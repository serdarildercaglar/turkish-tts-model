"""Review kliplerine klonlama referansı atar — konuşmacı gömmelerinden.

Review bölümünde `speaker_id` ve `ref_id` boş geliyor; bu haliyle 1.340 saat
yalnızca düz TTS'e katkı sağlar, klonlamaya sıfır. Oysa veri hattının sqlite
veritabanında her klip için 256 boyutlu konuşmacı gömmesi var
(`work/db/state-v2.sqlite`, `embeddings` tablosu).

Klonlama için konuşmacı ETİKETİ gerekmez; her klip için *benzer sesli* bir eş
yeterli. Dolayısıyla kümeleme yerine en yakın komşu araması yapılır — eşik
dışında ayarlanacak hiperparametre kalmaz.

İki kısıt:
  - eş AYNI KANALDAN olur (farklı kanal = farklı kayıt ortamı)
  - eş FARKLI KAYITTAN olur (aynı dosyanın komşu parçası olmasın; model
    "aynı dosyayı sürdür" kestirmesini öğrenmesin)

Kimlik eşlemesi: manifest kimliği `source-<hash>-<baslangic_ms>-<bitis_ms>`
biçiminde; veritabanındaki klip (channel, start_sec, end_sec) ile eşleşir.

    python scripts/pair_clone_refs.py \
        --manifest artifacts/manifest/review_hyg.jsonl \
        --db /path/work/db/state-v2.sqlite \
        --out artifacts/manifest/review_paired.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ID = re.compile(r"^(?P<src>source-[0-9a-f]+)-(?P<a>\d{10})-(?P<b>\d{10})$")


def parse_id(cid: str):
    m = _ID.match(cid)
    if not m:
        return None
    return m.group("src"), int(m.group("a")), int(m.group("b"))


def load_embeddings(db: Path, decision: str):
    """(channel, start_ms, end_ms) -> (source_id, vektor) tablosu."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.execute(
        "SELECT c.channel, c.start_sec, c.end_sec, c.source_id, "
        "       e.vector, e.dimensions "
        "FROM clips c JOIN embeddings e ON e.clip_id = c.id "
        "WHERE c.decision = ?", (decision,))
    table = {}
    for ch, s, e, src, blob, dim in cur:
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.size != dim:
            continue
        table[(ch, int(round(s * 1000)), int(round(e * 1000)))] = (src, vec)
    con.close()
    return table


def nearest_within_channel(vecs: np.ndarray, srcs: np.ndarray,
                           min_sim: float, device: str, chunk: int = 2048):
    """Her satır için farklı `source_id`'den gelen en benzer satırın indisi.

    Kosinüs benzerliği; vektörler önceden birim uzunluğa getirilir. Aynı
    kayıttan gelen adaylar -inf ile maskelenir.
    """
    import torch

    x = torch.from_numpy(vecs)
    x = torch.nn.functional.normalize(x, dim=1).to(device)
    src = torch.from_numpy(srcs).to(device)
    n = x.shape[0]
    best_i = np.full(n, -1, dtype=np.int64)
    best_s = np.zeros(n, dtype=np.float32)

    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        sim = x[s0:s1] @ x.T                      # [c, n]
        same_src = src[s0:s1, None] == src[None, :]
        sim.masked_fill_(same_src, float("-inf"))
        v, i = sim.max(dim=1)
        best_i[s0:s1] = i.cpu().numpy()
        best_s[s0:s1] = v.float().cpu().numpy()
        del sim, same_src
    keep = best_s >= min_sim
    best_i[~keep] = -1
    return best_i, best_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decision", default="REVIEW")
    ap.add_argument("--min-sim", type=float, default=0.70,
                    help="kosinus esigi; altinda kalan klip ref_id'siz kalir")
    ap.add_argument("--device", default="cpu",
                    help="benzerlik matrisi icin; bos bir GPU cok hizlandirir")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8")
            if l.strip()]
    print(f"manifest: {len(rows):,} satir", file=sys.stderr)

    emb = load_embeddings(args.db, args.decision)
    print(f"gomme: {len(emb):,} klip", file=sys.stderr)

    # kanal bazinda topla
    per_channel: dict[str, list] = defaultdict(list)
    missing = 0
    for r in rows:
        p = parse_id(r["id"])
        if p is None:
            missing += 1
            continue
        key = (r.get("channel"), p[1], p[2])
        hit = emb.get(key)
        if hit is None:
            missing += 1
            continue
        per_channel[r["channel"]].append((r, hit[0], hit[1]))
    print(f"gomme eslesmeyen: {missing:,}", file=sys.stderr)

    paired = 0
    sims = []
    spk_assigned = 0
    for ch, items in sorted(per_channel.items(),
                            key=lambda kv: -len(kv[1])):
        vecs = np.stack([v for _, _, v in items])
        srcs = np.asarray([s for _, s, _ in items], dtype=np.int64)
        idx, sim = nearest_within_channel(vecs, srcs, args.min_sim, args.device)
        for k, (r, _, _) in enumerate(items):
            # konusmaci kimligi yoksa kanali vekil yap (dedupe ve bolunme icin)
            if r.get("speaker_id") in (None, "", "None"):
                r["speaker_id"] = f"review-{ch}"
                spk_assigned += 1
            j = int(idx[k])
            if j >= 0:
                r["ref_id"] = items[j][0]["id"]
                r["ref_sim"] = round(float(sim[k]), 4)
                paired += 1
                sims.append(float(sim[k]))
        print(f"  {ch:<24} {len(items):>7,} klip -> "
              f"{sum(1 for k in range(len(items)) if idx[k] >= 0):>7,} eslendi",
              file=sys.stderr, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    sims_a = np.asarray(sims) if sims else np.zeros(1)
    rep = {
        "satir": len(rows),
        "gomme_eslesmeyen": missing,
        "ref_atanan": paired,
        "ref_orani_yuzde": round(paired / max(len(rows), 1) * 100, 2),
        "konusmaci_vekili_atanan": spk_assigned,
        "benzerlik": {
            "ort": round(float(sims_a.mean()), 4),
            "p10": round(float(np.percentile(sims_a, 10)), 4),
            "p50": round(float(np.percentile(sims_a, 50)), 4),
            "p90": round(float(np.percentile(sims_a, 90)), 4),
        },
        "min_sim": args.min_sim,
    }
    text = json.dumps(rep, ensure_ascii=False, indent=1)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
