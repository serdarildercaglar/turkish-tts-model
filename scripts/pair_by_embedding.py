"""Konuşmacı etiketi OLMAYAN harici manifest'e klon referansı atar.

`pair_clone_refs.py`'nin genellemesi: orada gömmeler veri hattının sqlite'ından
geliyordu; burada kliplerden pyannote ile hesaplanır (npz'e önbelleklenir,
sürdürülebilir). Eşleme aynı: kanal içinde, FARKLI kayıttan (`src`), kosinüs
en yakın komşu; eşik altı satır ref'siz kalır ve dışa aktarımda düşer.

Konuşmacı kimliği bilinmediğinden `speaker_id` "emb-<kanal>" VEKİLİ yazılır;
`export_qwen3tts_ft.py` bu önekte aynı-konuşmacı yedeğine düşmez (kanal ≠
konuşmacı), yalnız buradaki gömme eşleşmesini kullanır.

    python scripts/pair_by_embedding.py \
        --manifest artifacts/manifest/ext_afk.jsonl \
        --out artifacts/manifest/ext_afk_paired.jsonl --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pair_clone_refs import nearest_within_channel


def embed_all(rows, cache: Path, device: str) -> np.ndarray:
    """Manifest sırasında [N, D] gömme; npz önbelleği kimlikle doğrulanır."""
    ids = [r["id"] for r in rows]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == ids:
            return z["vecs"]
        print("onbellek kimlikleri uyusmuyor, yeniden hesaplanacak",
              file=sys.stderr)
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained("pyannote/embedding")
    inf = Inference(model, window="whole", device=device)
    vecs = np.zeros((len(rows), 512), dtype=np.float32)
    for i, r in enumerate(rows):
        vecs[i] = np.asarray(inf(r["audio"]), dtype=np.float32)
        if (i + 1) % 2000 == 0:
            print(f"  gomme {i+1:,}/{len(rows):,}", file=sys.stderr, flush=True)
            np.savez(cache, ids=ids, vecs=vecs)  # ara kayit
    np.savez(cache, ids=ids, vecs=vecs)
    return vecs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-sim", type=float, default=0.70)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", type=Path,
                    help="gomme npz'i (varsayilan: <manifest>.emb.npz)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8")
            if l.strip()]
    cache = args.cache or args.manifest.with_suffix(".emb.npz")
    vecs = embed_all(rows, cache, args.device)

    by_ch: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_ch[r.get("channel") or ""].append(i)

    paired = 0
    for ch, idxs in by_ch.items():
        v = vecs[idxs]
        # ayni kayittan es secilmesin: src -> tamsayi
        srcs = {s: k for k, s in enumerate(
            {rows[i].get("src") or rows[i]["id"] for i in idxs})}
        sarr = np.asarray([srcs[rows[i].get("src") or rows[i]["id"]]
                           for i in idxs], dtype=np.int64)
        nn, sim = nearest_within_channel(v, sarr, args.min_sim, args.device)
        for k, i in enumerate(idxs):
            r = rows[i]
            if not r.get("speaker_id"):
                r["speaker_id"] = f"emb-{ch}"
            j = int(nn[k])
            if j >= 0:
                r["ref_id"] = rows[idxs[j]]["id"]
                r["ref_sim"] = round(float(sim[k]), 4)
                paired += 1
        print(f"  {ch:<12} {len(idxs):>7,} klip -> esli", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(satir=len(rows), ref_atanan=paired,
                          oran=round(paired / max(len(rows), 1), 3)),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
