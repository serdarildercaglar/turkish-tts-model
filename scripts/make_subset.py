"""Pilot için alt küme: en temiz klipler, prozodi kovalarının tamamından.

İki kısıt aynı anda gözetilir:

1. **Temizlik.** Kalite alanlarının üst dilimleri (DNSMOS, konuşma oranı,
   kırpma, müzik, ASR CER) ve cümle parçası olmayan klipler. Eşikler yüzdelik
   olarak verilir, böylece korpus değişse de anlam korunur.
2. **Prozodi kapsaması.** Hız (5) × seviye (3) = 15 kova eşit doldurulur;
   böylece pilot "hep aynı tempoda okunmuş temiz kitap" olmaz. Kova içinde
   konuşmacılar sırayla gezilir (round-robin), tek okuyucu kovayı kapatmasın.

Hedef süreye ulaşınca durur.

    python scripts/make_subset.py \
        --manifest artifacts/manifest/train_all.jsonl \
        --hours 100 --out artifacts/manifest/pilot_100h.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_hygiene import speaker_key
from src.prosody import Prosody


def _f(row: dict, key: str):
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=100.0)
    ap.add_argument("--prosody", type=Path,
                    help="prosody.json dizini (varsayilan: manifest yani)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dnsmos-pct", type=float, default=50.0,
                    help="DNSMOS icin alt yuzdelik esigi (yuksek = daha temiz)")
    ap.add_argument("--speech-pct", type=float, default=40.0)
    ap.add_argument("--max-clip-ratio", type=float, default=0.0)
    ap.add_argument("--max-music", type=float, default=0.05)
    ap.add_argument("--max-cer", type=float, default=0.0)
    ap.add_argument("--allow-fragments", action="store_true",
                    help="cumle parcalarini da al (varsayilan: alma)")
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--splits", nargs="+",
                    help="yalnizca bu source_split degerlerinden sec")
    args = ap.parse_args()

    rows = []
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    print(f"manifest: {len(rows):,} satir", file=sys.stderr)

    if args.splits:
        rows = [r for r in rows if r.get("source_split") in set(args.splits)]
        print(f"bolum suzgeci sonrasi: {len(rows):,}", file=sys.stderr)

    # yuzdelik esikler korpusun kendisinden
    dns = np.array([v for v in (_f(r, "quality_dnsmos_ovrl") for r in rows)
                    if v is not None])
    spc = np.array([v for v in (_f(r, "quality_speech_ratio") for r in rows)
                    if v is not None])
    dns_min = float(np.percentile(dns, args.dnsmos_pct)) if dns.size else -1e9
    spc_min = float(np.percentile(spc, args.speech_pct)) if spc.size else -1e9
    print(f"esikler: dnsmos >= {dns_min:.4f}, konusma orani >= {spc_min:.4f}",
          file=sys.stderr)

    drop = Counter()

    def clean(r: dict) -> bool:
        d = _f(r, "duration") or 0
        if not (args.min_sec <= d <= args.max_sec):
            drop["sure araligi"] += 1
            return False
        if not args.allow_fragments and (r.get("frag_start") or r.get("frag_end")):
            drop["cumle parcasi"] += 1
            return False
        v = _f(r, "quality_dnsmos_ovrl")
        if v is not None and v < dns_min:
            drop["dnsmos"] += 1
            return False
        v = _f(r, "quality_speech_ratio")
        if v is not None and v < spc_min:
            drop["konusma orani"] += 1
            return False
        v = _f(r, "quality_clip_ratio")
        if v is not None and v > args.max_clip_ratio:
            drop["kirpma"] += 1
            return False
        v = _f(r, "quality_music_score")
        if v is not None and v > args.max_music:
            drop["muzik"] += 1
            return False
        v = _f(r, "quality_asr_cer")
        if v is not None and v > args.max_cer:
            drop["asr cer"] += 1
            return False
        return True

    pool = [r for r in rows if clean(r)]
    print(f"temizlik sonrasi havuz: {len(pool):,} klip "
          f"({sum(_f(r, 'duration') or 0 for r in pool) / 3600:.1f} saat)",
          file=sys.stderr)
    for k, v in drop.most_common():
        print(f"   elenen {k}: {v:,}", file=sys.stderr)
    if not pool:
        raise SystemExit("havuz bos; esikleri gevsetin")

    # ---------------------------------------------------- prozodi tabakalari
    pros = Prosody.load(args.prosody or args.manifest.parent)
    cells: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in pool:
        rb = pros.rate_bucket(r.get("text", ""), _f(r, "duration"))
        lb = pros.loud_bucket(r.get("quality_lufs"))
        cells[(rb, lb)][speaker_key(r)].append(r)

    rng = random.Random(args.seed)
    for spk_map in cells.values():
        for lst in spk_map.values():
            rng.shuffle(lst)

    target = args.hours * 3600.0
    chosen: list[dict] = []
    total = 0.0
    # kovalar arasinda sirayla, kova icinde konusmacilar arasinda sirayla gez
    cell_keys = sorted(cells, key=lambda k: (k[0] is None, k))
    cursors = {k: 0 for k in cell_keys}
    spk_lists = {k: sorted(cells[k]) for k in cell_keys}
    exhausted: set = set()

    while total < target and len(exhausted) < len(cell_keys):
        for ck in cell_keys:
            if ck in exhausted or total >= target:
                continue
            spks = spk_lists[ck]
            took = False
            for _ in range(len(spks)):
                s = spks[cursors[ck] % len(spks)]
                cursors[ck] += 1
                bucket = cells[ck][s]
                if bucket:
                    r = bucket.pop()
                    chosen.append(r)
                    total += _f(r, "duration") or 0
                    took = True
                    break
            if not took:
                exhausted.add(ck)

    # Klon ornekleri referans klibin de kumede olmasini gerektirir; secilen
    # kliplerin ref_id esleri yoksa klon egitimi bos kalir. Esleri havuzdan
    # degil TUM manifestten cekeriz: referans yalnizca istem oneki olarak
    # kullanilir, uzerinde kayip hesaplanmaz, dolayisiyla temizlik kapisindan
    # gecmesi sart degildir.
    by_id = {r["id"]: r for r in rows}
    have = {r["id"] for r in chosen}
    added = 0
    for r in list(chosen):
        ref_id = r.get("ref_id")
        if not ref_id or ref_id in ("None", "") or ref_id in have:
            continue
        ref = by_id.get(ref_id)
        if ref is None:
            continue
        chosen.append(ref)
        have.add(ref_id)
        total += _f(ref, "duration") or 0
        added += 1
    if added:
        print(f"klon icin {added:,} referans klibi eklendi", file=sys.stderr)

    rng.shuffle(chosen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------- rapor
    per_cell = Counter()
    per_spk = Counter()
    for r in chosen:
        rb = pros.rate_bucket(r.get("text", ""), _f(r, "duration"))
        lb = pros.loud_bucket(r.get("quality_lufs"))
        per_cell[(rb, lb)] += _f(r, "duration") or 0
        per_spk[speaker_key(r)] += 1
    rep = {
        "klip": len(chosen),
        "saat": round(total / 3600, 2),
        "hedef_saat": args.hours,
        "konusmaci": len(per_spk),
        "kova_saat": {f"hiz{rb}_seviye{lb}": round(v / 3600, 2)
                      for (rb, lb), v in sorted(per_cell.items(),
                                                key=lambda kv: str(kv[0]))},
        "esikler": {"dnsmos_min": round(dns_min, 4),
                    "konusma_orani_min": round(spc_min, 4),
                    "max_clip_ratio": args.max_clip_ratio,
                    "max_music": args.max_music, "max_cer": args.max_cer},
        "havuz_klip": len(pool),
        "eklenen_referans": added,
    }
    (args.out.with_suffix(".report.json")).write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
