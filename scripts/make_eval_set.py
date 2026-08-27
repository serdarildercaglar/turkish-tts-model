"""Sabit değerlendirme kümesi: düz + klon istemleri.

Hijyenden geçmiş bir manifestten (`prepare_manifest.py`) sabit tohumla örnekler.
Klon istemleri `ref_id` çiftlerinden gelir; referans klibin sesi diskte olmalıdır.

Rakam içeren cümleler ayrı işaretlenir (`has_digits`): ASR bunları rakamla geri
yazar ve naif normalizasyon doğru okunuşu tutturamaz, dolayısıyla toplu WER'e
karıştırılmamalıdır.

    python scripts/make_eval_set.py \
        --manifest artifacts/manifest/train_all.jsonl \
        --out artifacts/eval_set.jsonl --exclude artifacts/manifest/pilot_100h.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

_DIGIT = re.compile(r"\d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, action="append", default=[],
                    help="bu manifestlerdeki kimlikler alinmaz (egitim kumesi)")
    ap.add_argument("--n-plain", type=int, default=100)
    ap.add_argument("--n-clone", type=int, default=100)
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    banned: set[str] = set()
    for p in args.exclude:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    banned.add(json.loads(line)["id"])

    rows: dict[str, dict] = {}
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r

    def dur(r):
        try:
            return float(r.get("duration") or 0)
        except (TypeError, ValueError):
            return 0.0

    rng = random.Random(args.seed)
    ordered = sorted(rows.values(), key=lambda r: r["id"])
    rng.shuffle(ordered)

    usable = [r for r in ordered
              if r["id"] not in banned and args.min_sec <= dur(r) <= args.max_sec]
    plain = usable[: args.n_plain]

    clone = []
    for r in usable:
        ref_id = r.get("ref_id")
        if not ref_id or ref_id in ("None", ""):
            continue
        ref = rows.get(ref_id)
        if ref is None or not ref.get("audio"):
            continue
        if not Path(ref["audio"]).is_file():
            continue
        clone.append((r, ref))
        if len(clone) >= args.n_clone:
            break

    out = []
    for r in plain:
        out.append({"kind": "plain", "text": r["text"], "clip_id": r["id"],
                    "has_digits": bool(_DIGIT.search(r["text"]))})
    for r, ref in clone:
        out.append({"kind": "clone", "text": r["text"], "clip_id": r["id"],
                    "ref_audio": ref["audio"], "ref_text": ref["text"],
                    "ref_id": ref["id"],
                    "has_digits": bool(_DIGIT.search(r["text"]))})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_dig = sum(1 for r in out if r["has_digits"])
    print(f"{len(plain)} duz + {len(clone)} klon istem -> {args.out}")
    print(f"  rakam iceren: {n_dig} (ayri raporlanmali)")
    if len(clone) < args.n_clone:
        print(f"UYARI: yalnizca {len(clone)} klon cifti bulundu "
              "(referans sesi diskte olan)")


if __name__ == "__main__":
    main()
