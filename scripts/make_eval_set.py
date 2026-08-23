"""Sabit değerlendirme kümesi üretir: 100 düz + 100 klon istemi.

Validation manifestinden sabit tohumla örnekler; klon istemleri için
referans klip aynı kaynak kayıttan DEĞİL, aynı split içindeki ref_id
çiftlerinden gelir. Validation'da ref_id azsa (yalnız 1.604 klipte var)
eksik kalan klon istemleri raporlanır.

    python scripts/make_eval_set.py --manifest .../hf_validation.jsonl \
        --out artifacts/eval_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-plain", type=int, default=100)
    ap.add_argument("--n-clone", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = {}
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r

    rng = random.Random(args.seed)
    ordered = sorted(rows.values(), key=lambda r: r["id"])
    rng.shuffle(ordered)

    plain = [r for r in ordered if 3.0 <= r["duration"] <= 15.0][: args.n_plain]
    clone_pool = [r for r in ordered
                  if r.get("ref_id") and r["ref_id"] in rows
                  and Path(rows[r["ref_id"]]["audio"]).is_file()]
    clone = clone_pool[: args.n_clone]

    out = []
    for r in plain:
        out.append({"kind": "plain", "text": r["text"], "clip_id": r["id"]})
    for r in clone:
        ref = rows[r["ref_id"]]
        out.append({"kind": "clone", "text": r["text"], "clip_id": r["id"],
                    "ref_audio": ref["audio"], "ref_text": ref["text"],
                    "ref_id": ref["id"]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(plain)} duz + {len(clone)} klon istem -> {args.out}")
    if len(clone) < args.n_clone:
        print(f"UYARI: validation'da yalnizca {len(clone)} klon cifti bulundu; "
              "eksigi train konusmacilarindan tamamlamayi degerlendirin")


if __name__ == "__main__":
    main()
