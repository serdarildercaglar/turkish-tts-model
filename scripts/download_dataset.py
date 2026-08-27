"""Veri kümesini Hub'dan indirir (train + review + validation).

Kapılı depodan tek akışlı indirme ~140 KB/s'te takılıyor; `hf_transfer` paralel
parça indirmesi bunu iki mertebe açıyor, betik onu varsayılan olarak açar.

    pip install hf_transfer
    python scripts/download_dataset.py --out /veri/turkish-tts --splits train review

Sürdürülebilir: var olan dosyalar atlanır, kesilirse kaldığı yerden devam eder.
Yalnızca `data/<split>-*.parquet` çekilir; ses parquet içinde gömülüdür,
ayrıca klip dosyası yoktur.

Boyutlar: train ~75,6 GB (258 shard), review ~83,7 GB (249 shard),
validation ~1,2 GB (5 shard).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO = "serdarcaglar/turkish-tts-audiobooks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="veri kumesinin inecegi dizin")
    ap.add_argument("--splits", nargs="+", default=["train", "review"],
                    choices=["train", "review", "validation"])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int,
                    help="bolum basina ilk N shard (deneme icin)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    files = api.list_repo_files(args.repo, repo_type="dataset")

    patterns = []
    for sp in args.splits:
        shards = sorted(f for f in files
                        if f.startswith(f"data/{sp}-") and f.endswith(".parquet"))
        if not shards:
            print(f"! '{sp}' icin shard bulunamadi", file=sys.stderr)
            continue
        if args.limit:
            shards = shards[: args.limit]
        patterns.extend(shards)
        print(f"{sp}: {len(shards)} shard", file=sys.stderr)

    if not patterns:
        raise SystemExit("indirilecek dosya yok")

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snapshot_download(
        args.repo, repo_type="dataset", local_dir=str(args.out),
        allow_patterns=patterns, max_workers=args.workers,
    )
    total = sum((args.out / p).stat().st_size for p in patterns
                if (args.out / p).is_file())
    dt = time.time() - t0
    print(f"bitti: {len(patterns)} dosya, {total / 2**30:.1f} GB, "
          f"{dt / 60:.1f} dk ({total / max(dt, 1) / 2**20:.1f} MB/s) -> {args.out}")


if __name__ == "__main__":
    main()
