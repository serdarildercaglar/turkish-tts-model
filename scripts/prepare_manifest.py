"""Veri kümesinden eğitime girecek TÜRETİLMİŞ manifesti üretir.

Kaynağa (parquet ya da jsonl) yazmaz. `src.data_hygiene.Policy` uyarınca süzer,
metni düzeltir, `train` ve `review` bölümlerini birleştirir ve tek bir jsonl
yazar. Ses hiç açılmaz — parquet sütunlu olduğu için üstveri okumak ucuzdur.

    # indirilen veri kumesinden, train + review birlesik
    python scripts/prepare_manifest.py \
        --data /veri/turkish-tts --splits train review \
        --out artifacts/manifest/train_all.jsonl \
        --report artifacts/manifest/train_all_report.json

    # eski akis: yerel jsonl manifestler
    python scripts/prepare_manifest.py \
        --manifest .../hf_train.jsonl --manifest .../hf_review.jsonl --out ...

Rapor, her elemenin sayısını ve gerekçesini verir; sessizce düşen satır yoktur.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_hygiene import Hygiene, Policy
from src.dataset_source import available_splits, iter_jsonl, iter_metadata
from src.prosody import Prosody


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("kaynak (biri gerekli)")
    src.add_argument("--data", type=Path,
                     help="indirilen veri kumesi dizini (parquet)")
    src.add_argument("--splits", nargs="+", default=["train", "review"],
                     help="--data ile kullanilacak bolumler")
    src.add_argument("--manifest", type=Path, action="append", default=[],
                     help="jsonl manifest; birden fazla verilebilir")

    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--prosody-out", type=Path,
                    help="kova sinirlarinin yazilacagi dizin (varsayilan: --out yani)")
    ap.add_argument("--fragments", choices=("mark", "drop", "keep"),
                    default="mark")
    ap.add_argument("--verbalize-digits", action="store_true",
                    help="rakamlari sozlu forma cevir (varsayilan kapali)")
    ap.add_argument("--require-audio", action="store_true",
                    help="jsonl akisinda ses dosyasi diskte yoksa ele")
    ap.add_argument("--keep-reason", action="append", default=[],
                    help="bu review gerekcesini eleme listesinden cikar")
    ap.add_argument("--drop-reason", action="append", default=[],
                    help="bu review gerekcesini eleme listesine ekle")
    args = ap.parse_args()

    if not args.data and not args.manifest:
        ap.error("--data ya da --manifest verin")
    if args.data and not available_splits(args.data):
        ap.error(f"{args.data} altinda parquet shard bulunamadi")

    pol = Policy(
        fragments=args.fragments,
        verbalize_digits=args.verbalize_digits,
        require_audio=args.require_audio,
    )
    reasons = (set(pol.drop_reasons) - set(args.keep_reason)) | set(args.drop_reason)
    pol.drop_reasons = tuple(sorted(reasons))

    hy = Hygiene(pol)
    per_source: dict[str, dict] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, object]] = []
    if args.data:
        for sp in args.splits:
            sources.append((sp, iter_metadata(args.data, sp)))
    for m in args.manifest:
        sources.append((m.stem, iter_jsonl(m)))

    n_in = 0
    kept_rows: list[dict] = []
    with args.out.open("w", encoding="utf-8") as fh:
        for name, rows in sources:
            k0 = hy.stats.kept
            d0 = sum(hy.stats.dropped.values())
            n0 = n_in
            for row in rows:
                n_in += 1
                exists = None
                if pol.require_audio and row.get("audio"):
                    a = row["audio"]
                    exists = os.path.isfile(a) if isinstance(a, str) else True
                out = hy.process(row, audio_exists=exists)
                if out is None:
                    continue
                out["source_split"] = name
                # `audio` alani korunur: yerel FLAC akisinda ingest_audio.py
                # dogrudan bu yolu okur. Parquet akisinda alan zaten yoktur.
                kept_rows.append({"text": out["text"],
                                  "duration": out.get("duration"),
                                  "quality_lufs": out.get("quality_lufs")})
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            per_source[name] = {
                "girdi": n_in - n0,
                "tutulan": hy.stats.kept - k0,
                "elenen": sum(hy.stats.dropped.values()) - d0,
            }
            print(f"{name}: {per_source[name]}", file=sys.stderr, flush=True)

    # prozodi kova sinirlari: TUTULAN satirlardan hesaplanir
    pros = Prosody.from_rows(kept_rows)
    pros_dir = args.prosody_out or args.out.parent
    pros.save(pros_dir)

    rep = {
        "politika": {
            "fragments": pol.fragments,
            "verbalize_digits": pol.verbalize_digits,
            "require_audio": pol.require_audio,
            "drop_reasons": list(pol.drop_reasons),
            "esikler": {
                "max_clip_ratio": pol.max_clip_ratio,
                "max_music_score": pol.max_music_score,
                "min_speech_ratio": pol.min_speech_ratio,
                "min_dnsmos_ovrl": pol.min_dnsmos_ovrl,
                "min_speaker_cosine": pol.min_speaker_cosine,
                "rate": [pol.rate_min, pol.rate_max],
            },
        },
        "girdi_satir": n_in,
        "kaynak_basina": per_source,
        "prozodi": pros.to_dict(),
        "saat": round(sum(float(r["duration"] or 0) for r in kept_rows) / 3600, 1),
        **hy.stats.as_dict(),
    }
    text = json.dumps(rep, ensure_ascii=False, indent=1)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
