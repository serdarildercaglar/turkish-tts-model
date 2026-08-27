"""Harici HF veri kümesini klip + manifest'e çevirir (FT hattı için).

Kaynak, ses'i parquet içinde gömülü taşıyan herhangi bir HF TTS kümesi olabilir.
Shard indir → satırları çöz → hijyenden geçir → `<id>.flac` + manifest satırı
yaz → shard'ı sil. Sürdürülebilir: bitmiş shard'lar `done/` imleriyle atlanır.

Desteklenen kümeler için örnek çağrılar (bkz. docs/ft_qwen3tts.md):

    # tek konuşmacı, 48 kHz, lisanssız (ticari risk kullanıcı kararı)
    python scripts/ingest_hf_dataset.py --repo Anilosan15/Turkish_TTS_Data \
        --tag anil --speaker-const sila --clips /veri/ext/anil \
        --out artifacts/manifest/ext_anil.jsonl

    # konuşmacı etiketi YOK → sonra pair_by_embedding.py şart
    python scripts/ingest_hf_dataset.py --repo afkfatih/turkish-tts-combined-raw \
        --tag afk --clips /veri/ext/afk --out artifacts/manifest/ext_afk.jsonl

    # 2.158 etiketli konuşmacı + duygu; kayıt anahtarı original_filename
    python scripts/ingest_hf_dataset.py --repo Codyfederer/tr-combined \
        --tag cody --speaker-col speaker_id --src-col original_filename \
        --clips /veri/ext/cody --out artifacts/manifest/ext_cody.jsonl

Manifest satırı, mevcut hat ile aynı sözleşmeyi taşır (id, audio, text,
duration, speaker_id, channel) + `src` (kayıt anahtarı: aynı kayıttan referans
seçilmesin). Metin/süre/mükerrer süzgeci `src.data_hygiene.Hygiene`'dir;
`--dnsmos-min` verilirse ses kalitesi de elenir (bizim eşik pratiği ~3,0).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_hygiene import Hygiene, Policy

AUDIO_COLS = ("audio", "wav", "speech")


def list_shards(repo: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True, help="kimlik/kanal öneki (örn. anil)")
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speaker-col")
    ap.add_argument("--speaker-const", help="tüm satırlara sabit konuşmacı")
    ap.add_argument("--src-col", help="kayıt anahtarı sütunu (örn. original_filename)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--dnsmos-min", type=float,
                    help="raw-dalga DNSMOS OVRL alt sınırı (örn. 3.0)")
    ap.add_argument("--dnsmos-onnx",
                    default="/mnt/310C8DBF109E2BFC/projects/turkish-tts/voxcpm/"
                            "work/models/dnsmos_sig_bak_ovr.onnx")
    ap.add_argument("--max-shards", type=int, help="deneme için sınırla")
    ap.add_argument("--keep-parquet", action="store_true")
    args = ap.parse_args()

    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    dns = None
    if args.dnsmos_min is not None:
        from src.evaluate import DNSMOS

        dns = DNSMOS(args.dnsmos_onnx)

    hy = Hygiene(Policy())
    args.clips.mkdir(parents=True, exist_ok=True)
    done_dir = args.clips / "done"
    done_dir.mkdir(exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    shards = list_shards(args.repo)
    if args.max_shards:
        shards = shards[: args.max_shards]
    print(f"{args.repo}: {len(shards)} shard", file=sys.stderr)

    stats = dict(satir=0, tutulan=0, hijyen=0, dnsmos=0, ses_yok=0, saat=0.0)
    mode = "a" if any(done_dir.iterdir()) else "w"
    fout = args.out.open(mode, encoding="utf-8")

    for si, shard in enumerate(shards):
        mark = done_dir / f"{Path(shard).stem}.done"
        if mark.exists():
            continue
        local = hf_hub_download(args.repo, shard, repo_type="dataset")
        pf = pq.ParquetFile(local)
        names = [f.name for f in pf.schema_arrow]
        audio_col = next((c for c in AUDIO_COLS if c in names), None)
        if audio_col is None:
            raise SystemExit(f"{shard}: ses sutunu yok; sema {names}")
        ri = 0
        for batch in pf.iter_batches(batch_size=64):
            for r in batch.to_pylist():
                stats["satir"] += 1
                cid = f"{args.tag}-s{si:04d}r{ri:06d}"
                ri += 1
                cell = r.get(audio_col)
                data = cell.get("bytes") if isinstance(cell, dict) else cell
                if data is None:
                    stats["ses_yok"] += 1
                    continue
                wav, sr = sf.read(io.BytesIO(data), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                dur = len(wav) / sr
                spk = (args.speaker_const
                       or (r.get(args.speaker_col) if args.speaker_col else "")
                       or "")
                row = dict(id=cid, text=r.get(args.text_col) or "",
                           duration=dur, speaker_id=str(spk),
                           channel=args.tag,
                           # src-col bos gelirse (tr-combined'da cogu satir oyle)
                           # klibin kendisi kayit sayilir; ayni-kayit kisiti
                           # hic ref'i engellemesin
                           src=(str(r.get(args.src_col) or "").strip()
                                if args.src_col else "") or cid,
                           sr=int(sr))
                out = hy.process(dict(row), audio_exists=lambda _: True)
                if out is None:
                    stats["hijyen"] += 1
                    continue
                if dns is not None and dns.ovrl(wav, sr) < args.dnsmos_min:
                    stats["dnsmos"] += 1
                    continue
                path = args.clips / f"{cid}.flac"
                sf.write(path, wav, sr)
                out["audio"] = str(path)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["tutulan"] += 1
                stats["saat"] += dur / 3600
        fout.flush()
        mark.touch()
        if not args.keep_parquet:
            Path(local).unlink(missing_ok=True)
        print(f"  {shard} bitti; toplam {stats['tutulan']:,} klip "
              f"({stats['saat']:.1f} sa)", file=sys.stderr, flush=True)

    fout.close()
    stats["saat"] = round(stats["saat"], 1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
