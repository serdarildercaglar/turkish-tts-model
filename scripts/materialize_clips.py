"""FT JSONL'inin ses dosyalarını Hub parquet'lerinden diske döker.

Bulut makinesi akışı (Qwen3-TTS ince ayarı): veri kümesi parquet içinde gömülü
ses taşır, Qwen'in `prepare_data.py`'si ise dosya yolu ister. Bu betik,
`export_qwen3tts_ft.py` çıktısındaki hedef + referans kliplerin SES'ini
parquet'lerden okuyup `<out>/<id>.flac` olarak yazar ve JSONL'deki yolları
yeniden yazar.

    # bulutta: önce parquet'ler (download_dataset.py ya da huggingface-cli)
    python scripts/materialize_clips.py \
        --ft-jsonl artifacts/qwen3tts_ft/train.jsonl \
        --data /veri/turkish-tts --splits train review \
        --clips-out /veri/clips \
        --out artifacts/qwen3tts_ft/train_cloud.jsonl

Sürdürülebilir: var olan .flac atlanır. Kimlik eşlemesi dosya adından
yapıldığı için yerel yolların bulut'ta bulunmaması sorun değildir.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_source import find_shards, iter_audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-jsonl", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True,
                    help="parquet kökü (data/<split>-*.parquet)")
    ap.add_argument("--splits", nargs="+", default=["train", "review"])
    ap.add_argument("--clips-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="yolları yeniden yazılmış FT JSONL'i")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.ft_jsonl.open(encoding="utf-8")
            if l.strip()]
    if rows and "id" not in rows[0]:
        raise SystemExit("ft-jsonl'de 'id' alani yok; export_qwen3tts_ft.py'yi "
                         "guncel surumle yeniden kosun")

    need = ({r["id"] for r in rows}
            | {r["ref_id"] for r in rows if r.get("ref_id")})
    args.clips_out.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in args.clips_out.glob("*.flac")}
    todo = need - have
    print(f"gereken {len(need):,} klip; diskte {len(need)-len(todo):,}, "
          f"yazilacak {len(todo):,}", file=sys.stderr)

    if todo:
        import soundfile as sf

        written = 0
        for split in args.splits:
            for shard in find_shards(args.data, split):
                for k, cell in iter_audio(shard, keep_ids=todo):
                    data = cell.get("bytes") if isinstance(cell, dict) else cell
                    if data is None:
                        continue
                    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
                    sf.write(args.clips_out / f"{k}.flac", wav, sr)
                    todo.discard(k)
                    written += 1
                    if written % 20000 == 0:
                        print(f"  {written:,} yazildi, {len(todo):,} kaldi",
                              file=sys.stderr, flush=True)
                if not todo:
                    break
            if not todo:
                break
        print(f"yazilan: {written:,}; bulunamayan: {len(todo):,}",
              file=sys.stderr)

    def newp(k: str | None, old: str) -> str:
        # sabit-ref satirlarinda ref_id yoktur; verilen yol aynen kalir
        return str(args.clips_out / f"{k}.flac") if k else old

    kept = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            a = newp(r["id"], r["audio"])
            ra = newp(r.get("ref_id"), r["ref_audio"])
            if Path(a).exists() and Path(ra).exists():
                fh.write(json.dumps({"audio": a, "text": r["text"],
                                     "ref_audio": ra},
                                    ensure_ascii=False) + "\n")
                kept += 1
    print(json.dumps({"satir": kept, "atilan": len(rows) - kept}))


if __name__ == "__main__":
    main()
