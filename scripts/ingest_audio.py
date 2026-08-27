"""Sesi SNAC kodlarına çevirir — parquet'ten ya da yerel kliplerden.

Kaynak indirilmiş veri kümesi ise ses parquet içinde gömülüdür; shard shard
okunur, yalnızca hijyenden geçen klipler kodlanır. Yerel FLAC klipler varsa
manifestteki `audio` yolu kullanılır.

Çıktı, bölüm başına `shard-XXXXX.npz` (ids / lens / tokens). `build_dataset.py`
tüm bölümlerin kodlarını tek dizin ağacından okur.

    # indirilen veri kumesinden (train + review)
    python scripts/ingest_audio.py \
        --data /veri/turkish-tts --splits train review \
        --manifest artifacts/manifest/train_all.jsonl \
        --out artifacts/codes

    # yerel FLAC kliplerden
    python scripts/ingest_audio.py \
        --manifest artifacts/manifest/train_all.jsonl --out artifacts/codes/train

Sürdürülebilir: yazılmış shard'lar atlanır. `--delete-parquet` ile işlenen
parquet silinerek tepe disk kullanımı bir shard'a indirilebilir.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codec import SAMPLES_PER_FRAME, SNAC_SR, encode_batch, load_snac
from src.dataset_source import find_shards, iter_audio

LOCAL_SHARD_SIZE = 4096


def decode_audio(cell):
    """HF `Audio` hücresi: {'bytes':..., 'path':...} ya da doğrudan bayt."""
    import soundfile as sf

    if cell is None:
        return None
    data = cell.get("bytes") if isinstance(cell, dict) else cell
    if data is None:
        path = cell.get("path") if isinstance(cell, dict) else None
        if not path or not os.path.isfile(path):
            return None
        return sf.read(path, dtype="float32", always_2d=False)
    return sf.read(io.BytesIO(data), dtype="float32", always_2d=False)


def to_24k(wav: np.ndarray, sr: int) -> torch.Tensor:
    import torchaudio.functional as AF

    x = torch.from_numpy(np.ascontiguousarray(wav)).float()
    if x.ndim > 1:
        x = x.mean(dim=-1)
    if sr != SNAC_SR:
        x = AF.resample(x, sr, SNAC_SR)
    return x


def encode_group(model, items, device, batch_sec: float):
    """(id, wav24k) listesini süreye göre sıralayıp batch'leyerek kodlar."""
    items = sorted(items, key=lambda t: t[1].shape[-1])
    ids, codes = [], []
    batch, budget = [], 0.0

    def flush():
        nonlocal batch, budget
        if not batch:
            return
        n_samples = [w.shape[-1] for _, w in batch]
        pad_to = -(-max(n_samples) // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
        x = torch.zeros(len(batch), 1, pad_to)
        for i, (_, w) in enumerate(batch):
            x[i, 0, : w.shape[-1]] = w
        for (cid, _), flat in zip(batch, encode_batch(model, x.to(device), n_samples)):
            ids.append(cid)
            codes.append(flat)
        batch, budget = [], 0.0

    for cid, w in items:
        batch.append((cid, w))
        budget += w.shape[-1] / SNAC_SR
        if budget >= batch_sec:
            flush()
    flush()
    return ids, codes


def write_shard(path: Path, ids: list[str], codes: list[list[int]]) -> None:
    lens = np.array([len(c) for c in codes], dtype=np.int32)
    tokens = (np.concatenate([np.asarray(c, dtype=np.uint16) for c in codes])
              if codes else np.zeros(0, dtype=np.uint16))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ids=np.array(ids), lens=lens, tokens=tokens)
    tmp.rename(path)


def run_parquet(args, model, keep, keep_split) -> None:
    for split in args.splits:
        shards = find_shards(args.data, split)
        if not shards:
            print(f"! {split}: shard yok, atlandi", file=sys.stderr)
            continue
        out_dir = args.out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        ids_wanted = keep_split.get(split, keep)
        t0 = time.time()
        n_clip = n_tok = done = skipped = 0
        for si, shard in enumerate(shards):
            out_path = out_dir / f"shard-{si:05d}.npz"
            if out_path.exists():
                skipped += 1
                continue
            items = []
            for cid, cell in iter_audio(shard, ids_wanted):
                dec = decode_audio(cell)
                if dec is not None:
                    items.append((cid, to_24k(*dec)))
            ids, codes = encode_group(model, items, args.device, args.batch_sec)
            write_shard(out_path, ids, codes)
            done += 1
            n_clip += len(ids)
            n_tok += sum(len(c) for c in codes)
            if args.delete_parquet:
                os.remove(shard)
            print(f"[{split}] shard {si + 1}/{len(shards)}  klip={len(ids)}  "
                  f"toplam={n_clip:,}  gecen={(time.time() - t0) / 60:.1f}dk",
                  flush=True)
        print(f"[{split}] bitti: {done} yazildi, {skipped} atlandi, "
              f"{n_clip:,} klip, {n_tok:,} token -> {out_dir}")


def run_local(args, model, rows) -> None:
    """Manifestteki `audio` yolundan okur (eski akış)."""
    import soundfile as sf

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r.get("audio")]
    shards = [rows[i:i + LOCAL_SHARD_SIZE]
              for i in range(0, len(rows), LOCAL_SHARD_SIZE)]
    t0 = time.time()
    done = skipped = n_clip = 0
    for si, chunk in enumerate(shards):
        out_path = args.out / f"shard-{si:05d}.npz"
        if out_path.exists():
            skipped += 1
            continue
        items = []
        for r in chunk:
            if not os.path.isfile(r["audio"]):
                continue
            wav, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
            items.append((r["id"], to_24k(wav, sr)))
        ids, codes = encode_group(model, items, args.device, args.batch_sec)
        write_shard(out_path, ids, codes)
        done += 1
        n_clip += len(ids)
        print(f"shard {si + 1}/{len(shards)}  klip={len(ids)}  "
              f"toplam={n_clip:,}  gecen={(time.time() - t0) / 60:.1f}dk", flush=True)
    print(f"bitti: {done} yazildi, {skipped} atlandi, {n_clip:,} klip -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="hijyenden gecmis manifest (tutulacak kimlikler)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, help="indirilen veri kumesi dizini")
    ap.add_argument("--splits", nargs="+", default=["train", "review"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-sec", type=float, default=240.0)
    ap.add_argument("--delete-parquet", action="store_true",
                    help="islenen parquet'i sil (tepe disk kullanimini dusurur)")
    args = ap.parse_args()

    keep: set[str] = set()
    keep_split: dict[str, set[str]] = {}
    rows = []
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            keep.add(r["id"])
            keep_split.setdefault(r.get("source_split", ""), set()).add(r["id"])
            rows.append(r)
    print(f"hijyenden gecen klip: {len(keep):,}", file=sys.stderr)

    model = load_snac(args.device)
    if args.data:
        run_parquet(args, model, keep, keep_split)
    else:
        run_local(args, model, rows)


if __name__ == "__main__":
    main()
