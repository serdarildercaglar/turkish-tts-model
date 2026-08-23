"""Korpus kliplerini SNAC kodlarına çevirir (bir kezlik, sürdürülebilir geçiş).

Girdi: hattın hf_*.jsonl manifestleri + diskteki FLAC klipler (16 kHz mono).
Her klip 24 kHz'e yeniden örneklenir, SNAC ile kodlanır ve düzleştirilmiş
kodlar (bkz. src/codec.py) shard'lanmış .npz dosyalarına yazılır:

    out/shard-00042.npz : ids (S,), lens (S,), tokens (toplam, uint16)

Kaldığı yerden devam eder: mevcut shard dosyaları atlanır. Depo kökünden:

    python scripts/tokenize_audio.py --manifest .../hf_train.jsonl --out artifacts/codes/train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codec import SAMPLES_PER_FRAME, SNAC_SR, encode_batch, load_snac

SRC_SR = 16000
SHARD_SIZE = 4096  # klip / shard


def load_clip(path: str) -> torch.Tensor:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SRC_SR, (path, sr)
    return torch.from_numpy(np.ascontiguousarray(wav))


def resample_24k(wav: torch.Tensor) -> torch.Tensor:
    import torchaudio.functional as AF

    return AF.resample(wav, SRC_SR, SNAC_SR)


def encode_shard(model, rows, device, batch_sec: float) -> tuple[list[str], list[list[int]]]:
    """Uzunluğa göre sıralayıp dolgu israfını azaltarak kodlar."""
    rows = sorted(rows, key=lambda r: r["duration"])
    ids, codes = [], []
    batch: list[tuple[str, torch.Tensor]] = []
    budget = 0.0

    def flush():
        nonlocal batch, budget
        if not batch:
            return
        n_samples = [w.shape[-1] for _, w in batch]
        pad_to = -(-max(n_samples) // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
        x = torch.zeros(len(batch), 1, pad_to)
        for i, (_, w) in enumerate(batch):
            x[i, 0, : w.shape[-1]] = w
        flats = encode_batch(model, x.to(device), n_samples)
        for (cid, _), flat in zip(batch, flats):
            ids.append(cid)
            codes.append(flat)
        batch, budget = [], 0.0

    for r in rows:
        wav = resample_24k(load_clip(r["audio"]))
        batch.append((r["id"], wav))
        budget += r["duration"]
        if budget >= batch_sec:
            flush()
    flush()
    return ids, codes


def write_shard(path: Path, ids: list[str], codes: list[list[int]]) -> None:
    lens = np.array([len(c) for c in codes], dtype=np.int32)
    tokens = np.concatenate([np.asarray(c, dtype=np.uint16) for c in codes])
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ids=np.array(ids), lens=lens, tokens=tokens)
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-sec", type=float, default=240.0,
                    help="GPU batch'i basina toplam ses suresi")
    ap.add_argument("--limit", type=int, help="ilk N klip (deneme icin)")
    args = ap.parse_args()

    rows = []
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows.append({k: r[k] for k in ("id", "audio", "duration")})
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_snac(args.device)
    shards = [rows[i : i + SHARD_SIZE] for i in range(0, len(rows), SHARD_SIZE)]
    done = skipped = 0
    for si, shard_rows in enumerate(shards):
        path = args.out / f"shard-{si:05d}.npz"
        if path.exists():
            skipped += 1
            continue
        ids, codes = encode_shard(model, shard_rows, args.device, args.batch_sec)
        write_shard(path, ids, codes)
        done += 1
        n_tok = sum(len(c) for c in codes)
        print(f"shard {si+1}/{len(shards)}  klip={len(ids)}  token={n_tok:,}", flush=True)
    print(f"bitti: {done} shard yazildi, {skipped} atlandi -> {args.out}")


if __name__ == "__main__":
    main()
