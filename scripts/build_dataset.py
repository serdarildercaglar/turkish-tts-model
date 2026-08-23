"""Eğitim örneklerini paketler: token dizileri + uzunluk/etiket indeksi.

Girdi: hf_train.jsonl, tokenizer ve tokenize_audio.py'nin kod shard'ları.
Çıktı (--out dizininde):

    tokens.bin   int32 memmap, bütün örnekler art arda
    index.npz    offsets (N+1, int64), lengths (N, int32),
                 label_starts (N, int32)  # kaybın başladığı pozisyon

İki örnek türü (bkz. plan):
  düz:  <|bos|><|plain_tts|><|text_start|> metin <|text_end|><|audio_start|>
        SES <|audio_end|><|eos|>                       (kayıp: SES'ten itibaren)
  klon: <|bos|><|clone_tts|><|text_start|> ref_metin hedef_metin <|text_end|>
        <|audio_start|> REF_SES HEDEF_SES <|audio_end|><|eos|>
                                                (kayıp: HEDEF_SES'ten itibaren)

Depo kökünden:
    python scripts/build_dataset.py --manifest .../hf_train.jsonl \
        --codes artifacts/codes/train --tokenizer artifacts/tokenizer --out artifacts/packed
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vocab import (
    AUDIO_BASE, AUDIO_END, AUDIO_START, BOS, CLONE_TTS, EOS,
    PLAIN_TTS, TEXT_END, TEXT_START,
)

MAX_LEN = 3072
MAX_REF_FRAMES = 117  # ~10 s (117 * 2048 / 24000)
CLONE_REPEAT = 2


def load_codes(codes_dir: Path) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for shard in sorted(codes_dir.glob("shard-*.npz")):
        z = np.load(shard, allow_pickle=False)
        ids, lens, tokens = z["ids"], z["lens"], z["tokens"]
        off = 0
        for cid, n in zip(ids, lens):
            table[str(cid)] = tokens[off : off + n]
            off += int(n)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--codes", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clone-repeat", type=int, default=CLONE_REPEAT)
    args = ap.parse_args()

    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    codes = load_codes(args.codes)
    print(f"kod tablosu: {len(codes):,} klip", file=sys.stderr)

    rows = {}
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r

    def text_ids(s: str) -> list[int]:
        return tok(s, add_special_tokens=False)["input_ids"]

    def audio_ids(cid: str, max_frames: int | None = None) -> list[int] | None:
        flat = codes.get(cid)
        if flat is None:
            return None
        if max_frames is not None:
            flat = flat[: max_frames * 7]
        return (flat.astype(np.int32) + AUDIO_BASE).tolist()

    examples: list[tuple[list[int], int]] = []  # (dizi, label_start)
    miss_audio = too_long = miss_ref = 0

    for r in rows.values():
        au = audio_ids(r["id"])
        if au is None:
            miss_audio += 1
            continue
        seq = [BOS, PLAIN_TTS, TEXT_START, *text_ids(r["text"]), TEXT_END,
               AUDIO_START, *au, AUDIO_END, EOS]
        if len(seq) > MAX_LEN:
            too_long += 1
            continue
        examples.append((seq, seq.index(AUDIO_START) + 1))

    n_plain = len(examples)
    for r in rows.values():
        ref_id = r.get("ref_id")
        if not ref_id:
            continue
        ref = rows.get(ref_id)
        au_t = audio_ids(r["id"])
        au_r = audio_ids(ref_id, max_frames=MAX_REF_FRAMES) if ref else None
        if ref is None or au_t is None or au_r is None:
            miss_ref += 1
            continue
        seq = [BOS, CLONE_TTS, TEXT_START,
               *text_ids(ref["text"] + " " + r["text"]), TEXT_END,
               AUDIO_START, *au_r, *au_t, AUDIO_END, EOS]
        if len(seq) > MAX_LEN:
            too_long += 1
            continue
        label_start = seq.index(AUDIO_START) + 1 + len(au_r)
        for _ in range(args.clone_repeat):
            examples.append((seq, label_start))

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    args.out.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(s) for s, _ in examples], dtype=np.int32)
    offsets = np.zeros(len(examples) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    label_starts = np.array([ls for _, ls in examples], dtype=np.int32)

    mm = np.memmap(args.out / "tokens.bin", dtype=np.int32, mode="w+",
                   shape=(int(offsets[-1]),))
    for (seq, _), o, n in zip(examples, offsets[:-1], lengths):
        mm[o : o + n] = seq
    mm.flush()
    np.savez(args.out / "index.npz", offsets=offsets, lengths=lengths,
             label_starts=label_starts)

    stats = dict(
        examples=len(examples), plain=n_plain, clone=len(examples) - n_plain,
        tokens=int(offsets[-1]), mean_len=float(lengths.mean()),
        p95_len=int(np.percentile(lengths, 95)), max_len=int(lengths.max()),
        miss_audio=miss_audio, miss_ref=miss_ref, too_long=too_long,
    )
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
