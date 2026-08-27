"""Eğitim örneklerini paketler: token dizileri + uzunluk/etiket indeksi.

Girdi: hijyenden geçmiş manifest (`prepare_manifest.py`), SNAC kod shard'ları
(`ingest_audio.py`) ve metin tokenizer'ı. Çıktı (--out dizininde):

    tokens.bin   int32 memmap, bütün örnekler art arda
    index.npz    offsets (N+1, int64), lengths (N, int32),
                 label_starts (N, int32)  # kaybın başladığı pozisyon
    prosody.json kova sınırları (çıkarımda aynısı okunur)
    stats.json   özet

İstem biçimi:

  düz:  <|bos|><|plain_tts|><|rate_k|><|loud_k|>[<|frag_start|>][<|frag_end|>]
        <|text_start|> metin <|text_end|><|audio_start|> SES <|audio_end|><|eos|>
        (kayıp: SES'ten itibaren)

  klon: <|bos|><|clone_tts|><|rate_k|><|loud_k|>[parça]<|text_start|>
        ref_metin hedef_metin <|text_end|><|audio_start|> REF_SES HEDEF_SES
        <|audio_end|><|eos|>            (kayıp: HEDEF_SES'ten itibaren)

Kontrol tokenları `--control-dropout` olasılığıyla `<|rate_any|>`/`<|loud_any|>`
ile değiştirilir; böylece çıkarımda kontrol verilmediğinde model çuvallamaz.

    python scripts/build_dataset.py \
        --manifest artifacts/manifest/train_all.jsonl \
        --codes artifacts/codes --tokenizer artifacts/tokenizer \
        --out artifacts/packed
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codec import remap_slot_to_level
from src.prosody import Prosody
from src.vocab import (
    AUDIO_BASE, AUDIO_END, AUDIO_START, BOS, CLONE_TTS, EOS, FRAG_END,
    FRAG_START, PLAIN_TTS, TEXT_END, TEXT_START, loud_token, rate_token,
)

MAX_LEN = 3072
MAX_REF_FRAMES = 117  # ~10 s (117 * 2048 / 24000)
CLONE_REPEAT = 2


def load_codes(codes_dir: Path, layout: str) -> dict[str, np.ndarray]:
    """`codes_dir` altındaki tüm shard'ları (alt dizinler dahil) okur."""
    table: dict[str, np.ndarray] = {}
    shards = sorted(codes_dir.rglob("shard-*.npz"))
    if not shards:
        raise SystemExit(f"{codes_dir} altinda shard bulunamadi")
    for shard in shards:
        z = np.load(shard, allow_pickle=False)
        ids, lens, tokens = z["ids"], z["lens"], z["tokens"]
        off = 0
        for cid, n in zip(ids, lens):
            flat = tokens[off:off + n]
            if layout == "slot":
                flat = remap_slot_to_level(flat)
            table[str(cid)] = np.asarray(flat, dtype=np.int32)
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
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--control-dropout", type=float, default=0.15,
                    help="kontrol tokenini <|..._any|> ile degistirme olasiligi")
    ap.add_argument("--codes-layout", choices=("level", "slot"), default="level",
                    help="'slot': eski 7-yuva ofsetli shard'lari donustur")
    ap.add_argument("--prosody", type=Path,
                    help="prosody.json dizini (varsayilan: manifest yani)")
    args = ap.parse_args()

    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    codes = load_codes(args.codes, args.codes_layout)
    print(f"kod tablosu: {len(codes):,} klip", file=sys.stderr)

    rows: dict[str, dict] = {}
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    print(f"manifest: {len(rows):,} satir", file=sys.stderr)

    pros = Prosody.load(args.prosody or args.manifest.parent)
    rng = random.Random(args.seed)

    def text_ids(s: str) -> list[int]:
        return tok(s, add_special_tokens=False)["input_ids"]

    def audio_ids(cid: str, max_frames: int | None = None) -> list[int] | None:
        flat = codes.get(cid)
        if flat is None:
            return None
        if max_frames is not None:
            flat = flat[: max_frames * 7]
        return (flat.astype(np.int64) + AUDIO_BASE).tolist()

    def control(r: dict) -> list[int]:
        """Prozodi kontrol tokenları; dropout ile `any` varyantına düşer."""
        rb = pros.rate_bucket(r.get("text", ""), r.get("duration"))
        lb = pros.loud_bucket(r.get("quality_lufs"))
        if rng.random() < args.control_dropout:
            rb = None
        if rng.random() < args.control_dropout:
            lb = None
        return [rate_token(rb), loud_token(lb)]

    def frag(r: dict) -> list[int]:
        out = []
        if r.get("frag_start"):
            out.append(FRAG_START)
        if r.get("frag_end"):
            out.append(FRAG_END)
        return out

    # Diziler bellekte TUTULMAZ: train+review'da ~1,8 milyar token, Python
    # listesi olarak ~58 GB eder. Her ornek uretildigi anda diske yazilir,
    # bellekte yalnizca uzunluk/etiket indisi kalir.
    args.out.mkdir(parents=True, exist_ok=True)
    bin_path = args.out / "tokens.bin"
    starts: list[int] = []
    lengths_l: list[int] = []
    labels_l: list[int] = []
    cursor = 0
    fbin = open(bin_path, "wb")

    def emit(seq: list[int], label_start: int, times: int = 1) -> None:
        """Diziyi bir kez yazar, indekse `times` kez kaydeder.

        Klon tekrarı aynı diziyi yeniden yazmaz; ofset paylaşılır. Örnekleyici
        açısından fark yoktur, disk yarıya iner.
        """
        nonlocal cursor
        fbin.write(np.asarray(seq, dtype=np.int32).tobytes())
        for _ in range(times):
            starts.append(cursor)
            lengths_l.append(len(seq))
            labels_l.append(label_start)
        cursor += len(seq)

    n_plain = 0
    miss_audio = too_long = miss_ref = 0

    for r in rows.values():
        au = audio_ids(r["id"])
        if au is None:
            miss_audio += 1
            continue
        seq = [BOS, PLAIN_TTS, *control(r), *frag(r), TEXT_START,
               *text_ids(r["text"]), TEXT_END, AUDIO_START, *au, AUDIO_END, EOS]
        if len(seq) > args.max_len:
            too_long += 1
            continue
        emit(seq, seq.index(AUDIO_START) + 1)
        n_plain += 1

    for r in rows.values():
        ref_id = r.get("ref_id")
        if not ref_id or ref_id in ("None", ""):
            continue
        ref = rows.get(ref_id)
        au_t = audio_ids(r["id"])
        au_r = audio_ids(ref_id, max_frames=MAX_REF_FRAMES) if ref else None
        if ref is None or au_t is None or au_r is None:
            miss_ref += 1
            continue
        seq = [BOS, CLONE_TTS, *control(r), *frag(r), TEXT_START,
               *text_ids(ref["text"] + " " + r["text"]), TEXT_END,
               AUDIO_START, *au_r, *au_t, AUDIO_END, EOS]
        if len(seq) > args.max_len:
            too_long += 1
            continue
        label_start = seq.index(AUDIO_START) + 1 + len(au_r)
        emit(seq, label_start, times=args.clone_repeat)

    fbin.close()
    n_examples = len(starts)
    if n_examples == 0:
        raise SystemExit("hic ornek uretilmedi; manifest ve kodlar eslesmiyor olabilir")

    # Karistirma yalnizca INDISTE yapilir; tokens.bin uretim sirasinda kalir.
    # PackedTTSDataset ofsetle eristigi icin bu, veriyi diskte tasimadan
    # karistirmaya esdegerdir.
    starts_a = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths_l, dtype=np.int32)
    label_starts = np.asarray(labels_l, dtype=np.int32)
    perm = np.random.default_rng(args.seed).permutation(n_examples)
    offsets = np.concatenate([starts_a[perm], np.array([cursor], dtype=np.int64)])
    lengths = lengths[perm]
    label_starts = label_starts[perm]

    np.savez(args.out / "index.npz", offsets=offsets, lengths=lengths,
             label_starts=label_starts)
    pros.save(args.out)

    stats = dict(
        examples=n_examples, plain=n_plain, clone=n_examples - n_plain,
        tokens=int(cursor), mean_len=float(lengths.mean()),
        p95_len=int(np.percentile(lengths, 95)), max_len=int(lengths.max()),
        miss_audio=miss_audio, miss_ref=miss_ref, too_long=too_long,
        control_dropout=args.control_dropout, prosody=pros.to_dict(),
    )
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
