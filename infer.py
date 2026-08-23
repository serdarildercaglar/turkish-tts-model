"""Yerel çıkarım: metin (+ isteğe bağlı referans ses) → 24 kHz wav.

Düz TTS:
    python infer.py --model ckpt/final --tokenizer artifacts/tokenizer \
        --text "Merhaba, bu bir deneme." --out cikti.wav

Ses klonlama (referans ses + referans metin):
    python infer.py ... --ref-audio ref.wav --ref-text "Referans cümlesi."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.codec import SNAC_SR, decode_to_wav, encode_batch, load_snac, SAMPLES_PER_FRAME
from src.vocab import (
    AUDIO_END, AUDIO_START, BOS, CLONE_TTS, EOS, PLAIN_TTS,
    TEXT_END, TEXT_START, is_audio_id, lm_id_to_audio,
)

MAX_REF_FRAMES = 117


def encode_ref(snac, path: str, device: str) -> list[int]:
    import soundfile as sf
    import torchaudio.functional as AF

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = torch.from_numpy(wav)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if sr != SNAC_SR:
        wav = AF.resample(wav, sr, SNAC_SR)
    n = wav.shape[-1]
    pad_to = -(-n // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    x = torch.zeros(1, 1, pad_to)
    x[0, 0, :n] = wav
    flat = encode_batch(snac, x.to(device), [n])[0]
    return flat[: MAX_REF_FRAMES * 7]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    ap.add_argument("--out", default="cikti.wav")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=1750)
    args = ap.parse_args()
    if bool(args.ref_audio) != bool(args.ref_text):
        ap.error("--ref-audio ve --ref-text birlikte verilmeli")

    import soundfile as sf
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    snac = load_snac(args.device)

    def tids(s: str) -> list[int]:
        return tok(s, add_special_tokens=False)["input_ids"]

    from src.vocab import AUDIO_BASE
    if args.ref_audio:
        ref_codes = encode_ref(snac, args.ref_audio, args.device)
        prompt = [BOS, CLONE_TTS, TEXT_START,
                  *tids(args.ref_text + " " + args.text), TEXT_END,
                  AUDIO_START, *(c + AUDIO_BASE for c in ref_codes)]
    else:
        prompt = [BOS, PLAIN_TTS, TEXT_START, *tids(args.text), TEXT_END,
                  AUDIO_START]

    ids = torch.tensor([prompt], device=args.device)
    with torch.inference_mode():
        out = model.generate(
            ids,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=[AUDIO_END, EOS],
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0, ids.shape[1]:].tolist()
    flat = [lm_id_to_audio(t) for t in gen if is_audio_id(t)]
    if len(flat) < 7:
        raise SystemExit("model ses tokeni uretmedi; checkpoint/istem kontrol edin")
    wav = decode_to_wav(snac, flat)
    sf.write(args.out, wav.numpy(), SNAC_SR)
    print(f"{args.out}: {wav.shape[-1]/SNAC_SR:.2f} s  ({len(flat)} ses tokeni)")


if __name__ == "__main__":
    main()
