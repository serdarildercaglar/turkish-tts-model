"""Korpus transkriptlerinden 4k'lık Türkçe ByteLevel BPE eğitir.

Nihai tokenizer üç bölgeyi tek sözlükte birleştirir (bkz. src/vocab.py):
metin BPE [0,4096), ses tokenları [4096,16384), özel+kontrol [16384,...).
Çıktı: --out dizinine HF `PreTrainedTokenizerFast` (tokenizer.json + config).

Depo kökünden:  python scripts/train_tokenizer.py \
    --manifest /path/to/hf_train.jsonl --out artifacts/tokenizer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codec import AUDIO_VOCAB
from src.vocab import ALL_ADDED, CONTROLS, SPECIALS, TEXT_VOCAB, VOCAB_SIZE, audio_token_name


def iter_texts(manifest: Path):
    """Hijyenden geçmiş manifestin metinleri — model tam olarak bunları görecek."""
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                t = json.loads(line).get("text")
                if t:
                    yield t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/tokenizer"))
    args = ap.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=TEXT_VOCAB,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_texts(args.manifest), trainer=trainer)
    got = tok.get_vocab_size()
    assert got == TEXT_VOCAB, f"BPE sozlugu {got}, beklenen {TEXT_VOCAB}"

    # Ses tokenlarini 8192'den, ozel tokenlari 36864'ten baslatacak sirada ekle.
    added = tok.add_tokens([audio_token_name(i) for i in range(AUDIO_VOCAB)])
    assert added == AUDIO_VOCAB
    tok.add_special_tokens(ALL_ADDED)
    for name, want in (("<|bos|>", None), (audio_token_name(0), TEXT_VOCAB)):
        if want is not None:
            assert tok.token_to_id(name) == want, (name, tok.token_to_id(name))

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        model_max_length=4096,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(args.out)
    # Sozluk ustveri dogrulamasi
    report = {
        "text_vocab": TEXT_VOCAB,
        "audio_vocab": AUDIO_VOCAB,
        "specials": {s: fast.convert_tokens_to_ids(s) for s in SPECIALS},
        "controls": {s: fast.convert_tokens_to_ids(s) for s in CONTROLS},
        "audio_0": fast.convert_tokens_to_ids(audio_token_name(0)),
        "audio_last": fast.convert_tokens_to_ids(audio_token_name(AUDIO_VOCAB - 1)),
        "len": len(fast),
        "vocab_size_padded": VOCAB_SIZE,
    }
    (args.out / "vocab_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
