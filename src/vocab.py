"""LM sözlük düzeni — tek doğruluk kaynağı.

Metin BPE, ses tokenları ve özel tokenlar tek embedding tablosunu paylaşır.
Ses tokenları tokenizer'da `<custom_token_N>` adlı GERÇEK tokenlardır;
vLLM motoru (vllm-omni'nin AR aşaması) bu sayede onları sıradan token olarak
üretir; SNAC çözücü aşaması ya da istemci `N`'i koda çevirir.
"""
from src.codec import AUDIO_VOCAB

TEXT_VOCAB = 8192
AUDIO_BASE = TEXT_VOCAB                      # 8192
SPECIAL_BASE = TEXT_VOCAB + AUDIO_VOCAB      # 36864

SPECIALS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|text_start|>",
    "<|text_end|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|plain_tts|>",
    "<|clone_tts|>",
]
# 64'ün katına yuvarlanmış toplam sözlük (verimli matmul/embedding için).
VOCAB_SIZE = 36928

BOS = SPECIAL_BASE + SPECIALS.index("<|bos|>")
EOS = SPECIAL_BASE + SPECIALS.index("<|eos|>")
PAD = SPECIAL_BASE + SPECIALS.index("<|pad|>")
TEXT_START = SPECIAL_BASE + SPECIALS.index("<|text_start|>")
TEXT_END = SPECIAL_BASE + SPECIALS.index("<|text_end|>")
AUDIO_START = SPECIAL_BASE + SPECIALS.index("<|audio_start|>")
AUDIO_END = SPECIAL_BASE + SPECIALS.index("<|audio_end|>")
PLAIN_TTS = SPECIAL_BASE + SPECIALS.index("<|plain_tts|>")
CLONE_TTS = SPECIAL_BASE + SPECIALS.index("<|clone_tts|>")

assert SPECIAL_BASE + len(SPECIALS) <= VOCAB_SIZE


def audio_token_name(i: int) -> str:
    return f"<custom_token_{i}>"


def audio_id_to_lm(flat_code: int) -> int:
    return AUDIO_BASE + flat_code


def lm_id_to_audio(lm_id: int) -> int:
    return lm_id - AUDIO_BASE


def is_audio_id(lm_id: int) -> bool:
    return AUDIO_BASE <= lm_id < AUDIO_BASE + AUDIO_VOCAB
