"""İstem kurulumu — eğitim, çıkarım ve değerlendirme aynı yerden okur.

`build_dataset.py`'nin ürettiği diziyle birebir aynı biçimi üretir; istem
biçimi tek bir yerde tanımlı olsun ki eğitimle çıkarım sessizce ayrışmasın.
"""
from __future__ import annotations

from src.vocab import (
    AUDIO_BASE, AUDIO_START, BOS, CLONE_TTS, FRAG_END, FRAG_START, PLAIN_TTS,
    TEXT_END, TEXT_START, loud_token, rate_token,
)


def build_prompt(text_ids: list[int], *, clone: bool = False,
                 ref_codes: list[int] | None = None,
                 rate: int | None = None, loud: int | None = None,
                 frag_start: bool = False, frag_end: bool = False) -> list[int]:
    """Üretime hazır istem token listesi.

    `rate` / `loud` None ise `<|rate_any|>` / `<|loud_any|>` yazılır: modelin
    eğitimde kontrol düşürülerek öğrendiği "ortalama" davranış.
    """
    head = [BOS, CLONE_TTS if clone else PLAIN_TTS,
            rate_token(rate), loud_token(loud)]
    if frag_start:
        head.append(FRAG_START)
    if frag_end:
        head.append(FRAG_END)
    seq = head + [TEXT_START, *text_ids, TEXT_END, AUDIO_START]
    if clone:
        if not ref_codes:
            raise ValueError("klon istemi icin ref_codes gerekir")
        seq += [c + AUDIO_BASE for c in ref_codes]
    return seq
