import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codec import (
    AUDIO_VOCAB, CODEBOOK_SIZE, LEVELS, SLOTS_PER_FRAME, SLOT_LEVEL,
    flatten_codes, remap_slot_to_level, unflatten_codes,
)
from src.prompt import build_prompt
from src.vocab import (
    ALL_ADDED, AUDIO_BASE, LOUD_ANY, LOUD_BASE, RATE_ANY, RATE_BASE,
    SPECIAL_BASE, TEXT_VOCAB, VOCAB_SIZE, audio_id_to_lm, is_audio_id,
    lm_id_to_audio, loud_token, rate_token,
)


def test_flatten_roundtrip():
    t = 13
    l0 = torch.randint(0, CODEBOOK_SIZE, (t,))
    l1 = torch.randint(0, CODEBOOK_SIZE, (2 * t,))
    l2 = torch.randint(0, CODEBOOK_SIZE, (4 * t,))
    flat = flatten_codes(l0, l1, l2)
    assert len(flat) == t * SLOTS_PER_FRAME
    # her yuva, KENDI DUZEYININ araliginda olmali
    for i, v in enumerate(flat):
        lv = SLOT_LEVEL[i % SLOTS_PER_FRAME]
        assert lv * CODEBOOK_SIZE <= v < (lv + 1) * CODEBOOK_SIZE
    r0, r1, r2 = unflatten_codes(flat)
    assert torch.equal(r0, l0) and torch.equal(r1, l1) and torch.equal(r2, l2)


def test_audio_vocab_is_three_codebooks():
    assert LEVELS == 3
    assert AUDIO_VOCAB == 3 * CODEBOOK_SIZE == 12288
    # yuva basina ofset olsaydi 28.672 olurdu; 16.384 giris kazandik
    assert 7 * CODEBOOK_SIZE - AUDIO_VOCAB == 16384


def test_unflatten_trims_partial_frame():
    t = 5
    z = torch.zeros(t, dtype=torch.long)
    flat = flatten_codes(z, torch.zeros(2 * t, dtype=torch.long),
                         torch.zeros(4 * t, dtype=torch.long)) + [3, 7]
    r0, _, _ = unflatten_codes(flat)
    assert r0.shape[-1] == t


def test_remap_slot_to_level_matches_flatten():
    """Eski yuva-ofsetli kodlar, yeniden kodlamadan duzey ofsetine cevrilir."""
    t = 9
    l0 = torch.randint(0, CODEBOOK_SIZE, (t,))
    l1 = torch.randint(0, CODEBOOK_SIZE, (2 * t,))
    l2 = torch.randint(0, CODEBOOK_SIZE, (4 * t,))
    # eski duzen: her yuva kendi 4096'lik araliginda
    old = np.empty(t * 7, dtype=np.int64)
    old[0::7] = l0.numpy() + 0 * CODEBOOK_SIZE
    old[1::7] = l1.numpy()[0::2] + 1 * CODEBOOK_SIZE
    old[2::7] = l1.numpy()[1::2] + 2 * CODEBOOK_SIZE
    old[3::7] = l2.numpy()[0::4] + 3 * CODEBOOK_SIZE
    old[4::7] = l2.numpy()[1::4] + 4 * CODEBOOK_SIZE
    old[5::7] = l2.numpy()[2::4] + 5 * CODEBOOK_SIZE
    old[6::7] = l2.numpy()[3::4] + 6 * CODEBOOK_SIZE
    assert np.array_equal(remap_slot_to_level(old),
                          np.asarray(flatten_codes(l0, l1, l2)))


def test_vocab_layout():
    assert AUDIO_BASE == TEXT_VOCAB == 4096
    assert SPECIAL_BASE == TEXT_VOCAB + AUDIO_VOCAB == 16384
    assert SPECIAL_BASE + len(ALL_ADDED) <= VOCAB_SIZE
    assert VOCAB_SIZE % 64 == 0
    assert is_audio_id(audio_id_to_lm(0))
    assert is_audio_id(audio_id_to_lm(AUDIO_VOCAB - 1))
    assert not is_audio_id(audio_id_to_lm(AUDIO_VOCAB))
    assert lm_id_to_audio(audio_id_to_lm(1234)) == 1234


def test_control_tokens():
    assert rate_token(None) == RATE_ANY
    assert rate_token(0) == RATE_BASE and rate_token(4) == RATE_BASE + 4
    assert loud_token(None) == LOUD_ANY
    assert loud_token(2) == LOUD_BASE + 2
    # kontrol tokenlari ses araligiyla cakismamali
    for t in (RATE_ANY, LOUD_ANY, RATE_BASE, LOUD_BASE):
        assert not is_audio_id(t) and t < VOCAB_SIZE


def test_prompt_shape():
    plain = build_prompt([1, 2, 3])
    clone = build_prompt([1, 2, 3], clone=True, ref_codes=[0, 5], rate=2, loud=1)
    # duz istem ses tokeniyla bitmez, klon istemi referans sesle biter
    assert not is_audio_id(plain[-1])
    assert is_audio_id(clone[-1])
    assert RATE_ANY in plain and LOUD_ANY in plain
    assert RATE_BASE + 2 in clone and LOUD_BASE + 1 in clone
    assert build_prompt([1], frag_start=True, frag_end=True) != build_prompt([1])
