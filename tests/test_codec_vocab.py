import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codec import (
    AUDIO_VOCAB, CODEBOOK_SIZE, SLOTS_PER_FRAME, flatten_codes, unflatten_codes,
)
from src.vocab import (
    AUDIO_BASE, SPECIAL_BASE, SPECIALS, VOCAB_SIZE,
    audio_id_to_lm, is_audio_id, lm_id_to_audio,
)


def test_flatten_roundtrip():
    t = 13
    l0 = torch.randint(0, CODEBOOK_SIZE, (t,))
    l1 = torch.randint(0, CODEBOOK_SIZE, (2 * t,))
    l2 = torch.randint(0, CODEBOOK_SIZE, (4 * t,))
    flat = flatten_codes(l0, l1, l2)
    assert len(flat) == t * SLOTS_PER_FRAME
    # her yuva kendi araligindadir
    for i, v in enumerate(flat):
        slot = i % SLOTS_PER_FRAME
        assert slot * CODEBOOK_SIZE <= v < (slot + 1) * CODEBOOK_SIZE
    r0, r1, r2 = unflatten_codes(flat)
    assert torch.equal(r0, l0) and torch.equal(r1, l1) and torch.equal(r2, l2)


def test_unflatten_trims_partial_frame():
    t = 5
    l0 = torch.zeros(t, dtype=torch.long)
    l1 = torch.zeros(2 * t, dtype=torch.long)
    l2 = torch.zeros(4 * t, dtype=torch.long)
    flat = flatten_codes(l0, l1, l2) + [3, 7]  # bozuk kuyruk
    r0, _, _ = unflatten_codes(flat)
    assert r0.shape[-1] == t


def test_vocab_layout():
    assert AUDIO_BASE == 8192
    assert SPECIAL_BASE == 8192 + AUDIO_VOCAB == 36864
    assert SPECIAL_BASE + len(SPECIALS) <= VOCAB_SIZE
    assert is_audio_id(audio_id_to_lm(0))
    assert is_audio_id(audio_id_to_lm(AUDIO_VOCAB - 1))
    assert not is_audio_id(audio_id_to_lm(AUDIO_VOCAB))
    assert lm_id_to_audio(audio_id_to_lm(1234)) == 1234
