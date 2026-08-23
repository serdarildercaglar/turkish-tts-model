import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PackedTTSDataset, TTSCollator
from src.vocab import PAD


def make_packed(tmp_path: Path):
    seqs = [([5, 6, 7, 100, 101, 102, 103], 3),   # label_start=3
            ([9, 10, 200, 201], 2)]
    lengths = np.array([len(s) for s, _ in seqs], dtype=np.int32)
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    mm = np.memmap(tmp_path / "tokens.bin", dtype=np.int32, mode="w+",
                   shape=(int(offsets[-1]),))
    for (s, _), o, n in zip(seqs, offsets[:-1], lengths):
        mm[o:o + n] = s
    mm.flush()
    np.savez(tmp_path / "index.npz", offsets=offsets, lengths=lengths,
             label_starts=np.array([ls for _, ls in seqs], dtype=np.int32))
    return seqs


def test_dataset_and_collator_masking(tmp_path):
    seqs = make_packed(tmp_path)
    ds = PackedTTSDataset(tmp_path)
    assert len(ds) == 2
    batch = TTSCollator()([ds[0], ds[1]])
    ids, labels, att = batch["input_ids"], batch["labels"], batch["attention_mask"]
    assert ids.shape == (2, 7)
    # ornek 0: ilk 3 pozisyon baglam (-100), kalan 4 etiketli
    assert (labels[0, :3] == -100).all()
    assert torch.equal(labels[0, 3:7], ids[0, 3:7])
    # ornek 1: dolgu -100 ve PAD, attention 0
    assert (labels[1, :2] == -100).all()
    assert torch.equal(labels[1, 2:4], ids[1, 2:4])
    assert (labels[1, 4:] == -100).all()
    assert (ids[1, 4:] == PAD).all()
    assert att[1].tolist() == [1, 1, 1, 1, 0, 0, 0]


def test_model_configs_param_counts():
    from transformers import LlamaConfig, LlamaForCausalLM

    for name, lo, hi in (("model_95m.json", 90e6, 100e6),
                         ("model_74m.json", 65e6, 80e6),
                         ("model_145m.json", 130e6, 160e6)):
        cfg = LlamaConfig(**json.loads((ROOT / "configs" / name).read_text()))
        with torch.device("meta"):
            m = LlamaForCausalLM(cfg)
        n = sum(p.numel() for p in m.parameters())
        assert lo < n < hi, (name, n)
