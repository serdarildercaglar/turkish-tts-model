"""Paketlenmiş veri kümesi ve collator.

build_dataset.py'nin memmap çıktısını okur. Kayıp maskesi: her örnekte
`label_start` öncesi (bağlam: metin + varsa referans ses) ve dolgu -100.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.vocab import PAD


class PackedTTSDataset(Dataset):
    def __init__(self, packed_dir: str | Path):
        packed_dir = Path(packed_dir)
        idx = np.load(packed_dir / "index.npz")
        self.offsets = idx["offsets"]
        self.lengths = idx["lengths"]
        self.label_starts = idx["label_starts"]
        self.tokens = np.memmap(packed_dir / "tokens.bin", dtype=np.int32,
                                mode="r", shape=(int(self.offsets[-1]),))

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, i: int) -> dict:
        o, n = int(self.offsets[i]), int(self.lengths[i])
        ids = torch.from_numpy(np.asarray(self.tokens[o : o + n], dtype=np.int64))
        return {"input_ids": ids, "label_start": int(self.label_starts[i])}


class TTSCollator:
    """Sağ-dolgu; labels = input_ids, bağlam ve dolgu -100."""

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), PAD, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            n = len(ids)
            input_ids[i, :n] = ids
            attention[i, :n] = 1
            ls = b["label_start"]
            labels[i, ls:n] = ids[ls:]
        return {"input_ids": input_ids, "attention_mask": attention,
                "labels": labels}
