"""Paketlenmiş veri kümesi ve collator.

build_dataset.py'nin memmap çıktısını okur. Kayıp maskesi: her örnekte
`label_start` öncesi (bağlam: metin + varsa referans ses) ve dolgu -100.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

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


class TokenBudgetSampler(Sampler):
    """Sabit örnek sayısı yerine sabit TOKEN bütçesiyle batch kurar.

    Örnek uzunlukları 300 ile 2600 arasında değişiyor. Sabit batch boyutu,
    bütçeyi en uzun kovaya göre ayarlamayı zorunlu kılar; kısa kovalarda GPU'nun
    yarısı boşa gider, uzun kovada da bellek taşar. Token bütçesi ikisini de
    çözer: batch başına dolgulu token sayısı sabit kalır, dolayısıyla bellek de.

    Uzunluğa göre sıralanmış bloklar içinde batch kurulur (dolgu israfı düşük),
    sonra batch sırası karıştırılır (gradyan uzunluğa göre önyargılı olmasın).
    """

    def __init__(self, lengths, tokens_per_batch: int, seed: int = 0,
                 block_multiplier: int = 64, drop_last: bool = False):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.tokens_per_batch = int(tokens_per_batch)
        self.seed = seed
        self.block = max(1, block_multiplier) * max(
            1, self.tokens_per_batch // max(1, int(self.lengths.max())))
        self.drop_last = drop_last
        self.epoch = 0
        if self.lengths.max() > self.tokens_per_batch:
            raise ValueError(
                f"en uzun ornek {self.lengths.max()} token, batch butcesi "
                f"{self.tokens_per_batch}; butceyi buyutun ya da max_len'i kisin")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths))
        batches = []
        for s in range(0, len(order), self.block):
            blk = order[s:s + self.block]
            blk = blk[np.argsort(self.lengths[blk], kind="stable")]
            cur, longest = [], 0
            for i in blk:
                n = int(self.lengths[i])
                new_longest = max(longest, n)
                # dolgulu maliyet: en uzun ornek x batch boyutu
                if cur and new_longest * (len(cur) + 1) > self.tokens_per_batch:
                    batches.append(cur)
                    cur, longest = [int(i)], n
                else:
                    cur.append(int(i))
                    longest = new_longest
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
