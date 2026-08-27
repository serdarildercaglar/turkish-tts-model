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

    for name, lo, hi in (("model_63m.json", 60e6, 65e6),
                         ("model_51m.json", 48e6, 55e6),
                         ("model_58m.json", 54e6, 60e6)):
        cfg = LlamaConfig(**json.loads((ROOT / "configs" / name).read_text()))
        with torch.device("meta"):
            m = LlamaForCausalLM(cfg)
        n = sum(p.numel() for p in m.parameters())
        assert lo < n < hi, (name, n)
        # 65M ustu hicbir varyant olmamali: 3090'da rahat egitim hedefi
        assert n < 65e6, (name, n)
        # gomme tablosu modelin dortte birini yememeli (duzey-ofsetli ses sozlugu)
        emb = cfg.vocab_size * cfg.hidden_size
        assert emb / n < 0.20, (name, emb / n)


def test_packed_index_permutation_roundtrip(tmp_path):
    """build_dataset diziyi diske akitip yalnizca INDISI karistirir.

    tokens.bin uretim sirasinda kalir; offsets[i] artik kumulatif degil, i'nci
    ornegin dosyadaki baslangicidir. PackedTTSDataset'in bunu dogru okudugunu
    ve karistirmanin veriyi bozmadigini dogrular.
    """
    seqs = [[10, 11, 12], [20, 21], [30, 31, 32, 33], [40]]
    label_starts_build = [1, 1, 2, 0]

    starts, cursor = [], 0
    with open(tmp_path / "tokens.bin", "wb") as fh:
        for s in seqs:
            fh.write(np.asarray(s, dtype=np.int32).tobytes())
            starts.append(cursor)
            cursor += len(s)

    perm = np.array([2, 0, 3, 1])
    offsets = np.concatenate([np.asarray(starts, dtype=np.int64)[perm],
                              np.array([cursor], dtype=np.int64)])
    lengths = np.asarray([len(s) for s in seqs], dtype=np.int32)[perm]
    labels = np.asarray(label_starts_build, dtype=np.int32)[perm]
    np.savez(tmp_path / "index.npz", offsets=offsets, lengths=lengths,
             label_starts=labels)

    ds = PackedTTSDataset(tmp_path)
    assert len(ds) == len(seqs)
    for i, src in enumerate(perm):
        assert ds[i]["input_ids"].tolist() == seqs[src]
        assert ds[i]["label_start"] == label_starts_build[src]


def test_token_budget_sampler_caps_padded_cost():
    """Dolgulu maliyet (en uzun ornek x batch boyutu) butceyi asmamali.

    Sabit batch boyutu uzun kovada belleği taşırır; bütçe onu sabitler.
    """
    from src.data import TokenBudgetSampler

    rng = np.random.default_rng(0)
    lengths = rng.integers(300, 2600, size=5000)
    budget = 20480
    s = TokenBudgetSampler(lengths, budget, seed=1)

    seen = []
    for batch in s:
        assert batch, "bos batch"
        cost = int(lengths[batch].max()) * len(batch)
        assert cost <= budget, (cost, budget)
        seen.extend(batch)
    # her ornek tam olarak bir kez gorulmeli
    assert sorted(seen) == list(range(len(lengths)))
    assert len(s) == len(list(iter(s)))


def test_token_budget_sampler_rejects_oversized_example():
    from src.data import TokenBudgetSampler

    try:
        TokenBudgetSampler([100, 5000], 4096)
    except ValueError as exc:
        assert "butce" in str(exc)
    else:
        raise AssertionError("buyuk ornek icin hata bekleniyordu")
