"""SNAC codec köprüsü: ses ↔ düzleştirilmiş LM token kimlikleri.

SNAC 24 kHz **üç** RVQ kod kitabı kullanır: kaba L0 (T çerçeve), L1 (2T), L2 (4T).
Orpheus tarzı düzleştirme, kaba çerçeve başına 7 token yazar:

    [L0[i], L1[2i], L1[2i+1], L2[4i], L2[4i+1], L2[4i+2], L2[4i+3]]

**Sözlük düzeni: yuva değil, DÜZEY başına ofset.** Yedi yuvanın her birine kendi
4096'lık aralığını vermek (28.672 giriş) L1'i iki, L2'yi dört kez kopyalar —
oysa bunlar aynı kod kitabından gelir. Düzey başına ofset 3×4096 = 12.288 girişe
iner ve 16.384 gömme satırı kazandırır; 63M'lik bir modelde bu, gömme tablosunun
%25'ten %13'e düşmesi demektir. Çerçeve içindeki yuva bilgisi zaten konumdan
(RoPE) gelir, kimlikten gelmesi gerekmez.

Eski (yuva-ofsetli) kod shard'ları `remap_slot_to_level` ile dönüştürülebilir;
yeniden kodlama gerekmez, dönüşüm saf aritmetiktir.

LM sözlük düzeni (vocab.py'de sabitlenir):
    [0, TEXT_VOCAB)                      metin BPE
    [TEXT_VOCAB, TEXT_VOCAB+3*4096)      ses tokenları
    [SPECIAL_BASE, ...)                  özel + kontrol tokenları
"""
from __future__ import annotations

import torch

CODEBOOK_SIZE = 4096
SLOTS_PER_FRAME = 7
LEVELS = 3
# Düzleştirilmiş çerçevede her yuvanın hangi RVQ düzeyinden geldiği.
SLOT_LEVEL = (0, 1, 1, 2, 2, 2, 2)
AUDIO_VOCAB = LEVELS * CODEBOOK_SIZE  # 12.288
SNAC_MODEL = "hubertsiuzdak/snac_24khz"
SNAC_SR = 24000
# L0 çerçevesi başına örnek sayısı (24 kHz'te ~11,7 Hz kare hızı).
SAMPLES_PER_FRAME = 2048

_SLOT_OFFSET = torch.tensor(
    [lv * CODEBOOK_SIZE for lv in SLOT_LEVEL], dtype=torch.long)


def load_snac(device: str = "cuda"):
    from snac import SNAC

    model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)
    return model


def flatten_codes(l0: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor) -> list[int]:
    """Tek klibin üç düzey kodunu düzey-ofsetli düz listeye çevirir."""
    t = l0.shape[-1]
    assert l1.shape[-1] == 2 * t and l2.shape[-1] == 4 * t, (
        l0.shape, l1.shape, l2.shape)
    out = torch.empty(t * SLOTS_PER_FRAME, dtype=torch.int32)
    out[0::7] = l0 + 0 * CODEBOOK_SIZE
    out[1::7] = l1[0::2] + 1 * CODEBOOK_SIZE
    out[2::7] = l1[1::2] + 1 * CODEBOOK_SIZE
    out[3::7] = l2[0::4] + 2 * CODEBOOK_SIZE
    out[4::7] = l2[1::4] + 2 * CODEBOOK_SIZE
    out[5::7] = l2[2::4] + 2 * CODEBOOK_SIZE
    out[6::7] = l2[3::4] + 2 * CODEBOOK_SIZE
    return out.tolist()


def unflatten_codes(flat: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Düz listeyi SNAC'ın beklediği üç düzeye geri çevirir.

    Bozuk kuyruklar (7'nin katı olmayan uzunluk) kırpılır; yanlış aralığa düşen
    kimlikler kod kitabı aralığına kenetlenir — üretim sırasında LM'in tek tük
    hatası ses çıkışını tümden bozmasın.
    """
    t = len(flat) // SLOTS_PER_FRAME
    if t == 0:
        raise ValueError("en az bir tam çerçeve (7 token) gerekir")
    x = torch.tensor(flat[: t * SLOTS_PER_FRAME], dtype=torch.long).view(t, 7)
    x = (x - _SLOT_OFFSET).clamp_(0, CODEBOOK_SIZE - 1)
    l0 = x[:, 0]
    l1 = torch.stack([x[:, 1], x[:, 2]], dim=1).reshape(-1)
    l2 = torch.stack([x[:, 3], x[:, 4], x[:, 5], x[:, 6]], dim=1).reshape(-1)
    return l0, l1, l2


def remap_slot_to_level(flat):
    """Eski 7-yuva ofsetli kodları (0..28671) düzey ofsetine (0..12287) çevirir.

    Saf aritmetik: `kod + yuva*4096` → `kod + düzey*4096`. Kayıpsızdır ve sesi
    yeniden kodlamayı gerektirmez. `numpy` dizisi ya da liste kabul eder.
    """
    import numpy as np

    a = np.asarray(flat)
    n = (a.shape[-1] // SLOTS_PER_FRAME) * SLOTS_PER_FRAME
    a = a[..., :n].reshape(-1, SLOTS_PER_FRAME).astype(np.int64)
    slot_off = np.arange(SLOTS_PER_FRAME) * CODEBOOK_SIZE
    lvl_off = np.array([lv * CODEBOOK_SIZE for lv in SLOT_LEVEL])
    out = a - slot_off + lvl_off
    return out.reshape(-1)


@torch.inference_mode()
def encode_batch(model, wavs_24k: torch.Tensor, n_samples: list[int]) -> list[list[int]]:
    """SAMPLES_PER_FRAME katına sıfır-dolgulu [B, 1, T] batch'i kodlar.

    Her klibin kod uzunluğu kendi (dolgusuz) örnek sayısından türetilir;
    batch'teki en uzun klibe göre eklenen dolgu çerçeveleri atılır.
    """
    codes = model.encode(wavs_24k)
    out = []
    for i, n in enumerate(n_samples):
        frames = (n + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
        out.append(flatten_codes(
            codes[0][i, :frames].cpu(),
            codes[1][i, : 2 * frames].cpu(),
            codes[2][i, : 4 * frames].cpu(),
        ))
    return out


@torch.inference_mode()
def decode_to_wav(model, flat: list[int]) -> torch.Tensor:
    """Düz token listesinden 24 kHz mono dalga biçimi [T] üretir."""
    l0, l1, l2 = unflatten_codes(flat)
    device = next(model.parameters()).device
    codes = [c.unsqueeze(0).to(device) for c in (l0, l1, l2)]
    wav = model.decode(codes)
    return wav.squeeze().cpu()
