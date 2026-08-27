"""Üretilen sesi telefon hattının kodeğine çevirir (G.711 μ-law / A-law, 8 kHz).

IVR çıkışı SIP/RTP üzerinden G.711'de taşınır: 8 kHz örnekleme, 8 bit
logaritmik nicemleme, 64 kbit/s. Model 24 kHz üretiyor; kullanıcının gerçekte
duyacağı şey bu dönüşümden sonrasıdır. Değerlendirmeyi ham 24 kHz üzerinde
yapmak, hattan asla teslim edilmeyecek bir kaliteyi ölçmek olur.

    python scripts/to_telephony.py ses.wav --out-dir cikti

Üretilenler:
    <ad>_g711u.wav   μ-law 8 kHz (Kuzey Amerika/Japonya, çoğu VoIP varsayılanı)
    <ad>_g711a.wav   A-law 8 kHz (Avrupa/Türkiye PSTN)
    <ad>_g711u_16k.wav  μ-law'dan geri açılmış 16 kHz (ASR'ye vermek için)

Ayrıca bandın ne götürdüğünü ölçer: 4 kHz üstü enerji kaybı ve nicemleme
gürültüsü.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TEL_SR = 8000


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(np.ascontiguousarray(x)).float()
    return AF.resample(t, sr_in, sr_out).numpy()


def band_energy_loss(orig: np.ndarray, sr: int) -> float:
    """4 kHz üstünde kalan enerjinin oranı — G.711'in attığı kısım."""
    spec = np.abs(np.fft.rfft(orig))
    freq = np.fft.rfftfreq(len(orig), 1 / sr)
    total = float(np.sum(spec ** 2))
    if total <= 0:
        return 0.0
    return float(np.sum(spec[freq > 4000] ** 2) / total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path, nargs="+")
    ap.add_argument("--out-dir", type=Path, default=Path("telephony"))
    ap.add_argument("--law", choices=("u", "a", "both"), default="both")
    args = ap.parse_args()

    import soundfile as sf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.wav:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        lost = band_energy_loss(x, sr)

        x8 = _resample(x, sr, TEL_SR) if sr != TEL_SR else x
        x8 = np.clip(x8, -1.0, 1.0)
        stem = path.stem

        made = []
        laws = ("u", "a") if args.law == "both" else (args.law,)
        for law in laws:
            subtype = "ULAW" if law == "u" else "ALAW"
            out = args.out_dir / f"{stem}_g711{law}.wav"
            sf.write(out, x8, TEL_SR, subtype=subtype)
            made.append(out)

        # kodekten geri acilmis hali: ASR'ye ve dinlemeye verilecek olan
        back = args.out_dir / f"{stem}_g711{laws[0]}_16k.wav"
        decoded, _ = sf.read(made[0], dtype="float32", always_2d=False)
        sf.write(back, _resample(decoded, TEL_SR, 16000), 16000)

        # nicemleme gurultusu: kodek oncesi/sonrasi fark
        noise = decoded - x8[: len(decoded)]
        snr = 10 * np.log10(
            (np.mean(x8[: len(decoded)] ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12))

        print(f"{path.name}  ({len(x) / sr:.2f} sn, {sr} Hz)")
        print(f"  4 kHz ustu atilan enerji : %{lost * 100:.2f}")
        print(f"  G.711 nicemleme SNR      : {snr:.1f} dB")
        for m in made:
            print(f"  -> {m}")
        print(f"  -> {back}  (geri acilmis, ASR/dinleme icin)")


if __name__ == "__main__":
    main()
