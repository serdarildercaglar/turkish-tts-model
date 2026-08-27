"""Paketlenmiş kümenin doğruluğunu uçtan uca doğrular — eğitimden ÖNCE.

Rastgele örnekler seçer, istem yapısını çözer ve ses tokenlarını gerçek dalga
biçimine geri çevirir. Sessiz bir kodlama/ofset hatası ancak burada yakalanır;
eğitim başladıktan sonra kayıp düşüyor gibi görünüp çıktı gürültü olabilir.

    python scripts/verify_packed.py --packed artifacts/packed_pilot \
        --tokenizer artifacts/tokenizer --out artifacts/verify --n 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codec import SNAC_SR, decode_to_wav, load_snac
from src.data import PackedTTSDataset
from src.prosody import Prosody
from src.vocab import (
    ALL_ADDED, AUDIO_END, AUDIO_START, CLONE_TTS, EOS, PLAIN_TTS, SPECIAL_BASE,
    TEXT_END, TEXT_START, is_audio_id, lm_id_to_audio,
)

NAME = {SPECIAL_BASE + i: n for i, n in enumerate(ALL_ADDED)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/verify"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    ds = PackedTTSDataset(args.packed)
    pros = Prosody.load(args.packed)
    print(f"ornek: {len(ds):,} | prozodi sinirlari: {pros.to_dict()}")

    args.out.mkdir(parents=True, exist_ok=True)
    snac = load_snac(args.device)
    rng = np.random.default_rng(args.seed)

    n_clone = n_plain = 0
    for k, i in enumerate(rng.choice(len(ds), size=args.n, replace=False)):
        item = ds[int(i)]
        ids = item["input_ids"].tolist()
        ls = item["label_start"]

        kind = "klon" if CLONE_TTS in ids[:4] else "duz"
        n_clone += kind == "klon"
        n_plain += kind == "duz"

        head = [NAME.get(t, str(t)) for t in ids[: ids.index(TEXT_START)]]
        text = tok.decode(ids[ids.index(TEXT_START) + 1: ids.index(TEXT_END)])

        audio = [lm_id_to_audio(t) for t in ids if is_audio_id(t)]
        # kayip yalnizca label_start'tan itibaren; klon isteminde referans ses
        # bunun ONCESINDE kalir
        target = [lm_id_to_audio(t) for t in ids[ls:] if is_audio_id(t)]

        assert ids[-1] == EOS and ids[-2] == AUDIO_END, "dizi EOS ile bitmiyor"
        assert AUDIO_START in ids, "AUDIO_START yok"
        assert len(audio) % 7 == 0, f"ses tokeni 7'nin katı degil: {len(audio)}"

        wav = decode_to_wav(snac, target)
        path = args.out / f"{k:02d}_{kind}.wav"
        sf.write(path, wav.numpy(), SNAC_SR)

        print(f"\n[{k}] {kind}  uzunluk={len(ids)}  label_start={ls}")
        print(f"  istem basi : {' '.join(head)}")
        print(f"  metin      : {text[:90]}")
        print(f"  ses tokeni : toplam {len(audio)} | hedef {len(target)} "
              f"({len(target) / 7 / (SNAC_SR / 2048):.2f} sn)")
        print(f"  cozuldu    : {path}  ({wav.shape[-1] / SNAC_SR:.2f} sn, "
              f"tepe {wav.abs().max():.3f})")

    print(f"\n{n_plain} duz + {n_clone} klon ornek dogrulandi -> {args.out}")


if __name__ == "__main__":
    main()
