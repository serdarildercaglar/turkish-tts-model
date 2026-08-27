"""Checkpoint'lerin AYRILMIŞ veri üzerindeki kaybını ölçer — ezber noktasını bulur.

Eğitim kaybı tek başına converge ile ezberi ayırt etmez: model veriyi
ezberlemeye başladığında eğitim kaybı düşmeye devam ederken ayrılmış kayıp
yükselir. Dönüm noktası "daha fazla epoch fayda etmiyor" demektir.

Eğitim koşusunu değiştirmeye gerek yok; kaydedilmiş checkpoint'ler sonradan
taranır, dolayısıyla koşan bir eğitimi kesmez.

    python scripts/heldout_loss.py --ckpt-dir ckpt_pilot \
        --packed artifacts/packed_heldout --out artifacts/heldout_loss.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import PackedTTSDataset, TokenBudgetSampler, TTSCollator

_STEP = re.compile(r"checkpoint-(\d+)$")


@torch.inference_mode()
def evaluate(model, ds, collator, tokens_per_batch, device, max_batches=None):
    sampler = TokenBudgetSampler(ds.lengths, tokens_per_batch, seed=0)
    total_nll = 0.0
    total_tok = 0
    for k, idx in enumerate(sampler):
        if max_batches and k >= max_batches:
            break
        batch = collator([ds[i] for i in idx])
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        att = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=att).logits
        # kayip yalnizca etiketli (ses) pozisyonlarda; kaydirma HF ile ayni
        sl = logits[:, :-1].float()
        tl = labels[:, 1:]
        mask = tl != -100
        if not mask.any():
            continue
        nll = torch.nn.functional.cross_entropy(
            sl[mask], tl[mask], reduction="sum")
        total_nll += float(nll)
        total_tok += int(mask.sum())
    return total_nll / max(total_tok, 1), total_tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--packed", type=Path, required=True,
                    help="AYRILMIS paketlenmis kume (egitimde gorulmemis)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens-per-batch", type=int, default=8192)
    ap.add_argument("--max-batches", type=int, default=40,
                    help="checkpoint basina batch sayisi (hiz icin)")
    args = ap.parse_args()

    from transformers import LlamaForCausalLM

    ds = PackedTTSDataset(args.packed)
    collator = TTSCollator()
    print(f"ayrilmis kume: {len(ds):,} ornek", file=sys.stderr)

    ckpts = sorted(
        (p for p in args.ckpt_dir.iterdir() if _STEP.search(p.name)),
        key=lambda p: int(_STEP.search(p.name).group(1)))
    if (args.ckpt_dir / "final").is_dir():
        ckpts.append(args.ckpt_dir / "final")
    if not ckpts:
        raise SystemExit(f"{args.ckpt_dir} altinda checkpoint yok")

    rows = []
    best = None
    for c in ckpts:
        m = LlamaForCausalLM.from_pretrained(c, torch_dtype=torch.bfloat16)
        m = m.to(args.device).eval()
        loss, ntok = evaluate(m, ds, collator, args.tokens_per_batch,
                              args.device, args.max_batches)
        step = int(_STEP.search(c.name).group(1)) if _STEP.search(c.name) else -1
        rows.append({"checkpoint": c.name, "step": step,
                     "heldout_loss": round(loss, 4),
                     "heldout_ppl": round(float(np.exp(loss)), 2),
                     "tokens": ntok})
        flag = ""
        if best is None or loss < best:
            best, flag = loss, "  <- en iyi"
        print(f"{c.name:<20} kayip {loss:.4f}  ppl {np.exp(loss):8.2f}{flag}",
              flush=True)
        del m
        torch.cuda.empty_cache()

    # ezber donum noktasi: kaybin yukselmeye basladigi ilk checkpoint
    turn = None
    for i in range(1, len(rows)):
        if rows[i]["heldout_loss"] > rows[i - 1]["heldout_loss"]:
            turn = rows[i - 1]["checkpoint"]
            break
    print(f"\nen dusuk ayrilmis kayip: "
          f"{min(rows, key=lambda r: r['heldout_loss'])['checkpoint']}")
    print(f"ezber donum noktasi    : {turn or 'henuz yok (kayip hala dusuyor)'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"rows": rows, "donum": turn}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
