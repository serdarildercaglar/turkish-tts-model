"""Eğitim girişi: sıfırdan LlamaForCausalLM, HF Trainer, tek GPU.

    python -m src.train --config configs/train.yaml [--resume]

Checkpoint'ler atomik kaydedilir; elektrik kesintisinden sonra --resume ile
kaldığı yerden devam eder. Kaydedilen model stok Llama'dır: vllm-omni'nin AR
aşaması ek ağırlık olmadan yükler (docs/serving_vllm_omni.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import (
    LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
    Trainer, TrainingArguments,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

from src.data import PackedTTSDataset, TTSCollator


class LengthBucketTrainer(Trainer):
    """Uzunluk kovalı örnekleyici: dolgu israfını düşürür."""

    def _get_train_sampler(self, *args, **kwargs):
        ds = self.train_dataset
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=ds,
            lengths=[int(x) for x in ds.lengths],
        )


def build_model(cfg_path: Path, tokenizer: PreTrainedTokenizerFast) -> LlamaForCausalLM:
    cfg = LlamaConfig(**json.loads(cfg_path.read_text()))
    cfg.bos_token_id = tokenizer.bos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id
    model = LlamaForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"model: {n/1e6:.1f}M parametre")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    c = yaml.safe_load(args.config.read_text())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(c["tokenizer_dir"])
    dataset = PackedTTSDataset(c["packed_dir"])
    model = build_model(Path(c["model_config"]), tokenizer)

    targs = TrainingArguments(
        output_dir=c["output_dir"],
        per_device_train_batch_size=c.get("micro_batch", 16),
        gradient_accumulation_steps=c.get("grad_accum", 4),
        num_train_epochs=c.get("epochs", 8),
        learning_rate=c.get("lr", 3e-4),
        lr_scheduler_type="cosine",
        warmup_steps=c.get("warmup_steps", 1000),
        weight_decay=c.get("weight_decay", 0.1),
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=c.get("logging_steps", 50),
        save_steps=c.get("save_steps", 2000),
        save_total_limit=c.get("save_total_limit", 4),
        dataloader_num_workers=c.get("num_workers", 4),
        report_to=c.get("report_to", "none"),
        seed=c.get("seed", 42),
    )
    trainer = LengthBucketTrainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=TTSCollator(),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(Path(c["output_dir"]) / "final"))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
