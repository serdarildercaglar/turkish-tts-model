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
    AutoConfig, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerFast,
    Trainer, TrainingArguments,
)
from torch.utils.data import DataLoader

from src.data import PackedTTSDataset, TokenBudgetSampler, TTSCollator
from src.prosody import Prosody


class TokenBudgetTrainer(Trainer):
    """Token bütçeli batching: batch başına dolgulu token sayısı sabit.

    Örnek uzunlukları 300–2600 arasında değiştiği için sabit batch boyutu ya
    kısa kovalarda GPU'yu boş bırakır ya uzun kovada belleği taşırır. Bütçe
    her ikisini de sabitler.
    """

    def __init__(self, *a, tokens_per_batch: int, **kw):
        super().__init__(*a, **kw)
        self.tokens_per_batch = tokens_per_batch

    def get_train_dataloader(self) -> DataLoader:
        ds = self.train_dataset
        sampler = TokenBudgetSampler(ds.lengths, self.tokens_per_batch,
                                     seed=self.args.seed)
        return DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def build_model(cfg_path: Path, tokenizer: PreTrainedTokenizerFast) -> PreTrainedModel:
    # model_type json'da (llama, qwen3, ...); config sinifi oradan secilir.
    d = json.loads(cfg_path.read_text())
    cfg = AutoConfig.for_model(d.pop("model_type"), **d)
    cfg.bos_token_id = tokenizer.bos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_config(cfg)
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
    if model.config.vocab_size < len(tokenizer):
        raise SystemExit(
            f"model sozlugu {model.config.vocab_size} < tokenizer {len(tokenizer)}")
    # Prozodi kova sinirlari modelin sozlesmesinin parcasi: cikarim ayni
    # sinirlari okumali, yoksa <|rate_k|> baska bir hizi ifade eder.
    prosody = Prosody.load(c["packed_dir"])

    targs = TrainingArguments(
        output_dir=c["output_dir"],
        per_device_train_batch_size=1,  # gercek batch TokenBudgetSampler'da
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
        # Trainer, modelin forward imzasinda olmayan alanlari collator'a
        # ulasmadan siler; `label_start` bizim kayip maskemizin kaynagi.
        remove_unused_columns=False,
        report_to=c.get("report_to", "none"),
        seed=c.get("seed", 42),
    )
    trainer = TokenBudgetTrainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=TTSCollator(),
        processing_class=tokenizer,
        tokens_per_batch=c.get("tokens_per_batch", 24576),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    final = Path(c["output_dir"]) / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    prosody.save(final)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
