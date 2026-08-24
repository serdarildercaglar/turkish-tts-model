"""vLLM decode hızı ölçümü: adım süresi → RTF (83 token = 1 sn ses).

Rastgele ağırlıklı (ya da eğitilmiş) Llama'yı vLLM ile yükler; batch 1/8/16/32/64'te
1.000 tokenlik üretimi zamanlar. Sunum GPU'sunda (H100/H200) koşturup
docs/serving_vllm_omni.md'deki 3090 tablosuyla karşılaştırın.

    VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/bench_vllm_decode.py \
        --config configs/model_95m.json --tokenizer artifacts/tokenizer --out /tmp/bench95m
    # eğitilmiş checkpoint: --model ckpt/final (config/tokenizer gerekmez)

FlashInfer örnekleyicisi nvcc ister; env değişkeni onu kapatır (ölçümü etkilemez).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def build_random_model(config: str, tokenizer: str, out: Path) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    if not (out / "config.json").exists():
        torch.manual_seed(0)
        m = LlamaForCausalLM(LlamaConfig(**json.load(open(config)))).to(torch.bfloat16)
        print(f"params: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
        m.save_pretrained(out)
        PreTrainedTokenizerFast.from_pretrained(tokenizer).save_pretrained(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="eğitilmiş HF checkpoint dizini")
    ap.add_argument("--config", help="rastgele model için configs/model_*.json")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--out", default="/tmp/bench_model")
    ap.add_argument("--tokens", type=int, default=1000)
    ap.add_argument("--prompt-len", type=int, default=200)
    ap.add_argument("--batches", default="1,1,8,16,32,64")
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    args = ap.parse_args()
    if not args.model and not args.config:
        ap.error("--model ya da --config verin")
    mdir = Path(args.model) if args.model else build_random_model(args.config, args.tokenizer, Path(args.out))

    from vllm import LLM, SamplingParams

    llm = LLM(model=str(mdir), dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enforce_eager=False,
              enable_prefix_caching=False, max_num_seqs=128)
    n = args.tokens
    sp = SamplingParams(temperature=1.0, max_tokens=n, min_tokens=n, ignore_eos=True)
    prompt = [36864] + list(range(100, 100 + args.prompt_len))
    for bs in (int(b) for b in args.batches.split(",")):
        reqs = [{"prompt_token_ids": prompt}] * bs
        torch.cuda.synchronize()
        t = time.perf_counter()
        outs = llm.generate(reqs, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        tot = sum(len(o.outputs[0].token_ids) for o in outs)
        per_seq = tot / bs / dt
        print(f"batch={bs:3d} step={1000 * dt / n:.2f} ms  per-seq={per_seq:,.0f} tok/s  "
              f"agg={tot / dt:,.0f} tok/s (={tot / dt / 83:.0f} ses-s/s)  "
              f"RTF={83 / per_seq:.3f}", flush=True)


if __name__ == "__main__":
    main()
