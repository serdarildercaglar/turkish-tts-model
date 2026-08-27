"""Seçilen checkpoint'i Hugging Face model deposu olarak paketler/yükler.

    python scripts/export_hf.py --checkpoint ckpt/checkpoint-XXXX \
        --tokenizer artifacts/tokenizer --repo serdarildercaglar/turkish-tts-model \
        [--push]

--push verilmezse yalnızca yerel export dizini hazırlanır. Model kartı
MODEL_CARD.md'den alınır; eval özeti --eval-json ile karta işlenir.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--eval-json", type=Path)
    ap.add_argument("--out", type=Path, default=Path("export"))
    ap.add_argument("--packed", type=Path,
                    help="prosody.json'un arananacagi paketlenmis veri dizini")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

    args.out.mkdir(parents=True, exist_ok=True)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint)
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    # prosody.json checkpoint ile birlikte tasinmali: kova sinirlari olmadan
    # <|rate_k|> tokenlari cikarimda baska bir hizi ifade eder
    for src_dir in (args.checkpoint, args.checkpoint.parent, args.packed):
        if src_dir and (Path(src_dir) / "prosody.json").is_file():
            shutil.copy(Path(src_dir) / "prosody.json", args.out / "prosody.json")
            break
    else:
        print("! prosody.json bulunamadi; cikarim yedek sinirlari kullanir")

    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    if args.eval_json and args.eval_json.is_file():
        summary = json.loads(args.eval_json.read_text()).get("summary", {})
        table = "\n".join(f"| {k} | {v} |" for k, v in summary.items())
        card = card.replace("<!--EVAL-->", f"| metrik | değer |\n|---|---|\n{table}")
    (args.out / "README.md").write_text(card, encoding="utf-8")
    for extra in ("LICENSE",):
        shutil.copy(ROOT / extra, args.out / extra)
    print(f"export hazir: {args.out}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="model",
                          folder_path=str(args.out),
                          commit_message="Upload turkish-tts-model checkpoint")
        print(f"yuklendi: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
