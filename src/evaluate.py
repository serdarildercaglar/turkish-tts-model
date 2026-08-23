"""Checkpoint değerlendirme döngüsü: üret → çöz → CER / konuşmacı-kosinüs / DNSMOS.

    python -m src.evaluate --config configs/train.yaml --checkpoint ckpt/checkpoint-2000

Sabit istem kümesi eval_set.jsonl'dan okunur (scripts/build_dataset.py'nin
kardeşi olarak scripts/make_eval_set.py üretir): her satır
{"kind": "plain"|"clone", "text": ..., "ref_audio": path, "ref_text": ...}.

Metrikler:
- CER: OpenAI-uyumlu Whisper servisi (config: whisper_url, whisper_model);
  normalizasyon: küçük harf + noktalama temizliği.
- Konuşmacı kosinüsü (yalnız clone): pyannote WeSpeaker ResNet34 gömmesi,
  üretim ↔ referans.
- DNSMOS OVRL: config'te verilen sig_bak_ovr.onnx ile.
Eksik bağımlılıkta ilgili metrik atlanır ve raporda null kalır.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codec import SNAC_SR, decode_to_wav, load_snac
from src.vocab import (
    AUDIO_BASE, AUDIO_END, AUDIO_START, BOS, CLONE_TTS, EOS, PLAIN_TTS,
    TEXT_END, TEXT_START, is_audio_id, lm_id_to_audio,
)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).casefold()
    s = "".join(c for c in s if not unicodedata.category(c).startswith("P"))
    return " ".join(s.split())


def cer(hyp: str, ref: str) -> float:
    h, r = norm_text(hyp), norm_text(ref)
    if not r:
        return float("nan")
    prev = list(range(len(r) + 1))
    for i, ch in enumerate(h, 1):
        cur = [i]
        for j, cr in enumerate(r, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ch != cr)))
        prev = cur
    return prev[-1] / len(r)


def transcribe(url: str, model: str, wav: np.ndarray) -> str | None:
    try:
        import requests
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, wav, SNAC_SR, format="WAV")
        buf.seek(0)
        r = requests.post(
            f"{url.rstrip('/')}/v1/audio/transcriptions",
            files={"file": ("gen.wav", buf, "audio/wav")},
            data={"model": model, "language": "tr", "temperature": 0},
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("text", "")
    except Exception as exc:  # metrik atlanir, kosu durmaz
        print(f"  ! whisper: {exc}", file=sys.stderr)
        return None


class SpeakerEmbedder:
    def __init__(self, device: str):
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        self.inf = Inference(model, window="whole", device=torch.device(device))

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import torchaudio.functional as AF

        x = torch.from_numpy(wav).float().unsqueeze(0)
        if sr != 16000:
            x = AF.resample(x, sr, 16000)
        emb = self.inf({"waveform": x, "sample_rate": 16000})
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)


class DNSMOS:
    """Ham dalga biçimli sig_bak_ovr.onnx — veri hattının dnsmos aşamasıyla
    birebir aynı pencereleme (9,01 sn, 1 sn kaydırma) ve polinom eşleme."""

    SR = 16000
    WIN = int(9.01 * SR)
    P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def ovrl(self, wav: np.ndarray, sr: int) -> float:
        import torchaudio.functional as AF

        x = wav.astype(np.float32)
        if sr != self.SR:
            x = AF.resample(torch.from_numpy(x), sr, self.SR).numpy()
        if x.size < self.WIN:
            reps = int(np.ceil(self.WIN / max(x.size, 1)))
            wins = [np.tile(x, reps)[: self.WIN]]
        else:
            wins = [x[s : s + self.WIN]
                    for s in range(0, x.size - self.WIN + 1, self.SR)]
        raws = []
        for s in range(0, len(wins), 8):
            batch = np.stack(wins[s : s + 8]).astype(np.float32)
            raws.append(self.sess.run(None, {self.input_name: batch})[0])
        raw = np.concatenate(raws).mean(axis=0)
        return float(self.P_OVR(raw[2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-set", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("eval"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import soundfile as sf
    import yaml
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

    c = yaml.safe_load(args.config.read_text())
    eval_path = args.eval_set or Path(c["eval_set"])
    prompts = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    tok = PreTrainedTokenizerFast.from_pretrained(c["tokenizer_dir"])
    model = LlamaForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16).to(args.device).eval()
    snac = load_snac(args.device)

    try:
        spk = SpeakerEmbedder(args.device)
    except Exception as exc:
        print(f"! konusmaci gommesi devre disi: {exc}", file=sys.stderr)
        spk = None
    try:
        dns = DNSMOS(c["dnsmos_onnx"]) if c.get("dnsmos_onnx") else None
    except Exception as exc:
        print(f"! dnsmos devre disi: {exc}", file=sys.stderr)
        dns = None

    from infer import encode_ref  # ayni istem kurulumu

    step = Path(args.checkpoint).name
    wav_dir = args.out_dir / step
    wav_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, p in enumerate(prompts):
        def tids(s):
            return tok(s, add_special_tokens=False)["input_ids"]

        if p["kind"] == "clone":
            ref_codes = encode_ref(snac, p["ref_audio"], args.device)
            prompt = [BOS, CLONE_TTS, TEXT_START,
                      *tids(p["ref_text"] + " " + p["text"]), TEXT_END,
                      AUDIO_START, *(x + AUDIO_BASE for x in ref_codes)]
        else:
            prompt = [BOS, PLAIN_TTS, TEXT_START, *tids(p["text"]), TEXT_END,
                      AUDIO_START]
        ids = torch.tensor([prompt], device=args.device)
        with torch.inference_mode():
            out = model.generate(ids, do_sample=True, temperature=0.7, top_p=0.9,
                                 repetition_penalty=1.1, max_new_tokens=1750,
                                 eos_token_id=[AUDIO_END, EOS],
                                 pad_token_id=tok.pad_token_id)
        gen = out[0, ids.shape[1]:].tolist()
        flat = [lm_id_to_audio(t) for t in gen if is_audio_id(t)]
        row = {"i": i, "kind": p["kind"], "n_tokens": len(flat)}
        if len(flat) >= 7:
            wav = decode_to_wav(snac, flat).numpy()
            sf.write(wav_dir / f"{i:03d}.wav", wav, SNAC_SR)
            row["dur_sec"] = round(len(wav) / SNAC_SR, 2)
            if c.get("whisper_url"):
                hyp = transcribe(c["whisper_url"], c.get("whisper_model", ""), wav)
                row["cer"] = cer(hyp, p["text"]) if hyp is not None else None
            if spk and p["kind"] == "clone":
                import soundfile as sf2
                ref_wav, ref_sr = sf2.read(p["ref_audio"], dtype="float32")
                row["spk_cos"] = float(np.dot(spk(wav, SNAC_SR), spk(ref_wav, ref_sr)))
            if dns:
                row["dnsmos_ovrl"] = dns.ovrl(wav, SNAC_SR)
        results.append(row)
        print(row, flush=True)

    def agg(key, kind=None):
        vals = [r[key] for r in results
                if r.get(key) is not None and (kind is None or r["kind"] == kind)]
        return float(np.mean(vals)) if vals else None

    summary = {
        "checkpoint": args.checkpoint,
        "n": len(results),
        "cer_plain": agg("cer", "plain"), "cer_clone": agg("cer", "clone"),
        "cer": agg("cer"), "spk_cos": agg("spk_cos"),
        "dnsmos_ovrl": agg("dnsmos_ovrl"),
        "empty_generations": sum(1 for r in results if r["n_tokens"] < 7),
    }
    (args.out_dir / f"{step}.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
