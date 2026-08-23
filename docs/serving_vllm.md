# vLLM ile servis

Checkpoint stok `LlamaForCausalLM` olduğu için vLLM'e özel hiçbir şey
gerekmez:

```bash
vllm serve serdarildercaglar/turkish-tts-model --port 8001
# ya da yerel checkpoint: vllm serve ckpt/final --tokenizer artifacts/tokenizer --port 8001
```

Model, ses tokenlarını `<custom_token_N>` adlı gerçek tokenlar olarak üretir;
yani vLLM'in gözünde bu sıradan bir metin tamamlama isteğidir. İstemci akışı:

1. İstemi kur (düz TTS ya da klonlama — biçim için README).
2. `/v1/completions` çağır; durdurucular: `<|audio_end|>`, `<|eos|>`.
3. Dönen metindeki `<custom_token_N>` dizisini ayrıştır, `N` değerlerini
   SNAC kodlarına çevir ve sesi çöz.

## Örnek istemci

```python
import re, requests, soundfile as sf, sys
sys.path.insert(0, ".")  # depo kökünden çalıştırın
from src.codec import decode_to_wav, load_snac, SNAC_SR

PROMPT = ("<|bos|><|plain_tts|><|text_start|>"
          "Merhaba, bugün hava çok güzel."
          "<|text_end|><|audio_start|>")

r = requests.post("http://localhost:8001/v1/completions", json={
    "model": "serdarildercaglar/turkish-tts-model",
    "prompt": PROMPT,
    "max_tokens": 1750,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "stop": ["<|audio_end|>", "<|eos|>"],
})
text = r.json()["choices"][0]["text"]
flat = [int(m) for m in re.findall(r"<custom_token_(\d+)>", text)]

snac = load_snac("cuda")           # CPU'da da çalışır: "cpu"
wav = decode_to_wav(snac, flat)
sf.write("cikti.wav", wav.numpy(), SNAC_SR)
```

Klonlama isteminde referans sesin SNAC tokenları da isteme eklenir
(`infer.py` içindeki `encode_ref` ile üretilir, `<custom_token_N>` olarak
serileştirilir).

Notlar:

- `repetition_penalty` ses tokenlarında kelime tekrarını bastırır; 1.05–1.2
  aralığında ayarlayın.
- Akışlı (streaming) kullanımda tokenlar 7'lik çerçeveler halinde biriktikçe
  parça parça çözülebilir; SNAC çözücüsü kısa pencerelerde sınır
  artefaktları üretebilir, pencereleri örtüştürün.
