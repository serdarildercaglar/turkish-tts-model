# turkish-tts-model

Türkçe metinden konuşma sentezi için **sıfırdan eğitilen**, ~95M parametreli,
**vLLM ile doğrudan servis edilebilen**, ses klonlamalı bir konuşma dil
modeli. Eğitim verisi:
[`serdarcaglar/turkish-tts-audiobooks`](https://huggingface.co/datasets/serdarcaglar/turkish-tts-audiobooks)
(1.292 saat temiz Türkçe okuma konuşması; hattı da açık:
[turkish-tts-audiobooks](https://github.com/serdarildercaglar/turkish-tts-audiobooks)).

## Mimari

Stok `LlamaForCausalLM` — hiçbir özel modelleme kodu yok; olağan bir dil
modeli, sözlüğü ses tokenlarıyla genişletilmiş:

- **Codec:** [SNAC 24 kHz](https://github.com/hubertsiuzdak/snac) (MIT).
  Üç RVQ düzeyi, kaba çerçeve başına 7 token olarak düzleştirilir
  (Orpheus şeması), ~83 token/saniye. Korpus 16 kHz olduğundan giriş 24 kHz'e
  yeniden örneklenir; çıkış 24 kHz'tir.
- **Sözlük (36.928):** 8.192 Türkçe BPE (korpustan eğitilir) + 28.672 ses
  tokenı (`<custom_token_N>` — vLLM'in metin olarak üretebilmesi için gerçek
  tokenlar) + özel tokenlar.
- **Model:** hidden 640 / 14 katman / 10 başlık / FFN 1792, RoPE, bağlı
  embedding → ~95M parametre (`configs/model_95m.json`; 74M ve 145M
  varyantları da var).
- **İstem biçimi:**
  - Düz TTS: `<|bos|><|plain_tts|><|text_start|>metin<|text_end|><|audio_start|>` → model ses tokenları + `<|audio_end|><|eos|>` üretir.
  - Klonlama: `<|bos|><|clone_tts|><|text_start|>ref_metin hedef_metin<|text_end|><|audio_start|>REF_SES` → model referans sesin devamı olarak hedef sesi üretir (Llasa tarzı devam).

## Eğitim (tek RTX 3090)

```bash
pip install -r requirements.txt
cp configs/train.example.yaml configs/train.yaml   # yolları doldur

# 1) Metin tokenizer'ı (8k BPE)
python scripts/train_tokenizer.py --manifest .../hf_train.jsonl --out artifacts/tokenizer

# 2) Ses tokenizasyonu: FLAC klipler -> SNAC kodları (~0,8 GB, 3-5 sa GPU, sürdürülebilir)
python scripts/tokenize_audio.py --manifest .../hf_train.jsonl --out artifacts/codes/train
python scripts/tokenize_audio.py --manifest .../hf_validation.jsonl --out artifacts/codes/validation

# 3) Paketlenmiş eğitim kümesi (~550M token/epoch; düz + ref_id klon çiftleri x2)
python scripts/build_dataset.py --manifest .../hf_train.jsonl \
    --codes artifacts/codes/train --tokenizer artifacts/tokenizer --out artifacts/packed

# 4) Eğitim (bf16, uzunluk-kovali batching, atomik checkpoint; --resume ile devam)
python -m src.train --config configs/train.yaml

# 5) Checkpoint değerlendirme: CER (Whisper) + konuşmacı kosinüsü + DNSMOS
python scripts/make_eval_set.py --manifest .../hf_validation.jsonl --out artifacts/eval_set.jsonl
python -m src.evaluate --config configs/train.yaml --checkpoint ckpt/checkpoint-2000
```

Veri kümesi kapılıdır (otomatik onaylı form); klipler ve manifestler yerelde
hazırsa yolları doğrudan gösterin, değilse Hub'dan indirip `audio` alanlarını
diske açın.

## Çıkarım

```bash
# Düz TTS
python infer.py --model ckpt/final --tokenizer artifacts/tokenizer \
    --text "Merhaba, bugün hava çok güzel." --out merhaba.wav

# Ses klonlama (3-10 sn referans + transkripti)
python infer.py --model ckpt/final --tokenizer artifacts/tokenizer \
    --text "Bu cümleyi referans sesle söyle." \
    --ref-audio ref.wav --ref-text "Referans kaydın transkripti." --out klon.wav
```

vLLM ile servis: [docs/serving_vllm.md](docs/serving_vllm.md) — checkpoint
stok Llama olduğu için `vllm serve <model>` doğrudan çalışır; istemci
`<custom_token_N>` çıktısını SNAC ile sese çözer.

## Lisans ve yükümlülükler

Kod ve ağırlıklar **Apache-2.0**. Eğitim verisinin erişim şartları gereği bu
modelin ağırlıkları ticari kullanıma izin veren bir lisansla açık yayımlanır
ve veri kümesine atıf verilir; bu depo her iki yükümlülüğü de yerine getirir.
SNAC codec'i MIT lisanslıdır (Hubert Siuzdak).

## Atıf

```bibtex
@misc{caglar2026turkishttsaudiobooks,
  title        = {Turkish TTS Audiobooks: a 2,724-hour Turkish read-speech
                  corpus for text-to-speech},
  author       = {Serdar I. {\c{C}}a{\u{g}}lar},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/serdarcaglar/turkish-tts-audiobooks}}
}
```
