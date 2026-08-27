# Proje hafızası (bulut oturumları için)

Yerel oturumların biriktirdiği hafıza `.claude/memory/` altında; ayrıntı
gerektiğinde ilgili dosyayı oku. Kullanıcı tam cümleli Türkçe yanıt tercih
eder.

## Güncel durum (28 Ağu 2026)

**Tek aktif dal — Qwen3-TTS ince ayarı:** sıfırdan eğitim dalı ve o dala ait
mimari dosyalar/kararlar bilinçli olarak silindi. Model:
Qwen3-TTS-12Hz-0.6B-Base (Apache-2.0, resmî FT hattı, vllm-omni day-0 yerli).
2.000 saatlik `{audio, text, ref_audio}` JSONL'i dışa aktarıldı
(`artifacts/qwen3tts_ft/train.jsonl`, 638.929 satır, heldout dışarıda).
Review kliplerinin FLAC'i yerelde YOK — bulutta `scripts/download_dataset.py`
ile parquet'ler indirilir, `scripts/materialize_clips.py` klipleri döker ve
yolları yeniden yazar. Plan ve riskler: `docs/ft_qwen3tts.md`.

## Dizin

- [ft-qwen3tts](memory/ft-qwen3tts.md) — ince ayar kararı, veri dışa aktarımı,
  bilinen riskler (tek-konuşmacı FT sınırı, dil-kimliği koşullaması,
  transformers<5 şartı), metin-mükerrerliği BİLEREK korunuyor
- [dataset-audit](memory/dataset-audit.md) — review seti değerli, %93
  konuşmacı sızıntısı, hijyen sonrası 2.623,8 saat
- [eval-digit-scoring-trap](memory/eval-digit-scoring-trap.md) — sayılı
  cümlelerde WER yapay şişiyor; naif normalizasyonla düzelmez
- [ema-tts-baseline](memory/ema-tts-baseline.md) — NAR flow-matching RTF
  paritesi; prozodide üstünlük bizde
- [user-and-environment](memory/user-and-environment.md) — kullanıcı
  tercihleri, yerel 3090, veri yolları

Depodaki `configs/model_*.json`, `src/train.py`, `docs/serving_vllm_omni.md`
gibi dosyalar eski sıfırdan-eğitim dalından kalmadır; FT dalı bunları
KULLANMAZ (eğitim = QwenLM/Qwen3-TTS finetuning hattı, servis = vllm-omni
yerli Qwen3-TTS). Kullanıcı isterse onlar da temizlenecek.
