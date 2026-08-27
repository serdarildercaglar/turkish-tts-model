# Proje hafızası (bulut oturumları için)

Yerel oturumların biriktirdiği hafıza `.claude/memory/` altında; ayrıntı
gerektiğinde ilgili dosyayı oku. Kullanıcı tam cümleli Türkçe yanıt tercih
eder.

## Güncel durum (28 Ağu 2026) — İŞLER BURADAN (BULUTTAN) DEVAM EDECEK

**Tek aktif dal — Qwen3-TTS ince ayarı:** sıfırdan eğitim dalı silindi.
Model: Qwen3-TTS-12Hz-0.6B-Base (Apache-2.0, resmî FT hattı, vllm-omni
day-0 yerli). Ana kümenin 2.000 saatlik `{audio, text, ref_audio}` JSONL'i
yerelde üretildi (638.929 satır, heldout dışarıda) ama gitignore'da —
bulutta aynı tohumla yeniden üretilir. Sıradaki adımlar (ayrıntı ve tam
komutlar `docs/ft_qwen3tts.md` "Bulut sırası"):

1. Ana küme parquet'leri → `scripts/materialize_clips.py` (review FLAC'leri
   yalnız Hub'da).
2. Üç harici küme → `scripts/ingest_hf_dataset.py` (hijyen dahil;
   Anilosan LİSANSSIZ, afkfatih'te Khan-NC payı, tr-combined YouTube
   kırpımı — ticari karar kullanıcının).
3. afkfatih konuşmacısız → `scripts/pair_by_embedding.py` (pyannote).
4. Birleşik `scripts/export_qwen3tts_ft.py` → Qwen resmî `prepare_data.py`
   → `sft_12hz.py` (ayrı env; Qwen paketi transformers>=5 ile uyumsuz).
5. Eval üçlüsü: WER / DNSMOS / klon benzerliği, `heldout.jsonl`.

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
