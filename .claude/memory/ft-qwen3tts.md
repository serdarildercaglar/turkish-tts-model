---
name: ft-qwen3tts
description: "Aktif dal — Qwen3-TTS-0.6B-Base Türkçe FT; 2000 saat export hazır, 3 harici HF kümesi denetlendi, tüm işler BULUTTA devam edecek"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6341b935-9bb8-4f7e-ac17-c33d1613d07d
  modified: 2026-08-27T22:19:53.593Z
---

28 Ağu 2026: sıfırdan eğitim silindi; tek dal **Qwen3-TTS-12Hz-0.6B-Base
ince ayarı** (Apache-2.0, resmî FT hattı `QwenLM/Qwen3-TTS/finetuning`,
vllm-omni day-0 yerli). B planı: Chatterbox Multilingual v3 (MIT, Türkçe
hazır, vllm-omni yok). Plan + komutlar: `docs/ft_qwen3tts.md`.

**Kalınan yer — bundan sonrası BULUTTA (H100/H200):**
1. Parquet'ler + `materialize_clips.py` (ana küme kliplerini döker; review
   FLAC'leri yalnız Hub'da), harici kümeler `ingest_hf_dataset.py` ile
   dökülür (+hijyen, istenirse `--dnsmos-min 3.0`).
2. afkfatih konuşmacısız → `pair_by_embedding.py` (pyannote/embedding) şart.
3. Birleşik `export_qwen3tts_ft.py` (ana 2.000 sa hazır: 638.929 satır,
   heldout dışarıda; hariciler eklenince `--max-hours` büyütülür).
4. Qwen resmî `prepare_data.py` → `sft_12hz.py` (lr ~2e-5, 2–3 epoch,
   heldout erken durdurma; Qwen paketi transformers>=5 ile uyumsuz → ayrı env).
5. Eval üçlüsü (WER/DNSMOS/klon benzerliği) + rakam tuzağına dikkat.

**Harici kümeler (birer shard'la denetlendi, sonra silindi):**
- `Anilosan15/Turkish_TTS_Data`: 30,6k satır, tek konuşmacı "sıla", 48 kHz,
  sesli-kitap tarzı, DNSMOS ort 3,34. **LİSANSSIZ** — ticari risk açık konu.
- `afkfatih/turkish-tts-combined-raw`: 81,5k satır, 16/48 kHz karışık,
  DNSMOS 2,13–3,64, **konuşmacı sütunu yok**; CC-BY-SA-3.0, Khan payı NC riski.
- `Codyfederer/tr-combined`: 221,5k satır, 2.158 etiketli konuşmacı + duygu,
  44,1 kHz YouTube kırpımı, DNSMOS 2,38–3,29; CC-BY-4.0 etiketi şüpheli
  (kazıma). Kapılı küme — HF hesabı `serdarcaglar` erişimi kabul etti.

Riskler: resmî FT tek-konuşmacı odaklı (çok-konuşmacı için satır başına
farklı ref_audio; vspeech/Qwen3-TTS-Train referans); Türkçe modelin 10
dilinde yok → `codec_language_ids` koşullaması ilk teknik kontrol.
Kullanıcı notu: aynı metnin farklı seslendirmeleri BİLEREK tutuluyor
(prozodi çeşitliliği) — metin-bazlı tekilleştirme YAPMA; hijyen yalnız
(konuşmacı, metin, ~süre) mükerrerini düşürür.

İlgili: [[dataset-audit]]
