---
name: ft-qwen3tts
description: Karar — sıfırdan eğitim rafta; Qwen3-TTS-12Hz-0.6B-Base 2.000 saat Türkçe ile FT edilecek; veri dışa aktarımı yapıldı
metadata: 
  node_type: memory
  type: project
  originSessionId: 6341b935-9bb8-4f7e-ac17-c33d1613d07d
  modified: 2026-08-27T21:40:01.826Z
---

28 Ağu 2026 kararı: sıfırdan eğitim (63M/288M) rafa kalktı; **ince ayar dalı**
seçildi. Model: **Qwen3-TTS-12Hz-0.6B-Base** (Apache-2.0, resmî FT hattı
`QwenLM/Qwen3-TTS/finetuning`, vllm-omni day-0 yerli — eklenti gerekmez).
B planı: Chatterbox Multilingual v3 (MIT, Türkçe hazır, ama vllm-omni yok).
Plan: `docs/ft_qwen3tts.md`.

- Veri: `scripts/export_qwen3tts_ft.py` → `artifacts/qwen3tts_ft/train.jsonl`
  = **638.929 satır, tam 2.000 saat** (train_only + review_paired, heldout
  dışarıda). Referans: manifest ref'i ≤15 sn ise o; yoksa aynı konuşmacı +
  farklı kayıt yedeği; `review-*` vekil konuşmacıda yedek kapalı.
- **Review kliplerinin FLAC'i yerelde YOK** (kaynak m4a'lar + Hub parquet var);
  bulutta `scripts/materialize_clips.py` parquet'ten `<id>.flac` döker ve
  yolları yeniden yazar (id/ref_id alanları JSONL'de taşınıyor).
- Resmî FT bugün tek-konuşmacı odaklı; çok-konuşmacılı klon için satır başına
  farklı ref_audio + vspeech/Qwen3-TTS-Train referans. Türkçe, modelin 10
  dilinde yok → dil-kimliği koşullaması (`codec_language_ids`) ilk teknik
  kontrol. Qwen paketi transformers>=5 ile uyumsuz → ayrı env.
- Kullanıcı notu: veri kümesinde birebir aynı metinler farklı seslendirmelerle
  var ve BİLEREK tutuluyor (prozodi çeşitliliği). Hijyen zaten yalnız
  (konuşmacı, metin, ~süre) üçlüsü aynıysa düşürüyor — metin-bazlı
  tekilleştirme YAPMA.

İlgili: [[dataset-audit]]
