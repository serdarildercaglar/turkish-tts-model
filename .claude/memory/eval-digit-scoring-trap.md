---
name: eval-digit-scoring-trap
description: "TTS eval'de rakam içeren cümleler WER'i yapay olarak şişiriyor — ASR rakamla yazıyor, naif rakam→kelime normalizasyonu doğru okunuşu asla tutturamıyor"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 294fa856-6770-4d4d-bf19-4f5d0446e990
  modified: 2026-08-27T14:30:58.146Z
---

TTS değerlendirmesinde referans metin sayı/tarih/para içeriyorsa, ASR çıktısını
rakam olarak yazar ve puanlama normalizasyonundaki naif `\d+ → kelime` dönüşümü
doğru sözlü okunuşu **asla** tutturamaz. Ölçülen hata sentezden değil harness'ten gelir.

27 Ağustos 2026, ema-tts ölçümünde görüldü. 9 cümlede whisper-large-v3-turbo ile
ham WER %13,8 çıktı; başarısız 4 cümlenin hepsi sayı ağırlıklıydı ve incelendiğinde
sentez doğruydu:

- TTS "dokuz buçukta" dedi; ASR `09:30'da` yazdı; normalizasyon "dokuz otuz da" üretti.
- TTS "virgül" ve "yüzde" dedi; ASR `,` ve `%` yazdı; noktalama olarak silindi.
- TTS telefonu rakam rakam okudu; ASR `0850 123 45 67` yazdı; normalizasyon
  tek büyük sayı sanıp "sekiz milyar beş yüz bir milyon..." üretti.

Sayı içermeyen 6 cümlede gerçek WER %1,4 (tek gerçek hata "sürecine"→"sürecin").
Yani gerçek performans ham sayının ~10 katı daha iyi.

**Why:** kendi src/evaluate.py'miz aynı tuzağa düşerse pilot ve tam eğitim
checkpoint'lerini yanlış sıralarız — sayı okuması iyileşen bir model daha kötü
görünebilir. ema-tts'in kendi score_norm'u da bu kusuru taşıyor, dolayısıyla
raporlanan %3,0 WER'i de bu açıdan temkinli okumak gerekir.

**How to apply:** eval setinde sayılı cümleleri ayrı raporla, ya da referansı
frontend çıktısıyla (sözel okunuş) karşılaştırıp ASR hipotezini de aynı frontend'den
geçir. Tek bir toplu WER sayısına bakıp karar verme.
Ayrıca aynı ölçümde 8 kHz telefon bandına indirmenin anlaşılırlığa maliyeti
yok çıktı (%12,8 vs %13,8) — IVR hedefi için iyi haber.
İlgili: [[ema-tts-baseline]]
