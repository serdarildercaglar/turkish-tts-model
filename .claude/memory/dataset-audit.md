---
name: dataset-audit
description: "Korpus denetiminin sonuçları — review setinin değeri, sızıntı, mükerrer kayıt, metin hataları ve hangi düzeltmelerin uygulandığı"
metadata: 
  node_type: memory
  type: project
  originSessionId: 09e30beb-ae55-4b9d-9a05-881026afc21f
  modified: 2026-08-27T15:27:43.826Z
---

27 Ağustos 2026'da `scripts/audit_dataset.py` ile salt-okunur denetim yapıldı
(manifestler değiştirilmedi). Sonuçlar `artifacts/audit/` altında.

**Ses kalite kapısı zaten temiz.** Kırpma, konuşma oranı, müzik, DNSMOS,
konuşmacı kosinüsü — train'de tek aykırı satır yok; voxcpm hattı elemiş.
Sorun tamamen metinde ve bölünmede.

**review seti değerli, `sentetik_ses_suphesi` yanlış pozitif.** 432.306 klip /
1.412,7 saat, tamamı `decision: REVIEW`. Bayrakların %83'ü sentetik ses şüphesi,
ama kullanıcıya göre bu tek bir okuyucunun ses değiştirip tiyatral okuması —
yani tam da istediğimiz prozodi çeşitliliği. Kalite metrikleri train'le birebir
aynı (DNSMOS 3,3586 vs 3,3604). **Review sesi yerelde YOK**, HF'te
`data/review-*.parquet` (249 shard, 83,7 GB) içinde; `speaker_id` tüm review
satırlarında `None`, vekil olarak `channel` kullanılıyor.

**Gerçek hatalar:** ASR döngüsü (155 train / 265 review), harf-sn aykırılığı,
`asr_cift_gecis_uyusmazligi` (CER max 49 — metin yanlış), `farkli_konusmaci_suphesi`
(kosinüs p50 0,39 — klonlamayı bozar). Mükerrer kayıt: aynı kaydın farklı
`source_id` altında ikinci alımı; grup içi süre yayılımı p50 **0,000 s** ile
doğrulandı. Aynı metnin FARKLI konuşmacıyla tekrarı korunur — o prozodi
çeşitliliğidir, elenmez.

**Metin:** Whisper Türkçe'de çift tırnağı `''` yazıyor (%9,56) ve bu, ek sınırı
işaretleyen kesme ile çakışıyor (`Baba'ya`, satırların %18,4'ünde). `…` %2,43.
Cümle parçası: %9,42 küçük harfle başlıyor, %16,33 noktalamasız bitiyor.
Rakam %5,08 — sözelleştirme kararı kullanıcıya bırakıldı, varsayılan KAPALI
(bkz. [[eval-digit-scoring-trap]]).

**Değerlendirme bölünmesi kirli:** source_id kesişimi 0 ama validation
kliplerinin **%93,2'si eğitimde görülmüş konuşmacıya** ait, 270 klip birebir
aynı kayıt. Kullanıcı ayrı bir validation seti hazırlanacağını söyledi.

**Hijyen sonrası:** train 416.315 → 412.943, review 432.306 → 405.832.
**Birleşik 818.775 klip / 2.623,8 saat.**

**Why:** "garbage in garbage out olmasın" — mimariyi kurmadan önce neyin
düzeltileceğinin ölçülmesi istendi.

**How to apply:** `src/data_hygiene.py` politikayı taşır, `scripts/prepare_manifest.py`
türetilmiş manifesti üretir. Kaynak manifestlere asla yazılmaz.
İlgili: [[ft-qwen3tts]]
