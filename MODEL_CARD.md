---
language:
- tr
license: apache-2.0
pipeline_tag: text-to-speech
datasets:
- serdarcaglar/turkish-tts-audiobooks
tags:
- tts
- turkish
- speech-lm
- snac
- vllm-omni
---

# turkish-tts-model

Türkçe metinden konuşma sentezi için sıfırdan eğitilmiş **62,8M parametreli**
konuşma dil modeli: stok `LlamaForCausalLM`, SNAC 24 kHz codec tokenları
üzerinde otoregresif üretim, zero-shot ses klonlama ve prozodi kontrolü.
**vllm-omni ile `/v1/audio/speech` olarak servis edilir**
(`vllm serve <model> --omni`, akışlı çıkış, `ref_audio` klonlama); ses
tokenları `<custom_token_N>` biçiminde gerçek tokenlardır, SNAC çözücü aşaması
bunları 24 kHz sese çevirir.

Sözlük 16.448 giriş: 4.096 Türkçe BPE + 12.288 ses tokenı + özel ve kontrol
tokenları. Ses sözlüğü SNAC'ın **üç kod kitabı** başına ofsetlenir, yedi yuva
başına değil; yuva bilgisi konumdan (RoPE) geldiği için yuva başına ofset
L1'i iki, L2'yi dört kez kopyalamaktan ibarettir. Bu düzeltme gömme tablosunu
modelin %25'inden %13'üne indirir.

**Prozodi kontrolü:** istemin başındaki `<|rate_k|>` (5 kova, konuşma hızı) ve
`<|loud_k|>` (3 kova, ses seviyesi) tokenları. Kova sınırları `prosody.json`
içinde checkpoint ile birlikte taşınır. Kontrol verilmezse `<|rate_any|>` /
`<|loud_any|>` ile öğrenilmiş ortalamaya oturur.

Eğitim ve çıkarım kodu:
https://github.com/serdarildercaglar/turkish-tts-model

## Eğitim verisi

[`serdarcaglar/turkish-tts-audiobooks`](https://huggingface.co/datasets/serdarcaglar/turkish-tts-audiobooks)
`train` ve `review` bölümleri birlikte: hijyenden sonra 818.775 klip /
2.623,8 saat Türkçe sesli kitap okuma konuşması, 16 kHz (eğitim için 24 kHz'e
yeniden örneklendi), makine transkriptli (örneklem denetiminde insan referanslı
CER 0,0012). Klonlama örnekleri kayıtlar-arası `ref_id` çiftlerinden kurulur.

`review` bölümü `sentetik_ses_suphesi` bayrağı taşır; bu bayrak tek bir
okuyucunun ses değiştirip tiyatral okumasından doğan yanlış pozitiftir ve
bölüm prozodi çeşitliliği açısından değerlidir (kalite metrikleri `train` ile
birebir aynı: DNSMOS 3,3586 vs 3,3604). Gerçek hatalar — ASR döngüsü, çift
geçiş uyuşmazlığı, farklı konuşmacı şüphesi, mükerrer kayıt — elenir.

## Değerlendirme

<!--EVAL-->

## Sınırlar

- Tek alan (sesli kitap anlatımı) ve tek dil; kayıt-dışı konuşma tarzlarında
  başarım düşer. 62,8M parametre bu model ailesinin bilinen örneklerinden
  (0,5B–3B) çok küçüktür; uzun cümlelerde kelime atlama/tekrar görülebilir.
- Klonlama geniş tınıyı yakalar; konuşmacı benzerliği büyük modellerin
  gerisindedir.
- Çıkış bant genişliği ~8 kHz'tir (kaynak korpus 16 kHz).

## Etik ve kullanım şartları

Bu model, eğitim verisinin erişim şartlarına uygun olarak **açık ağırlıklı ve
ticari kullanıma izin veren** lisansla yayımlanmıştır (Apache-2.0) ve veri
kümesine atıf verir. Kimsenin sesini rızası olmadan klonlamayın. Eğitim
verisinin kaynak kayıtlarının telif durumu doğrulanmamıştır; ayrıntı için
veri kümesi kartına bakın.

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

Codec: [SNAC](https://github.com/hubertsiuzdak/snac) (MIT, Hubert Siuzdak).
