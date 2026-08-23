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
- vllm
---

# turkish-tts-model

Türkçe metinden konuşma sentezi için sıfırdan eğitilmiş ~95M parametreli
konuşma dil modeli: stok `LlamaForCausalLM`, SNAC 24 kHz codec tokenları
üzerinde otoregresif üretim, zero-shot ses klonlama. **vLLM ile doğrudan
servis edilir** (`vllm serve`); ses tokenları `<custom_token_N>` biçiminde
metin olarak üretilir ve istemci tarafında SNAC ile 24 kHz sese çözülür.

Eğitim ve çıkarım kodu:
https://github.com/serdarildercaglar/turkish-tts-model

## Eğitim verisi

[`serdarcaglar/turkish-tts-audiobooks`](https://huggingface.co/datasets/serdarcaglar/turkish-tts-audiobooks)
temiz havuzu: 416.315 klip / 1.292 saat Türkçe sesli kitap okuma konuşması,
16 kHz (eğitim için 24 kHz'e yeniden örneklendi), makine transkriptli
(örneklem denetiminde insan referanslı CER 0,0012). Klonlama örnekleri veri
kümesinin kayıtlar-arası `ref_id` çiftlerinden kurulur.

## Değerlendirme

<!--EVAL-->

## Sınırlar

- Tek alan (sesli kitap anlatımı) ve tek dil; kayıt-dışı konuşma tarzlarında
  başarım düşer. 95M parametre bu model ailesinin bilinen örneklerinden
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
