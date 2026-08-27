# turkish-tts-model

Türkçe metinden konuşma sentezi için **sıfırdan eğitilen**, **63M parametreli**,
ses klonlamalı ve prozodi kontrollü, gerçek zamanlı IVR hedefli bir konuşma dil
modeli. Tek RTX 3090'da eğitilebilir. Eğitim verisi:
[`serdarcaglar/turkish-tts-audiobooks`](https://huggingface.co/datasets/serdarcaglar/turkish-tts-audiobooks)
(hattı da açık: [turkish-tts-audiobooks](https://github.com/serdarildercaglar/turkish-tts-audiobooks)).

## Mimari

Stok `LlamaForCausalLM` — hiçbir özel modelleme kodu yok; olağan bir dil modeli,
sözlüğü ses tokenlarıyla genişletilmiş.

- **Codec:** [SNAC 24 kHz](https://github.com/hubertsiuzdak/snac) (MIT). Üç RVQ
  düzeyi, kaba çerçeve başına 7 token olarak düzleştirilir (Orpheus şeması),
  ~82 token/saniye. Korpus 16 kHz olduğundan giriş 24 kHz'e yeniden örneklenir.
- **Sözlük (16.448):** 4.096 Türkçe BPE + **12.288 ses tokenı** + özel ve
  prozodi kontrol tokenları.

  Ses sözlüğü **düzey** başına ofsetlenir, yuva başına değil. Yedi yuvanın her
  birine kendi 4.096'lık aralığını vermek (28.672 giriş) L1'i iki, L2'yi dört
  kez kopyalar — oysa bunlar aynı kod kitabından gelir. Çerçeve içindeki yuva
  bilgisi zaten konumdan (RoPE) gelir. Düzey ofseti 16.384 gömme satırı
  kazandırır: gömme tablosu modelin %25'inden **%13'üne** iner, kazanılan
  parametre katmanlara gider.

- **Model:** hidden 512 / 20 katman / 8 başlık (GQA, 2 KV başı) / FFN 1344,
  RoPE, bağlı embedding → **62,8M parametre** (`configs/model_63m.json`;
  51M ve 58M varyantları da var).

- **Batching token bütçesiyle yapılır**, sabit örnek sayısıyla değil. Örnek
  uzunlukları 300–2600 token arasında değişiyor; sabit batch boyutu ya kısa
  kovalarda GPU'yu boş bırakır ya uzun kovada belleği taşırır. `tokens_per_batch`
  dolgulu maliyeti (en uzun örnek × batch boyutu) sabitler.
  3090'da ölçülen: **20.480 token/batch → 19,1 GB, ~1,05 s/adım**.

- **İstem biçimi:**
  - Düz: `<|bos|><|plain_tts|><|rate_k|><|loud_k|>[parça]<|text_start|>metin<|text_end|><|audio_start|>`
  - Klon: `<|bos|><|clone_tts|><|rate_k|><|loud_k|>[parça]<|text_start|>ref_metin hedef_metin<|text_end|><|audio_start|>REF_SES`

  Tek tanım `src/prompt.py`'dedir; eğitim, çıkarım ve değerlendirme aynı yerden
  okur ki biçim sessizce ayrışmasın.

- **Prozodi kontrolü:** klip başına manifestten bedava gelen iki öznitelik
  kovalanır — konuşma hızı (harf/sn, 5 kova) ve ses seviyesi (LUFS, 3 kova).
  Kova sınırları korpustan hesaplanır ve `prosody.json` olarak checkpoint'le
  taşınır. Eğitimde kontrol tokenı `--control-dropout` olasılığıyla
  `<|rate_any|>`/`<|loud_any|>`'e düşer, böylece çıkarımda kontrol verilmezse
  model öğrenilmiş ortalamaya oturur. Kalite (DNSMOS) kovası bilerek yok:
  korpusta aralık 2,94–3,58 ile çok dar, öğrenilecek sinyal taşımıyor.

## Veri hattı

Kaynak veriye asla yazılmaz; her adım türetilmiş çıktı üretir ve ne düştüğünü
gerekçesiyle raporlar.

```bash
pip install -r requirements.txt
cp configs/train.example.yaml configs/train.yaml

# 0) (istege bagli) veri kumesini denetle — salt okunur
python scripts/audit_dataset.py --manifest .../hf_train.jsonl \
    --val .../hf_validation.jsonl --out artifacts/audit

# 1) veri kumesini indir (train + review; ses parquet icinde gomulu)
pip install hf_transfer          # kapili depoda tek akis ~140 KB/s'te takiliyor
python scripts/download_dataset.py --out /veri/turkish-tts --splits train review

# 2) hijyen + train/review birlestirme -> turetilmis manifest
python scripts/prepare_manifest.py --data /veri/turkish-tts --splits train review \
    --out artifacts/manifest/train_all.jsonl \
    --report artifacts/manifest/train_all_report.json

# 3) metin tokenizer'i (4k BPE, hijyenden gecmis metin uzerinde)
python scripts/train_tokenizer.py --manifest artifacts/manifest/train_all.jsonl \
    --out artifacts/tokenizer

# 4) ses -> SNAC kodlari (parquet'ten akisla; --delete-parquet ile disk tasarrufu)
python scripts/ingest_audio.py --data /veri/turkish-tts --splits train review \
    --manifest artifacts/manifest/train_all.jsonl --out artifacts/codes

# 5) paketlenmis egitim kumesi
python scripts/build_dataset.py --manifest artifacts/manifest/train_all.jsonl \
    --codes artifacts/codes --tokenizer artifacts/tokenizer --out artifacts/packed

# 6) egitim
python -m src.train --config configs/train.yaml        # --resume ile devam
```

### Hijyen politikası

`src/data_hygiene.py` tek karar noktasıdır; hiçbir düzeltme sessizce uygulanmaz.

**Kayıpsız metin düzeltmeleri.** Whisper Türkçe'de çift tırnağı `''` (iki ASCII
kesme) yazıyor ve bu, ek sınırı işaretleyen kesme ile çakışıyor (`Baba'ya`).
`''` → `"`, `…` → `...`, NFC, boşluk sadeleştirme. Ek kesmesi korunur.

**Elenenler.** ASR döngüsü, harf/sn aykırılığı, aynı kaydın farklı `source_id`
altında mükerrer alımı, `asr_cift_gecis_uyusmazligi` ve
`farkli_konusmaci_suphesi` gerekçeleri, kalite eşikleri.

Aynı metnin **farklı konuşmacıyla** tekrarı elenmez — o prozodi çeşitliliğidir.
Mükerrer kayıt ayrımı süre eşitliğiyle yapılır (grup içi yayılım p50 0,000 s).

`sentetik_ses_suphesi` bilerek eleme listesinde değil: review bölümünde tek bir
okuyucunun ses değiştirip tiyatral okuması bu bayrağı yanlış tetikliyor, veri
prozodi açısından değerli.

**Cümle parçaları.** Korpusun ~dörtte biri cümle ortasından başlıyor ya da
bitiyor. Atmak %25 veri kaybı demek; onun yerine `<|frag_start|>` /
`<|frag_end|>` ile işaretlenir. Çıkarımda tokenı vermeyerek tam cümle isteriz.

**Rakamlar varsayılan olarak ham bırakılır.** Sözelleştirme ayrı bir karardır,
`--verbalize-digits` ile açılır (`src/text_frontend.py`).

### Pilot alt kümesi

```bash
python scripts/make_subset.py --manifest artifacts/manifest/train_all.jsonl \
    --hours 100 --out artifacts/manifest/pilot_100h.jsonl
```

Sıkı kalite kapısı (DNSMOS/konuşma oranı üst dilimleri, sıfır kırpma, sıfır ASR
CER, parça yok) **ve** 5×3 prozodi kovasının eşit doldurulması. Kova içinde
konuşmacılar sırayla gezilir, tek okuyucu kovayı kapatmasın.

## Çıkarım

```bash
python infer.py --model ckpt/final --tokenizer artifacts/tokenizer \
    --text "Merhaba, bugün hava çok güzel." --out merhaba.wav

# prozodi kontrolu: hiz 0 (yavas) - 4 (hizli), seviye 0-2
python infer.py ... --rate 1 --loud 1

# ses klonlama (3-10 sn referans + transkripti)
python infer.py ... --ref-audio ref.wav --ref-text "Referans kaydın transkripti."
```

Servis: [docs/serving_vllm_omni.md](docs/serving_vllm_omni.md). Checkpoint stok
Llama'dır; vllm-omni tarafında küçük bir out-of-tree eklenti (AR aşaması + SNAC
çözücü aşaması) modeli hatta bağlar.

## Değerlendirme notları

- **Telefon bandı.** IVR çıkışı G.711'de 8 kHz'e düşer; değerlendirme sesi
  ASR'den önce 8 kHz'e indirmelidir, yoksa hattan teslim edilmeyecek bir
  kaliteyi ölçeriz.
- **Rakam tuzağı.** Referans metin sayı içeriyorsa ASR çıktıyı rakamla yazar ve
  naif `\d+ → kelime` normalizasyonu doğru okunuşu tutturamaz; ölçülen hata
  sentezden değil harness'ten gelir. Sayılı cümleleri ayrı raporlayın.
- **Bölünme.** Kaynak `validation`'ın %93,2'si eğitimde görülmüş konuşmacıya
  ait; genelleme iddiası için konuşmacı-ayrık ayrı bir küme gerekir.
- Örnekleyici stokastiktir; tek koşu tek çekiliştir. 3 tohum ortalaması ± ile
  raporlayın.

## Lisans ve yükümlülükler

Kod ve ağırlıklar **Apache-2.0**. SNAC codec'i MIT (Hubert Siuzdak).
`src/text_frontend.py`, `canberkkkkkk/ema-tts` (Apache-2.0) içindeki `text.py`'den
uyarlanmıştır.

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
