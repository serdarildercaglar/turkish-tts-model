# İnce ayar planı: Qwen3-TTS-12Hz-0.6B-Base × 2.000 saat Türkçe (28 Ağu 2026)

Karar: sıfırdan eğitim rafa; ön eğitimli, düşük parametreli, ticari lisanslı
bir modele Türkçe kazandırılacak. Aday karşılaştırması ve seçim gerekçesi
aşağıda; hepsi kaynaklardan doğrulandı (bağlantılar dosya sonunda).

## Aday karşılaştırması

| | **Qwen3-TTS-12Hz-0.6B-Base (SEÇİLDİ)** | Chatterbox Multilingual v3 (0.5B) | CosyVoice3-0.5B | Fish/OpenAudio S1-mini |
|---|---|---|---|---|
| Lisans | Apache-2.0 (ticari OK) | MIT (ticari OK) | kontrol gerekir | CC-BY-NC → **elendi** |
| İnce ayar dokümanı | **Resmî** (`QwenLM/Qwen3-TTS/finetuning`: prepare_data + sft_12hz) + topluluk (vspeech/Qwen3-TTS-Train, çok-konuşmacılı) | Yalnız topluluk | sınırlı | — |
| Türkçe | Yok (10 dil) → 2.000 saatle **öğretilecek** | **Var** (23 dil) | yok | — |
| Klonlama | ref_audio koşullaması, eğitim biçiminin parçası | zero-shot, güçlü | var | — |
| Prozodi | 12 Hz talker + code predictor; sesli-kitap FT'si domain prozodisini getirir | emotion exaggeration kontrolü | iyi | — |
| **vllm-omni** | **Yerli, day-0** (`Qwen3TTSForConditionalGeneration`) — eklenti bile gerekmez | **Yok** → IVR servis katmanı sıfırdan | yerli var | — |
| Büyüklerle yarış | 1.7B kardeşi SOTA sınıfı; 0.6B aynı codec/desen | ElevenLabs'a karşı tercih testleri | — | — |

Seçimi belirleyen eksen servis: IVR hedefi vllm-omni'ye sabitlendi
(serving_vllm_omni.md) ve Qwen3-TTS orada **yerli** çalışıyor — SNAC eklentisi,
özel pipeline, hiçbiri gerekmiyor. Chatterbox'ın Türkçe'yi hazır bilmesi
cazip ama vllm-omni desteği olmadığından gerçek-zamanlı eşzamanlı servis
katmanı bize kalıyordu; ayrıca mimarisi (T3 + S3Gen flow-matching) iki ayrı
ağır bileşen. Dürüst not: Chatterbox, "hazır Türkçe + MIT" ile meşru bir
B planıdır; A dalı klon/prozodide hedefi tutturamazsa ilk denenecek odur.

## Bilinen riskler (önceden doğrulanmış)

1. **Resmî FT hattı bugün tek-konuşmacı odaklı** ("currently supports
   single-speaker fine-tuning"). Çok-konuşmacılı klon FT'si için satır başına
   farklı `ref_audio` verilir (biçim buna izin veriyor); topluluk deposu
   (vspeech/Qwen3-TTS-Train) çok-konuşmacılıyı açıkça destekliyor. İlk
   deneyde ikisi de koşulup heldout klon benzerliğiyle karşılaştırılacak.
2. **Türkçe ön eğitimde yok.** Metin tarafı sorun değil (Qwen 151k sözlüğü
   Türkçe'yi verimli kodlar); risk, dil-kimliği koşullaması: config'te sabit
   `codec_language_ids` listesi var (zh/en/es/de/ja/ko/fr/it/pt/ru). FT'de
   Türkçe için ya en yakın davranışlı kimlik sabitlenir ya da yeni kimlik
   tokenı eklenir — `prepare_data.py`'nin dil alanını nasıl işlediği
   uygulamanın ilk kontrol maddesi.
3. **`Qwen3TTSForConditionalGeneration` stok transformers'ta yok** (4.57 ve
   5.14'te yerelde doğrulandı); eğitim QwenLM deposunun kendi paketiyle yapılır
   ve `transformers>=5` ile uyumsuz — FT ortamı ayrı conda env ister.
4. **SNAC shard'ları bu dalda kullanılmaz.** Ses Qwen3-TTS-Tokenizer-12Hz ile
   yeniden kodlanır (`prepare_data.py` bunu kendisi yapar). 2.000 saat için
   H100'de kabaca birkaç saat–yarım gün sınıfı bir iş; SNAC hattı v1/sıfırdan
   dalı için dokunulmadan durur.

## Hat (yerelde doğrulanmış durumla)

1. **Veri dışa aktarımı — YAPILDI** — `scripts/export_qwen3tts_ft.py`:
   train_only + review_paired → `{audio, text, ref_audio, id, ref_id}` JSONL.
   Referans: manifest `ref_id`'si (train: konuşmacı-içi, review: gömme
   en-yakın-komşu, sim ≥ 0,7) uygunsa o; yoksa aynı konuşmacı + farklı
   kayıttan ≤ 15 sn yedek. `review-*` vekil konuşmacılarda yedek KAPALI
   (kanal ≠ konuşmacı; klon eğitimi kirlenmesin). Sonuç
   (`artifacts/qwen3tts_ft/train.jsonl`): **638.929 satır = tam 2.000,0 saat**
   (aday havuz 648k satır; 170k review satırı kısa/eşleşmiş referans
   olmadığı için düştü). Heldout dışarıda.
2. **Klip dökümü (bulutta)** — review kliplerinin FLAC'i YERELDE YOK
   (400/400 örneklem kayıp; yalnız kaynak m4a'lar + Hub parquet'leri var).
   Bulut makinesinde: `scripts/download_dataset.py` ile
   `serdarcaglar/turkish-tts-audiobooks` (train+review parquet) →
   `scripts/materialize_clips.py` id-eşlemeli klipleri `<id>.flac` olarak
   döker ve JSONL yollarını yeniden yazar (`train_cloud.jsonl`, yalnız
   `{audio, text, ref_audio}` kalır).
3. **Ön işleme** — resmî `prepare_data.py` (Qwen tokenizer'la kod çıkarımı),
   H100'de.
3. **SFT** — `sft_12hz.py`; başlangıç: lr 2e-5 civarı (resmî örnek), 2–3
   epoch, erken durdurma heldout kaybıyla. Dil genişletme tam-FT ister
   (LoRA yeni dil fonolojisi için dar kalır); tek-konuşmacı kararlılık
   tarifi (sabit ref) IVR marka sesi için AYRI kısa bir 2. aşama olarak
   uygulanabilir: önce çok-konuşmacılı Türkçe FT, sonra marka sesiyle
   kısa sabit-ref FT (`--fixed-ref`).
4. **Değerlendirme** — mevcut üçlü (WER/whisper, DNSMOS, klon benzerliği)
   `heldout.jsonl` üzerinde; rakam-normalizasyon tuzağına dikkat
   (eval-digit-scoring-trap).
5. **Servis** — vllm-omni yerli Qwen3-TTS hattı; `/v1/audio/speech` +
   akış. SNAC/özel eklenti bu dalda YOK; `to_telephony.py` 24 kHz → 8 kHz
   tarafı aynen geçerli.

## Ek harici veri kümeleri (28 Ağu 2026 — birer shard'la yerinde denetlendi)

Kullanıcı üç HF kümesinin eklenmesini istedi. Denetim: her kümeden tek shard
indirilip şema + DNSMOS (raw-dalga, bizim onnx) örneklendi, sonra silindi.
Tam döküm YALNIZ bulutta yapılır (`ingest_hf_dataset.py`).

| Küme | Boyut | Denetim bulgusu | Lisans / risk |
|---|---|---|---|
| `Anilosan15/Turkish_TTS_Data` | 30.606 satır, tek konuşmacı ("sıla"), 48 kHz | Sesli-kitap tarzı, temiz; DNSMOS ort **3,34** (3,11–3,51) | **LİSANS YOK** — ticari kullanım riski kullanıcı kararı |
| `afkfatih/turkish-tts-combined-raw` | 81.513 satır, 7 kaynak, 16/48 kHz karışık | DNSMOS 2,13–3,64 dalgalı; **konuşmacı sütunu YOK** → `pair_by_embedding.py` şart | CC-BY-SA-3.0; içindeki Khan Academy payı muhtemelen NC — ticari için o alt küme dışlanabilir olmalı |
| `Codyfederer/tr-combined` | 221.531 satır, **2.158 etiketli konuşmacı**, duygu etiketi, 44,1 kHz | YouTube kırpımı (Vyvo builder); DNSMOS 2,38–3,29; `original_filename` çoğu satırda boş | CC-BY-4.0 deniyor ama YouTube kazıması — etiketin hukuki değeri şüpheli |

Mükerrer metin notu (kullanıcı): aynı metnin farklı seslendirmeleri BİLEREK
tutulur (prozodi çeşitliliği); hijyen yalnız (konuşmacı, metin, ~süre) üçlüsü
aynı olan gerçek kayıt mükerrerlerini düşürür. Kümeler arası aynı kayıt
gelirse (afkfatih ↔ tr-combined kaynak çakışması olası) aynı üçlü yakalar.

Bulut sırası (harici kümeler):

    # 1) dök + hijyen (+ istenirse --dnsmos-min 3.0 kalite süzgeci)
    python scripts/ingest_hf_dataset.py --repo Anilosan15/Turkish_TTS_Data \
        --tag anil --speaker-const sila --clips /veri/ext/anil \
        --out artifacts/manifest/ext_anil.jsonl
    python scripts/ingest_hf_dataset.py --repo afkfatih/turkish-tts-combined-raw \
        --tag afk --clips /veri/ext/afk --out artifacts/manifest/ext_afk.jsonl
    python scripts/ingest_hf_dataset.py --repo Codyfederer/tr-combined \
        --tag cody --speaker-col speaker_id --src-col original_filename \
        --clips /veri/ext/cody --out artifacts/manifest/ext_cody.jsonl
    # 2) konuşmacısız kümeye gömme eşlemesi (pyannote/embedding, GPU)
    python scripts/pair_by_embedding.py --manifest artifacts/manifest/ext_afk.jsonl \
        --out artifacts/manifest/ext_afk_paired.jsonl --device cuda
    # 3) birleşik dışa aktarım (saat sınırını ihtiyaca göre büyüt)
    python scripts/export_qwen3tts_ft.py \
        --manifest artifacts/manifest/train_only.jsonl \
                   artifacts/manifest/review_paired.jsonl \
                   artifacts/manifest/ext_anil.jsonl \
                   artifacts/manifest/ext_afk_paired.jsonl \
                   artifacts/manifest/ext_cody.jsonl \
        --out artifacts/qwen3tts_ft/train.jsonl --max-hours 2600 --no-check-files
    # 4) ana kümenin klipleri: materialize_clips.py (harici kliplerin yolu
    #    zaten yerel/dökülmüş olduğundan yalnız ana küme id'leri dökülür)

Not: harici manifest'ler `src` (kayıt anahtarı) taşır; exporter aynı kayıttan
referans seçmez. DNSMOS süzgeci açılırsa eşiğimiz pratiği ~3,0'dır — tr-combined
ve afkfatih'in düşük ucunu eler, Anilosan neredeyse tamamen geçer.

## Kaynaklar

- https://github.com/QwenLM/Qwen3-TTS (Apache-2.0; "day-0 vLLM-Omni")
- https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning (JSONL biçimi,
  prepare_data.py, sft_12hz.py, tek-konuşmacı notu, örnek hiperparametreler)
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- https://github.com/vspeech/Qwen3-TTS-Train (çok-konuşmacılı FT)
- https://www.resemble.ai/learn/models/chatterbox-multilingual (B planı;
  MIT, 23 dil — Türkçe dahil, 0.5B Llama omurga)
- https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/
