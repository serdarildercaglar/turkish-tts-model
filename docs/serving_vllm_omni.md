# vllm-omni ile servis (`/v1/audio/speech`)

**Neden vllm-omni, vLLM değil:** vLLM'in OpenAI sunucusunda TTS ucu
(`/v1/audio/speech`) yoktur; yalnız `/v1/completions` ve Whisper tarafı
(`/v1/audio/transcriptions`) vardır. Ham vLLM'de bu model yalnız
`<custom_token_N>` metni üretebilir, sesi istemci çözmek zorunda kalır.
[vllm-omni](https://github.com/vllm-project/vllm-omni) ise `vllm serve <model>
--omni` ile OpenAI uyumlu **`POST /v1/audio/speech`**, WebSocket
**`/v1/audio/speech/stream`** (artımlı metin girişi + PCM akışı) ve
`/v1/audio/speech/batch` uçlarını sunar; SSE ya da ham PCM akışı, `ref_audio`
ile klonlama ve `voice` ön ayarları serving katmanında hazırdır. Gerçek zamanlı
IVR hedefi için doğru katman budur.

**Model tarafında değişiklik yok.** Checkpoint stok `LlamaForCausalLM` +
SNAC codec tokenları (Orpheus şeması); vllm-omni'nin AR aşaması tam olarak bu
tür modelleri vLLM'in yerel Llama katmanlarıyla (KV önbelleği, sürekli
batching, CUDA graph) koşturur. Değişen yalnız servis katmanı: küçük bir
**out-of-tree eklenti** (fork gerekmez) modeli iki aşamalı bir hatta bağlar.

## Eklenti mimarisi

vllm-omni üç kayıt noktasını dışarıdan doldurmaya izin verir
(kaynak: `vllm_omni/plugins/__init__.py`, `config/pipeline_registry.py:register_pipeline`,
`entrypoints/openai/tts_adapters/__init__.py:register_tts_adapter`; 23 Ağu 2026,
commit `1c917b2` ile doğrulandı):

```
serving/turkish_tts_omni/            # pip install -e serving/
├── pyproject.toml                   # [project.entry-points."vllm_omni.general_plugins"]
│                                    #   turkish_tts = "turkish_tts_omni:register"
├── __init__.py                      # register(): ModelRegistry.register_model(...),
│                                    #   register_pipeline(PIPELINE), adapter import
├── model.py                         # TurkishTTSForConditionalGeneration (birleşik sınıf)
│                                    #   model_stage == "ar":      vLLM LlamaForCausalLM sarmalayıcı
│                                    #   model_stage == "decoder": SNACDecoder
├── pipeline.py                      # PipelineConfig(model_type="turkish_tts", 2 aşama)
├── stage_input.py                   # ar2decoder (toplu) + ar2decoder_async_chunk (akış)
├── adapter.py                       # @register_tts_adapter class TurkishTTSAdapter(ARTTSAdapter)
└── deploy/turkish_tts.yaml          # --deploy-config; `pipeline: turkish_tts` ile hattı sabitler
```

### Aşama 0 — AR (`StageExecutionType.LLM_AR`, `owns_tokenizer=True`)

`init_vllm_registered_model` ile stok Llama yüklenir; sarmalayıcının tek
işi her adımda `multimodal_outputs["codes"]["audio"]`'ya SNAC kodunu koymaktır.
Örnekleme forward'dan sonra olduğundan model kendi ürettiği tokenı o adımda
bilmez; ama bir sonraki adımın `input_ids`'i tam o tokendır — dolayısıyla
"bir önceki adımda örneklenen token" (`id − AUDIO_BASE`, `is_audio_id` ise)
raporlanır. Durdurucular: `<|audio_end|>`, `<|eos|>` (`sampling_constraints`
içinde `stop_token_ids`, `detokenize=False`). `config.json` stok Llama kalır;
sınıf adı deploy yaml'daki `hf_overrides: {architectures: [...]}` ile verilir,
böylece aynı checkpoint HF/transformers ile de değişmeden yüklenir.

### Aşama 1 — SNAC çözücü (`StageExecutionType.LLM_GENERATION`, `final_output_type="audio"`)

`src/codec.py`'deki `unflatten_codes` + `snac.decode` ile aynı iş; akış için
`chunked_decode_streaming(codes, chunk_frames, left_context_frames)`
(vllm-omni'nin Qwen3-TTS/Voxtral çözücüleriyle aynı sözleşme). Çerçeve birimi
7 token = 2.048 örnek = **85,3 ms** ses (~11,7 çerçeve/s). Aşamalar arası
aktarım `SharedMemoryConnector` (`codec_streaming: true`); yaml'daki
`codec_chunk_frames`, `codec_chunk_frames_at_begin`,
`codec_left_context_frames`, `crossfade_sec` alanları pencereyi belirler.

### Adapter (`/v1/audio/speech` isteği → engine istemi)

- `validate`: `input` boş olamaz; `ref_audio` verildiyse `ref_text` zorunlu
  (klon istemi transkript ister); `voice` verildiyse ön ayar listesinde olmalı.
- `build`: istem token dizisini `src/data.py`'deki kurucuyla üretir —
  düz: `<|bos|><|plain_tts|><|text_start|>T<|text_end|><|audio_start|>`;
  klon: `<|bos|><|clone_tts|><|text_start|>ref_T T<|text_end|><|audio_start|>REF_SES`.
  `ref_audio` API sürecinde SNAC ile kodlanır (`infer.py:encode_ref`, ≤117
  çerçeve ≈ 10 sn). **Ön ayarlı sesler** (`voice: "..."`) bir dizinden
  `(ref_codes, ref_text)` çifti olarak önceden yüklenir — IVR'ın sabit marka
  sesi budur; sabit referans öneki tüm isteklerde aynı olduğundan aşama 0'da
  `enable_prefix_caching: true` ile neredeyse bedavaya gelir.
- `apply_sampling_overrides`: temp 0,7 / top_p 0,9 / rep-pen 1,1 varsayılanları
  ve `max_new_tokens` → `max_tokens` (1 sn ses ≈ 83 token; 30 sn için 2.500).

### Deploy yaml (taslak)

```yaml
async_chunk: true
pipeline: turkish_tts
connectors:
  shm:
    name: SharedMemoryConnector
    extra:
      codec_streaming: true
      codec_chunk_frames: 12          # ≈1,0 s
      codec_chunk_frames_at_begin: 4  # ≈340 ms — ilk sese hızlı ulaş
      codec_left_context_frames: 8
      crossfade_sec: 0.02
stages:
  - stage_id: 0                       # AR — stok Llama
    devices: "0"
    max_num_seqs: 32
    max_model_len: 3072
    gpu_memory_utilization: 0.35
    enable_prefix_caching: true
    async_scheduling: true
    dtype: bfloat16
    hf_overrides: {architectures: [TurkishTTSForConditionalGeneration]}
    output_connectors: {to_stage_1: shm}
    default_sampling_params:
      temperature: 0.7
      top_p: 0.9
      repetition_penalty: 1.1
      max_tokens: 2500
      detokenize: false
  - stage_id: 1                       # SNAC çözücü
    devices: "0"
    max_num_seqs: 32
    gpu_memory_utilization: 0.1
    enforce_eager: true
    hf_overrides: {architectures: [TurkishTTSForConditionalGeneration]}
    input_connectors: {from_stage_0: shm}
```

Kullanım:

```bash
pip install vllm-omni && pip install -e serving/
vllm serve serdarildercaglar/turkish-tts-model --omni \
    --deploy-config serving/turkish_tts_omni/deploy/turkish_tts.yaml --port 8091

curl -s http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Merhaba, bugün hava çok güzel.","voice":"ivr_kadin","response_format":"wav"}' \
  -o cikti.wav
# klonlama: {"input": "...", "ref_audio": "data:audio/wav;base64,...", "ref_text": "referansın transkripti"}
# akış: "stream": true, "stream_format": "audio"  (ham PCM parçaları)
```

## Gerçek zamanlı IVR notları

### Ölçülen hız (24 Ağu 2026, boş RTX 3090, vLLM 0.26, CUDA graph FULL_AND_PIECEWISE, rastgele ağırlıklı 95M)

`scripts/bench_vllm_decode.py` ile; 200 tokenlik istem + 1.000 token üretim.
Sayı, **adım süresi** ile belirlenir (model 190 MB — bellek değil, kernel
başlatma tabanında koşar):

| eşzamanlı akış | ms/adım | akış başı tok/s | akış başı RTF | toplam ses-s/s |
|---:|---:|---:|---:|---:|
| 1 | 1,12–1,22 | 820–890 | **0,09–0,10** | 10 |
| 8 | 1,18 | 851 | 0,10 | 82 |
| 16 | 1,47 | 681 | 0,12 | 131 |
| 32 | 2,11 | 475 | 0,18 | 183 |
| 64 | 3,44 | 290 | 0,29 | 224 |

H100/H200'de tek akış adımı tahminen 0,7–1,0 ms (RTF ≈ 0,06–0,08); 32–64
akışta 3090'daki büyümenin nedeni KV okuması (MHA, 10 KV başı) olduğu için
Hopper'da eğri daha düz kalır. **Aynı betiği H100'de koşturup bu tabloyu
güncelleyin.**

### "RTF 0,05" hedefi üzerine (ajan tartışmasının sonucu)

- Tek akışta RTF 0,05 = 0,6 ms/adım demek; 7 token/çerçeve düzeninde 83
  ardışık adım/s gerektiğinden bu, vLLM'in küçük-model tabanının (H100'de
  ~0,7–1 ms) altındadır. **Bu mimariyle dürüst vaat: H100 tek akış RTF
  0,06–0,10, 32 akışta ≤ 0,2.** Spekülatif çözme ses tokenlarında kazandırmaz
  (LLaSA-8B'de 0,98×, PCG 1,4× ama WER bozuluyor — arXiv 2511.13732).
- IVR için bağlayıcı ölçütler RTF 0,05 değil: **ilk paket p95 ≤ 250 ms**,
  hedef eşzamanlılıkta akış başı **RTF p95 ≤ 0,3**, bekleme (underrun) oranı
  %0, 20 ms RTP kadansı + 60–80 ms jitter tamponu. Yukarıdaki tablo 3090'da
  bile 32 çağrıyı 5× payla taşır; tek H100 100+ çağrı sınıfındadır.
- RTF 0,03–0,05 gerçekten şartsa yol, çerçeve başına **tek adımda 7 yuva**
  üreten çok-başlı/delay-pattern model (Qwen3-TTS code-predictor, MOSS delay
  deseni): adım sayısı 83 → 12/s. Bedeli: stok Llama'dan çıkış (özel vLLM
  sınıfı + çok-başlı örnekleme), HF `generate` uyumunun bitmesi, düz delay
  deseninin ilk paketi +0,5 s geciktirmesi ([0,1,1,1,1,1,1] ile +85 ms),
  SNAC'ın 12/24/47 Hz hiyerarşisinin MusicGen-tipi eşit-oranlı delay'e tam
  oturmaması ve 1.292 saatlik veride kalite riski (MusicGen'de "parallel"
  varyantı belirgin kötü, delay düzleştirmeye yakın). Karar: **v1 stok Llama +
  düzleştirme; çok-başlı varyant pilot sonrası A/B kalemi** (SNAC kodları
  seviye bazında saklandığından veri yeniden kodlanmaz, yalnız paketleme + 7
  baş değişir).
- Ucuz kaldıraçlar (stok Llama'yı bozmaz): **GQA** (`num_key_value_heads`
  10 → 2; 64 akışta adım büyümesinin kaynağı KV okuması), sampler diyeti
  (yalnız temperature + ses-token aralığı maskesi), `async_scheduling`,
  Stage 0 `max_num_seqs` 32–64, ilk SNAC parçası 2–4 çerçeve.

### Dağıtım notları

- **Gecikme bütçesi:** ilk ses = prefill (~1.000 token, referans öneki
  prefix-cache'te → birkaç ms) + 4 çerçeve × 7 = 28 adım (~30 ms H100) + SNAC
  parça çözümü (~5 ms) → **ilk paket ~50–80 ms**, ağ hariç.
- **Telefon hattı:** çıkış 24 kHz PCM; G.711 (8 kHz μ-law/A-law) ya da 16 kHz
  gerekiyorsa yeniden örnekleme SIP/medya geçidinde yapılır (`response_format:
  pcm`). SNAC 24 kHz'e sabittir.
- **Artımlı metin:** LLM'den cümle cümle gelen metin için WebSocket
  `/v1/audio/speech/stream` (`session.config` → `input.text` … → `input.done`)
  oturumu tekrar kullanılabilir; cümle sınırında `input.done` aruzu korur.
- **Sabit ses:** IVR marka sesi = sabit `voice` ön ayarı (ref kodları + ref
  metni); `enable_prefix_caching: true` ile referans öneki bir kez prefill edilir.
- **Uzunluk:** max_len 3072 ≈ 10 sn referans + ~25 sn hedef; uzun metin
  adapter'da cümlelere bölünür.
- **B planı (vllm-omni kırılırsa):** checkpoint stok Llama olduğundan düz
  `vllm serve` + `/v1/completions` (`stream`, `return_token_ids`) + ince
  FastAPI/WS geçidi (SNAC çözümü GPU'da 20–40 ms pencereli mikro-batch) bir
  günlük iştir; Orpheus üretim kurulumları (Baseten, Simplismart) bu desenle
  koşar. vllm-omni tercih nedeni: hazır `/v1/audio/speech`, WebSocket akışı,
  ses deposu ve toplu çözücü aşaması; riski: alfa API, iki haftalık sürüm
  kadansı — **sürümü sabitleyin** (`vllm-omni==0.26.x`).

## Durum ve sonraki adım

Eklenti paketi (`serving/turkish_tts_omni/`) **pilot checkpoint çıkınca**
yazılıp duman testinden geçirilecek; API'yi sabitlemeden yazılmış kodun
vllm-omni'nin hızlı değişen iç sözleşmesine karşı test edilmesi gerekir.
vllm-omni kendi vLLM/torch sürümünü kilitler; eğitim ortamından (`main`,
torch 2.5.1) ayrı bir conda ortamı kurun (mevcut `vllm` ortamındaki vLLM
0.26.0 muhtemelen eski, `pip install vllm-omni` doğru vLLM'i getirir).

## Yedek: ham vLLM ile metin-tamamlama

Hata ayıklama ya da vllm-omni olmayan ortamlar için eski yol çalışmaya
devam eder: `vllm serve ckpt/final --tokenizer artifacts/tokenizer` →
`/v1/completions`, durdurucular `<|audio_end|>`/`<|eos|>`; dönen
`<custom_token_N>` dizisi `re.findall(r"<custom_token_(\d+)>", text)` ile
alınıp `src.codec.decode_to_wav` ile çözülür (`repetition_penalty` 1,05–1,2).
Bu yolda ses ucu ve akışlı çözüm yoktur; üretim için değildir.
