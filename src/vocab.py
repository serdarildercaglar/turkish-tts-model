"""LM sözlük düzeni — tek doğruluk kaynağı.

Metin BPE, ses tokenları, özel tokenlar ve prozodi kontrol tokenları tek
embedding tablosunu paylaşır. Ses tokenları tokenizer'da `<custom_token_N>`
adlı GERÇEK tokenlardır; vLLM motoru (vllm-omni'nin AR aşaması) bu sayede
onları sıradan token olarak üretir.

Düzen (63M için toplam 16.448 giriş → gömme 8,4M, modelin %13'ü):

    [0, 4.096)              metin BPE
    [4.096, 16.384)         ses tokenları (3 SNAC düzeyi × 4.096)
    [16.384, ...)           özel tokenlar + prozodi kontrol tokenları
    16.448'e kadar          64'ün katına yastıklama (verimli matmul)

Önceki düzende ses tokenları 7 yuva × 4.096 = 28.672 giriş kaplıyordu ve metin
BPE 8.192'ydi; gömme tablosu modelin %25'iydi. Düzey başına ofset (bkz.
`src.codec`) ve daha küçük BPE ile aynı bilgi %13'e sığıyor, kazanılan
parametreler katmanlara gidiyor.
"""
from src.codec import AUDIO_VOCAB

TEXT_VOCAB = 4096
AUDIO_BASE = TEXT_VOCAB                      # 4.096
SPECIAL_BASE = TEXT_VOCAB + AUDIO_VOCAB      # 16.384

SPECIALS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|text_start|>",
    "<|text_end|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|plain_tts|>",
    "<|clone_tts|>",
    # Klip cümle ortasından başlıyor / bitiyor. Korpusun ~dörtte biri cümle
    # parçası; atmak yerine işaretlenir, böylece model "devam" başlangıcıyla
    # "başlatma"yı ayırt eder ve çıkarımda tokenı vermeyerek tam cümle isteriz.
    "<|frag_start|>",
    "<|frag_end|>",
]

# --------------------------------------------------------- prozodi kontrolü
# Klip başına ölçülebilir özniteliklerden kova tokenları. Eğitimde istemin
# başına yazılır, çıkarımda istenen değer verilir. `<|rate_any|>` / `<|loud_any|>`
# eğitim sırasında belirli bir olasılıkla konur (classifier-free tarzı), böylece
# çıkarımda kontrol verilmediğinde model öğrenilmiş ortalamaya düşer.
RATE_BUCKETS = 5
LOUD_BUCKETS = 3
CONTROLS = (
    [f"<|rate_{i}|>" for i in range(RATE_BUCKETS)] + ["<|rate_any|>"]
    + [f"<|loud_{i}|>" for i in range(LOUD_BUCKETS)] + ["<|loud_any|>"]
)

ALL_ADDED = SPECIALS + CONTROLS
# 64'ün katına yuvarlanmış toplam sözlük.
VOCAB_SIZE = 16448


def _sid(name: str) -> int:
    return SPECIAL_BASE + ALL_ADDED.index(name)


BOS = _sid("<|bos|>")
EOS = _sid("<|eos|>")
PAD = _sid("<|pad|>")
TEXT_START = _sid("<|text_start|>")
TEXT_END = _sid("<|text_end|>")
AUDIO_START = _sid("<|audio_start|>")
AUDIO_END = _sid("<|audio_end|>")
PLAIN_TTS = _sid("<|plain_tts|>")
CLONE_TTS = _sid("<|clone_tts|>")
FRAG_START = _sid("<|frag_start|>")
FRAG_END = _sid("<|frag_end|>")

RATE_BASE = _sid("<|rate_0|>")
RATE_ANY = _sid("<|rate_any|>")
LOUD_BASE = _sid("<|loud_0|>")
LOUD_ANY = _sid("<|loud_any|>")

assert SPECIAL_BASE + len(ALL_ADDED) <= VOCAB_SIZE, (
    f"{SPECIAL_BASE + len(ALL_ADDED)} > {VOCAB_SIZE}")


def rate_token(bucket: int | None) -> int:
    """Konuşma hızı kova tokenı; None ise `<|rate_any|>`."""
    return RATE_ANY if bucket is None else RATE_BASE + bucket


def loud_token(bucket: int | None) -> int:
    """Ses seviyesi kova tokenı; None ise `<|loud_any|>`."""
    return LOUD_ANY if bucket is None else LOUD_BASE + bucket


def audio_token_name(i: int) -> str:
    return f"<custom_token_{i}>"


def audio_id_to_lm(flat_code: int) -> int:
    return AUDIO_BASE + flat_code


def lm_id_to_audio(lm_id: int) -> int:
    return lm_id - AUDIO_BASE


def is_audio_id(lm_id: int) -> bool:
    return AUDIO_BASE <= lm_id < AUDIO_BASE + AUDIO_VOCAB
