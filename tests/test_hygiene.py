import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_hygiene import (
    Hygiene, Policy, clean_text, is_fragment_end, is_fragment_start, speaker_key,
)
from src.prosody import Prosody, bucket_of, compute_edges


def row(**kw):
    base = dict(id="x", text="Merhaba dünya.", duration=1.5,
                speaker_id="spk-1", channel="kanal")
    base.update(kw)
    return base


# ------------------------------------------------------------------ metin
def test_whisper_double_apostrophe_becomes_quote():
    """Whisper Turkce'de tirnagi '' yaziyor; bu ek kesmesiyle cakisiyor."""
    t = clean_text("Baba'ya ''Bizi kurtar'' der gibi baktı.")
    assert '"Bizi kurtar"' in t
    assert "Baba'ya" in t          # ek kesmesi KORUNUR
    assert "''" not in t


def test_ellipsis_and_spacing():
    assert clean_text("Bir  şey…") == 'Bir şey...'
    assert clean_text("  boşluk  ") == "boşluk"


def test_fragment_detection():
    assert is_fragment_start("kalbin de içinde") and not is_fragment_start("Kalp.")
    assert is_fragment_end("devam ediyor") and not is_fragment_end("Bitti.")


# ------------------------------------------------------------------ eleme
def test_drops_asr_loop():
    hy = Hygiene(Policy())
    assert hy.process(row(text="eğiliyor eğiliyor eğiliyor"), None) is None
    assert hy.stats.dropped["ASR dongusu"] == 1


def test_drops_rate_outlier():
    hy = Hygiene(Policy())
    assert hy.process(row(text="Çok kısa.", duration=30.0), None) is None
    assert hy.stats.dropped["harf/sn aykiri"] == 1


def test_same_recording_deduped_but_different_speaker_kept():
    """Ayni kaydin mukerrer alimi elenir; FARKLI konusmaci prozodi cesitliligidir."""
    hy = Hygiene(Policy())
    assert hy.process(row(id="a"), None) is not None
    assert hy.process(row(id="b"), None) is None                    # ayni kayit
    assert hy.process(row(id="c", speaker_id="spk-2"), None) is not None


def test_review_reason_filtering():
    hy = Hygiene(Policy())
    # sentetik ses suphesi YANLIS POZITIF: tek okuyucunun tiyatral okumasi
    assert hy.process(row(review_reasons=["sentetik_ses_suphesi"]), None) is not None
    assert hy.process(row(id="y", review_reasons=["farkli_konusmaci_suphesi"]),
                      None) is None


def test_quality_thresholds():
    hy = Hygiene(Policy())
    assert hy.process(row(quality_music_score=0.9), None) is None
    assert hy.stats.dropped["esik: muzik"] == 1


def test_speaker_key_falls_back_to_channel():
    """Review satirlarinda speaker_id None geliyor."""
    assert speaker_key(row(speaker_id=None)) == "channel:kanal"
    assert speaker_key(row()) == "spk-1"


def test_fragments_marked_not_dropped_by_default():
    hy = Hygiene(Policy())
    out = hy.process(row(text="devam eden cümle"), None)
    assert out is not None and out["frag_start"] and out["frag_end"]
    hy2 = Hygiene(Policy(fragments="drop"))
    assert hy2.process(row(text="devam eden cümle"), None) is None


def test_digits_untouched_by_default():
    hy = Hygiene(Policy())
    out = hy.process(row(text="Saat 09:30'da geldi."), None)
    assert "09:30" in out["text"]


# ---------------------------------------------------------------- prozodi
def test_prosody_buckets():
    edges = compute_edges(list(range(100)), 5)
    assert len(edges) == 4
    assert bucket_of(-1, edges) == 0 and bucket_of(1000, edges) == 4
    assert bucket_of(None, edges) is None


def test_prosody_roundtrip(tmp_path):
    p = Prosody.from_rows([
        {"text": "a" * n, "duration": 1.0, "quality_lufs": -20 - n / 10}
        for n in range(5, 105)
    ])
    p.save(tmp_path)
    q = Prosody.load(tmp_path)
    assert q.rate_edges == p.rate_edges and q.loud_edges == p.loud_edges
