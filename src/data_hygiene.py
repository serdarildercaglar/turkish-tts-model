"""Modele giren veriyi düzelten katman — kaynak manifestleri DEĞİŞTİRMEZ.

`scripts/audit_dataset.py` ile ölçülen hatalara karşılık gelir. Her karar
yapılandırılabilir; hiçbiri sessizce uygulanmaz, `Policy` alanlarıyla açılıp
kapatılır ve `Stats` her elemenin sayısını raporlar.

Metin düzeltmeleri kayıpsızdır (karakter değişimi), eleme ise satır düşürür.
Rakam sözelleştirme VARSAYILAN OLARAK KAPALIDIR — ayrı bir karardır ve
`Policy.verbalize_digits` ile açılır.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ------------------------------------------------------------------ metin
# Whisper Türkçe'de çift tırnağı '' (iki ASCII kesme) olarak yazıyor; bu
# karakter Türkçe'de ek sınırı işaretleyen kesme ile çakışıyor (Baba'ya).
_DOUBLE_APOS = re.compile(r"''+")
_FANCY_QUOTE = re.compile(r"[“”„«»]")
_FANCY_APOS = re.compile(r"[‘’‚‛′´`ʼ]")
_DASH = re.compile(r"[‐‑‒–—―−]")
_INVISIBLE = re.compile(r"[​‎‏﻿­]")
_NBSP = re.compile(r"[   ]")
_WS = re.compile(r"\s+")

# ASR döngüsü: aynı kelime arka arkaya 3+ kez
_REPEAT_RUN = re.compile(r"(\b\w+\b)(\s+\1\b){2,}", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

TERMINAL = ".!?"


def clean_text(t: str) -> str:
    """Kayıpsız karakter düzeltmeleri. Kelime silmez, sözcük sırasını bozmaz."""
    t = unicodedata.normalize("NFC", t)
    t = _INVISIBLE.sub("", t)
    t = _NBSP.sub(" ", t)
    t = _DOUBLE_APOS.sub('"', t)      # ''alıntı'' -> "alıntı"
    t = _FANCY_QUOTE.sub('"', t)
    t = _FANCY_APOS.sub("'", t)       # kıvrık kesme -> düz (ek sınırı)
    t = _DASH.sub("-", t)
    t = t.replace("…", "...")
    t = _WS.sub(" ", t)
    return t.strip()


def is_fragment_start(t: str) -> bool:
    """Klip cümle ortasından başlıyor mu (metin küçük harfle açılıyor)."""
    s = t.lstrip('"\'(')
    return bool(s) and s[0].isalpha() and s[0].islower()


def is_fragment_end(t: str) -> bool:
    """Klip cümle ortasında bitiyor mu (sonda cümle noktalaması yok)."""
    s = t.rstrip('"\')')
    return bool(s) and s[-1] not in TERMINAL


# --------------------------------------------------------------- politika
@dataclass
class Policy:
    """Neyin düşeceği ve neyin düzeleceği. Tümü açıkça yapılandırılır."""

    # --- metin düzeltmeleri (kayıpsız)
    clean_text: bool = True
    verbalize_digits: bool = False       # ayrı karar; varsayılan KAPALI

    # --- satır elemeleri
    drop_asr_loop: bool = True           # 'eğiliyor eğiliyor eğiliyor'
    drop_rate_outlier: bool = True       # metin-ses uyuşmazlığı işareti
    rate_min: float = 6.0                # harf/sn
    rate_max: float = 25.0
    min_words: int = 2
    drop_duplicate_recording: bool = True  # aynı kayıt mükerrer alınmış
    # aynı metnin FARKLI konuşmacıyla tekrarı prozodi çeşitliliğidir, korunur

    # --- review bayrakları: elenecek gerekçeler
    # 'sentetik_ses_suphesi' kasıtlı olarak listede yok: tek kişinin ses
    # değiştirip tiyatral okuması yanlış pozitif üretiyor, veri değerli.
    drop_reasons: tuple[str, ...] = (
        "asr_cift_gecis_uyusmazligi",    # CER p99 15, max 49 -> metin yanlış
        "farkli_konusmaci_suphesi",      # kosinüs p50 0.39 -> klonlamayı bozar
    )
    # eşik tabanlı elemeler (manifest kalite alanları)
    max_clip_ratio: float | None = 0.02
    max_music_score: float | None = 0.70
    min_speech_ratio: float | None = 0.70
    min_dnsmos_ovrl: float | None = 2.50
    min_speaker_cosine: float | None = 0.55

    # --- parça klipler
    # 'mark': bağlam tokenıyla işaretle (veri korunur)
    # 'drop': ele (~%25 kayıp)
    # 'keep': hiçbir şey yapma
    fragments: str = "mark"

    require_audio: bool = True           # ses dosyası diskte yoksa ele


@dataclass
class Stats:
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    fixed: Counter = field(default_factory=Counter)
    frag_start: int = 0
    frag_end: int = 0

    def as_dict(self) -> dict:
        return {
            "tutulan": self.kept,
            "elenen": dict(self.dropped.most_common()),
            "elenen_toplam": sum(self.dropped.values()),
            "duzeltilen": dict(self.fixed.most_common()),
            "parca_baslangic": self.frag_start,
            "parca_bitis": self.frag_end,
        }


def _reasons(row: dict) -> set[str]:
    v = row.get("review_reasons")
    if not v:
        return set()
    if isinstance(v, str):
        v = json.loads(v) if v.strip().startswith("[") else [v]
    return set(v)


def _num(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def speaker_key(row: dict) -> str:
    """Konuşmacı kimliği; review satırlarında `speaker_id` None geliyor.

    Review kümesi tek bir okuyucunun karakter seslerinden oluşuyor ve kanal
    başına bir anlatıcı düşüyor, dolayısıyla kanal makul bir vekil. Train'in
    kimlik uzayıyla karışmasın diye ad alanı ayrılır.
    """
    sid = row.get("speaker_id")
    if sid in (None, "", "None"):
        return f"channel:{row.get('channel')}"
    return str(sid)


class Hygiene:
    """Satırları politikaya göre süzer ve düzeltir; kararları sayar."""

    def __init__(self, policy: Policy | None = None):
        self.p = policy or Policy()
        self.stats = Stats()
        self._seen: dict[tuple, str] = {}
        self._verbalize = None
        if self.p.verbalize_digits:
            from src.text_frontend import prepare as _prep
            self._verbalize = _prep

    # -------------------------------------------------------------- satır
    def process(self, row: dict, audio_exists=None) -> dict | None:
        """Tek satırı işler. Elenirse None, tutulursa yeni satır sözlüğü."""
        p, st = self.p, self.stats
        text = row.get("text") or ""
        dur = _num(row, "duration") or 0.0

        if p.clean_text:
            cleaned = clean_text(text)
            if cleaned != text:
                st.fixed["metin temizligi"] += 1
            text = cleaned

        if not text:
            st.dropped["metin bos"] += 1
            return None
        if len(_WORD.findall(text)) < p.min_words:
            st.dropped["kelime sayisi yetersiz"] += 1
            return None
        if dur <= 0:
            st.dropped["sure gecersiz"] += 1
            return None

        if p.drop_asr_loop and _REPEAT_RUN.search(text):
            st.dropped["ASR dongusu"] += 1
            return None

        if p.drop_rate_outlier:
            rate = len(text) / dur
            if not (p.rate_min <= rate <= p.rate_max):
                st.dropped["harf/sn aykiri"] += 1
                return None

        # review bayrakları
        hit = _reasons(row) & set(p.drop_reasons)
        if hit:
            st.dropped[f"gerekce: {sorted(hit)[0]}"] += 1
            return None

        for key, lim, cmp_, tag in (
            ("quality_clip_ratio", p.max_clip_ratio, "max", "kirpma"),
            ("quality_music_score", p.max_music_score, "max", "muzik"),
            ("quality_speech_ratio", p.min_speech_ratio, "min", "konusma orani"),
            ("quality_dnsmos_ovrl", p.min_dnsmos_ovrl, "min", "dnsmos"),
            ("quality_speaker_cosine", p.min_speaker_cosine, "min",
             "konusmaci kosinusu"),
        ):
            if lim is None:
                continue
            v = _num(row, key)
            if v is None:
                continue
            if (cmp_ == "max" and v > lim) or (cmp_ == "min" and v < lim):
                st.dropped[f"esik: {tag}"] += 1
                return None

        if p.require_audio and audio_exists is not None and not audio_exists:
            st.dropped["ses dosyasi yok"] += 1
            return None

        # mükerrer kayıt: aynı konuşmacı + aynı metin + ~aynı süre.
        # farklı konuşmacının aynı metni okuması KORUNUR (prozodi çeşitliliği).
        if p.drop_duplicate_recording:
            key = (speaker_key(row), text, round(dur, 1))
            if key in self._seen:
                st.dropped["mukerrer kayit"] += 1
                return None
            self._seen[key] = row["id"]

        fs, fe = is_fragment_start(text), is_fragment_end(text)
        if fs:
            st.frag_start += 1
        if fe:
            st.frag_end += 1
        if p.fragments == "drop" and (fs or fe):
            st.dropped["parca klip"] += 1
            return None

        if self._verbalize is not None:
            v = self._verbalize(text)
            if v != text:
                st.fixed["rakam sozellestirme"] += 1
            text = v

        st.kept += 1
        out = dict(row)
        out["text"] = text
        out["speaker_key"] = speaker_key(row)
        if p.fragments == "mark":
            out["frag_start"] = fs
            out["frag_end"] = fe
        return out
