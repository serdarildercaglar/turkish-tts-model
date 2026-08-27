"""Türkçe metin ön işleme: unicode sadeleştirme + sayı/tarih/para sözelleştirme.

Korpus transkriptleri Whisper'dan geldiği için ses sözlü formu söylerken metin
rakam yazar (satırların ~%7,3'ünde rakam var). Modelin metni ile duyduğu ses
arasındaki bu uyuşmazlığı kapatmak için hem EĞİTİM hem ÇIKARIM metni buradan
geçer. Aynı işlev değerlendirmede ASR hipotezine de uygulanır; yoksa ASR'nin
rakamla yazdığı çıktı doğru okunuşu asla tutturamaz ve WER yapay olarak şişer.

`canberkkkkkk/ema-tts` (Apache-2.0) içindeki `text.py`'den uyarlandı. Farklar:

- **Büyük/küçük harf korunur.** Kaynak modelin karakter sözlüğü yalnız küçük
  harfti; bizim 8k BPE'miz karışık harfli metinle eğitiliyor, dolayısıyla
  küçültme bilgi kaybı olur (`IBAN` → `ıban` hatası da böyle doğuyordu).
- **Alfanümerik içindeki rakamlar genişletilir.** `TR33` → `TR 33`; kaynakta
  `\b` sınırı tutmadığı için ham rakam modele düşüyordu.
- **Eksi işareti korunur.** `-350 TL` kaynakta `-üç yüz elli` oluyordu; para
  kuralı sayıyı önce kaptığı için işaret dışarıda kalıyordu.
- **Bileşik hız birimi.** `120 km/sa` → `saatte yüz yirmi kilometre`.
- **Para birimi kesirleri.** `1.245,50 TL` → `... lira elli kuruş`
  (kaynakta `virgül elli te le`).
- **Simgeler sözelleşir.** `@ # & + *` — IVR'da tuş adı olarak geçiyorlar.
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------- unicode
# kıvrık tırnaklar, primler ve tireler aynı sesi verir
PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "′": "'", "´": "'", "`": "'", "ʼ": "'",
    "“": '"', "”": '"', "„": '"', "«": '"',
    "»": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", "​": "",
    "﻿": "", "‎": "", "‏": "",
}

# şapkalı harfleri neredeyse kimse yazmaz; sese katkısı yok
CIRCUMFLEX = {
    "â": "a", "Â": "A", "î": "i", "Î": "İ", "û": "u", "Û": "U",
    "ê": "e", "Ê": "E", "ô": "o", "Ô": "O",
}

TR_LETTER = "A-Za-zÇĞİÖŞÜçğıöşü"

TERMINAL = ".!?"
PUNCT_AFTER_SPACE = ".,!?;:"

_MULTI_DOT = re.compile(r"\.{2,}")
_MULTI_MARK = re.compile(r"([!?])[!?]+")
_MIXED_MARK = re.compile(r"[!?]{2,}")
_MULTI_COMMA = re.compile(r",{2,}")
_SPACE_BEFORE = re.compile(r"\s+([.,!?;:])")
_SPACE_AFTER = re.compile(rf"([{re.escape(PUNCT_AFTER_SPACE)}])(?=[^\s\d])")
_SPACE_AROUND_APOS = re.compile(r"\s*'\s*")
_WS = re.compile(r"\s+")


def normalize(t: str, add_final: bool = True) -> str:
    """Unicode ve noktalama sadeleştirmesi. Harf büyüklüğü korunur."""
    t = unicodedata.normalize("NFC", t)

    for k, v in PUNCT_MAP.items():
        t = t.replace(k, v)

    # NFC sonrası artakalan birleşen işaretler (görünmez, kendi kod noktası)
    t = "".join(c for c in t if not unicodedata.combining(c))

    for k, v in CIRCUMFLEX.items():
        t = t.replace(k, v)

    # yinelenen işaretler sohbet vurgusu; model bunu hiç görmedi
    t = _MULTI_DOT.sub(".", t)
    t = _MIXED_MARK.sub(lambda m: "?" if "?" in m.group(0) else "!", t)
    t = _MULTI_MARK.sub(r"\1", t)
    t = _MULTI_COMMA.sub(",", t)

    t = _SPACE_BEFORE.sub(r"\1", t)
    t = _SPACE_AFTER.sub(r"\1 ", t)
    t = _SPACE_AROUND_APOS.sub("'", t)
    t = _WS.sub(" ", t).strip()

    if add_final and t and t[-1] not in TERMINAL:
        t = t.rstrip(",;:") + "."
    return t


# ---------------------------------------------------------------- sayılar
_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz",
         "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş",
         "seksen", "doksan"]
_SCALE = [(10**15, "katrilyon"), (10**12, "trilyon"), (10**9, "milyar"),
          (10**6, "milyon"), (10**3, "bin")]

# Türkçede sıra sayısı, asıl sayı + ek değil; düzensizleri tabloya değer
_ORD = {1: "birinci", 2: "ikinci", 3: "üçüncü", 4: "dördüncü", 5: "beşinci",
        6: "altıncı", 7: "yedinci", 8: "sekizinci", 9: "dokuzuncu",
        10: "onuncu", 20: "yirminci", 30: "otuzuncu", 40: "kırkıncı",
        50: "ellinci", 60: "altmışıncı", 70: "yetmişinci",
        80: "sekseninci", 90: "doksanıncı", 100: "yüzüncü",
        1000: "bininci", 10**6: "milyonuncu"}

MONTHS = ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
          "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

# kuruş alt birimi olanlar: tam kısım + iki haneli kesir ayrı okunur
CURRENCY = {
    "₺": ("lira", "kuruş"), "TL": ("lira", "kuruş"),
    "$": ("dolar", "sent"), "€": ("avro", "sent"),
    "£": ("sterlin", "peni"), "¥": ("yen", None),
}

UNITS = {
    "km": "kilometre", "cm": "santimetre", "mm": "milimetre",
    "m": "metre", "m²": "metrekare", "m³": "metreküp",
    "kg": "kilogram", "gr": "gram", "g": "gram", "mg": "miligram",
    "lt": "litre", "l": "litre", "ml": "mililitre",
    "dk": "dakika", "sn": "saniye", "sa": "saat",
    "GB": "gigabayt", "MB": "megabayt", "TB": "terabayt", "KB": "kilobayt",
    "kW": "kilovat", "MW": "megavat", "W": "vat",
    "°C": "derece", "°": "derece", "%": "yüzde",
}

# bölü işaretiyle yazılan hız birimleri: Türkçesi sayıyı sona alır
PER_UNITS = {"sa": "saatte", "h": "saatte", "s": "saniyede",
             "sn": "saniyede", "dk": "dakikada"}

# IVR'da tuş adı olarak geçen simgeler
SYMBOLS = {"@": " et ", "&": " ve ", "+": " artı ", "*": " yıldız ",
           "#": " kare ", "_": " alt çizgi ", "§": " paragraf "}


def _u1000(n: int) -> str:
    o = []
    h, r = divmod(n, 100)
    if h:
        o.append(("" if h == 1 else _ONES[h] + " ") + "yüz")
    t, x = divmod(r, 10)
    if t:
        o.append(_TENS[t])
    if x:
        o.append(_ONES[x])
    return " ".join(o)


def int2tr(n: int) -> str:
    """Asıl sayı okunuşu. `bir bin` denmez, `bin` denir."""
    if n < 0:
        return "eksi " + int2tr(-n)
    if n == 0:
        return "sıfır"
    p = []
    for v, name in _SCALE:
        q, n = divmod(n, v)
        if q:
            pre = _u1000(q)
            p.append("bin" if (v == 1000 and q == 1)
                     else (pre + " " if pre else "") + name)
    if n:
        p.append(_u1000(n))
    return " ".join(p)


def ord2tr(n: int) -> str:
    """Sıra sayısı. Yalnız son bileşen sıra biçimi alır: `yirmi üçüncü`."""
    if n in _ORD:
        return _ORD[n]
    for div in (10**6, 1000, 100, 10):
        if n > div:
            head, tail = divmod(n, div)
            if tail == 0:
                return (int2tr(head) + " " + _ORD[div]).strip()
            return (int2tr(n - tail) + " " + ord2tr(tail)).strip()
    return _ORD.get(n, int2tr(n) + "ıncı")


def dec2tr(whole: int, frac: str, sep_word: str = "virgül") -> str:
    """Ondalık. Kesir tek sayı olarak okunur: `üç virgül on dört`.

    Baştaki sıfır söylenir, çünkü `0,05` ile `0,5` farklıdır.
    """
    lead = "sıfır " * (len(frac) - len(frac.lstrip("0")))
    rest = frac.lstrip("0")
    tail = (lead + (int2tr(int(rest)) if rest else "")).strip() or "sıfır"
    return f"{int2tr(whole)} {sep_word} {tail}"


def _ungroup(s: str) -> int:
    """`1.543` tek sayıdır; noktalar ayırıcı, ondalık işareti değil."""
    return int(s.replace(".", ""))


# ------------------------------------------------------------- ön geçişler

_SIGN = re.compile(r"(?<![\w.,])-(?=\d)")
# harf-rakam sınırı: `TR33` -> `TR 33`. Bilinen birim/para kısaltmaları
# ayrılmaz, yoksa `120km` gibi yazımlar ölçü kuralından kaçar.
_ALNUM_LD = re.compile(rf"(?<=[{TR_LETTER}])(?=\d)")
_ALNUM_DL = re.compile(rf"(?<=\d)(?=[{TR_LETTER}])")
_KNOWN_TAIL = tuple(sorted(set(UNITS) | set(CURRENCY), key=len, reverse=True))


def _split_alnum(t: str) -> str:
    """Harf ve rakamın yapıştığı yere boşluk koyar; birimleri korur."""
    def dl(m: re.Match) -> str:
        rest = t[m.start():]
        for u in _KNOWN_TAIL:
            if rest.startswith(u):
                nxt = rest[len(u):len(u) + 1]
                if not nxt or not nxt.isalnum():
                    return ""          # `120km` — ölçü kuralına bırak
        return " "
    return _ALNUM_LD.sub(" ", _ALNUM_DL.sub(dl, t))


# ---------------------------------------------------------------- kurallar
# Sırayla denenir. Bir aralığı ilk eşleşen kural sahiplenir; kısa biçimle
# karışabilecek her şey önce gelmeli.

def _date(m):
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    if y < 100:
        y += 2000 if y < 50 else 1900
    return f"{int2tr(d)} {MONTHS[mo]} {int2tr(y)}"


def _day_month(m):
    d, mo = int(m.group(1)), int(m.group(2))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    return f"{int2tr(d)} {MONTHS[mo]}"


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    if mi == 0:
        return int2tr(h)
    if mi == 30:
        return f"{int2tr(h)} buçuk"
    return f"{int2tr(h)} {int2tr(mi)}"


def _percent(m):
    body = m.group(1)
    if "," in body:
        w, f = body.split(",")
        return "yüzde " + dec2tr(_ungroup(w), f)
    return "yüzde " + int2tr(_ungroup(body))


def _money(body: str, sym: str) -> str:
    major, minor = CURRENCY.get(sym, CURRENCY.get(sym.upper(), (sym, None)))
    if "," not in body:
        return f"{int2tr(_ungroup(body))} {major}"
    w, f = body.split(",")
    whole = int2tr(_ungroup(w))
    # iki haneli kesir + alt birimi olan para: `elli kuruş`
    if minor and len(f) <= 2:
        cents = int(f.ljust(2, "0"))
        if cents == 0:
            return f"{whole} {major}"
        return f"{whole} {major} {int2tr(cents)} {minor}"
    return f"{dec2tr(_ungroup(w), f)} {major}"


def _currency(m):
    return _money(m.group(1), m.group(2))


def _currency_pre(m):
    return _money(m.group(2), m.group(1))


def _speed(m):
    """`120 km/sa` -> `saatte yüz yirmi kilometre` (Türkçe sayıyı sona alır)."""
    body, unit, per = m.group(1), m.group(2), m.group(3)
    per_word = PER_UNITS.get(per) or PER_UNITS.get(per.lower())
    unit_word = UNITS.get(unit, UNITS.get(unit.lower(), unit))
    if per_word is None:
        return None
    if "," in body:
        w, f = body.split(",")
        num = dec2tr(_ungroup(w), f)
    else:
        num = int2tr(_ungroup(body))
    return f"{per_word} {num} {unit_word}"


def _measure(m):
    body, unit = m.group(1), m.group(2)
    word = UNITS.get(unit, UNITS.get(unit.lower(), unit))
    if "," in body:
        w, f = body.split(",")
        return dec2tr(_ungroup(w), f) + " " + word
    return int2tr(_ungroup(body)) + " " + word


def _range(m):
    return f"{int2tr(int(m.group(1)))} ila {int2tr(int(m.group(2)))}"


def _score(m):
    return f"{int2tr(int(m.group(1)))} {int2tr(int(m.group(2)))}"


def _ordinal(m):
    return ord2tr(_ungroup(m.group(1)))


def _decimal(m):
    return dec2tr(_ungroup(m.group(1)), m.group(2))


def _grouped(m):
    return int2tr(_ungroup(m.group(1)))


def _plain(m):
    """Yalın tam sayı. Baştaki sıfır söylenir (kod, dahili numara)."""
    d = m.group(1)
    if len(d) > 1 and d[0] == "0":
        return " ".join(int2tr(int(c)) for c in d)
    return int2tr(int(d))


def _ratio(m):
    return f"{int2tr(int(m.group(1)))} bölü {int2tr(int(m.group(2)))}"


def _phone(m):
    """Rakam rakam, yazıldığı gruplar hâlinde."""
    return " ".join(" ".join(int2tr(int(d)) for d in g)
                    for g in m.group(0).split())


# Türkçe ek rakama değil OKUNUŞA bağlanır: `2026'da` -> `iki bin yirmi altıda`.
SUFFIX = r"(?:'(?P<sfx>\w+))?"

RULES = [
    ("phone",     re.compile(r"\b0\d{3}\s\d{3}\s\d{2}\s\d{2}\b"), _phone),
    ("date",      re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b"
                             + SUFFIX), _date),
    ("time",      re.compile(r"\b(\d{1,2}):(\d{2})\b" + SUFFIX), _time),
    ("day_month", re.compile(r"\b(\d{1,2})\.(\d{1,2})\b" + SUFFIX),
     _day_month),
    ("percent",   re.compile(r"%\s?(\d[\d.]*(?:,\d+)?)" + SUFFIX), _percent),
    ("currency",  re.compile(r"\b(\d[\d.]*(?:,\d+)?)\s?(TL|₺|\$|€|£|¥)"
                             + SUFFIX), _currency),
    ("currency_pre", re.compile(r"(₺|\$|€|£|¥)\s?(\d[\d.]*(?:,\d+)?)"
                                + SUFFIX), _currency_pre),
    ("speed",     re.compile(r"\b(\d[\d.]*(?:,\d+)?)\s?"
                             r"(km|m|mm|cm|kg|lt|l|MB|GB)/(sa|h|sn|s|dk)\b"),
     _speed),
    ("measure",   re.compile(r"\b(\d[\d.]*(?:,\d+)?)\s?"
                             r"(km|cm|mm|m²|m³|kg|gr|mg|ml|lt|dk|sn|sa|"
                             r"GB|MB|TB|KB|kW|MW|°C|°|m|g|l|W)\b" + SUFFIX),
     _measure),
    ("range",     re.compile(r"\b(\d+)\s?-\s?(\d+)\s+(?=aras)"), _range),
    ("score",     re.compile(r"\b(\d{1,2})-(\d{1,2})\b(?=\s|$|[,.!?])"),
     _score),
    ("ordinal",   re.compile(r"\b(\d[\d.]*)\.(?=\s|$)" + SUFFIX), _ordinal),
    ("decimal",   re.compile(r"\b(\d[\d.]*),(\d+)\b" + SUFFIX), _decimal),
    ("grouped",   re.compile(r"\b(\d{1,3}(?:\.\d{3})+)" + SUFFIX), _grouped),
    ("ratio",     re.compile(r"\b(\d+)/(\d+)\b" + SUFFIX), _ratio),
    # sondaki \b yok: ek kesmeden sonra gelir, `1985'ten` sınırı tutmaz
    ("plain",     re.compile(r"\b(\d+)" + SUFFIX), _plain),
]


BACK = set("aıou")
HARD = set("fstkçşhp")

# bulunma ve ayrılma hâlinin dört biçimi; hangisinin doğru olduğu OKUNUŞUN
# sesine bağlı, rakamın sesine değil
_HARMONY = {
    "da": ("da", "de", "ta", "te"), "de": ("da", "de", "ta", "te"),
    "ta": ("da", "de", "ta", "te"), "te": ("da", "de", "ta", "te"),
    "dan": ("dan", "den", "tan", "ten"),
    "den": ("dan", "den", "tan", "ten"),
    "tan": ("dan", "den", "tan", "ten"),
    "ten": ("dan", "den", "tan", "ten"),
}


def _apply_suffix(reading: str, suffix: str | None) -> str:
    """Ek, rakama değil okunuşa eklenir.

    Yazarın seçtiği ek RAKAMIN okunuşuyla uyumludur — `2026'da`, çünkü
    *iki bin yirmi altı* arka ünlüyle biter. Rakam kelimeye dönüşünce sonu
    değişebilir: `09:30` *dokuz buçuk* okunur ve `ta` alır, yazılan `da` değil.
    Bu yüzden hem ünsüz hem ünlü okunuştan yeniden hesaplanır.
    """
    if not suffix:
        return reading
    forms = _HARMONY.get(suffix.lower())
    if not forms:
        return reading + suffix

    tail = reading.rstrip()
    last_vowel = next((c for c in reversed(tail.lower())
                       if c in "aeıioöuü"), "a")
    last_char = tail[-1].lower() if tail else "a"

    back = last_vowel in BACK
    hard = last_char in HARD
    return tail + forms[(0 if back else 1) + (2 if hard else 0)]


def verbalize(text: str, trace: bool = False):
    """Her sayısal ifadeyi sözlü okunuşuna açar."""
    out = _SIGN.sub("eksi ", _split_alnum(text))
    changes: list[tuple[str, str, str]] = []

    for name, rx, fn in RULES:
        def sub(m: re.Match) -> str:
            try:
                reading = fn(m)
            except Exception:
                return m.group(0)
            if reading is None:
                return m.group(0)
            full = _apply_suffix(reading, m.groupdict().get("sfx"))
            changes.append((name, m.group(0), full))
            return full
        out = rx.sub(sub, out)

    for sym, word in SYMBOLS.items():
        if sym in out:
            out = out.replace(sym, word)

    out = _WS.sub(" ", out).strip()
    return (out, changes) if trace else out


def has_digits(t: str) -> bool:
    return bool(re.search(r"\d", t))


def prepare(text: str, add_final: bool = True) -> str:
    """Modele verilecek nihai biçim: sözelleştir, sonra sadeleştir.

    Sıra önemli: sözelleştirme kesme işaretine ve noktalara bakar, bu yüzden
    boşluk/noktalama düzenlemesinden ÖNCE koşar.
    """
    return normalize(verbalize(text), add_final=add_final)
