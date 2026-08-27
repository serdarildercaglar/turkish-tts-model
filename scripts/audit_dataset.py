"""Veri kümesi denetimi — SALT OKUNUR. Manifesti değiştirmez, klipleri açmaz.

Modele giren metnin ve manifest kalite alanlarının hatalarını sayar; her bulgu
için sayı, oran ve örnek verir. Amaç, düzeltme hattını yazmadan önce nelerin
düzeltilmesi gerektiğini ölçmek (garbage in / garbage out).

    python scripts/audit_dataset.py \
        --manifest /path/hf_train.jsonl \
        --val /path/hf_validation.jsonl \
        --out artifacts/audit

Çıktı:
    <out>/audit_<split>.json   her bulgunun sayısı/oranı + örnek kimlikleri
    <out>/summary.md           insan okur özet

Bulgular üç ağırlıkta işaretlenir:
    BLOCK  modele bu haliyle girmemeli
    FIX    düzeltme hattında ele alınmalı
    INFO   bilgi; karar gerektirir ama hata olmayabilir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TR_LOWER = "abcçdefgğhıijklmnoöprsştuüvyzâîû"
TR_UPPER = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZÂÎÛ"
TR_LETTER = TR_LOWER + TR_UPPER
# Türkçe olmayan ama metinde geçebilen latin harfleri
FOREIGN = "qwxQWX"
ALLOWED_PUNCT = " .,!?:;-'\"()%&/…"

_WORD = re.compile(rf"[{TR_LETTER}{FOREIGN}0-9]+")
_SUFFIX_APOS = re.compile(rf"[{TR_LETTER}]'[{TR_LOWER}]")
_REPEAT_RUN = re.compile(r"(\b\w+\b)(\s+\1\b){2,}", re.IGNORECASE)


def sev(tag: str) -> str:
    return tag


class Audit:
    """Bulguları sayar ve her biri için ilk N örneği saklar."""

    def __init__(self, keep: int = 5):
        self.count: Counter = Counter()
        self.examples: dict[str, list] = defaultdict(list)
        self.severity: dict[str, str] = {}
        self.keep = keep
        self.n = 0

    def hit(self, key: str, severity: str, cid: str, detail: str = "") -> None:
        self.count[key] += 1
        self.severity[key] = severity
        if len(self.examples[key]) < self.keep:
            self.examples[key].append({"id": cid, "detail": detail[:160]})

    def report(self) -> dict:
        rows = []
        order = {"BLOCK": 0, "FIX": 1, "INFO": 2}
        for k, v in sorted(self.count.items(),
                           key=lambda kv: (order[self.severity[kv[0]]], -kv[1])):
            rows.append({
                "bulgu": k,
                "agirlik": self.severity[k],
                "sayi": v,
                "oran_yuzde": round(v / self.n * 100, 3),
                "ornekler": self.examples[k],
            })
        return {"satir": self.n, "bulgular": rows}


def audit_rows(rows: list[dict], a: Audit, known_ids: set[str]) -> dict:
    """Satır başına metin ve kalite kontrolleri."""
    text_seen: dict[str, str] = {}
    stats = defaultdict(list)
    charset: Counter = Counter()

    for r in rows:
        a.n += 1
        cid = r["id"]
        t = r.get("text") or ""
        dur = float(r["duration"])
        stats["duration"].append(dur)
        charset.update(t)

        # ---------------------------------------------------------- BLOCK
        if not t.strip():
            a.hit("metin bos", "BLOCK", cid)
            continue
        if len(_WORD.findall(t)) < 2:
            a.hit("iki kelimeden az", "BLOCK", cid, t)
        if dur <= 0:
            a.hit("sure sifir/negatif", "BLOCK", cid, str(dur))
        if t in text_seen:
            a.hit("birebir ayni metin (tekrar)", "BLOCK", cid,
                  f"{text_seen[t]} ile ayni: {t[:80]}")
        else:
            text_seen[t] = cid
        if _REPEAT_RUN.search(t):
            a.hit("ASR dongusu (kelime 3+ tekrar)", "BLOCK", cid,
                  _REPEAT_RUN.search(t).group(0))

        cer = r.get("quality_asr_cer")
        if cer is not None and float(cer) > 0.15:
            a.hit("ASR CER > 0.15 (hizalama supheli)", "BLOCK", cid, str(cer))

        # ------------------------------------------------------------ FIX
        if "''" in t:
            a.hit("'' tirnak (Whisper konvansiyonu)", "FIX", cid, t[:100])
        if '"' in t:
            a.hit('" duz tirnak', "FIX", cid, t[:100])
        if "…" in t:
            a.hit("… tek karakter uc nokta", "FIX", cid, t[:100])
        if re.search(r"[“”‘’«»„]", t):
            a.hit("kivrik/aciklamali tirnak", "FIX", cid, t[:100])
        if re.search(r"\d", t):
            a.hit("rakam (sesli formla uyusmuyor)", "FIX", cid, t[:100])
        if t != t.strip():
            a.hit("bas/son bosluk", "FIX", cid, repr(t[:20]))
        if re.search(r"\s{2,}", t):
            a.hit("cift bosluk", "FIX", cid, t[:80])
        if re.search(r"[̀-ͯ]", unicodedata.normalize("NFD", t)):
            nfc = unicodedata.normalize("NFC", t)
            if unicodedata.normalize("NFD", nfc) != unicodedata.normalize("NFD", t):
                a.hit("unicode normalizasyon farki", "FIX", cid, t[:80])
        bad = {c for c in t if c not in TR_LETTER + FOREIGN + ALLOWED_PUNCT
               and not c.isdigit()}
        if bad:
            a.hit("sozluk disi karakter", "FIX", cid,
                  f"{sorted(bad)} :: {t[:80]}")

        # parca klipler
        first = t.strip()[0]
        if first.isalpha() and first not in TR_UPPER + FOREIGN.upper():
            a.hit("kucuk harfle basliyor (parca)", "FIX", cid, t[:80])
        if t.strip()[-1] not in ".!?\"'":
            a.hit("noktalamasiz bitiyor (parca)", "FIX", cid, t[-80:])

        # ----------------------------------------------------------- INFO
        if _SUFFIX_APOS.search(t) and "''" in t:
            a.hit("ek kesmesi + tirnak ayni satirda", "INFO", cid, t[:100])
        if re.search(rf"[{FOREIGN}]", t):
            a.hit("q/w/x iceriyor (yabanci sozcuk)", "INFO", cid, t[:80])
        letters = [c for c in t if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
            a.hit("cogunlukla BUYUK HARF", "INFO", cid, t[:80])

        ref = r.get("ref_id")
        if ref and ref not in ("None", ""):
            if ref not in known_ids:
                a.hit("ref_id manifestte yok", "BLOCK", cid, str(ref))
            elif ref == cid:
                a.hit("ref_id kendisini gosteriyor", "BLOCK", cid, str(ref))

        # kalite alanlari
        for key, lo, hi, tag in (
            ("quality_clip_ratio", None, 0.01, "kirpma orani > %1"),
            ("quality_speech_ratio", 0.5, None, "konusma orani < 0.5"),
            ("quality_music_score", None, 0.5, "muzik skoru > 0.5"),
            ("quality_dnsmos_ovrl", 2.5, None, "DNSMOS < 2.5"),
            ("quality_speaker_cosine", 0.5, None, "konusmaci kosinusu < 0.5"),
        ):
            v = r.get(key)
            if v in (None, "", "None"):
                continue
            v = float(v)
            stats[key].append(v)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                a.hit(tag, "FIX", cid, f"{key}={v:.4f}")

        # konusma hizi aykiriliklari (metin/ses uyusmazligi isareti)
        rate = len(t) / dur if dur > 0 else 0
        stats["rate"].append(rate)
        if dur > 0 and (rate < 6 or rate > 25):
            a.hit("harf/sn araligin disinda (metin-ses uyusmazligi)", "BLOCK",
                  cid, f"{rate:.1f} harf/sn, {dur:.1f}s :: {t[:60]}")

    return {"charset": charset, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--val", type=Path)
    ap.add_argument("--out", type=Path, default=Path("artifacts/audit"))
    ap.add_argument("--keep", type=int, default=5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": args.manifest}
    if args.val:
        splits["validation"] = args.val

    loaded = {}
    for name, path in splits.items():
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        loaded[name] = rows
        print(f"{name}: {len(rows):,} satir okundu", file=sys.stderr)

    reports = {}
    for name, rows in loaded.items():
        ids = {r["id"] for r in rows}
        a = Audit(keep=args.keep)
        extra = audit_rows(rows, a, ids)
        rep = a.report()

        st = extra["stats"]
        rep["dagilim"] = {
            k: {
                "ort": round(float(np.mean(v)), 4),
                "p1": round(float(np.percentile(v, 1)), 4),
                "p50": round(float(np.percentile(v, 50)), 4),
                "p99": round(float(np.percentile(v, 99)), 4),
            }
            for k, v in st.items() if v
        }
        rep["karakter_envanteri"] = {
            repr(c): n for c, n in extra["charset"].most_common()
            if not c.isalnum() or c in FOREIGN
        }
        reports[name] = rep
        (args.out / f"audit_{name}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------------------------------------------ bolumler arasi sizinti
    cross = {}
    if len(loaded) == 2:
        tr, va = loaded["train"], loaded["validation"]
        for key in ("source_id", "speaker_id", "channel"):
            s_tr = {r[key] for r in tr}
            s_va = {r[key] for r in va}
            ortak = s_tr & s_va
            cross[key] = {
                "train_benzersiz": len(s_tr),
                "validation_benzersiz": len(s_va),
                "ortak": len(ortak),
                "validation_kapsanan_yuzde": round(
                    sum(1 for r in va if r[key] in ortak) / len(va) * 100, 2),
            }
        t_tr = {r["text"] for r in tr}
        cross["birebir_ayni_metin"] = sum(1 for r in va if r["text"] in t_tr)
        (args.out / "audit_cross.json").write_text(
            json.dumps(cross, ensure_ascii=False, indent=1), encoding="utf-8")

    # --------------------------------------------------------------- ozet
    lines = ["# Veri kümesi denetimi", ""]
    for name, rep in reports.items():
        lines += [f"## {name} — {rep['satir']:,} satır", "",
                  "| ağırlık | bulgu | sayı | oran |", "|---|---|---:|---:|"]
        for b in rep["bulgular"]:
            lines.append(f"| {b['agirlik']} | {b['bulgu']} | {b['sayi']:,} | "
                         f"%{b['oran_yuzde']:.2f} |")
        lines.append("")
    if cross:
        lines += ["## Bölümler arası sızıntı", "",
                  "| alan | train | validation | ortak | val kapsanan |",
                  "|---|---:|---:|---:|---:|"]
        for k, v in cross.items():
            if isinstance(v, dict):
                lines.append(f"| {k} | {v['train_benzersiz']:,} | "
                             f"{v['validation_benzersiz']:,} | {v['ortak']:,} | "
                             f"%{v['validation_kapsanan_yuzde']:.1f} |")
        lines.append("")
        lines.append(f"validation'da train ile birebir aynı metin: "
                     f"{cross['birebir_ayni_metin']:,}")
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
