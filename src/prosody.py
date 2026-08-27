"""Prozodi kontrol kovaları: konuşma hızı ve ses seviyesi.

Model mimarisini değiştirmez — kovalar yalnızca istemin başına yazılan
tokenlardır (bkz. `src.vocab.CONTROLS`). Eğitimde klibin ölçülen değerinden,
çıkarımda kullanıcının istediği değerden gelir.

Öznitelikler manifestten bedava gelir, ses okumak gerekmez:
  - hız   = metin uzunluğu / süre  (harf/sn)
  - seviye = `quality_lufs`

Kova sınırları korpustan nicelik dilimleriyle hesaplanır ve `prosody.json`
olarak paketlenmiş veriyle birlikte yazılır; çıkarım aynı dosyayı okur. Sınırlar
modelin sözleşmesinin parçasıdır, checkpoint ile birlikte taşınmalıdır.

Kalite (DNSMOS) kovası bilerek YOK: korpusta aralık 2,94–3,58 ile çok dar,
öğrenilecek sinyal taşımıyor. Perde (F0) kovası da yok; ses okumayı gerektirir,
manifestte hazır değil — ileride eklenecekse `EXTRA` alanı ayrılmıştır.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.vocab import LOUD_BUCKETS, RATE_BUCKETS

FILENAME = "prosody.json"

# Korpus ölçülmeden kullanılacak yedek sınırlar (hf_train, ham metin).
# `build_dataset.py` gerçek veriden yeniden hesaplar ve üzerine yazar.
DEFAULT = {
    "rate_edges": [12.216, 13.136, 13.882, 14.735],
    "loud_edges": [-24.411, -20.800],
}


def compute_edges(values, n_buckets: int) -> list[float]:
    """Eşit dolulukta kova sınırları (n_buckets-1 adet iç sınır)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        raise ValueError("kova sınırı için değer yok")
    qs = [100.0 * i / n_buckets for i in range(1, n_buckets)]
    return [round(float(x), 4) for x in np.percentile(v, qs)]


def bucket_of(value: float | None, edges: list[float]) -> int | None:
    """Değerin kova indisi; değer yoksa None (→ `<|..._any|>`)."""
    if value is None:
        return None
    return int(np.searchsorted(np.asarray(edges), float(value), side="right"))


class Prosody:
    """Kova sınırlarını taşır; hız/seviye → kova indisi çevirir."""

    def __init__(self, rate_edges=None, loud_edges=None):
        self.rate_edges = list(rate_edges or DEFAULT["rate_edges"])
        self.loud_edges = list(loud_edges or DEFAULT["loud_edges"])
        assert len(self.rate_edges) == RATE_BUCKETS - 1
        assert len(self.loud_edges) == LOUD_BUCKETS - 1

    # ------------------------------------------------------------ oznitelik
    @staticmethod
    def rate_of(text: str, duration: float) -> float | None:
        """Konuşma hızı (harf/sn)."""
        if not text or not duration or duration <= 0:
            return None
        return len(text) / float(duration)

    def rate_bucket(self, text: str, duration: float) -> int | None:
        return bucket_of(self.rate_of(text, duration), self.rate_edges)

    def loud_bucket(self, lufs) -> int | None:
        if lufs in (None, "", "None"):
            return None
        try:
            return bucket_of(float(lufs), self.loud_edges)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------- kalicilik
    def to_dict(self) -> dict:
        return {"rate_edges": self.rate_edges, "loud_edges": self.loud_edges,
                "rate_buckets": RATE_BUCKETS, "loud_buckets": LOUD_BUCKETS}

    def save(self, directory: str | Path) -> Path:
        p = Path(directory) / FILENAME
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, directory: str | Path) -> "Prosody":
        """`prosody.json` varsa oradan, yoksa yedek sınırlarla."""
        p = Path(directory) / FILENAME
        if not p.is_file():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(d.get("rate_edges"), d.get("loud_edges"))

    @classmethod
    def from_rows(cls, rows) -> "Prosody":
        """Manifest satırlarından sınırları hesaplar."""
        rates, louds = [], []
        for r in rows:
            v = cls.rate_of(r.get("text", ""), r.get("duration"))
            if v is not None:
                rates.append(v)
            lu = r.get("quality_lufs")
            if lu not in (None, "", "None"):
                try:
                    louds.append(float(lu))
                except (TypeError, ValueError):
                    pass
        return cls(compute_edges(rates, RATE_BUCKETS),
                   compute_edges(louds, LOUD_BUCKETS))
