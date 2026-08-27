"""Veri kaynağı soyutlaması: parquet dizini ya da jsonl manifest.

Kullanıcı veri kümesini `scripts/download_dataset.py` ile indirdiğinde elimizde
`<kok>/data/<split>-XXXXX-of-YYYYY.parquet` bulunur; ses bu dosyaların içinde
gömülüdür, ayrı klip dosyası yoktur. Eski akışta ise yerelde jsonl manifest +
FLAC klipler vardı. Her iki kaynağı da aynı arayüzle okuruz.

Önemli: üstveri okurken ses sütunu ATLANIR. Parquet sütunlu olduğu için bu,
84 GB'lik dosyalardan yalnızca birkaç yüz MB okumak demektir; manifest
hazırlama adımı sesi hiç açmaz.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

AUDIO_COLUMNS = ("audio", "wav", "speech")
ID_COLUMNS = ("id", "clip_id", "utt_id")

_SHARD_RE = re.compile(r"^(?P<split>[a-z_]+)-\d+-of-\d+\.parquet$")


def find_shards(root: str | Path, split: str) -> list[Path]:
    """`<kok>/data/<split>-*.parquet` dosyalarını sıralı döndürür.

    `<kok>` doğrudan parquet'leri içeriyorsa (data/ alt dizini yoksa) o da
    kabul edilir.
    """
    root = Path(root)
    for base in (root / "data", root):
        if base.is_dir():
            hits = sorted(base.glob(f"{split}-*.parquet"))
            if hits:
                return hits
    return []


def available_splits(root: str | Path) -> list[str]:
    root = Path(root)
    found = set()
    for base in (root / "data", root):
        if not base.is_dir():
            continue
        for p in base.glob("*.parquet"):
            m = _SHARD_RE.match(p.name)
            if m:
                found.add(m.group("split"))
    return sorted(found)


def _pick(names: list[str], candidates) -> str | None:
    return next((c for c in candidates if c in names), None)


def iter_metadata(root: str | Path, split: str, batch_size: int = 4096):
    """Ses sütunu hariç tüm alanları satır sözlükleri olarak üretir."""
    import pyarrow.parquet as pq

    for shard in find_shards(root, split):
        pf = pq.ParquetFile(shard)
        names = [f.name for f in pf.schema_arrow]
        audio = _pick(names, AUDIO_COLUMNS)
        cols = [n for n in names if n != audio]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            for row in batch.to_pylist():
                row["_shard"] = shard.name
                row["_split"] = split
                yield row


def iter_audio(shard: str | Path, keep_ids: set[str] | None = None,
               batch_size: int = 64):
    """Tek shard'dan (id, ses_hucresi) üretir; `keep_ids` verilirse süzer."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shard)
    names = [f.name for f in pf.schema_arrow]
    id_col = _pick(names, ID_COLUMNS)
    audio_col = _pick(names, AUDIO_COLUMNS)
    if id_col is None or audio_col is None:
        raise ValueError(f"{shard}: kimlik/ses sutunu yok; sema {names}")
    for batch in pf.iter_batches(batch_size=batch_size,
                                 columns=[id_col, audio_col]):
        for row in batch.to_pylist():
            cid = row[id_col]
            if keep_ids is not None and cid not in keep_ids:
                continue
            yield cid, row[audio_col]


def iter_jsonl(path: str | Path):
    """Eski akış: jsonl manifest satırları."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_rows(source: str | Path, split: str | None = None):
    """Kaynak bir dizin ise parquet üstverisi, dosya ise jsonl olarak okur."""
    p = Path(source)
    if p.is_dir():
        if split is None:
            raise ValueError("parquet dizini icin --split gerekir")
        yield from iter_metadata(p, split)
    else:
        yield from iter_jsonl(p)
