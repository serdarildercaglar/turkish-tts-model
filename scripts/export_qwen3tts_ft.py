"""Manifest → Qwen3-TTS ince ayar JSONL'i.

Qwen3-TTS-Finetuning hattı satır başına üç alan ister:

    {"audio": <hedef.wav/flac>, "text": <transkript>, "ref_audio": <referans>}

Referans, klonlama koşullamasının ta kendisi: her satıra FARKLI bir referans
vermek çok-konuşmacılı klon ince ayarı yapar (tek-ses istenirse resmî tarifin
dediği gibi tüm satırlara aynı referans verilir — bkz. --fixed-ref).

Referans seçimi: manifestteki `ref_id` varsa ve `--max-ref-dur`'a sığıyorsa o;
yoksa AYNI KONUŞMACIDAN, FARKLI KAYITTAN, süre sınırına uyan rastgele bir eş
(train'in %65'i ref'siz ve mevcut ref'lerin çoğu 15 sn'den uzun — bu yedek
olmadan 2.000 saat toplanamıyor; pair_clone_refs'in kısıtlarıyla tutarlı).

    python scripts/export_qwen3tts_ft.py \
        --manifest artifacts/manifest/train_only.jsonl \
                   artifacts/manifest/review_paired.jsonl \
        --out artifacts/qwen3tts_ft/train.jsonl --max-hours 2000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=2000.0)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--max-ref-dur", type=float, default=15.0,
                    help="daha uzun referansı olan satır atılır")
    ap.add_argument("--fixed-ref", type=Path,
                    help="tek-ses ince ayarı: tüm satırlara bu referans")
    ap.add_argument("--drop-frag", action="store_true",
                    help="cümle ortasından başlayan/biten klipleri atla")
    ap.add_argument("--no-check-files", action="store_true",
                    help="yol var mı bakma (ses bulutta/Hub'da olacaksa)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for m in args.manifest:
        with m.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows[r["id"]] = r
    print(f"manifest: {len(rows):,} satir", file=sys.stderr)

    rng = random.Random(args.seed)

    def src_of(r: dict) -> str:
        # harici kumeler kayit anahtarini `src` alaninda tasir; bizim
        # manifest'te kimlik bicimi source-<hash>-<bas_ms>-<bit_ms>
        return r.get("src") or r["id"].rsplit("-", 2)[0]

    # Yedek referans havuzu: konusmaci -> sure sinirina uyan klipler.
    pool: dict[str, list[dict]] = {}
    if not args.fixed_ref:
        for r in rows.values():
            if float(r.get("duration") or 0) <= args.max_ref_dur:
                spk = r.get("speaker_id")
                if spk not in (None, "", "None"):
                    pool.setdefault(spk, []).append(r)

    def pick_ref(r: dict) -> dict | None:
        """Manifest ref'i uygunsa o; degilse ayni konusmaci + farkli kayit.

        `review-*` konusmacilar kanal VEKILIDIR (pair_clone_refs): ayni kanalda
        farkli gercek sesler olabilir, yedege dusmek klon egitimini kirletir —
        onlarda yalniz gomme-eslesmeli manifest ref'i kabul edilir.
        """
        ref = rows.get(r.get("ref_id") or "")
        if ref is not None and float(ref.get("duration") or 0) <= args.max_ref_dur:
            return ref
        if str(r.get("speaker_id") or "").startswith(("review-", "emb-")):
            return None
        cands = pool.get(r.get("speaker_id"), [])
        for _ in range(8):
            if not cands:
                break
            c = rng.choice(cands)
            if c["id"] != r["id"] and src_of(c) != src_of(r):
                return c
        return None

    cand = []
    drop = dict(dur=0, frag=0, ref_yok=0, dosya=0)
    for r in rows.values():
        d = float(r.get("duration") or 0)
        if not (args.min_dur <= d <= args.max_dur):
            drop["dur"] += 1
            continue
        if args.drop_frag and (r.get("frag_start") or r.get("frag_end")):
            drop["frag"] += 1
            continue
        if args.fixed_ref:
            ref_path, ref_id = str(args.fixed_ref), None
        else:
            ref = pick_ref(r)
            if ref is None:
                drop["ref_yok"] += 1
                continue
            ref_path, ref_id = ref["audio"], ref["id"]
        if not args.no_check_files and not Path(r["audio"]).exists():
            drop["dosya"] += 1
            continue
        # id/ref_id, materialize_clips.py'nin parquet eslemesi icin tasinir;
        # Qwen'in prepare_data.py'sine giden son dosyada yalniz uc alan kalir.
        cand.append((d, {"audio": r["audio"], "text": r["text"],
                         "ref_audio": ref_path, "id": r["id"],
                         "ref_id": ref_id}))

    # Saat sınırı: kaynak çeşitliliği bozulmasın diye rastgele örneklenir.
    random.Random(args.seed).shuffle(cand)
    total, kept = 0.0, []
    for d, row in cand:
        if total + d > args.max_hours * 3600:
            continue
        total += d
        kept.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rep = dict(satir=len(kept), saat=round(total / 3600, 1),
               aday=len(cand), atilan=drop)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
