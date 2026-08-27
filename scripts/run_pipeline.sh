#!/usr/bin/env bash
# Veri hattını uçtan uca koşar: indirilmiş veri kümesi -> eğitime hazır paket.
#
#   bash scripts/run_pipeline.sh /veri/turkish-tts
#   bash scripts/run_pipeline.sh /veri/turkish-tts --pilot 100
#
# Her adım sürdürülebilir; yarıda kesilirse aynı komut kaldığı yerden devam
# eder. Kaynak veri kümesine yazılmaz.
set -euo pipefail

DATA="${1:?kullanim: run_pipeline.sh <veri_kumesi_dizini> [--pilot SAAT]}"
shift || true

PILOT_HOURS=""
SPLITS=(train review)
ART=artifacts
PY="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot) PILOT_HOURS="$2"; shift 2 ;;
    --splits) shift; SPLITS=(); while [[ $# -gt 0 && "$1" != --* ]]; do SPLITS+=("$1"); shift; done ;;
    --artifacts) ART="$2"; shift 2 ;;
    *) echo "bilinmeyen secenek: $1" >&2; exit 2 ;;
  esac
done

MANIFEST="$ART/manifest/train_all.jsonl"
CODES="$ART/codes"
PACKED="$ART/packed"

echo "== 1/5  hijyen + bolum birlestirme (${SPLITS[*]})"
if [[ ! -f "$MANIFEST" ]]; then
  $PY scripts/prepare_manifest.py --data "$DATA" --splits "${SPLITS[@]}" \
      --out "$MANIFEST" --report "$ART/manifest/train_all_report.json"
else
  echo "   $MANIFEST var, atlandi"
fi

TRAIN_MANIFEST="$MANIFEST"
if [[ -n "$PILOT_HOURS" ]]; then
  TRAIN_MANIFEST="$ART/manifest/pilot_${PILOT_HOURS}h.jsonl"
  PACKED="$ART/packed_pilot"
  echo "== 1b/5  ${PILOT_HOURS} saatlik temiz + prozodi-tabakali alt kume"
  if [[ ! -f "$TRAIN_MANIFEST" ]]; then
    $PY scripts/make_subset.py --manifest "$MANIFEST" \
        --hours "$PILOT_HOURS" --out "$TRAIN_MANIFEST"
  else
    echo "   $TRAIN_MANIFEST var, atlandi"
  fi
fi

echo "== 2/5  metin tokenizer'i (4k BPE)"
if [[ ! -f "$ART/tokenizer/tokenizer.json" ]]; then
  # BPE her zaman TAM manifestten egitilir; pilot alt kumesi sozlugu daraltmasin
  $PY scripts/train_tokenizer.py --manifest "$MANIFEST" --out "$ART/tokenizer"
else
  echo "   $ART/tokenizer var, atlandi"
fi

echo "== 3/5  ses -> SNAC kodlari"
$PY scripts/ingest_audio.py --data "$DATA" --splits "${SPLITS[@]}" \
    --manifest "$TRAIN_MANIFEST" --out "$CODES"

echo "== 4/5  paketlenmis egitim kumesi"
$PY scripts/build_dataset.py --manifest "$TRAIN_MANIFEST" \
    --codes "$CODES" --tokenizer "$ART/tokenizer" --out "$PACKED"

echo "== 5/5  degerlendirme kumesi"
if [[ ! -f "$ART/eval_set.jsonl" ]]; then
  $PY scripts/make_eval_set.py --manifest "$MANIFEST" \
      --exclude "$TRAIN_MANIFEST" --out "$ART/eval_set.jsonl"
else
  echo "   $ART/eval_set.jsonl var, atlandi"
fi

cat <<EOF

hazir. egitim:
  python -m src.train --config configs/$( [[ -n "$PILOT_HOURS" ]] && echo train_pilot.yaml || echo train.yaml )

paketlenmis veri: $PACKED
EOF
