#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/version2026draft}"
PY="${PYTHON_BIN:-$PROJECT/.venv/bin/python}"
OUTPUT="${OUTPUT_ROOT:-$PROJECT/outputs/autoformer_itransformer_official_version3_10seeds_200ep}"
SEEDS=(7 17 27 37 47 57 67 77 87 97)
DATASETS=(nasa calce tju)

cd "$PROJECT"
mkdir -p "$OUTPUT"

for model in autoformer itransformer; do
    "$PY" -u Compare-Models/run_autoformer_itransformer.py \
        --model "$model" \
        --datasets "${DATASETS[@]}" \
        --seeds "${SEEDS[@]}" \
        --device cuda \
        --max-epochs 200 \
        --data-version version3 \
        --output-root "$OUTPUT"
done

printf 'Completed Autoformer and iTransformer at %s\n' "$(date -Is)" \
    > "$OUTPUT/TRAINING_COMPLETE.txt"
