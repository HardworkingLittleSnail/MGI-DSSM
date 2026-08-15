#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/version2026draft}"
PY="${PYTHON_BIN:-$PROJECT/.venv/bin/python}"
OUT="${OUTPUT_ROOT:-$PROJECT/outputs/ic2ml_paper_method_version3_10seeds_200ep}"
SEEDS=(7 17 27 37 47 57 67 77 87 97)

cd "$PROJECT/Compare-Models/IC2ML"
mkdir -p "$OUT"

for dataset in nasa calce tju; do
    case "$dataset" in
        nasa) seq_len=16; battery=B0005; learning_rate=1e-4; voltage=(--voltage-start 3.6 --voltage-end 3.7) ;;
        calce) seq_len=64; battery=CS2_35; learning_rate=1e-4; voltage=(--voltage-start 3.6 --voltage-end 3.7) ;;
        tju) seq_len=64; battery=CY25-1; learning_rate=1e-4; voltage=(--voltage-start 3.6 --voltage-end 3.7) ;;
    esac
    "$PY" -u run_rul_benchmark.py \
        --dataset "$dataset" --data-version version3 \
        --model-variant direct --use-capacity-history \
        --initialize-history-readout-ridge --test-batteries "$battery" \
        --seeds "${SEEDS[@]}" --seq-len "$seq_len" \
        --epochs 200 --patience 20 --batch-size 128 --hidden-dim 256 \
        --learning-rate "$learning_rate" --capacity-scaling rated \
        --validation-mode chronological --selection-objective capacity_mae \
        --history-loss-weight 1 --trajectory-loss-weight 1 --rul-loss-weight 0.5 \
        --capacity-loss mse "${voltage[@]}" --output-root "$OUT"
done

printf 'IC2ML paper-method runs completed at %s\n' "$(date -Is)" > "$OUT/TRAINING_COMPLETE.txt"
