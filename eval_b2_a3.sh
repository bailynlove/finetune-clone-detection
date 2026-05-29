#!/bin/bash
# Evaluate A3 (P0+P1 combined) on XLCoST 6-pair + CodeNet test_sd + test_dd
set -e
PYTHON=/data1/env/envs/graphenhanced/bin/python3
BASE=/data1/models/Qwen3.5-4B
CKPT=runs/b2_a3_combined_20k
OUT=results/ablations
XLPAIRS="xlcost_java_python xlcost_java_cpp xlcost_java_csharp xlcost_cpp_python xlcost_cpp_csharp xlcost_python_csharp"

mkdir -p $OUT

echo "=== A3 eval on GPU 6 ==="
export CUDA_VISIBLE_DEVICES=6

for P in $XLPAIRS; do
    $PYTHON finetune/eval.py \
        --mode b2 --checkpoint $CKPT --base_model $BASE \
        --test_files dataset/finetune_data/b2/${P}/test.jsonl \
        --output $OUT/b2_a3_${P}.json --batch_size 4
done

for SPLIT in test_sd test_dd; do
    $PYTHON finetune/eval.py \
        --mode b2 --checkpoint $CKPT --base_model $BASE \
        --test_files dataset/finetune_data_v2/b2/${SPLIT}/python_java.jsonl \
        --output $OUT/b2_a3_codenet_${SPLIT}.json --batch_size 4
done

echo ""
echo "=== A3 Results ==="
SUM=0; N=0
for P in $XLPAIRS; do
    F=$OUT/b2_a3_${P}.json
    $PYTHON -c "import json; d=json.load(open('$F')); print(f'  $P: F1={d[\"f1\"]:.3f}  P={d[\"precision\"]:.3f}  R={d[\"recall\"]:.3f}')"
    V=$($PYTHON -c "import json; print(json.load(open('$F'))['f1'])")
    SUM=$($PYTHON -c "print($SUM + $V)"); N=$((N+1))
done
$PYTHON -c "print(f'  XLCoST avg F1: {round($SUM/$N, 3)}')"
for SPLIT in test_sd test_dd; do
    F=$OUT/b2_a3_codenet_${SPLIT}.json
    $PYTHON -c "import json; d=json.load(open('$F')); print(f'  CodeNet $SPLIT: F1={d[\"f1\"]:.3f}  P={d[\"precision\"]:.3f}  R={d[\"recall\"]:.3f}')"
done

echo "[DONE]"
