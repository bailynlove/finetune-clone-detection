#!/bin/bash
# Evaluate A1 (image-dropout) and A2 (image-augment) ablation checkpoints
# on XLCoST 6-pair average and CodeNet test_sd.
# Run after both training jobs finish.
set -e
PYTHON=/data1/env/envs/graphenhanced/bin/python3
BASE=/data1/models/Qwen3.5-4B
OUT=results/ablations
XLPAIRS="xlcost_java_python xlcost_java_cpp xlcost_java_csharp xlcost_cpp_python xlcost_cpp_csharp xlcost_python_csharp"

mkdir -p $OUT

for LABEL in a1_dropout a2_augment; do
    CKPT=runs/b2_${LABEL}_5k
    GPU=$( [[ "$LABEL" == "a1_dropout" ]] && echo 6 || echo 7 )

    echo "=== B2 $LABEL (GPU $GPU) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/eval.py \
        --mode b2 --checkpoint $CKPT --base_model $BASE \
        --test_files dataset/finetune_data_v2/b2/test_sd/python_java.jsonl \
        --output $OUT/b2_${LABEL}_codenet_test_sd.json --batch_size 4 &

    for P in $XLPAIRS; do
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON finetune/eval.py \
            --mode b2 --checkpoint $CKPT --base_model $BASE \
            --test_files dataset/finetune_data/b2/${P}/test.jsonl \
            --output $OUT/b2_${LABEL}_${P}.json --batch_size 4
    done
    wait
done

echo ""
echo "=== Results ==="
for LABEL in a1_dropout a2_augment; do
    echo "--- B2 $LABEL ---"
    # XLCoST average
    SUM=0; N=0
    for P in $XLPAIRS; do
        F=$OUT/b2_${LABEL}_${P}.json
        [ -f "$F" ] || continue
        V=$($PYTHON -c "import json; print(json.load(open('$F'))['f1'])")
        echo "  $P: $V"
        SUM=$($PYTHON -c "print($SUM + $V)")
        N=$((N+1))
    done
    AVG=$($PYTHON -c "print(round($SUM/$N,3))" 2>/dev/null || echo "?")
    echo "  XLCoST avg: $AVG"
    # CodeNet
    F=$OUT/b2_${LABEL}_codenet_test_sd.json
    [ -f "$F" ] && $PYTHON -c "import json; d=json.load(open('$F')); print(f'  CodeNet test_sd: F1={d[\"f1\"]:.3f}')"
done

echo "[DONE]"
