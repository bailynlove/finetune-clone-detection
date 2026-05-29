#!/bin/bash
# GPU 5: Coder-B1 family (ZS / old-v2 / v2real) on CodeNet + XLCoST
set -e
export CUDA_VISIBLE_DEVICES=5
cd /data1/clone-test

PYTHON=/data1/env/envs/graphenhanced/bin/python3
BASE=/data1/models/Qwen2.5-Coder-3B-Instruct
OLD=runs/v2_qwen25coder_b1_train_a_seed42
NEW=runs/v2real_qwen25coder_b1_train_a_seed42
OUT=results/v2real
XLPAIRS="xlcost_java_python xlcost_java_cpp xlcost_java_csharp xlcost_cpp_python xlcost_cpp_csharp xlcost_python_csharp"

mkdir -p $OUT

echo "=== Coder ZS B1 ==="
for SPLIT in test_sd test_dd; do
    $PYTHON finetune/eval.py --mode b1 --no_lora --base_model $BASE \
        --test_files dataset/finetune_data_v2/b1/${SPLIT}/python_java.jsonl \
        --output $OUT/coder_zs_b1_codenet_${SPLIT}.json --batch_size 4
done
for P in $XLPAIRS; do
    $PYTHON finetune/eval.py --mode b1 --no_lora --base_model $BASE \
        --test_files dataset/finetune_data/b1/${P}/test.jsonl \
        --output $OUT/coder_zs_b1_${P}.json --batch_size 4
done

echo "=== Coder B1 old (v2_qwen25coder) ==="
for SPLIT in test_sd test_dd; do
    $PYTHON finetune/eval.py --mode b1 --checkpoint $OLD --base_model $BASE \
        --test_files dataset/finetune_data_v2/b1/${SPLIT}/python_java.jsonl \
        --output $OUT/coder_b1old_codenet_${SPLIT}.json --batch_size 4
done
for P in $XLPAIRS; do
    $PYTHON finetune/eval.py --mode b1 --checkpoint $OLD --base_model $BASE \
        --test_files dataset/finetune_data/b1/${P}/test.jsonl \
        --output $OUT/coder_b1old_${P}.json --batch_size 4
done

echo "=== Coder B1 v2real ==="
for SPLIT in test_sd test_dd; do
    $PYTHON finetune/eval.py --mode b1 --checkpoint $NEW --base_model $BASE \
        --test_files dataset/finetune_data_v2/b1/${SPLIT}/python_java.jsonl \
        --output $OUT/coder_b1v2real_codenet_${SPLIT}.json --batch_size 4
done
for P in $XLPAIRS; do
    $PYTHON finetune/eval.py --mode b1 --checkpoint $NEW --base_model $BASE \
        --test_files dataset/finetune_data/b1/${P}/test.jsonl \
        --output $OUT/coder_b1v2real_${P}.json --batch_size 4
done

echo "[DONE] Coder-B1 evaluations → $OUT"
