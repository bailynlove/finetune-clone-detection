#!/bin/bash
# GPU 6: 3.5-B2 family (ZS / old-v2 / v2real) on CodeNet + XLCoST
set -e
export CUDA_VISIBLE_DEVICES=6
cd /data1/clone-test

PYTHON=/data1/env/envs/graphenhanced/bin/python3
BASE=/data1/models/Qwen3.5-4B
OLD=runs/v2_b2_train_a_seed42
NEW=runs/v2real_b2_train_a_seed42
OUT=results/v2real
XLPAIRS="xlcost_java_python xlcost_java_cpp xlcost_java_csharp xlcost_cpp_python xlcost_cpp_csharp xlcost_python_csharp"

mkdir -p $OUT

echo "=== 3.5 ZS B2 ==="
for SPLIT in test_sd test_dd; do
    $PYTHON finetune/eval.py --mode b2 --no_lora --base_model $BASE \
        --test_files dataset/finetune_data_v2/b2/${SPLIT}/python_java.jsonl \
        --output $OUT/qw35_zs_b2_codenet_${SPLIT}.json --batch_size 4
done
for P in $XLPAIRS; do
    $PYTHON finetune/eval.py --mode b2 --no_lora --base_model $BASE \
        --test_files dataset/finetune_data/b2/${P}/test.jsonl \
        --output $OUT/qw35_zs_b2_${P}.json --batch_size 4
done

echo "=== 3.5 B2 old (v2_b2 — trained on v1 data) ==="
for SPLIT in test_sd test_dd; do
    $PYTHON finetune/eval.py --mode b2 --checkpoint $OLD --base_model $BASE \
        --test_files dataset/finetune_data_v2/b2/${SPLIT}/python_java.jsonl \
        --output $OUT/qw35_b2old_codenet_${SPLIT}.json --batch_size 4
done
for P in $XLPAIRS; do
    $PYTHON finetune/eval.py --mode b2 --checkpoint $OLD --base_model $BASE \
        --test_files dataset/finetune_data/b2/${P}/test.jsonl \
        --output $OUT/qw35_b2old_${P}.json --batch_size 4
done

echo "=== 3.5 B2 v2real ==="
for SPLIT in test_sd test_dd; do
    $PYTHON finetune/eval.py --mode b2 --checkpoint $NEW --base_model $BASE \
        --test_files dataset/finetune_data_v2/b2/${SPLIT}/python_java.jsonl \
        --output $OUT/qw35_b2v2real_codenet_${SPLIT}.json --batch_size 4
done
for P in $XLPAIRS; do
    $PYTHON finetune/eval.py --mode b2 --checkpoint $NEW --base_model $BASE \
        --test_files dataset/finetune_data/b2/${P}/test.jsonl \
        --output $OUT/qw35_b2v2real_${P}.json --batch_size 4
done

echo "[DONE] 3.5-B2 evaluations → $OUT"
