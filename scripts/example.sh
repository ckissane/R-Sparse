# Environment Setup
# conda env create -f env.yml
# Evals: mmlu, winogrande, arc_challenge, arc_easy, hellaswag, openbookqa, piqa

# Install lighteval
# pip install lighteval[extended_tasks]==0.12.0
# ==0.10.0 # -E extended_tasks

# Generate Low-Rank Weights (Offline)
 python -u utils/prepare_low_rank_weight.py \
     --model_name unsloth/Llama-3.2-1B \
     --output_dir ../low_rank_models/llama-3.2-1b
#  python -u utils/prepare_low_rank_weight.py \
#      --model_name unsloth/Llama-3.2-3B \
#      --output_dir ../low_rank_models/llama-3.2-3b

# Task definitions for lighteval 0.12.0
# Format: suite|task:subset|num_fewshot
# Tasks using acc_norm (length-normalized accuracy): hellaswag, arc:challenge, arc:easy, openbookqa, piqa, winogrande
# MMLU uses standard accuracy
# TASKS="leaderboard|mmlu|5|0,leaderboard|winogrande|5|0,leaderboard|arc:challenge|25|0,lighteval|arc:easy|25|0,leaderboard|hellaswag|10|0,lighteval|openbookqa|0|0,lighteval|piqa|0|0"
# TASKS="leaderboard|mmlu|0|0,leaderboard|winogrande|0|0,leaderboard|arc:challenge|0|0,lighteval|arc:easy|0|0,leaderboard|hellaswag|0|0,lighteval|openbookqa|0|0,lighteval|piqa|0|0"

TASKS="leaderboard|hellaswag|0|0"
GPU=0
model=unsloth/Llama-3.2-1B

# Baseline: Full evaluation
CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
    --model_name ${model} \
    --method full \
    --tasks "${TASKS}" \
    --output_dir "./evals_1b/full"

# # Baseline: Relufiction
# CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
#     --model_name ${model} \
#     --method relufiction \
#     --tasks "${TASKS}" \
#     --output_dir "./evals/relufiction"

# R-Sparse
# CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
#     --model_name ${model} \
#     --method r_sparse \
#     --config_file config/llama-3.2-1b_default.json \
#     --tasks "${TASKS}" \
#     --output_dir "$./evals/{model}_{0.5}" \
#     --target_sparsity=0.5

CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
    --model_name ${model} \
    --method r_sparse \
    --config_file config/llama-3.2-1b_default.json \
    --tasks "${TASKS}" \
    --output_dir "./evals_1b/r_sparse_3" \
    --target_sparsity=0.66666666666
#CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
    # --model_name ${model} \
    # --method r_sparse \
    # --config_file config/llama-3.2-1b_default.json \
    # --tasks "${TASKS}" \
    # --output_dir "./evals/r_sparse_4" \
    # --target_sparsity=0.75
#CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
    # --model_name ${model} \
    # --method r_sparse \
    # --config_file config/llama-3.2-1b_default.json \
    # --tasks "${TASKS}" \
    # --output_dir "./evals/r_sparse_5" \
    # --target_sparsity=0.8
CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
    --model_name ${model} \
    --method r_sparse \
    --config_file config/llama-3.2-1b_default.json \
    --tasks "${TASKS}" \
    --output_dir "./evals_1b/r_sparse_6" \
    --target_sparsity=0.83333333

    # --sparse_config_file config/llama3_sparsity_50_evolutionary_search.npy \
