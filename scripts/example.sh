# Environment Setup
# conda env create -f env.yml
# Evals: mmlu, winogrande, arc_challenge, arc_easy, hellaswag, openbookqa, piqa

# Install lighteval
# pip install lighteval==0.10.0

# Generate Low-Rank Weights (Offline)
python -u utils/prepare_low_rank_weight.py \
    --model_name unsloth/Llama-3.2-1B \
    --output_dir ../low_rank_models/llama-3.2-1b

# Task definitions for lighteval 0.10.0
# Format: suite|task:subset|num_fewshot|truncate_fewshots
# Tasks using acc_norm (length-normalized accuracy): hellaswag, arc:challenge, arc:easy, openbookqa, piqa, winogrande
# MMLU uses standard accuracy
TASKS="leaderboard|mmlu|5|0,leaderboard|winogrande|5|0,leaderboard|arc:challenge|25|0,lighteval|arc:easy|25|0,leaderboard|hellaswag|10|0,lighteval|openbookqa|0|0,lighteval|piqa|0|0"

GPU=0
model=unsloth/Llama-3.2-1B

# Baseline: Full evaluation
CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
    --model_name ${model} \
    --method full \
    --tasks "${TASKS}" \
    --output_dir "./evals/full"

# # Baseline: Relufiction
# CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
#     --model_name ${model} \
#     --method relufiction \
#     --tasks "${TASKS}" \
#     --output_dir "./evals/relufiction"

# R-Sparse
CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation_lighteval.py \
    --model_name ${model} \
    --method r_sparse \
    --sparse_config_file config/llama3_sparsity_50_evolutionary_search.npy \
    --config_file config/llama-3.2-1b_default.json \
    --tasks "${TASKS}" \
    --output_dir "./evals/r_sparse"
