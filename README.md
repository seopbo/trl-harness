# trl-harness
Complete examples based on [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)

## preliminary
```bash
git clone https://github.com/seopbo/trl-harness.git
```
```bash
conda create trl-harness python=3.13
conda activate trl-harness
pip install --no-cache-dir uv
```

## setup
```bash
# flash-attn
uv pip install --system --no-cache-dir packaging
uv pip install --system --no-cache-dir psutil
uv pip install --system --no-cache-dir ninja

MAX_JOBS=16 uv pip install --system --no-cache-dir flash-attn==2.7.3 --no-build-isolation
```

```bash
# vllm
uv pip install --system --no-cache-dir vllm==0.14.1 --torch-backend=auto
```

```bash
# trl-harness
uv pip install --system --no-cache-dir -r requirements.txt
uv pip install --system --no-cache-dir -r requirements-utils.txt
```

## launch
### preliminary
```bash
# model
mkdir -p checkpoints/base/qwen2.5-1.5b
hf download Qwen/Qwen2.5-1.5B --local-dir checkpoints/base/qwen2.5-1.5b
```

### sft
```bash
bash download_sft-source-datasets.sh && \
    python prepare_sft-dataset.py
```

```bash
# A100 80GB x 4
# global_batch_size=128
WANDB_PROJECT=sft-qwen2.5 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_configs/deepspeed_zero1_for_sft.yaml \
    --main_process_port 8000 \
    run_sft.py \
    --use_wandb_logging \
    --pretrained_model_name_or_path checkpoints/base/qwen2.5-1.5b \
    --chat_template_path assets/qwen3_instruct/chat_template.jinja \
    --data_dirpath datasets/sft \
    --output_dirpath checkpoints/sft/qwen2.5-1.5b \
    --save_strategy epoch \
    --no_save_only_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --weight_decay 0.01 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine
```

### rlvrif
```bash
# download & preprocessing
mkdir -p datasets/source/rl/ai2/if/dolci-rl-zero-if-7b && \
    hf download allenai/Dolci-RL-Zero-IF-7B --repo-type dataset --local-dir datasets/source/rl/ai2/if/dolci-rl-zero-if-7b && \
    python prepare_rlvrif-dataset.py
```

```bash
# serving model on A100x1
CUDA_VISIBLE_DEVICES=3 trl vllm-serve \
    --model checkpoints/sft/qwen2.5-1.5b \
    --data_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --dtype auto \
    --max_model_len 32768 \
    --port 8000
```

```bash
# training model on A100x3
WANDB_PROJECT=rlvrif-qwen2.5 CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file accelerate_configs/deepspeed_zero1_for_rl.yaml \
    --main_process_port 8001 \
    run_rlvrif.py \
    --pretrained_model_name_or_path checkpoints/sft/qwen2.5-1.5b \
    --chat_template_path checkpoints/sft/qwen2.5-1.5b/chat_template.jinja \
    --data_dirpath datasets/rl/if \
    --output_dirpath checkpoints/rl/rlvrif-qwen2.5-1.5b \
    --save_strategy steps \
    --save_total_limit 5 \
    --save_steps 100 \
    --torch_empty_cache_steps 100 \
    --num_train_epochs 2 \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --steps_per_generation 2 \
    --max_completion_length 2048 \
    --weight_decay 0.0 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type constant \
    --use_wandb_logging
```

### rlvrmath
```bash
# download & preprocessing
mkdir -p datasets/source/rl/ai2/math/rlvr-gsm-math-if-mixed-constraints && \
    hf download allenai/RLVR-GSM-MATH-IF-Mixed-Constraints --repo-type dataset --local-dir datasets/source/rl/ai2/math/rlvr-gsm-math-if-mixed-constraints && \
    python prepare_rlvrmath-dataset.py
```

```bash
# serving model on A100x1
CUDA_VISIBLE_DEVICES=3 trl vllm-serve \
    --model checkpoints/sft/qwen2.5-1.5b \
    --data_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --dtype auto \
    --max_model_len 32768 \
    --port 8000
```

```bash
# training model on A100x3
WANDB_PROJECT=rlvrmath-qwen2.5 CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file accelerate_configs/deepspeed_zero1_for_rl.yaml \
    --main_process_port 8001 \
    run_rlvrmath.py \
    --pretrained_model_name_or_path checkpoints/sft/qwen2.5-1.5b \
    --chat_template_path checkpoints/sft/qwen2.5-1.5b/chat_template.jinja \
    --data_dirpath datasets/rl/math \
    --output_dirpath checkpoints/rl/rlvrmath-qwen2.5-1.5b \
    --save_strategy steps \
    --save_total_limit 5 \
    --save_steps 100 \
    --torch_empty_cache_steps 100 \
    --num_train_epochs 2 \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --steps_per_generation 2 \
    --max_completion_length 2048 \
    --weight_decay 0.0 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type constant \
    --use_wandb_logging
```

### rlvrcode
```bash
# download & preprocessing
mkdir -p datasets/source/rl/ai2/code/dolci-rl-zero-code-7b && \
    hf download allenai/Dolci-RL-Zero-Code-7B --repo-type dataset --local-dir datasets/source/rl/ai2/code/dolci-rl-zero-code-7b && \
    python prepare_rlvrcode-dataset.py
```

```bash
# serving env for verfiable code reward
uvicorn utils.code_utils.api:app --host 0.0.0.0 --port 1234
```

```bash
# serving model on A100x1
CUDA_VISIBLE_DEVICES=3 trl vllm-serve \
    --model checkpoints/sft/qwen2.5-1.5b \
    --data_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --dtype auto \
    --max_model_len 32768 \
    --port 8000
```

```bash
# training model on A100x3
WANDB_PROJECT=rlvrcode-qwen2.5 CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file accelerate_configs/deepspeed_zero1_for_rl.yaml \
    --main_process_port 8001 \
    run_rlvrcode.py \
    --pretrained_model_name_or_path checkpoints/sft/qwen2.5-1.5b \
    --chat_template_path checkpoints/sft/qwen2.5-1.5b/chat_template.jinja \
    --data_dirpath datasets/rl/code \
    --output_dirpath checkpoints/rl/rlvrcode-qwen2.5-1.5b \
    --save_strategy steps \
    --save_total_limit 5 \
    --save_steps 100 \
    --torch_empty_cache_steps 100 \
    --num_train_epochs 2 \
    --num_generations 8 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --steps_per_generation 2 \
    --max_completion_length 2048 \
    --weight_decay 0.0 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type constant \
    --use_wandb_logging
```