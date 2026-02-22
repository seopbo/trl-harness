# trl-harness
Complete examples based on [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)

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
# trl-harness
uv pip install --system --no-cache-dir -r requirements.txt
```

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

## launch
### preliminary

```bash
# model
mkdir -p checkpoints/base/qwen3-1.7b-base
hf download Qwen/Qwen3-1.7B-Base --local-dir checkpoints/base/qwen3-1.7b-base
```

### sft
```bash
bash download_sft-source-datasets.sh
python prepare_sft-dataset.py
```

```bash
# A100 80GB x 8
# global_batch_size=128
export WANDB_PROJECT=sft
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_configs/multi_gpu.yaml \
    --main_process_port 8000 \
    run_sft.py \
    --use_wandb_logging \
    --pretrained_model_name_or_path checkpoints/base/qwen3-1.7b-base \
    --chat_template_path assets/qwen3_instruct/chat_template.jinja \
    --data_dirpath datasets/sft \
    --output_dirpath checkpoints/sft/qwen3-1.7b \
    --save_strategy epoch \
    --no_save_only_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 8192 \
    --weight_decay 0.1 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine
```