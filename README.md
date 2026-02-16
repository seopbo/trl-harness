# trl-harness
Complete examples based on [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
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
### sft
```bash
mkdir -p datasets/tulu-3-sft-mixture
hf download --repo-type dataset allenai/tulu-3-sft-mixture --local-dir datasets/tulu-3-sft-mixture
```

```bash
# A100 80GB x 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 accelerate launch --config_file accelerate_configs/multi_gpu.yaml run_sft.py
```