import os
import torch
from pathlib import Path
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.generation.vllm_client import VLLMClient
from utils.groud_truth_utils import CodeVerifier, CodeVerifierConfig


class TokenizerSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            kwargs["processing_class"].save_pretrained(args.output_dir)


def get_main_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="checkpoints/sft/qwen2.5-1.5b")
    parser.add_argument("--chat_template_path", type=str, default="checkpoints/sft/qwen2.5-1.5b/chat_template.jinja")
    parser.add_argument("--use_wandb_logging", action="store_true")
    parser.add_argument("--data_dirpath", type=str, default="datasets/rl/code")
    parser.add_argument("--output_dirpath", type=str, default="checkpoints/rl/rlvrcode-qwen2.5-1.5b")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--save_strategy", type=str, choices=["epoch", "steps"], default="steps")
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--torch_empty_cache_steps", type=int, default=100)
    parser.add_argument("--no_save_only_model", action="store_false", dest="save_only_model")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--steps_per_generation", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, choices=["linear", "cosine", "constant", "constant_with_warmup"], default="constant")
    args = parser.parse_args()
    return args


def main():
    args = get_main_args()
    list_of_data_filepaths = [
        str(filepath.absolute()) for filepath in Path(args.data_dirpath).absolute().rglob("*.parquet")
    ]
    code_ds = load_dataset("parquet", data_files=list_of_data_filepaths, split="train", num_proc=os.cpu_count() // 4)
    config = CodeVerifierConfig(
        code_api_url="http://localhost:1234/test_program",  
        code_max_execution_time=6.0,
        code_pass_rate_reward_threshold=0.99,
        code_apply_perf_penalty=False
    )

    verifier = CodeVerifier(config)

    def make_reward(verifier):
        def code_reward(completions, ground_truth, **kwargs):
            rewards = []
            for completion, label in zip(completions, ground_truth):
                result = verifier(
                    tokenized_prediction=[],
                    prediction=completion[0]["content"],
                    label=label
                )
                rewards.append(result.score)
            return rewards
        return code_reward
    code_reward = make_reward(verifier)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    chat_template = open(args.chat_template_path).read()
    tokenizer.chat_template = chat_template
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        attn_implementation="flash_attention_2",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    # grpo_config = GRPOConfig(
    #     report_to="wandb" if args.use_wandb_logging else "none",
    #     run_name=args.output_dirpath.split("/")[-1],
    #     output_dir=args.output_dirpath,
    #     remove_unused_columns=False,
    #     eval_strategy="no",
    #     save_strategy=args.save_strategy,
    #     save_total_limit=args.save_total_limit,
    #     save_steps=args.save_steps,
    #     save_only_model=args.save_only_model,
    #     torch_empty_cache_steps=args.torch_empty_cache_steps,
    #     logging_steps=5,
    #     log_completions=False,
    #     num_completions_to_print=1,
    #     bf16=True,
    #     disable_dropout=True,
    #     cast_lm_head_to_fp32=False,
    #     num_train_epochs=args.num_train_epochs,
    #     num_generations=args.num_generations,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     steps_per_generation=args.steps_per_generation,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     num_iterations=1,
    #     beta=0.04,
    #     epsilon=0.2,
    #     scale_rewards="group",
    #     loss_type="grpo",
    #     lr_scheduler_type=args.lr_scheduler_type,
    #     warmup_ratio=args.warmup_ratio,
    #     learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     adam_beta1=0.9,
    #     adam_beta2=0.95,
    #     adam_epsilon=1e-8,
    #     max_grad_norm=1.0,
    #     ds3_gather_for_generation=False,
    #     chat_template_kwargs={"add_generation_prompt": True},
    #     use_vllm=True,
    #     vllm_mode="server",
    #     vllm_server_base_url="http://localhost:8000",
    #     max_completion_length=args.max_completion_length,
    #     temperature=1.0,
    #     top_p=1.0,
    #     top_k=0,
    # )

    grpo_config = GRPOConfig(
        report_to="wandb" if args.use_wandb_logging else "none",
        run_name=args.output_dirpath.split("/")[-1],
        output_dir=args.output_dirpath,
        remove_unused_columns=False,
        eval_strategy="no",
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        torch_empty_cache_steps=args.torch_empty_cache_steps,
        logging_steps=5,
        log_completions=False,
        num_completions_to_print=1,
        bf16=True,
        disable_dropout=True,
        cast_lm_head_to_fp32=False,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        steps_per_generation=args.steps_per_generation,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_iterations=1,
        beta=0.04,
        epsilon=0.2,
        epsilon_high=0.28,
        scale_rewards="group",
        loss_type="dapo",
        mask_truncated_completions=True,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        ds3_gather_for_generation=False,
        chat_template_kwargs={"add_generation_prompt": True},
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://localhost:8000",
        max_completion_length=args.max_completion_length,
        temperature=0.85,
        top_p=1.0,
        top_k=0,
    )

    # monkey-patch VLLMClient to remove tools from the request
    _original_chat = VLLMClient.chat
    def _patched_chat(self, *args, **kwargs):
        if "tools" in kwargs and not kwargs["tools"]:
            kwargs["tools"] = None
        return _original_chat(self, *args, **kwargs)
    VLLMClient.chat = _patched_chat
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=code_reward,
        args=grpo_config,
        train_dataset=code_ds,
        processing_class=tokenizer,
        callbacks=[TokenizerSaveCallback()],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dirpath)


if __name__ == "__main__":
    main()
