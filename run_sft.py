import os
import torch
from pathlib import Path
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig


class TokenizerSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            kwargs["processing_class"].save_pretrained(args.output_dir)


def get_main_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="checkpoints/base/qwen3-1.7b-base")
    parser.add_argument("--chat_template_path", type=str, default="assets/qwen3_instruct/chat_template.jinja")
    parser.add_argument("--use_wandb_logging", action="store_true")
    parser.add_argument("--data_dirpath", type=str, default="datasets/sft")
    parser.add_argument("--output_dirpath", type=str, default="checkpoints/sft/qwen3-1.7b-tulu3-subsets")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--save_strategy", type=str, choices=["epoch", "steps"], default="steps")
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--torch_empty_cache_steps", type=int, default=100)
    parser.add_argument("--no_save_only_model", action="store_false", dest="save_only_model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, choices=["linear", "cosine"], default="cosine")
    args = parser.parse_args()
    return args


def main():
    args = get_main_args()
    list_of_data_filepaths = [
        str(filepath.absolute()) for filepath in Path(args.data_dirpath).absolute().rglob("*.zst.parquet")
    ]
    sft_ds = load_dataset(
        "parquet", data_files=list_of_data_filepaths, split="train", num_proc=os.cpu_count() // 4
    ).select_columns(["messages"])

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    chat_template = open(args.chat_template_path).read()
    tokenizer.chat_template = chat_template
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        attn_implementation="flash_attention_2",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    sft_config = SFTConfig(
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        logging_first_step=True,
        logging_steps=5,
        report_to="wandb" if args.use_wandb_logging else "none",
        run_name=args.output_dirpath.split("/")[-1],
        chat_template_path=args.chat_template_path,
        output_dir=args.output_dirpath,
        dataset_num_proc=os.cpu_count() // 4,
        bf16=True,
        assistant_only_loss=True,
        packing=True,
        packing_strategy="bfd",
        padding_free=True,
        group_by_length=False,
        eval_strategy="no",
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        torch_empty_cache_steps=args.torch_empty_cache_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
    )

    # Initialize the SFT trainer (subclass adds precomputed length column when group_by_length=True)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_ds,
        processing_class=tokenizer,
        callbacks=[TokenizerSaveCallback()],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dirpath)


if __name__ == "__main__":
    main()
