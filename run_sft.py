import os
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


class TokenizerSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            kwargs["processing_class"].save_pretrained(args.output_dir)


def get_main_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/data/ib-huawei-nas-lmt_980/users/mat/workspace/trl-harness/checkpoints/qwen3-1.7b-base")
    parser.add_argument("--chat_template_path", type=str, default="/data/ib-huawei-nas-lmt_980/users/mat/workspace/trl-harness/assets/qwen3_instruct_trl.jinja")
    parser.add_argument("--data_dirpath", type=str, default="/data/ib-huawei-nas-lmt_980/users/mat/workspace/trl-harness/datasets/tulu-3-sft-mixture/data")
    parser.add_argument("--output_dirpath", type=str, default="/data/ib-huawei-nas-lmt_980/users/mat/workspace/trl-harness/test_checkpoints/qwen3-1.7b-tulu3-sft-mixure")
    args = parser.parse_args()
    return args


def main():
    args = get_main_args()
    sft_ds = load_dataset(
        "parquet",
        num_proc=os.cpu_count() // 2,
        split="train",
        data_dir=args.data_dirpath
    )
    sft_ds = sft_ds.select_columns(["messages"])

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    training_chat_template = open(args.chat_template_path).read()
    tokenizer.chat_template = training_chat_template

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype="auto"
    )

    # Initialize the SFT config
    sft_config = SFTConfig(
        chat_template_path=args.chat_template_path,
        output_dir=args.output_dirpath,
        dataset_num_proc=os.cpu_count() // 2,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        bf16=True,
        assistant_only_loss=True,
        max_length=8192,
        packing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        learning_rate=2e-5,
        eval_strategy="no",
        save_steps=50,
    )


    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_ds,
        processing_class=tokenizer,
        callbacks=[TokenizerSaveCallback()]
    )

    trainer.train()

if __name__ == "__main__":
    main()
