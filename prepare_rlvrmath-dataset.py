import os
import math
from datasets import load_dataset
from tqdm.contrib.concurrent import process_map
from pathlib import Path
from functools import partial
from datasets import load_dataset

MiB = 1024 * 1024


def calculate_proper_number_of_shards(dataset_size_in_bytes):
    return math.ceil(dataset_size_in_bytes / (128 * MiB))


def save_shard(idx, num_shards, output_dirpath, dset):
    shard_dataset = dset.shard(num_shards=num_shards, index=idx, contiguous=True)
    shard_filename = f"shard_{idx:09d}.zst.parquet"
    shard_filepath = f"{output_dirpath}/{shard_filename}"
    shard_dataset.to_parquet(shard_filepath, compression="zstd")


def main():
    list_of_allenai_data_filepaths = [
        str(filepath.absolute()) for filepath in Path("datasets/source/rl/ai2/math/rlvr-gsm-math-if-mixed-constraints").rglob("*.parquet")
    ]
    rl_ds = load_dataset("parquet", num_proc=os.cpu_count() // 4, data_files=list_of_allenai_data_filepaths, split="train")
    rl_ds = rl_ds.filter(lambda x: x["dataset"] in ["MATH", "gsm8k"], num_proc=os.cpu_count() // 4)
    rl_ds = rl_ds.select_columns(["messages", "ground_truth"])
    rl_ds = rl_ds.rename_column("messages", "prompt")
    rl_ds[0]
    num_shards = calculate_proper_number_of_shards(rl_ds._estimate_nbytes())

    save_shard_ = partial(
        save_shard,
        num_shards=num_shards,
        output_dirpath="datasets/rl/math",
        dset=rl_ds,
    )

    _ = process_map(
        save_shard_,
        range(num_shards),
        max_workers=os.cpu_count() // 4,
        chunksize=max(math.ceil(num_shards // (os.cpu_count() // 4)), 1),
    )


if __name__ == "__main__":
    main()
