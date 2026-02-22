import os
import math
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial

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
        str(filepath.absolute()) for filepath in Path("datasets/source/sft/allenai").rglob("*.parquet")
    ]
    dict_of_allenai_datasets = {
        allenai_data_filepath.split("sft/")[-1]
        .split("/data")[0]: load_dataset(
            "parquet", num_proc=os.cpu_count() // 4, data_files=allenai_data_filepath, split="train"
        )
        .select_columns(["messages"])
        for allenai_data_filepath in list_of_allenai_data_filepaths
    }
    keys_of_dict_of_allenai_datasets = list(dict_of_allenai_datasets.keys())
    list_of_allenai_datasets = []

    for key in keys_of_dict_of_allenai_datasets:
        allenai_dataset = dict_of_allenai_datasets.pop(key)
        allenai_dataset = allenai_dataset.map(
            lambda example: {"messages": example["messages"], "source": key}, num_proc=os.cpu_count() // 4
        )
        list_of_allenai_datasets.append(allenai_dataset)
    else:
        del dict_of_allenai_datasets

    sft_dataset = concatenate_datasets(list_of_allenai_datasets)
    sft_dataset = sft_dataset.shuffle(seed=42)
    num_shards = calculate_proper_number_of_shards(sft_dataset._estimate_nbytes())

    save_shard_ = partial(
        save_shard,
        num_shards=num_shards,
        output_dirpath="datasets/sft",
        dset=sft_dataset,
    )

    _ = process_map(
        save_shard_,
        range(num_shards),
        max_workers=os.cpu_count() // 4,
        chunksize=max(math.ceil(num_shards // (os.cpu_count() // 4)), 1),
    )


if __name__ == "__main__":
    main()
