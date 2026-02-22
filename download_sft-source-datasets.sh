#!/usr/bin/env bash

mkdir -p datasets/source/sft/allenai
for repo in \
  allenai/tulu-3-sft-personas-math-grade-filtered \
  allenai/tulu-3-sft-personas-code \
  allenai/tulu-3-sft-personas-instruction-following
do
  name="${repo#*/}"
  mkdir -p "datasets/source/sft/allenai/${name}"
  hf download "${repo}" --local-dir "datasets/source/sft/allenai/${name}" --repo-type dataset
done
