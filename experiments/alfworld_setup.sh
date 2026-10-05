#!/usr/bin/env bash
# Install ALFWorld (text-only) and download the TextWorld-PDDL game files into data/alfworld.
# TextWorld's Z-machine backend (jericho) is not needed for ALFWorld and is skipped
# (alfworld_eval.py stubs it); fast-downward is compiled from source (needs g++ and cmake).
set -euo pipefail
pip install -q --no-deps "textworld==1.7.0" "alfworld==0.4.2"
pip install -q "tatsu==5.8.3" hashids mementos termcolor prompt_toolkit more_itertools networkx tqdm \
    "fast-downward-textworld==20.6.4" openai tokenizers
mkdir -p data/alfworld
if [ ! -d data/alfworld/json_2.1.1/valid_unseen ]; then
  curl -sSL -o data/alfworld/tw-pddl.zip \
    https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip
  unzip -q -o data/alfworld/tw-pddl.zip -d data/alfworld
fi
echo "valid_unseen games: $(find data/alfworld/json_2.1.1/valid_unseen -name game.tw-pddl | wc -l)"   # expect 134
