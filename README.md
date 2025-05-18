# Finetune LLM

A comprehensive framework to fine tune and evaluate LLMs.

## Setting Up

The recommended option for installing all dependencies is `uv`:

1. Make sure you have `uv` installed. One way to install `uv` is via pip: `pip install uv`
2. Create the virtual environment with the required packages using `uv sync`

## Fine Tuning

To fine tune on a dataset, you must define a **processor** for this dataset (more on processors [here](processors/README.md)).

To run the fine tuning:

```bash
accelerate launch finetune_sft.py \
  --model_id=google/gemma-3-4b-it \
  --dataset_id=d4nieldev/qpl-composer-ds \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --learning_rate=2e-4 \
  --num_train_epochs=4 \
  --gradient_checkpointing=True \
  --logging_steps=0.01 \
  --save_steps=0.5 \
  --random_seed=1 \
  --lora \
  --r=16 \
  --alpha=32 \
  --dropout=0.05
```

To see information about the different arguments run: `python finetune_sft.py --help`.
