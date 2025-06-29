# Finetune LLM

![FineTuna Overview](assets/FineTuna-Inside%20View.drawio.png FineTuna Overview)

A comprehensive framework to fine tune and evaluate LLMs.

## Setting Up

The recommended option for installing all dependencies is `uv`:

1. Make sure you have `uv` installed. One way to install `uv` is via pip: `pip install uv`
2. Create the virtual environment with the required packages using `uv sync`

## Fine Tuning

### Prerequisites

1. Make sure your python is configured to the project directory by executing: `export PYTHONPATH=/path/to/project/dir`.
2. Login to HuggingFace Hub by executing: `huggingface-cli login` and insert your huggingface token (for pulling models and datasets).
3. Login to Weights & Biases by executing: `wandb login` and insert your W&B api key (for reporting training statistics to W&B).
4. Define a **processor** for your dataset (more on processors [here](processors/README.md)).


To run the fine tuning:

```bash
accelerate launch src/training/finetune_sft.py \
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

To see information about the different arguments run: `python src/training/finetune_sft.py --help`.
