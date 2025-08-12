# Finetune LLM

![FineTuna Overview](assets/FineTuna-Inside%20View.drawio.png "FineTuna Overview")

A comprehensive framework to fine tune and evaluate LLMs.

## Setting Up

The recommended option for installing all dependencies is `uv`:

1. Make sure you have `uv` installed. One way to install `uv` is via pip: `pip install uv`
2. Create the virtual environment with the required packages using `uv sync`

## Fine Tuning

### Prerequisites

1. Set up a `.env` file with your weights & biases and huggingface api keys (`WANDB_API_KEY` and `HF_TOKEN` respectively)
2. Make sure your python is configured to the project directory by executing: `export PYTHONPATH=/path/to/project/dir`.
3. Install microsoft odbc driver: `./install_odbc_driver_ubuntu.sh`
4. Install flash attention 2: `uv pip install "flash-attn==2.8.2" --no-build-isolation`


To run the fine tuning:

```bash
nohup uv run deepspeed --no_ssh --node_rank 0 --master_addr localhost --master_port 12355 --num_nodes=1 --num_gpus=1 src/training/finetune_sft.py --model_id "d4nieldev/Qwen3-4B-QPL-AIO" --dataset_id "d4nieldev/qpl-merged-cot-ds" --warmup_ratio 0.05 --num_train_epochs 12 --deepspeed_config "src/training/deepspeed_configs/stage-2-offloading-warmup-cosine-lr.json" --eval_steps 0.08333333 --save_steps 0.08333333 --load_best_model_at_end --metric_for_best_model eval_execution_accuracy --greater_is_better > output.log 2>&1 &
```

To see information about the different arguments run: `python src/training/finetune_sft.py --help`.

