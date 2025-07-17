"""
!! IMPORTANT !!
This script is currently under development and is not fully functional.
For now, the training process is not stable, it seems like we have exploding gradients even though we are using gradient clipping.
The model is not converging and the loss is not decreasing.
"""
import os
import argparse
import logging
from datetime import datetime

import torch
from datasets import load_dataset
from unsloth import FastModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig

from src.callbacks import MemoryLoggingCallback
from src.prompters import PrompterRegistry


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"  # for LoRA: https://github.com/pytorch/pytorch/issues/93661
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a CausalLM on a dataset.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    hf_ids_group = parser.add_argument_group("Hugging Face IDs")
    hf_ids_group.add_argument("--model_id", type=str, default="unsloth/gemma-3-4b-it", help="Model ID to use for fine-tuning.")
    hf_ids_group.add_argument("--dataset_id", type=str, default="dair-ai/emotion", help="Dataset ID to use for fine-tuning.")
    
    train_config_group = parser.add_argument_group("Training config")
    train_config_group.add_argument("--train_batch_size", type=int, default=1, help="Training batch size (per GPU).")
    train_config_group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    train_config_group.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    train_config_group.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    train_config_group.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False, help="Enable gradient checkpointing.")
    
    monitoring_group = parser.add_argument_group("Monitoring")
    monitoring_group.add_argument("--logging_steps", type=int, default=500, help="Log every N steps. If between 0 to 1, part of the epoch.")
    monitoring_group.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps. If between 0 to 1, part of the epoch.")
    monitoring_group.add_argument("--random_seed", type=int, default=1, help="Random seed for reproduction.")
    
    lora_config_group = parser.add_argument_group("LoRA config")
    lora_config_group.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True, help="Use LoRA for fine-tuning.")
    lora_config_group.add_argument("--r", type=int, default=16, help="LoRA rank.")
    lora_config_group.add_argument("--alpha", type=int, default=32, help="LoRA alpha.")
    lora_config_group.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout.")
    
    args = parser.parse_args()

    return args


def args_str(args):
    model_id = args.model_id.split("/")[-1]
    dataset_id = args.dataset_id.split("/")[-1]
    other_args = "_".join([f"{k}={v}" for k, v in vars(args).items() if k not in ["model_id", "dataset_id"]])
    return f"{model_id}-{dataset_id}_{other_args}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def train(
    args
):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}={v}")
    
    max_seq_length = 1024

    # Step 1. Load model & tokenizer
    model, tokenizer = FastModel.from_pretrained(
        args.model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )
    
    if args.lora:
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers = False,
            finetune_language_layers = True,
            finetune_attention_modules = True,
            finetune_mlp_modules = True,
            r = args.r,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias = "none",
            random_state = args.random_seed,
        )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3"
    )
    
    
    # Step 3. Data preperation
    train_dataset = load_dataset(args.dataset_id, split="train")  # type: ignore
    prompter = PrompterRegistry.get(args.dataset_id)()
    train_dataset = train_dataset.map(prompter.to_chat_template, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.rename_column("messages", "conversations")  # for standardization
    train_dataset = standardize_data_formats(train_dataset)

    print(f"AFTER STANDARTIZATION:\n\n{train_dataset[0]}\n\n")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize = False,
                add_generation_prompt = False
            ).removeprefix('<bos>') 
            for convo in convos
        ]
        return { "text" : texts, }
    
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    print(f"AFTER FORMATTING:\n\n{train_dataset[0]['text']}\n\n")
    
    
    # Step 4. Training
    dirname = args_str(args)
    training_args = SFTConfig(
        dataset_text_field          = "text",
        per_device_train_batch_size = args.train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_train_epochs            = args.num_train_epochs,
        warmup_steps                = 5,
        learning_rate               = args.learning_rate,
        logging_steps               = args.logging_steps,
        output_dir                  = f"./output/unsloth/{dirname}",
        run_name                    = dirname,
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = args.random_seed,
        report_to                   = ["wandb"],
        dataset_num_proc            = 2,
        max_seq_length              = max_seq_length,
        # packing                     = True,
        # fp16                        = not is_bfloat16_supported(),
        # bf16                        = is_bfloat16_supported(),
        # logging_strategy            = "steps",
        # save_strategy               = "steps",
        # save_steps                  = args.save_steps,
        # max_grad_norm               = 1.0
    )

    trainer = SFTTrainer(
        model              = model,
        processing_class   = tokenizer,
        train_dataset      = train_dataset, 
        args               = training_args,
        callbacks          = [MemoryLoggingCallback()],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n"
    )

    print(f"SHOULD HAVE SINGLE <bos> TOKEN:\n\n{tokenizer.decode(trainer.train_dataset[0]['input_ids'])}\n\n")
    print(f"MASKED EXAMPLE:\n\n{tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]['labels']]).replace(tokenizer.pad_token, ' ')}\n\n")


    trainer.train()
    # tokenizer.save_pretrained(training_args.output_dir)
    

if __name__ == "__main__":
    args = parse_args()
        
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train(
        args=args,
    )
