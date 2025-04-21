import argparse
from typing import Callable, List, Dict, Any, TypedDict

from callbacks import MemoryLoggingCallback
from utils.argparse_utils import collect_kwargs

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


class ChatMessage(TypedDict):
    role: str
    content: str

class ChatTemplate(TypedDict):
    messages: List[ChatMessage]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a CausalLM on a dataset.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    hf_ids_group = parser.add_argument_group("Hugging Face IDs")
    hf_ids_group.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Model ID to use for fine-tuning.")
    hf_ids_group.add_argument("--dataset_id", type=str, default="dair-ai/emotion", help="Dataset ID to use for fine-tuning.")
    
    train_config_group = parser.add_argument_group("Training config")
    train_config_group.add_argument("--train_batch_size", type=int, default=1, help="Training batch size (per GPU).")
    train_config_group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    train_config_group.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    train_config_group.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    
    monitoring_group = parser.add_argument_group("Monitoring")
    monitoring_group.add_argument("--logging_steps", type=int, default=500, help="Log every N steps.")
    monitoring_group.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps.")
    monitoring_group.add_argument("--random_seed", type=int, default=1, help="Random seed for reproduction.")
    
    lora_config_group = parser.add_argument_group("LoRA config")
    lora_config_group.add_argument("--r", type=int, default=16, help="LoRA rank.")
    lora_config_group.add_argument("--alpha", type=int, default=32, help="LoRA alpha.")
    lora_config_group.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout.")
    
    memory_management_group = parser.add_argument_group("Memory management")
    memory_management_group.add_argument("--max_input_tokens", type=int, help="Maximum number of input tokens. Don't provide this argument for automatic inference (according to dataset).")
    memory_management_group.add_argument("--max_output_tokens", type=int, help="Maximum number of output tokens. Don't provide this argument for automatic inference (according to dataset).")
    
    args = parser.parse_args()

    return args


def train(
    to_chat_template: Callable[[List[Dict[str, Any]]], List[ChatTemplate]],
    args
):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}={v}")
    
    # Step 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if 'gemma' in args.model_id:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id)
    
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["lm_head", "embed_token"],
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)
    model = torch.compile(model)
    model.print_trainable_parameters()
    
    # Step 2. Data preperation
    train_dataset = load_dataset(args.dataset_id, split="train")
    train_dataset = train_dataset.map(to_chat_template, batched=True, remove_columns=train_dataset.column_names)

    # Step 3. Training
    dirname = f"{args.model_id.split('/')[-1]}-{args.dataset_id.split('/')[-1]}"
    training_args = SFTConfig(
        output_dir                  = f"./output/{dirname}",
        per_device_train_batch_size = args.train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate               = args.learning_rate,
        num_train_epochs            = args.num_train_epochs,
        eval_strategy               = "no",       # manual via callback
        logging_strategy            = "steps",
        logging_steps               = args.logging_steps,
        save_strategy               = "steps",
        save_steps                  = args.save_steps,
        seed                        = args.random_seed,
        report_to                   = ["wandb"],
        max_length                  = 2048,  # FIXME: This should be generic!
    )

    trainer = SFTTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        processing_class = tokenizer,
        callbacks        = [MemoryLoggingCallback()]
    )

    trainer.train()
    
    
if __name__ == "__main__":
    args = parse_args()
    
    labels_list = ["sadness","joy","love","anger","fear","surprise"]
    
    def to_chat_template(batch: List[Dict[str, Any]]) -> List[ChatTemplate]:
        outputs = []
        for example in batch:
            prompt = f"Below is a piece of text. Classify it into one of: {', '.join(labels_list)}.\n\n"
            prompt += f"\"{example['text']}\"\n\nThe emotion in the above text is: "
            response = f"{labels_list[example['label']]}"
            outputs.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            })
        return outputs
    
    train(
        to_chat_template=to_chat_template,
        args=args,
    )
