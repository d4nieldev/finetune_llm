import argparse
from typing import Type
import importlib

from processors import BaseProcessor
from callbacks import MemoryLoggingCallback
from utils.argparse_utils import collect_kwargs

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a CausalLM on a dataset.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    hf_ids_group = parser.add_argument_group("Hugging Face IDs")
    hf_ids_group.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Model ID to use for fine-tuning.")
    hf_ids_group.add_argument("--dataset_id", type=str, default="bgunlp/question_decomposer_ds", help="Dataset ID to use for fine-tuning.")

    processor_group = parser.add_argument_group("Processor")
    processor_group.add_argument("--processor_class", type=str, default="QPLDecomposerProcessor", 
                                 help="Processor class to use for fine-tuning. If the processor requires additional keyword arguments, "
                                      "please provide them in the format 'p_key=value' - the acceptable types are [int | str | float | list | dict].")
    
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

    args, unknown = parser.parse_known_args()
    processor_kwargs = collect_kwargs(unknown, prefix="p_")

    return args, processor_kwargs


def train(processor_cls: Type[BaseProcessor], args, **processor_kwargs):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}={v}")
    print(f"\nProcessor ({processor_cls}) kwargs:")
    for k, v in processor_kwargs.items():
        print(f"{k}={v}")
    
    # Step 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    
    # Step 2. Data preperation
    dataset = load_dataset(args.dataset_id)
    if not isinstance(dataset, dict):
        raise ValueError("Dataset should be a `DatasetDict` with train and validation splits.")
    
    processor = processor_cls(
        tokenizer=tokenizer,
        model=model,
        dataset=dataset,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        **processor_kwargs
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Step 3. Training
    dirname = f"{args.model_id.split('/')[-1]}-{args.dataset_id.split('/')[-1]}"
    training_args = TrainingArguments(
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
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = processor.processed_train,
        processing_class = tokenizer,
        data_collator    = data_collator,
        callbacks        = [MemoryLoggingCallback()] + processor.get_callbacks(),
    )

    trainer.train()
    
    
if __name__ == "__main__":
    args, processor_kwargs = parse_args()

    processors = importlib.import_module("processors")
    processor_cls = getattr(processors, args.processor_class)
    
    train(
        processor_cls=processor_cls,
        args=args,
        **processor_kwargs
    )
