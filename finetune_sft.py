import os
import argparse
import logging

from callbacks import MemoryLoggingCallback
from processors import ProcessorRegistry

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"  # for LoRA: https://github.com/pytorch/pytorch/issues/93661
logger = logging.getLogger(__name__)


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
    monitoring_group.add_argument("--logging_steps", type=int, default=500, help="Log every N steps. If between 0 to 1, part of the epoch.")
    monitoring_group.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps. If between 0 to 1, part of the epoch.")
    monitoring_group.add_argument("--random_seed", type=int, default=1, help="Random seed for reproduction.")
    
    lora_config_group = parser.add_argument_group("LoRA config")
    lora_config_group.add_argument("--r", type=int, default=16, help="LoRA rank.")
    lora_config_group.add_argument("--alpha", type=int, default=32, help="LoRA alpha.")
    lora_config_group.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout.")
    
    args = parser.parse_args()

    return args


def args_str(args):
    model_id = args.model_id.split("/")[-1]
    dataset_id = args.dataset_id.split("/")[-1]
    other_args = "_".join([f"{k}={v}" for k, v in vars(args).items() if k not in ["model_id", "dataset_id"]])
    return f"{model_id}_{dataset_id}_{other_args}"


def train(
    args
):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}={v}")
    
    # Step 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    original_vocab_size = len(tokenizer)
    if not tokenizer.pad_token or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    logger.info(f"==== Length of Tokenizer: {len(tokenizer)} ====")
    
    
    # Step 2. Load model
    if 'gemma' in args.model_id:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id)
    
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules="all-linear",
    )
    model = torch.compile(model)
    
    # Resize token embeddings after adding special <|pad|> token
    if len(tokenizer) != original_vocab_size:
        print("Tokenizer size was changed from {} to {}".format(original_vocab_size, len(tokenizer)))
        model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Memory footprint: {model.get_memory_footprint()}")
    
    
    # Step 3. Data preperation
    train_dataset: Dataset = load_dataset(args.dataset_id, split="train")  # type: ignore
    processor = ProcessorRegistry.get(args.dataset_id)()
    train_dataset = train_dataset.map(processor.to_chat_template, remove_columns=train_dataset.column_names)
    def to_model_prompt(example):
        # example["messages"] is a list of {"role": "...", "content": "..."}
        input_ids = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,               # returns a tensor of token‑IDs
            add_generation_prompt=False, # keep existing answers in the text
            continue_final_message=False,
            return_tensors="pt"
        )[0]                             # remove batch dim
        return {"input_ids": input_ids}
    train_dataset = train_dataset.map(to_model_prompt, remove_columns=train_dataset.column_names)
    
    # Find instruction and response prefixes
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "!"}, {"role": "assistant", "content": "?"}],
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=False,
    )
    
    a_id = tokenizer.convert_tokens_to_ids('!')
    b_id = tokenizer.convert_tokens_to_ids('?')
    
    instruction_template = ids[:ids.index(a_id)]
    response_template = ids[ids.index(a_id)+1:ids.index(b_id)]
    
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    # Find maximum number of tokens in a request and set max_length
    max_tokens = 0
    for example in train_dataset:
        max_tokens = max(max_tokens, len(example["input_ids"]))  # type: ignore
    max_length = int(max_tokens * 1.2)
    print(f"Max tokens in a request: {max_tokens}. Setting max_length to {max_length}.")
    
    
    # Step 4. Training
    dirname = args_str(args)
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
        max_length                  = max_length,
        remove_unused_columns       = False, # 
        save_safetensors            = False  # Safetensors can't save tensors that share the same memory
    )

    trainer = SFTTrainer(
        model            = model,  # type: ignore
        peft_config      = lora_cfg,
        args             = training_args,
        data_collator    = data_collator,
        train_dataset    = train_dataset,
        processing_class = tokenizer,
        callbacks        = [MemoryLoggingCallback()]
    )

    trainer.train()
    tokenizer.save_pretrained(training_args.output_dir)
    
    
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
