import os
import argparse
from typing import Callable, List, Dict, Any, TypedDict
import logging

from callbacks import MemoryLoggingCallback

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

class ChatMessage(TypedDict):
    role: str
    content: str

class ChatTemplates(TypedDict):
    messages: List[List[ChatMessage]]


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
    
    args = parser.parse_args()

    return args


def train(
    to_chat_template: Callable[[Dict[str, List[Any]]], ChatTemplates],
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
        inference_mode=False,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj"],
        # modules_to_save=["lm_head", "embed_token"],
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)
    model = torch.compile(model)
    model.print_trainable_parameters()
    
    # Resize token embeddings after adding special <|pad|> token
    if len(tokenizer) != original_vocab_size:
        print("Tokenizer size was changed from {} to {}".format(original_vocab_size, len(tokenizer)))
        model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Memory footprint: {model.get_memory_footprint()}")
    
    
    # Step 3. Data preperation
    train_dataset = load_dataset(args.dataset_id, split="train")
    train_dataset = train_dataset.map(to_chat_template, batched=True, remove_columns=train_dataset.column_names)
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
        return_tensors="pt"
    )[0]
    
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
        max_tokens = max(max_tokens, len(example["input_ids"]))
    max_length = int(max_tokens * 1.2)
    print(f"Max tokens in a request: {max_tokens}. Setting max_length to {max_length}.")
    
    
    # Step 4. Training
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
        max_length                  = max_length,
        remove_unused_columns       = False, # 
        save_safetensors            = False  # Safetensors can't save tensors that share the same memory
    )

    trainer = SFTTrainer(
        model            = model,
        args             = training_args,
        data_collator    = data_collator,
        train_dataset    = train_dataset,
        processing_class = tokenizer,
        callbacks        = [MemoryLoggingCallback()]
    )

    trainer.train()
    
    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    
    
if __name__ == "__main__":
    args = parse_args()
    
    labels_list = ["sadness","joy","love","anger","fear","surprise"]
    
    def to_chat_template(batch: Dict[str, List[Any]]) -> ChatTemplates:
        outputs = []
        for text, label in zip(batch['text'], batch['label']):
            prompt = f"Below is a piece of text. Classify it into one of: {', '.join(labels_list)}.\n\n\"{text}\""
            response = f"The emotion in the above text is: {labels_list[label]}"
            outputs.append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ])
        return {
            'messages': outputs
        }
        
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    train(
        to_chat_template=to_chat_template,
        args=args,
    )
