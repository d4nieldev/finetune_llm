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
import wandb


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

    if 0 < args.logging_steps < 1:
        args.logging_steps /= args.gradient_accumulation_steps * args.train_batch_size
    if 0 < args.save_steps < 1:
        args.save_steps /= args.gradient_accumulation_steps * args.train_batch_size

    return args


def args_str(args, run_id):
    model_id = args.model_id.split("/")[-1]
    dataset_id = args.dataset_id.split("/")[-1]
    other_args = "_".join([f"{k}={v}" for k, v in vars(args).items() if k not in ["model_id", "dataset_id"]])
    return f"{run_id}_{model_id}-{dataset_id}_{other_args}"


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
        # Make sure that pad_token is different from eos_token, because it requires a special treatment in the collator
        # https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#5-when-pad_token-equals-eos_token
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    logger.info(f"==== Length of Tokenizer: {len(tokenizer)} ====")
    
    
    # Step 2. Load model
    if 'gemma' in args.model_id:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id)
    
    trainer_kwargs = {}
    if args.lora:
        trainer_kwargs['peft_config'] = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules="all-linear",
        )
    else:
        # torch.compile is not compatible with LoRA
        # https://huggingface.co/docs/peft/en/developer_guides/torch_compile?utm_source=chatgpt.com
        model = model.compile(model)
    
    # Resize token embeddings after adding special <|pad|> token
    if len(tokenizer) != original_vocab_size:
        print("Tokenizer size was changed from {} to {}".format(original_vocab_size, len(tokenizer)))
        model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Memory footprint: {model.get_memory_footprint()}")
    
    
    # Step 3. Data preperation
    train_dataset: Dataset = load_dataset(args.dataset_id, split="train")  # type: ignore
    processor = ProcessorRegistry.get(args.dataset_id)()
    train_dataset = train_dataset.map(lambda ex: processor.to_chat_template(ex, train=True), remove_columns=train_dataset.column_names)
    def to_model_prompt(example):
        # example["messages"] is a list of {"role": "...", "content": "..."}
        # https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#6-applying-the-chat-template-is-not-a-homomorphism-with-respect-to-concatenation
        input_ids = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,   # generation is included in the messages
            continue_final_message=False,  # put eos token at the end of the last message
        )[0]                               # remove batch dim
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
        instruction_template=instruction_template,  # can be None for single-turn conversations
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
    run = wandb.init(
        project=f"{args.model_id}-{args.dataset_id}",
        config=vars(args),
        resume="allow"
    )
    run_id = run.id
    
    dirname = args_str(args, run_id)
    training_args = SFTConfig(
        output_dir                  = f"./output/{dirname}",
        run_name                    = dirname,
        per_device_train_batch_size = args.train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        gradient_checkpointing      = args.gradient_checkpointing,
        learning_rate               = args.learning_rate,
        num_train_epochs            = args.num_train_epochs,
        eval_strategy               = "no",
        logging_strategy            = "steps",
        logging_steps               = args.logging_steps,
        save_strategy               = "steps",
        save_steps                  = args.save_steps,
        seed                        = args.random_seed,
        report_to                   = ["wandb"],
        max_seq_length              = max_length,
        remove_unused_columns       = False,
        save_safetensors            = False # Safetensors can't save tensors that share the same memory
    )

    if args.lora:
        # to supress warning: No label_names provided for model class PeftModelForCausalLM. Since PeftModel hides base models input arguments, if label_names is not given, label_names can't be set automatically within Trainer
        # https://github.com/unslothai/unsloth/issues/1788#issuecomment-2772497747
        training_args.label_names = ["labels"]

    trainer = SFTTrainer(
        model            = model,  # type: ignore
        args             = training_args,
        data_collator    = data_collator,
        train_dataset    = train_dataset,
        processing_class = tokenizer,
        callbacks        = [MemoryLoggingCallback()],
        **trainer_kwargs,
    )

    trainer.train()
    tokenizer.save_pretrained(training_args.output_dir)

    run.finish()
    
    
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
