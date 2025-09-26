import argparse
import logging
from pathlib import Path

import torch
import wandb
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from unsloth import add_new_tokens
from trl.trainer.sft_config import SFTConfig
from transformers.training_args import OptimizerNames
from datasets import Dataset
from dotenv import load_dotenv

from src.callbacks import MemoryLoggingCallback
from src.prompters import PrompterRegistry
import src.utils.paths as p
from src.training import trainers

load_dotenv()

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a LLM on a dataset.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    hf_ids_group = parser.add_argument_group("Hugging Face IDs")
    hf_ids_group.add_argument("--model_id_or_path", type=str, required=True, help="Model ID to use for fine-tuning.")
    hf_ids_group.add_argument("--resume_from_checkpoint", type=Path, default=None, help="Path to a checkpoint to resume training from.")
    hf_ids_group.add_argument("--dataset_id", type=str, required=True, help="Dataset ID to use for fine-tuning.")

    train_config_group = parser.add_argument_group("Training config")
    train_config_group.add_argument("--sort_data", action=argparse.BooleanOptionalAction, default=False, help="Sort data in ascending order by prompt length before training.")
    train_config_group.add_argument("--shuffle_data", action=argparse.BooleanOptionalAction, default=True, help="Shuffle data before training.")
    train_config_group.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    train_config_group.add_argument("--train_batch_size", type=int, default=1, help="Training batch size (per GPU).")
    train_config_group.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing.")
    train_config_group.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps.")
    train_config_group.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    train_config_group.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type.")
    train_config_group.add_argument("--optim", type=OptimizerNames, choices=[opt.value for opt in OptimizerNames], default=OptimizerNames.ADAMW_TORCH, help="Optimizer to use for training.")
    train_config_group.add_argument("--warmup_ratio", type=float, default=0.15, help="Warmup ratio for learning rate scheduler.")
    train_config_group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer.")
    train_config_group.add_argument("--num_train_epochs", type=int, default=4, help="Number of training epochs.")
    train_config_group.add_argument("--max_seq_length", type=int, default=32768, help="Maximum sequence length for training.")
    train_config_group.add_argument("--quantization", type=int, choices=[4, 8, None], default=None, help="Quantization bits (4 or 8). Default is None (bf16/fp16)")
    
    monitoring_group = parser.add_argument_group("Monitoring")
    monitoring_group.add_argument("--logging_steps", type=float, default=1, help="Log every N steps. If between 0 to 1, part of total_steps.")
    monitoring_group.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size (per GPU).")
    monitoring_group.add_argument("--eval_steps", type=float, default=0.25, help="Evaluate every N steps. If between 0 to 1, part of total steps.")
    monitoring_group.add_argument("--save_steps", type=float, default=0.25, help="Save checkpoint every N steps. If between 0 to 1, part of total steps.")
    monitoring_group.add_argument("--save_total_limit", type=int, default=1, help="Maximum number of checkpoints to keep.")
    monitoring_group.add_argument("--random_seed", type=int, default=1, help="Random seed for reproduction.")
    
    best_model_group = parser.add_argument_group("Best model logic")
    best_model_group.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True, help="Load the best model at the end of training.")
    best_model_group.add_argument("--greater_is_better", action=argparse.BooleanOptionalAction, default=False, help="Whether a higher metric value is better.")
    best_model_group.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Metric to use for determining the best model.")

    lora_config_group = parser.add_argument_group("LoRA config")
    lora_config_group.add_argument("--lora", action=argparse.BooleanOptionalAction, default=False, help="Use LoRA for fine-tuning.")
    lora_config_group.add_argument("--r", type=int, default=256, help="LoRA rank.")
    lora_config_group.add_argument("--alpha", type=int, default=256, help="LoRA alpha.")
    lora_config_group.add_argument("--dropout", type=float, default=0, help="LoRA dropout.")
    
    args = parser.parse_args()

    return args


def args_str(args, run_id):
    model_id = args.model_id_or_path.split("/")[-1]
    dataset_id = args.dataset_id.split("/")[-1]
    shortname = {
        'sort_data': 'sort',
        'train_batch_size': 'bsz',
        'gradient_checkpointing': 'gc',
        'gradient_accumulation_steps': 'ga',
        'learning_rate': 'lr',
        'lr_scheduler_type': 'lrs',
        'optim': 'opt',
        'warmup_ratio': 'wr',
        'weight_decay': 'wd',
        'quantization': 'q',
        'num_train_epochs': 'epochs',
        'max_seq_length': 'maxlen',
        'random_seed': 'seed',
    }
    if args.lora:
        shortname.update({
            'lora': 'lora',
            'r': 'r',
            'alpha': 'a',
            'dropout': 'dp',
        })
    other_args = "_".join([
        f"{shortname[k]}={v.replace('/', '-')}" if isinstance(v, str) else 
        (shortname[k] if isinstance(v, bool) else (
            f"{shortname[k]}={v.stem}" if isinstance(v, Path) else f"{shortname[k]}={v}"
        ))
        for k, v in vars(args).items() 
        if k in shortname and v is not None and (not isinstance(v, bool) or v is True)
    ])
    return f"{run_id}_{model_id}-{dataset_id}_{other_args}"


def train(
    args
):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}={v}")

    # Step 1. Load prompter
    prompter = PrompterRegistry.get(args.dataset_id)(with_assistant=True)
    
    # Step 2. Load model & tokenizer, and configure if needed
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_id_or_path,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.quantization == 4,
        load_in_8bit = args.quantization == 8,
        full_finetuning = not args.lora,
        trust_remote_code = True,
        attn_implementation="sdpa"
    )
    # TODO: solve bug in unsloth!
    # add_new_tokens(model, tokenizer, list(prompter.special_tokens_to_add().values()))
    if args.lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.r,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = args.alpha,
            lora_dropout = args.dropout,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = args.random_seed,
        )
    
    # Step 3. Data preperation
    train_dataset: Dataset = prompter.load_dataset()['train']
    if args.sort_data:
        # sort by prompt length
        train_dataset = train_dataset.map(
            lambda ex: {'len_text': sum(len(m['content']) for m in ex['messages'] if m['role'] == 'user')}, 
        )
        train_dataset = train_dataset.sort("len_text", reverse=False)
    if args.shuffle_data:
        train_dataset = train_dataset.shuffle(seed=args.random_seed)
    train_dataset = train_dataset.map(
        lambda ex: {'text': tokenizer.apply_chat_template(prompter.to_chat_template(ex)['messages'], tokenize=False, add_generation_prompt=False)}, 
        remove_columns=train_dataset.column_names
    )

    eval_dataset: Dataset = prompter.load_dataset()['validation']
    eval_dataset = eval_dataset.map(
        lambda ex: {'text': tokenizer.apply_chat_template(prompter.to_chat_template(ex)['messages'], tokenize=False, add_generation_prompt=False)}, 
        remove_columns=eval_dataset.column_names
    )
    
    # Step 4. Training
    run = wandb.init(
        project=f"{args.model_id_or_path.replace('/', '-')}_{args.dataset_id.replace('/', '-')}",
        config=vars(args),
        resume="allow"
    )
    run_id = run.id
    
    dirname = args_str(args, run_id)
    local_output_dir = str(p.TRAINED_MODELS_DIR / f"{dirname}")
    training_args = SFTConfig(
        dataset_text_field="text",
        # local_rank                    = args.local_rank,
        output_dir                    = local_output_dir,
        per_device_train_batch_size   = args.train_batch_size,
        per_device_eval_batch_size    = args.eval_batch_size,
        # gradient_checkpointing        = args.gradient_checkpointing,
        gradient_accumulation_steps   = args.gradient_accumulation_steps,
        learning_rate                 = args.learning_rate,
        optim                         = args.optim,
        num_train_epochs              = args.num_train_epochs,
        lr_scheduler_type             = args.lr_scheduler_type,
        warmup_ratio                  = args.warmup_ratio,
        weight_decay                  = args.weight_decay,
        ## ------ Monitoring ------
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        ## ---- best-model logic ----
        load_best_model_at_end=args.load_best_model_at_end,
        greater_is_better=args.greater_is_better,
        metric_for_best_model=args.metric_for_best_model,
        ## --------------------------
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        log_on_each_node=False,
        push_to_hub=False,
        disable_tqdm=False,
        # ddp_backend='nccl' if int(os.environ.get("WORLD_SIZE", 1)) > 1 else None,
        # ddp_find_unused_parameters=False,  # Enable for MoE
        # ddp_timeout=43200,  # 12 hours
        save_total_limit=args.save_total_limit,
        # max_length=args.max_seq_length,
        # pad_token=tokenizer.pad_token,
        # assistant_only_loss=True,
        # packing=False,
        padding_free=True if args.train_batch_size > 1 else False,
        # fsdp=["full_shard", "offload"],
        # adam_beta1=args.adam_beta1,
        # adam_beta2=args.adam_beta2,
        # adam_epsilon=args.adam_epsilon,

        run_name                      = dirname,
        seed                          = args.random_seed,
        report_to                     = "wandb",
        # remove_unused_columns         = False,
        # save_safetensors              = args.lora  # Safetensors can't save tensors that share the same memory
    )

    trainer_cls = getattr(trainers, prompter.trainer_cls_name)
    trainer = trainer_cls(
        model            = model,
        processing_class = tokenizer,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        # eval_packing     = False,
        callbacks        = [MemoryLoggingCallback()],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save the model - LoRA weights only by default
    model.save_pretrained(local_output_dir)  # Local saving
    tokenizer.save_pretrained(local_output_dir)

    print(f"======= Model and tokenizer saved to {local_output_dir} =======")

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
