import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from peft import LoraConfig, TaskType
import wandb
from dotenv import load_dotenv

from src.callbacks import MemoryLoggingCallback
from src.prompters import PrompterRegistry
import src.utils.paths as p
from src.training import trainers as t

load_dotenv()

# run with:
# deepspeed --no_ssh --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --num_nodes=$NUM_NODES --num_gpus=$NUM_GPUS \
#           src/training/finetune_sft.py --model_id $MODEL_ID --dataset_id $DATASET_ID ...
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "12355"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a CausalLM on a dataset.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    hf_ids_group = parser.add_argument_group("Hugging Face IDs")
    hf_ids_group.add_argument("--model_id_or_path", type=str, required=True, help="Model ID to use for fine-tuning.")
    hf_ids_group.add_argument("--resume_from_checkpoint", type=Path, default=None, help="Path to a checkpoint to resume training from.")
    hf_ids_group.add_argument("--model_revision", type=str, default="main", help="Model revision to use for fine-tuning.")
    hf_ids_group.add_argument("--dataset_id", type=str, required=True, help="Dataset ID to use for fine-tuning.")

    train_config_group = parser.add_argument_group("Training config")
    train_config_group.add_argument("--sort_data", action=argparse.BooleanOptionalAction, default=True, help="Sort data in ascending order by prompt length before training.")
    train_config_group.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    train_config_group.add_argument("--train_batch_size", type=int, default=1, help="Training batch size (per GPU).")
    train_config_group.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing.")
    train_config_group.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps.")
    train_config_group.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate.")
    train_config_group.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    train_config_group.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer to use for training.")
    train_config_group.add_argument("--warmup_ratio", type=float, default=0.15, help="Warmup ratio for learning rate scheduler.")
    train_config_group.add_argument("--weight_decay", type=float, default=1, help="Weight decay for optimizer.")
    train_config_group.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False, help="Use 16-bit floating point precision.")
    train_config_group.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Use bfloat16 precision (requires PyTorch 1.10+).")
    train_config_group.add_argument("--num_train_epochs", type=int, default=4, help="Number of training epochs.")
    train_config_group.add_argument("--max_seq_length", type=int, default=32768, help="Maximum sequence length for training.")
    train_config_group.add_argument("--deepspeed_config", type=Path, required=False, default=None, help="Path to the deepspeed config file.")
    
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
    lora_config_group.add_argument("--r", type=int, default=16, help="LoRA rank.")
    lora_config_group.add_argument("--alpha", type=int, default=32, help="LoRA alpha.")
    lora_config_group.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout.")
    lora_config_group.add_argument("--neftune_noise_alpha", type=float, default=0, help="Neftune noise alpha for LoRA.")
    
    args = parser.parse_args()

    return args


def args_str(args, run_id):
    model_id = args.model_id_or_path.split("/")[-1]
    dataset_id = args.dataset_id.split("/")[-1]
    shortname = {
        'model_revision': 'rev',
        'sort_data': 'sort',
        'train_batch_size': 'bsz',
        'gradient_checkpointing': 'gc',
        'gradient_accumulation_steps': 'gacc',
        'learning_rate': 'lr',
        'lr_scheduler_type': 'lrs',
        'optim': 'opt',
        'warmup_ratio': 'wr',
        'weight_decay': 'wd',
        'fp16': 'fp16',
        'bf16': 'bf16',
        'num_train_epochs': 'epochs',
        'max_seq_length': 'maxlen',
        'deepspeed_config': 'ds',
        'random_seed': 'seed',
    }
    if args.lora:
        shortname.update({
            'lora': 'lora',
            'r': 'r',
            'alpha': 'alpha',
            'dropout': 'dropout',
            'neftune_noise_alpha': 'neftune',
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
    
    # Step 1. Load model
    model_kwargs = {
        "device_map": None,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
    }
    if 'gemma' in args.model_id_or_path:
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(args.model_id_or_path, **model_kwargs, revision=args.model_revision)

    peft_kwargs = {}
    if args.lora:
        peft_kwargs['peft_config'] = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules="all-linear",
        )
        # for LoRA: https://github.com/pytorch/pytorch/issues/93661
        os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
    else:
        # torch.compile is not compatible with LoRA
        # https://huggingface.co/docs/peft/en/developer_guides/torch_compile?utm_source=chatgpt.com
        pass
        # model = model.compile(model)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    # Step 2. Load and adjust tokenizer
    # Padding side should be right for training and left for inference:
    # https://github.com/huggingface/transformers/issues/34842
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id_or_path,
        padding_side="right", 
        model_max_length=args.max_seq_length,
        revision=args.model_revision,
    )
    logger.info(f"==== Tokenizer Max Length: {tokenizer.model_max_length} ====")

    original_vocab_size = len(tokenizer)
    if not tokenizer.pad_token or tokenizer.pad_token == tokenizer.eos_token:
        # Make sure that pad_token is different from eos_token, because it requires a special treatment in the collator
        # https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#5-when-pad_token-equals-eos_token
        logger.info("==== Adding Special pad token: <|pad|> ====")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    
    deepspeed_config_dict = None
    if args.deepspeed_config:
        with open(args.deepspeed_config, 'r') as f:
            deepspeed_config_dict = json.load(f)
            print(f"Using deepspeed config: {deepspeed_config_dict}")

    prompter = PrompterRegistry.get(args.dataset_id)(with_assistant=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(prompter.special_tokens_to_add().values())})
    logger.info(f"==== Length of Tokenizer: {len(tokenizer)} ====")

    # Resize token embeddings after adding special <|pad|> token
    if len(tokenizer) != original_vocab_size:
        print("Tokenizer size was changed from {} to {}".format(original_vocab_size, len(tokenizer)))
        model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Memory footprint: {model.get_memory_footprint()}")
    
    
    # Step 3. Data preperation
    train_dataset: Dataset = prompter.load_dataset()['train'] # type: ignore
    train_dataset = train_dataset.map(
        lambda ex: prompter.to_chat_template(ex), 
        remove_columns=train_dataset.column_names
    )
    train_dataset = train_dataset.map(
        lambda ex: {'len_text': sum(len(m['content']) for m in ex['messages'] if m['role'] == 'user')}, 
    )
    if args.sort_data:
        train_dataset = train_dataset.sort("len_text", reverse=False)
    eval_dataset: Dataset = prompter.load_dataset()['validation'] # type: ignore
    eval_dataset = eval_dataset.map(lambda ex: prompter.to_chat_template(ex), remove_columns=eval_dataset.column_names)
    

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
        # local_rank                    = args.local_rank,
        output_dir                    = local_output_dir,
        per_device_train_batch_size   = args.train_batch_size,
        per_device_eval_batch_size    = args.eval_batch_size,
        gradient_checkpointing        = args.gradient_checkpointing,
        gradient_checkpointing_kwargs = {'use_reentrant': False},
        gradient_accumulation_steps   = args.gradient_accumulation_steps,
        learning_rate                 = args.learning_rate,
        optim                         = args.optim,
        num_train_epochs              = args.num_train_epochs,
        lr_scheduler_type             = args.lr_scheduler_type,
        warmup_ratio                  = args.warmup_ratio,
        weight_decay                  = args.weight_decay,
        fp16                          = args.fp16,
        bf16                          = args.bf16,
        deepspeed                     = deepspeed_config_dict,
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
        ddp_backend='nccl' if int(os.environ.get("WORLD_SIZE", 1)) > 1 else None,
        ddp_find_unused_parameters=False,  # Enable for MoE
        # ddp_timeout=43200,  # 12 hours
        neftune_noise_alpha=args.neftune_noise_alpha if args.neftune_noise_alpha > 0 else None,
        save_total_limit=args.save_total_limit,
        max_length=args.max_seq_length,
        pad_token=tokenizer.pad_token,
        assistant_only_loss=True,
        packing=False,
        padding_free=False,
        # fsdp=["full_shard", "offload"],
        # adam_beta1=args.adam_beta1,
        # adam_beta2=args.adam_beta2,
        # adam_epsilon=args.adam_epsilon,

        run_name                      = dirname,
        seed                          = args.random_seed,
        report_to                     = ["wandb"],
        remove_unused_columns         = False,
        save_safetensors              = args.lora  # Safetensors can't save tensors that share the same memory
    )

    if args.lora:
        # to supress warning: No label_names provided for model class PeftModelForCausalLM. Since PeftModel hides base models input arguments, if label_names is not given, label_names can't be set automatically within Trainer
        # https://github.com/unslothai/unsloth/issues/1788#issuecomment-2772497747
        training_args.label_names = ["labels"]

    trainer_cls = getattr(t, prompter.trainer_cls_name)
    trainer = trainer_cls(
        model            = model,
        processing_class = tokenizer,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        # eval_packing     = False,
        callbacks        = [MemoryLoggingCallback()],
        **peft_kwargs,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # All ranks must call save_model() for DeepSpeed
    # See Model Checkpointing in https://www.deepspeed.ai/getting-started/
    print(f"[RANK {dist.get_rank()}] Saving model...")
    trainer.save_model(output_dir=local_output_dir)

    # Only rank 0 saves tokenizer and does file operations
    if trainer.is_world_process_zero():
        print(f"[RANK 0] Saving tokenizer...")
        tokenizer.save_pretrained(local_output_dir)
        print("[RANK 0] Local save complete.")

    print(f"[RANK {dist.get_rank()}] About to enter barrier...")
    sys.stdout.flush()  # Force immediate output
    dist.barrier()     # Wait for rank 0 to finish saving locally before other ranks proceed
    print(f"[RANK {dist.get_rank()}] Barrier completed!")

    # Final barrier to ensure the script doesn't exit before rank 0 is done uploading.
    dist.barrier()

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
