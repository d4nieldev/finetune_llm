import torch
import wandb
from transformers.trainer_callback import TrainerCallback


class MemoryLoggingCallback(TrainerCallback):
    """
    A custom callback for logging memory usage during training.
    """
    def on_step_end(self, args, state, control, **kwargs):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        wandb.log({
            "memory_allocated_bytes": allocated,
            "memory_reserved_bytes": reserved,
            "step": state.global_step
        })