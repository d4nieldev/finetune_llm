from typing import List, Optional, Callable
from typing_extensions import TypedDict

from src.utils.chat_types import ChatML

import torch
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding


class ModelPrompt(TypedDict):
    prompt: str


class GreedyFirstStep(LogitsProcessor):
    def __init__(self, start_length: int):
        self.start_length = start_length  # prompt length

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[1]
        if cur_len == self.start_length:          # first *generated* position
            idx = scores.argmax(-1, keepdim=True)
            mask = torch.full_like(scores, float("-inf"))
            scores = mask.scatter_(1, idx, 0.0)   # keep only arg-max
        return scores


def to_model_prompt(tokenizer: PreTrainedTokenizerBase, chat_template: ChatML, **kwargs) -> ModelPrompt:
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [msg for msg in chat_template["messages"] if msg['role'] in ['system', 'user']],  # only the system and user
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            **kwargs
        )
    else:
        prompt = "# Instructions:\n" +[m for m in chat_template['messages'] if m['role'] == 'system'][-1]['content']+"\n\n"
        prompt += "# Input:\n" + [m for m in chat_template['messages'] if m['role'] == 'user'][-1]['content']+"\n\n"
        prompt += "# Answer:\n"
    return ModelPrompt(prompt=prompt)


def to_model_inputs_cuda(tokenizer: PreTrainedTokenizerBase, model_prompts: List[ModelPrompt], device) -> BatchEncoding:
    return tokenizer(
        [mp['prompt'] for mp in model_prompts],
        padding=True,
        padding_side="left",  # https://huggingface.co/docs/transformers/llm_tutorial?padding=right+pad#padding-side
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)


def to_model_outputs(
        model_inputs_cuda: BatchEncoding,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 256,
        first_greedy: bool = False,
        **generation_params
    ) -> List[str]:
    if generation_params.get('do_sample') is False:
        generation_params['top_p'] = None
        generation_params['top_k'] = None
    logits_processor = [GreedyFirstStep(model_inputs_cuda['input_ids'].shape[1])] if first_greedy else None
    generation_ids = model.generate(**model_inputs_cuda, max_new_tokens=max_new_tokens, logits_processor=logits_processor, **generation_params)
    generation_ids = generation_ids[:, model_inputs_cuda["input_ids"].shape[1]:]
    return tokenizer.batch_decode(generation_ids, skip_special_tokens=True)


def generate_batch(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_prompts: List[ModelPrompt],
        batch_size: int = 8,
        max_new_tokens: int = 256,
        progress_bar: Optional[tqdm] = None,
        is_valid_output: Optional[Callable[[int, str], bool]] = None,
        max_retries: int = 3,
        first_greedy: bool = False,
        **generation_params
    ) -> List[Optional[str]]:
    if generation_params.get('do_sample') is False and is_valid_output is not None:
        raise ValueError("`retry_pred` is only supported when `do_sample` is True.")
    
    model_outputs: List[Optional[str]] = [None for _ in model_prompts]
    remaining_prompts_indices = list(range(len(model_prompts)))
    tries_count = 0

    while len(remaining_prompts_indices) > 0 and tries_count < max_retries:
        remaining_prompts = [model_prompts[i] for i in remaining_prompts_indices]
        to_remove = []
        
        for i in range(0, len(remaining_prompts), batch_size):
            # Generate model outputs for the current batch
            batch_model_prompts = remaining_prompts[i:i + batch_size]
            batch_model_inputs_cuda = to_model_inputs_cuda(tokenizer, batch_model_prompts, model.device)
            batch_model_outputs = to_model_outputs(batch_model_inputs_cuda, model, tokenizer, max_new_tokens, first_greedy, **generation_params)

            if is_valid_output is not None:
                for j, output in enumerate(batch_model_outputs):
                    original_prompt_idx = remaining_prompts_indices[i + j]
                    if is_valid_output(original_prompt_idx, output):
                        # The output is valid, so we can keep it
                        model_outputs[original_prompt_idx] = output
                        to_remove.append(original_prompt_idx)
                        if progress_bar is not None:
                            progress_bar.update(1)
            elif progress_bar is not None:
                # Save the output for all prompts in the batch, without checking validity
                model_outputs[i: i+batch_size] = batch_model_outputs
                remaining_prompts_indices = remaining_prompts_indices[len(batch_model_prompts):]
                progress_bar.update(len(batch_model_outputs))
        
        for idx in to_remove:
            remaining_prompts_indices.remove(idx)
        tries_count += 1
    return model_outputs


