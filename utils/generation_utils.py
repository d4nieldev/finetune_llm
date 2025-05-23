from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict

from custom_types import ChatTemplate

from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding

class ModelPrompt(TypedDict):
    prompt: str


def to_model_prompt(tokenizer: PreTrainedTokenizerBase, chat_template: ChatTemplate) -> ModelPrompt:
    prompt = tokenizer.apply_chat_template(
        [msg for msg in chat_template["messages"] if msg['role'] in ['system', 'user']],  # only the system and user
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )
    return ModelPrompt(prompt=prompt)


def to_model_inputs_cuda(tokenizer: PreTrainedTokenizerBase, model_prompts: List[ModelPrompt]) -> BatchEncoding:
    return tokenizer(
        [mp['prompt'] for mp in model_prompts],
        padding=True,
        padding_side="left",  # https://huggingface.co/docs/transformers/llm_tutorial?padding=right+pad#padding-side
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")


def to_model_outputs(
        model_inputs_cuda: BatchEncoding,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 256,
        **generation_params
    ) -> List[str]:
    if not generation_params['do_sample']:
        generation_params['top_p'] = None
        generation_params['top_k'] = None
    generation_ids = model.generate(**model_inputs_cuda, max_new_tokens=max_new_tokens, **generation_params)
    generation_ids = generation_ids[:, model_inputs_cuda["input_ids"].shape[1]:]
    return tokenizer.batch_decode(generation_ids, skip_special_tokens=True)


def generate_batch(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_prompts: List[ModelPrompt],
        batch_size: int = 8,
        max_new_tokens: int = 256,
        progress_bar: Optional[tqdm] = None,
        **generation_params
    ) -> List[str]:
    model_outputs = []
    for i in range(0, len(model_prompts), batch_size):
        batch_model_prompts = model_prompts[i:i + batch_size]
        batch_model_inputs_cuda = to_model_inputs_cuda(tokenizer, batch_model_prompts)
        batch_model_outputs = to_model_outputs(batch_model_inputs_cuda, model, tokenizer, max_new_tokens, **generation_params)
        model_outputs.extend(batch_model_outputs)
        if progress_bar is not None:
            progress_bar.update(len(batch_model_prompts))
    return model_outputs


