import re
from typing import Union, List

import torch
from tqdm import tqdm
from transformers.trainer_callback import TrainerCallback
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel, PeftMixedModel
from datasets import DatasetDict, IterableDatasetDict
from evaluate import EvaluationModule


class ClassEvalCallback(TrainerCallback):
    """
    A custom callback for evaluating a model during training for a classification task.
    """
    def __init__(
        self, 
        eval_ds: Union[DatasetDict, IterableDatasetDict], 
        model: Union[PreTrainedModel, PeftModel, PeftMixedModel], 
        tokenizer: PreTrainedTokenizer, 
        metrics: List[EvaluationModule],
        eval_steps: int, 
        batch_size: int,
        labels_list: List[str],
        max_out_tokens: int = 16
    ):
        """
        Args:
            eval_ds (Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]): The dataset to evaluate on.
            model (PreTrainedModel): The model to evaluate.
            tokenizer (PreTrainedTokenizer): The tokenizer used for the model.
            metrics (List[EvaluationModule]): A list of evaluation metrics to compute.
            eval_steps (int): The number of steps between evaluations.
            batch_size (int): The batch size for evaluation.
            labels_list (List[str]): A list of labels for classification.
            max_out_tokens (int): The maximum number of output tokens to generate during evaluation.
        """
        super().__init__()

        self.eval_ds    = eval_ds
        self.model      = model
        self.tokenizer  = tokenizer
        self.metrics    = metrics
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self.labels_list = labels_list
        self.max_out_tokens = max_out_tokens
        
        self._num_rows = len(self.eval_ds["input_ids"])


    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and state.global_step % self.eval_steps == 0:
            progress_bar = tqdm(total=self._num_rows, desc="Eval")
            predictions = []
            
            for i in range(0, self._num_rows, self.batch_size):
                # build batch
                input_ids = torch.tensor(self.eval_ds["input_ids"][i:i+self.batch_size]).to(args.device)
                attention_mask = torch.tensor(self.eval_ds["attention_mask"][i:i+self.batch_size]).to(args.device)

                # generate
                gen = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_out_tokens,
                    do_sample=False,
                )
                
                # decode
                raw_texts = self.tokenizer.batch_decode(gen.cpu(), skip_special_tokens=True)
                
                # get labels
                inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                texts = [txt[len(inp):].strip() for txt, inp in zip(raw_texts, inputs)]
                preds = [self._extract_label(txt) for txt in texts]
                
                # accumulate
                predictions.extend(preds)
                progress_bar.update(len(input_ids))
                
                # clean up
                del input_ids, attention_mask, gen
                torch.cuda.empty_cache()
            
            progress_bar.close()

            # true labels
            references = [self._extract_label(lbl_txt) for lbl_txt in self.eval_ds["labels"]]
            res = {}
            for metric in self.metrics:
                res = metric.compute(predictions=predictions, references=references)
            print(f"\nStep {state.global_step} Eval →", res)
            
            
    def _extract_label(self, txt: str) -> int:
        match = re.search(r"(\w+)", txt)
        if match:
            match = match.group(1).lower()
            if match in self.labels_list:
                return self.labels_list.index(match)
        return -1