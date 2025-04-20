from typing import List, TypeVar
from transformers.trainer_callback import TrainerCallback
import evaluate

from callbacks import ClassEvalCallback
from .base import BaseProcessor


CustomCallback = TypeVar("CustomCallback", bound=TrainerCallback)


class EmotionProcessor(BaseProcessor):
    def __init__(
        self, 
        *, 
        metrics: List[str] = ['accuracy'],
        eval_steps: int = 8000,
        eval_batch_size: int = 10,
        **kwargs
    ):
        """
        Args:
            metrics (List[str]): List of metrics to evaluate on. Default is ['accuracy'].
            eval_steps (int): The number of steps between evaluations. Default is 8000.
            eval_batch_size (int): Batch size for evaluation. Default is 10.
            **kwargs: Additional arguments passed to the BaseProcessor.
        """
        self._labels_list = ["sadness","joy","love","anger","fear","surprise"]
        
        super().__init__(**kwargs)
        
        self._metrics = metrics
        self._eval_steps = eval_steps
        self._eval_batch_size = eval_batch_size
        


    def process_row(self, row):
        prompt = f"Below is a piece of text. Classify it into one of: {', '.join(self._labels_list)}.\n\n"
        prompt += f"\"{row['text']}\"\n\nThe emotion in the above text is: "
        response = f"{self._labels_list[row['label']]}"
        return {
            "prompt": prompt,
            "response": response,
        }
        

    def get_callbacks(self) -> List[CustomCallback]:
        return [
            ClassEvalCallback(
                eval_ds=self.tokenized_validation,
                tokenizer=self._tokenizer,
                model=self._model,
                metrics=[evaluate.load(metric) for metric in self._metrics],
                eval_steps=self._eval_steps,
                batch_size=self._eval_batch_size,
                labels_list=self._labels_list,
                max_out_tokens=self._max_output_tokens
            )
        ]
