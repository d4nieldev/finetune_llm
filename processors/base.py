from typing import Dict, Literal, Any, List, Union, Optional
from abc import ABC, abstractmethod
from transformers.trainer_callback import TrainerCallback
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from peft import PeftModel, PeftMixedModel


class BaseProcessor(ABC):
    """
    Base class for all processors. This class is responsible for processing the dataset and preparing it for training.
    It handles prompt creation, tokenization, padding, and callbacks during the training.
    """
    def __init__(
        self,
        *,
        model: Union[PreTrainedModel, PeftModel, PeftMixedModel],
        tokenizer: PreTrainedTokenizer,
        dataset: Union[DatasetDict, IterableDatasetDict],
        max_input_tokens: Optional[int],
        max_output_tokens: Optional[int],
    ):
        """
        Args:
            model (PreTrainedModel): The model to be trained.
            tokenizer (PreTrainedTokenizer): The tokenizer used for the model.
            dataset (Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]): The dataset to be used - should have a train/validation split.
            max_input_tokens (Optional[int]): The maximum number of input tokens.
            max_output_tokens (Optional[int]): The maximum number of output tokens.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._dataset = dataset
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self.__autoset_input_output_tokens()
        
        self._processed_train = None
        self._tokenized_validation = None
        
        
    @property
    def processed_train(self):
        """
        Returns the processed training dataset.
        """
        if self._processed_train:
            return self._processed_train
        
        self._processed_train = self._dataset["train"].map(
            self._preprocess_train, batched=True,
            remove_columns=self._dataset["train"].column_names
        )
        return self._processed_train


    @property
    def tokenized_validation(self):
        """
        Returns the processed validation dataset.
        """
        if self._tokenized_validation:
            return self._tokenized_validation
        self._tokenized_validation = self._dataset["validation"].map(
            self._preprocess_eval, batched=True,
            remove_columns=self._dataset["validation"].column_names
        )
        return self._tokenized_validation
    
    
    def __autoset_input_output_tokens(self) -> None:
        """
        Automatically sets the maximum input and output tokens based on the dataset.
        """
        max_input = 0
        max_output = 0
        
        for dataset in self._dataset.values():
            for row in dataset:
                parsed_row = self.process_row(row)
                input = self._tokenizer(parsed_row['prompt'], truncation=True, max_length=self._max_input_tokens)
                output = self._tokenizer(parsed_row['response'], truncation=True, max_length=self._max_output_tokens)
                max_input = max(max_input, len(input.input_ids))
                max_output = max(max_output, len(output.input_ids))
        
        if self._max_input_tokens is None:
            self._max_input_tokens = int(max_input * 1.2)
            print(f"Auto-set max input tokens to {self._max_input_tokens}.")
        if self._max_output_tokens is None:
            self._max_output_tokens = int(max_output * 1.2)
            print(f"Auto-set max output tokens to {self._max_output_tokens}.")
        
        
    def _preprocess_train(
        self, 
        examples: Dict[str, List[Any]], 
    ):
        """
        Preprocesses the training dataset by tokenizing the input and output sequences.
        """
        input_ids, attn_mask, labels = [], [], []
        rows = [dict(zip(examples, t)) for t in zip(*examples.values())]
        for row in rows:
            parsed_row = self.process_row(row)
            p = self._tokenizer(parsed_row['prompt'], truncation=True, max_length=self._max_input_tokens)
            r = self._tokenizer(parsed_row['response'], truncation=True, max_length=self._max_output_tokens)

            seq_ids = p.input_ids + r.input_ids
            mask    = [1] * len(seq_ids)  # Attend to all input and output tokens
            lab     = [-100] * len(p.input_ids) + r.input_ids  # Loss will not be computed for input tokens

            # Right pad the sequence to the maximum length
            pad_len = self._max_input_tokens + self._max_output_tokens - len(seq_ids)
            seq_ids += [self._tokenizer.pad_token_id] * pad_len
            mask    += [0] * pad_len  # Do not attend to padding tokens
            lab     += [-100] * pad_len  # Loss will not be computed for padding tokens

            input_ids.append(seq_ids)
            attn_mask.append(mask)
            labels.append(lab)

        return {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "labels":         labels,
        }
        
    def _preprocess_eval(
        self, 
        examples: Dict[str, List[Any]], 
    ):
        """
        Preprocesses the evaluation dataset by tokenizing the input and output sequences.
        """
        rows = [dict(zip(examples, t)) for t in zip(*examples.values())]
        parsed_rows = [self.process_row(row) for row in rows]
        prompts = [row['prompt'] for row in parsed_rows]
        inputs = self._tokenizer(prompts, truncation=True, padding="max_length", max_length=self._max_input_tokens)
        inputs["labels"] = [row['response'] for row in parsed_rows]
        return inputs
    
    
    @abstractmethod
    def process_row(self, row: dict[str, Any]) -> Dict[Literal["prompt", "response"], str]:
        """
        Processes a single row of the dataset. This method should be implemented by subclasses.
        Args:
            row (dict[str, Any]): A single row of the dataset.
        Returns:
            Dict[Literal["prompt", "response"], str]: A dictionary containing the prompt and response for the given row.
        """
        pass
    
    @abstractmethod
    def get_callbacks(self) -> List[TrainerCallback]:
        """
        Returns a list of callbacks to be used during training. This method should be implemented by subclasses.
        Returns:
            List[TrainerCallback]: A list of callbacks to be used during training.
        """
        pass
        