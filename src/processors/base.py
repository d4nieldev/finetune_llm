from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Mapping
from transformers import PreTrainedTokenizerBase

from src.utils.chat_types import ChatML

from datasets import load_dataset, Dataset, DatasetDict


class BaseProcessor(ABC):
    dataset_id = None

    def __init__(self, with_assistant: bool = False) -> None:
        self.with_assistant = with_assistant

    def load_dataset(self, **kwargs) -> DatasetDict:
        return load_dataset(self.dataset_id, **kwargs)
    
    def _prepare_train_dataset(
            self, 
            train_dataset: Dataset,
            tokenizer: PreTrainedTokenizerBase,
            num_epochs: int,
            *,
            sort: bool = False, 
            shuffle: bool = False,
            random_seed: int = 42,
            **kwargs
    ) -> Dataset:
        train_dataset = train_dataset.map(
            lambda ex: {'text': tokenizer.apply_chat_template(self.to_chat_template(ex)['messages'], tokenize=False, add_generation_prompt=False)}, 
            remove_columns=train_dataset.column_names,
            desc="Applying chat template"
        )
        if sort:
            # sort by output length
            train_dataset = train_dataset.map(
                lambda ex: {'len_text': sum(len(m['content']) for m in ex['messages'] if m['role'] == 'assistant')}, 
                desc="Sorting by output length"
            )
            train_dataset = train_dataset.sort("len_text", reverse=False)
        if shuffle:
            train_dataset = train_dataset.shuffle(seed=random_seed)
        train_dataset = train_dataset.repeat(num_epochs)
        return train_dataset

    def prepare_dataset(
            self, 
            tokenizer: PreTrainedTokenizerBase,
            num_epochs: int,
            *,
            sort: bool = False, 
            shuffle: bool = False,
            random_seed: int = 42,
            **kwargs
        ) -> tuple[Dataset, Dataset]:
        dataset = self.load_dataset()
        train_dataset = dataset['train']
        train_dataset = self._prepare_train_dataset(
            train_dataset, tokenizer, num_epochs,
            sort=sort, shuffle=shuffle, random_seed=random_seed,
            **kwargs
        )

        eval_dataset = dataset['validation'].map(
            lambda ex: {'text': tokenizer.apply_chat_template(self.to_chat_template(ex)['messages'], tokenize=False, add_generation_prompt=False)}, 
            remove_columns=dataset['validation'].column_names,
            desc="Applying chat template"
        )
        return train_dataset, eval_dataset
    
    @property
    def trainer_cls_name(self) -> str:
        return "SFTTrainer"
    
    def special_tokens_to_add(self) -> dict[str, str]:
        """
        Returns a list of tokens to add to the tokenizer.
        This can be used to ensure that the tokenizer can handle special tokens used in the prompts.
        """
        return {}

    @abstractmethod
    def to_chat_template(self, example: Mapping[str, Any], **kwargs) -> ChatML:
        """
        Convert an example from the dataset to a chat template.

        Args:
            example (Mapping[str, Any]): A dictionary representing a single example from the dataset.

        Returns:
            ChatTemplate: A dictionary representing the chat template.
        """
        raise NotImplementedError
    

class processorRegistry:
    _registry: Dict[str, Type[BaseProcessor]] = {}

    @classmethod
    def register(cls, processor_cls: Type[BaseProcessor]) -> Type[BaseProcessor]:
        dataset_id = processor_cls.dataset_id
        if dataset_id is None:
            raise ValueError(f"processor {processor_cls.__name__} must define a class-level `dataset_id` attribute.")
        if dataset_id in cls._registry:
            raise ValueError(f"Duplicate processor dataset_id '{dataset_id}'")
        cls._registry[dataset_id] = processor_cls
        return processor_cls

    @classmethod
    def get(cls, processor_id: str) -> Type[BaseProcessor]:
        processor_cls = cls._registry.get(processor_id)
        if processor_cls is None:
            raise ValueError(f"processor with dataset_id '{processor_id}' not found. The available processors are: {list(cls._registry.keys())}")
        return processor_cls

    @classmethod
    def all(cls) -> Dict[str, Type[BaseProcessor]]:
        return dict(cls._registry)
