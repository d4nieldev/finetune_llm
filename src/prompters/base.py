from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Mapping

from src.utils.chat_types import ChatTemplate

from datasets import load_dataset, Dataset, DatasetDict


class BasePrompter(ABC):
    def __init__(self, with_assistant: bool = False) -> None:
        self.with_assistant = with_assistant

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        pass
    
    def load_dataset(self):
        return load_dataset(self.dataset_id)
    
    def tokens_to_add(self) -> list[str]:
        """
        Returns a list of tokens to add to the tokenizer.
        This can be used to ensure that the tokenizer can handle special tokens used in the prompts.
        """
        return []

    @abstractmethod
    def to_chat_template(self, example: Mapping[str, Any]) -> ChatTemplate:
        """
        Convert an example from the dataset to a chat template.

        Args:
            example (Mapping[str, Any]): A dictionary representing a single example from the dataset.

        Returns:
            ChatTemplate: A dictionary representing the chat template.
        """
        raise NotImplementedError
    

class PrompterRegistry:
    _registry: Dict[str, Type[BasePrompter]] = {}

    @classmethod
    def register(cls, prompter_cls: Type[BasePrompter]) -> Type[BasePrompter]:
        dataset_id = prompter_cls().dataset_id
        if dataset_id is None:
            raise ValueError(f"Prompter {prompter_cls.__name__} must define a class-level `dataset_id` attribute.")
        if dataset_id in cls._registry:
            raise ValueError(f"Duplicate prompter dataset_id '{dataset_id}'")
        cls._registry[dataset_id] = prompter_cls
        return prompter_cls

    @classmethod
    def get(cls, prompter_id: str) -> Type[BasePrompter]:
        prompter_cls = cls._registry.get(prompter_id)
        if prompter_cls is None:
            raise ValueError(f"Prompter with dataset_id '{prompter_id}' not found. The available prompters are: {list(cls._registry.keys())}")
        return prompter_cls

    @classmethod
    def all(cls) -> Dict[str, Type[BasePrompter]]:
        return dict(cls._registry)
