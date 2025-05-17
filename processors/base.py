from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Mapping

from custom_types import ChatTemplate


class BaseProcessor(ABC):    
    @abstractmethod
    def to_chat_template(self, example: Mapping[str, Any], train: bool = False) -> ChatTemplate:
        """
        Convert an example from the dataset to a chat template.

        Args:
            example (Mapping[str, Any]): A dictionary representing a single example from the dataset.

        Returns:
            ChatTemplate: A dictionary representing the chat template.
        """
        raise NotImplementedError
    

class ProcessorRegistry:
    _registry: Dict[str, Type[BaseProcessor]] = {}

    @classmethod
    def register(cls, processor_cls: Type[BaseProcessor]) -> Type[BaseProcessor]:
        dataset_id = getattr(processor_cls, "dataset_id", None)
        if dataset_id is None:
            raise ValueError(f"Processor {processor_cls.__name__} must define a class-level `dataset_id` attribute.")
        if dataset_id in cls._registry:
            raise ValueError(f"Duplicate processor dataset_id '{dataset_id}'")
        cls._registry[dataset_id] = processor_cls
        return processor_cls

    @classmethod
    def get(cls, processor_id: str) -> Type[BaseProcessor]:
        processor_cls = cls._registry.get(processor_id)
        if processor_cls is None:
            raise ValueError(f"Processor with dataset_id '{processor_id}' not found. The available processors are: {list(cls._registry.keys())}")
        return processor_cls

    @classmethod
    def all(cls) -> Dict[str, Type[BaseProcessor]]:
        return dict(cls._registry)
