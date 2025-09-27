from typing import Callable
from datasets import interleave_datasets, DatasetDict
import numpy as np

from src.utils.chat_types import ChatML
from src.processors.qpl.base import QPLProcessor
from src.processors.base import processorRegistry
from src.processors.qpl.decomposer_cot import QPLDecomposerCotProcessor
from src.processors.qpl.completer_cot import QPLCompleterCotProcessor

from datasets import load_dataset


@processorRegistry.register
class QPLMergedCotProcessor(QPLProcessor):
    dataset_id = "d4nieldev/qpl-merged-cot-ds"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.with_assistant:
            raise ValueError("Merged dataset requires with_assistant=True")
        self.decomposer_cot_processor = QPLDecomposerCotProcessor(*args, **kwargs)
        self.completer_cot_processor = QPLCompleterCotProcessor(*args, **kwargs)

    @property
    def trainer_cls_name(self) -> str:
        return "RecursiveEvalSFTTrainer"

    def load_dataset(self):
        decomposer_ds = self.decomposer_cot_processor.load_dataset("balanced")
        decomposer_ds = decomposer_ds.map(lambda _: {"task": "decomposer"})
        completer_ds = self.completer_cot_processor.load_dataset("balanced")
        completer_ds = completer_ds.map(lambda _: {"task": "completer"})
        
        # convert to decomposer dataset format for evaluation during training
        development_ds = load_dataset('d4nieldev/nl2qpl-ds', split='development')
        development_ds = development_ds.map(lambda x: {
            "task": "development", 
            "db_id": x['qpl'].split(' | ')[0], 
            "cot": "", 
            "sub_question_1": None,
            "sub_question_2": None,
            "op": x['query'] # gold sql
        })

        return DatasetDict({
            'train': interleave_datasets([decomposer_ds['train'], completer_ds['train']], stopping_strategy="all_exhausted"), # will oversample the smaller dataset (decomposer)
            'validation': development_ds 
        })

    def to_chat_template(self, example) -> ChatML:
        if example['task'] == 'decomposer':
            return self.decomposer_cot_processor.to_chat_template(example)
        elif example['task'] == 'completer':
            return self.completer_cot_processor.to_chat_template(example)
        elif example['task'] == 'development':
            return self.decomposer_cot_processor.to_chat_template(example)
        else:
            raise ValueError(f"Unknown task type: {example['task']}")
