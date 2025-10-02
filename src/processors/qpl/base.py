from enum import StrEnum
from typing import Any, Mapping
import numpy as np

from datasets import Dataset

from src.processors.base import BaseProcessor, chatml_to_dataset
import src.utils.paths as p
from src.utils.schema import DBSchema, NoiseStrategy, SchemaRepresentation
from src.utils.chat_types import ChatML


class NoiseSchemaScheduler(StrEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"

class QPLProcessor(BaseProcessor):
    dataset_id = None

    def __init__(
            self, 
            schema_representation: SchemaRepresentation = SchemaRepresentation.M_SCHEMA, 
            schema_noise_strategy: NoiseStrategy = NoiseStrategy.DEPTH,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.schema_representation = schema_representation
        self.noise_strategy = schema_noise_strategy

        self.__db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH, apply_lower=False)

    
    def special_tokens_to_add(self) -> dict[str, str]:
        if self.schema_representation == "m_schema":
            return {
                "m_schema_open": "【",
                "m_schema_close": "】"
            }
        return super().special_tokens_to_add()
    
    def to_chat_template(self, example: Mapping[str, Any], *, noise: float = 1.0, **kwargs) -> ChatML:
        raise NotImplementedError

    def _prepare_train_dataset(
            self,
            train_dataset,
            tokenizer,
            num_epochs,
            *, 
            sort: bool = False,
            shuffle: bool = False,
            random_seed: int = 42,
            noise_schema_sched: NoiseSchemaScheduler | None = None,
            min_noise: float = 1.0,
            max_noise: float = 1.0,
            **kwargs
        ) -> Dataset:
        if sort and not shuffle:
            # simple schemas first, break ties with same database id, then shortest output first
            train_dataset = train_dataset.map(
                lambda ex: {
                    'schema_length': len(getattr(self.__db_schemas[ex['db_id']], str(self.schema_representation))()), 
                    'output_length': -len([msg['content'] for msg in self.to_chat_template(ex)['messages'] if msg['role'] == 'assistant'][0]),
                },
                desc="Sorting by schema complexity"
            ).sort(["schema_length", "db_id", "output_length"])
        elif sort and shuffle:
            # keep same databases together, but shuffle within each database
            train_dataset = train_dataset.shuffle(seed=random_seed).sort('db_id')
        elif shuffle:
            # just shuffle
            train_dataset = train_dataset.shuffle(seed=random_seed)

        if noise_schema_sched is None:
            train_dataset = train_dataset.map(
                lambda ex: chatml_to_dataset(self.to_chat_template(ex), tokenizer), 
                remove_columns=train_dataset.column_names,
                desc="Applying chat template"
            )
            return train_dataset.repeat(num_epochs)

        if noise_schema_sched == NoiseSchemaScheduler.LINEAR:
            noises = np.linspace(min_noise, max_noise, num_epochs * len(train_dataset))
        elif noise_schema_sched == NoiseSchemaScheduler.COSINE:
            noises = np.sin(np.linspace(0, np.pi/2, num_epochs * len(train_dataset)))
        elif noise_schema_sched == NoiseSchemaScheduler.EXPONENTIAL:
            noises = np.logspace(np.log10(min_noise + 1e-5), np.log10(max_noise + 1e-5), num_epochs * len(train_dataset)) - 1e-5
        else:
            raise ValueError(f"Unknown noise schema schedule: {noise_schema_sched}. Known schedules are: {list(NoiseSchemaScheduler)}.")
        train_dataset = train_dataset.repeat(num_epochs)
        train_dataset = train_dataset.map(
            lambda ex, i: chatml_to_dataset(self.to_chat_template(ex, noise=float(noises[i])), tokenizer), 
            remove_columns=train_dataset.column_names,
            with_indices=True,
            desc="Applying schema noise & chat template"
        )
        return train_dataset
    
    def _get_schema_str(
        self, 
        db_id: str,
        link_table_cols: dict[str, set[str]] | None = None,
        noise: float = 0.0,
    ):
        # problematic ids
        if db_id == "car_11":
            db_id = "car_1"

        db_schema = self.__db_schemas[db_id]
        if link_table_cols is not None:
            db_schema = db_schema.linked(
                table_cols=link_table_cols,
                noise=noise,
                noise_strategy=self.noise_strategy
            )

        if self.schema_representation == SchemaRepresentation.DDL:
            return db_schema.ddl()
        elif self.schema_representation == SchemaRepresentation.M_SCHEMA:
            return db_schema.m_schema()
        else:
            raise ValueError(f"Unknown representation: {self.schema_representation}. Use 'ddl' or 'm_schema'.")
