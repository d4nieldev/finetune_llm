import re
from typing import Optional, Union

import torch
import pyodbc
from trl import SFTTrainer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from src.experiments.qpl.text_to_qpl import text_to_qpl, GenerationMode
from src.experiments.qpl.validate_qpl import compare_qpl_sql
from src.prompters import QPLDecomposerPrompter, QPLCompleterPrompter, QPLDecomposerCotPrompter, QPLCompleterCotPrompter


class RecursiveEvalSFTTrainer(SFTTrainer):
    def __init__(self, cot: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cot = cot

        self._evaluation_metrics = {}
        self.args.per_device_eval_batch_size = len(self.eval_dataset)
        self._conn = pyodbc.connect((
            'Driver={ODBC Driver 18 for SQL Server};'
            'Server=tcp:spider-sql.database.windows.net,1433;'
            'Database=test;'
            'Uid=iloveqpl;'
            'Pwd=P4$$w0rd!;'
            'Encrypt=yes;'
            'TrustServerCertificate=no;'
            'Connection Timeout=30;'
        ), autocommit=True)
    

    @torch.no_grad()
    def prediction_step(self, model: PreTrainedModel, inputs: dict[str, torch.Tensor], *args, **kwargs):
        model.eval()
        if self.processing_class is None:
            raise ValueError("Tokenizer is not initialized")
        tokenizer: PreTrainedTokenizer = self.processing_class  # type: ignore

        # Step 1. Extract db_id and question from input texts
        input_regex = re.compile(r'【DB_ID】 ([^\n]+).*\[Question\]\: ([^\n]+)', re.DOTALL)
        input_texts = tokenizer.batch_decode([inp[torch.where((inputs['labels'][i] == -100) & (inp != tokenizer.pad_token_id))[0]] for i, inp in enumerate(inputs['input_ids'])])
        matches = [re.search(input_regex, input_text) for input_text in input_texts]
        if not all(matches):
            raise ValueError("Input text does not match expected format")
        examples = [
            {
                'db_id': match.group(1),
                'question': match.group(2)
            }
            for match in matches
        ]

        # Step 2. Predict QPL using decomposition and reconstruction
        results = text_to_qpl(
            examples=examples,
            decomposer_model=model,
            decomposer_tokenizer=tokenizer,
            completer_model=model,
            completer_tokenizer=tokenizer,
            decomposer_prompter=QPLDecomposerPrompter(with_assistant=False) if not self.cot else QPLDecomposerCotPrompter(with_assistant=False),
            completer_prompter=QPLCompleterPrompter(with_assistant=False) if not self.cot else QPLCompleterCotPrompter(with_assistant=False),
            mode=GenerationMode.SAMPLING
        )

        # Step 3. Extract gold SQL queries
        gold_sql_regex = re.compile(r'</think>\n(.*)', re.DOTALL)
        labels_texts= tokenizer.batch_decode([lbl[torch.where(lbl != -100)[0]] for lbl in inputs['labels']], skip_special_tokens=True)
        gold_sqls = [re.search(gold_sql_regex, labels_text).group(1).strip() for labels_text in labels_texts]

        # Step 4. Connect to database
        cursor = self._conn.cursor()

        # Step 5. Compute statistics
        acc = 0
        errs = 0
        for pred, gold in zip(results, gold_sqls):
            same, err = compare_qpl_sql(pred['pred_qpl'], gold, pred['db_id'], cursor)
            acc += 1 if same else 0
            errs += 1 if err else 0
        
        self._evaluation_metrics = {
            'eval_execution_accuracy': acc / len(results),
            'eval_error_rate': errs / len(results)
        }

        return (None, None, None)
    

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        if self._evaluation_metrics:
            metrics.update(self._evaluation_metrics)

        self.log(metrics)

        return metrics
