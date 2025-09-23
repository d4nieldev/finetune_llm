import re
import json
import argparse
import logging as log

import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


from src.utils.paths import TRAINED_MODELS_DIR
import src.utils.paths as p
from src.prompters import QPLDecomposerCotPrompter, QPLDecomposerPrompter
from src.utils.generation import to_model_prompt, generate_batch
from src.experiments.qpl.text_to_qpl import get_generation_params, GenerationMode

log.basicConfig(
    level=log.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def parse_args():
    parser = argparse.ArgumentParser(description="QPL Decomposer Ablation Experiment")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Max new tokens to generate")
    parser.add_argument("--no_cot", action='store_true', help="Do not use Chain of Thought prompting")
    parser.add_argument("--generation_mode", type=GenerationMode, default=GenerationMode.SAMPLING, help="Generation mode")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for generation")
    
    args = parser.parse_args()

    # args.model_path = TRAINED_MODELS_DIR / args.model_dir # TODO: enable
    args.model_path = args.model_dir # TODO: disable
    args.output_dir = p.ABLATION_DECOMPOSER_OUTPUT_DIR / args.model_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args

if __name__ == "__main__":
    args = parse_args()

    # Load and process data
    if args.no_cot:
        prompter = QPLDecomposerPrompter(schema_representation="m_schema", with_assistant=True)
    else:
        prompter = QPLDecomposerCotPrompter(schema_representation="m_schema", with_assistant=True)

    test_dataset = list(prompter.load_dataset()['validation'])
    chat_templates = list(map(prompter.to_chat_template, test_dataset))

    # Load model & tokenizer
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()
    # model = model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # TODO: make general
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = 16*1024,
        load_in_4bit = True,
    )

    prompts = list(map(lambda ct: to_model_prompt(tokenizer, ct), chat_templates))

    # Decompose questions
    predictions = generate_batch(
        model=model,
        tokenizer=tokenizer,
        model_prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        progress_bar=tqdm(total=len(prompts), desc="Decomposing"),
        max_retries=args.max_retries,
        **get_generation_params(args.generation_mode)
    )

    op_correct = 0
    sum_similarity = 0
    sentences_count = 0

    op_to_id = {
        'aggregate': 0,
        'except': 1,
        'filter': 2,
        'intersect': 3,
        'join': 4,
        'scan': 5,
        'sort': 6,
        'topsort': 7,
        'union': 8,
        'other': 9
    }
    id_to_op = {v: k for k, v in op_to_id.items()}

    emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    output_pattern = re.compile(r"(?P<reasoning><think>.*?</think>)?\s*(?P<answer>.*)", re.DOTALL)
    y_pred = []
    y_true = []

    predictions_comparison = []
    for pred, chat_template in tqdm(zip(predictions, chat_templates), desc="Evaluating", total=len(predictions)):
        if pred is None:
            log.error("No prediction for chat template:", chat_template)
            continue

        # save prediction vs. gold
        predictions_comparison.append({
            "input": [ct for ct in chat_template['messages'] if ct['role'] in ['system', 'user']],
            "pred": pred,
            "gold": chat_template['messages'][-1]['content']
        })

        # process data
        gold = chat_template['messages'][-1]['content']
        if not (gold_match := output_pattern.match(gold)):
            log.error(f"Invalid gold format:\n\n{gold}\n\n---------------------------")
            continue
        if not (pred_match := output_pattern.match(pred)):
            log.warning(f"Invalid prediction format:\n\n{pred}\n\n---------------------------")
            continue

        pred_lines = pred_match.group("answer").split("\n")
        gold_lines = gold_match.group("answer").split("\n")

        # operator classification
        pred_op_id = op_to_id.get(pred_lines[0].lower(), op_to_id["other"])
        gold_op_id = op_to_id.get(gold_lines[0].lower(), op_to_id["other"])
        y_pred.append(pred_op_id)
        y_true.append(gold_op_id)

        # sentence similarity
        if pred_op_id == gold_op_id:
            op_correct += 1

            model_sentences = pred_lines[1:]
            label_sentences = gold_lines[1:]

            sentences_count += len(label_sentences)

            if len(model_sentences) != len(label_sentences):
                print("======================")
                print(pred)
                print("----")
                print(gold)
                print("======================")
            else:
                all_sentences = model_sentences + label_sentences
                embeddings = emb_model.encode(all_sentences, show_progress_bar=False)
                similarity_matrix = embeddings @ embeddings.T
                if len(model_sentences) == 0:
                    similarity = 0
                elif len(model_sentences) == 1:
                    similarity = similarity_matrix[0][1]
                else:
                    similarity = max(
                        similarity_matrix[0,2] + similarity_matrix[1,3],
                        similarity_matrix[0,3] + similarity_matrix[1,2]
                    ) / 2
                sum_similarity += similarity

    # Generate a classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Print nicely
    df = pd.DataFrame(report).transpose()
    df.index = df.index.map(lambda x: id_to_op[int(x)] if x.isdigit() else x)
    df['support'] = df['support'].astype(int)
    print(df)

    cm = confusion_matrix(y_true, y_pred, labels=list(op_to_id.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(op_to_id.keys()))
    disp.plot(cmap=plt.cm.Blues)

    stats = {
        'accuracy': op_correct / len(test_dataset),
        'sentence_similarity': float(sum_similarity / op_correct) if op_correct > 0 else 0,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    plt.xticks(rotation=45)
    plt.savefig(args.output_dir / "confusion_matrix.png")

    with open(args.output_dir / "predictions.json", 'w') as f:
        json.dump(predictions_comparison, f, indent=2)

    with open(args.output_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)