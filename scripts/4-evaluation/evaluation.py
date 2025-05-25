import os
import sys
import evaluate
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification, TrainingArguments

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from utils import *

def main(model_path, test_path):
    base_model = "microsoft/codebert-base"

    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)

    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        print("Please ensure the model path is correct and the model files are accessible.")
        return

    tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)

    if not os.path.exists(test_path):
        print(f"Error: Test dataset file not found at '{test_path}'")
        return

    try:
        dataset, label_list = construct_dataset_wo_train(model.config.to_dict(), test_path=test_path)

    except Exception as e:
        print(f"Error constructing dataset from '{test_path}': {e}")
        return

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load("seqeval")
    tokenized_datasets = dataset.map(lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True)
    testset = tokenized_datasets['test']

    lb2id = model.config.to_dict()['label2id']
    id2lb = model.config.to_dict()['id2label']

    def compute_metrics_wrapper(p):
        return compute_metrics_entity(p, id2lb, lb2id, p_match=True)

    trainer = Trainer(
        model=model,
        eval_dataset=testset,
        args=TrainingArguments(output_dir=".", per_device_eval_batch_size = 100, report_to="none"),
        data_collator=data_collator,
        compute_metrics = compute_metrics_wrapper
    )

    eval_results = trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SDLog model for sensitive attribute detection.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained or fine-tuned SDLog model.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to the test dataset file (e.g., for 'all' or 'net' attributes).")

    args = parser.parse_args()
    main(args.model_path, args.test_path)