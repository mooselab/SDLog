import os
import sys
import evaluate
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--train_path", type=str, default="./dataset/train_dataset.txt", help="Path to the training dataset.")
parser.add_argument("--test_path", type=str, default="./dataset/test_dataset.txt", help="Path to the testing dataset.")
parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for saving results.")
parser.add_argument("--eval_strategy", type=str, choices=["no", "steps", "epoch"], default="steps", help="Evaluation strategy to use.")
parser.add_argument("--eval_steps", type=int, default=300, help="Number of update steps between two evaluations when using evaluation strategy 'steps'.")
parser.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch"], default="steps", help="Save strategy to use.")
parser.add_argument("--save_steps", type=int, default=300, help="Number of steps between saving checkpoints.")
parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for saving logs.")
parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device for training.")
parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per device for evaluation.")
parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps between logging outputs.")
parser.add_argument("--load_best_model_at_end", action="store_true", help="Whether to load the best model at the end of training.")
parser.add_argument("--model_save_path", type=str, default="./ner_model", help="Directory to save the trained model.")
parser.add_argument("--cpu", action="store_true", help="Whether to only use CPU")
parser.add_argument("--base_model", type=str, default="microsoft/codebert-base", help="The base BERT model to use.")
parser.add_argument("--local_model_path", type=str, default=None, help="Path to a locally saved model. If specified, the model will be loaded from this path instead of downloading from the hub.")

def main():
    args = parser.parse_args()
    model_name = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    if args.local_model_path:
        model = AutoModelForTokenClassification.from_pretrained(args.local_model_path)
        model_config = model.config.to_dict()
        dataset, label_list = construct_dataset_with_prior(model_config, train_path = args.train_path, test_path = args.test_path)
    else:
        dataset, label_list = construct_dataset(train_path = args.train_path, test_path = args.test_path)
        label_to_id = {label: i for i, label in enumerate(label_list)}
        id_to_label = {i: label for i, label in enumerate(label_list)}
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), id2label = id_to_label, label2id = label_to_id)

    tokenized_datasets = dataset.map(lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        no_cuda=args.cpu,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        save_total_limit=3,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=2)

        true_labels = [
            [label_list[l] for l in label if l != -100]
            for label in labels
        ]
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["validation"],
        eval_dataset=tokenized_datasets["test"],
        processing_class = tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(args.model_save_path)
    return


if __name__ == "__main__":
    main()