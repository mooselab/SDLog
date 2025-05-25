import os
import sys
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from utils import *

def main(model_path, log_path):
    # Load the dataset
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at '{log_path}'")
        return
    
    # Load the SDLog model
    base_model = "microsoft/codebert-base"

    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)
    token_classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="first")

    output_path = os.path.join(project_root, "target_dataset", "4-anonymized_datasets", "main") 

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_raw_file = os.path.join(output_path, "dataset.txt") 
    output_anonymized_file = os.path.join(output_path, "dataset_anonymized.txt") 

    # Open files for writing the raw and anonymized logs
    with open(output_raw_file, "w", encoding="utf-8") as raw_file, \
         open(output_anonymized_file, "w", encoding="utf-8") as anonymized_file, \
         open(log_path, 'r') as test_file:

        logs = test_file.readlines()
        
        # Process the logs
        for i, log_entry_with_newline in enumerate(logs):
            log_entry = log_entry_with_newline.rstrip('\n')
            raw_file.write(log_entry_with_newline)

            ner_results = token_classifier(log_entry)
            anonymized_log = log_entry

            if ner_results:
                sorted_results = sorted(ner_results, key=lambda x: x['start'], reverse=True)

                # Replace sensitive attributes with their anonymized versions
                for result in sorted_results:
                    start_index = result['start']
                    end_index = result['end']
                    entity_group_name = 'B_' + result['entity_group']

                    # Perform replacement on the stripped log
                    anonymized_log = anonymized_log[:start_index] + entity_group_name + anonymized_log[end_index:]

                anonymized_file.write(anonymized_log + "\n")
            else:
                # If no sensitive attributes, anonymized log is the same as the original stripped log
                anonymized_file.write(anonymized_log + "\n")

    print(f"Processing complete. Successfully created anonymized logs at: {output_anonymized_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check predictions of an SDLog model on raw log entries.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or Hugging Face ID to the pre-trained or fine-tuned SDLog model.")
    parser.add_argument("--log_path", type=str, required=True,
                        help="Path to the test dataset file (e.g., for 'all' or 'net' attributes).")
    
    args = parser.parse_args()
    main(args.model_path, args.log_path)