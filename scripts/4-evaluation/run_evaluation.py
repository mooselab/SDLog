import argparse
import subprocess
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Define model paths ---
# Main models from Hugging Face
MAIN_MODEL_ALL_PATH = 'LogSensitiveResearcher/SDLog_main' 
MAIN_MODEL_NET_PATH = 'LogSensitiveResearcher/SDLog_net'
# Fine-tuned models
FINE_TUNED_MODEL_ALL_PATH = os.path.join(script_dir, "..", "..", "fine_tuned_model") 
FINE_TUNED_MODEL_NET_PATH = os.path.join(script_dir, "..", "..", "fine_tuned_model_net")

# --- Define test dataset paths ---
MAIN_DATASET_ALL_PATH = os.path.join(script_dir, "..", "..", "target_dataset", "2-processed_datasets", "main", "dataset.txt")
MAIN_DATASET_NET_PATH = os.path.join(script_dir, "..", "..", "target_dataset", "2-processed_datasets", "net", "dataset.txt")

# Define the path to the main evaluation script
EVALUATE_SCRIPT = os.path.join(script_dir, "evaluation.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SDLog model evaluation for specified model and attribute type.")
    parser.add_argument("--model", type=str, choices=["main", "finetuned"], required=True,
                        help="Specify the model category: 'main' (pre-trained SDLog models) or 'finetuned' (locally fine-tuned models).")
    parser.add_argument("--attribute", type=str, choices=["all", "net"], required=True,
                        help="Specify the attribute set to evaluate on: 'all' sensitive attributes or 'net' attributes.")

    args = parser.parse_args()

    selected_model_path = ""
    selected_test_path = ""

    # --- Determine which model and dataset paths to use based on --model and --attribute ---
    if args.model == "main":
        if args.attribute == "all":
            selected_model_path = MAIN_MODEL_ALL_PATH
            selected_test_path = MAIN_DATASET_ALL_PATH
        elif args.attribute == "net":
            selected_model_path = MAIN_MODEL_NET_PATH
            selected_test_path = MAIN_DATASET_NET_PATH
    elif args.model == "finetuned":
        if args.attribute == "all":
            selected_model_path = FINE_TUNED_MODEL_ALL_PATH
            selected_test_path = MAIN_DATASET_ALL_PATH
        elif args.attribute == "net":
            selected_model_path = FINE_TUNED_MODEL_NET_PATH
            selected_test_path = MAIN_DATASET_NET_PATH
    else:
        print(f"Internal Error: Invalid model type '{args.model}'.")
        exit(1)

    # --- Basic existence checks for local paths ---
    if not os.path.exists(selected_test_path):
        print(f"Error: Specified test dataset path '{selected_test_path}' does not exist.")
        print("Please check the path or ensure the file is present.")
        exit(1)

    if args.model == "finetuned":
        if not os.path.exists(selected_model_path):
            print(f"Error: Fine-tuned model directory not found at '{selected_model_path}'.")
            print("Please ensure the model has been trained and saved to this location.")
            exit(1)

    # Construct the command to run evaluate.py
    command = [
        "python",
        EVALUATE_SCRIPT,
        "--model_path", selected_model_path,
        "--test_path", selected_test_path
    ]

    print(f"\n--- Running evaluation for '{args.model}' model with '{args.attribute}' attributes ---")

    # Execute the evaluate.py script
    try:
        subprocess.run(command, check=True, text=True)
    except FileNotFoundError:
        print(f"\nError: Python executable or '{EVALUATE_SCRIPT}' not found.")
        print("Please ensure Python is in your system's PATH and evaluate.py exists at the expected location.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nEvaluation script failed with exit code {e.returncode}.")
        print(f"Error output:\n{e.stderr}")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        exit(1)