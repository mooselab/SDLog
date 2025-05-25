import argparse
import subprocess
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

# --- Define model paths ---
# Main models from Hugging Face
MAIN_MODEL_ALL_PATH = 'LogSensitiveResearcher/SDLog_main' 
# MAIN_MODEL_NET_PATH = 'LogSensitiveResearcher/SDLog_net'
# Fine-tuned models
FINE_TUNED_MODEL_ALL_PATH = os.path.join(project_root, "fine_tuned_model") 
# FINE_TUNED_MODEL_NET_PATH = os.path.join(project_root, "fine_tuned_model_net")

# --- Define test dataset paths ---
MAIN_DATASET_ALL_PATH = os.path.join(project_root, "target_dataset", "1-raw_datasets", "main", "logs.txt")
# MAIN_DATASET_NET_PATH = os.path.join(script_dir, "target_dataset", "1-raw_datasets", "net", "logs.txt")

# Define the path to the sdlog.py script
CHECK_PREDICTIONS_SCRIPT = os.path.join(script_dir, "sdlog.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SDLog model prediction checks for specified model and attribute type.")
    parser.add_argument("--model", type=str, choices=["main", "finetuned"], required=True,
                        help="Specify the model category: 'main' (pre-trained SDLog models) or 'finetuned' (locally fine-tuned models).")
    parser.add_argument("--attribute", type=str, choices=["all"], required=True,
                        help="Specify the attribute set to evaluate on: 'all' sensitive attributes or 'net' attributes. This determines the model to load.")

    args = parser.parse_args()

    selected_model_path = ""

    # --- Determine which model path to use based on --model and --attribute ---
    if args.model == "main":
        if args.attribute == "all":
            selected_model_path = MAIN_MODEL_ALL_PATH
            selected_log_path = MAIN_DATASET_ALL_PATH
        # elif args.attribute == "net":
        #     selected_model_path = MAIN_MODEL_NET_PATH
        #     selected_log_path = MAIN_DATASET_NET_PATH
    elif args.model == "finetuned":
        if args.attribute == "all":
            selected_model_path = FINE_TUNED_MODEL_ALL_PATH
            selected_log_path = MAIN_DATASET_ALL_PATH
        # elif args.attribute == "net":
            # selected_model_path = FINE_TUNED_MODEL_NET_PATH
            # selected_log_path = MAIN_DATASET_NET_PATH
    else:
        print(f"Internal Error: Invalid model type '{args.model}'.")
        sys.exit(1)

    # --- Basic existence checks for local paths ---
    if not os.path.exists(selected_log_path):
        print(f"Error: Specified test dataset path '{selected_log_path}' does not exist.")
        print("Please check the path or ensure the file is present.")
        exit(1)

    if args.model == "finetuned":
        if not os.path.exists(selected_model_path):
            print(f"Error: Fine-tuned model directory not found at '{selected_model_path}'.")
            sys.exit(1)

    # Construct the command to run check_predictions.py
    command = [
        sys.executable, # Use the same Python interpreter
        CHECK_PREDICTIONS_SCRIPT,
        "--model_path", selected_model_path,
        "--log_path", selected_log_path
    ]

    print(f"\n--- Running prediction check for '{args.model}' model with '{args.attribute}' attributes ---")

    # Execute the check_predictions.py script
    try:
        subprocess.run(command, check=True, text=True)
    except FileNotFoundError:
        print(f"\nError: Python executable or '{CHECK_PREDICTIONS_SCRIPT}' not found.")
        print("Please ensure Python is in your system's PATH and check_predictions.py exists at the expected location.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nPrediction check script failed with exit code {e.returncode}.")
        print(f"Error output:\n{e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        sys.exit(1)