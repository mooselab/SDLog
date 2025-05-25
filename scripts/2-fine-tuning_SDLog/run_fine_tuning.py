import os
import sys
import argparse
import subprocess

def run_fine_tuning():
    # --- Parse Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="Run fine-tuning script for SDLog model.")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs. Defaults to 2.")

    args = parser.parse_args()
    base_model = "microsoft/codebert-base"
    num_train_epochs = args.num_train_epochs

    # Resolve paths relative to the script's directory if they are relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    train_path_resolved = os.path.join(project_root, "target_dataset", "3-fine_tuned_datasets", "main", "train.txt")
    test_path_resolved = os.path.join(project_root, "target_dataset", "3-fine_tuned_datasets", "main", "test.txt")

    # Path to the base model
    base_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--LogSensitiveResearcher--SDLog_main/snapshots/7241fbb2cf8affe71d627496d3b5d592878a2b87")

    # Path where the fine-tuned model will be saved
    finetune_model_path = os.path.join(project_root, "fine_tuned_model")

    # --- Create the directory to save the fine-tuned model ---
    os.makedirs(finetune_model_path, exist_ok=True)

    # --- Verify dataset paths exist ---
    if not os.path.exists(train_path_resolved):
        print(f"Error: Training dataset not found at '{train_path_resolved}'")
        sys.exit(1)
    if not os.path.exists(test_path_resolved):
        print(f"Error: Test dataset not found at '{test_path_resolved}'")
        sys.exit(1)

    # --- Construct the command to run fine_tuning.py ---
    command = [
        sys.executable,
        "scripts/2-fine-tuning_SDLog/fine_tuning.py",
        "--train_path", train_path_resolved,
        "--test_path", test_path_resolved,
        "--model_save_path", finetune_model_path,
        "--base_model", base_model,
        "--num_train_epochs", str(num_train_epochs),
        "--load_best_model_at_end",
        "--local_model_path", base_model_path
    ]

    # --- Execute the command ---
    try:
        subprocess.run(command, check=True, text=True, capture_output=False)
        print("\nFine-tuning completed successfully!")
    except FileNotFoundError:
        print(f"\nError: 'train.py' script not found or Python executable not in PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Fine-tuning script failed with exit code {e.returncode}.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_fine_tuning()