import os
import argparse

def split_and_save_logs(raw_log_path, processed_log_dir, num_train_logs):
    train_logs = []
    test_logs = []
    current_log = []
    
    # Define the input file path 
    train_file_path = os.path.join(processed_log_dir, 'train.txt')
    test_file_path = os.path.join(processed_log_dir, 'test.txt')
    
    try:
        with open(raw_log_path, 'r') as file:
            for line in file:
                line = line.rstrip('\n')
                
                # Empty line indicates end of a log entry
                if not line and current_log:
                    # Check if the current log has any non-O labels
                    has_non_o = any(len(parts) > 1 and parts[1].strip() != 'O' for parts in 
                                  [token_label.split() for token_label in current_log if token_label])
                    
                    # Add to appropriate list based on criteria and count
                    if has_non_o and len(train_logs) < num_train_logs:
                        train_logs.append(current_log.copy())
                    else:
                        test_logs.append(current_log.copy())
                    
                    # Reset for next log
                    current_log = []
                else:
                    current_log.append(line)
            
            # Check the last log entry if the file doesn't end with an empty line
            if current_log:
                has_non_o = any(len(parts) > 1 and parts[1].strip() != 'O' for parts in 
                              [token_label.split() for token_label in current_log if token_label])
                
                if has_non_o and len(train_logs) < num_train_logs:
                    train_logs.append(current_log.copy())
                else:
                    test_logs.append(current_log.copy())

        # Write the training logs to file
        with open(train_file_path, 'w') as train_file:
            for log in train_logs:
                for line in log:
                    train_file.write(f"{line}\n")
                train_file.write("\n")  # Empty line between logs

        # Write the testing logs to file
        with open(test_file_path, 'w') as test_file:
            for log in test_logs:
                for line in log:
                    test_file.write(f"{line}\n")
                test_file.write("\n")  # Empty line between logs

        print(f"- Finetuning training set: {len(train_logs)} logs saved to {train_file_path}")
        print(f"- Finetuning testing set: {len(test_logs)} logs saved to {test_file_path}")
        
        return len(train_logs), len(test_logs)
        
    except FileNotFoundError:
        print(f"Error: File '{raw_log_path}' not found.")
        return 0, 0
    except Exception as e:
        print(f"Error processing file: {e}")
        return 0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw log entries into training and testing sets.")
    parser.add_argument("--num_finetuned_logs", type=int, default=9999999999999,
                        help="Number of logs to include in the fine-tuned set. "
                             "If not specified, all logs with non-O labels will be used for training.")
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    raw_logs_dir = os.path.join(project_root, 'target_dataset', '2-processed_datasets', 'main')
    processed_logs_dir = os.path.join(project_root, 'target_dataset', '3-fine_tuned_datasets', 'main')
    if not os.path.exists(processed_logs_dir):
        os.makedirs(processed_logs_dir)
    
    for item in os.listdir(raw_logs_dir):
        raw_log_path = os.path.join(raw_logs_dir, item)
        if os.path.isfile(raw_log_path) and item.endswith(".txt"):
            train_count, test_count = split_and_save_logs(raw_log_path, processed_logs_dir, args.num_finetuned_logs)
        elif os.path.isdir(raw_log_path):
            print(f"Skipping directory: {item}")