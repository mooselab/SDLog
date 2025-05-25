import os

def create_labeled_logs_from_txt(raw_data_dir, processed_logs_dir):
    logs_file_path = os.path.join(raw_data_dir, "logs.txt")
    labels_file_path = os.path.join(raw_data_dir, "labels.txt")

    # Define the output file path 
    processed_logs_filename = "dataset.txt" 
    processed_logs_path = os.path.join(processed_logs_dir, processed_logs_filename)

    try:
        with open(logs_file_path, 'r') as logs_file, \
             open(labels_file_path, 'r') as labels_file, \
             open(processed_logs_path, 'w') as output_file:

            log_entries = logs_file.readlines()
            label_entries = labels_file.readlines()

            # Sanity check: Ensure both files have the same number of lines
            if len(log_entries) != len(label_entries):
                raise ValueError(f"Mismatch in number of entries between '{logs_file_path}' "
                                 f"({len(log_entries)} lines) and '{labels_file_path}' "
                                 f"({len(label_entries)} lines).")

            for idx, (log_line, label_line) in enumerate(zip(log_entries, label_entries)):
                # Split tokens and labels
                tokens = log_line.strip().split()
                labels = label_line.strip().split()

                # Sanity check for token and label count within a line
                if len(tokens) != len(labels):
                    print(f'logs (line {idx+1}):', tokens)
                    print(f'label (line {idx+1}):', labels)
                    raise ValueError(f"Word count mismatch at line {idx+1}: "
                                    f"{len(tokens)} words in logs vs {len(labels)} in labels.")

                # Write each token and its label in token-label pairs
                for token, label in zip(tokens, labels):
                    output_file.write(f"{token}\t{label}\n")  # Token and label separated by a tab

                output_file.write("\n")  # Blank line to separate different log entries

        print(f"Successfully created processed logs at: {processed_logs_path}")

    except FileNotFoundError:
        print(f"Error: One or both of the input files ('{logs_file_path}' or '{labels_file_path}') were not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define input and output directories
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

raw_data_dir = os.path.join(project_root, 'target_dataset', '1-raw_datasets', 'main')
processed_logs_dir = os.path.join(project_root, 'target_dataset', '2-processed_datasets', 'main')

# Create processed_logs_dir if it doesn't exist
if not os.path.exists(processed_logs_dir):
    os.makedirs(processed_logs_dir)

# Call the function to process the text files
create_labeled_logs_from_txt(raw_data_dir, processed_logs_dir)