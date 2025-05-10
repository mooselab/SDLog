import os
import pandas as pd

def create_labeled_logs(raw_logs_dir, processed_logs_dir):
    # Process each CSV file in raw_logs_dir
    for filename in os.listdir(raw_logs_dir):
        if filename.endswith(".csv"):
            raw_logs_path = os.path.join(raw_logs_dir, filename)

            # Create the output path
            base_filename = os.path.splitext(filename)[0] # Gets filename
            processed_logs_filename = f"{base_filename}.txt"
            processed_logs_path = os.path.join(processed_logs_dir, processed_logs_filename) # Output .txt path

            # Read logs
            df = pd.read_csv(raw_logs_path, usecols=['logs', 'labels'])

            with open(processed_logs_path, 'w') as output_file:
                for idx, row in df.iterrows():
                    tokens = row['logs'].split()
                    labels = row['labels'].split()

                    # Sanity check
                    if len(tokens) != len(labels):
                        print('logs:', tokens)
                        print('label:', labels)
                        raise ValueError(f"Word count mismatch in {filename} at row {idx}: "
                                        f"{len(tokens)} words in Content vs {len(labels)} in final_label.")

                    # Write each token and its label in token-label pairs
                    for token, label in zip(tokens, labels):
                        output_file.write(f"{str(token)}\t{label}\n")  # Token and label separated by a tab

                    output_file.write("\n")  # Blank line to separate different log entries

# Define input and output directories
script_dir = os.path.dirname(__file__)
raw_logs_dir = os.path.join(script_dir, '..', '..', 'target_dataset', '1-raw_dataset', 'main')
processed_logs_dir = os.path.join(script_dir, '..', '..', 'target_dataset', '2-processed_dataset', 'main')

if not os.path.exists(processed_logs_dir):
    os.makedirs(processed_logs_dir)

create_labeled_logs(raw_logs_dir, processed_logs_dir)