# SDLog: Sensitivity Detector in Software Logs

## Do you need to find sensitive information in your software logs?

SDLog is a powerful, deep learning-based framework designed to **automatically identify sensitive information in software logs**. Unlike traditional regular expressions that struggle with the diverse and unstructured nature of real-world logs, SDLog leverages contextual understanding to accurately detect Personally Identifiable Information (PII). You can use the pre-trained SDLog, or, to achieve even better performance on your specific log formats, **you can easily fine-tune it with as few as 100 labeled log samples**, enabling near-perfect detection.

---

## Installation

To get started with SDLog, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mooselab/SDLog/
    cd SDLog/
    ```

2.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n sdlog python=3.10
    conda activate sdlog
    ```

3.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

4.  **Fetch Pre-trained Models:**
    ```bash
    python fetch_models.py
    ```

---

## Usage

SDLog can be used for sensitive information detection out-of-the-box or fine-tuned on your specific dataset.

**Important Note:** For just **running** SDLog to anonymize your logs, you **do not** need a labeled dataset. However, if you wish to **fine-tune** the model or **evaluate** its performance, you will need a dataset with both log entries and their corresponding labels.

### 1. Run SDLog on Your Dataset

To anonymize your logs using the pre-trained SDLog model:

* **Add your dataset:** Put your log file with the name `logs.txt` into the following directory: 
`target_dataset/1-raw_datasets/main/`
* **Run the anonymization script:**
    ```bash
    python scripts/3-run_SDLog/run_sdlog.py --model main --attribute all
    ```
    The anonymized logs will be saved in `target_dataset/4-anonymized_datasets/main/dataset_anonymized.txt`.

### 2. Fine-tune SDLog with Your Logs

For enhanced performance, you can fine-tune SDLog.

* **Add your dataset and labels:**
    * Put your log file named `logs.txt` and your corresponding labels file named `labels.txt` into: `target_dataset/1-raw_datasets/main/`
* **Run preprocessing steps:**
    ```bash
    python scripts/1-preprocessing/preprocessing_main.py
    python scripts/1-preprocessing/preprocessing_fine_tuning.py --num_finetuned_logs 200
    ```
    * The `--num_finetuned_logs` argument specifies the number of log entries (with sensitive information) from your dataset that will be used for fine-tuning. This argument is **optional**. If you remove it, the entire dataset found in `target_dataset/1-raw_datasets/main/logs.txt` will be used for fine-tuning.
* **Run the fine-tuning script:**
    ```bash
    python scripts/2-fine-tuning_SDLog/run_fine_tuning.py --num_train_epochs 3
    ```
    The `--num_finetuned_logs` argument specifies the number of epochs for fine-tuning. This argument is **optional** and the default epoch value is 2.
* **Run the anonymization script:**
    ```bash
    python scripts/3-run_SDLog/run_sdlog.py --model finetuned --attribute all
    ```
    The fine-tuned model will be saved, and your anonymized logs will be in `target_dataset/4-anonymized_datasets/main/dataset_anonymized.txt`.

### 3. Evaluate SDLog

To evaluate the performance of the SDLog model (either the pre-trained `main` model or your `finetuned` model):

* **Ensure you have labeled data:** As mentioned above, evaluation requires both `logs.txt` and `labels.txt` in `target_dataset/1-raw_datasets/main/`.
* **For the pre-trained `main` model:**
    ```bash
    python scripts/4-evaluation/run_evaluation.py --model main --attribute all
    ```
* **For your `finetuned` model:**
    ```bash
    python scripts/4-evaluation/run_evaluation.py --model finetuned --attribute all
    ```

---

## Citation

If you are interested in the performance of SDLog, you can find detailed evaluations in our paper. If you use SDLog in your research, please consider citing it:

```bibtex
@article{aghili2025sdlog,
  title={SDLog: A Deep Learning Framework for Detecting Sensitive Information in Software Logs},
  author={Roozbeh Aghili, Xingfang Wu, Foutse Khomh, and Heng Li},
  journal={arXiv preprint arXiv:2505.14976},
  year={2025}
}
```

---
