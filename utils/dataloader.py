from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value

def _load_ner_log_data(file_path):
    sentences = []
    labels = []
    sentence = []
    label = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                word, tag = line.strip().split("\t")
                sentence.append(word)
                label.append(tag)

    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return {"tokens": sentences, "ner_tags": labels}


def _encode_labels(data, label_to_id):
    # data["ner_tags"] = [[label_to_id.get(tag, label_to_id["O"]) for tag in tags] for tags in data["ner_tags"]]
    data["ner_tags"] = [[label_to_id[tag] for tag in tags] for tags in data["ner_tags"]]
    return data

def construct_dataset(train_path, test_path):
    train_data = _load_ner_log_data(train_path)
    test_data = _load_ner_log_data(test_path)
    label_list = sorted({tag for tags in train_data["ner_tags"] for tag in tags})
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    train_data = _encode_labels(train_data, label_to_id)
    test_data = _encode_labels(test_data, label_to_id)

    features = Features({
    "tokens": Sequence(feature=Value("string")),
    "ner_tags": Sequence(feature=ClassLabel(names=label_list))
    })

    train_dataset = Dataset.from_dict(train_data, features=features)
    test_dataset = Dataset.from_dict(test_data, features=features)

    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    return dataset, label_list

def construct_dataset_wo_train(model_config, test_path):
    test_data = _load_ner_log_data(test_path)
    label_list = sorted(model_config['label2id'].keys())

    label_to_id = model_config['label2id']
    test_data = _encode_labels(test_data, label_to_id)

    features = Features({
    "tokens": Sequence(feature=Value("string")),
    "ner_tags": Sequence(feature=ClassLabel(names=label_list))
    })

    test_dataset = Dataset.from_dict(test_data, features=features)

    dataset = DatasetDict({
        "test": test_dataset
    })

    return dataset, label_list

def construct_dataset_with_prior(model_config, train_path, test_path):
    # previous_train_path = _load_ner_log_data(previous_train_path)
    train_data = _load_ner_log_data(train_path)
    test_data = _load_ner_log_data(test_path)
    # label_list = sorted({tag for tags in previous_train_path["ner_tags"] for tag in tags})
    label_list = sorted(model_config['label2id'].keys())
    label_to_id = model_config['label2id']
    id_to_label = model_config['id2label']
    # label_to_id = {label: i for i, label in enumerate(label_list)}
    # id_to_label = {i: label for i, label in enumerate(label_list)}

    train_data = _encode_labels(train_data, label_to_id)
    test_data = _encode_labels(test_data, label_to_id)

    features = Features({
    "tokens": Sequence(feature=Value("string")),
    "ner_tags": Sequence(feature=ClassLabel(names=label_list))
    })

    train_dataset = Dataset.from_dict(train_data, features=features)
    test_dataset = Dataset.from_dict(test_data, features=features)

    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    return dataset, label_list



if __name__ == "__main__":
    train_data = _load_ner_log_data("train.txt")
    print(train_data)
    