from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def _get_bin_lbs(gt_tags, pre_tags):
    bin_gt_tags = []
    bin_pre_tags = []
    for i in range(len(gt_tags)):
        if gt_tags[i] != 'O' or pre_tags[i] != 'O':
            bin_gt_tags.append(False if gt_tags[i]=='O' else True)
            bin_pre_tags.append(False if pre_tags[i]=='O' else True)
    return bin_gt_tags, bin_pre_tags

def _calc_matrics(labels, predictions, id2lb, lb2id, p_match):
    index = 0
    cnt_pos = 0
    acc_correct = 0
    gt_tags = []
    pre_tags = []

    o_label_id = lb2id.get('O')

    while index < len(labels):
        if labels[index] == -100:
            index = index + 1
            continue
        elif o_label_id is not None and labels[index] == o_label_id:
            gt_tags.append('O')
            if predictions[index] == o_label_id:
                pre_tags.append('O')
            else:
                predicted_tag = id2lb.get(predictions[index], 'UNKNOWN')
                pre_tags.append(predicted_tag[2:] if len(predicted_tag) > 2 and predicted_tag[0] in ['B', 'I'] else 'O')
            index = index + 1
            continue
        
        try:
            BIO_tag = id2lb[labels[index]]
            c_start = BIO_tag[0]
            cur_entity_type = BIO_tag[2:]

            if c_start == 'B':
                gt_tags.append(cur_entity_type)
                cnt_pos = cnt_pos + 1

                current_entity_match_correct = True
                temp_index = index
                
                if labels[temp_index] != predictions[temp_index]:
                    current_entity_match_correct = False

                temp_index += 1
                while temp_index < len(labels) and labels[temp_index] != -100 and id2lb.get(labels[temp_index], '')[0] == 'I':
                    if labels[temp_index] != predictions[temp_index]:
                        current_entity_match_correct = False
                    temp_index += 1

                if current_entity_match_correct:
                    pre_tags.append(cur_entity_type)
                    acc_correct = acc_correct + 1
                elif p_match:
                    predicted_b_tag = id2lb.get(predictions[index], '')
                    if predicted_b_tag[2:] == cur_entity_type:
                        pre_tags.append(cur_entity_type)
                        acc_correct += 1
                    else:
                        pre_tags.append('O')
                else:
                    pre_tags.append('O')

                index = temp_index
            else:
                index = index + 1
        except KeyError:
            index += 1
            continue

    if cnt_pos!=0:
        acc = acc_correct / cnt_pos
    else:
        acc = 0
    
    if not gt_tags and not pre_tags:
        return {
            "accuracy": 0.0,
            "confusion_matrix": np.array([[]]),
            "report": "No entities found for evaluation."
        }

    bin_gt_tags, bin_pre_tags = _get_bin_lbs(gt_tags, pre_tags)

    unique_gt_tags = sorted(list(set(gt_tags) - {'O'}))
    unique_pre_tags = sorted(list(set(pre_tags) - {'O'}))
    target_names = sorted(list(set(unique_gt_tags + unique_pre_tags)))

    cm = confusion_matrix(gt_tags, pre_tags, labels=sorted(list(set(gt_tags + pre_tags))))
    report = classification_report(gt_tags, pre_tags, labels=target_names, zero_division=0, digits=3)
    bin_report = classification_report(bin_gt_tags, bin_pre_tags, zero_division=0, digits=3, output_dict=True)

    print('* Please ignore the class `O`, we did not count it in the report.\n',report)
    print('Confusion Matrix:\n', cm)
    print('Accuracy:\n', acc)
    print('\nBinary results (Sensitive or not):\n')
    print('Out of the predicted pieces of sensitive info, what portion is real sensitive:\n Precision (TP/TP+FP):', bin_report['True']['precision'])
    print('Out of all the sensitive information in the log, what portion is detected:\n Recall (TP/TP+FN):', bin_report['True']['recall'])
    print('F1 score of sensitive information:\n F1 (2*prec*rec/(prec+rec)):', bin_report['True']['f1-score'])
    
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report
    }

def compute_metrics_entity(p, id2lb, lb2id, p_match = True, cross_val = False):
    pred, lb = p
    if cross_val == True:
        assert len(pred) == len(lb), "The lengths of predictions and labels mismatch."
        labels = []
        predictions = []
        for i in range(len(pred)):
            cur_pred =  pred[i].argmax(axis=2)
            cur_lb = lb[i].reshape(-1)
            cur_pred = cur_pred.reshape(-1)
            predictions.append(cur_pred)
            labels.append(cur_lb)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
    else:
        predictions = pred.argmax(axis=2)
        labels = lb.reshape(-1)
        predictions = predictions.reshape(-1)
    return _calc_matrics(labels, predictions, id2lb, lb2id, p_match)


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    # Example for local testing (can be removed if not needed)
    try:
        model = AutoModelForTokenClassification.from_pretrained('LogSensitiveResearcher/SDLog_main')
    except Exception:
        from transformers import AutoConfig
        config_without_O = AutoConfig.from_pretrained('microsoft/codebert-base', num_labels=3, id2label={0: 'B-HOST', 1: 'B-IP', 2: 'B-PORT'}, label2id={'B-HOST': 0, 'B-IP': 1, 'B-PORT': 2})
        model = type('DummyModel', (object,), {'config': config_without_O})()

    lb2id = model.config.label2id
    id2lb = model.config.id2label

    # Dummy data for testing
    labels_test = np.array([0, -100, 1, 0, 2, -100, 2]) # Example with -100 and valid IDs
    predictions_test = np.array([0, -100, 1, 0, 0, -100, 2]) # Example with some misclassifications
    
    # Create dummy logits for `p` argument
    num_labels_in_model = len(id2lb)
    pred_logits = np.random.rand(1, len(labels_test), num_labels_in_model)
    # Ensure predictions_test has valid IDs for this model
    pred_logits[0, np.arange(len(labels_test)), predictions_test] = 10.0 # Make sure `argmax` yields `predictions_test`

    p = [pred_logits, labels_test[np.newaxis, :]] # Add batch dimension for Trainer output format
    compute_metrics_entity(p, id2lb, lb2id, p_match = True, cross_val = False)