import re
from collections import OrderedDict

# def add_space_special_chars(text, special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '[', ']', '{', '}', ';', ':', ',', '.', '<', '>', '?', '/', '~', '', '|', '\\', "'", '"']):
def add_space_special_chars(text, special_chars = []):
    pattern_l = fr'(?<!\s)([{"".join(map(re.escape, special_chars))}])'
    pattern_r = fr'([{"".join(map(re.escape, special_chars))}])(?!\s)'

    positions_l = []
    positions_r = []
    for match in re.finditer(pattern_l, text):
        positions_l.append(match.start())
    for match in re.finditer(pattern_r, text):
        positions_r.append(match.start())

    text = re.sub(pattern_l, r' \1', text)
    text = re.sub(pattern_r, r'\1 ', text)
    return text, positions_l, positions_r


def _remove_token_at_index(s, index):
    if s[index]==' ':
        return s[:index] + s[index+1:]
    else:
        return s

def _get_pos_dict(pos_l, pos_r):
    two_sides = list(set(pos_l) & set(pos_r))
    left_side = list(set(pos_l) - set(two_sides))
    right_side = list(set(pos_r) - set(two_sides))

    # consecutive two sides token
    two_sides.sort()
    for i in range(len(two_sides)-1):
        if two_sides[i] == two_sides[i+1]-1:
            # right_side.append(two_sides.pop(i+1))
            right_side.append(two_sides[i+1])
            i = i + 1

    pos_dict = OrderedDict()
    pos_dict.update({key: 'both' for key in two_sides})
    pos_dict.update({key: 'left' for key in left_side})
    pos_dict.update({key: 'right' for key in right_side})
    pos_dict = sorted(pos_dict.items())
    pos_dict = OrderedDict(pos_dict)
    return pos_dict

def remove_space_special_chars(text, pos_l, pos_r):
    pos_dict = _get_pos_dict(pos_l, pos_r)
    for key, type in pos_dict.items():
        if type == 'both':
            text = _remove_token_at_index(text,key)
            text = _remove_token_at_index(text,key+1)
        elif type == 'left':
            text = _remove_token_at_index(text,key)
        else:
            text = _remove_token_at_index(text,key+1)
    return text

def adjust_ner_results(ner_results, pos_l, pos_r):
    pos_dict = _get_pos_dict(pos_l, pos_r)
    for entity_group in ner_results:
        start = entity_group['start']
        end = entity_group['end']
        last_pos = -10
        for pos, side in pos_dict.items():
            if pos < start-1:
                if side == 'both':
                    if pos == last_pos + 1:
                        start = start -1
                        end = end -1
                    else:
                        start = start -2
                        end = end -2
                else:
                    if pos == last_pos + 1:
                        start = start 
                        end = end 
                    else:
                        start = start -1
                        end = end -1
            elif pos == start-1:
                if side == 'left':
                    start = start -1
                    end = end -1
                elif side == 'both':
                    start = start -1
                    end = end -2
            elif pos < end-1:
                if side == 'both':
                    end = end -2
                else:
                    end = end -1
            last_pos = pos
        entity_group['start'] = start
        entity_group['end'] = end
    return ner_results

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                # ignore index: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                label_ids.append(-100)
            elif word_id != previous_word_id:  # First token of the word
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)  # Set to -100 to ignore during training
            previous_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == "__main__":
    # sentence= "081109 204106 329 INFO dfs.DataNode$PacketResponder : PacketResponder 2 for block blk_-6670958622368987959 terminating"
    # sentence = "20171223-23:19:21:439|Step_SPUtils|30002312| getTodayTotalDetailSteps = 1514042280000##7214##549659##8661##16256##31004759"
    # sentence = "Jul 17 23:21:54 combo ftpd[25235]: connection from 82.68.222.195 (82-68-222-195.dsl.in-addr.zen.co.uk) at Sun Jul 17 23:21:54 2005"
    sentence = "- 1131566594 2005.11.09 tbird-admin1 Nov 9 12:03:14 local@tbird-admin1 /apps/x86_64/system/ganglia-3.0.1/sbin/gmetad[1682]: data_thread() got not answer from any [Thunderbird_C1] datasource"
    # sentences = ["081109 204106 329 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-6670958622368987959 terminating", "081109 204106 329 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-6670958622368987959 terminating", "081109 204106 329 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-6670958622368987959 terminating", "081109 204106 329 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-6670958622368987959 terminating"]
    # text, positions_l, positions_r = add_space_special_chars(sentence)
    # print(text, positions_l, positions_r)
    # ret = list(map(add_space_special_chars, sentences))
    # print(ret[1])
    text, positions_l, positions_r = add_space_special_chars(sentence)
    # print(positions_l)
    # print(positions_r)
    _get_pos_dict(positions_l,positions_r)
    text_reversed = remove_space_special_chars(text, positions_l, positions_r)
    # print('Original:',sentence)
    # print('added:',text)
    # print('reversed:',text_reversed)
