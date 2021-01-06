import eval.models.slot_tagger as slot_tagger
import eval.models.snt_classifier as snt_classifier
import eval.utils.gpu_selection as gpu_selection
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import logging
import time
import os


def read_vocab_file(vocab_path, bos_eos=False, no_pad=False, no_unk=False, separator=':'):
    word2id, id2word = {}, {}
    if not no_pad:
        word2id['<pad>'] = len(word2id)
        id2word[len(id2word)] = '<pad>'
    if not no_unk:
        word2id['<unk>'] = len(word2id)
        id2word[len(id2word)] = '<unk>'
    if bos_eos == True:
        word2id['<s>'] = len(word2id)
        id2word[len(id2word)] = '<s>'
        word2id['</s>'] = len(word2id)
        id2word[len(id2word)] = '</s>'
    with open(vocab_path, 'r') as f:
        for line in f:
            if separator in line:
                word, idx = line.strip('\r\n').split(' '+separator+' ')
                idx = int(idx)
            else:
                word = line.strip()
                idx = len(word2id)
            if word not in word2id:
                word2id[word] = idx
                id2word[idx] = word
    return word2id, id2word


def read_seqtag_data_with_class(data_path, word2idx, tag2idx, class2idx, separator=':', multiClass=False, keep_order=True, lowercase=False):
    print('Reading source data ...')
    input_seqs = []
    tag_seqs = []
    class_labels = []
    line_num = -1
    with open(data_path, 'r') as f:
        for ind, line in enumerate(f):
            line_num += 1
            slot_tag_line, class_name = line.strip('\r\n').split(' <=> ')
            if slot_tag_line == "":
                continue
            in_seq, tag_seq = [], []
            for item in slot_tag_line.split(' '):
                tmp = item.split(separator)
                assert len(tmp) >= 2
                word, tag = separator.join(tmp[:-1]), tmp[-1]
                if lowercase:
                    word = word.lower()
                in_seq.append(word2idx[word] if word in word2idx else word2idx['<unk>'])
                tag_seq.append(tag2idx[tag] if tag in tag2idx else (tag2idx['<unk>'], tag))
            if keep_order:
                in_seq.append(line_num)
            input_seqs.append(in_seq)
            tag_seqs.append(tag_seq)
            if multiClass:
                if class_name == '':
                    class_labels.append([])
                else:
                    class_labels.append([class2idx[val] for val in class_name.split(';')])
            else:
                if ';' not in class_name:
                    class_labels.append(class2idx[class_name])
                else:
                    class_labels.append((class2idx[class_name.split(';')[0]], class_name.split(';'))) # get the first class for training

    input_feats = {'data':input_seqs}
    tag_labels = {'data':tag_seqs}
    class_labels = {'data':class_labels}

    return input_feats, tag_labels, class_labels


def get_minibatch_with_class(input_seqs, tag_seqs, class_labels, word2idx, tag2idx, class2idx, train_data_indx, index,
                             batch_size, add_start_end=False, multiClass=False, keep_order=True, enc_dec_focus=False,
                             device=None):
    """Prepare minibatch."""
    input_seqs = [input_seqs[idx] for idx in train_data_indx[index:index + batch_size]]
    tag_seqs = [tag_seqs[idx] for idx in train_data_indx[index:index + batch_size]]
    class_labels = [class_labels[idx] for idx in train_data_indx[index:index + batch_size]]
    if add_start_end:
        input_seqs = [[word2idx['<s>']] + line + [word2idx['</s>']] for line in input_seqs]
        tag_seqs = [[tag2idx['O']] + line + [tag2idx['O']] for line in tag_seqs]
    else:
        pass

    data_mb = list(zip(input_seqs, tag_seqs, class_labels))
    data_mb.sort(key=lambda x: len(x[0]), reverse=True)  # sorted for pad setence

    raw_tags = [[item[1] if type(item) in {list, tuple} else item for item in tag] for seq, tag, cls in data_mb]
    data_mb = [(seq, [item[0] if type(item) in {list, tuple} else item for item in tag], cls) for seq, tag, cls in
               data_mb]
    if keep_order:
        line_nums = [seq[-1] for seq, _, _ in data_mb]
        data_mb = [(seq[:-1], tag, cls) for seq, tag, cls in data_mb]

    lens = [len(seq) for seq, _, _ in data_mb]
    max_len = max(lens)
    input_idxs = [
        seq + [word2idx['<pad>']] * (max_len - len(seq))
        for seq, _, _ in data_mb
    ]
    input_idxs = torch.tensor(input_idxs, dtype=torch.long, device=device)

    if not enc_dec_focus:
        tag_idxs = [
            seq + [tag2idx['<pad>']] * (max_len - len(seq))
            for _, seq, _ in data_mb
        ]
    else:
        tag_idxs = [
            [tag2idx['<s>']] + seq + [tag2idx['<pad>']] * (max_len - len(seq))
            for _, seq, _ in data_mb
        ]
    tag_idxs = torch.tensor(tag_idxs, dtype=torch.long, device=device)

    if multiClass:
        raw_classes = [class_list for _, _, class_list in data_mb]
        class_tensor = torch.zeros(len(data_mb), len(class2idx), dtype=torch.float)
        for idx, (_, _, class_list) in enumerate(data_mb):
            for w in class_list:
                class_tensor[idx][w] = 1
        class_idxs = class_tensor.to(device)
    else:
        raw_classes = [class_label[1] if type(class_label) in {list, tuple} else class_label for _, _, class_label in
                       data_mb]
        class_idxs = [class_label[0] if type(class_label) in {list, tuple} else class_label for _, _, class_label in
                      data_mb]
        class_idxs = torch.tensor(class_idxs, dtype=torch.long, device=device)

    ret = [input_idxs, tag_idxs, raw_tags, class_idxs, raw_classes, lens]
    if keep_order:
        ret.append(line_nums)

    return ret


def get_chunks(labels):
    """
        It supports IOB2 or IOBES tagging scheme.
        You may also want to try https://github.com/sighsmile/conlleval.
    """
    chunks = []
    start_idx,end_idx = 0,0
    for idx in range(1,len(labels)-1):
        chunkStart, chunkEnd = False,False
        if labels[idx-1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            prevTag, prevType = labels[idx-1][:1], labels[idx-1][2:]
        else:
            prevTag, prevType = 'O', 'O'
        if labels[idx] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            Tag, Type = labels[idx][:1], labels[idx][2:]
        else:
            Tag, Type = 'O', 'O'
        if labels[idx+1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            nextTag, nextType = labels[idx+1][:1], labels[idx+1][2:]
        else:
            nextTag, nextType = 'O', 'O'

        if Tag == 'B' or Tag == 'S' or (prevTag, Tag) in {('O', 'I'), ('O', 'E'), ('E', 'I'), ('E', 'E'), ('S', 'I'), ('S', 'E')}:
            chunkStart = True
        if Tag != 'O' and prevType != Type:
            chunkStart = True

        if Tag == 'E' or Tag == 'S' or (Tag, nextTag) in {('B', 'B'), ('B', 'O'), ('B', 'S'), ('I', 'B'), ('I', 'O'), ('I', 'S')}:
            chunkEnd = True
        if Tag != 'O' and Type != nextType:
            chunkEnd = True

        if chunkStart:
            start_idx = idx
        if chunkEnd:
            end_idx = idx
            chunks.append((start_idx,end_idx,Type))
            start_idx,end_idx = 0,0
    return chunks


def decode(seed_file, output_path, device, model_path, vocab_path, save_npy, states_path, test_batchSize):
    tag_to_idx, idx_to_tag = read_vocab_file(vocab_path + '.tag', bos_eos=False, no_pad=True, no_unk=True)
    class_to_idx, idx_to_class = read_vocab_file(vocab_path + '.class', bos_eos=False, no_pad=True, no_unk=True)
    word_to_idx, idx_to_word = read_vocab_file(vocab_path + '.in', bos_eos=False, no_pad=True, no_unk=True)

    test_feats, test_tags, test_class = \
        read_seqtag_data_with_class(seed_file, word_to_idx, tag_to_idx, class_to_idx, multiClass=False, lowercase=False)
    model_tag = slot_tagger.LSTMTagger(embedding_dim=1024, hidden_dim=200, vocab_size=len(word_to_idx),
                                       tagset_size=len(tag_to_idx), bidirectional=False, num_layers=1,
                                       dropout=0.5, device=device, extFeats_dim=None)
    model_class = snt_classifier.sntClassifier_hiddenAttention(hidden_dim=200, class_size=len(class_to_idx),
                                                               bidirectional=False, num_layers=1, dropout=0.5,
                                                               device=device, multi_class=False)
    encoder_info_filter = lambda info: info
    model_tag.to(device)
    model_class.to(device)
    model_tag.load_model(model_path + '.tag')
    model_class.load_model(model_path + '.class')

    # loss function
    weight_mask = torch.ones(len(tag_to_idx), device=device)
    weight_mask[tag_to_idx['<pad>']] = 0
    tag_loss_function = nn.NLLLoss(weight=weight_mask, size_average=False)
    class_loss_function = nn.NLLLoss(size_average=False)

    model_tag.eval()
    model_class.eval()

    data_index = np.arange(len(test_feats['data']))
    losses = []
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    TP2, FP2, FN2, TN2 = 0.0, 0.0, 0.0, 0.0

    with open(output_path, 'w') as f:
        for j in range(0, len(data_index), test_batchSize):
            inputs, tags, raw_tags, classes, raw_classes, lens, line_nums = \
                get_minibatch_with_class(test_feats['data'], test_tags['data'], test_class['data'], word_to_idx,
                                         tag_to_idx, class_to_idx, data_index, j, test_batchSize, add_start_end=False,
                                         multiClass=False, enc_dec_focus=False, device=device)

            tag_scores, (packed_h_t_c_t, lstm_out, lengths) = model_tag(inputs, lens, with_snt_classifier=True)
            encoder_info = (packed_h_t_c_t, lstm_out, lengths)
            tag_loss = tag_loss_function(tag_scores.contiguous().view(-1, len(tag_to_idx)), tags.view(-1))
            top_pred_slots = tag_scores.data.cpu().numpy().argmax(axis=-1)

            if save_npy:
                tmp_save_path = "/root/chatbot/DialTest/data/snips/states/tmp/out_" + str(j) + ".npy"
                np.save(tmp_save_path, lstm_out.cpu().detach().numpy())

            class_scores = model_class(encoder_info_filter(encoder_info))
            class_loss = class_loss_function(class_scores, classes)
            snt_probs = class_scores.data.cpu().numpy().argmax(axis=-1)
            losses.append([tag_loss.item() / sum(lens), class_loss.item() / len(lens)])

            inputs = inputs.data.cpu().numpy()
            # classes = classes.data.cpu().numpy()
            for idx, pred_line in enumerate(top_pred_slots):
                length = lens[idx]
                pred_seq = [idx_to_tag[tag] for tag in pred_line][:length]
                lab_seq = [idx_to_tag[tag] if type(tag) == int else tag for tag in raw_tags[idx]]
                pred_chunks = get_chunks(['O'] + pred_seq + ['O'])
                label_chunks = get_chunks(['O'] + lab_seq + ['O'])
                for pred_chunk in pred_chunks:
                    if pred_chunk in label_chunks:
                        TP += 1
                    else:
                        FP += 1
                for label_chunk in label_chunks:
                    if label_chunk not in pred_chunks:
                        FN += 1

                input_line = [idx_to_word[word] for word in inputs[idx]][:length]
                word_tag_line = [input_line[_idx] + ':[' + lab_seq[_idx] + ']:' + pred_seq[_idx] for _idx in
                                 range(len(input_line))]


                pred_class = idx_to_class[snt_probs[idx]]
                if type(raw_classes[idx]) == int:
                    gold_classes = {idx_to_class[raw_classes[idx]]}
                else:
                    gold_classes = set(raw_classes[idx])
                if pred_class in gold_classes:
                    TP2 += 1
                else:
                    FP2 += 1
                    FN2 += 1
                    # print("true intent: ", gold_classes, "pred intent:", pred_class)
                    gold_class_str = ';'.join(list(gold_classes))
                    pred_class_str = pred_class
                    f.write(str(line_nums[idx]) + ' : ' + ' '.join(word_tag_line) + ' <=> ' + '[' + gold_class_str + ']' + ' <=> ' + pred_class_str + '\n')

    # ---
    tmp_save_path_dir = "/root/chatbot/DialTest/data/snips/states/tmp/"
    states_num = len(os.listdir(tmp_save_path_dir))
    if states_num == 1:
        data = np.load(os.path.join(tmp_save_path_dir, os.listdir(tmp_save_path_dir)[0]))
        np.save(states_path, data)
    else:
        total_len, max_len = 0, 0
        for states_data_list in os.listdir(tmp_save_path_dir):
            sub_path = os.path.join(tmp_save_path_dir, states_data_list)
            print(sub_path)
            if os.path.isfile(sub_path):
                data = np.load(sub_path)
                total_len += data.shape[0]
                if data.shape[1] > max_len:
                    max_len = data.shape[1]

        save_data = np.empty([total_len, max_len, 200])
        p = 0
        for states_data_list in os.listdir(tmp_save_path_dir):
            sub_path = os.path.join(tmp_save_path_dir, states_data_list)
            if os.path.isfile(sub_path):
                data = np.load(sub_path)

                if data.shape[1] == max_len:
                    for k in range(len(data)):
                        save_data[p] = data[k]
                        p += 1
                elif data.shape[1] < max_len:
                    z = np.zeros((max_len - data.shape[1], 200))
                    for k in range(len(data)):
                        tmp_data = np.r_[data[k], z]
                        save_data[p] = np.array([tmp_data])
                        p += 1

        print(p)
        np.save(states_path, save_data)

        for states_data_list in os.listdir(tmp_save_path_dir):
            sub_path = os.path.join(tmp_save_path_dir, states_data_list)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
    # ---
    if TP == 0:
        p, r, f = 0, 0, 0
    else:
        p, r, f = 100 * TP / (TP + FP), 100 * TP / (TP + FN), 100 * 2 * TP / (2 * TP + FN + FP)

    mean_losses = np.mean(losses, axis=0)
    slot_accuracy = (TP + TN) / (TP + FP + TN + FN)
    intent_accuracy = (TP2 + TN2) / (TP2 + FP2 + TN2 + FN2)
    return slot_accuracy, intent_accuracy, mean_losses, p, r, f, 0 if 2 * TP2 + FN2 + FP2 == 0 else 100 * 2 * TP2 / (
                2 * TP2 + FN2 + FP2)


def eval_model(seed_file, out_path, states_path):
    deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu()
    torch.cuda.set_device(deviceId)
    device = torch.device("cuda")
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    model_path = "/root/chatbot/DialTest/save_profile/snips/model"
    vocab_path = "/root/chatbot/DialTest/save_profile/snips/vocab"

    start_time = time.time()
    acc_slot_te, acc_intent_te, loss_te, p_te, r_te, f_te, cf_te = \
        decode(seed_file=seed_file, output_path=out_path, device=device, model_path=model_path, vocab_path=vocab_path,
               save_npy=True, states_path=states_path, test_batchSize=500)
    print('Test:\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f\tSlot Acc : %.2f Intent Acc : %.2f' %
        (time.time() - start_time, loss_te[0], loss_te[1], f_te, cf_te, acc_slot_te, acc_intent_te))


if __name__ == '__main__':
    seed_file = "/root/chatbot/DialTest/data/snips/test"
    out_path = "/root/chatbot/DialTest/data/snips/out.txt"
    states_path = "/root/chatbot/DialTest/data/snips/states/out_" + str(0) + ".npy"
    eval_model(seed_file, out_path, states_path)
