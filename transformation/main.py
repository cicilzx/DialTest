from  BT_aug import augment as BT
from SR_aug import augment as SR
from WI_aug import augment as WI


def process_data(in_file, out_file):
    """
    Convert the original train/test/valid file into a file with only English sentences,
    each sentence occupies one line.
    :in_file: the original train/test/valid file with word slot label and sentence intent label
    :out_file: the transformed file with only sentences.
    """
    in_file = open(file=in_file, encoding='utf-8')
    out_file = open(file=out_file, encoding='utf-8', mode='a')
    for line in in_file.readlines():
        sentence, intent = line.split(' <=> ')
        words = []
        slots = []
        word_slots = sentence.split(' ')
        for word_slot in word_slots:
            flag = word_slot.rfind(':')
            word = word_slot[0:flag]
            slot_label = word_slot[flag+1:]
            words.append(word)
            slots.append(slot_label)
        new_sentence = ' '.join(words)
        new_sentence = new_sentence + '\n'
        # new_sentence = new_sentence + ' <=> ' + intent
        out_file.write(new_sentence)
        print(new_sentence)
        # print(words, slots)
    out_file.close()
    in_file.close()


def process_bert(in_file, out_file):
    """
    the sentences transformed by BERT contains #, so we remove the #
    in_file: the file processed by bert with word insertion,
    out_file: the file with clean sentences.
    """
    input_file = open(file=in_file, encoding='utf-8')
    output_file = open(file=out_file, encoding='utf-8', mode='w')

    for line in input_file.readlines():
        tmp_line = line.replace(' ##', '')
        new_line = tmp_line.replace('##', '')
        output_file.write(new_line + "\n")

    input_file.close()
    output_file.close()


def label_aug_sentence(src_file_path, input_file_path, out_file_path):
    """
    transform the clean sentence to sentences with labels
    src_file_path: original data
    input_file_path: only sentences (after transformation)
    out_file_path: transformed sentences with intent and slot labels
    """
    src_file = open(file=src_file_path, encoding='utf-8')
    input_file = open(file=input_file_path, encoding='utf-8')
    out_file = open(file=out_file_path, encoding='utf-8', mode='a')

    src_lines = src_file.readlines()
    flag_ = -1
    for line in input_file.readlines():
        words = []
        slots = []
        temp_src = src_lines[flag_]
        sentence, intent = temp_src.strip('\n').split(' <=> ')
        word_slots = sentence.split(' ')
        for word_slot in word_slots:
            flag = word_slot.rfind(':')
            word = word_slot[0:flag]
            slot_label = word_slot[flag + 1:]
            words.append(word.lower())
            slots.append(slot_label)
        dict_ = dict(zip(words, slots))
        if line == '\n':
            flag_ += 1
        else:
            new_dic = {}
            w = line.strip('\n').split(' ')
            if len(words) >= 2:
                if w[-1] == str(words[-2] + words[-1]):
                    w[-1] = words[-2]
                    w.append(words[-1])
            for each in w:
                if len(each) > 1 and (each[-1] == "?" or each[-1] == "."):
                    each = each[:-1]
                if each.lower() in dict_.keys():
                    new_dic[each.lower()] = dict_[each.lower()]
                else:
                    new_dic[each.lower()] = "O"
            new_sen = ""
            for keys, values in new_dic.items():
                new_sen = new_sen + keys + ":" + values + " "
            new_sen = new_sen + "<=> " + intent + "\n"
            out_file.write(new_sen)
            print(new_sen)


if __name__ == '__main__':
    original_data = "../dataset/atis/test"
    original_clean_data = "../dataset/atis/test_clean"
    model_dir = '/home/cici/major/NLPDataAugmentation/wwm_cased_L-24_H-1024_A-16/'  
    # bert pretrained model, the path is needed to be changed to your own path. It can be downloaded at https://github.com/google-research/bert
    process_data(original_data, original_clean_data)

    BT_out_file_path1 = "../dataset/atis/bt_test1"  # mid file of BT
    BT_out_file_path2 = "../dataset/atis/bt_test"
    SR_out_file_path = "../dataset/atis/sr_test"
    WI_out_file_path1 = "../dataset/atis/wi_test1"  # mid file of WI
    WI_out_file_path2 = "../dataset/atis/wi_test2"  # mid file of WI
    WI_out_file_path3 = "../dataset/atis/wi_test"

    BT(original_clean_data, BT_out_file_path1)
    SR(original_data, SR_out_file_path, 3)
    WI(original_clean_data, WI_out_file_path1, model_dir)
    process_bert(WI_out_file_path1, WI_out_file_path2)

    # # transform to data with slot label and intent label
    label_aug_sentence(original_data, BT_out_file_path1, BT_out_file_path2)
    label_aug_sentence(original_data, WI_out_file_path2, WI_out_file_path3)
    print("The BT, SR, WI is saved as bt_test, sr_test, wi_test.")
