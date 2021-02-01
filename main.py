from cov.coverage_analyzer import get_BSCov_BTCov
from eval import eval_model_test
import os
import numpy as np


def cov_guided():
    model_path = "/home/cici/major/DialTest/save_profile/snips-lstm/model"
    vocab_path = "/home/cici/major/DialTest/save_profile/snips-lstm/vocab"
    wrapper = "/home/cici/major/DialTest/save_profile/nlu-model/snips_abst_train_model/wrapper_lstm_nlu_3_10.pkl"
    test_data_path = "/media/cici/BackupDrive/data/snips/bert_aug-1/valid"
    out_path = "/home/cici/major/DialTest/data/out.txt"

    seed_data_path = "/home/cici/major/DialTest/data/snips/valid_bert_cov1"
    in_test_data = open(test_data_path, mode='r', encoding='utf-8')
    seed_data = open(seed_data_path, mode='a', encoding='utf-8')

    lines = in_test_data.readlines()
    cov_list = []
    for j, line in enumerate(lines):
        states_path = "/home/cici/major/DialTest/data/snips/states/out_" + str(j) + ".npy"
        tmp_seed_data_path = "/home/cici/major/DialTest/data/seed_tmp"
        tmp_seed_data = open(tmp_seed_data_path, mode='w', encoding='utf-8')
        tmp_seed_data.writelines(lines[j])
        tmp_seed_data.close()
        #eval_model_test.eval_model(seed_file=tmp_seed_data_path, out_path=out_path, states_path=states_path)
        eval_model_test.eval_model(model_path=model_path, vocab_path=vocab_path, seed_file=tmp_seed_data_path,
                                          out_path=out_path, states_path=states_path, save_nyp=True, save_softmax=False)
        BSCov, BTCov = get_BSCov_BTCov(wrapper, states_path)
        cov_list.append(float(BSCov))
        os.remove(states_path)

    select_num = int(0.1 * len(lines))
    sorted_arg = np.argsort(cov_list)[-select_num:]
    for item in sorted_arg:
        print(lines[item])
        seed_data.write(lines[item])


def gini_guided():
    test_data_path = "/media/cici/BackupDrive/data/atis/bert_aug-1/valid"
    out_path = "/home/cici/major/DialTest/data/out.txt"
    model_path = "/home/cici/major/DialTest/save_profile/atis-lstm/model"
    vocab_path = "/home/cici/major/DialTest/save_profile/atis-lstm/vocab"

    seed_data_path = "/home/cici/major/DialTest/data/atis/valid_bert_gini1"
    in_test_data = open(test_data_path, mode='r', encoding='utf-8')
    seed_data = open(seed_data_path, mode='a', encoding='utf-8')

    lines = in_test_data.readlines()
    gini_list = []
    for j, line in enumerate(lines):
        tmp_seed_data_path = "/home/cici/major/DialTest/data/seed_tmp"
        tmp_seed_data = open(tmp_seed_data_path, mode='w', encoding='utf-8')
        tmp_seed_data.writelines(lines[j])
        tmp_seed_data.close()
        gini = eval_model_test.eval_model(model_path=model_path, vocab_path=vocab_path, seed_file=tmp_seed_data_path,
                                          out_path=out_path, states_path="", save_nyp=False, save_softmax=True)
        float_gini = gini.item()
        gini_list.append(float(float_gini))
    select_num = int(0.1 * len(lines))
    sorted_arg = np.argsort(gini_list)[-select_num:]
    for item in sorted_arg:
        print(lines[item])
        seed_data.write(lines[item])


def gini_and_cov_guided():
    wrapper = "/home/cici/major/DialTest/save_profile/nlu-model/facebook_abst_train_model/wrapper_lstm_nlu_3_10.pkl"
    test_data_path = "/media/cici/BackupDrive/data/facebook/mix_aug/test"
    out_path = "/home/cici/major/DialTest/data/out.txt"
    model_path = "/home/cici/major/DialTest/save_profile/facebook-lstm/model"
    vocab_path = "/home/cici/major/DialTest/save_profile/facebook-lstm/vocab"

    seed_data_path_gini = "/media/cici/BackupDrive/data/facebook/mix_aug/gini_test"
    seed_data_path_cov = "/media/cici/BackupDrive/data/facebook/mix_aug/cov_test"
    in_test_data = open(test_data_path, mode='r', encoding='utf-8')
    seed_data_gini = open(seed_data_path_gini, mode='a', encoding='utf-8')
    seed_data_cov = open(seed_data_path_cov, mode='a', encoding='utf-8')

    lines = in_test_data.readlines()
    gini_list = []
    cov_list = []
    for j, line in enumerate(lines):
        states_path = "/home/cici/major/DialTest/data/facebook/states/out_" + str(j) + ".npy"
        tmp_seed_data_path = "/home/cici/major/DialTest/data/seed_tmp"
        tmp_seed_data = open(tmp_seed_data_path, mode='w', encoding='utf-8')
        tmp_seed_data.writelines(lines[j])
        tmp_seed_data.close()
        gini = eval_model_test.eval_model(model_path=model_path, vocab_path=vocab_path, seed_file=tmp_seed_data_path,
                                   out_path=out_path, states_path=states_path, save_nyp=True, save_softmax=True)
        float_gini = gini.item()
        gini_list.append(float(float_gini))
        BSCov, BTCov = get_BSCov_BTCov(wrapper, states_path)
        cov_list.append(float(BSCov))
        os.remove(states_path)

    # select_num = int(len(lines)*0.3)
    select_num = 8621
    sorted_arg1 = np.argsort(gini_list)[-select_num:]
    for item in sorted_arg1:
        print(lines[item])
        seed_data_gini.write(lines[item])

    sorted_arg2 = np.argsort(cov_list)[-select_num:]
    for item in sorted_arg2:
        print(lines[item])
        seed_data_cov.write(lines[item])


if __name__ == '__main__':
    # gini_guided()
    # cov_guided()
    gini_and_cov_guided()
