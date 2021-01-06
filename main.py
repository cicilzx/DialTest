from cov.coverage_analyzer import get_BSCov_BTCov
from eval import eval_model_test
import os


if __name__ == '__main__':
    wrapper = "/home/cici/major/DialTest/save_profile/nlu-model/snips_abst_train_model/wrapper_lstm_nlu_3_10.pkl"
    test_data_path = "/home/cici/Desktop/snips/standard/test"
    out_path = "/home/cici/major/DialTest/data/snips/out/out.txt"

    BSCov_now, BTCov_now = 0, 0
    seed_data_path = "/home/cici/major/DialTest/data/snips/seed"
    in_test_data = open(test_data_path, mode='r', encoding='utf-8')
    seed_data = open(seed_data_path, mode='a', encoding='utf-8')

    lines = in_test_data.readlines()
    for j, line in enumerate(lines):
        states_path = "/home/cici/major/DialTest/data/snips/states/out_" + str(j) + ".npy"
        tmp_seed_data_path = "/home/cici/major/DialTest/data/snips/seed_tmp"
        tmp_seed_data = open(tmp_seed_data_path, mode='w', encoding='utf-8')
        tmp_seed_data.writelines(lines[:j+1])
        tmp_seed_data.close()
        eval_model_test.eval_model(seed_file=tmp_seed_data_path, out_path=out_path, states_path=states_path)
        BSCov, BTCov = get_BSCov_BTCov(wrapper, states_path)
        if BSCov - BSCov_now >= 0.005 or BTCov - BTCov_now >= 0.0005:
            seed_data.write(line)
            BSCov_now = BSCov
            BTCov_now = BTCov
            print("----------------------------")
            print("line num: ", j, ", new BSCov: ", BSCov_now)
            print("line num: ", j, ", new BTCov: ", BTCov_now)
        else:
            os.remove(states_path)
            pass
