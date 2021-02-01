import os
from cov.coverage import Coverage
import numpy as np


def fuzzing_analyzer(dtmc_wrapper_f, states_data_path):
    BSCov, BTCov = 0, 0
    states = np.load(states_data_path)
    coverage_handlers = []

    for criteria, k_step in [("state", 0), ("transition", 0)]:  # , ("k-step", 3), ("k-step", 6)
        cov = Coverage(dtmc_wrapper_f, criteria, k_step)
        coverage_handlers.append(cov)

    for i, coverage_handler in enumerate(coverage_handlers):
        cov = coverage_handler.get_coverage_criteria(states)
        total = coverage_handler.get_total()
        if i == 0:
            print("Basic State Coverage(BSCov):", len(cov) / total)
            BSCov = len(cov) / total
        if i == 1:
            print("Basic Transition Coverage(BTCov):", len(cov) / total)
            BTCov = len(cov) / total
        if coverage_handler.mode != "k-step":  # to printout the weighted coverage metrics
            weight_dic = coverage_handler.get_weight_dic()
            # print(sum([weight_dic[e] for e in cov]))
            rev_weight_dic = coverage_handler.get_weight_dic(reverse=True)
            # print(sum([rev_weight_dic[e] for e in cov]))
    return BSCov, BTCov


def get_BSCov_BTCov(wrapper, states_data):
    return fuzzing_analyzer(wrapper, states_data)


if __name__ == '__main__':
    wrapper = "../save_profile/nlu-model/snips_abst_train_model/wrapper_lstm_nlu_3_10.pkl"
    states_data = "../save_profile/nlu-model/snips_test_states/out_0_batch_0.npy"
    BSCov, BTCov = get_BSCov_BTCov(wrapper, states_data)
    print("BSCov: ", BSCov, "BTCov: ", BTCov)
