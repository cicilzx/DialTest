import nltk
import tensorflow as tf
# from transformers import *
import heapq
from tensorflow.python.ops.gen_math_ops import mod
import numpy as np
from util import read_file
from bert import modeling as modeling, tokenization
from collections import defaultdict

print(tf.__version__)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

    return logits


class BertAugmentor(object):
    def __init__(self, model_dir, beam_size=5):
        self.beam_size = beam_size  # Each sentence with mask generates at most beam_size transformed senetences.
        # bert's setting profile
        self.bert_config_file = model_dir + 'bert_config.json'
        self.init_checkpoint = model_dir + 'bert_model.ckpt'
        self.bert_vocab_file = model_dir + 'vocab.txt'
        self.bert_config = modeling.BertConfig.from_json_file(
            self.bert_config_file)
        # self.token = tokenization.CharTokenizer(vocab_file=self.bert_vocab_file)
        self.token = tokenization.FullTokenizer(vocab_file=self.bert_vocab_file)
        self.mask_token = "[MASK]"
        self.mask_id = self.token.convert_tokens_to_ids([self.mask_token])[0]
        self.cls_token = "[CLS]"
        self.cls_id = self.token.convert_tokens_to_ids([self.cls_token])[0]
        self.sep_token = "[SEP]"
        self.sep_id = self.token.convert_tokens_to_ids([self.sep_token])[0]

        self.build()
        # sess init
        self.build_sess()

    def __del__(self):
        self.close_sess()

    def build(self):
        # placeholder
        self.input_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(
            tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='segment_ids')
        self.masked_lm_positions = tf.placeholder(
            tf.int32, shape=[None, None], name='masked_lm_positions')

        # Initialize BERT
        self.model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        self.masked_logits = get_masked_lm_output(
            self.bert_config, self.model.get_sequence_output(), self.model.get_embedding_table(),
            self.masked_lm_positions)
        self.predict_prob = tf.nn.softmax(self.masked_logits, axis=-1)

        # Load BERT model
        tvars = tf.trainable_variables()
        (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment)

    def build_sess(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def close_sess(self):
        # self.sess.close()
        pass

    def predict_single_mask(self, word_ids: list, mask_index: int, prob: float = None):
        """Enter a sentence token id list, and return self.beam_size candidate words and prob for the possible content of the mask_indexth mask"""
        word_ids_out = []
        word_mask = [1] * len(word_ids)
        word_segment_ids = [0] * len(word_ids)
        fd = {self.input_ids: [word_ids], self.input_mask: [word_mask], self.segment_ids: [
            word_segment_ids], self.masked_lm_positions: [[mask_index]]}
        mask_probs = self.sess.run(self.predict_prob, feed_dict=fd)
        for mask_prob in mask_probs:
            mask_prob = mask_prob.tolist()
            max_num_index_list = map(mask_prob.index, heapq.nlargest(self.beam_size, mask_prob))
            for i in max_num_index_list:
                if prob and mask_prob[i] < prob:
                    continue
                cur_word_ids = word_ids.copy()
                cur_word_ids[mask_index] = i
                word_ids_out.append([cur_word_ids, mask_prob[i]])
        return word_ids_out

    def predict_batch_mask(self, query_ids: list, mask_indexes: int, prob: float = 0.5):
        """Enter multiple token id lists, and return self.beam_size candidate words and prob for the possible content of the mask_index-th mask
        word_ids: [word_ids1:list, ], shape=[batch, query_lenght]
        mask_indexes: The mask_id to be predicted by the query, [[mask_id], ...], shape=[batch, 1, 1]
        """
        word_ids_out = []
        word_mask = [[1] * len(x) for x in query_ids]
        word_segment_ids = [[1] * len(x) for x in query_ids]
        fd = {self.input_ids: query_ids, self.input_mask: word_mask, self.segment_ids:
            word_segment_ids, self.masked_lm_positions: mask_indexes}
        mask_probs = self.sess.run(self.predict_prob, feed_dict=fd)
        for mask_prob, word_ids_, mask_index in zip(mask_probs, query_ids, mask_indexes):
            # each query of batch
            cur_out = []
            mask_prob = mask_prob.tolist()
            max_num_index_list = map(mask_prob.index, heapq.nlargest(self.n_best, mask_prob))
            for i in max_num_index_list:
                cur_word_ids = word_ids_.copy()
                cur_word_ids[mask_index[0]] = i
                cur_out.append([cur_word_ids, mask_prob[i]])
            word_ids_out.append(cur_out)
        return word_ids_out

    def gen_sen(self, word_ids: list, indexes: list):
        """
        The input is a word id list, which contains a mask, and produces corresponding words for the mask.
        Because the number of masks for each query is inconsistent, the prediction test is inconsistent, and it needs to be predicted separately.
        """
        out_arr = []
        for i, index_ in enumerate(indexes):
            if i == 0:
                out_arr = self.predict_single_mask(word_ids, index_)
            else:
                tmp_arr = out_arr.copy()
                out_arr = []
                for word_ids_, prob in tmp_arr:
                    cur_arr = self.predict_single_mask(word_ids_, index_)
                    cur_arr = [[x[0], x[1] * prob] for x in cur_arr]
                    out_arr.extend(cur_arr)
                # select the first beam size sentences
                out_arr = sorted(out_arr, key=lambda x: x[1], reverse=True)[:self.beam_size]
        for i, (each, _) in enumerate(out_arr):
            covert_ids_to_tokens_list = self.token.convert_ids_to_tokens(each)
            query_src = covert_ids_to_tokens_list
            # query_src = [x.convert_ids_to_tokens() for x in each]
            out_arr[i][0] = query_src
        return out_arr

    def word_insert(self, query):
        """Randomly mask some words and use bert to generate the content of the mask.

        max_query： The maximum number of all queries generated.
        """
        out_arr = []
        seg_list = query.split(' ')
        # Randomly select non-stop word mask.
        i, index_arr = 1, [1]
        for each in seg_list:
            # i += len(each)
            i += 1
            index_arr.append(i)
        # query to id
        split_tokens = self.token.tokenize(query)
        word_ids = self.token.convert_tokens_to_ids(split_tokens)
        word_ids.insert(0, self.cls_id)
        word_ids.append(self.sep_id)
        word_ids_arr, word_index_arr = [], []
        # Randomly insert n characters, 1<=n<=3
        for index_ in index_arr:
            insert_num = np.random.randint(1, 4)
            word_ids_ = word_ids.copy()
            word_index = []
            for i in range(insert_num):
                word_ids_.insert(index_, self.mask_id)
                word_index.append(index_ + i)
            word_ids_arr.append(word_ids_)
            word_index_arr.append(word_index)
        for word_ids, word_index in zip(word_ids_arr, word_index_arr):
            arr_ = self.gen_sen(word_ids, indexes=word_index)
            out_arr.extend(arr_)
            pass
        # This is the first beam size among all generated sentences.
        out_arr = sorted(out_arr, key=lambda x: x[1], reverse=True)
        out_arr = [" ".join(x[0][1:-1][:-1]) + " " + x[0][1:-1][-1] for x in out_arr[:self.beam_size]]
        return out_arr

    def word_replace(self, query):
        """Randomly mask some words and use bert to generate the content of the mask."""
        out_arr = []
        seg_list = query.split(' ')
        # Randomly select non-stop word mask.
        i, index_map = 1, {}
        for each in seg_list:
            # index_map[i] = len(each)
            # i += len(each)
            index_map[i] = 1
            i += 1
        # query to id
        split_tokens = self.token.tokenize(query)
        word_ids = self.token.convert_tokens_to_ids(split_tokens)
        word_ids.insert(0, self.cls_id)
        word_ids.append(self.sep_id)
        word_ids_arr, word_index_arr = [], []
        # Mask words sequentially
        for index_, word_len in index_map.items():
            word_ids_ = word_ids.copy()
            word_index = []
            for i in range(word_len):
                word_ids_[index_ + i] = self.mask_id
                word_index.append(index_ + i)
            word_ids_arr.append(word_ids_)
            word_index_arr.append(word_index)
        for word_ids, word_index in zip(word_ids_arr, word_index_arr):
            arr_ = self.gen_sen(word_ids, indexes=word_index)
            out_arr.extend(arr_)
            pass
        out_arr = sorted(out_arr, key=lambda x: x[1], reverse=True)
        out_arr = [" ".join(x[0][1:-1][:-1]) + " " + x[0][1:-1][-1] for x in out_arr[:self.beam_size]]
        return out_arr

    def insert_word2queries(self, query, beam_size=10):
        self.beam_size = beam_size
        out_map = defaultdict(list)
        out_map[query] = self.word_insert(query)
        return out_map

    def replace_word2queries(self, queries: list, beam_size=10):
        self.beam_size = beam_size
        out_map = defaultdict(list)
        for query in queries:
            out_map[query] = self.word_replace(query)
        return out_map


def augment(in_file, out_file, model_dir):
    """
    file_: Input file, each line is a sentence
    model_dir: bert pretrained model，it can be downloaded at https://github.com/google-research/bert
    """
    if not model_dir:
        raise Exception("must feed params:[model_dir]")
    f = open(in_file)

    mask_model = BertAugmentor(model_dir)

    # ------Random insertion: predict possible words by inserting a mask randomly------
    out = open(out_file, 'a', encoding='utf-8')
    for query in f.readlines():
        if query == "\n":
            out.write("\n")
        elif query != "\n":
            insert_result = mask_model.insert_word2queries(query, beam_size=1)
            print("Augmentor's result:", insert_result)
            # write to file
            for query, v in insert_result.items():
                print('\n'.join((v)))
                out.write(v[0] + "\n")


if __name__ == "__main__":
    # bert pretrained model，it can be downloaded at https://github.com/google-research/bert
    model_dir = '/home/cici/major/NLPDataAugmentation/wwm_cased_L-24_H-1024_A-16/'
    read_file_path = "../dataset/test/test_clean"
    write_file_path = "../dataset/test/test_clean.wiaug"
    augment(read_file_path, write_file_path, model_dir=model_dir)
    print("Finished!")
