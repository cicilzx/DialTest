from nltk.corpus import wordnet
import random
from random import choice

random.seed(1)

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']


def synonym_replace(dict_w_l, n):
    words = dict_w_l.keys()
    new_list_dict_w_l = []
    new_words = []
    for key, values in dict_w_l.items():
        if values == ":O":
            new_words.append(key)
    # print(new_words)
    random_word_list = list(set([word for word in new_words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            for synonym in synonyms:
                new_words1 = [synonym if word == random_word else word for word in words]
                new_lable1 = [dict_w_l[random_word] if word == random_word else dict_w_l[word] for word in words]
                new_dict_w_l = dict(zip(new_words1, new_lable1))
                new_list_dict_w_l.append(new_dict_w_l)
                num_replaced += 1
                if num_replaced >= n:
                    break
    return new_list_dict_w_l


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def get_words_and_lables_from_sentence(sentence):
    words = []
    lables = []
    word_lable_pairs = sentence.split(" ")
    for word_lable_pair in word_lable_pairs:
        temp = word_lable_pair.rfind(":")
        w = word_lable_pair[0:temp]
        l = word_lable_pair[temp:]
        words.append(w)
        lables.append(l)
    return dict(zip(words, lables))


def aug_sentece(ori_sentence, aug_num):
    aug = []
    sentence = ori_sentence.split(" <=> ")[0]
    intent = ori_sentence.split(" <=> ")[1]
    dict_word_lable = get_words_and_lables_from_sentence(sentence)
    # print("Original: ", dict_word_lable)

    new_dict_w_l = synonym_replace(dict_word_lable, aug_num)
    for li in new_dict_w_l:
        aug_sentence = ""
        for key, values in li.items():
            aug_sentence = aug_sentence + key + values + " "
        aug.append(aug_sentence + "<=> " + intent)
    return aug


if __name__ == '__main__':
    seed_sentence = "I'd:O like:O to:O have:O this:O track:B-music_item onto:O my:B-playlist_owner " \
                   "Classical:B-playlist Relaxations:I-playlist playlist.:O <=> AddToPlaylist"
    aug = aug_sentece(seed_sentence, aug_num=1)
    print(aug[0])
