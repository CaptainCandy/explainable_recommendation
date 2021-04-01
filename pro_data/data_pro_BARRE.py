import numpy as np
import re
import itertools
from collections import Counter
from tqdm import tqdm

import tensorflow.compat.v1 as tf
import csv
import dill as pickle
import os

dataset_name = "music"
embedding_size = 768
tf.flags.DEFINE_string("valid_data", "../data/%s_bert/%s_valid.csv" % (dataset_name, dataset_name), " Data for validation")
tf.flags.DEFINE_string("test_data", "../data/%s_bert/%s_test.csv" % (dataset_name, dataset_name), "Data for testing")
tf.flags.DEFINE_string("train_data", "../data/%s_bert/%s_train.csv" % (dataset_name, dataset_name), "Data for training")
tf.flags.DEFINE_string("user_review", "../data/%s_bert/user_review" % dataset_name, "User's reviews")
tf.flags.DEFINE_string("item_review", "../data/%s_bert/item_review" % dataset_name, "Item's reviews")
tf.flags.DEFINE_string("user_review_embeds", "../data/%s_bert/user_reviews_embeddings" % dataset_name, "User's reviews embeddings")
tf.flags.DEFINE_string("item_review_embeds", "../data/%s_bert/item_reviews_embeddings" % dataset_name, "Item's reviews embeddings")
tf.flags.DEFINE_string("user_review_id", "../data/%s_bert/user_rid" % dataset_name, "user_review_id")
tf.flags.DEFINE_string("item_review_id", "../data/%s_bert/item_rid" % dataset_name, "item_review_id")
tf.flags.DEFINE_string("stopwords", "../data/stopwords", "stopwords")


# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()


# def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):
#     """
#     Pads all sentences to the same length. The length is defined by the longest sentence.
#     Returns padded sentences.
#     """
#     review_num = u_len
#     review_len = u2_len
#
#     u_text2 = {}
#     for i in u_text.keys():
#         u_reviews = u_text[i]
#         padded_u_train = []
#         for ri in range(review_num):
#             if ri < len(u_reviews):
#                 sentence = u_reviews[ri]
#                 if review_len > len(sentence):
#                     num_padding = review_len - len(sentence)
#                     new_sentence = sentence + [padding_word] * num_padding
#                     padded_u_train.append(new_sentence)
#                 else:
#                     new_sentence = sentence[:review_len]
#                     padded_u_train.append(new_sentence)
#             else:
#                 new_sentence = [padding_word] * review_len
#                 padded_u_train.append(new_sentence)
#         u_text2[i] = padded_u_train
#
#     return u_text2


def pad_reviewid(u_train, u_valid, u_len, num):
    pad_u_train = []

    for i in range(len(u_train)):
        x = u_train[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_train.append(x)
    pad_u_valid = []

    for i in range(len(u_valid)):
        x = u_valid[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_valid.append(x)
    return pad_u_train, pad_u_valid


def pad_embeddings(u_text_embeds, i_text_embeds, u_len, i_len):
    u_text_embeds2 = {}
    for key in u_text_embeds.keys():
        u_embeds = u_text_embeds[key]
        padded_one_u = []
        count = 0
        for i in u_embeds:
            if count < u_len:
                padded_one_u.append(i)
            else:
                break
            count += 1
        for i in range(u_len - count):
            padded_one_u.append([1] * embedding_size)
        u_text_embeds2[key] = padded_one_u

    i_text_embeds2 = {}
    for key in i_text_embeds.keys():
        i_embeds = i_text_embeds[key]
        padded_one_i = []
        count = 0
        for i in i_embeds:
            if count < i_len:
                padded_one_i.append(i)
            else:
                break
            count += 1
        for i in range(i_len - count):
            padded_one_i.append([1] * embedding_size)
        i_text_embeds2[key] = np.array(padded_one_i)

    return u_text_embeds2, i_text_embeds2


# def build_vocab(sentences1, sentences2):
#     """
#     Builds a vocabulary mapping from word to index based on the sentences.
#     Returns vocabulary mapping and inverse vocabulary mapping.
#     """
#     # Build vocabulary
#     word_counts1 = Counter(itertools.chain(*sentences1))
#     # Mapping from index to word
#     vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
#     vocabulary_inv1 = list(sorted(vocabulary_inv1))
#     # Mapping from word to index
#     vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}
#
#     word_counts2 = Counter(itertools.chain(*sentences2))
#     # Mapping from index to word
#     vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
#     vocabulary_inv2 = list(sorted(vocabulary_inv2))
#     # Mapping from word to index
#     vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
#
#     vocab_list1 = list(vocabulary1.keys())
#     vocab_file = open("../data/%s/vocabulary_all.txt" % dataset_name, "w")
#     vocab_list2 = list(vocabulary2.keys())
#     for word in vocab_list2:
#         if word not in vocabulary1:
#             vocab_list1.append(word)
#     for word in vocab_list1:
#         vocab_file.write(word)
#         vocab_file.write('\n')
#     vocab_file.close()
#     print("len of vocabulary_all: ", len(vocab_list1))
#
#     return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    就是把一个个review的单词换成vocab里的id
    """
    l = len(u_text)
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([[vocabulary_u[word] for word in words] for words in u_reviews])
        u_text2[i] = u
    l = len(i_text)
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([[vocabulary_i[word] for word in words] for words in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2


def load_data(train_data, valid_data, user_review, item_review, user_rid, item_rid,
              user_review_embeds, item_review_embeds, stopwords):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    y_train, y_valid, u_len, i_len, uid_train, iid_train, uid_valid \
        , iid_valid, user_num, item_num \
        , reid_user_train, reid_item_train, reid_user_valid, reid_item_valid\
        , u_text_embeds, i_text_embeds = \
        load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid,
                             user_review_embeds, item_review_embeds, stopwords)
    print("load data done")
    # u_text = pad_sentences(u_text, u_len, u2_len)
    u_text_embeds, i_text_embeds = pad_embeddings(u_text_embeds, i_text_embeds, u_len, i_len)
    print("pad embedding done")
    reid_user_train, reid_user_valid = pad_reviewid(reid_user_train, reid_user_valid, u_len, item_num + 1)
    print("pad user done")
    # i_text = pad_sentences(i_text, i_len, i2_len)
    reid_item_train, reid_item_valid = pad_reviewid(reid_item_train, reid_item_valid, i_len, user_num + 1)
    print("pad item done")

    # user_voc = [xx for x in u_text.values() for xx in x]
    # item_voc = [xx for x in i_text.values() for xx in x]

    # vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    # print("len of vocabulary_user %s." % len(vocabulary_user))
    # print("len of vocabulary_item %s." % len(vocabulary_item))
    # u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)
    u_text_embeds = np.array(u_text_embeds)
    i_text_embeds = np.array(i_text_embeds)

    return [y_train, y_valid,
            uid_train, iid_train,
            uid_valid, iid_valid,
            u_len, i_len,
            user_num, item_num,
            reid_user_train, reid_item_train, reid_user_valid, reid_item_valid,
            u_text_embeds, i_text_embeds]


def load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid,
                         user_review_embeds, item_review_embeds, stopwords):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    f_train = open(train_data, "r")
    # f1 = open(user_review, 'rb')
    # f2 = open(item_review, 'rb')
    f3 = open(user_rid, 'rb')
    f4 = open(item_rid, 'rb')
    f5 = open(user_review_embeds, 'rb')
    f6 = open(item_review_embeds, 'rb')

    # user_reviews = pickle.load(f1)
    # item_reviews = pickle.load(f2)
    user_rids = pickle.load(f3)
    item_rids = pickle.load(f4)
    user_review_embeds = pickle.load(f5)
    item_review_embeds = pickle.load(f6)

    reid_user_train = []
    reid_item_train = []
    uid_train = []
    iid_train = []
    y_train = []
    # u_text = {}
    u_rid = {}
    # i_text = {}
    i_rid = {}
    u_text_embeds = {}
    i_text_embeds = {}

    for line in tqdm(f_train):
        line = line.split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        if int(line[0]) in u_text_embeds:
            reid_user_train.append(u_rid[int(line[0])])
        else:
            # u_text[int(line[0])] = []
            # for s in user_reviews[int(line[0])]:
            #     u_text[int(line[0])].append(s)
            u_rid[int(line[0])] = []
            for s in user_rids[int(line[0])]:
                u_rid[int(line[0])].append(int(s))
            u_text_embeds[int(line[0])] = []
            for s in user_review_embeds[int(line[0])]:
                u_text_embeds[int(line[0])].append(s)
            reid_user_train.append(u_rid[int(line[0])])

        if int(line[1]) in i_text_embeds:
            reid_item_train.append(i_rid[int(line[1])])
        else:
            # i_text[int(line[1])] = []
            # for s in item_reviews[int(line[1])]:
            #     i_text[int(line[1])].append(s)
            i_rid[int(line[1])] = []
            for s in item_rids[int(line[1])]:
                i_rid[int(line[1])].append(int(s))
            i_text_embeds[int(line[1])] = []
            for s in item_review_embeds[int(line[1])]:
                i_text_embeds[int(line[1])].append(s)
            reid_item_train.append(i_rid[int(line[1])])

        y_train.append(float(line[2]))
    print("finish train data.")

    reid_user_valid = []
    reid_item_valid = []
    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data, "r")
    for line in tqdm(f_valid):
        line = line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        if int(line[0]) in u_text_embeds:
            reid_user_valid.append(u_rid[int(line[0])])
        else:
            # bert vocab的第一个
            # u_text[int(line[0])] = [["[PAD]"]]
            # u_text[int(line[0])] = []
            # for s in user_reviews[int(line[0])]:
            #     u_text[int(line[0])].append(s)
            # 全1的向量不影响计算
            u_text_embeds[int(line[0])] = user_review_embeds[int(line[0])]
            u_rid[int(line[0])] = user_rids[int(line[0])]
            reid_user_valid.append(u_rid[int(line[0])])

        if int(line[1]) in i_text_embeds:
            reid_item_valid.append(i_rid[int(line[1])])
        else:
            # i_text[int(line[1])] = [["[PAD]"]]
            # i_text[int(line[1])] = []
            # for s in item_reviews[int(line[1])]:
            #     i_text[int(line[1])].append(s)
            i_text_embeds[int(line[1])] = item_review_embeds[int(line[1])]
            i_rid[int(line[1])] = item_rids[int(line[1])]
            reid_item_valid.append(i_rid[int(line[1])])

        y_valid.append(float(line[2]))
    print("finish valid data.")

    review_num_u = np.array([len(x) for x in u_text_embeds.values()])
    x = np.sort(review_num_u)
    u_len = x[int(0.9 * len(review_num_u)) - 1]

    review_num_i = np.array([len(x) for x in i_text_embeds.values()])
    y = np.sort(review_num_i)
    i_len = y[int(0.9 * len(review_num_i)) - 1]

    print("u_len:", u_len)
    print("i_len:", i_len)
    user_num = len(u_text_embeds)
    item_num = len(i_text_embeds)
    print("user_num:", user_num)
    print("item_num:", item_num)

    return [y_train, y_valid,
            u_len, i_len,
            uid_train, iid_train,
            uid_valid, iid_valid,
            user_num, item_num,
            reid_user_train, reid_item_train,
            reid_user_valid, reid_item_valid,
            u_text_embeds, i_text_embeds]


if __name__ == '__main__':
    TPS_DIR = '../data/%s_bert' % dataset_name
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()

    y_train, y_valid, uid_train, iid_train, uid_valid, iid_valid, u_len, i_len, user_num, item_num, reid_user_train, \
    reid_item_train, reid_user_valid, reid_item_valid, u_text_embeds, i_text_embeds = \
        load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.user_review, FLAGS.item_review, FLAGS.user_review_id,
                  FLAGS.item_review_id, FLAGS.user_review_embeds, FLAGS.item_review_embeds, FLAGS.stopwords)

    np.random.seed(2017)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    reid_user_train = reid_user_train[shuffle_indices]
    reid_item_train = reid_item_train[shuffle_indices]

    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    batches_train = list(zip(userid_train, itemid_train, reid_user_train, reid_item_train, y_train))
    batches_test = list(zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid))
    print('write begin')
    output = open(os.path.join(TPS_DIR, ('%s.train' % dataset_name)), 'wb')
    pickle.dump(batches_train, output)
    output = open(os.path.join(TPS_DIR, ('%s.test' % dataset_name)), 'wb')
    pickle.dump(batches_test, output)

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['review_num_u'] = u_len
    para['review_num_i'] = i_len
    # print(u_text_embeds[0].shape[0], i_text_embeds[0].shape[0])
    # para['user_vocab'] = vocabulary_user
    # para['item_vocab'] = vocabulary_item
    para['train_length'] = len(y_train)
    para['test_length'] = len(y_valid)
    # para['u_text'] = u_text
    # para['i_text'] = i_text
    output = open(os.path.join(TPS_DIR, ('%s.para' % dataset_name)), 'wb')
    pickle.dump(para, output)

    # pickle.dump(u_text_embeds, open(os.path.join(TPS_DIR, 'u_text_embeds_input'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(i_text_embeds, open(os.path.join(TPS_DIR, 'i_text_embeds_input'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(TPS_DIR, 'u_text_embeds_input.npy'), u_text_embeds, allow_pickle=True)
    np.save(os.path.join(TPS_DIR, 'i_text_embeds_input.npy'), i_text_embeds, allow_pickle=True)
    # para['u_text_embeds'] = u_text_embeds
    # para['i_text_embeds'] = i_text_embeds


    # vocab_inv = {}
    # vocab_inv['user'] = vocabulary_inv_user
    # vocab_inv['item'] = vocabulary_inv_item
    # output = open(os.path.join(TPS_DIR, ('%s.vocab' % dataset_name)), 'wb')
    # pickle.dump(vocab_inv, output)
