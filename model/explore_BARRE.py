import tensorflow.compat.v1 as tf
import numpy as np
import dill as pickle
import time
import json
import matplotlib.pyplot as plt
# import sys
# sys.path.append("model")
# sys.path.append("data")
import BARRE
import wordcloud
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


dataset_name = "movies"
valid_data = "../data/%s_bert/%s.test" % (dataset_name, dataset_name)
para_data = "../data/%s_bert/%s.para" % (dataset_name, dataset_name)
train_data = "../data/%s_bert/%s.train" % (dataset_name, dataset_name)
u_text_embeds_input = "../data/%s_bert/u_text_embeds_input.npy" % dataset_name
i_text_embeds_input = "../data/%s_bert/i_text_embeds_input.npy" % dataset_name
u_text_path = "../data/%s_bert/user_review" % dataset_name
i_text_path = "../data/%s_bert/item_review" % dataset_name

embedding_dim = 768
dropout_keep_prob = 0.5
l2_reg_lambda = 0.1
# Training parameters
allow_soft_placement = True
log_device_placement = False

time_str = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))


def print_items_meta(asins, meta):
    for asin in asins:
        try:
            info = meta[asin]
            print("asin:", info["asin"])
            print("title:", info["title"])
            print("brand:", info["brand"])
            print("feature:", info["feature"])
            print("image:", info["image"])
            print("price:", info["price"], "\n")
        except KeyError:
            pass


def write_items_meta(asins, meta, f):
    for asin in asins:
        try:
            info = meta[asin]
            f.write("asin:")
            f.write(info["asin"])
            f.write("\n")
            f.write("title:")
            f.write(info["title"])
            f.write("\n")
            f.write("brand:")
            f.write(info["brand"])
            f.write("\n")
            f.write("feature:")
            f.write(info["feature"])
            f.write("\n")
            f.write("image:")
            f.write(info["image"])
            f.write("\n\n")
        except KeyError:
            pass


def make_wordcloud(s, test_id, reviewerID, asin, type, stop_words):
    w = wordcloud.WordCloud(width=800, height=400, background_color='white', stopwords=stop_words)
    w.generate(s)
    w.to_file("../example/%s_%s_%s_%s_%s_%s.png" % (dataset_name, test_id, type, reviewerID, asin, time_str))


def make_tfidf_wordcloud(top_reviews, corpus, test_id, reviewerID, asin, type, stop_words):
    tfidf = TfidfVectorizer(max_features=500, stop_words=stop_words)  # 默认值
    tfidf.fit(corpus)
    wordlist = tfidf.get_feature_names()
    Ya = tfidf.transform(top_reviews).toarray()
    word_fre = {}
    for i in range(Ya.shape[0]):
        for j in range(Ya.shape[1]):
            if wordlist[j] in word_fre:
                word_fre[wordlist[j]] = max(word_fre[wordlist[j]], Ya[i][j])
            else:
                word_fre[wordlist[j]] = Ya[i][j]

    w = wordcloud.WordCloud(width=800, height=400, background_color='white', stopwords=stop_words)
    w.generate_from_frequencies(word_fre)
    w.to_file("../example/%s_%s_%s_%s_%s_%s.png" % (dataset_name, test_id, type, reviewerID, asin, time_str))


# if __name__ == '__main__':
# FLAGS = tf.flags.FLAGS
# FLAGS.flag_values_dict()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

print("Loading trained model...")
para = pickle.load(open(para_data, 'rb'))
user_num = para['user_num']
item_num = para['item_num']
review_num_u = para['review_num_u']
review_num_i = para['review_num_i']
train_length = para['train_length']
test_length = para['test_length']

u_text_embeds = np.load(u_text_embeds_input, allow_pickle=True)
i_text_embeds = np.load(i_text_embeds_input, allow_pickle=True)
u_text_embeds = dict(u_text_embeds.item())
i_text_embeds = dict(i_text_embeds.item())

u_text = pickle.load(open(u_text_path, "rb"))
i_text = pickle.load(open(i_text_path, "rb"))


jsf = open('../data/%s/id2reviewerID.json' % dataset_name, 'r')
id2reviewerID = json.loads(jsf.readline())
jsf = open('../data/%s/id2asin.json' % dataset_name, 'r')
id2asin = json.loads(jsf.readline())

# jsf = open('../data/All_Amazon_Meta.json', 'r')
# jsf = open('../data/%s/meta_Digital_Music.json' % dataset_name, 'r')
jsf = open('../data/%s/meta_Movies_and_TV.json' % dataset_name, 'r')
meta = {}
for line in jsf:
    js = json.loads(line)
    meta[js["asin"]] = js
jsf.close()

random_seed = 2021
print("user_num", user_num)
print("item_num", item_num)
print("review_num_u", review_num_u)
print("review_num_i", review_num_i)
print("train_length", train_length)
print("test_length", test_length)

#%%
test_data = pickle.load(open(valid_data, 'rb'))
test_data = np.array(test_data)
# 要查看的某个测试点
test_id = 131
print("test_id:", test_id)
test_data = test_data[test_id:test_id+1]
userid_test, itemid_test, reuid, reiid, y_test = zip(*test_data)
item_purchased_asin = []
for id in reuid[0]:
    if id == item_num or id == item_num+1:
        continue
    item_purchased_asin.append(id2asin[str(id)])
current_user_ID = id2reviewerID[str(userid_test[0][0])]
print("user %s historical purchases:" % current_user_ID)
print_items_meta(item_purchased_asin, meta)
current_item_asin = id2asin[str(itemid_test[0][0])]
print("current item %s's information:" % current_item_asin)
print_items_meta([current_item_asin], meta)

session_conf = tf.ConfigProto(
    allow_soft_placement=allow_soft_placement,
    log_device_placement=log_device_placement)
session_conf.gpu_options.allow_growth = True
# sess = tf.Session(config=session_conf)

tf.reset_default_graph()
with tf.Session(config=session_conf) as sess:
    deep = BARRE.BARRE(
        review_num_u=review_num_u,
        review_num_i=review_num_i,
        user_num=user_num,
        item_num=item_num,
        num_classes=1,
        embedding_size=embedding_dim,
        embedding_id=32,
        l2_reg_lambda=l2_reg_lambda,
        attention_size=32,
        n_latent=32)
    tf.set_random_seed(random_seed)

    # saver = tf.train.import_meta_graph('./checkpoints/NARRE_instruments_2021-02-08_22h47m03s.ckpt-50102.meta')
    saver = tf.train.Saver()
    saver.restore(sess, "../model/checkpoints/BARRE_movies_2021-03-05_20h01m43s.ckpt-1043798")

    u_batch = [u_text_embeds[userid_test[0][0]]]
    i_batch = [i_text_embeds[itemid_test[0][0]]]
    y_batch = [y_test[0]]
    u_batch = np.array(u_batch)
    i_batch = np.array(i_batch)

    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: userid_test,
        deep.input_iid: itemid_test,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }

    pred_ratings, item_attention, loss, rmse, mae = \
        sess.run([deep.predictions, deep.i_a, deep.loss, deep.accuracy, deep.mae], feed_dict)
    print("pred:", pred_ratings[0][0], "y:", y_batch[0][0], "\n", loss, rmse, mae)

    item_attention = np.array(item_attention).flatten()
    # print(item_attention)
    sortedidx = np.argsort(item_attention)[::-1]

    # item的所有review
    sentences = i_text[itemid_test[0][0]]
    ori_num_i = len(sentences)
    item_reviews = []
    if ori_num_i >= review_num_i:
        item_reviews = sentences[:review_num_i]
    else:
        for i in range(review_num_i - ori_num_i):
            sentences.append("</null>")
        item_reviews = sentences

    # 排完序的固定数量的item的review
    item_reviews = np.array(item_reviews)[sortedidx]

    # stop_words = ["music", "song", "songs", "pink", "moon", "album", "albums", "quot", "null"] + list(wordcloud.STOPWORDS)
    stop_words = ["movie", "movies", "film", "watch", "show", "shows", "one", "find", "will", "quot", "null",
                  "season", "seasons", "series", "episode", "episodes", ""] + \
                 list(wordcloud.STOPWORDS) + \
                ['aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're',
                 'shan','shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']
    make_wordcloud(";".join(sentences), test_id, current_user_ID, current_item_asin, "all", stop_words)
    make_tfidf_wordcloud(item_reviews[:10], sentences, test_id, current_user_ID, current_item_asin, "ranked", stop_words)
    print(item_reviews)