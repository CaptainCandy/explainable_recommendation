import tensorflow.compat.v1 as tf
import numpy as np
import dill as pickle
import time
import json
import matplotlib.pyplot as plt
import BARRE
import wordcloud
from tqdm import tqdm


dataset_name = "music"
tf.flags.DEFINE_string("valid_data", "../data/%s_bert/%s.test" % (dataset_name, dataset_name), " Data for validation")
tf.flags.DEFINE_string("para_data", "../data/%s_bert/%s.para" % (dataset_name, dataset_name), "Data parameters")
tf.flags.DEFINE_string("train_data", "../data/%s_bert/%s.train" % (dataset_name, dataset_name), "Data for training")
tf.flags.DEFINE_string("u_text_embeds_input", "../data/%s_bert/u_text_embeds_input.npy" % dataset_name, "User text embedding")
tf.flags.DEFINE_string("i_text_embeds_input", "../data/%s_bert/i_text_embeds_input.npy" % dataset_name, "Item text embedding")
u_text_path = "../data/%s_bert/user_review" % dataset_name
i_text_path = "../data/%s_bert/item_review" % dataset_name

tf.flags.DEFINE_integer("embedding_dim", 768, "Dimensionality of character embedding ")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

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
            print("price:", info["price"])
            print("\n")
        except KeyError:
            pass


def make_wordcloud(s, test_id, reviewerID, asin, type, stop_words):
    w = wordcloud.WordCloud(width=800, height=400, background_color='white', stopwords=stop_words)
    w.generate(s)
    w.to_file("../example/%s_%s_%s_%s_%s_%s.png" % (dataset_name, type, test_id, reviewerID, asin, time_str))


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading trained model...")
    print(FLAGS.para_data)
    para = pickle.load(open(FLAGS.para_data, 'rb'))
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    train_length = para['train_length']
    test_length = para['test_length']

    u_text_embeds = np.load(FLAGS.u_text_embeds_input, allow_pickle=True)
    i_text_embeds = np.load(FLAGS.i_text_embeds_input, allow_pickle=True)
    u_text_embeds = dict(u_text_embeds.item())
    i_text_embeds = dict(i_text_embeds.item())

    u_text = pickle.load(open(u_text_path, "rb"))
    i_text = pickle.load(open(i_text_path, "rb"))

    jsf = open('../data/%s/id2reviewerID.json' % dataset_name, 'r')
    id2reviewerID = json.loads(jsf.readline())
    jsf = open('../data/%s/id2asin.json' % dataset_name, 'r')
    id2asin = json.loads(jsf.readline())

    # jsf = open('../data/All_Amazon_Meta.json', 'r')
    jsf = open('../data/%s/meta_Digital_Music.json' % dataset_name, 'r')
    meta = {}
    for line in jsf:
        js = json.loads(line)
        meta[js["asin"]] = js
    jsf.close()

    random_seed = 2021
    print("user_num", user_num)
    print("item_num", item_num)
    print("user_num_real", len(u_text_embeds))
    print("item_num_real", len(i_text_embeds))
    print("review_num_u", review_num_u)
    print("review_num_i", review_num_i)
    print("train_length", train_length)
    print("test_length", test_length)

    test_data = pickle.load(open(FLAGS.valid_data, 'rb'))
    test_data = np.array(test_data)

    # 要查看的某个测试点
    test_id = 600
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
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    # sess = tf.Session(config=session_conf)

    with tf.Session(config=session_conf) as sess:
        deep = BARRE.BARRE(
            review_num_u=review_num_u,
            review_num_i=review_num_i,
            user_num=user_num,
            item_num=item_num,
            num_classes=1,
            embedding_size=FLAGS.embedding_dim,
            embedding_id=32,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            attention_size=32,
            n_latent=32)
        tf.set_random_seed(random_seed)

        # saver = tf.train.import_meta_graph('./checkpoints/NARRE_instruments_2021-02-08_22h47m03s.ckpt-50102.meta')
        saver = tf.train.Saver()
        saver.restore(sess, "./checkpoints/BARRE_music_2021-02-23_22h27m53s.ckpt-72720")

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

        stop_words = ["music", "song", "songs", "pink", "moon", "album", "albums", "quot", "null"] + list(wordcloud.STOPWORDS)
        make_wordcloud(";".join(sentences), test_id, current_user_ID, current_item_asin, "all", stop_words)
        make_wordcloud(";".join(item_reviews[:5]), test_id, current_user_ID, current_item_asin, "ranked", stop_words)
        print(item_reviews)