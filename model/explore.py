import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import NARRE
from tqdm import tqdm


dataset_name = "music"
tf.flags.DEFINE_string("word2vec", "../data/GoogleNews-vectors-negative300.txt",
                       "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("valid_data", "../data/%s/%s.test" % (dataset_name, dataset_name), " Data for validation")
tf.flags.DEFINE_string("para_data", "../data/%s/%s.para" % (dataset_name, dataset_name), "Data parameters")
tf.flags.DEFINE_string("vocab_data", "../data/%s/%s.vocab" % (dataset_name, dataset_name), "Id2word")
tf.flags.DEFINE_string("train_data", "../data/%s/%s.train" % (dataset_name, dataset_name), "Data for training")

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 96, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

time_str = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading trained model...")
    print(FLAGS.para_data)
    pkl_file = open(FLAGS.para_data, 'rb')
    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    review_len_u = para['review_len_u']
    review_len_i = para['review_len_i']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']

    pkl_file = open(FLAGS.vocab_data, 'rb')
    vocab_inv = pickle.load(pkl_file)
    vocabulary_user_inv = vocab_inv['user']
    vocabulary_item_inv = vocab_inv['item']

    random_seed = 2021
    print(user_num)
    print(item_num)
    print(review_num_u)
    print(review_len_u)
    print(review_num_i)
    print(review_len_i)

    pkl_file = open(FLAGS.valid_data, 'rb')
    test_data = pickle.load(pkl_file)
    test_data = np.array(test_data)
    pkl_file.close()

    # 要查看的某个测试点
    test_id = 200
    test_data = test_data[test_id:test_id+1]
    userid_test, itemid_test, reuid, reiid, y_test = zip(*test_data)

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    with tf.Session() as sess:
        deep = NARRE.NARRE(
            review_num_u=review_num_u,
            review_num_i=review_num_i,
            review_len_u=review_len_u,
            review_len_i=review_len_i,
            user_num=user_num,
            item_num=item_num,
            num_classes=1,
            user_vocab_size=len(vocabulary_user),
            item_vocab_size=len(vocabulary_item),
            embedding_size=FLAGS.embedding_dim,
            embedding_id=32,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            attention_size=32,
            n_latent=32)
        tf.set_random_seed(random_seed)

        # saver = tf.train.import_meta_graph('./checkpoints/NARRE_instruments_2021-02-08_22h47m03s.ckpt-50102.meta')
        saver = tf.train.Saver()
        saver.restore(sess, "./checkpoints/NARRE_music_2021-02-12_21h22m27s.ckpt-148810")

        u_batch = [u_text[userid_test[0][0]]]
        i_batch = [i_text[itemid_test[0][0]]]
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

        pred_ratings, item_attention, loss, rmse, mae = sess.run([deep.predictions, deep.i_a, deep.loss, deep.accuracy, deep.mae],
                                                 feed_dict)
        print("pred:", pred_ratings[0][0], "y:", y_batch[0][0], "\n", loss, rmse, mae)

        item_attention = np.array(item_attention).flatten()
        print(item_attention)
        sortidx = np.argsort(item_attention)[::-1]

        sentences = i_text[itemid_test[0][0]]
        item_reviews = []
        for sentence in sentences:
            if sentence[0] == 1:
                item_reviews.append("<null>")
                continue
            review = ""
            for id in sentence:
                if id == 1:
                    break
                review += str(vocabulary_item_inv[id])
                review += " "
            item_reviews.append(review)
        item_reviews = np.array(item_reviews)[sortidx]

        print(item_reviews[:6])