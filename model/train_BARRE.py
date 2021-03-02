'''
BARRE
@author:
Xinze Tang

@ created:
23/2/2021
@references:

'''

import os
import numpy as np
import tensorflow.compat.v1 as tf
import dill as pickle
import time
import matplotlib.pyplot as plt
import BARRE
from tensorflow.python import debug as tf_debug
from tqdm import tqdm


dataset_name = "instruments"
tf.flags.DEFINE_string("valid_data", "../data/%s_bert/%s.test" % (dataset_name, dataset_name), " Data for validation")
tf.flags.DEFINE_string("para_data", "../data/%s_bert/%s.para" % (dataset_name, dataset_name), "Data parameters")
tf.flags.DEFINE_string("train_data", "../data/%s_bert/%s.train" % (dataset_name, dataset_name), "Data for training")
tf.flags.DEFINE_string("u_text_embeds_input", "../data/%s_bert/u_text_embeds_input.npy" % dataset_name, "User text embedding")
tf.flags.DEFINE_string("i_text_embeds_input", "../data/%s_bert/i_text_embeds_input.npy" % dataset_name, "Item text embedding")
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 768, "Dimensionality of character embedding")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes ")
# tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 16384, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

time_str = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))


def train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, batch_num):
    """
    A single training step
    """
    # print("u_batch: ", np.any(np.isnan(u_batch)))
    # print("i_batch: ", np.any(np.isnan(i_batch)))
    # print("uid: ", np.any(np.isnan(uid)))
    # print("iid: ", np.any(np.isnan(iid)))
    # print("reuid: ", np.any(np.isnan(reuid)))
    # print("reiid: ", np.any(np.isnan(reiid)))
    # print("y_batch: ", np.any(np.isnan(y_batch)))
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 0.8,
        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae, u_a, i_a, fm = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.u_a, deep.i_a, deep.score],
        feed_dict)
    return accuracy, mae, u_a, i_a, fm


def dev_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)

    return [loss, accuracy, mae]


# def load_word2vec_embedding(vocab_size, embedding_size, vocabulary, type):
#     """
#
#     Args:
#         vocab_size:
#         embedding_size:
#         vocabulary: vocabulary_user or vocabulary_item
#         type: "user" or "item"
#
#     Returns:
#         initW
#     """
#     # print("./%s_initW_%s.npy" % (type, dataset_name))
#     if os.path.exists("./%s_initW_%s.npy" % (type, dataset_name)):
#         initW = np.load("./%s_initW_%s.npy" % (type, dataset_name))
#         return initW
#
#     initW = np.random.uniform(-1.0, 1.0, (vocab_size, embedding_size))
#     # load any vectors from the word2vec
#     print("\nLoad word2vec i file {}\n".format(FLAGS.word2vec))
#     with open(FLAGS.word2vec, "r", encoding="utf-8") as f:
#         header = f.readline()
#         vocab_size_google, layer1_size = map(int, header.split())
#         # binary_len = np.dtype('float32').itemsize * layer1_size
#         for line in tqdm(range(vocab_size_google), ncols=80):
#             word = []
#             while True:
#                 ch = f.read(1)
#                 if ch == ' ':
#                     word = ''.join(word)
#                     break
#                 word.append(ch)
#             if word in vocabulary:
#                 idx = vocabulary[word]
#                 initW[idx] = np.fromstring(f.readline(), dtype='float32', count=FLAGS.embedding_dim)
#             else:
#                 f.readline()
#     np.save("./%s_initW_%s.npy" % (type, dataset_name), initW)
#     return initW


def plot_train_process(rmse_train, rmse_test, mae_train, mae_test):
    # 绘制曲线
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.plot(rmse_train)
    plt.plot(rmse_test)
    plt.legend(('Train RMSE', 'Val RMSE'))
    plt.title("RMSE")
    plt.subplot(122)
    plt.plot(mae_train)
    plt.plot(mae_test)
    plt.title("MAE")
    plt.legend(('Train MAE', 'Val MAE'))
    plt.savefig('./results/%s_%s.jpg' % (dataset_name, time_str))
    plt.close()


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading data...")
    t0 = time.time()
    para = pickle.load(open(FLAGS.para_data, 'rb'))
    u_text_embeds = np.load(FLAGS.u_text_embeds_input, allow_pickle=True)
    i_text_embeds = np.load(FLAGS.i_text_embeds_input, allow_pickle=True)
    u_text_embeds = dict(u_text_embeds.item())
    i_text_embeds = dict(i_text_embeds.item())

    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    # vocabulary_user = para['user_vocab']
    # vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    test_length = para['test_length']
    # u_text = para['u_text']
    # i_text = para['i_text']
    # u_text_embeds = para['u_text_embeds']
    # i_text_embeds = para['i_text_embeds']

    random_seed = 2021
    print("user_num", user_num)
    print("item_num", item_num)
    print("user_num_real", len(u_text_embeds))
    print("item_num_real", len(i_text_embeds))
    print("review_num_u", review_num_u)
    print("review_num_i", review_num_i)
    print("train_length", train_length)
    print("test_length", test_length)
    t1 = time.time()
    print("t1-t0:%.2f" % (t1 - t0))

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        # tensorboard_writer = tf.summary.FileWriter('./logs', sess.graph)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan) # run -f has_inf_or_nan
        with sess.as_default():
            deep = BARRE.BARRE(
                review_num_u=review_num_u,
                review_num_i=review_num_i,
                user_num=user_num,
                item_num=item_num,
                num_classes=1,
                # user_vocab_size=len(vocabulary_user),
                # item_vocab_size=len(vocabulary_item),
                embedding_size=FLAGS.embedding_dim,
                embedding_id=32,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                attention_size=32,
                n_latent=32)
            tf.set_random_seed(random_seed)

            global_step = tf.Variable(0, name="global_step", trainable=False)

            optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999,
                                               epsilon=1e-8).minimize(deep.loss, global_step=global_step)
            # optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(deep.loss, global_step=global_step)
            # optimizer = tf.train.GradientDescentOptimizer(0.01)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.99, momentum=0.0,
            #                                       epsilon=1e-10, use_locking=False, name='RMSProp').minimize(deep.loss, global_step=global_step)

            # params = tf.trainable_variables()
            # gradients = tf.gradients(deep.loss, params)
            # clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5)
            train_op = optimizer  # .apply_gradients(zip(clipped_gradients, params), global_step=global_step)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()

            # if FLAGS.word2vec:
            #     # initial matrix with random uniform
            #
            #     initW = load_word2vec_embedding(len(vocabulary_user), FLAGS.embedding_dim, vocabulary_user, "user")
            #     sess.run(deep.W1.assign(initW))
            #
            #     initW = load_word2vec_embedding(len(vocabulary_item), FLAGS.embedding_dim, vocabulary_item, "item")
            #     sess.run(deep.W2.assign(initW))

            best_mae = 10
            best_rmse = 10
            train_mae = 0
            train_rmse = 0

            pkl_file = open(FLAGS.train_data, 'rb')
            train_data = pickle.load(pkl_file)
            train_data = np.asarray(train_data)
            pkl_file.close()

            pkl_file = open(FLAGS.valid_data, 'rb')
            test_data = pickle.load(pkl_file)
            test_data = np.asarray(test_data)
            pkl_file.close()

            data_size_train = len(train_data)
            data_size_test = len(test_data)
            batch_size = FLAGS.batch_size
            ll = int(len(train_data) / batch_size)
            print("batch_num_all: ", ll)

            # 训练过程记录
            rmse_train_listforplot = []
            mae_train_listforplot = []
            rmse_test_listforplot = []
            mae_test_listforplot = []


            saver = tf.train.Saver(max_to_keep=1)

            for epoch in tqdm(range(FLAGS.num_epochs), ncols=80):
                # Shuffle the data at each epoch
                # t1 = time.time()
                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                # t2 = time.time()
                # print("t2-t1:%.2f" % (t2 - t1))
                # for batch_num in tqdm(range(ll), ncols=10):
                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]

                    uid, iid, reuid, reiid, y_batch = zip(*data_train)
                    # t4 = time.time()
                    # u_batch = np.zeros([batch_size, review_num_u, FLAGS.embedding_dim])
                    # i_batch = np.zeros([batch_size, review_num_i, FLAGS.embedding_dim])
                    # for i in range(batch_size):
                    #     u_batch[i] = np.asarray(u_text_embeds[uid[i][0]])
                    #     i_batch[i] = np.asarray(i_text_embeds[iid[i][0]])
                    u_batch = []
                    i_batch = []
                    for i in range(batch_size):
                        u_batch.append(u_text_embeds[uid[i][0]])
                        i_batch.append(i_text_embeds[iid[i][0]])
                    u_batch = np.array(u_batch)
                    i_batch = np.array(i_batch)
                    # t5 = time.time()
                    # print("t5-t4:%.2f" % (t5 - t4))
                    t_rmse, t_mae, u_a, i_a, fm = train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch,
                                                             batch_num)
                    # t6 = time.time()
                    # print("t6-t5:%.2f" % (t6 - t5))
                    # print(t_rmse, t_mae)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mae += t_mae

                    # Evaluate without an epoch
                    # if batch_num % 900 == 0 and batch_num > 1:
                    #     print("\nEvaluation:")
                    #     print(batch_num)
                    #
                    #     loss_s = 0
                    #     accuracy_s = 0
                    #     mae_s = 0
                    #
                    #     ll_test = int(len(test_data) / batch_size) + 1
                    #     for batch_num in range(ll_test):
                    #         start_index = batch_num * batch_size
                    #         end_index = min((batch_num + 1) * batch_size, data_size_test)
                    #         data_test = test_data[start_index:end_index]
                    #
                    #         userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
                    #         u_valid = []
                    #         i_valid = []
                    #         for i in range(len(userid_valid)):
                    #             u_valid.append(u_text[userid_valid[i][0]])
                    #             i_valid.append(i_text[itemid_valid[i][0]])
                    #         u_valid = np.array(u_valid)
                    #         i_valid = np.array(i_valid)
                    #
                    #         loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid,
                    #                                        y_valid)
                    #         loss_s = loss_s + len(u_valid) * loss
                    #         accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                    #         mae_s = mae_s + len(u_valid) * mae
                    #
                    #     rmse = np.sqrt(accuracy_s / test_length)
                    #     mae = mae_s / test_length
                    #     print("loss_valid {:.4f}, rmse_valid {:.4f}, mae_valid {:.4f}".format(loss_s / test_length,
                    #                                                                           rmse,
                    #                                                                           mae))
                    #     if best_rmse > rmse:
                    #         best_rmse = rmse
                    #     if best_mae > mae:
                    #         best_mae = mae
                    #     print("")

                print("epoch: " + str(epoch))
                print("\nEvaluation:")
                print("train: rmse, mae:", train_rmse / ll, train_mae / ll)
                rmse_train_listforplot.append(train_rmse / ll)
                mae_train_listforplot.append(train_mae / ll)
                u_a = np.reshape(u_a[0], (1, -1))
                i_a = np.reshape(i_a[0], (1, -1))

                train_rmse = 0
                train_mae = 0

                loss_s = 0
                accuracy_s = 0
                mae_s = 0

                t7 = time.time()
                ll_test = int(len(test_data) / batch_size) + 1
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index:end_index]

                    userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
                    u_valid = []
                    i_valid = []
                    for i in range(len(userid_valid)):
                        u_valid.append(u_text_embeds[userid_valid[i][0]])
                        i_valid.append(i_text_embeds[itemid_valid[i][0]])
                    u_valid = np.array(u_valid)
                    i_valid = np.array(i_valid)

                    loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid, y_valid)
                    loss_s = loss_s + len(u_valid) * loss
                    accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                    mae_s = mae_s + len(u_valid) * mae
                t8 = time.time()
                print("t8-t7:%.2f" % (t8 - t7))
                print("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length,
                                                                                np.sqrt(accuracy_s / test_length),
                                                                                mae_s / test_length))
                rmse = np.sqrt(accuracy_s / test_length)
                mae = mae_s / test_length
                rmse_test_listforplot.append(rmse)
                mae_test_listforplot.append(mae)
                if best_rmse > rmse:
                    best_rmse = rmse
                    saver.save(sess, "./checkpoints/NARRE_%s_%s.ckpt" % (dataset_name, time_str),
                               global_step=global_step)
                if best_mae > mae:
                    best_mae = mae
                print("")
            print('best rmse:', best_rmse)
            print('best mae:', best_mae)
            np.savez("./criterion_%s_%s.npz" % (dataset_name, time_str), rmse_train=rmse_train_listforplot, rmse_test=rmse_test_listforplot,
                     mae_train=mae_train_listforplot, mae_test=mae_test_listforplot)
            plot_train_process(rmse_train_listforplot, rmse_test_listforplot,
                               mae_train_listforplot, mae_test_listforplot)
            # deep.save('./checkpoints/NARRE_%s_%s' % (dataset_name, time_str))