'''
BARER
@author:
Xinze Tang

@ created:
30/3/2021
@references:
'''

import tensorflow.compat.v1 as tf


class BRR(object):
    # embedding_id应该是id表达的维度
    def __init__(
            self, review_num_u, review_num_i, user_num, item_num, num_classes,
            n_latent, embedding_id, attention_size,
            embedding_size, l2_reg_lambda=0.0):
        # input_u较原来改成直接改成输入embedding，不需要lookup了
        self.input_u = tf.placeholder(tf.float32, [None, review_num_u, embedding_size], name="input_u")
        self.input_i = tf.placeholder(tf.float32, [None, review_num_i, embedding_size], name="input_i")

        self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reiid')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss_x = tf.constant(0.0)

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.input_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.input_i, 1.0)
            # self.h_drop_u = tf.Print(self.h_drop_u, ["h_drop_u: ", self.h_drop_u])
            # self.h_drop_i = tf.Print(self.h_drop_i, ["h_drop_i: ", self.h_drop_i])

        with tf.name_scope("add_reviews"):
            self.u_feas = tf.reduce_mean(self.h_drop_u, axis=1)
            self.u_feas = tf.nn.dropout(self.u_feas, self.dropout_keep_prob)
            self.i_feas = tf.reduce_mean(self.h_drop_i, axis=1)
            self.i_feas = tf.nn.dropout(self.i_feas, self.dropout_keep_prob)
            # self.u_feas = tf.Print(self.u_feas, ["u_feas: ", self.u_feas, tf.shape(self.u_feas)], summarize=50)
            # self.i_feas = tf.Print(self.i_feas, ["i_feas: ", self.i_feas, tf.shape(self.u_feas)], summarize=50)
        with tf.name_scope("get_fea"):

            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")
            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            # uidmf = tf.Print(uidmf, ["uidmf: ", uidmf, tf.shape(uidmf)], summarize=50)
            # iidmf = tf.Print(iidmf, ["iidmf: ", iidmf, tf.shape(iidmf)], summarize=50)

            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            # self.uid = tf.Print(self.uid, ["uid: ", self.uid, tf.shape(self.uid)], summarize=50)
            # self.iid = tf.Print(self.iid, ["iid: ", self.iid, tf.shape(self.iid)], summarize=50)
            self.uid = tf.reshape(self.uid, [-1, embedding_id])
            self.iid = tf.reshape(self.iid, [-1, embedding_id])
            # self.uid = tf.Print(self.uid, ["uid: ", self.uid, tf.shape(self.uid)], summarize=50)
            # self.iid = tf.Print(self.iid, ["iid: ", self.iid, tf.shape(self.iid)], summarize=50)
            Wu = tf.Variable(
                tf.random_uniform([embedding_size, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            # qu(即uid)+Xu
            self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu

            Wi = tf.Variable(
                tf.random_uniform([embedding_size, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            # pi+Yi(W0*Oi+b0)
            self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi

            # self.u_feas = tf.Print(self.u_feas, ["u_feas: ", self.u_feas, tf.shape(self.u_feas)], summarize=50)
            # self.i_feas = tf.Print(self.i_feas, ["i_feas: ", self.i_feas, tf.shape(self.u_feas)], summarize=50)

        with tf.name_scope('prediction'):
            # h0
            self.FM = tf.multiply(self.u_feas, self.i_feas, name="h0")
            self.FM = tf.nn.relu(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)
            # self.FM = tf.Print(self.FM, ["FM: ", self.FM, tf.shape(self.FM)], summarize=50)

            # Wmul = tf.Variable(
            #     tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')
            Wmul = tf.constant(1, shape=[n_latent, 1], name='wmul', dtype=tf.float32)
            # Wmul = tf.Print(Wmul, ["Wmul:", Wmul], summarize=50)

            # W1T*h0
            self.mul = tf.matmul(self.FM, Wmul)
            self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)
            # self.score = tf.Print(self.score, ["score: ", self.score, tf.shape(self.score)], summarize=50)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))
            self.loss = losses + l2_reg_lambda * l2_loss_x

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))
