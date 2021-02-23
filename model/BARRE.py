'''
NARRE
@author:
Xinze Tang

@ created:
20/2/2021
@references:
'''

import tensorflow as tf


class BARRE(object):
    # embedding_id应该是id表达的维度
    def __init__(
            self, review_num_u, review_num_i, user_num, item_num, num_classes,
            n_latent, embedding_id, attention_size,
            embedding_size, l2_reg_lambda=0.0):
        # input_u较原来改成直接改成输入embedding，不需要lookup了
        # self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        self.input_u = tf.placeholder(tf.float32, [None, review_num_u, embedding_size], name="input_u")
        # self.input_u = tf.reshape(self.input_u, [-1, review_len_u, embedding_size, 1])
        # self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        self.input_i = tf.placeholder(tf.float32, [None, review_num_i, embedding_size], name="input_i")
        # self.input_i = tf.reshape(self.input_i, [-1, review_len_i, embedding_size, 1])

        self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss_x = tf.constant(0.0)
        # with tf.name_scope("user_embedding"):
        #     self.W1 = tf.Variable(
        #         tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
        #         name="W1")
        #     self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
        #     self.embedded_users = tf.expand_dims(self.embedded_user, -1)
        #     # self.embedded_users = tf.Print(self.embedded_users, ["embedded_users: ", self.embedded_users,
        #     #                                                      tf.shape(self.embedded_users)], summarize=50)
        #
        # with tf.name_scope("item_embedding"):
        #     self.W2 = tf.Variable(
        #         tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
        #         name="W2")
        #     self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
        #     self.embedded_items = tf.expand_dims(self.embedded_item, -1)
        #     # self.embedded_items = tf.Print(self.embedded_items, ["embedded_items: ", self.embedded_items,
        #     #                                                      tf.shape(self.embedded_items)], summarize=50)
        #
        # # 卷积层是问题所在！通过clip来解决矩阵乘法数值过大的问题
        # pooled_outputs_u = []
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope("user_conv-maxpool-%s" % filter_size):
        #         # Convolution Layer
        #         filter_shape = [filter_size, embedding_size, 1, num_filters]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #         # W = tf.Print(W, ["conv W shape:", tf.shape(W)], summarize=50)
        #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        #         self.embedded_users = tf.reshape(self.embedded_users, [-1, review_len_u, embedding_size, 1])
        #
        #         conv = tf.nn.conv2d(
        #             self.embedded_users,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name="conv")
        #         # Apply nonlinearity
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #         # Maxpooling over the outputs
        #         pooled = tf.nn.max_pool(
        #             h,
        #             ksize=[1, review_len_u - filter_size + 1, 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name="pool")
        #         pooled_outputs_u.append(pooled)
        # num_filters_total = num_filters * len(filter_sizes)
        # self.h_pool_u = tf.concat(pooled_outputs_u, 3)
        # self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, review_num_u, num_filters_total])
        # self.h_pool_flat_u = tf.clip_by_value(self.h_pool_flat_u, clip_value_min=0, clip_value_max=5, name="clip")
        # # self.h_pool_flat_u = tf.Print(self.h_pool_flat_u, ["h_pool_flat_u: ", self.h_pool_flat_u,
        # #                                                    tf.shape(self.h_pool_flat_u)], summarize=50)
        #
        # pooled_outputs_i = []
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope("item_conv-maxpool-%s" % filter_size):
        #         # Convolution Layer
        #         # filter_shape = 3*300*1*100
        #         filter_shape = [filter_size, embedding_size, 1, num_filters]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #         # W = tf.Print(W, ["W: ", W, tf.shape(W)], summarize=50)
        #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        #         self.embedded_items = tf.reshape(self.embedded_items, [-1, review_len_i, embedding_size, 1])
        #         # self.embedded_items = tf.Print(self.embedded_items, ["after reshape: ", tf.shape(self.embedded_items)], summarize=50)
        #         conv = tf.nn.conv2d(
        #             self.embedded_items,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name="conv")
        #         # Apply nonlinearity
        #         # N*review_len_i - filter_size + 1*1*num_filters
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #         # h = tf.Print(h, ["h after conv and ReLU: ", h, tf.shape(h)], summarize=50)
        #         # Maxpooling over the outputs
        #         # N*1*1*1
        #         pooled = tf.nn.max_pool(
        #             h,
        #             ksize=[1, review_len_i - filter_size + 1, 1, 1], # [batch, height, width, channels]
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name="pool")
        #         pooled_outputs_i.append(pooled)
        # # num_filters_total也就是文本卷积之后的输出特征个数
        # num_filters_total = num_filters * len(filter_sizes)
        # # 根据kernel的个数concat起来，几个kernel就有几个池化后的数
        # self.h_pool_i = tf.concat(pooled_outputs_i, 3)
        # # self.h_pool_i = tf.Print(self.h_pool_i, ["h_pool_i after pooled: ", self.h_pool_i,
        # #                                                    tf.shape(self.h_pool_i)], summarize=50)
        # # batch_size*32(k)*卷积后的特征个数
        # self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, review_num_i, num_filters_total])
        # self.h_pool_flat_i = tf.clip_by_value(self.h_pool_flat_i, clip_value_min=0, clip_value_max=5, name="clip")
        # # self.h_pool_flat_i = tf.Print(self.h_pool_flat_i, ["h_pool_flat_i after clipped: ", self.h_pool_flat_i,
        # #                                                    tf.shape(self.h_pool_flat_i)], summarize=50)

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.input_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.input_i, 1.0)
            # self.h_drop_u = tf.Print(self.h_drop_u, ["h_drop_u: ", self.h_drop_u])
            # self.h_drop_i = tf.Print(self.h_drop_i, ["h_drop_i: ", self.h_drop_i])
        with tf.name_scope("attention"):
            Wau = tf.Variable(
                tf.random_uniform([embedding_size, attention_size], -0.1, 0.1), name='Wau')
            Wru = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.1, 0.1), name='Wru')
            Wpu = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.1, 0.1), name='Wpu')
            bau = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bau")
            bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")
            self.iid_a = tf.nn.relu(tf.nn.embedding_lookup(iidW, self.input_reuid))
            self.u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_u, Wau) + tf.einsum('ajk,kl->ajl', self.iid_a, Wru) + bau),
                                 Wpu) + bbu  # None*u_len*1

            self.u_a = tf.nn.softmax(self.u_j, 1)  # none*u_len*1
            # self.u_a = tf.Print(self.u_a, ["u_a:", self.u_a, tf.shape(self.u_a)], summarize=50)


            Wai = tf.Variable(
                tf.random_uniform([embedding_size, attention_size], -0.1, 0.1), name='Wai')
            Wri = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.1, 0.1), name='Wri')
            Wpi = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.1, 0.1), name='Wpi')
            bai = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bai")
            bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
            self.uid_a = tf.nn.relu(tf.nn.embedding_lookup(uidW, self.input_reiid))
            self.i_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_i, Wai) + tf.einsum('ajk,kl->ajl', self.uid_a, Wri) + bai),
                                 Wpi) + bbi

            self.i_a = tf.nn.softmax(self.i_j, 1)  # none*len*1
            # self.i_a = tf.Print(self.i_a, ["i_a:", self.i_a, tf.shape(self.i_a)], summarize=50)

            l2_loss_x += tf.nn.l2_loss(Wau)
            l2_loss_x += tf.nn.l2_loss(Wru)
            l2_loss_x += tf.nn.l2_loss(Wri)
            l2_loss_x += tf.nn.l2_loss(Wai)

        with tf.name_scope("add_reviews"):
            self.u_feas = tf.reduce_sum(tf.multiply(self.u_a, self.h_drop_u), 1)
            self.u_feas = tf.nn.dropout(self.u_feas, self.dropout_keep_prob)
            self.i_feas = tf.reduce_sum(tf.multiply(self.i_a, self.h_drop_i), 1)
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

        with tf.name_scope('ncf'):
            # h0
            self.FM = tf.multiply(self.u_feas, self.i_feas, name="h0")
            self.FM = tf.nn.relu(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)
            # self.FM = tf.Print(self.FM, ["FM: ", self.FM, tf.shape(self.FM)], summarize=50)

            Wmul = tf.Variable(
                tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')

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
