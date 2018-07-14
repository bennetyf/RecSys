import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import inspect

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The GAN AE Model #######################################################
# Define the class for Minimax AE
class Adv_AE(object):
    def __init__(self,
                 sess,
                 top_K,
                 num_factors = 32,
                 ae_regs = [0.0,0.0,0.0,0.0],
                 user_node_reg = 0.0,

                 eps = 0.5,
                 num_noise_factor = 64,

                 drop_out_rate = 0.0,
                 lr=0.001,
                 is_user_node = False,

                 noise_pos = 'W2',
                 noise_type='random',

                 robust_test = False,
                 adv_training = False,

                 noise_loss_ratio = 0.0,
                 org_loss_ratio=0.0,

                 is_prec = False,

                 epochs=100, batch_size=128, T=10**3, verbose=False):

        # Parse the arguments and store them in the model
        self.session = sess

        self.num_factors = num_factors
        self.num_noise_factor = num_noise_factor
        self.ae_regs = ae_regs
        self.user_node_regs = user_node_reg
        self.is_user_node = is_user_node

        self.eps = eps
        self.topK = top_K

        self.dropout_rate = drop_out_rate

        self.noise_pos = noise_pos
        self.noise_type = noise_type

        self.lr = lr

        self.org_loss_ratio = org_loss_ratio
        self.noise_loss_ratio = noise_loss_ratio

        self.robust_test = robust_test
        self.adv_training = adv_training

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

        self.is_prec = is_prec

        if self.noise_pos == 'USER':
            assert self.is_user_node

        assert not (self.robust_test and self.adv_training)

        gtl.print_paras(inspect.currentframe())

    def prepare_data(self, original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape
        self.train_array = train_matrix.toarray()

        if self.is_prec:
            _, self.ranking_dict, self.test_dict = \
                mtl.negdict_mat(original_matrix, test_matrix, mod='precision', random_state=20)
        else:
            _, self.ranking_dict, self.test_dict = \
                mtl.negdict_mat(original_matrix, test_matrix, num_neg=199, mod='others', random_state=0)

        self.num_training = self.train_array.shape[0]
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope('Model',reuse=tf.AUTO_REUSE):
            with tf.name_scope('Inputs'):
                # Model Feeds
                self.ratings = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item], name='ratings')
                self.uid = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
                self.istraining = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
                self.layer1_dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='layer1_dropout_rate')

            #########################################################################################################
            with tf.name_scope('Variables'):
                input = self.ratings

                # Encoder Variables
                layer1_w = tf.get_variable(name='encoder_weights',
                                           shape=[self.num_item, self.num_factors],
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

                layer1_b = tf.get_variable(name='encoder_bias',
                                           shape=[self.num_factors],
                                           initializer=tf.zeros_initializer())
                if self.is_user_node:
                    user_embedding = tf.get_variable(name='user_embedding',
                                                     shape=[self.num_user, self.num_factors],
                                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                                     dtype=tf.float32)  # (users, embedding_size)

                # Decoder Variables
                layer2_w1 = tf.get_variable(name='decoder_weights',
                                            shape=[self.num_factors, self.num_item],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

                layer2_w2 = tf.get_variable(name='decoder_concat',
                                            shape=[self.num_noise_factor, self.num_item],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

                layer2_b = tf.get_variable(name='decoder_bias',
                                           shape=[self.num_item],
                                           initializer=tf.zeros_initializer())

                # Noise Variables
                item_w1_noise = tf.get_variable(name='item_w1_noise',
                                                shape=[self.num_item, self.num_factors],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32,
                                                trainable=False)
                if self.is_user_node:
                    user_w_noise = tf.get_variable(name='user_w_noise',
                                                 shape=[self.num_user, self.num_factors],
                                                 initializer=tf.zeros_initializer(),
                                                 dtype=tf.float32,
                                                 trainable=False)

                item_w2_noise = tf.get_variable(name='item_w2_noise',
                                                shape=[self.num_factors, self.num_item],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32,
                                                trainable=False)

                hidden_noise_tr = tf.get_variable(name='hidden_noise_tr',
                                                    shape=[self.batch_size, self.num_factors],
                                                    initializer=tf.zeros_initializer(),
                                                    dtype=tf.float32,
                                                    trainable=False)

                hidden_noise_eval = tf.get_variable(name='hidden_noise_eval',
                                                    shape=[self.num_user, self.num_factors],
                                                    initializer=tf.zeros_initializer(),
                                                    dtype=tf.float32,
                                                    trainable=False)

                noise_vector_tr = tf.get_variable(name='encoder_noise_tr',
                                                  shape=[self.batch_size, self.num_noise_factor],
                                                  initializer=tf.zeros_initializer(),
                                                  dtype=tf.float32,
                                                  trainable=False)

                noise_vector_eval = tf.get_variable(name='encoder_noise_eval',
                                                    shape=[self.num_user, self.num_noise_factor],
                                                    initializer=tf.zeros_initializer(),
                                                    dtype=tf.float32,
                                                    trainable=False)

            #########################################################################################################
            with tf.name_scope('Original_AE'):
                ############# Original AE Model
                org_w1, org_w2 = layer1_w, layer2_w1

                if self.robust_test:
                    if self.noise_pos == 'W1':
                        org_w1 += item_w1_noise
                    elif self.noise_pos == 'W2':
                        org_w2 += item_w2_noise
                    elif self.noise_pos == 'USER':
                        user_embedding += user_w_noise

                if self.is_user_node:
                    user_node = tf.nn.embedding_lookup(user_embedding, self.uid)
                    org_encoder = tf.sigmoid(tf.matmul(input, org_w1) + layer1_b + user_node)
                else:
                    org_encoder = tf.sigmoid(tf.matmul(input, org_w1) + layer1_b)

                if self.robust_test and self.noise_pos == 'HID':
                    org_encoder += hidden_noise_eval

                org_encoder = tf.cond(self.istraining,
                                      lambda: tf.layers.dropout(org_encoder, rate=self.layer1_dropout_rate, name='layer1_dropout'),
                                      lambda: org_encoder)

                org_decoder = tf.identity(tf.matmul(org_encoder, org_w2) + layer2_b)

                self.org_output = org_decoder

                org_base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.org_output))
                # org_base_loss = tf.nn.l2_loss(tf.sigmoid(self.org_output) - input)
                org_base_loss = org_base_loss / tf.cast(tf.shape(input)[0], dtype=org_base_loss.dtype)

            #########################################################################################################
            ###### The Noisy Auto-Encoder
            if self.adv_training:
                if self.noise_pos == 'CON':
                    # ConCat Noise AE
                    with tf.name_scope("ConCat_AE"):

                        if self.is_user_node:
                            user_node = tf.nn.embedding_lookup(user_embedding, self.uid)
                            concat_noise_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b + user_node)
                        else:
                            concat_noise_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b)

                        concat_noise_encoder = tf.cond(self.istraining,
                                                       lambda:tf.concat([concat_noise_encoder, noise_vector_tr], axis=1),
                                                       lambda:tf.concat([concat_noise_encoder, noise_vector_eval], axis=1))

                        concat_noise_encoder = tf.cond(self.istraining,
                                                        lambda: tf.layers.dropout(concat_noise_encoder, rate=self.layer1_dropout_rate, name='layer1_dropout'),
                                                        lambda: concat_noise_encoder)

                        concat_w2 = tf.concat([layer2_w1, layer2_w2], axis=0)
                        # out_vector = tf.sigmoid(tf.matmul(concat_noise_encoder, layer2_concat_w) + layer2_b)
                        concat_noise_decoder = tf.identity(tf.matmul(concat_noise_encoder, concat_w2) + layer2_b)

                        # Output
                        self.concat_noise_output = concat_noise_decoder
                        # Noisy Model Loss
                        concat_noise_base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.concat_noise_output))
                        # concat_noise_base_loss = tf.nn.l2_loss(tf.sigmoid(self.concat_noise_output) - input)
                        concat_noise_base_loss = concat_noise_base_loss / tf.cast(tf.shape(input)[0], dtype=concat_noise_base_loss.dtype)

                if self.noise_pos == 'W1':
                    with tf.name_scope("W1_AE"):
                        w1_noise_w1 = layer1_w + item_w1_noise

                        if self.is_user_node:
                            user_node = tf.nn.embedding_lookup(user_embedding, self.uid)
                            w1_noise_encoder = tf.sigmoid(tf.matmul(input, w1_noise_w1) + layer1_b + user_node)
                        else:
                            w1_noise_encoder = tf.sigmoid(tf.matmul(input, w1_noise_w1) + layer1_b)

                        w1_noise_encoder = tf.cond(self.istraining,
                                                   lambda: tf.layers.dropout(w1_noise_encoder, rate=self.layer1_dropout_rate,name='layer1_dropout'),
                                                   lambda: w1_noise_encoder)

                        w1_noise_decoder = tf.identity(tf.matmul(w1_noise_encoder, layer2_w1) + layer2_b)

                        # Output
                        self.w1_noise_output = w1_noise_decoder

                        # Noisy Model Loss
                        w1_noise_base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.w1_noise_output))
                        # weight_noise_base_loss = tf.nn.l2_loss(tf.sigmoid(self.weight_noise_output) - input)
                        w1_noise_base_loss = w1_noise_base_loss / tf.cast(tf.shape(input)[0], dtype=w1_noise_base_loss.dtype)

                if self.noise_pos == 'W2':
                    with tf.name_scope("W2_AE"):
                        w2_noise_w2 = layer2_w1 + item_w2_noise

                        if self.is_user_node:
                            user_node = tf.nn.embedding_lookup(user_embedding, self.uid)
                            w2_noise_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b + user_node)
                        else:
                            w2_noise_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b)

                        w2_noise_encoder = tf.cond(self.istraining,
                                                   lambda: tf.layers.dropout(w2_noise_encoder, rate=self.layer1_dropout_rate,name='layer1_dropout'),
                                                   lambda: w2_noise_encoder)

                        w2_noise_decoder = tf.identity(tf.matmul(w2_noise_encoder, w2_noise_w2) + layer2_b)

                        # Output
                        self.w2_noise_output = w2_noise_decoder

                        # Noisy Model Loss
                        w2_noise_base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.w2_noise_output))
                        # weight_noise_base_loss = tf.nn.l2_loss(tf.sigmoid(self.weight_noise_output) - input)
                        w2_noise_base_loss = w2_noise_base_loss / tf.cast(tf.shape(input)[0], dtype=w2_noise_base_loss.dtype)

                if self.noise_pos == 'USER':
                    with tf.name_scope("USER_AE"):
                        user_embedding += user_w_noise
                        user_node = tf.nn.embedding_lookup(user_embedding, self.uid)
                        user_noise_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b + user_node)

                        user_noise_encoder = tf.cond(self.istraining,
                                                   lambda: tf.layers.dropout(user_noise_encoder, rate=self.layer1_dropout_rate,name='layer1_dropout'),
                                                   lambda: user_noise_encoder)

                        user_noise_decoder = tf.identity(tf.matmul(user_noise_encoder, layer2_w1) + layer2_b)

                        # Output
                        self.user_noise_output = user_noise_decoder

                        # Noisy Model Loss
                        user_noise_base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.user_noise_output))
                        # weight_noise_base_loss = tf.nn.l2_loss(tf.sigmoid(self.weight_noise_output) - input)
                        user_noise_base_loss = user_noise_base_loss / tf.cast(tf.shape(input)[0], dtype=user_noise_base_loss.dtype)

                if self.noise_pos == 'HID':
                    with tf.name_scope("Hidden_AE"):
                        if self.is_user_node:
                            user_node = tf.nn.embedding_lookup(user_embedding, self.uid)
                            hidden_noise_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b + user_node) + hidden_noise_tr
                        else:
                            hidden_noise_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b) + hidden_noise_tr

                        hidden_noise_encoder = tf.cond(self.istraining,
                                                        lambda: tf.layers.dropout(hidden_noise_encoder, rate=self.layer1_dropout_rate, name='layer1_dropout'),
                                                        lambda: hidden_noise_encoder)

                        hidden_noise_decoder = tf.identity(tf.matmul(hidden_noise_encoder, layer2_w1) + layer2_b)

                        # Output
                        self.hidden_noise_output = hidden_noise_decoder

                        hidden_noise_base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.hidden_noise_output))
                        # hidden_noise_base_loss = tf.nn.l2_loss(tf.sigmoid(self.hidden_noise_output) - input)
                        hidden_noise_base_loss = hidden_noise_base_loss / tf.cast(tf.shape(input)[0], dtype=self.hidden_noise_output.dtype)

            ############# Final Outputs
            with tf.name_scope('Prediction'):
                # self.mixed_output = (1-self.output_mix_ratio) * self.org_output + self.output_mix_ratio * self.noisy_output
                self.pred_y = tf.sigmoid(self.org_output)
                # self.pred_y = tf.sigmoid(self.mixed_output)

            ############# Overall Losses
            with tf.name_scope('Loss'):
                if self.adv_training:
                    if self.noise_pos == 'W1':
                        base_loss = self.org_loss_ratio * org_base_loss + self.noise_loss_ratio * w1_noise_base_loss
                        reg_loss =  self.ae_regs[0] * tf.nn.l2_loss(w1_noise_w1) + \
                                    self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
                                    self.ae_regs[2] * tf.nn.l2_loss(layer2_w1) + \
                                    self.ae_regs[3] * tf.nn.l2_loss(layer2_b)

                    if self.noise_pos == 'W2':
                        base_loss = self.org_loss_ratio * org_base_loss + self.noise_loss_ratio * w2_noise_base_loss
                        reg_loss =  self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + \
                                    self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
                                    self.ae_regs[2] * tf.nn.l2_loss(w2_noise_w2) + \
                                    self.ae_regs[3] * tf.nn.l2_loss(layer2_b)

                    if self.noise_pos == 'HID':
                        base_loss = self.org_loss_ratio * org_base_loss + self.noise_loss_ratio * hidden_noise_base_loss
                        reg_loss =  self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + \
                                    self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
                                    self.ae_regs[2] * tf.nn.l2_loss(layer2_w1) + \
                                    self.ae_regs[3] * tf.nn.l2_loss(layer2_b)

                    if self.noise_pos == 'USER':
                        base_loss = self.org_loss_ratio * org_base_loss + self.noise_loss_ratio * user_noise_base_loss
                        reg_loss =  self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + \
                                    self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
                                    self.ae_regs[2] * tf.nn.l2_loss(layer2_w1) + \
                                    self.ae_regs[3] * tf.nn.l2_loss(layer2_b)

                    if self.noise_pos == 'CON':
                        base_loss = self.org_loss_ratio * org_base_loss + self.noise_loss_ratio * concat_noise_base_loss
                        reg_loss =  self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + \
                                    self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
                                    self.ae_regs[2] * tf.nn.l2_loss(concat_w2) + \
                                    self.ae_regs[3] * tf.nn.l2_loss(layer2_b)

                else:
                    base_loss = org_base_loss
                    reg_loss = self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + \
                               self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
                               self.ae_regs[2] * tf.nn.l2_loss(layer2_w1) + \
                               self.ae_regs[3] * tf.nn.l2_loss(layer2_b)

                if self.is_user_node:
                    reg_loss += self.user_node_regs * tf.nn.l2_loss(user_embedding)

                self.loss = base_loss + reg_loss

            ############# Optimizer
            with tf.name_scope('Optimizer'):
                self.opt = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)

            ########### Robustness Testing (Random or Adversial)
            with tf.name_scope('Noise_Adding'):
                if self.adv_training or self.robust_test:
                    if self.noise_type == 'random':
                        if self.noise_pos == 'W1':
                            random_noise = tf.random_normal(shape=tf.shape(org_w1), mean=tf.reduce_mean(org_w1), stddev=0.01)
                            self.update_delta = item_w1_noise.assign(self.eps * random_noise / tf.norm(random_noise))
                        if self.noise_pos == 'W2':
                            random_noise = tf.random_normal(shape=tf.shape(org_w2), mean=tf.reduce_mean(org_w2), stddev=0.01)
                            self.update_delta = item_w2_noise.assign(self.eps * random_noise / tf.norm(random_noise))
                        if self.noise_pos == 'USER':
                            random_noise = tf.random_normal(shape=tf.shape(user_embedding), mean=tf.reduce_mean(user_embedding), stddev=0.01)
                            self.update_delta = user_w_noise.assign(self.eps * random_noise / tf.norm(random_noise))
                        if self.noise_pos == 'HID':
                            random_noise = tf.random_normal(shape=tf.shape(org_encoder), mean=tf.reduce_mean(org_encoder), stddev=0.01)
                            if self.robust_test:
                                self.update_delta = hidden_noise_eval.assign(self.eps * random_noise / tf.norm(random_noise))
                            else:
                                self.update_delta = hidden_noise_tr.assign(self.eps * random_noise / tf.norm(random_noise))
                    if self.noise_type == 'adv':
                        if self.noise_pos == 'W1':
                            if self.robust_test:
                                self.grad_delta = tf.gradients(ys=org_base_loss, xs=item_w1_noise)[0]
                            else:
                                self.grad_delta = tf.gradients(ys=base_loss, xs=item_w1_noise)[0]
                            self.grad_delta_dense = tf.stop_gradient(self.grad_delta)
                            self.update_delta = item_w1_noise.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))
                        if self.noise_pos == 'W2':
                            if self.robust_test:
                                self.grad_delta = tf.gradients(ys=org_base_loss, xs=item_w2_noise)[0]
                            else:
                                self.grad_delta = tf.gradients(ys=base_loss, xs=item_w2_noise)[0]
                            self.grad_delta_dense = tf.stop_gradient(self.grad_delta)
                            self.update_delta = item_w2_noise.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))
                        if self.noise_pos == 'USER':
                            if self.robust_test:
                                self.grad_delta = tf.gradients(ys=org_base_loss, xs=user_w_noise)[0]
                            else:
                                self.grad_delta = tf.gradients(ys=base_loss, xs=user_w_noise)[0]
                            self.grad_delta_dense = tf.stop_gradient(self.grad_delta)
                            self.update_delta = user_w_noise.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))
                        if self.noise_pos == 'HID':
                            if self.robust_test:
                                self.grad_delta = tf.gradients(ys=org_base_loss, xs=hidden_noise_eval)[0]
                                self.grad_delta_dense = tf.stop_gradient(self.grad_delta)
                                self.update_delta = hidden_noise_eval.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))
                            else:
                                self.grad_delta = tf.gradients(ys=base_loss, xs=hidden_noise_tr)[0]
                                self.grad_delta_dense = tf.stop_gradient(self.grad_delta)
                                self.update_delta = hidden_noise_tr.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))
            print('Model Building Completed.')

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        random_idx = np.random.permutation(self.num_user)

        n_batches, total_loss = 0, 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                break
                # batch_idx = random_idx[i * self.batch_size:]
                # batch_ratings = self.train_array[batch_idx, :]
                # batch_neg_masks = self.negative_output_mask[batch_idx, :]
            else:
                batch_idx = random_idx[i * self.batch_size: (i + 1) * self.batch_size]
                batch_ratings = self.train_array[batch_idx, :]

            if self.is_user_node:
                feed_dict = {
                                self.ratings:               batch_ratings,
                                self.uid:                   batch_idx,
                                self.istraining:            True,
                                self.layer1_dropout_rate:   self.dropout_rate
                            }
            else:
                feed_dict = {
                                self.ratings:               batch_ratings,
                                self.istraining:            True,
                                self.layer1_dropout_rate:   self.dropout_rate
                            }

            # Update the Noises in Training (Adversial Training)
            if self.adv_training:
                self.session.run(self.update_delta,feed_dict)

            # Train the model
            _, l = self.session.run([self.opt, self.loss], feed_dict)

            n_batches += 1
            total_loss += l

            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("[All] Training Epoch {0} Batch {1}: [Loss] = {2}".format(epoch, n_batches, total_loss / n_batches))

        if self.verbose:
            print("[Epoch Average] Training Epoch {0}: [Loss] {1}".format(epoch, total_loss / n_batches))

        return total_loss / n_batches

    def eval_one_epoch(self, epoch):
        if self.is_user_node:
            feed_dict = {
                         self.ratings:              self.train_array,
                         self.uid:                  range(self.num_user),
                         self.istraining:           False,
                         self.layer1_dropout_rate:  0
                        }
        else:
            feed_dict = {
                            self.ratings:               self.train_array,
                            self.istraining:            False,
                            self.layer1_dropout_rate:   0
                        }

        if self.robust_test:
            print('[Pos={0}] {1} Noise Added [Robust Test]'.format(self.noise_pos, self.noise_type))
            print('[Eps={0}] {1} Noise Level [Robust Test]'.format(self.eps, self.noise_type))
            self.session.run(self.update_delta,feed_dict)

        # Input the uncorrupted training data
        pred_y = self.session.run(self.pred_y, feed_dict)

        # pred_y = pred_y.clip(min=0,max=1)

        n_batches = 0
        if self.is_prec:
            total_prec, total_recall = np.zeros(len(self.topK)), np.zeros(len(self.topK))
        else:
            total_hr, total_ndcg = np.zeros(len(self.topK)), np.zeros(len(self.topK))

        # Loop for each user (generate the ranking lists for different users)
        for u in self.ranking_dict:

            if len(self.test_dict[u]) == 0:
                continue

            iid = self.ranking_dict[u]  # The ranking item ids for user u
            rk = pred_y[u, np.array(iid)]  # The predicted item values for user u
            n_batches += 1

            if self.is_prec:
                precision, recall,_,_,_ = evl.rankingMetrics(rk, iid, self.topK, self.test_dict[u], mod='precision', is_map=False)
                total_prec += precision
                total_recall += recall
            else:
                hr, ndcg = evl.rankingMetrics(rk, iid, self.topK, self.test_dict[u], mod='hr')
                total_hr += hr
                total_ndcg += ndcg

        if self.is_prec:
            avg_prec, avg_recall = total_prec / n_batches, total_recall / n_batches
            for i in range(len(self.topK)):
                print('-' * 55)
                print("Epoch {0}: [Precision@{1}] {2}".format(epoch, self.topK[i], avg_prec[i]))
                print("Epoch {0}: [Recall@{1}] {2}".format(epoch, self.topK[i], avg_recall[i]))
            print('=' * 55)
            return avg_prec[0], avg_recall[0]
        else:
            avg_hr, avg_ndcg = total_hr / n_batches, total_ndcg / n_batches

            for i in range(len(self.topK)):
                print('-' * 55)
                print("Epoch {0}: [HR@{1}] {2}".format(epoch, self.topK[i], avg_hr[i]))
                print("Epoch {0}: [nDCG@{1}] {2}".format(epoch, self.topK[i], avg_ndcg[i]))
            print('=' * 55)
            return avg_hr[0], avg_ndcg[0]

    # Final Training of the model
    def train(self, restore=False, save=False, restore_datafile=None, save_datafile=None):

        if restore: # Restore the model from checkpoint
            self.restore_model(restore_datafile, verbose=True)
        else:
            self.session.run(tf.global_variables_initializer())

        if not save: # Do not save the model
            self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                self.eval_one_epoch(i)

        else: # Save the model while training
            previous_metric1,_ = self.eval_one_epoch(-1)
            # previous_metric1 = 0
            for i in range(self.epochs):
                self.train_one_epoch(i)
                metric1,_ = self.eval_one_epoch(i)
                if metric1 > previous_metric1:
                    previous_metric1 = metric1
                    self.save_model(save_datafile, verbose=False)

    # Save the model
    def save_model(self, datafile, verbose=False):
        saver = tf.train.Saver()
        path = saver.save(self.session, datafile)
        if verbose:
            print("Model Saved in Path: {0}".format(path))

    # Restore the model
    def restore_model(self, datafile, verbose=False):
        saver = tf.train.Saver()
        saver.restore(self.session, datafile)
        if verbose:
            print("Model Restored from Path: {0}".format(datafile))

    # Evaluate the model
    def evaluate(self, datafile):
        self.restore_model(datafile,True)
        self.eval_one_epoch(-1)

########################################################################################################################

######################################## Parse Arguments ###############################################################
# if __name__ == "__main__":
    # original_matrix \
    #         = mtl.load_original_matrix(datafile='Data/ml-1m/ratings.dat', header=['uid', 'iid', 'ratings','time'], sep='::')
    # original_matrix = mtl.matrix_to_binary(original_matrix,0)
    # train_matrix, test_matrix \
    #     = mtl.matrix_split(original_matrix, opt='ranking', mode='mat', n_item_per_user=1, random_state=10)
    # train_matrix, test_matrix \
    #     = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.2, random_state=10)

    # Calculate Precision
    # original_matrix = gtl.load_mat_as_matrix('Data/Ciao.mat', opt='coo')['rating']
    # train_matrix, test_matrix \
    #     = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.2, random_state=10)
    #
    # gtl.matrix_to_mat('Data/Ciao_Precision_Data.mat', opt='coo', original=original_matrix, train=train_matrix, test=test_matrix)
    #
    # path_list = ['Data/ML1M_Rank_HR.mat',
    #             'Data/ML10M_Rank_HR.mat',
    #             'Data/Ciao_Rank_HR.mat',
    #             'Data/Douban_Rank_HR.mat',
    #             'Data/Epinions_Rank_HR.mat',
    #             'Data/GoodBooks_Rank_HR.mat',
    #             'Data/YahooMovie_Rank_HR.mat',
    #             'Data/YahooMusic_Rank_HR.mat',
    #             'Data/FilmTrust_Rank_HR.mat',
    #             'Data/Flixster_Rank_HR.mat',
    #             'Data/BX_Rank_HR.mat']
    #
    # dataset_list = ['ML1M','ML10M','Ciao','Douban','Epinions','GoodBooks','YahooMovie','YahooMusic','FilmTrust','Flixster','BX']
    # # data = gtl.load_mat_as_matrix('Data/ML1M_Precision_Data.mat', opt='coo')
    # # data = gtl.load_mat_as_matrix('Data/Ciao_Precision_Data.mat', opt='coo')
    #
    # path = path_list[0]
    # dataset = dataset_list[0]
    #
    # print('Loading Data From {0}'.format(path))
    #
    # data = gtl.load_mat_as_matrix(path, opt='coo')
    # original_matrix, train_matrix, test_matrix = data['original'], data['train'], data['test']
    #
    # print(original_matrix.shape)
    # print("Number of Nonzeros in Training: {0}".format(train_matrix.nnz))
    #
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
    #                                       intra_op_parallelism_threads=8,
    #                                       inter_op_parallelism_threads=8,
    #                                       gpu_options=gpu_options)) as sess:
    #     model = Adv_AE(sess,
    #                     top_K=[5,10],
    #
    #                     num_factors=200,
    #                     # ae_regs=[0.01,0.01,0.01,0.01],
    #                     ae_regs = [0.001]*4,
    #                     lr=0.001,
    #
    #                     is_user_node=True,
    #                     user_node_reg=0.001,
    #
    #                     robust_test=False,
    #                     adv_training=True,
    #                     noise_pos='W2',
    #                     noise_type='adv',
    #                     num_noise_factor=64,
    #                     eps=0.5,
    #                     org_loss_ratio=1.0,
    #                     noise_loss_ratio=500.0,
    #
    #                     is_prec=False,
    #                     epochs=2000, batch_size=128, T=200, verbose=True)
    #
    #     model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
    #
    #     # for i in range(16):
    #     #     model.eps = i
    #     model.build_model()
    #     #     model.evaluate('SavedModel/GAN_AE/ML-1M/CAE_ML1M_CE.ckpt')
    #     #     model.evaluate('SavedModel/GAN_AE/Ciao/CAE_Ciao_CE.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/YahooMovie/CAE_YahooMovie_CE.ckpt')
    #     #     model.evaluate('SavedModel/GAN_AE/Douban/CAE_Douban_CE.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/FilmTrust/CAE_FilmTrust_CE.ckpt')
    #
    #     # model.evaluate('SavedModel/GAN_AE/Douban/Weight_Noise/Pure_AE_Douban_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/Epinions/Weight_Noise/Pure_AE_Epinions_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/YahooMovie/Weight_Noise/Pure_AE_YahooMovie_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/YahooMusic/Weight_Noise/Pure_AE_YahooMusic_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/FilmTrust/Weight_Noise/Pure_AE_FilmTrust_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/ML-10M/Weight_Noise/Pure_AE_ML10M_SQ_MIX_ALL_0_01.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/BookCrossing/Weight_Noise/Pure_AE_BX_SQ_MIX_ALL_1_0.ckpt')
    #
    #     if model.adv_training:
    #         ckpt_save_path = "Pretrain/%s/GANAE/embed_%d/%s/" % (dataset, model.num_factors, '20180713') + 'checkpoint.ckpt'
    #         ckpt_restore_path = "Pretrain/%s/CAE/embed_%d/%s/" % (dataset, model.num_factors, '20180713') + 'checkpoint.ckpt'
    #     else:
    #         ckpt_save_path = "Pretrain/%s/CAE/embed_%d/%s/" % (dataset, model.num_factors, '20180713') + 'checkpoint.ckpt'
    #         ckpt_restore_path = None
    #         # ckpt_restore_path = 0 if args.restore is None else "Pretrain/%s/MF_BPR/embed_%d/%s/" % (args.dataset, args.embed_size, args.restore)
    #
    #     if not os.path.exists(ckpt_save_path):
    #         os.makedirs(ckpt_save_path)
    #     if ckpt_restore_path and not os.path.exists(ckpt_restore_path):
    #         os.makedirs(ckpt_restore_path)
    #
    #     model.train(restore=True, save=True,
    #                 save_datafile=ckpt_save_path, restore_datafile=ckpt_restore_path)
    #                 # save_datafile = 'SavedModel/GAN_AE/ML-1M/CAE_ML1M_CE_Emb120_20180713.ckpt')
    #                 # save_datafile='SavedModel/GAN_AE/ML-1M/CAE_ML1M_PREC_CE.ckpt')
    #
    #                 # save_datafile = 'SavedModel/GAN_AE/Ciao/CAE_Ciao_CE.ckpt')
    #                 # save_datafile = 'SavedModel/GAN_AE/Ciao/AE_Ciao_PREC_CE.ckpt')
    #
    #                 # save_datafile = 'SavedModel/GAN_AE/GoodBooks/Weight_Noise/AE_GB_RANK_CE.ckpt')
    #
    #                 # restore_datafile = 'SavedModel/GAN_AE/YahooMovie/AE_YahooMovie_CE.ckpt')
    #                 # restore_datafile = 'SavedModel/GAN_AE/Douban/AE_Douban_CE.ckpt')
    #                 # save_datafile = 'SavedModel/GAN_AE/FilmTrust/AE_FilmTrust_CE.ckpt')
    #                 # save_datafile = 'SavedModel/GAN_AE/YahooMusic/AE_YahooMusic_CE.ckpt')