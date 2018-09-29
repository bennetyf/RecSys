# Training user embeddings using AutoRec
# import sys, os
# sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import numpy as np
import GenUtils as gtl

############################################### The AutoRec Model ######################################################
# Define the class for AutoRec
class AutoRec(object):
    def __init__(self, sess,
                 num_factors=200,
                 reg=0.0,
                 model='user',
                 dropout = 0.0,
                 lr=0.001,
                 epochs=100, batch_size=128, T=10**3, verbose=False):
        '''Constructor'''
        # Parse the arguments and store them in the model
        self.session = sess

        self.num_factors = num_factors
        self.reg = reg
        self.model = model
        self.dropout = dropout

        self.lr = lr

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self, original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape

        # gtl.set_random_seed(100) # Set random seed for the program (shuffling)

        if self.model == 'item':
            self.train_array, self.test_array = train_matrix.toarray().T, test_matrix.toarray().T
        else:
            self.train_array, self.test_array = train_matrix.toarray(), test_matrix.toarray()

        self.num_training = self.train_array.shape[0]
        self.num_batch = self.num_training // self.batch_size

        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope('AutoRec',reuse=tf.AUTO_REUSE):
            self.istraining = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')

            if self.model == 'item':
                self.ratings = tf.placeholder(dtype=tf.float32, shape=[None,self.num_user], name='ratings')
                # scale = np.sqrt(1.0 / (self.num_factors))
            else:
                self.ratings = tf.placeholder(dtype=tf.float32, shape=[None,self.num_item], name='ratings')
                # scale = np.sqrt(1.0 / (self.num_factors))

            # Auto Encoder
            layer1 = tf.layers.dense(self.ratings,
                                    units=self.num_factors, activation=tf.nn.sigmoid, use_bias=True,
                                    # kernel_initializer=tf.random_uniform_initializer(-scale, scale),
                                    kernel_initializer=tf.truncated_normal_initializer(0,0.03),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='layer1')

            self.embedding_out = layer1  # The embedding output of the auto-encoder

            layer1_out = tf.cond(self.istraining,
                                 lambda: tf.layers.dropout(layer1, rate=self.dropout_rate, name='layer1_dropout'),
                                 lambda: layer1)

            if self.model == 'item':
                out_vector = tf.layers.dense(layer1_out,
                                            units=self.num_user, activation=tf.identity, use_bias=True,
                                            # kernel_initializer=tf.random_uniform_initializer(-scale, scale),
                                            kernel_initializer=tf.truncated_normal_initializer(0, 0.03),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg),
                                            bias_initializer=tf.zeros_initializer(),
                                            name='output')
            else:
                out_vector = tf.layers.dense(layer1_out,
                                            units=self.num_item, activation=tf.identity, use_bias=False,
                                            # kernel_initializer=tf.random_uniform_initializer(-scale, scale),
                                            kernel_initializer=tf.truncated_normal_initializer(0, 0.03),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg),
                                            bias_initializer=tf.zeros_initializer(),
                                            name='output')

            # Output
            out_mask = tf.sign(self.ratings)
            self.pred_y = tf.cond(self.istraining,
                                  lambda: tf.multiply(out_vector, out_mask),
                                  lambda: out_vector)

            # Loss
            base_loss = tf.reduce_sum(tf.square(self.ratings - self.pred_y))
            # reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
            self.loss = base_loss + reg_loss

            # Optimizer
            optimizer = tf.train.RMSPropOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]

            self.opt = optimizer.apply_gradients(capped_gvs)

            # Metrics
            self.rms = tf.sqrt(tf.reduce_sum(tf.square(self.ratings - self.pred_y)) / tf.reduce_sum(out_mask))
            self.mae = tf.reduce_sum(tf.abs(self.ratings - self.pred_y)) / tf.reduce_sum(out_mask)

        print('Model {0}-AutoRec Building Completed.'.format(self.model))

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        random_idx = np.random.permutation(self.train_array.shape[0])
        # np.random.shuffle(self.train_array)

        n_batches,total_loss,total_mae,total_rms = 0,0,0,0
        for i in range(self.num_batch):

            if i == self.num_batch -1:
                batch_ratings = self.train_array[random_idx[i * self.batch_size:],:]
                # batch_ratings = self.train_array[i * self.batch_size:, :]
            else:
                batch_ratings = self.train_array[random_idx[i * self.batch_size: (i + 1) * self.batch_size],:]
                # batch_ratings = self.train_array[i * self.batch_size: (i + 1) * self.batch_size, :]

            _, l, mae, rms = \
                self.session.run([self.opt, self.loss, self.mae, self.rms],
                                 feed_dict={self.ratings:       batch_ratings,
                                            self.istraining:    True,
                                            self.dropout_rate:  self.dropout})

            n_batches += 1
            total_loss += l
            total_mae += mae
            total_rms += rms

            # if self.verbose:
            #     if n_batches % self.skip_step == 0:
            #         print("Training Epoch {0} Batch {1}: [Loss] = {2} [MAE] = {3}"
            #               .format(epoch, n_batches, total_loss / n_batches, total_mae / n_batches))
        if self.verbose:
            print("="*80)
            print("Training Epoch {0}: [Loss] {1}".format(epoch, total_loss / n_batches))
            print("Training Epoch {0}: [MAE] {1} and [RMS] {2}".format(epoch, total_mae / n_batches, total_rms / n_batches))

    def eval_one_epoch(self, epoch):
        # Input the training data to predict the ratings in the test array
        pred_y = self.session.run(self.pred_y,
                                  feed_dict={self.ratings:      self.train_array,
                                             self.istraining:   False,
                                             self.dropout_rate: self.dropout})
        pred_y = pred_y.clip(min=1, max=5)

        # Extract the nonzero rating indices of the test array from the prediction and test array
        prediction = pred_y[self.test_array.nonzero()]
        truth = self.test_array[self.test_array.nonzero()]

        # Calculate Metrics
        mae = np.mean(np.abs(truth-prediction))
        rms = np.sqrt(np.mean(np.square(truth-prediction)))

        print("Testing Epoch {0} :  [MAE] {1} and [RMS] {2}".format(epoch, mae, rms))
        return mae, rms

    def gen_embedding(self, datafile):
        embedding_array = self.session.run(self.embedding_out,
                                           feed_dict={self.ratings:     self.train_array,
                                                      self.istraining:  False,
                                                      self.dropout_rate:self.dropout})

        gtl.array_to_mat(datafile, embedding = embedding_array)

    # Final Training of the model
    def train(self, restore=False, save=False, datafile=None):

        if restore: # Restore the model from checkpoint
            self.restore_model(datafile, verbose=True)
        else:
            self.session.run(tf.global_variables_initializer())

        if not save: # Do not save the model
            self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                self.eval_one_epoch(i)

        else: # Save the model while training
            _, previous_rms = self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                _,rms = self.eval_one_epoch(i)
                if rms < previous_rms:
                    previous_rms = rms
                    self.save_model(datafile, verbose=False)

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
