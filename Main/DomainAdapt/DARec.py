# import sys,os
# sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
import numpy as np

from DAUtils import flip_gradient

# import Utils.RecEval as evl
# import Utils.MatUtils as mtl
# import Utils.GenUtils as gtl

########################################### The Cross-domain DA Model ##################################################
class DARec(object):
    def __init__(self, sess,
                 top_K =[5,10],
                 input_dim = 200,

                 pred_dim = 100,
                 pred_lambda = 1.0,
                 pred_reg = 0.0,

                 cls_layers=[100, 2],
                 cls_reg = 0.0,

                 pred_cls_lambda = 1.0,
                 grl_lambda = 1.0,

                 drop_out_rate = 0.0,

                 lr=0.001,
                 epochs=100, batch_size=128, T=10**3, verbose=False):
        '''Constructor'''
        # Parse the arguments and store them in the model
        self.session = sess

        self.input_dim = input_dim

        self.pred_dim = pred_dim
        self.pred_lambda = pred_lambda
        self.pred_reg = pred_reg

        self.cls_layers = cls_layers
        self.cls_reg = cls_reg

        self.pred_cls_lambda = pred_cls_lambda

        self.grl_coeff = grl_lambda
        self.drop_out = drop_out_rate

        self.topK = top_K
        self.lr = lr

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self, original_matrix_sc, train_matrix_sc, test_matrix_sc,
                            original_matrix_tg, train_matrix_tg, test_matrix_tg,
                            embedding_arr_sc, embedding_arr_tg):

        self.num_user_sc, self.num_item_sc = original_matrix_sc.shape
        self.num_user_tg, self.num_item_tg = original_matrix_tg.shape
        assert self.num_user_sc == self.num_user_tg

        self.train_array_sc, self.test_array_sc = train_matrix_sc.toarray(), test_matrix_sc.toarray()
        self.train_array_tg, self.test_array_tg = train_matrix_tg.toarray(), test_matrix_tg.toarray()

        self.embed_arr = np.vstack((embedding_arr_sc, embedding_arr_tg))
        self.domain_arr = np.vstack((np.tile([1., 0.], [self.num_user_sc, 1]),
                                    np.tile([0., 1.], [self.num_user_tg, 1])))

        self.num_training = self.num_user_sc * 2  # There are two domains with the same number of users
        self.num_batch = self.num_training // self.batch_size

        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope("DARec_Model", reuse=tf.AUTO_REUSE):
            self.istraining = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')

            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='features')
            self.ratings_sc = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item_sc], name='ratings_sc')
            self.ratings_tg = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item_tg], name='ratings_tg')

            self.domain = tf.placeholder(tf.float32, [None, 2])
            self.grl = tf.placeholder(tf.float32, [])

            with tf.variable_scope("Rating_Predictions",reuse=tf.AUTO_REUSE): # Map the two domains into a shared space
                pred_w1 = tf.get_variable(name='shared_w1',
                                            shape=[self.pred_dim, self.input_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                pred_b1 = tf.get_variable(name='shared_b1',
                                            shape=[self.pred_dim],
                                            initializer=tf.zeros_initializer())

                pred_w2_sc = tf.get_variable(name='shared_w2_sc',
                                            shape=[self.pred_dim, self.num_item_sc],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                pred_b2_sc = tf.get_variable(name='shared_b2_sc',
                                            shape=[self.num_item_sc],
                                            initializer=tf.zeros_initializer())

                pred_w2_tg = tf.get_variable(name='shared_w2_tg',
                                            shape=[self.pred_dim, self.num_item_tg],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                pred_b2_tg = tf.get_variable(name='shared_b2_tg',
                                            shape=[self.num_item_tg],
                                            initializer=tf.zeros_initializer())

                # AutoEncoder for Shared Feature Extraction
                pred_encode = tf.sigmoid(tf.matmul(pred_w1, self.input) + pred_b1)

                shared_vec = tf.cond(self.istraining,
                                     lambda: tf.layers.dropout(pred_encode, rate=self.dropout_rate, name='encode_dropout'),
                                     lambda: pred_encode)

                pred_decode_sc = tf.identity(tf.matmul(pred_w2_sc, shared_vec) + pred_b2_sc)
                pred_decode_tg = tf.identity(tf.matmul(pred_w2_tg, shared_vec) + pred_b2_tg)

                # Rating Prediction
                self.pred_y_sc = tf.cond(self.istraining,
                                         lambda: tf.multiply(pred_decode_sc, tf.sign(self.ratings_sc)),
                                         lambda: pred_decode_sc)

                self.pred_y_tg = tf.cond(self.istraining,
                                         lambda: tf.multiply(pred_decode_tg, tf.sign(self.ratings_tg)),
                                         lambda: pred_decode_tg)

                # Losses for Rating Prediction
                base_loss = tf.reduce_sum(tf.square(self.ratings_sc - self.pred_y_sc)) \
                            + self.pred_lambda * tf.reduce_sum(tf.square(self.ratings_tg - self.pred_y_tg))
                reg_loss = self.pred_reg * \
                           (tf.nn.l2_loss(pred_w1) + tf.nn.l2_loss(pred_b1) +
                            tf.nn.l2_loss(pred_w2_sc) + tf.nn.l2_loss(pred_b2_sc) +
                            tf.nn.l2_loss(pred_w2_tg) + tf.nn.l2_loss(pred_b2_tg))
                self.pred_loss = base_loss + reg_loss

            with tf.variable_scope("Domain_Classifier",reuse=tf.AUTO_REUSE):
                # Flip the gradient when backpropagating through this operation
                feat = flip_gradient(shared_vec, self.grl)
                # MLP for domain classification
                mlp_vector = feat

                assert self.cls_layers[-1] == 2 # Check whether the classification is binary
                for i in range(len(self.cls_layers)):
                    mlp_vector = tf.layers.dense(mlp_vector, units=self.cls_layers[i], activation=tf.nn.relu,
                                                 use_bias=True,
                                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.cls_reg),
                                                 bias_initializer=tf.zeros_initializer(),
                                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.cls_reg),
                                                 name='cls_layer{0}'.format(i))

                self.cls_y = tf.nn.softmax(mlp_vector)
                self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=mlp_vector, labels=self.domain) + \
                                   tf.losses.get_regularization_loss(scope="DARec_Model/Domain_Classifier")

            self.total_loss = self.pred_loss + self.pred_cls_lambda * self.domain_loss
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

            # Metrics
            self.train_rms_sc = tf.sqrt(tf.reduce_mean(tf.square(self.ratings_sc - self.pred_y_sc)))
            self.train_rms_tg = tf.sqrt(tf.reduce_mean(tf.square(self.ratings_tg - self.pred_y_tg)))
            self.train_rms = tf.sqrt(tf.square(self.train_rms_sc) + tf.square(self.train_array_tg))

            self.correct_domain_pred = tf.equal(tf.argmax(self.domain, 1), tf.argmax(self.cls_y, 1))
            self.domain_acc = tf.reduce_mean(tf.cast(self.correct_domain_pred, tf.float32))

            print('Model Building Completed.')

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        n_batches = 0
        total_loss = 0
        total_rms = 0
        total_domain_acc = 0

        # Select users from the two different domains in an unbalanced way
        # random_idx = np.random.permutation(self.num_batch * self.batch_size)

        # Select users from the two different domains in a balanced way
        random_idx = np.vstack((np.random.permutation(self.num_user_sc),
                                np.random.permutation(self.num_user_sc) + self.num_user_sc))\
                        .flatten('F')

        for i in range(self.num_batch):
            # Prepare feeding data for one batch
            if i == self.num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            else:
                batch_idx = random_idx[i * self.batch_size: (i + 1) * self.batch_size]

            idx = [k if k < self.num_user_sc else k - self.num_user_sc for k in batch_idx]

            feed_config = {
                self.istraining: True,
                self.dropout_rate: self.drop_out,
                self.input: self.embed_arr[batch_idx],
                self.ratings_sc: self.train_array_sc[idx],
                self.ratings_tg: self.train_array_tg[idx],
                self.domain: self.domain_arr[batch_idx],
                self.grl: self.grl_coeff
            }
            _, batch_loss, batch_rms, batch_domain_acc \
                = self.session.run([self.opt, self.total_loss, self.train_rms, self.domain_acc],
                                    feed_dict=feed_config)

            n_batches += 1
            total_loss += batch_loss
            total_rms += batch_rms
            total_domain_acc += batch_domain_acc

        if self.verbose:
            print("Epoch {0}: [Loss] {1}, [RMSE] {2}, [DomAcc] {3}"
                  .format(epoch, total_loss/n_batches, total_rms/n_batches, total_domain_acc/n_batches))

    def eval_one_epoch(self, epoch):
        # Input the training data to predict the ratings in the test array
        feed_config = {
            self.istraining: False,
            self.dropout_rate: self.drop_out,
            self.input: self.embed_arr,
            self.ratings_sc: self.train_array_sc,
            self.ratings_tg: self.train_array_tg,
            self.domain: self.domain_arr,
            self.grl: self.grl_coeff
        }

        pred_y_sc, pred_y_tg, domain_acc = self.session.run([self.pred_y_sc, self.pred_y_tg, self.domain_acc],
                                                            feed_dict=feed_config)
        pred_y_sc, pred_y_tg = pred_y_sc.clip(min=1, max=5), pred_y_tg.clip(min=1, max=5)

        # Extract the nonzero rating indices of the test array from the prediction and test array
        test_pred_y_sc, test_pred_y_tg = pred_y_sc[self.test_array_sc.nonzero()], \
                                         pred_y_tg[self.test_array_tg.nonzero()]

        truth_sc, truth_tg = self.test_array_sc[self.test_array_sc.nonzero()], \
                             self.test_array_tg[self.test_array_tg.nonzero()]

        # Calculate Metrics
        mae_sc, mae_tg = np.mean(np.abs(truth_sc - test_pred_y_sc)), \
                         np.mean(np.abs(truth_tg - test_pred_y_tg))
        rms_sc, rms_tg = np.sqrt(np.mean(np.square(truth_sc - test_pred_y_sc))), \
                         np.sqrt(np.mean(np.square(truth_tg - test_pred_y_tg)))
        mae_total = (mae_sc + mae_tg) / 2
        rms_total = np.sqrt(np.square(rms_sc) + np.square(rms_tg))

        print("Testing Epoch {0} :  [Source_MAE] {1} and [Source_RMS] {2}".format(epoch, mae_sc, rms_sc))
        print("Testing Epoch {0} :  [Target_MAE] {1} and [Target_RMS] {2}".format(epoch, mae_tg, rms_tg))
        print("Testing Epoch {0} :  [MAE] {1} and [RMS] {2}".format(epoch, mae_total, rms_total))
        print("Testing Epoch {0} :  [DomAcc] {1}".format(epoch, domain_acc))
        return mae_total, rms_total

    # Final Training of the model
    def train(self, restore=False, save=False, datafile=None):

        if restore:  # Restore the model from checkpoint
            self.restore_model(datafile, verbose=True)
        else:
            self.session.run(tf.global_variables_initializer())

        if not save:  # Do not save the model
            self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                self.eval_one_epoch(i)

        else:  # Save the model while training
            _, previous_rms = self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                _, rms = self.eval_one_epoch(i)
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

########################################################################################################################
