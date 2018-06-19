import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import argparse

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The SVD Model ##########################################################
# Define the class for SVDMF
class SVD(object):
    def __init__(self, sess,
                 num_factors = 16, regs_ui=[0,0], regs_bias=[0,0],
                 lr=0.001,
                 epochs=100, batch_size=128, T=10 ** 3, verbose=False):

        self.session = sess
        self.num_factors = num_factors
        self.regs_user, self.regs_item = regs_ui
        self.regs_bias_user, self.regs_bias_item = regs_bias

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self,original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape
        self.train_uid, self.train_iid, self.train_ratings = mtl.matrix_to_list(train_matrix)
        self.test_uid, self.test_iid, self.test_ratings = mtl.matrix_to_list(test_matrix)
        self.mu = np.mean(self.train_ratings)
        self.num_training = len(self.train_ratings)
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope('Model',reuse=tf.AUTO_REUSE):
            # Inputs
            self.uid = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
            self.iid = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.ratings = tf.placeholder(dtype=tf.float32, shape=[None], name='ratings')

            # Latent factor matrices for users and items
            user_lf_matrix = tf.get_variable(name='user_latent_factors',
                                             shape=[self.num_user, self.num_factors],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            item_lf_matrix = tf.get_variable(name='item_latent_factors',
                                             shape=[self.num_item, self.num_factors],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            # Baseline (Biases)
            bias_user = tf.get_variable(name='user_bias',
                                        shape=[self.num_user],
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            bias_item = tf.get_variable(name='item_bias',
                                        shape=[self.num_item],
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            # Get the latent factors for users and items
            user_latent_factor = tf.nn.embedding_lookup(user_lf_matrix, self.uid,name='lf_user')
            item_latent_factor = tf.nn.embedding_lookup(item_lf_matrix, self.iid,name='lf_item')
            b_u = tf.nn.embedding_lookup(bias_user,self.uid,name='bias_user')
            b_i = tf.nn.embedding_lookup(bias_item,self.iid,name='bias_item')

            # Prediction Outputs
            self.base_pred_y = self.mu + b_i + b_u
            self.pred_y = tf.einsum('ij,ij->i',user_latent_factor,item_latent_factor) + self.mu + b_i + b_u

            # Loss
            base_loss = tf.reduce_sum(tf.square(self.ratings - self.pred_y))
            regs_loss = self.regs_user * tf.nn.l2_loss(user_latent_factor)+\
                        self.regs_item * tf.nn.l2_loss(item_latent_factor)+\
                        self.regs_bias_user * tf.nn.l2_loss(b_u)+\
                        self.regs_bias_item * tf.nn.l2_loss(b_i)
            self.loss = base_loss + regs_loss

            # Optimizer
            self.opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

            # Metrics
            self.rms = tf.sqrt(tf.reduce_mean(tf.square(self.ratings - self.pred_y)))
            self.mae = tf.reduce_mean(tf.abs(self.ratings-self.pred_y))

            print('Model Building Completed.')

    # Build the model for customized SGD algorithm
    def build_model_np(self):
        # Initialize all the parameters
        self.bu = np.random.normal(scale = 1. / self.num_factors, size=[self.num_user])
        self.bi = np.random.normal(scale = 1. / self.num_factors, size=[self.num_item])
        self.P = np.random.normal(scale=1. / self.num_factors, size=[self.num_user, self.num_factors])
        self.Q = np.random.normal(scale=1. / self.num_factors, size=[self.num_factors, self.num_item])
        print("Parameter Initialization Completed.")

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        uid, iid, ratings = gtl.shuffle_list(self.train_uid, self.train_iid, self.train_ratings)

        n_batches,total_loss,total_mae,total_rms = 0,0,0,0
        for i in range(self.num_batch):
            batch_user = uid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = iid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_ratings = ratings[i * self.batch_size:(i + 1) * self.batch_size]

            _, l, mae, rms = self.session.run([self.opt, self.loss, self.mae, self.rms],
                                        feed_dict={self.uid: batch_user,self.iid: batch_item,
                                                   self.ratings: batch_ratings})
            n_batches += 1
            total_loss += l
            total_mae += mae
            total_rms += rms

            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("Epoch {0} Batch {1}: [Loss] = {2} [MAE] = {3}"
                          .format(epoch, n_batches, total_loss / n_batches, total_mae / n_batches))
        if self.verbose:
            print("Epoch {0}: [Loss] {1}".format(epoch, total_loss / n_batches))
            print("Epoch {0} Training:  [MAE] {1} and [RMS] {2}".format(epoch, total_mae / n_batches, total_rms / n_batches))

    def eval_one_epoch(self, epoch):
        # Get the ranking list for each user
        mae, rms = self.session.run([self.mae, self.rms],
                                    feed_dict={self.uid:self.test_uid, self.iid:self.test_iid, self.ratings:self.test_ratings})
        print("Epoch {0} Testing:  [MAE] {1} and [RMS] {2}".format(epoch, mae, rms))

    # Final Training of the model
    def train(self):
        self.session.run(tf.global_variables_initializer())
        self.eval_one_epoch(-1)
        for i in range(self.epochs):
            self.train_one_epoch(i)
            self.eval_one_epoch(i)

    # Training using SGD algorithm
    def train_one_epoch_np(self):
        for uid, iid, ratings in list(zip(self.train_uid, self.train_iid, self.train_ratings)):
            # The estimated rating
            pred_r = self.mu + self.bu[uid] + self.bi[iid] + np.dot(self.Q[:, iid], self.P[uid, :])

            # Calculate the loss of this specific user-item pair
            error = ratings - pred_r

            # Update the parameters
            self.bu[uid] = self.bu[uid] + self.lr * (error - self.regs_bias_user * self.bu[uid])
            self.bi[iid] = self.bi[iid] + self.lr * (error - self.regs_item * self.bi[iid])
            self.P[uid, :] = self.P[uid, :] + self.lr * (error * self.Q[:, iid] - self.regs_user * self.P[uid, :])
            self.Q[:, iid] = self.Q[:, iid] + self.lr * (error * self.P[uid, :] - self.regs_bias_item * self.Q[:, iid])

    # Eval on the testing dataset
    def eval_one_epoch_np(self,epoch):
        mae_list, rms_list = [], []
        for uid, iid, ratings in list(zip(self.test_uid, self.test_iid, self.test_ratings)):
            mae_list.append(np.abs(self.mu + self.bu[uid] + self.bi[iid] + np.dot(self.Q[:, iid], self.P[uid, :]) - ratings))
            rms_list.append((self.mu + self.bu[uid] + self.bi[iid] + np.dot(self.Q[:, iid], self.P[uid, :]) - ratings) ** 2)
        mae = np.mean(mae_list)
        rms = np.sqrt(np.mean(rms_list))
        print("Epoch {0} Testing:  [MAE] {1} and [RMS] {2}".format(epoch, mae, rms))

    # Final training using the SGD algorithm
    def train_np(self):
        self.eval_one_epoch_np(-1)
        for i in range(self.epochs):
            self.train_one_epoch_np()
            self.eval_one_epoch_np(i)

########################################################################################################################

######################################## Parse Arguments ###############################################################
def parseArgs():
    parser = argparse.ArgumentParser(description="SVD")

    parser.add_argument('--nfactors', type=int, default=100,
                        help='Embedding size.')
    parser.add_argument('--regs_ui', nargs='?', default='[0.02,0.02]', type=str,
                        help="Regularization constants for user and item embeddings.")

    parser.add_argument('--regs_bias', nargs='?', default='[0.01,0.01]', type=str)

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0015,
                        help='Learning rate.')

    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, lr,\
    regs_ui, regs_bias, num_factors = \
    args.epochs, args.batch_size, args.lr,\
    args.regs_ui, args.regs_bias,args.nfactors

    regs_ui = list(np.float32(eval(regs_ui)))
    regs_bias = list(np.float32(eval(regs_bias)))

    original_matrix \
        = mtl.load_original_matrix(datafile='Data/ml-1m/ratings.dat', header=['uid', 'iid', 'ratings', 'time'], sep='::')

    train_matrix, test_matrix = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.1, seed=42)
    # gtl.matrix_to_mat('svd_all.mat',opt='all',train_all=train_matrix, test_all=test_matrix)
    # gtl.matrix_to_excel('svd_all.xlsx',opt='coo',train_all=train_matrix, test_all=test_matrix)
    # print("Saved!")

    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:
        model=SVD(sess,
                 num_factors = num_factors, regs_ui=regs_ui, regs_bias=regs_bias,
                 lr=lr,
                 epochs=num_epochs, batch_size=batch_size, T=500, verbose=False)

        # for train_matrix, test_matrix in mtl.matrix_cross_validation(original_matrix, n_splits=5, seed=0):
        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
        model.build_model()
        model.train()
        # model.build_model_np()
        # model.train_np()