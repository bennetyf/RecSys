import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import argparse

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The WRMF Model #########################################################
# Define the class for WRMF
# Note that WRMF is for binary u-i matrix
class WRMF(object):
    def __init__(self, sess,
                 num_factors =32, regs_ui=[0,0],alpha=1,
                 lr=0.001, topk=10, num_ranking_neg =0,
                 epochs=100, batch_size=128, T=10 ** 3, verbose=False):

        self.session = sess
        self.num_factors = num_factors
        self.regs_user, self.regs_item = regs_ui
        self.alpha = alpha

        self.topk = topk
        self.num_ranking_neg = num_ranking_neg

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self,original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape

        # Negative Dicts for Ranking
        _, self.ranking_dict, self.test_dict = mtl.negdict_mat(original_matrix, test_matrix, num_neg=self.num_ranking_neg)

        # To Lists
        # Contain all the explicit and implicit ratings for training
        self.train_uid, self.train_iid, self.train_ratings = mtl.get_full_matrix(train_matrix, test_matrix)

        # The Cui and Pui parameters in the loss function
        self.weights_ui = [1 + self.alpha * ele for ele in self.train_ratings]
        self.indicators_ui = mtl.list_to_binary(self.train_ratings,0)

        # This is for the ALS algorithm
        self.train_matrix_csr, self.train_matrix_csc = train_matrix.tocsr(), train_matrix.tocsc()

        self.num_training = len(self.train_ratings)
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    def build_model(self,reuse=tf.AUTO_REUSE):
        with tf.variable_scope('Model'):
            # Inputs
            self.uid = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
            self.iid = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.weights = tf.placeholder(dtype=tf.float32, shape=[None], name='weights')
            self.indicators = tf.placeholder(dtype=tf.float32,shape=[None], name='indicator')

            # Latent factor matrices for users and items
            user_lf_matrix = tf.get_variable(name='user_latent_factors',
                                             shape=[self.num_user, self.num_factors],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            item_lf_matrix = tf.get_variable(name='item_latent_factors',
                                             shape=[self.num_item, self.num_factors],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            # Get the latent factors for users and items
            user_latent_factor = tf.nn.embedding_lookup(user_lf_matrix, self.uid, name='lf_user')
            item_latent_factor = tf.nn.embedding_lookup(item_lf_matrix, self.iid, name='lf_item')

            # Outputs
            self.pred_y = tf.einsum('ij,ij->i',user_latent_factor,item_latent_factor)

            # Loss
            base_loss = tf.reduce_sum(tf.multiply(self.weights, tf.square(self.indicators - self.pred_y)))
            reg_loss = self.regs_user * tf.nn.l2_loss(user_latent_factor)+\
                        self.regs_item * tf.nn.l2_loss(item_latent_factor)
            self.loss = base_loss + reg_loss

            # Opt
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            # Metrics
            self.rms = tf.sqrt(tf.reduce_mean(tf.square(self.indicators - self.pred_y)))
            self.mae = tf.reduce_mean(tf.abs(self.indicators - self.pred_y))

            print('Model Building Completed.')

    # The model for ALS algorithm
    def build_model_np(self):
        # Initialization of the latent factors
        self.P = np.random.normal(scale=1. / self.num_factors, size=[self.num_user, self.num_factors])
        self.Q = np.random.normal(scale=1. / self.num_factors, size=[self.num_item, self.num_factors])
        print("Parameter Initialization Completed.")

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        uid, iid, weights, indicators = gtl.shuffle_list(self.train_uid, self.train_iid, self.weights_ui, self.indicators_ui)

        n_batches,total_loss,total_mae,total_rms = 0,0,0,0
        for i in range(self.num_batch):
            batch_user = uid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = iid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_w = weights[i * self.batch_size:(i + 1) * self.batch_size]
            batch_ind = indicators[i * self.batch_size:(i + 1) * self.batch_size]

            _, l, mae, rms = self.session.run([self.opt, self.loss, self.mae, self.rms],
                                                feed_dict={ self.uid: batch_user,self.iid: batch_item,
                                                            self.weights: batch_w,self.indicators:batch_ind})

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
        n_batches, total_hr, total_ndcg, total_mrr = 0, 0, 0, 0
        for u in self.ranking_dict:
            iid = self.ranking_dict[u]
            uid = [u] * len(iid)

            rk = self.session.run(self.pred_y, feed_dict={self.uid: uid, self.iid: iid})

            hr, ndcg, mrr = evl.rankingMetrics(rk, iid, self.topk, self.test_dict[u])

            n_batches += 1
            total_hr += hr
            total_ndcg += ndcg
            total_mrr += mrr

        print("Epoch {0}: [HR] {1} and [MRR] {2} and [nDCG@{3}] {4}".format(epoch,
                total_hr / n_batches, total_mrr / n_batches, self.topk, total_ndcg / n_batches))

    # Final Training of the model
    def train(self):
        self.session.run(tf.global_variables_initializer())
        self.eval_one_epoch(-1)
        for i in range(self.epochs):
            self.train_one_epoch(i)
            self.eval_one_epoch(i)

    # Training using ALS algorithm
    def train_one_epoch_np(self):
        PtP = np.dot(self.P.T,self.P) # P Transpose by P (the matrix for users)
        QtQ = np.dot(self.Q.T,self.Q) # Q Transpose by Q (the matrix for items)

        # Fix item parameters and update user paramters
        for uid in self.train_uid:
            iids_for_u = self.train_matrix_csr.getrow(uid).nonzero()[1]
            if iids_for_u.size:
                self.P[uid,:] = self._als_update(iids_for_u, self.Q, QtQ, 'user')
            else:
                self.P[uid,:] = np.zeros(self.num_factors)

        # Fix user parameters and update item parameters
        for iid in self.train_iid:
            uids_for_i = self.train_matrix_csc.getcol(iid).nonzero()[0]
            if uids_for_i.size:
                self.Q[iid,:] = self._als_update(uids_for_i, self.P, PtP, 'item')
            else:
                self.Q[iid,:] = np.zeros(self.num_factors)

    # Update the user and item parameters in each iteration
    def _als_update(self, indices, X, XX, opt='user'):
        # Update the parameters in ALS algorithm
        Xix = X[indices, :]
        if opt == 'user':
            M = XX + self.alpha * Xix.T.dot(Xix) + np.diag(self.regs_user * np.ones(self.num_factors))
        if opt == 'item':
            M = XX + self.alpha * Xix.T.dot(Xix) + np.diag(self.regs_item * np.ones(self.num_factors))
        else:
            M = np.empty(XX.shape)
        return np.dot(np.linalg.inv(M), (1 + self.alpha) * Xix.sum(axis=0)) # Not working due to sometimes get to singular matrix

    # Evaluate one epoch
    def eval_one_epoch_np(self,epoch):
        # Ranking in all the testing data
        n_batches, n_mrr, total_hr, total_ndcg, total_mrr = 0, 0, 0, 0, 0

        for uid in self.ranking_dict:
            # Prediction to form the ranking list
            rk = []
            for iid in self.ranking_dict[uid]:
                rk.append(np.dot(self.P[uid,:], self.Q[iid,:]))

            _, hr, ndcg, mrr = evl.evalTopK(rk, self.test_dict[uid], self.topk)

            n_batches += 1
            total_hr += hr
            total_ndcg += ndcg
            if np.isinf(mrr):
                pass
            else:
                n_mrr += 1
                total_mrr += mrr

        print("Epoch {0}: [HR] {1} and [nDCG@{2}] {3}".format(epoch, total_hr / n_batches, self.topk,
                                                                  total_ndcg / n_batches))

    # Final training using the SGD algorithm
    def train_np(self):
        self.eval_one_epoch_np(-1)
        for i in range(self.epochs):
            self.train_one_epoch_np()
            self.eval_one_epoch_np(i)

########################################################################################################################

######################################## Parse Arguments ###############################################################
def parseArgs():
    parser = argparse.ArgumentParser(description="WRMF")

    parser.add_argument('--nfactors', type=int, default=100,
                        help='Embedding size.')
    parser.add_argument('--regs_ui', nargs='?', default='[0.01,0.01]', type=str,
                        help="Regularization constants for user and item embeddings.")

    parser.add_argument('--alpha', type=float, default=1,
                        help='Confidence weight, confidence c = 1 + alpha*r where r is the observed "rating"')

    parser.add_argument('--ntest', type=int, default=1,
                        help='Number of test items per user (Leave-N-Out).')

    parser.add_argument('--ndcgk', type=int, default=10,
                        help='The K value of the Top-K ranking list.')
    parser.add_argument('--num_rk', type=int, default=100,
                        help='The total number of negative items to be ranked when testing')

    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, lr, topk, num_rk, num_test,\
    regs_ui, num_factors, alpha = \
    args.epochs, args.batch_size, args.lr, args.ndcgk, args.num_rk, args.ntest,\
    args.regs_ui,args.nfactors,args.alpha

    regs_ui = list(np.float32(eval(regs_ui)))

    # original_matrix, train_matrix, test_matrix,num_users, num_items \
    #     = mtl.load_as_matrix(datafile='Data/ml-100k/u.data', header=['uid', 'iid', 'ratings', 'time'], sep='\t')

    original_matrix \
        = mtl.load_original_matrix(datafile='Data/ml-100k/u.data', header=['uid', 'iid', 'ratings', 'time'], sep='\t')

    original_matrix = mtl.matrix_theshold(original_matrix,threshold=2)

    train_matrix, test_matrix = mtl.matrix_split(original_matrix,n_item_per_user=num_test)

    test_matrix = mtl.matrix_to_binary(test_matrix,0)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:
        model=WRMF(sess,
                 num_factors = num_factors, regs_ui=regs_ui, alpha=alpha,
                 lr=lr, topk=topk, num_ranking_neg=num_rk-num_test,
                 epochs=num_epochs, batch_size=batch_size, T=500, verbose=False)

        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)

        model.build_model()
        model.train()
        # model.build_model_np()
        # model.train_np()