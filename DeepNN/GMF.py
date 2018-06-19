import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import argparse
import time

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The GMF Model ##########################################################
# Define the class for GMF
class GMF(object):
    def __init__(self, sess, num_neg, top_K = 10, num_ranking_neg = 0,
                 num_factors=32, regs_emb=[0,0],
                 lr=0.001,
                 epochs=100, batch_size=128, T=10**3, verbose=False):
        '''Constructor'''
        # Parse the arguments and store them in the model
        self.session = sess
        self.num_neg = num_neg
        self.topk = top_K
        self.num_ranking_neg = num_ranking_neg

        self.num_factors = num_factors
        self.regs_user, self.regs_item = regs_emb

        self.lr = lr

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self, original_matrix, train_matrix, test_matrix):
        self.num_user,self.num_item = original_matrix.shape
        self.train_uid, self.train_iid, self.train_labels = mtl.matrix_to_list(train_matrix)
        self.neg_dict, self.ranking_dict, self.test_dict = mtl.negdict_mat(original_matrix, test_matrix,num_neg=self.num_ranking_neg)

        # Negative Sampling on Lists
        print("Enter NegSa")
        start_time = time.time()
        self.train_uid, self.train_iid, self.train_labels = \
            mtl.negative_sample_list(user_list=self.train_uid,item_list=self.train_iid,rating_list=self.train_labels,
                                 num_neg=self.num_neg,neg_val=0,neg_dict=self.neg_dict)
        print("Leaving NegSa")
        print("Negative Sampling Time: {0}".format(time.time() - start_time))

        self.num_training = len(self.train_labels)
        self.num_batch = int(self.num_training / self.batch_size)

        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope('Model',reuse=tf.AUTO_REUSE):
            self.uid = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
            self.iid = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

            # One-hot input
            lf_matrix_user = tf.get_variable(name='user_latent_factors',
                                              shape=[self.num_user, self.num_factors],
                                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            lf_matrix_item = tf.get_variable(name='item_latent_factors',
                                               shape=[self.num_item, self.num_factors],
                                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            latent_factor_user = tf.nn.embedding_lookup(lf_matrix_user, self.uid)
            latent_factor_item = tf.nn.embedding_lookup(lf_matrix_item, self.iid)

            # Extra weights to be learned
            output_w = tf.get_variable(name='output_weights',
                                       shape=[self.num_factors],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            # Output layer
            vector_product = tf.einsum('ij,ij->ij',latent_factor_user,latent_factor_item)
            self.logits = tf.einsum('ij,j->i',vector_product,output_w)
            self.pred_y = tf.sigmoid(self.logits)

            # Loss
            base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            reg_loss  = self.regs_user * tf.nn.l2_loss(latent_factor_user)+\
                         self.regs_item * tf.nn.l2_loss(latent_factor_item)
            self.loss = base_loss + reg_loss

            # Opt
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            # Metrics
            self.rms = tf.sqrt(tf.reduce_mean(tf.square(self.labels - self.pred_y)))
            self.mae = tf.reduce_mean(tf.abs(self.labels - self.pred_y))

            print('Model Building Completed.')

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        uid, iid, lb = gtl.shuffle_list(self.train_uid, self.train_iid, self.train_labels)

        n_batches,total_loss,total_mae,total_rms = 0,0,0,0
        for i in range(self.num_batch):
            batch_user = uid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = iid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_labels = lb[i * self.batch_size:(i + 1) * self.batch_size]

            _, l, mae, rms = self.session.run([self.opt, self.loss, self.mae, self.rms],
                                                feed_dict={self.uid: batch_user,self.iid: batch_item,self.labels: batch_labels})
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
            print("Epoch {0}: [MAE] {1} and [RMS] {2}".format(epoch, total_mae / n_batches, total_rms / n_batches))

    def eval_one_epoch(self, epoch):
        n_batches,total_hr,total_ndcg,total_mrr = 0,0,0,0
        for u in self.ranking_dict:
            iid = self.ranking_dict[u]
            uid = [u] * len(iid)
            rk = self.session.run(self.pred_y, feed_dict={self.uid: uid, self.iid: iid})
            hr, ndcg, mrr = evl.rankingMetrics(rk, iid, self.topk, self.test_dict[u])
            n_batches += 1
            total_hr += hr
            total_ndcg += ndcg
            total_mrr += mrr
        print("Epoch {0}: [HR] {1} and [MRR] {2} and [nDCG@{3}] {4}".format(epoch, total_hr/n_batches, total_mrr/n_batches, self.topk, total_ndcg/n_batches))

    # Final Training of the model
    def train(self):
        self.session.run(tf.global_variables_initializer())
        self.eval_one_epoch(-1)
        for i in range(self.epochs):
            self.train_one_epoch(i)
            self.eval_one_epoch(i)
########################################################################################################################

######################################## Parse Arguments ###############################################################
def parseArgs():
    parser = argparse.ArgumentParser(description="GMF Recommendation")

    parser.add_argument('--nfactors', type=int, default=16,
                        help='Embedding size.')

    parser.add_argument('--ebregs', nargs='?', default='[0.001,0.001]', type=str,
                        help="Regularization constants for user and item embeddings.")

    parser.add_argument('--ntest', type=int, default=1,
                        help='Number of test items per user (Leave-N-Out).')

    parser.add_argument('--num_neg', type=int, default=5,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--ndcgk', type=int, default=10,
                        help='The K value of the Top-K ranking list.')
    parser.add_argument('--num_rk', type=int, default=100,
                        help='The total number of negative items to be ranked when testing')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, \
    reg_embs, num_neg, lr, ndcgk, num_factors, num_ranking_list, num_test = \
    args.epochs, args.batch_size,\
    args.ebregs, args.num_neg, args.lr, args.ndcgk, args.nfactors, args.num_rk, args.ntest

    reg_embs = list(np.float32(eval(reg_embs)))

    # original_matrix, train_matrix, test_matrix, num_users, num_items \
    #     = mtl.load_as_matrix(datafile='Data/books_and_elecs_merged.csv')

    original_matrix \
        = mtl.load_original_matrix(datafile='Data/ml-100k/u.data',header=['uid','iid','ratings','time'],sep='\t')
    original_matrix = mtl.matrix_to_binary(original_matrix, 0)
    train_matrix, test_matrix = mtl.matrix_split(original_matrix, opt='ranking', n_item_per_user=num_test)


    num_users, num_items = original_matrix.shape
    print("Number of users is {0}".format(num_users))
    print("Number of items is {0}".format(num_items))
    print("Number of ratings for all is {0}".format(original_matrix.nnz))
    print("Number of ratings for training is {0}".format(train_matrix.nnz))
    print("Ratings density for training is {0}".format(train_matrix.nnz / (num_users * num_items)))

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:

        model = GMF(sess,
                    num_neg=num_neg, top_K=ndcgk, num_ranking_neg=num_ranking_list-num_test,
                    num_factors=num_factors,regs_emb=reg_embs,
                    lr=lr,
                    epochs=num_epochs, batch_size=batch_size,
                    T=10 ** 3, verbose=False)

        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
        model.build_model()
        model.train()
