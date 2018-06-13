import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import argparse

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The BPR Model ##########################################################

# Define the class for BPRMF
class BPRMF(object):
    def __init__(self, sess, top_K = 10, num_ranking_list = 100,
                    num_factors =32, regs_ui=[0,0],
                    lr=0.001,
                    epochs=100, batch_size=128, T=10 ** 3, verbose=False):

        self.session = sess
        self.topK = top_K
        self.num_ranking_list = num_ranking_list

        self.num_factors = num_factors
        self.num_neg = 1

        self.regs_user, self.regs_item = regs_ui

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self,original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape
        self.train_uid, self.train_iid, self.train_labels = mtl.matrix_to_list(train_matrix)
        self.test_uid, self.test_iid, self.test_labels = mtl.matrix_to_list(test_matrix)
        self.neg_dict, self.test_dict = mtl.negdict_mat(original_matrix, test_matrix,
                                                        num_neg=self.num_ranking_list - 1)
        self.num_training = len(self.train_labels)
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    def model(self):
        with tf.variable_scope('Model'):
            self.uid = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
            self.iid = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.neg_iid = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_item_id')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

            # Latent factor matrices for users and items
            user_lf_matrix = tf.get_variable(name='user_latent_factors',
                                            shape=[self.num_user, self.num_factors],
                                            initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))
            item_lf_matrix = tf.get_variable(name='item_latent_factors',
                                           shape=[self.num_item, self.num_factors],
                                           initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))

            # Get the latent factors for users and items
            user_latent_factor = tf.nn.embedding_lookup(user_lf_matrix, self.uid)
            item_latent_factor = tf.nn.embedding_lookup(item_lf_matrix, self.iid)
            neg_item_latent_factor = tf.nn.embedding_lookup(item_lf_matrix, self.neg_iid)

            # Get the estimated probabilities of user u ranking positive item i higher than negative item j
            # Notions follow the original BPR paper
            self.x_ui = tf.einsum('ij,ij->i',user_latent_factor,item_latent_factor)
            self.x_uj = tf.einsum('ij,ij->i',user_latent_factor,neg_item_latent_factor)
            self.x_uij = self.x_ui - self.x_uj
            self.pred_y = self.x_ui

    def regularizer(self):
        with tf.variable_scope('Model', reuse=True):
            if self.regs_user:
                tf.add_to_collection('regs',
                                    tf.contrib.layers.apply_regularization(
                                    tf.contrib.layers.l2_regularizer(scale=self.regs_user),
                                    [tf.get_variable('user_latent_factors')]))
            if self.regs_item:
                tf.add_to_collection('regs',
                                    tf.contrib.layers.apply_regularization(
                                    tf.contrib.layers.l2_regularizer(scale=self.regs_item),
                                    [tf.get_variable('item_latent_factors')]))

    def loss(self):
        with tf.name_scope('Loss'):
            base_loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.x_uij)))
            self.regularizer()
            regs_loss= tf.get_collection('regs')
            self.loss = tf.add_n([base_loss] + regs_loss, name="loss")

    def optimizer(self):
        with tf.name_scope('Optimizer'):
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def build(self):
        self.model()
        self.loss()
        self.optimizer()
        print('Model Building Completed.')

############################################### Functions to run the model #############################################

    def train_one_epoch(self,epoch):
        uid, iid, lb = gtl.shuffle_list(self.train_uid, self.train_iid, self.train_labels)

        n_batches, total_loss = 0,0
        for i in range(self.num_batch):
            batch_user = uid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = iid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_labels = lb[i * self.batch_size:(i + 1) * self.batch_size]
            # Randomly select one negative item j for each user
            batch_neg_iid = [np.random.choice(self.neg_dict[u],1).item() for u in batch_user]

            _, l = self.session.run([self.opt, self.loss],
                                    feed_dict={ self.uid:batch_user,self.iid:batch_item,
                                                self.labels:batch_labels,self.neg_iid:batch_neg_iid})
            n_batches += 1
            total_loss += l

            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("Epoch {0} Batch {1}: [Loss] = {2}"
                          .format(epoch, n_batches, total_loss / n_batches))

    def eval_one_epoch(self,epoch):
        # Get the ranking list for each user
        uid, iid = [], []
        for u in range(self.num_user):
            uid.extend([u] * self.num_ranking_list)
            iid.extend(self.test_dict[u])

        batch_size = self.num_ranking_list

        n_batches, n_mrr, total_hr, total_ndcg, total_mrr = 0, 0, 0, 0, 0
        for i in range(self.num_user):
            batch_uid, batch_iid = uid[i * batch_size:(i + 1) * batch_size], iid[i * batch_size:(i + 1) * batch_size]
            rk = self.session.run(self.pred_y, feed_dict={self.uid: batch_uid, self.iid: batch_iid})
            _, hr, ndcg, mrr = evl.evalTopK(rk, batch_iid, self.topK)
            n_batches += 1
            total_hr += hr
            total_ndcg += ndcg
            if np.isinf(mrr):
                pass
            else:
                n_mrr += 1
                total_mrr += mrr
        print("Epoch {0}: [HR] {1} and [nDCG@{2}] {3}".format(epoch, total_hr / n_batches, self.topK, total_ndcg / n_batches))

    # Final Training of the model
    def train(self):
        self.session.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
        self.eval_one_epoch(-1)
        for i in range(self.epochs):
            self.train_one_epoch(i)
            self.eval_one_epoch(i)

########################################################################################################################

######################################## Parse Arguments ###############################################################
def parseArgs():
    parser = argparse.ArgumentParser(description="BPRMF")

    parser.add_argument('--nfactors', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0.001,0.001]', type=str,
                        help="Regularization constants for user and item embeddings.")

    parser.add_argument('--ndcgk', type=int, default=10,
                        help='The K value of the Top-K ranking list.')
    parser.add_argument('--num_rk', type=int, default=100,
                        help='The total number of negative items to be ranked when testing')


    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, \
    regs, lr, ndcgk, num_factors, num_ranking_list = \
    args.epochs, args.batch_size,\
    args.regs, args.lr, args.ndcgk, args.nfactors, args.num_rk


    regs_ui = list(np.float32(eval(regs)))

    original_matrix, train_matrix, test_matrix, num_users, num_items \
        = mtl.load_as_matrix(datafile='Data/books_and_elecs_merged.csv')

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:
        model=BPRMF(sess,
                    top_K = ndcgk, num_ranking_list = num_ranking_list,
                    num_factors = num_factors, regs_ui=regs_ui,
                    lr=lr,
                    epochs=num_epochs, batch_size=batch_size, T=10 ** 3, verbose=False)

        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)

        model.build()
        model.train()