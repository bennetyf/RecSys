import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import random
import argparse
import inspect

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The BPR Model ##########################################################

# Define the class for BPRMF
class BPRMF(object):
    def __init__(self, sess, top_K = 10, num_ranking_neg = 0,
                    num_factors =32, regs_ui=[0,0],
                    lr=0.001,
                    epochs=100, batch_size=128, T=10 ** 3, verbose=False):

        self.session = sess
        self.topK = top_K
        self.num_ranking_neg = num_ranking_neg

        self.num_factors = num_factors
        self.num_neg = 1

        self.regs_user, self.regs_item = regs_ui

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

        gtl.print_paras(inspect.currentframe())

    def prepare_data(self,original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape
        self.train_uid, self.train_iid, _ = mtl.matrix_to_list(train_matrix)
        self.neg_dict, self.ranking_dict, self.test_dict = \
            mtl.negdict_mat(original_matrix, test_matrix, mod='precision', random_state=20)
        # self.neg_dict, self.ranking_dict, self.test_dict = mtl.negdict_mat(original_matrix, test_matrix, num_neg=self.num_ranking_neg)
        self.num_training = len(self.train_uid)
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope('Model',reuse=tf.AUTO_REUSE):
            self.uid = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
            self.iid = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.neg_iid = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_item_id')

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

            # Loss
            self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.x_uij)))
            self.loss +=    self.regs_user * tf.nn.l2_loss(user_lf_matrix)+\
                            self.regs_item * tf.nn.l2_loss(item_lf_matrix)
                            # self.regs_item * tf.nn.l2_loss(neg_item_latent_factor)

            # Opt
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            print('Model Building Completed.')

############################################### Functions to run the model #############################################

    def train_one_epoch(self,epoch):
        uid, iid, lb = gtl.shuffle_list(self.train_uid, self.train_iid, self.train_labels)
        neg_iid = [random.choice(self.neg_dict[u]) for u in uid]
        # neg_iid = [np.random.choice(self.neg_dict[u], 1).item() for u in self.train_uid] #This is very slow
        # neg_iid = [self.neg_dict[u][np.random.randint(len(self.neg_dict[u]))] for u in uid] #This is much faster

        n_batches, total_loss = 0,0
        for i in range(self.num_batch):
            batch_user = uid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = iid[i * self.batch_size:(i + 1) * self.batch_size]
            batch_labels = lb[i * self.batch_size:(i + 1) * self.batch_size]
            batch_neg_iid = neg_iid[i * self.batch_size:(i + 1) * self.batch_size]
            # Randomly select one negative item j for each user
            # batch_neg_iid = [np.random.choice(self.neg_dict[u],1).item() for u in batch_user]

            _, l = self.session.run([self.opt, self.loss],
                                    feed_dict={ self.uid:batch_user,self.iid:batch_item,
                                                self.neg_iid:batch_neg_iid})
            n_batches += 1
            total_loss += l

            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("Epoch {0} Batch {1}: [Loss] = {2}"
                          .format(epoch, n_batches, total_loss / n_batches))
        if self.verbose:
            print("Epoch {0}: [Loss] {1}".format(epoch, total_loss / n_batches))

    def eval_one_epoch(self,epoch):
        n_batches, total_prec, total_recall, total_ap = 0, 0, 0, 0
        # n_batches, total_hr, total_ndcg, total_mrr = 0, 0, 0, 0
        for u in self.ranking_dict:

            if len(self.test_dict[u]) == 0:
                continue

            iid = self.ranking_dict[u]
            uid = [u] * len(iid)

            rk = self.session.run(self.pred_y, feed_dict={self.uid: uid, self.iid: iid})

            # hr, ndcg, mrr = evl.rankingMetrics(rk, iid, self.topK, self.test_dict[u], mod='hr')
            prec, recall,_,_,_ = evl.rankingMetrics(rk, iid, [self.topK], self.test_dict[u], mod='precision',is_map=False)
            n_batches += 1
            # total_hr += hr
            # total_ndcg += ndcg
            # total_mrr += mrr
            total_prec += prec[0]
            total_recall += recall[0]

        # avg_hr, avg_mrr, avg_ndcg = total_hr / n_batches, total_mrr / n_batches, total_ndcg / n_batches
        # print("Epoch {0}: [HR] {1} and [MRR] {2} and [nDCG@{3}] {4}".format(epoch, avg_hr, avg_mrr, self.topK, avg_ndcg))
        # return avg_hr, avg_mrr,avg_ndcg
        avg_prec, avg_recall, avg_ap = total_prec / n_batches, total_recall / n_batches, total_ap / n_batches
        print("Epoch {0}: [Precision@{1}] {2}".format(epoch, self.topK, avg_prec))
        print("Epoch {0}: [Recall@{1}] {2}".format(epoch, self.topK, avg_recall))
        print("Epoch {0}: [MAP@{1}] {2}".format(epoch, self.topK, avg_ap))
        print("=" * 40)

        return avg_prec, avg_recall, avg_ap

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
            _, _, previous_ndcg = self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                _, _, ndcg = self.eval_one_epoch(i)
                if ndcg < previous_ndcg:
                    previous_ndcg = ndcg
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
        self.restore_model(datafile, True)
        self.eval_one_epoch(-1)

########################################################################################################################

######################################## Parse Arguments ###############################################################
def parseArgs():
    parser = argparse.ArgumentParser(description="BPRMF")

    parser.add_argument('--nfactors', type=int, default=50,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0.0001,0.0001]', type=str,
                        help="Regularization constants for user and item embeddings.")

    parser.add_argument('--ntest', type=int, default=1,
                        help='Number of test items per user (Leave-N-Out).')
    parser.add_argument('--ndcgk', type=int, default=10,
                        help='The K value of the Top-K ranking list.')
    parser.add_argument('--num_rk', type=int, default=100,
                        help='The total number of negative items to be ranked when testing')


    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, \
    regs, lr, ndcgk, num_factors, num_ranking_list, num_test = \
    args.epochs, args.batch_size,\
    args.regs, args.lr, args.ndcgk, args.nfactors, args.num_rk, args.ntest

    regs_ui = list(np.float32(eval(regs)))

    # original_matrix, train_matrix, test_matrix, num_users, num_items \
    #     = mtl.load_as_matrix(datafile='Data/books_and_elecs_merged.csv')

    original_matrix \
        = mtl.load_original_matrix(datafile='Data/ml-1m/ratings.dat', header=['uid', 'iid', 'ratings', 'time'], sep='::')

    # original_matrix = mtl.matrix_theshold(original_matrix,threshold=2)

    original_matrix = mtl.matrix_to_binary(original_matrix,0)

    # train_matrix, test_matrix = mtl.matrix_split(original_matrix,n_item_per_user=num_test)
    train_matrix, test_matrix = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.2, random_state=10)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:
        model=BPRMF(sess,
                    top_K = ndcgk, num_ranking_neg = num_ranking_list-num_test,
                    num_factors = num_factors, regs_ui=regs_ui,
                    lr=lr,
                    epochs=num_epochs, batch_size=batch_size, T=10 ** 3, verbose=False)

        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)

        model.build_model()
        model.train()
