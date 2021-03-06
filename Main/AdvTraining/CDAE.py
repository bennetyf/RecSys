import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli
import numpy as np
import argparse

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The CDAE Model #########################################################
# Define the class for Collaborative Denoising Auto-Encoder
class CDAE(object):
    def __init__(self, sess, top_K =[5,10],
                 num_factors=32, ae_regs=[0,0,0,0], user_node_reg=0, noise_keep_prob=0.5,
                 is_neg_sa = False, output_neg_num = 0,
                 lr=0.001,
                 epochs=100, batch_size=128, T=10**3, verbose=False):
        '''Constructor'''
        # Parse the arguments and store them in the model
        self.session = sess

        self.num_factors = num_factors
        self.ae_regs = ae_regs
        self.user_node_regs = user_node_reg

        self.noise_keep_prob = noise_keep_prob
        self.is_neg_sa = is_neg_sa
        self.output_neg_num = output_neg_num

        self.topK = top_K

        self.lr = lr

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self, original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape
        self.train_array = train_matrix.toarray()

        _, self.ranking_dict, self.test_dict = \
            mtl.negdict_mat(original_matrix, test_matrix, num_neg=199, mod='others', random_state=0)

        # self.negative_output_mask = mtl.neg_mask_array(original_matrix, train_matrix, num_neg=self.output_neg_num)

        self.num_training = self.train_array.shape[0]
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope('Model',reuse=tf.AUTO_REUSE):
            # Model Feeds
            self.ratings = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item], name='ratings')
            # self.output_mask = tf.placeholder(dtype=tf.bool, shape=[None, self.num_item], name='output_mask')
            self.uid = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')

            self.istraining = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
            self.isnegsample = tf.placeholder(dtype=tf.bool, shape=[], name='negative_sample_flag')
            self.layer1_dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='layer1_dropout_rate')

            # Add Noise to the input ratings
            bernoulli_generator = Bernoulli(probs=self.noise_keep_prob, dtype=self.ratings.dtype)
            corruption_mask = bernoulli_generator.sample(tf.shape(self.ratings))
            corrupted_input = tf.multiply(self.ratings, corruption_mask)

            # Decide the input of the auto-encoder (corrupted at training time and uncorrupted at testing time)
            input = tf.cond(self.istraining, lambda:corrupted_input, lambda:self.ratings)

            # Encoder
            layer1_w = tf.get_variable(name='encoder_weights',
                                        shape=[self.num_item, self.num_factors],
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            layer1_b = tf.get_variable(name='encoder_bias',
                                        shape=[self.num_factors],
                                        initializer=tf.zeros_initializer())

            user_embedding = tf.get_variable(name='user_embedding',
                                             shape=[self.num_user, self.num_factors],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            # Decoder
            layer2_w = tf.get_variable(name='decoder_weights',
                                        shape=[self.num_factors, self.num_item],
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            layer2_b = tf.get_variable(name='decoder_bias',
                                        shape=[self.num_item],
                                        initializer=tf.zeros_initializer())

            user_node = tf.nn.embedding_lookup(user_embedding, self.uid)
            layer1 = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b + user_node)
            layer1_out = tf.cond(self.istraining,
                                 lambda: tf.layers.dropout(layer1, rate=self.layer1_dropout_rate, name='layer1_dropout'),
                                 lambda: layer1)
            # layer1 = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b)
            out_vector = tf.identity(tf.matmul(layer1_out, layer2_w) + layer2_b)

            # Output
            # Determine whether negative samples should be considered
            # mask = tf.cond(self.isnegsample,
            #                lambda : tf.cast(self.output_mask, dtype=out_vector.dtype),
            #                lambda : tf.sign(self.ratings))
            #
            # self.output = tf.cond(self.istraining,
            #                       lambda : tf.multiply(out_vector, mask),
            #                       lambda : out_vector)

            self.output = out_vector
            self.pred_y = tf.sigmoid(self.output)

            # Loss
            base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.output))
            # base_loss = tf.nn.l2_loss(self.pred_y - input)
            base_loss = base_loss / tf.cast(tf.shape(input)[0], dtype=base_loss.dtype) # Average over the batches
            reg_loss = self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + self.ae_regs[1] * tf.nn.l2_loss(layer1_b) +\
                       self.ae_regs[2] * tf.nn.l2_loss(layer2_w) + self.ae_regs[3] * tf.nn.l2_loss(layer2_b) +\
                       self.user_node_regs * tf.nn.l2_loss(user_embedding)

            # reg_loss = self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
            #            self.ae_regs[2] * tf.nn.l2_loss(layer2_w) + self.ae_regs[3] * tf.nn.l2_loss(layer2_b)
            self.loss = base_loss + reg_loss

            # Optimizer
            self.opt = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)

        print('Model Building Completed.')

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        random_idx = np.random.permutation(self.num_user)
        # np.random.shuffle(self.train_array)
        # index = list(range(self.num_user))

        n_batches, total_loss = 0, 0
        for i in range(self.num_batch):

            if i == self.num_batch - 1:
                # batch_idx = index[i * self.batch_size:]
                batch_idx = random_idx[i * self.batch_size:]
                batch_ratings = self.train_array[batch_idx,:]
                # batch_neg_masks = self.negative_output_mask[batch_idx,:]
            else:
                # batch_idx = index[i * self.batch_size: (i + 1) * self.batch_size]
                batch_idx = random_idx[i * self.batch_size: (i + 1) * self.batch_size]
                batch_ratings = self.train_array[batch_idx,:]
                # batch_neg_masks = self.negative_output_mask[batch_idx,:]

            _, l = \
                self.session.run([self.opt, self.loss],
                                 feed_dict={self.ratings:       batch_ratings,
                                            # self.output_mask:   batch_neg_masks,
                                            self.uid:           batch_idx,
                                            self.istraining:    True,
                                            self.isnegsample:   self.is_neg_sa,
                                            self.layer1_dropout_rate: 0.05})

            n_batches += 1
            total_loss += l

            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("Training Epoch {0} Batch {1}: [Loss] = {2}"
                          .format(epoch, n_batches, total_loss / n_batches))
        if self.verbose:
            print("Training Epoch {0}: [Loss] {1}".format(epoch, total_loss / n_batches))

    def eval_one_epoch(self, epoch):
        # Input the uncorrupted training data
        pred_y = self.session.run(self.pred_y,
                                  feed_dict={self.ratings: self.train_array,
                                             # self.output_mask: self.negative_output_mask,
                                             self.uid: range(self.num_user),
                                             self.istraining:False,
                                             self.isnegsample:self.is_neg_sa,
                                             self.layer1_dropout_rate: 0})
        pred_y = pred_y.clip(min=0, max=1)

        n_batches=0
        total_hr, total_ndcg = np.zeros(len(self.topK)), np.zeros(len(self.topK))
        # Loop for each user (generate the ranking lists for different users)
        for u in self.ranking_dict:
            iid = self.ranking_dict[u] # The ranking item ids for user u
            rk = pred_y[u,np.array(iid)] # The predicted item values for user u
            n_batches += 1

            hr, ndcg = evl.rankingMetrics(rk, iid, self.topK, self.test_dict[u], mod='hr')
            total_hr += hr
            total_ndcg += ndcg

        avg_hr, avg_ndcg = total_hr / n_batches, total_ndcg / n_batches

        for i in range(len(self.topK)):
            print('-' * 55)
            print("Epoch {0}: [HR@{1}] {2}".format(epoch, self.topK[i], avg_hr[i]))
            print("Epoch {0}: [nDCG@{1}] {2}".format(epoch, self.topK[i], avg_ndcg[i]))
        print('=' * 55)
        return avg_hr[0], avg_ndcg[0]

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
            pre_metric1, pre_metric2 = self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                metric1, metric2 = self.eval_one_epoch(i)
                if metric1 < pre_metric1:
                    pre_metric1 = metric1
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

######################################## Parse Arguments ###############################################################
def parseArgs():
    parser = argparse.ArgumentParser(description="CDAE")

    parser.add_argument('--nfactors', type=int, default=100,
                        help='Embedding size.')

    parser.add_argument('--ae_regs', nargs='?', default='[0.015,0.015,0.015,0.015]', type=str,
                        help="Network variable regularization constants.")
    parser.add_argument('--user_regs', nargs='?', default=0.015, type=float,
                        help="User node regularization constants.")

    parser.add_argument('--noise_ratio', default=1.0, type=float)
    parser.add_argument('--out_neg_num', default=4, type=float)

    parser.add_argument('--topk', default=[5,10],
                        help='The K value of the Top-K ranking list.')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0004,
                        help='Learning rate.')
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, lr,\
    topK, num_factors, noise_keep_prob, neg_num,\
    ae_regs, user_regs = \
    args.epochs, args.batch_size, args.lr,\
    args.topk, args.nfactors, args.noise_ratio, args.out_neg_num,\
    args.ae_regs, args.user_regs

    ae_regs = list(np.float32(eval(ae_regs)))

    # path = 'Data/ciao-hr-45.mat'
    # path = 'Data/ml1m-hr-45.mat'
    path = 'Data/filmtrust-hr-34.mat'
    print(path)

    data = gtl.load_mat_as_matrix(path, opt='coo')
    original_matrix, train_matrix, test_matrix = data['original'], data['train'], data['test']

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:
        model = CDAE(sess,
                     top_K = topK,
                     is_neg_sa = False,
                     output_neg_num = neg_num,
                     num_factors = num_factors,
                     ae_regs = ae_regs,
                     user_node_reg = user_regs,
                     noise_keep_prob = noise_keep_prob,
                     lr = lr,

                     epochs = num_epochs, batch_size = batch_size, T = 10**3, verbose = False)

        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
        model.build_model()
        model.train(restore=False, save=False, datafile='SavedModel/CDAE/CDAE_ML1M_80.ckpt')