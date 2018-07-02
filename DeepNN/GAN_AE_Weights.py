import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import argparse

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The GAN AE Model #######################################################
# Define the class for Minimax AE
class GAN_AE(object):
    def __init__(self, sess, top_K = 10, ranking_list_ratio = 0,
                 num_factors = 32, ae_regs = [0,0,0,0], user_node_reg = 0, noise_keep_prob = 0.5, neg_ratio = 0,
                 eps = 0.5,
                 lr=0.001,
                 epochs=100, batch_size=128, T=10**3, verbose=False):
        # Parse the arguments and store them in the model
        self.session = sess

        self.num_factors = num_factors
        self.ae_regs = ae_regs
        self.user_node_regs = user_node_reg

        self.eps = eps

        self.noise_keep_prob = noise_keep_prob
        self.neg_ratio = neg_ratio


        self.topK = top_K
        self.ranking_list_ratio = ranking_list_ratio

        self.lr = lr

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self, original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape
        self.train_array = train_matrix.toarray()
        _, self.ranking_dict, self.test_dict = \
            mtl.negdict_mat(original_matrix, test_matrix, num_neg=None, neg_ratio=self.ranking_list_ratio)

        self.negative_output_mask = mtl.neg_mask_array(original_matrix, train_matrix, neg_ratio = self.neg_ratio)

        self.num_training = self.train_array.shape[0]
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope('Model',reuse=tf.AUTO_REUSE):
            # Model Feeds
            self.ratings = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item], name='ratings')
            self.output_mask = tf.placeholder(dtype=tf.bool, shape=[None, self.num_item], name='output_mask')

            self.istraining = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
            self.isnegsample = tf.placeholder(dtype=tf.bool, shape=[], name='negative_sample_flag')

            self.layer1_dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='layer1_dropout_rate')

            # self.noise_vector = tf.placeholder(dtype=tf.float32, shape=[None, self.num_noise_factor], name='noise')

            input = self.ratings

            # Encoder
            layer1_w = tf.get_variable(name='encoder_weights',
                                       shape=[self.num_item, self.num_factors],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                       )

            layer1_b = tf.get_variable(name='encoder_bias',
                                       shape=[self.num_factors],
                                       initializer=tf.zeros_initializer(),
                                       )

            # user_node = tf.get_variable(name='user_nodes',
            #                             shape=[self.num_factors],
            #                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03)
            #                             )

            layer1_delta = tf.get_variable(name='encoder_noise',
                                           shape=[self.num_item, self.num_factors],
                                           initializer=tf.zeros_initializer(),
                                           dtype=tf.float32,
                                           trainable=False
                                           )

            # self.update_delta = layer1_delta
            layer1 = tf.sigmoid(tf.matmul(input, layer1_w + layer1_delta) + layer1_b)

            # layer1 = tf.cond(self.istraining,
            #                 lambda : tf.sigmoid(tf.matmul(input, layer1_w + layer1_delta) + layer1_b),
            #                 lambda : tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b))

            layer1_out = tf.cond(self.istraining,
                                 lambda: tf.layers.dropout(layer1, rate=self.layer1_dropout_rate,
                                                           name='layer1_dropout'),
                                 lambda: layer1)
            self.w1 = layer1_w
            self.delta = layer1_delta

            # Decoder
            layer2_w = tf.get_variable(name='decoder_weights',
                                       shape=[self.num_factors, self.num_item],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03)
                                       )

            layer2_b = tf.get_variable(name='decoder_bias',
                                       shape=[self.num_item],
                                       initializer=tf.zeros_initializer()
                                       )

            out_vector = tf.sigmoid(tf.matmul(layer1_out, layer2_w) + layer2_b)
            # out_vector = tf.identity(tf.matmul(layer1_out, layer2_w) + layer2_b)


            # self.b2 = layer2_b
            # self.w2_delta = tf.squeeze(tf.matmul(tf.expand_dims(layer1_delta,0), layer2_w))

            # Output
            # Determine whether negative samples should be considered
            mask = tf.cond(self.isnegsample,
                           lambda: tf.cast(self.output_mask, dtype=out_vector.dtype),
                           lambda: tf.sign(self.ratings))

            self.output = tf.cond(self.istraining,
                                  lambda: tf.multiply(out_vector, mask),
                                  lambda: out_vector)

            # self.pred_y = tf.sigmoid(self.output)
            self.pred_y = self.output

            # Loss
            # coeff = tf.constant(2.0, dtype=tf.float32)
            base_loss = tf.nn.l2_loss(input - self.pred_y)
            # base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.output))

            base_loss = base_loss / tf.cast(tf.shape(input)[0], dtype=base_loss.dtype)  # Average over the batches
            reg_loss = self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
                       self.ae_regs[2] * tf.nn.l2_loss(layer2_w) + self.ae_regs[3] * tf.nn.l2_loss(layer2_b)
            self.loss = base_loss + reg_loss

            # Optimizer
            self.opt = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
            # self.opt2 = tf.train.AdagradOptimizer(self.lr).minimize(-self.loss)

            # Gradients Computation of the Noise
            self.grad_delta = tf.gradients(ys = base_loss, xs = layer1_delta)[0]
            # convert the IndexedSlice Data to Dense Tensor
            self.grad_delta_dense = tf.stop_gradient(self.grad_delta)
            # self.grad_shape = tf.shape(self.grad_delta_dense)
            # normalization: new_grad = (grad / |grad|) * eps

            self.update_delta = layer1_delta.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))

            # self.grad_norm = tf.norm(self.grad_delta_dense)
            # self.update_delta = layer1_delta.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))
            # self.update_delta = layer1_delta.assign(self.eps * tf.nn.l2_normalize(self.grad_delta_dense,0))
            print('Model Building Completed.')

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        random_idx = np.random.permutation(self.num_user)
        n_batches, total_loss = 0, 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                break
                # batch_idx = index[i * self.batch_size:]
                # batch_idx = random_idx[i * self.batch_size:]
                # batch_ratings = self.train_array[batch_idx, :]
                # batch_neg_masks = self.negative_output_mask[batch_idx, :]
            else:
                # batch_idx = index[i * self.batch_size: (i + 1) * self.batch_size]
                batch_idx = random_idx[i * self.batch_size: (i + 1) * self.batch_size]
                batch_ratings = self.train_array[batch_idx, :]
                batch_neg_masks = self.negative_output_mask[batch_idx, :]
                # print(np.shape(batch_ratings))
            # Only calculate the loss of the observed items
            grad, delta = self.session.run([self.grad_delta_dense, self.update_delta],
                                            feed_dict={ self.ratings: batch_ratings, self.output_mask: batch_neg_masks,
                                                        self.istraining: True, self.isnegsample: False,
                                                        self.layer1_dropout_rate: 0.05})

            # Train the model
            _, l = \
                self.session.run([self.opt, self.loss],
                                 feed_dict={self.ratings: batch_ratings, self.output_mask: batch_neg_masks,
                                            self.istraining: True, self.isnegsample: False,
                                            self.layer1_dropout_rate: 0.05})

            n_batches += 1
            total_loss += l

            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("Training Epoch {0} Batch {1}: [Loss] = {2}"
                          .format(epoch, n_batches, total_loss / n_batches))
                    print("Training Epoch {0} Batch {1}: [Grad] = {2}"
                          .format(epoch, n_batches, np.linalg.norm(grad)))
                    # print(delta[:5])

        if self.verbose:
            print("Training Epoch {0}: [Loss] {1}".format(epoch, total_loss / n_batches))

    def eval_one_epoch(self, epoch):
        # Input the uncorrupted training data
        pred_y = self.session.run(self.pred_y,
                                  feed_dict={self.ratings: self.train_array, self.output_mask: self.negative_output_mask,
                                             self.istraining:False, self.isnegsample:False,
                                             self.layer1_dropout_rate: 0})
        pred_y = pred_y.clip(min=0,max=1)

        n_batches, total_hr, total_ndcg, total_mrr = 0, 0, 0, 0
        # Loop for each user (generate the ranking lists for different users)
        for u in self.ranking_dict:
            iid = self.ranking_dict[u]  # The ranking item ids for user u
            rk = pred_y[u, :][np.array(iid)]  # The predicted item values for user u

            hr, ndcg, mrr = evl.rankingMetrics(rk, iid, self.topK, self.test_dict[u])

            n_batches += 1
            total_hr += hr
            total_ndcg += ndcg
            total_mrr += mrr

        avg_hr, avg_mrr, avg_ndcg = total_hr / n_batches, total_mrr / n_batches, total_ndcg / n_batches
        print("Epoch {0}: [HR] {1} and [MRR] {2} and [nDCG@{3}] {4}".format(epoch, avg_hr, avg_mrr, self.topK, avg_ndcg))
        return avg_hr, avg_mrr, avg_ndcg

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
            _,_, previous_ndcg = self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                _,_,ndcg = self.eval_one_epoch(i)
                if ndcg > previous_ndcg:
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
        self.restore_model(datafile,True)
        self.eval_one_epoch(-1)

########################################################################################################################

######################################## Parse Arguments ###############################################################
def parseArgs():
    parser = argparse.ArgumentParser(description="GAE_AE")

    parser.add_argument('--nfactors', type=int, default=64,
                        help='Embedding size.')

    parser.add_argument('--ae_regs', nargs='?', default='[0.5,0.5,0.1,0.1]', type=str,
                        help="Network variable regularization constants.")
    # parser.add_argument('--user_regs', nargs='?', default=1, type=float,
    #                     help="User node regularization constants.")

    # parser.add_argument('--noise_ratio', default=0.6, type=float)
    parser.add_argument('--out_neg_ratio', default=4.0, type=float)

    parser.add_argument('--epsilon', default=1.0, type=float)

    parser.add_argument('--topk', type=int, default=10,
                        help='The K value of the Top-K ranking list.')
    parser.add_argument('--ranking_ratio', type=int, default=0.1,
                        help='The total number of negative items to be ranked when testing')

    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, lr,\
    topK, ranking_list_ratio, num_factors, neg_ratio, epsilon,\
    ae_regs = \
    args.epochs, args.batch_size, args.lr,\
    args.topk, args.ranking_ratio, args.nfactors, args.out_neg_ratio, args.epsilon,\
    args.ae_regs

    ae_regs = list(np.float32(eval(ae_regs)))

    # original_matrix \
    #     = mtl.load_original_matrix(datafile='Data/ml-1m/ratings.dat', header=['uid', 'iid', 'ratings', 'time'], sep='::')

    # train_matrix, test_matrix = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.1, seed=10)

    # original_matrix = mtl.matrix_to_binary(original_matrix, 0)
    # train_matrix, test_matrix = mtl.matrix_to_binary(train_matrix,0), mtl.matrix_to_binary(test_matrix,0)

    # gtl.matrix_to_mat('Data/ML1M_90_Data.mat', opt='coo', original=original_matrix, train=train_matrix, test=test_matrix)
    #
    data = gtl.load_mat_as_matrix('Data/ML1M_90_Data.mat', opt='coo')
    original_matrix, train_matrix, test_matrix = data['original'], data['train'], data['test']

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:
        model = GAN_AE(sess,
                     top_K=topK, ranking_list_ratio=ranking_list_ratio, neg_ratio=neg_ratio,
                     num_factors=num_factors, ae_regs=ae_regs, eps=epsilon,
                     lr=lr,
                     epochs=num_epochs, batch_size=batch_size, T=50, verbose=False)

        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
        model.build_model()
        model.train(restore=False, save=False, datafile='SavedModel/GAN_AE/GAN_AE_ML1M_90.ckpt')