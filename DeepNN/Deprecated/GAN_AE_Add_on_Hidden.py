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
    def __init__(self, sess, top_K = 10,
                 num_factors = 32, ae_regs = [0,0,0,0], user_node_reg = 0, neg_ratio = 0,
                 eps = 0.5, num_noise_factor = 64, loss_ratio = 0.0,
                 lr=0.001, is_neg_sa = False,
                 robust_test=False,
                 add_hidden_noise = False, add_weight_noise = False,
                 add_random_noise = False, add_adv_noise = False,
                 epochs=100, batch_size=128, T=10**3, verbose=False):
        # Parse the arguments and store them in the model
        self.session = sess

        self.num_factors = num_factors
        self.num_noise_factor = num_noise_factor
        self.ae_regs = ae_regs
        self.user_node_regs = user_node_reg

        self.eps = eps
        self.neg_ratio = neg_ratio
        self.topK = top_K

        self.loss_ratio = loss_ratio

        self.lr = lr
        self.is_neg_sa = is_neg_sa

        self.robust_test = robust_test
        self.hidden_noise = add_hidden_noise
        self.weight_noise = add_weight_noise
        self.random_noise = add_random_noise
        self.adv_noise = add_adv_noise

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self, original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape
        self.train_array = train_matrix.toarray()

        _, self.ranking_dict, self.test_dict = \
            mtl.negdict_mat(original_matrix, test_matrix, num_neg=199, mod='others', random_state=0)

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
            self.add_hidden_noise = tf.placeholder(dtype=tf.bool, shape=[], name='add_hidden_noise')
            self.add_weight_noise = tf.placeholder(dtype=tf.bool, shape=[], name='add_weight_noise')

            self.layer1_dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='layer1_dropout_rate')

            input = self.ratings

            # Encoder Variables
            layer1_w = tf.get_variable(name='encoder_weights',
                                       shape=[self.num_item, self.num_factors],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03))

            layer1_b = tf.get_variable(name='encoder_bias',
                                       shape=[self.num_factors],
                                       initializer=tf.zeros_initializer())

            # user_node = tf.get_variable(name='user_nodes',
            #                             shape=[self.num_factors],
            #                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03)
            #                             )

            w2_noise = tf.get_variable(name='w2_noise',
                                        shape=[self.num_factors, self.num_item],
                                        initializer=tf.zeros_initializer(),
                                        dtype=tf.float32,
                                        trainable=False)

            hidden_noise_tr = tf.get_variable(  name='hidden_noise_tr',
                                                shape=[self.batch_size, self.num_factors],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32,
                                                trainable=False)

            # Decoder Variables
            layer2_w = tf.get_variable(name='decoder_weights',
                                       shape=[self.num_factors, self.num_item],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03))
            layer2_b = tf.get_variable(name='decoder_bias',
                                       shape=[self.num_item],
                                       initializer=tf.zeros_initializer())


            # Original AE Model
            org_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b)
            org_encoder = tf.cond(self.istraining,
                                 lambda: tf.layers.dropout(org_encoder, rate=self.layer1_dropout_rate, name='layer1_dropout'),
                                 lambda: org_encoder)

            if self.robust_test:  # Robustness Testing on W2
                org_decoder = tf.identity(tf.matmul(org_encoder, layer2_w + w2_noise) + layer2_b)
                self.w2 = layer2_w
                self.w2_org = layer2_w - w2_noise
            else:
                org_decoder = tf.identity(tf.matmul(org_encoder, layer2_w) + layer2_b)

            mask = tf.cond(self.isnegsample,
                           lambda: tf.cast(self.output_mask, dtype=org_decoder.dtype),
                           lambda: tf.sign(self.ratings))

            self.org_output = tf.cond(self.istraining,
                                        lambda: tf.multiply(org_decoder, mask),
                                        lambda: org_decoder)

            org_base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.org_output))
            org_base_loss = org_base_loss / tf.cast(tf.shape(input)[0], dtype=org_base_loss.dtype)

            # org_reg_loss = self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
            #            self.ae_regs[2] * tf.nn.l2_loss(layer2_w2) + self.ae_regs[3] * tf.nn.l2_loss(layer2_b)
            # org_loss = org_base_loss + org_reg_loss

            # Noisy Model
            if not self.robust_test:
                noisy_encoder = tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b)
                noisy_encoder = tf.cond(self.istraining,
                                 lambda: noisy_encoder + hidden_noise_tr,
                                 lambda: noisy_encoder)

                # layer1 = tf.cond(self.istraining,
                #                 lambda : tf.sigmoid(tf.matmul(input, layer1_w + layer1_delta) + layer1_b),
                #                 lambda : tf.sigmoid(tf.matmul(input, layer1_w) + layer1_b))

                noisy_encoder = tf.cond(self.istraining,
                                        lambda: tf.layers.dropout(noisy_encoder, rate=self.layer1_dropout_rate, name='layer1_dropout'),
                                        lambda: noisy_encoder)

                self.w2 = layer2_w
                self.w2_org = tf.cond(self.add_weight_noise, lambda: layer2_w - w2_noise, lambda: layer2_w)

                noisy_decoder = tf.identity(tf.matmul(noisy_encoder, layer2_w) + layer2_b)

                # Output
                # Determine whether negative samples should be considered
                mask = tf.cond(self.isnegsample,
                               lambda: tf.cast(self.output_mask, dtype=noisy_decoder.dtype),
                               lambda: tf.sign(self.ratings))

                self.noisy_output = tf.cond(self.istraining,
                                      lambda: tf.multiply(noisy_decoder, mask),
                                      lambda: noisy_decoder)

                # self.pred_y = tf.sigmoid(self.output)
                # self.pred_y = self.output

                # Noisy Model Loss
                # base_loss = tf.nn.l2_loss(input - self.pred_y)
                noisy_base_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=self.noisy_output))
                noisy_base_loss = noisy_base_loss / tf.cast(tf.shape(input)[0], dtype=noisy_base_loss.dtype)  # Average over the batches

            ################## Mix the outputs
            # self.mixed_output = (1-self.loss_ratio) * self.org_output + self.loss_ratio * self.output
            self.pred_y = tf.sigmoid(self.org_output)
            # self.pred_y = self.mixed_output

            # Mix the losses
            # base_loss = tf.cond(tf.logical_or(self.add_concat_noise, self.add_weight_noise),
            #                     lambda : org_base_loss + self.loss_ratio * base_loss,
            #                     lambda : base_loss)
            if not self.robust_test:
                base_loss = org_base_loss + self.loss_ratio * noisy_base_loss
            else:
                base_loss = org_base_loss

            reg_loss = self.ae_regs[0] * tf.nn.l2_loss(layer1_w) + self.ae_regs[1] * tf.nn.l2_loss(layer1_b) + \
                       self.ae_regs[2] * tf.nn.l2_loss(layer2_w) + self.ae_regs[3] * tf.nn.l2_loss(layer2_b)

            self.loss = base_loss + reg_loss

            # Optimizer
            self.opt = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)

            # Add Noise (Random or Adversial)
            if self.robust_test:  # Robust Testing in Evaluation
                if self.random_noise:
                    random_noise = tf.random_normal(shape=tf.shape(layer2_w), mean=tf.reduce_mean(layer2_w), stddev=0.01)
                    self.update_delta = w2_noise.assign(self.eps * random_noise / tf.norm(random_noise))
                if self.adv_noise:
                    self.grad_delta = tf.gradients(ys=org_base_loss, xs=w2_noise)[0]
                    self.grad_delta_dense = tf.stop_gradient(self.grad_delta)
                    self.update_delta = w2_noise.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))
            else:
                # Gradients Computation of the Noise in Training
                self.grad_delta = tf.gradients(ys=base_loss, xs=hidden_noise_tr)[0]
                # convert the IndexedSlice Data to Dense Tensor
                self.grad_delta_dense = tf.stop_gradient(self.grad_delta)
                # self.grad_shape = tf.shape(self.grad_delta_dense)
                # normalization: new_grad = (grad / |grad|) * eps
                self.update_delta = hidden_noise_tr.assign(self.eps * self.grad_delta_dense / tf.norm(self.grad_delta_dense))

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

            if not self.robust_test:
            # Only calculate the loss of the observed items
                grad, delta = self.session.run([self.grad_delta_dense, self.update_delta],
                                                feed_dict={ self.ratings: batch_ratings, self.output_mask: batch_neg_masks,
                                                            self.istraining: True, self.isnegsample: self.is_neg_sa,
                                                            self.add_hidden_noise: self.hidden_noise,
                                                            self.add_weight_noise: self.weight_noise,
                                                            self.layer1_dropout_rate: 0})

                if self.verbose:
                    if n_batches % self.skip_step == 0:
                        print("Training Epoch {0} Batch {1}: [Grad] = {2}, [Noise] = {3}"
                              .format(epoch, n_batches, np.linalg.norm(grad), delta[0,0]))
                        # print("Training Epoch {0} Batch {1}: [W2] = {2}"
                        #       .format(epoch, n_batches, w2[0,0]))

            # Train the model
            _, l = \
                self.session.run([self.opt, self.loss],
                                 feed_dict={self.ratings: batch_ratings, self.output_mask: batch_neg_masks,
                                            self.istraining: True, self.isnegsample: self.is_neg_sa,
                                            self.add_hidden_noise: self.hidden_noise,
                                            self.add_weight_noise: self.weight_noise,
                                            self.layer1_dropout_rate: 0.05})

            n_batches += 1
            total_loss += l

            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("Training Epoch {0} Batch {1}: [Loss] = {2}"
                          .format(epoch, n_batches, total_loss / n_batches))
                    # print("Training Epoch {0} Batch {1}: [Grad] = {2}"
                    #       .format(epoch, n_batches, np.linalg.norm(grad)))
                    # print(delta[:5])

        if self.verbose:
            print("Training Epoch {0}: [Loss] {1}".format(epoch, total_loss / n_batches))

    def eval_one_epoch(self, epoch):
        if self.robust_test:
            delta = self.session.run(self.update_delta,
                                     feed_dict={self.ratings: self.train_array,
                                                self.output_mask: self.negative_output_mask,
                                                self.istraining: False, self.isnegsample: False,
                                                self.add_hidden_noise: self.hidden_noise,
                                                self.add_weight_noise: self.weight_noise,
                                                self.layer1_dropout_rate: 0})

            layer2_w, layer2_w_org = self.session.run([self.w2, self.w2_org],
                                                      feed_dict={self.ratings: self.train_array,
                                                                 self.output_mask: self.negative_output_mask,
                                                                 self.istraining: False, self.isnegsample: False,
                                                                 self.add_hidden_noise: self.hidden_noise,
                                                                 self.add_weight_noise: self.weight_noise,
                                                                 self.layer1_dropout_rate: 0})

            print("Evaluation Epoch {0}: [Delta] = {1} [W2]={2} [W2_ORG]={3}"
                  .format(epoch, delta[10, 0], layer2_w[10, 0], layer2_w_org[10, 0]))

        # Input the uncorrupted training data
        pred_y = self.session.run(self.pred_y,
                                  feed_dict={self.ratings: self.train_array, self.output_mask: self.negative_output_mask,
                                             self.istraining: False, self.isnegsample: self.is_neg_sa,
                                             self.add_hidden_noise: self.hidden_noise,
                                             self.add_weight_noise: self.weight_noise,
                                             self.layer1_dropout_rate: 0})
        pred_y = pred_y.clip(min=0,max=1)
        # n_batches, total_prec, total_ap = 0, 0, 0
        n_batches, total_hr, total_ndcg = 0, 0, 0
        # Loop for each user (generate the ranking lists for different users)
        for u in self.ranking_dict:
            iid = self.ranking_dict[u]  # The ranking item ids for user u
            rk = pred_y[u, np.array(iid)]  # The predicted item values for user u

            hr, ndcg = evl.rankingMetrics(rk, iid, self.topK, self.test_dict[u], mod='hr')

            n_batches += 1
            total_hr += hr
            total_ndcg += ndcg
            # total_prec += precision
            # total_ap += ap

        avg_hr, avg_ndcg = total_hr / n_batches, total_ndcg / n_batches
        print('='*50)
        print("Epoch {0}: [HR@{1}] {2}".format(epoch, self.topK, avg_hr))
        print("Epoch {0}: [nDCG@{1}] {2}".format(epoch, self.topK, avg_ndcg))
        return avg_hr, avg_ndcg

        # avg_prec, avg_ap = total_prec / n_batches, total_ap / n_batches
        # print("Epoch {0}: [Precision@{1}] {2}".format(epoch, self.topK, avg_prec))
        # print("Epoch {0}: [MAP@{1}] {2}".format(epoch, self.topK, avg_ap))
        # return avg_prec, avg_ap

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
def parseArgs():
    parser = argparse.ArgumentParser(description="GAE_AE")

    parser.add_argument('--nfactors', type=int, default=64,
                        help='Embedding size.')

    parser.add_argument('--ae_regs', nargs='?', default='[0.5,0.5,0.1,0.1]', type=str,
                        help="Network variable regularization constants.")
    # parser.add_argument('--user_regs', nargs='?', default=1, type=float,
    #                     help="User node regularization constants.")

    # parser.add_argument('--noise_ratio', default=0.6, type=float)
    parser.add_argument('--out_neg_ratio', default=1.0, type=float)

    parser.add_argument('--epsilon', default=50, type=float)

    parser.add_argument('--topk', type=int, default=5,
                        help='The K value of the Top-K ranking list.')

    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, lr,\
    topK, num_factors, neg_ratio, epsilon,\
    ae_regs = \
    args.epochs, args.batch_size, args.lr,\
    args.topk, args.nfactors, args.out_neg_ratio, args.epsilon,\
    args.ae_regs

    ae_regs = list(np.float32(eval(ae_regs)))

    # original_matrix \
    #     = mtl.load_original_matrix(datafile='Data/ml-1m/ratings.dat', header=['uid', 'iid', 'ratings', 'time'], sep='::')

    # train_matrix, test_matrix \
    #     = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.2, random_state=10)

    # train_matrix, test_matrix \
    #     = mtl.matrix_split(original_matrix, opt='ranking', mode='mat', n_item_per_user=1, random_state=10)

    # original_matrix = mtl.matrix_to_binary(original_matrix, 0)
    # train_matrix = mtl.matrix_to_binary(train_matrix, 0)
    # test_matrix = mtl.matrix_to_binary(test_matrix, 0)

    # gtl.matrix_to_mat('Data/ML1M_Rank_200_1_Data.mat', opt='coo', original=original_matrix, train=train_matrix, test=test_matrix)

    data = gtl.load_mat_as_matrix('Data/ML1M_Rank_200_1_Data.mat', opt='coo')
    original_matrix, train_matrix, test_matrix = data['original'], data['train'], data['test']

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:
        model = GAN_AE(sess,
                        top_K=topK, neg_ratio=neg_ratio,
                        num_factors=num_factors, ae_regs=ae_regs, eps=40,
                        lr=lr, num_noise_factor = 64, loss_ratio = 1,
                        is_neg_sa= False,
                        robust_test=False,
                        add_hidden_noise=True, add_weight_noise=False,
                        add_random_noise=False, add_adv_noise=True,
                        epochs=num_epochs, batch_size=batch_size, T=50, verbose=True)

        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
        model.build_model()
        # model.evaluate('SavedModel/GAN_AE/Hidden_Noise/Hidden_Noise_EPS_10_ML1M_CE_MIX.ckpt')

        # model.evaluate('SavedModel/GAN_AE/Weight_Noise/Weight_Noise_AE_ML1M_CE_200_1.ckpt')
        # model.evaluate('SavedModel/GAN_AE/Pure_AE/Pure_AE_ML1M_CE_200_1.ckpt')
        model.train(restore=True, save=False,
                    # restore_datafile='SavedModel/GAN_AE/Weight_Noise/Weight_Noise_AE_ML1M_CE_200_1.ckpt')
                    # restore_datafile='SavedModel/GAN_AE/Pure_AE/Pure_AE_ML1M_CE_200_1.ckpt',
                    restore_datafile = 'SavedModel/GAN_AE/Hidden_Noise/Pure_AE_ML1M_CE_MIX.ckpt',
                    # save_datafile= 'SavedModel/GAN_AE/Hidden_Noise/Pure_AE_ML1M_CE_MIX.ckpt')
                    save_datafile = 'SavedModel/GAN_AE/Hidden_Noise/Hidden_Noise_EPS_10_ML1M_CE_MIX.ckpt')