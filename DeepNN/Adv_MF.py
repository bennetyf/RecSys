import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

# os.environ["CUDA_VISIBLE_DEVICES"]="5"

import tensorflow as tf
import numpy as np
import inspect
import random

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The AMF Model ##########################################################

# Define the class for Minimax AE
class AMF(object):
    def __init__(self,
                 sess,
                 top_K,
                 num_factors = 32,
                 reg = 0.0,
                 reg_adv = 0.0,
                 noise_type = 'random',
                 is_adv = False,
                 eps = 0.5,
                 lr=0.001,
                 is_prec=False,
                 save_T = 50,
                 epochs=100,
                 batch_size=128,
                 T=10**3,
                 verbose=False):

        # Parse the arguments and store them in the model
        self.session = sess

        self.num_factors = num_factors
        self.reg = reg
        self.reg_adv = reg_adv
        self.noise_type = noise_type
        self.is_adv = is_adv

        self.eps = eps
        self.topK = top_K

        self.lr = lr
        self.is_prec = is_prec

        self.save_T = save_T

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

        self.metric1, self.metric2 = [], []

        gtl.print_paras(inspect.currentframe())


    def prepare_data(self, original_matrix, train_matrix, test_matrix):
        self.num_user, self.num_item = original_matrix.shape
        self.train_uid, self.train_iid, _ = mtl.matrix_to_list(train_matrix)

        if self.is_prec:
            self.neg_dict, self.ranking_dict, self.test_dict = \
                mtl.negdict_mat(original_matrix, test_matrix,  mod='precision', random_state=20)
        else:
            self.neg_dict, self.ranking_dict, self.test_dict = \
                mtl.negdict_mat(original_matrix, test_matrix, num_neg=199, mod='others', random_state=0)

        self.num_training = len(self.train_uid)
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope('Model',reuse=tf.AUTO_REUSE):
            with tf.name_scope('Input'):
                self.uid = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
                self.pos_iid = tf.placeholder(dtype=tf.int32, shape=[None], name='pos_item_id')
                self.neg_iid = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_item_id')

            with tf.name_scope('Embedding'):
                self.embedding_P = tf.get_variable(name='embedding_P',
                                                   shape=[self.num_user, self.num_factors],
                                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                                   dtype=tf.float32)  # (users, embedding_size)

                self.embedding_Q = tf.get_variable(name='embedding_Q',
                                                   shape=[self.num_item, self.num_factors],
                                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                                   dtype=tf.float32)  # (items, embedding_size)

                self.delta_P = tf.get_variable(name='delta_P',
                                                shape=[self.num_user, self.num_factors],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32, trainable=False)  # (users, embedding_size)

                self.delta_Q = tf.get_variable(name='delta_Q',
                                                shape=[self.num_item, self.num_factors],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32, trainable=False)  # (items, embedding_size)

            with tf.name_scope('Prediction'):
                self.embedding_p = tf.nn.embedding_lookup(self.embedding_P, self.uid) #[batch, num_factors]
                self.embedding_q_pos = tf.nn.embedding_lookup(self.embedding_Q, self.pos_iid)
                self.embedding_q_neg = tf.nn.embedding_lookup(self.embedding_Q, self.neg_iid)

                self.P_plus_delta = self.embedding_p + tf.nn.embedding_lookup(self.delta_P, self.uid)
                self.Q_plus_delta_pos = self.embedding_q_pos + tf.nn.embedding_lookup(self.delta_Q, self.pos_iid)
                self.Q_plus_delta_neg = self.embedding_q_neg + tf.nn.embedding_lookup(self.delta_Q, self.neg_iid)

                self.pred_y_pos = tf.einsum('ij,ij->i',self.embedding_p,self.embedding_q_pos)
                self.pred_y_neg = tf.einsum('ij,ij->i',self.embedding_p,self.embedding_q_neg)

                self.pred_y_pos_adv = tf.einsum('ij,ij->i',self.P_plus_delta,self.Q_plus_delta_pos)
                self.pred_y_neg_adv = tf.einsum('ij,ij->i',self.P_plus_delta,self.Q_plus_delta_neg)

            with tf.name_scope('Loss'):
                if self.is_adv:
                    # self.base_loss_adv = tf.reduce_sum(tf.nn.softplus(self.pred_y_neg_adv - self.pred_y_pos_adv))
                    self.base_loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pred_y_pos - self.pred_y_neg)))
                    self.reg_loss = self.reg * (tf.reduce_sum(tf.square(self.embedding_P)) +
                                                tf.reduce_sum(tf.square(self.embedding_Q)))+\
                                    self.reg_adv * (- tf.reduce_sum(tf.log(tf.sigmoid(self.pred_y_pos_adv - self.pred_y_neg_adv))))
                else:
                    self.base_loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pred_y_pos - self.pred_y_neg)))
                    self.reg_loss = self.reg * (tf.reduce_sum(tf.square(self.embedding_P)) +
                                                tf.reduce_sum(tf.square(self.embedding_Q)))

                self.loss = self.base_loss + self.reg_loss

            with tf.name_scope('Noise_Adding'):
                if self.noise_type == 'random':
                    # generation
                    self.adv_P = tf.truncated_normal(shape=[self.num_user, self.num_factors], mean=0.0, stddev=0.01)
                    self.adv_Q = tf.truncated_normal(shape=[self.num_item, self.num_factors], mean=0.0, stddev=0.01)

                    # normalization and multiply epsilon
                    self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.adv_P, 1) * self.eps)
                    self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.adv_Q, 1) * self.eps)

                elif self.noise_type == 'grad':
                    self.grad_P, self.grad_Q = tf.gradients(self.base_loss, [self.embedding_P, self.embedding_Q])

                    # convert the IndexedSlice Data to Dense Tensor
                    self.grad_P_dense = tf.stop_gradient(self.grad_P)
                    self.grad_Q_dense = tf.stop_gradient(self.grad_Q)

                    # normalization: new_grad = (grad / |grad|) * eps
                    self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.grad_P_dense, 1) * self.eps)
                    self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.grad_Q_dense, 1) * self.eps)

            with tf.name_scope('Optimizer'):
                self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)

            print('Model Building Completed.')

    def train_one_epoch(self, epoch):
        uid, iid = gtl.shuffle_list(self.train_uid, self.train_iid)

        # start_time = time.time()
        iid_neg = [random.choice(self.neg_dict[u]) for u in uid]
        # iid_neg = [np.random.choice(self.neg_dict[u]).item() for u in uid]
        # iid_neg = [self.neg_dict[u][np.random.randint(len(self.neg_dict[u]))] for u in uid]
        # print("Time={0}".format(time.time()-start_time))

        n_batches, total_loss = 0, 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                # break
                batch_uids = uid[i * self.batch_size:]
                batch_iids_pos = iid[i * self.batch_size:]
                batch_iids_neg = iid_neg[i * self.batch_size:]
            else:
                batch_uids = uid[i * self.batch_size: (i + 1) * self.batch_size]
                batch_iids_pos = iid[i * self.batch_size: (i + 1) * self.batch_size]
                batch_iids_neg = iid_neg[i * self.batch_size:(i + 1) * self.batch_size]
                # Randomly select one negative item j for each user
                # batch_iids_neg = [np.random.choice(self.neg_dict[u], 1).item() for u in batch_uids]

            feed_dict = {
                            self.uid:       batch_uids,
                            self.pos_iid:   batch_iids_pos,
                            self.neg_iid:   batch_iids_neg,
                        }

            if self.is_adv:
                self.session.run([self.update_P, self.update_Q], feed_dict)
            _, l = self.session.run([self.optimizer, self.loss], feed_dict)

            n_batches += 1
            total_loss += l

            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("[All] Training Epoch {0} Batch {1}: [Loss] = {2}".format(epoch, n_batches, total_loss / n_batches))

        if self.verbose:
            print("[Epoch Average] Training Epoch {0}: [Loss] {1}".format(epoch, total_loss / n_batches))

    def eval_one_epoch(self, epoch):
        n_batches = 0
        if self.is_prec:
            total_prec, total_recall = np.zeros(len(self.topK)), np.zeros(len(self.topK))
        else:
            total_hr, total_ndcg = np.zeros(len(self.topK)), np.zeros(len(self.topK))

        # if self.robust_test:
        #     print('[Eps={0}] {1} Noise Level [Robust Test]'.format(self.eps, self.noise_type))
        #     self.session.run([self.update_P, self.update_Q], feed_dict={self.uid: uid, self.pos_iid: iid})

        for u in self.ranking_dict:

            if len(self.test_dict[u]) == 0:
                continue

            iid = self.ranking_dict[u]
            uid = [u] * len(iid)
            n_batches += 1

            if self.is_adv:
                rk = self.session.run(self.pred_y_pos_adv, feed_dict={self.uid: uid, self.pos_iid: iid})
            else:
                rk = self.session.run(self.pred_y_pos, feed_dict={self.uid: uid, self.pos_iid: iid})

            if self.is_prec:
                precision, recall,_,_,_ = evl.rankingMetrics(rk, iid, self.topK, self.test_dict[u], mod='precision', is_map=False)
                total_prec += precision
                total_recall += recall
            else:
                hr, ndcg = evl.rankingMetrics(rk, iid, self.topK, self.test_dict[u], mod='hr')
                total_hr += hr
                total_ndcg += ndcg

        if self.is_prec:
            avg_prec, avg_recall = total_prec / n_batches, total_recall / n_batches
            for i in range(len(self.topK)):
                print('-' * 55)
                print("Epoch {0}: [Precision@{1}] {2}".format(epoch, self.topK[i], avg_prec[i]))
                print("Epoch {0}: [Recall@{1}] {2}".format(epoch, self.topK[i], avg_recall[i]))
            print('=' * 55)
            return avg_prec[0], avg_recall[0]
        else:
            avg_hr, avg_ndcg = total_hr / n_batches, total_ndcg / n_batches

            for i in range(len(self.topK)):
                print('-' * 55)
                print("Epoch {0}: [HR@{1}] {2}".format(epoch, self.topK[i], avg_hr[i]))
                print("Epoch {0}: [nDCG@{1}] {2}".format(epoch, self.topK[i], avg_ndcg[i]))
            print('=' * 55)
            return avg_hr[0], avg_ndcg[0]

    # Final Training of the model
    def train(self, restore=False, save=False, restore_datafile=None, save_datafile=None):

        if restore: # Restore the model from checkpoint
            self.restore_model(restore_datafile, verbose=True)
        else:
            self.session.run(tf.global_variables_initializer())

        if not save: # Do not save the model
            previous_metric1, previous_metric2 = self.eval_one_epoch(-1)
            self.metric1.append(previous_metric1)
            self.metric2.append(previous_metric2)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                metric1, metric2 = self.eval_one_epoch(i)
                self.metric1.append(metric1)
                self.metric2.append(metric2)

        else: # Save the model while training
            previous_metric1,previous_metric2 = self.eval_one_epoch(-1)
            self.metric1.append(previous_metric1)
            self.metric2.append(previous_metric2)
            # previous_metric1 = 0
            for i in range(self.epochs):
                self.train_one_epoch(i)
                metric1,metric2 = self.eval_one_epoch(i)
                self.metric1.append(metric1)
                self.metric2.append(metric2)
                # if i % self.save_T == 0:
                #     self.save_model(save_datafile, verbose=True)
                # if metric1 > previous_metric1:
                #     previous_metric1 = metric1
                if i == 100:
                    self.save_model(save_datafile, verbose=True)

    # Save the model
    def save_model(self, datafile, verbose=False):
        # saver = tf.train.Saver({'embedding_P':self.embedding_P,
        #                         'embedding_Q':self.embedding_Q})
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
# if __name__ == "__main__":

    # original_matrix \
    #         = mtl.load_original_matrix(datafile='Data/ml-100k/u.data', header=['uid', 'iid', 'ratings','time'], sep='\t')
    # original_matrix = mtl.matrix_to_binary(original_matrix, 0)
    # train_matrix, test_matrix \
    #     = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.2, random_state=10)

    # gtl.matrix_to_mat('Data/ML1M_Precision_Data.mat', opt='coo', original=original_matrix, train=train_matrix, test=test_matrix)

    # original_matrix = gtl.load_mat_as_matrix('Data/Ciao.mat', opt='coo')['rating']
    # train_matrix, test_matrix \
    #     = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.2, random_state=10)

    # data = gtl.load_mat_as_matrix('Data/ML1M_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/ML10M_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/Ciao_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/Douban_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/Epinions_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/GoodBooks_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/YahooMovie_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/YahooMusic_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/FilmTrust_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/Flixster_Rank_200_1_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/BX_Rank_200_1_Data.mat', opt='coo')

    # data = gtl.load_mat_as_matrix('Data/ML1M_Precision_Data.mat', opt='coo')
    # data = gtl.load_mat_as_matrix('Data/Ciao_Precision_Data.mat', opt='coo')

    # path_list = ['Data/ML1M_Rank_HR.mat',
    #              'Data/ML10M_Rank_HR.mat',
    #              'Data/Ciao_Rank_HR.mat',
    #              'Data/Douban_Rank_HR.mat',
    #              'Data/Epinions_Rank_HR.mat',
    #              'Data/GoodBooks_Rank_HR.mat',
    #              'Data/YahooMovie_Rank_HR.mat',
    #              'Data/YahooMusic_Rank_HR.mat',
    #              'Data/FilmTrust_Rank_HR.mat',
    #              'Data/Flixster_Rank_HR.mat',
    #              'Data/BX_Rank_HR.mat']
    # dataset_list = ['ML1M','ML10M','Ciao','Douban','Epinions','GoodBooks','YahooMovie','YahooMusic','FilmTrust','Flixster','BX']
    #
    # path = path_list[0]
    # dataset = dataset_list[0]
    #
    # print('Loading Data From {0}'.format(path))
    #
    # data = gtl.load_mat_as_matrix(path, opt='coo')
    # original_matrix, train_matrix, test_matrix = data['original'], data['train'], data['test']
    #
    # print(original_matrix.shape)
    # print("Number of Nonzeros in Training: {0}".format(train_matrix.nnz))
    #
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
    #                                       intra_op_parallelism_threads=8,
    #                                       inter_op_parallelism_threads=8,
    #                                       gpu_options=gpu_options)) as sess:
    #     model = AMF(sess,
    #                 top_K=[5,10],
    #                 num_factors=50,
    #                 reg=0.002,
    #
    #                 reg_adv=10,
    #                 noise_type='grad',
    #                 is_adv=True,
    #                 eps=1,
    #                 lr=0.001,
    #
    #                 is_prec=False,
    #                 epochs=2000,
    #                 batch_size=1024,
    #                 T=500,
    #                 verbose=True)
    #
    #     model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
    #
    #     model.build_model()
    #
    #     # for i in range(16):
    #     #     model.eps = i
    #     # model.evaluate('SavedModel/AMF/ML-1M/BPR_HR_CE.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/Ciao/Weight_Noise/Pure_AE_Ciao_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/Douban/Weight_Noise/Pure_AE_Douban_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/Epinions/Weight_Noise/Pure_AE_Epinions_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/YahooMovie/Weight_Noise/Pure_AE_YahooMovie_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/YahooMusic/Weight_Noise/Pure_AE_YahooMusic_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/FilmTrust/Weight_Noise/Pure_AE_FilmTrust_SQ_MIX_ALL.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/ML-10M/Weight_Noise/Pure_AE_ML10M_SQ_MIX_ALL_0_01.ckpt')
    #     # model.evaluate('SavedModel/GAN_AE/BookCrossing/Weight_Noise/Pure_AE_BX_SQ_MIX_ALL_1_0.ckpt')
    #
    #     if model.is_adv:
    #         ckpt_save_path = "Pretrain/%s/APR/embed_%d/%s/" % (dataset, model.num_factors, '20180713') + 'checkpoint.ckpt'
    #         ckpt_restore_path = "Pretrain/%s/BPR/embed_%d/%s/" % (dataset, model.num_factors, '20180713') + 'checkpoint.ckpt'
    #     else:
    #         ckpt_save_path = "Pretrain/%s/BPR/embed_%d/%s/" % (dataset, model.num_factors, '20180713') + 'checkpoint.ckpt'
    #         ckpt_restore_path = None
    #
    #     if not os.path.exists(ckpt_save_path):
    #         os.makedirs(ckpt_save_path)
    #     if ckpt_restore_path and not os.path.exists(ckpt_restore_path):
    #         os.makedirs(ckpt_restore_path)
    #
    #     model.train(restore=True, save=True,
    #                 save_datafile= ckpt_save_path, restore_datafile=ckpt_restore_path)
    #
    #                 # save_datafile = 'SavedModel/AMF/ML-1M/BPR_HR_CE.ckpt')
    #                 # save_datafile = 'SavedModel/AMF/Ciao/BPR_HR_CE.ckpt')
    #
    #     # model.train(restore=False, save=True,
    #     #             save_datafile = 'SavedModel/AMF/ML-1M/BPR_HR_CE.ckpt')
    #
    #                 # save_datafile='SavedModel/AMF/Ciao/AMF_ORG.ckpt')
    #                 #  save_datafile= 'SavedModel/AMF/Ciao/AMF_Precision.ckpt')
    #
    #                 # save_datafile = 'SavedModel/AMF/Ciao/BPR_Precision_ORG.ckpt')
    #                 # save_datafile = 'SavedModel/AMF/FilmTrust/BPR_HR_CE_ORG.ckpt')