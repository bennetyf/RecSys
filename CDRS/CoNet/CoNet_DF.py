import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import argparse

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.DfUtils as dfl

import time
############################################### The CoNet Model ########################################################

# Define the class for CoNet
class CoNetRec(object):
    def __init__(self, sess, num_neg, top_K = 10, num_ranking_list = 100, lr=0.001, regs=[0.001,0.001,0.001],
                 alpha = 0.5, epochs=100, batch_size=128, T=10**3, verbose=False):
        '''Constructor'''
        # Parse the arguments and store them in the model
        self.session = sess
        self.regs_user, self.dom1_regs_item, self.dom2_regs_item = regs
        self.topk = top_K
        self.lr = lr
        self.num_neg = num_neg
        self.num_ranking_list = num_ranking_list
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self, original_df1, train_df1, test_df1, test_dict1,
                     original_df2, train_df2, test_df2, test_dict2):

        # Meta Info
        self.num_user1, self.num_item1 = train_df1['uid'].unique().shape[0],train_df1['iid'].unique().shape[0]
        self.num_user2, self.num_item2 = train_df2['uid'].unique().shape[0],train_df2['iid'].unique().shape[0]

        # Upsampling
        train1, train2 = dfl.data_upsample(train_df1, train_df2)

        self.ui_dict1 = dfl.get_user_item_dict(original_df1)
        self.ui_dict2 = dfl.get_user_item_dict(original_df2)

        # Negative Sampling if Required
        if self.num_neg > 0:
            print("Enter NegSa")
            start_time = time.time()
            train1 = dfl.negative_sample_df(original_df1, self.ui_dict1, train1, test_df1, num_neg=1, neg_val=0, opt='train')
            train2 = dfl.negative_sample_df(original_df2, self.ui_dict2, train2, test_df2, num_neg=1, neg_val=0, opt='train')
            print("Leaving NegSa")
            print("Negative Sampling Time: {0}".format(time.time()-start_time))

        self.train_uid1, self.train_iid1, self.train_labels1 = dfl.df_to_list(train1)
        self.train_uid2, self.train_iid2, self.train_labels2 = dfl.df_to_list(train2)

        length1, length2 = len(self.train_labels1), len(self.train_labels2)
        self.num_training = max(length1, length2)
        self.num_batch = int(self.num_training / self.batch_size)

        # Generating the testing dictionary
        if self.num_ranking_list > 1:
            self.test_dict1 = \
                dfl.negative_sample_df(original_df1, self.ui_dict1, train1, test_df1, num_neg=self.num_ranking_list-1, opt='test')
            self.test_dict2 = \
                dfl.negative_sample_df(original_df2, self.ui_dict2, train2, test_df2, num_neg=self.num_ranking_list-1, opt='test')
        else:
            self.test_dict1, self.test_dict2 = test_dict1, test_dict2
        print("Data Preparation Completed.")

    def model(self, num_factors = 16):
        '''
        The Collaborative Cross Network Model
        '''
        with tf.variable_scope('Model'):
            self.dom1_uid = tf.placeholder(dtype=tf.int32, shape=[None], name='dom1_user_id')
            self.dom2_uid = tf.placeholder(dtype=tf.int32, shape=[None], name='dom2_user_id')
            self.dom1_iid = tf.placeholder(dtype=tf.int32, shape=[None], name='dom1_item_id')
            self.dom2_iid = tf.placeholder(dtype=tf.int32, shape=[None], name='dom2_item_id')
            self.dom1_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='dom1_labels')
            self.dom2_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='dom2_labels')

            # One-hot input
            embeddings_user = tf.get_variable(name='user_embed',
                                              shape=[self.num_user1, num_factors],
                                              initializer=tf.random_uniform_initializer(-1,1))

            embeddings_item1 = tf.get_variable(name='item1_embed',
                                               shape=[self.num_item1, num_factors],
                                               initializer=tf.random_uniform_initializer(-1,1))

            embeddings_item2 = tf.get_variable(name='item2_embed',
                                               shape=[self.num_item2, num_factors],
                                               initializer=tf.random_uniform_initializer(-1,1))

            # The Embedding Layer
            embed_layer_user1 = tf.nn.embedding_lookup(embeddings_user, self.dom1_uid)
            embed_layer_user2 = tf.nn.embedding_lookup(embeddings_user, self.dom2_uid)
            embed_layer_item1 = tf.nn.embedding_lookup(embeddings_item1, self.dom1_iid)
            embed_layer_item2 = tf.nn.embedding_lookup(embeddings_item2, self.dom2_iid)

            # First Hidden Layer
            mlp_vector1_tmp = tf.concat([embed_layer_user1,embed_layer_item1], axis=1)
            mlp_vector2_tmp = tf.concat([embed_layer_user2,embed_layer_item2], axis=1)


            # mlp_vector1 = tf.layers.dense(mlp_vector1_tmp, units=64, activation=tf.nn.relu)
            # mlp_vector2 = tf.layers.dense(mlp_vector2_tmp, units=64, activation=tf.nn.relu)
            # mlp_vector1 = tf.layers.dropout(mlp_vector1, rate=0.5)
            # mlp_vector2 = tf.layers.dropout(mlp_vector2, rate=0.5)
            #
            # # Cross Stitching
            # W11 = tf.get_variable('W11', shape=[64,64], initializer=tf.random_normal_initializer(0.0,0.01))
            # W12 = tf.get_variable('W12', shape=[64,64], initializer=tf.random_normal_initializer(0.0,0.01))
            # H1 = tf.get_variable('H1', shape=[64,64], initializer=tf.random_normal_initializer(0.0,0.01))
            #
            # mlp_vector1_tmp = tf.nn.relu(tf.matmul(mlp_vector1, W11)+tf.matmul(mlp_vector2,H1))
            # mlp_vector2_tmp = tf.nn.relu(tf.matmul(mlp_vector2, W12)+tf.matmul(mlp_vector1,H1))


            # Second Hidden Layer
            # mlp_vector1 = tf.layers.dense(mlp_vector1_tmp, units=32, activation=tf.nn.relu)
            # mlp_vector2 = tf.layers.dense(mlp_vector2_tmp, units=32, activation=tf.nn.relu)
            # mlp_vector1 = tf.layers.dropout(mlp_vector1,rate=0.5)
            # mlp_vector2 = tf.layers.dropout(mlp_vector2, rate=0.5)
            #
            # W21 = tf.get_variable('W21', shape=[32,32], initializer=tf.random_uniform_initializer(-1,1))
            # W22 = tf.get_variable('W22', shape=[32,32], initializer=tf.random_uniform_initializer(-1,1))
            # H2 = tf.get_variable('H2', shape=[32,32], initializer=tf.random_uniform_initializer(-1,1))
            #
            # mlp_vector1_tmp = tf.nn.relu(tf.matmul(mlp_vector1,W21) + tf.matmul(mlp_vector2,H2))
            # mlp_vector2_tmp = tf.nn.relu(tf.matmul(mlp_vector2,W22) + tf.matmul(mlp_vector1,H2))

            # Third Hidden Layer
            mlp_vector1 = tf.layers.dense(mlp_vector1_tmp, units=16, activation=tf.nn.relu)
            mlp_vector2 = tf.layers.dense(mlp_vector2_tmp, units=16, activation=tf.nn.relu)
            mlp_vector1 = tf.layers.dropout(mlp_vector1, rate=0.5)
            mlp_vector2 = tf.layers.dropout(mlp_vector2, rate=0.5)

            W31 = tf.get_variable('W31', shape=[16,16], initializer=tf.random_uniform_initializer(-1,1))
            W32 = tf.get_variable('W32', shape=[16,16], initializer=tf.random_uniform_initializer(-1,1))
            H3 = tf.get_variable('H3', shape=[16,16], initializer=tf.random_uniform_initializer(-1,1))

            mlp_vector1_tmp = tf.nn.relu(tf.matmul(mlp_vector1,W31) + tf.matmul(mlp_vector2,H3))
            mlp_vector2_tmp = tf.nn.relu(tf.matmul(mlp_vector2,W32) + tf.matmul(mlp_vector1,H3))

            # Fourth Layer
            mlp_vector1 = tf.layers.dense(mlp_vector1_tmp, units=8, activation=tf.nn.relu)
            mlp_vector2 = tf.layers.dense(mlp_vector2_tmp, units=8, activation=tf.nn.relu)
            mlp_vector1 = tf.layers.dropout(mlp_vector1, rate=0.5)
            mlp_vector2 = tf.layers.dropout(mlp_vector2, rate=0.5)

            # W41 = tf.get_variable('W41', shape=[8, 8], initializer=tf.random_uniform_initializer(-1,1))
            # W42 = tf.get_variable('W42', shape=[8, 8], initializer=tf.random_uniform_initializer(-1,1))
            # H4 = tf.get_variable('H4', shape=[8, 8], initializer=tf.random_uniform_initializer(-1,1))
            # mlp_vector1_4 = tf.nn.relu(tf.matmul(mlp_vector1_3, W41) + tf.matmul(mlp_vector2_3, H4))
            # mlp_vector2_4 = tf.nn.relu(tf.matmul(mlp_vector2_3, W42) + tf.matmul(mlp_vector1_3, H4))

            # Output
            mlp_vector1_out = tf.layers.dense(mlp_vector1, units=1, activation=tf.identity)
            mlp_vector2_out = tf.layers.dense(mlp_vector2, units=1, activation=tf.identity)

            self.logits1 = tf.reshape(mlp_vector1_out,shape=[-1])
            self.logits2 = tf.reshape(mlp_vector2_out,shape=[-1])

            self.scores1 = tf.sigmoid(self.logits1)
            self.scores2 = tf.sigmoid(self.logits2)

    def regularizer(self):
        with tf.variable_scope('Model',reuse=True):
            if self.regs_user:
                for name in ['reg1', 'reg2']:
                    tf.add_to_collection(name,
                                        tf.contrib.layers.apply_regularization(
                                        tf.contrib.layers.l2_regularizer(scale=self.regs_user),
                                        [tf.get_variable('user_embed')]))
            if self.dom1_regs_item:
                tf.add_to_collection('reg1',
                                        tf.contrib.layers.apply_regularization(
                                        tf.contrib.layers.l2_regularizer(scale=self.dom1_regs_item),
                                        [tf.get_variable('item1_embed')]))
            if self.dom2_regs_item:
                tf.add_to_collection('reg2',
                                        tf.contrib.layers.apply_regularization(
                                        tf.contrib.layers.l2_regularizer(scale=self.dom2_regs_item),
                                        [tf.get_variable('item2_embed')]))

    def loss(self):
        with tf.name_scope('Loss'):
            loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.dom1_labels,tf.float32), logits=self.logits1)
            loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.dom2_labels,tf.float32), logits=self.logits2)

            # Regularization
            self.regularizer()
            loss1 += tf.reduce_sum(tf.get_collection('reg1'))
            loss2 += tf.reduce_sum(tf.get_collection('reg2'))
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            self.loss = tf.reduce_sum(alpha * loss1 + (1 - alpha) * loss2)

    def optimizer(self):
        with tf.name_scope('Optimizer'):
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def metrics(self):
        with tf.name_scope('Metrics'):
            pred1 = tf.cast(tf.round(self.scores1), tf.int32)
            pred2 = tf.cast(tf.round(self.scores2), tf.int32)

            # Calculate the MAP and RMSE of the prediction results
            _, self.map1 = tf.metrics.mean_absolute_error(predictions=pred1, labels=self.dom1_labels)
            _, self.map2 = tf.metrics.mean_absolute_error(predictions=pred2, labels=self.dom2_labels)
            _, self.rms1 = tf.metrics.root_mean_squared_error(predictions=pred1, labels=self.dom1_labels)
            _, self.rms2 = tf.metrics.root_mean_squared_error(predictions=pred2, labels=self.dom2_labels)

    def build(self, num_factors = 16):
        self.model(num_factors)
        self.loss()
        self.optimizer()
        self.metrics()
        print('Model Building Completed.')

    ############################################### Functions to run the model ######################################

    def train_one_epoch(self,epoch):
        uid1, iid1, lb1, uid2, iid2, lb2 = mtl.shuffle_list(self.train_uid1, self.train_iid1, self.train_labels1,
                                            self.train_uid2, self.train_iid2, self.train_labels2)
        uid1, iid1, lb1, uid2, iid2, lb2 = list(uid1), list(iid1), list(lb1), list(uid2), list(iid2), list(lb2)

        n_batches = 0
        total_loss = 0
        total_map1 = 0
        total_map2 = 0
        for i in range(self.num_batch):
            batch_user1 = uid1[i * self.batch_size:(i+1) * self.batch_size]
            batch_user2 = uid2[i * self.batch_size:(i+1) * self.batch_size]
            batch_item1 = iid1[i * self.batch_size:(i+1) * self.batch_size]
            batch_item2 = iid2[i * self.batch_size:(i+1) * self.batch_size]
            batch_labels1 = lb1[i * self.batch_size:(i+1) * self.batch_size]
            batch_labels2 = lb2[i * self.batch_size:(i+1) * self.batch_size]

            _, l, map1, map2 = self.session.run([self.opt, self.loss, self.map1, self.map2],
                                                feed_dict={self.dom1_uid: batch_user1, self.dom2_uid: batch_user2,
                                                           self.dom1_iid: batch_item1, self.dom2_iid: batch_item2,
                                                           self.dom1_labels: batch_labels1, self.dom2_labels: batch_labels2})
            n_batches += 1
            total_loss += l
            total_map1 += map1
            total_map2 += map2
            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("Epoch {0} Batch {1}: [Loss] = {2} [MAE] = {3}"
                          .format(epoch, n_batches, total_loss/n_batches,(total_map1+total_map2)/(2*n_batches)))

        print("Epoch {0}: [Loss] {1}".format(epoch, total_loss/n_batches))
        print("Epoch {0}: [MAE] {1} and {2}".format(epoch, total_map1/n_batches, total_map2/n_batches))

    def eval_one_epoch(self, epoch):
        uid1, iid1, uid2, iid2 = [],[],[],[]

        for u in range(self.num_user1):
            uid1 = uid1 + [u] * self.num_ranking_list
            iid1 = iid1 + self.test_dict1[u]
            uid2 = uid2 + [u] * self.num_ranking_list
            iid2 = iid2 + self.test_dict2[u]

        batch_size = self.num_ranking_list

        n_batches = 0
        n_mrr1 = 0
        n_mrr2 = 0
        total_hr1 = 0
        total_hr2 = 0
        total_ndcg1 = 0
        total_ndcg2 = 0
        total_mrr1 = 0
        total_mrr2 = 0

        for i in range(self.num_user1):
            batch_uid1, batch_iid1, batch_uid2, batch_iid2 = \
                uid1[i * batch_size:(i+1) * batch_size],\
                iid1[i * batch_size:(i+1) * batch_size],\
                uid2[i * batch_size:(i+1) * batch_size],\
                iid2[i * batch_size:(i+1) * batch_size]

            rk1, rk2 = self.session.run([self.scores1, self.scores2],
                                        feed_dict={self.dom1_uid: batch_uid1, self.dom1_iid: batch_iid1,
                                                   self.dom2_uid: batch_uid2, self.dom2_iid: batch_iid2})

            _, hr1, ndcg1, mrr1 = evl.evalTopK(rk1, batch_iid1, self.topk)
            _, hr2, ndcg2, mrr2 = evl.evalTopK(rk2, batch_iid2, self.topk)

            n_batches += 1
            total_hr1 += hr1
            total_hr2 += hr2
            total_ndcg1 += ndcg1
            total_ndcg2 += ndcg2
            if np.isinf(mrr1):
                pass
            else:
                n_mrr1 += 1
                total_mrr1 += mrr1
            if np.isinf(mrr2):
                pass
            else:
                n_mrr2 +=1
                total_mrr2 += mrr2
        print("Epoch {0}: [HR] {1} and {2}".format(epoch, total_hr1/n_batches, total_hr2/n_batches))
        print("Epoch {0}: [nDCG@{1}] {2} and {3}".format(epoch, self.topk, total_ndcg1/n_batches, total_ndcg2/n_batches))
        # print("Epoch {0}: [MRR] {1} and {2}".format(epoch, total_mrr1/n_batches, total_mrr2/n_batches))

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
    parser = argparse.ArgumentParser(description="CoNet for Cross Domain Recommendation")
    # parser.add_argument('--path1', nargs='?', default='Data/domain1/', type=str,
    #                     help='Input data path for domain1.')
    # parser.add_argument('--path2', nargs='?', default='Data/domain2/', type=str,
    #                     help='Input data path for domain2.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--nfactors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--ebregs', nargs='?', default='[0.001,0.001,0.001]', type=str,
                        help="Regularization constants for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--alpha', type=int, default=0.6,
                        help='The loss split ration between the two domains.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--ndcgk', type=int, default=10,
                        help='The K value of the Top-K ranking list.')
    parser.add_argument('--learner', nargs='?', default='adam', choices=('adam','adagrad','rmsprop','sgd'),
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, regs, num_neg, alpha, lr, ndcgk, num_factors = \
        args.epochs, args.batch_size, args.ebregs,args.num_neg,args.alpha,args.lr,args.ndcgk, args.nfactors

    regs = list(np.float32(eval(regs)))

    original_df1, train_df1, test_df1, test_dict1, num_users1, num_items1\
        = dfl.load_as_df(datafile='Data/books_small/original.csv')

    original_df2, train_df2, test_df2, test_dict2, num_users2, num_items2 \
        = dfl.load_as_df(datafile='Data/elec_small/original.csv')

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          intra_op_parallelism_threads=12,
                                          inter_op_parallelism_threads=12,
                                          gpu_options=gpu_options)) as sess:

        conet = CoNetRec(sess,num_neg=num_neg,top_K=ndcgk,
                         num_ranking_list=100,lr=lr,regs=regs,
                         alpha=alpha,epochs=num_epochs,batch_size=batch_size,
                         T=10**3, verbose=True)

        conet.prepare_data(original_df1=original_df1,train_df1=train_df1,
                           test_df1=test_df1,test_dict1=test_dict1,
                           original_df2=original_df2,train_df2=train_df2,
                           test_df2=test_df2,test_dict2=test_dict2)
        # print(len(conet.test_dict1[0]),conet.test_dict1[0])

        conet.build(num_factors)
        conet.train()
