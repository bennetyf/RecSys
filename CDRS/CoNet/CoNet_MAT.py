import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import argparse
import time
import multiprocessing as mp

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

############################################### The CoNet Model ########################################################

# Define the class for CoNet
class CoNetRec(object):
    def __init__(self, sess, num_neg=0, top_K = 10, num_ranking_list = 100,
                 num_factors=16, regs=[0.001,0.001,0.001],alpha = 0.5,
                 lr=0.001,
                 epochs=100, batch_size=128, T=10**3, verbose=False):
        '''Constructor'''
        # Parse the arguments and store them in the model
        self.session = sess
        self.num_neg = num_neg
        self.topk = top_K
        self.num_ranking_list = num_ranking_list

        self.num_factors = num_factors
        self.regs_user, self.dom1_regs_item, self.dom2_regs_item = regs
        self.alpha = alpha

        self.lr = lr

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

    def prepare_data(self, original_matrix1, train_matrix1, test_matrix1,
                           original_matrix2, train_matrix2, test_matrix2):

        # Meta Info
        self.num_user1, self.num_item1 = train_matrix1.shape
        self.num_user2, self.num_item2 = train_matrix2.shape

        self.neg_dict1,self.test_dict1 = mtl.negdict_mat(original_matrix1,test_matrix1,num_neg=self.num_ranking_list-1)
        self.neg_dict2,self.test_dict2 = mtl.negdict_mat(original_matrix2,test_matrix2,num_neg=self.num_ranking_list-1)

        self.train_uid1, self.train_iid1, self.train_labels1 = mtl.matrix_to_list(train_matrix1)
        self.train_uid2, self.train_iid2, self.train_labels2 = mtl.matrix_to_list(train_matrix2)

        # Extend the shorter training data
        length1, length2 = len(self.train_labels1), len(self.train_labels2)
        if length1 < length2:
            self.train_uid1, self.train_iid1, self.train_labels1 =\
                mtl.data_upsample_list(self.train_uid1, self.train_iid1, self.train_labels1,num_ext=length2-length1)
        if length2 < length1:
            self.train_uid2, self.train_iid2, self.train_labels2 =\
                mtl.data_upsample_list(self.train_uid2, self.train_iid2, self.train_labels2,num_ext=length1-length2)

        assert len(self.train_labels1) == len(self.train_labels2)

        # Negative Sampling on Lists
        print("Enter NegSa")
        start_time = time.time()

        results = mp.Pool(processes=2).map(gtl.mphelper,
                                           [(mtl.negative_sample_list, self.neg_dict1, self.train_uid1, self.train_iid1, self.train_labels1, self.num_neg,
                                             0),
                                            (mtl.negative_sample_list, self.neg_dict2, self.train_uid2, self.train_iid2, self.train_labels2, self.num_neg,
                                             0)])

        self.train_uid1, self.train_iid1, self.train_labels1 = results[0]
        self.train_uid2, self.train_iid2, self.train_labels2 = results[1]

        # self.train_uid1, self.train_iid1, self.train_labels1 \
        #    = mtl.negative_sample_list(self.neg_dict1, self.train_uid1, self.train_iid1, self.train_labels1,num_neg=self.num_neg,neg_val=0)
        #
        # self.train_uid2, self.train_iid2, self.train_labels2 \
        #    = mtl.negative_sample_list(self.neg_dict2, self.train_uid2, self.train_iid2, self.train_labels2, num_neg=self.num_neg, neg_val=0)
        print("Leaving NegSa")
        print("Negative Sampling Time: {0}".format(time.time() - start_time))

        assert len(self.train_labels1) == len(self.train_labels2)
        self.num_training = len(self.train_labels1)
        self.num_batch = int(self.num_training / self.batch_size)

        print("Data Preparation Completed.")

    def build_model(self):
        '''
        The Collaborative Cross Network Model
        '''
        with tf.variable_scope('Model'):
            self.dom1_uid = tf.placeholder(dtype=tf.int32, shape=[None], name='dom1_user_id')
            self.dom2_uid = tf.placeholder(dtype=tf.int32, shape=[None], name='dom2_user_id')
            self.dom1_iid = tf.placeholder(dtype=tf.int32, shape=[None], name='dom1_item_id')
            self.dom2_iid = tf.placeholder(dtype=tf.int32, shape=[None], name='dom2_item_id')
            self.dom1_labels = tf.placeholder(dtype=tf.float32, shape=[None], name='dom1_labels')
            self.dom2_labels = tf.placeholder(dtype=tf.float32, shape=[None], name='dom2_labels')

            # One-hot input
            embeddings_user = tf.get_variable(name='user_embed',
                                              shape=[self.num_user1, self.num_factors],
                                              initializer=tf.random_uniform_initializer(-1,1))

            embeddings_item1 = tf.get_variable(name='item1_embed',
                                               shape=[self.num_item1, self.num_factors],
                                               initializer=tf.random_uniform_initializer(-1,1))

            embeddings_item2 = tf.get_variable(name='item2_embed',
                                               shape=[self.num_item2, self.num_factors],
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

            # Cross Stitching
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
            mlp_vector1 = tf.layers.dense(mlp_vector1_tmp, units=8, activation=tf.nn.relu)
            mlp_vector2 = tf.layers.dense(mlp_vector2_tmp, units=8, activation=tf.nn.relu)
            mlp_vector1 = tf.layers.dropout(mlp_vector1, rate=0.5)
            mlp_vector2 = tf.layers.dropout(mlp_vector2, rate=0.5)

            W31 = tf.get_variable('W31', shape=[8,4], initializer=tf.random_uniform_initializer(-1,1))
            W32 = tf.get_variable('W32', shape=[8,4], initializer=tf.random_uniform_initializer(-1,1))
            H3 = tf.get_variable('H3', shape=[8,4], initializer=tf.random_uniform_initializer(-1,1))

            mlp_vector1_tmp = tf.nn.relu(tf.matmul(mlp_vector1,W31) + tf.matmul(mlp_vector2,H3))
            mlp_vector2_tmp = tf.nn.relu(tf.matmul(mlp_vector2,W32) + tf.matmul(mlp_vector1,H3))

            # Fourth Layer
            mlp_vector1 = tf.layers.dense(mlp_vector1_tmp, units=4, activation=tf.nn.relu)
            mlp_vector2 = tf.layers.dense(mlp_vector2_tmp, units=4, activation=tf.nn.relu)
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

            self.pred_y1 = tf.sigmoid(self.logits1)
            self.pred_y2 = tf.sigmoid(self.logits2)

            # Loss
            loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.dom1_labels, logits=self.logits1))
            loss2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.dom2_labels, logits=self.logits2))

            loss1 += self.regs_user * tf.nn.l2_loss(embed_layer_user1) +\
                     self.dom1_regs_item * tf.nn.l2_loss(embed_layer_item1)

            loss2 += self.regs_user * tf.nn.l2_loss(embed_layer_user2) +\
                     self.dom2_regs_item * tf.nn.l2_loss(embed_layer_item2)

            alpha = tf.constant(self.alpha, dtype=tf.float32)
            self.loss = alpha * loss1 + (1 - alpha) * loss2

            # Opt
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            # Metrics
            pred1 = tf.cast(tf.round(self.pred_y1), tf.float32)
            pred2 = tf.cast(tf.round(self.pred_y2), tf.float32)
            self.mae1 = tf.reduce_mean(tf.abs(pred1 - self.dom1_labels))
            self.mae2 = tf.reduce_mean(tf.abs(pred2 - self.dom2_labels))
            self.rms1 = tf.sqrt(tf.reduce_mean(tf.square(pred1 - self.dom1_labels)))
            self.rms2 = tf.sqrt(tf.reduce_mean(tf.square(pred2 - self.dom2_labels)))

            print('Model Building Completed.')

    ############################################### Functions to run the model ######################################

    def train_one_epoch(self,epoch):
        uid1, iid1, lb1, uid2, iid2, lb2 = gtl.shuffle_list(self.train_uid1, self.train_iid1, self.train_labels1,
                                            self.train_uid2, self.train_iid2, self.train_labels2)
        # uid1, iid1, lb1, uid2, iid2, lb2 = list(uid1), list(iid1), list(lb1), list(uid2), list(iid2), list(lb2)

        n_batches = 0
        total_loss = 0
        total_mae1 = 0
        total_mae2 = 0
        for i in range(self.num_batch):
            batch_user1 = uid1[i * self.batch_size:(i+1) * self.batch_size]
            batch_user2 = uid2[i * self.batch_size:(i+1) * self.batch_size]
            batch_item1 = iid1[i * self.batch_size:(i+1) * self.batch_size]
            batch_item2 = iid2[i * self.batch_size:(i+1) * self.batch_size]
            batch_labels1 = lb1[i * self.batch_size:(i+1) * self.batch_size]
            batch_labels2 = lb2[i * self.batch_size:(i+1) * self.batch_size]

            _, l, mae1, mae2 = self.session.run([self.opt, self.loss, self.mae1, self.mae2],
                                                feed_dict={self.dom1_uid: batch_user1, self.dom2_uid: batch_user2,
                                                           self.dom1_iid: batch_item1, self.dom2_iid: batch_item2,
                                                           self.dom1_labels: batch_labels1, self.dom2_labels: batch_labels2})
            n_batches += 1
            total_loss += l
            total_mae1 += mae1
            total_mae2 += mae2
            # total_mae1 += evl.evalMAE(batch_labels1,np.round(pred1))
            # total_mae2 += evl.evalMAE(batch_labels2,np.round(pred2))
            if self.verbose:
                if n_batches % self.skip_step == 0:
                    print("Epoch {0} Batch {1}: [Loss] = {2} [MAE] = {3}"
                          .format(epoch, n_batches, total_loss/n_batches,(total_mae1+total_mae2)/(2*n_batches)))

        print("Epoch {0}: [Loss] {1}".format(epoch, total_loss/n_batches))
        print("Epoch {0}: [MAE] {1} and {2}".format(epoch, total_mae1/n_batches, total_mae2/n_batches))

    def eval_one_epoch(self, epoch):
        uid1, iid1, uid2, iid2 = [],[],[],[]

        for u in range(self.num_user1):
            uid1.extend([u] * self.num_ranking_list)
            iid1.extend(self.test_dict1[u])
            uid2.extend([u] * self.num_ranking_list)
            iid2.extend(self.test_dict2[u])

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

            rk1, rk2 = self.session.run([self.pred_y1, self.pred_y2],
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

    parser.add_argument('--nfactors', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--ebregs', nargs='?', default='[0.01,0.005,0.005]', type=str,
                        help="Regularization constants for user and item embeddings.")
    parser.add_argument('--alpha', type=int, default=0.6,
                        help='The loss split ration between the two domains.')

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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam', choices=('adam','adagrad','rmsprop','sgd'),
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    num_epochs, batch_size, regs, num_neg, alpha, lr, ndcgk, num_factors, num_ranking_list = \
        args.epochs, args.batch_size, args.ebregs,args.num_neg,args.alpha,args.lr,args.ndcgk, args.nfactors, args.num_rk

    regs = list(np.float32(eval(regs)))

    original_matrix1, train_matrix1, test_matrix1, num_users1, num_items1\
        = mtl.load_as_matrix(datafile='Data/books_small/original.csv')

    original_matrix2, train_matrix2, test_matrix2, num_users2, num_items2\
        = mtl.load_as_matrix(datafile='Data/elec_small/original.csv')

    print("Number of users in domain 1 is {0}".format(num_users1))
    print("Number of items in domain 1 is {0}".format(num_items1))
    print("Number of ratings in domain 1 in all is {0}".format(original_matrix1.nnz))
    print("Number of ratings in domain 1 for training is {0}".format(train_matrix1.nnz))
    print("Ratings density of domain 1 for training is {0}".format(train_matrix1.nnz/(num_users1*num_items1)))

    print("Number of users in domain 2 is {0}".format(num_users2))
    print("Number of items in domain 2 is {0}".format(num_items2))
    print("Number of ratings in domain 2 in all is {0}".format(original_matrix2.nnz))
    print("Number of ratings in domain 2 for training is {0}".format(train_matrix2.nnz))
    print("Ratings density of domain 2 for training is {0}".format(train_matrix2.nnz/(num_users2*num_items2)))

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=12,
                                          inter_op_parallelism_threads=12,
                                          gpu_options=gpu_options)) as sess:

        conet = CoNetRec(sess,num_neg=num_neg,top_K=ndcgk,num_ranking_list=num_ranking_list,
                         lr=lr,
                         num_factors=num_factors, regs=regs, alpha=alpha,
                         epochs=num_epochs,batch_size=batch_size,T=10**3, verbose=False)

        conet.prepare_data(original_matrix1=original_matrix1,train_matrix1=train_matrix1,
                           test_matrix1=test_matrix1,
                           original_matrix2=original_matrix2,train_matrix2=train_matrix2,
                           test_matrix2=test_matrix2)
        # print(len(conet.test_dict1[0]),conet.test_dict1[0])

        conet.build_model()
        conet.train()
