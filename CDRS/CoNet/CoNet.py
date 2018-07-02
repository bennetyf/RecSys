import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

import tensorflow as tf
import numpy as np
import argparse

from CDRS.CoNet.Dataset import Dataset
import Utils.RecEval as evl

######################################## Parse Arguments ###############################################################
def parseArgs():
    parser = argparse.ArgumentParser(description="CoNet for Cross Domain Recommendation")
    parser.add_argument('--path1', nargs='?', default='Data/domain1/', type=str,
                        help='Input data path for domain1.')
    parser.add_argument('--path2', nargs='?', default='Data/domain2/', type=str,
                        help='Input data path for domain2.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size.')
    parser.add_argument('--nfactors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--ebregs', nargs='?', default='[0.001,0.0005,0.0005]', type=str,
                        help="Regularization constants for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--alpha', type=int, default=0.6,
                        help='The loss split ration between the two domains.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--ndcgk', type=int, default=10,
                        help='The K value of the Top-K ranking list.')
    parser.add_argument('--learner', nargs='?', default='adam', choices=('adam','adagrad','rmsprop','sgd'),
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model or not.')
    return parser.parse_args()

############################################### The CoNet Model ########################################################

# Define the class for CoNet
class CoNetRec(object):
    def __init__(self, params):
        '''Constructor'''
        # Parse the arguments and store them in the model
        self.dom1_path = params.path1 # The data path for domain1 data
        self.dom2_path = params.path2 # The data path for domain2 data
        self.num_factors = params.nfactors # Number of the hidden neurons in the embedding layer
        self.regs_user, self.dom1_regs_item, \
        self.dom2_regs_item\
            = np.float32(eval(params.ebregs)) # Regularizers for the embedding layer parameters
        self.num_neg = params.num_neg # Number of negative instances for one explicit rating in training
        self.optimizer = params.learner # The optimizer for training the model
        self.learning_rate = params.lr # The learning rate of the model training
        self.alpha = params.alpha # The weight split of the two losses of the domains
        self.epochs = params.epochs # Number of epochs
        self.batch_size = params.batch_size # The batch size
        self.K = params.ndcgk
        self.verbose = params.verbose # Flag to decide whether intermediate results should be shown or not

        # Global step variable (This records how many times were parameter updated)
        self.global_step = tf.get_variable('global_step',
                                           shape=[],initializer=tf.constant_initializer(0),
                                           dtype=tf.int64, trainable=False)

        # Indicators
        self.training = False

    def data(self):
        '''
        Construct the datasets for training and testing
        '''
        # Simple PreProcessing of the dataset
        def datasetProcess(self, dataset):
            ################################ Training Data Processing ####################################
            # Shuffle the training data
            dataset.train.shuffle(buffer_size=2048)
            # Batch the training data
            if self.batch_size <= 0:
                dataset.train = dataset.train.batch(batch_size=1)
            else:
                dataset.train = dataset.train.batch(batch_size=self.batch_size)

            # Make a general iterator that goes through the training, testing and top-K testing datasets
            # It is important to have the same structrue of the three datasets
            dataset.iter = tf.data.Iterator.from_structure(dataset.train.output_types, dataset.train.output_shapes)

            # Generate iterator and initializer for the training dataset
            dataset.train.init_iter = dataset.iter.make_initializer(dataset.train)

            ################################ Testing Data Processing #####################################
            if self.batch_size <= 0:
                dataset.test = dataset.test.batch(batch_size=1)
            else:
                dataset.test = dataset.test.batch(batch_size=self.batch_size)
            dataset.test.init_iter = dataset.iter.make_initializer(dataset.test)

            ################################ Testing TopK Data Processing ################################
            dataset.test_topk = dataset.test_topk.batch(100)
            dataset.test_topk.init_iter = dataset.iter.make_initializer(dataset.test_topk)
            return dataset

        # Generate Dataset Objects (Using the Dataset API)
        self.dom1_dataset = datasetProcess(self, Dataset(self.dom1_path, num_neg=self.num_neg))
        self.dom2_dataset = datasetProcess(self, Dataset(self.dom2_path, num_neg=self.num_neg))

        # Generate the User Input
        assert self.dom1_dataset.num_user == self.dom2_dataset.num_user
        self.num_user = self.dom1_dataset.num_user

        ###### Extract the data from the datasets
        self.dom1_uid, self.dom1_iid, self.dom1_label = self.dom1_dataset.iter.get_next()
        self.dom2_uid, self.dom2_iid, self.dom2_label = self.dom2_dataset.iter.get_next()

    def model(self):
        '''
        The Collaborative Cross Network Model
        '''
        with tf.variable_scope('Model'):
            # One-hot input
            embeddings_user = tf.get_variable('user_embed',
                                              shape=[self.num_user, self.num_factors],
                                              initializer=tf.random_uniform_initializer(-1.0,1.0))

            embeddings_item1 = tf.get_variable('item1_embed',
                                               shape=[self.dom1_dataset.num_item, self.num_factors],
                                               initializer=tf.random_uniform_initializer(-1.0,1.0))

            embeddings_item2 = tf.get_variable('item2_embed',
                                               shape=[self.dom2_dataset.num_item, self.num_factors],
                                               initializer=tf.random_uniform_initializer(-1.0,1.0))

            # The Embedding Layer
            embed_layer_user1 = tf.nn.embedding_lookup(embeddings_user, self.dom1_uid)
            embed_layer_user2 = tf.nn.embedding_lookup(embeddings_user, self.dom2_uid)
            embed_layer_item1 = tf.nn.embedding_lookup(embeddings_item1, self.dom1_iid)
            embed_layer_item2 = tf.nn.embedding_lookup(embeddings_item2, self.dom2_iid)

            # First Hidden Layer
            mlp_vector1 = tf.concat([embed_layer_user1,embed_layer_item1], axis=1)
            mlp_vector2 = tf.concat([embed_layer_user2,embed_layer_item2], axis=1)


            mlp_vector1 = tf.layers.dense(mlp_vector1, units=64, activation=tf.nn.relu)
            mlp_vector2 = tf.layers.dense(mlp_vector2, units=64, activation=tf.nn.relu)

            # Cross Stitching
            W11 = tf.get_variable('W11', shape=[64,16], initializer=tf.random_uniform_initializer(-1.0,1.0))
            W12 = tf.get_variable('W12', shape=[64,16], initializer=tf.random_uniform_initializer(-1.0,1.0))
            H1 = tf.get_variable('H1', shape=[64,16], initializer=tf.random_uniform_initializer(-1.0,1.0))

            mlp_vector1_1 = tf.nn.relu(tf.matmul(mlp_vector1, W11)+tf.matmul(mlp_vector2,H1))
            mlp_vector2_1 = tf.nn.relu(tf.matmul(mlp_vector2, W12)+tf.matmul(mlp_vector1,H1))


            # Second Hidden Layer
            mlp_vector1_1 = tf.layers.dense(mlp_vector1_1, units=32, activation=tf.nn.relu)
            mlp_vector2_1 = tf.layers.dense(mlp_vector2_1, units=32, activation=tf.nn.relu)

            W21 = tf.get_variable('W21', shape=[32,16], initializer=tf.random_uniform_initializer(-1.0,1.0))
            W22 = tf.get_variable('W22', shape=[32,16], initializer=tf.random_uniform_initializer(-1.0,1.0))
            H2 = tf.get_variable('H2', shape=[32,16], initializer=tf.random_uniform_initializer(-1.0,1.0))

            mlp_vector1_2 = tf.nn.relu(tf.matmul(mlp_vector1_1,W21) + tf.matmul(mlp_vector2_1,H2))
            mlp_vector2_2 = tf.nn.relu(tf.matmul(mlp_vector2_1,W22) + tf.matmul(mlp_vector1_1,H2))

            # Third Hidden Layer
            mlp_vector1_2 = tf.layers.dense(mlp_vector1_2, units=16, activation=tf.nn.relu)
            mlp_vector2_2 = tf.layers.dense(mlp_vector2_2, units=16, activation=tf.nn.relu)

            W31 = tf.get_variable('W31', shape=[16,16], initializer=tf.random_uniform_initializer(-1.0,1.0))
            W32 = tf.get_variable('W32', shape=[16,16], initializer=tf.random_uniform_initializer(-1.0,1.0))
            H3 = tf.get_variable('H3', shape=[16,16], initializer=tf.random_uniform_initializer(-1.0,1.0))

            mlp_vector1_3 = tf.nn.relu(tf.matmul(mlp_vector1_2,W31) + tf.matmul(mlp_vector2_2,H3))
            mlp_vector2_3 = tf.nn.relu(tf.matmul(mlp_vector2_2,W32) + tf.matmul(mlp_vector1_2,H3))

            # Fourth Layer
            mlp_vector1_3 = tf.layers.dense(mlp_vector1_3, units=8, activation=tf.nn.relu)
            mlp_vector2_3 = tf.layers.dense(mlp_vector2_3, units=8, activation=tf.nn.relu)

            # W41 = tf.get_variable('W41', shape=[8, 8], initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # W42 = tf.get_variable('W42', shape=[8, 8], initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # H4 = tf.get_variable('H4', shape=[8, 8], initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # mlp_vector1_4 = tf.nn.relu(tf.matmul(mlp_vector1_3, W41) + tf.matmul(mlp_vector2_3, H4))
            # mlp_vector2_4 = tf.nn.relu(tf.matmul(mlp_vector2_3, W42) + tf.matmul(mlp_vector1_3, H4))

            # Output
            mlp_vector1_out = tf.layers.dense(mlp_vector1_3, units=1, activation=tf.identity)
            mlp_vector2_out = tf.layers.dense(mlp_vector2_3, units=1, activation=tf.identity)

            self.logits1 = tf.reshape(mlp_vector1_out,shape=[-1])
            self.logits2 = tf.reshape(mlp_vector2_out,shape=[-1])

    def regularizer(self):
        with tf.variable_scope('Model',reuse=True):
            # Regularization in the model
            if self.regs_user:
                tf.add_to_collections(['reg1', 'reg2'],
                                        tf.contrib.layers.apply_regularization(
                                        tf.contrib.layers.l2_regularizer(scale=self.regs_user),
                                        [tf.get_variable('user_embed')]))
            if self.dom1_regs_item:
                tf.add_to_collections(['reg1'],
                                        tf.contrib.layers.apply_regularization(
                                        tf.contrib.layers.l2_regularizer(scale=self.dom1_regs_item),
                                        [tf.get_variable('item1_embed')]))
            if self.dom2_regs_item:
                tf.add_to_collections(['reg2'],
                                        tf.contrib.layers.apply_regularization(
                                        tf.contrib.layers.l2_regularizer(scale=self.dom2_regs_item),
                                        [tf.get_variable('item2_embed')]))

    def loss(self):
        with tf.name_scope('Loss'):
            # It is better to use
            loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.dom1_label,tf.float32), logits=self.logits1)
            loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.dom2_label,tf.float32), logits=self.logits2)

            # Regularization
            self.regularizer()
            loss1 += tf.reduce_sum(tf.get_collection('reg1'))
            loss2 += tf.reduce_sum(tf.get_collection('reg2'))
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            self.loss = tf.reduce_sum(alpha * loss1 + (1 - alpha) * loss2)

    def opt_algo(self):
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def output(self):
        with tf.name_scope('Output'):
            # Probabilities
            prob1 = tf.sigmoid(self.logits1)
            prob2 = tf.sigmoid(self.logits2)
            pred1 = tf.cast(tf.round(prob1), tf.int32)
            pred2 = tf.cast(tf.round(prob2), tf.int32)

            # Calculate the MAP and RMSE of the prediction results
            _, self.map1 = tf.metrics.mean_absolute_error(predictions=pred1, labels=self.dom1_label)
            _, self.map2 = tf.metrics.mean_absolute_error(predictions=pred2, labels=self.dom2_label)
            _, self.rms1 = tf.metrics.root_mean_squared_error(predictions=pred1, labels=self.dom1_label)
            _, self.rms2 = tf.metrics.root_mean_squared_error(predictions=pred2, labels=self.dom2_label)

            # For Top-K evaluation
            self.scores1, self.scores2 = prob1, prob2

    def build_all(self):
        self.data()
        self.model()
        self.loss()
        self.opt_algo()
        self.output()

    ############################################### Functions to run the model ######################################

    def train_one_epoch(self, session, init, epoch):
        session.run(init)
        self.training = True
        n_batches = 0
        total_loss = 0
        total_map1 = 0
        total_map2 = 0
        try:
            while True:
                _, l, map1, map2 = session.run([self.opt, self.loss, self.map1, self.map2])
                self.global_step = self.global_step + 1
                n_batches += 1
                total_loss += l
                total_map1 += map1
                total_map2 += map2
        except tf.errors.OutOfRangeError:
            pass
        print("Epoch {0}: [Loss] {1}".format(epoch, total_loss/n_batches))
        print("Epoch {0}: [MAE] {1} and {2}".format(epoch, total_map1/n_batches, total_map2/n_batches))

    def eval_one_epoch(self, session, init, epoch):
        session.run(init)
        self.training = False
        n_batches = 0
        n_mrr1 = 0
        n_mrr2 = 0
        total_hr1 = 0
        total_hr2 = 0
        total_ndcg1 = 0
        total_ndcg2 = 0
        total_mrr1 = 0
        total_mrr2 = 0

        try:
            while True:
                rk1, rk2, iid1, iid2 = session.run([self.scores1, self.scores2, self.dom1_iid, self.dom2_iid])
                _, hr1, ndcg1, mrr1 = evl.evalTopK(rk1, iid1, self.K)
                _, hr2, ndcg2, mrr2 = evl.evalTopK(rk2, iid2, self.K)
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
        except tf.errors.OutOfRangeError:
            pass

        print("Epoch {0}: [HR] {1} and {2}".format(epoch, total_hr1/n_batches, total_hr2/n_batches))
        print("Epoch {0}: [nDCG@{1}] {2} and {3}".format(epoch, self.K, total_ndcg1/n_batches, total_ndcg2/n_batches))
        print("Epoch {0}: [MRR] {1} and {2}".format(epoch, total_mrr1/n_batches, total_mrr2/n_batches))

    # Final Training of the model
    def train(self):
        train_init = [self.dom1_dataset.train.init_iter, self.dom2_dataset.train.init_iter]
        test_init = [self.dom1_dataset.test_topk.init_iter, self.dom2_dataset.test_topk.init_iter]

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

            # Initial evaluation before training
            self.eval_one_epoch(sess, test_init, -1)
            for i in range(self.epochs):
                self.train_one_epoch(sess,train_init,i)
                self.eval_one_epoch(sess,test_init,i)

########################################################################################################################

if __name__ == "__main__":
    model = CoNetRec(parseArgs())
    model.build_all()
    model.train()

    # train_init = [model.dom1_dataset.train.init_iter, model.dom2_dataset.train.init_iter]
    # test_init = [model.dom1_dataset.test_topk.init_iter, model.dom2_dataset.test_topk.init_iter]
    # var_init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    #
    # with tf.Session() as sess:
    #     # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #     # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    #
    #     sess.run(var_init)
    #     sess.run(train_init)
    #     # print(sess.run([model.s0, model.s, model.s1, model.s2, model.s3]))
    #     # for i in range(5):
    #         # print(sess.run(model.dom1_dataset.iter.get_next()))
    #
    #     for i in range(5):
    #         model.train_one_epoch(sess,train_init,i)
    #         model.eval_one_epoch(sess,test_init,i)