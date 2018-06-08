import tensorflow as tf
import numpy as np

from Dataset import Dataset
import Utils.RecEval as evl

class BPRMF():
    def __init__(self):
        self.path = 'Data/'
        self.num_factors = 32
        self.reg_rate = 0.1
        self.num_neg = 1
        self.learning_rate = 0.001
        self.epochs = 100
        self.batch_size = 128
        self.K = 10
        # Global step variable (This records how many times were parameter updated)
        self.global_step = tf.get_variable('global_step',
                                           shape=[], initializer=tf.constant_initializer(0),
                                           dtype=tf.int64, trainable=False)
        # Indicators
        self.training = False

    def data(self):
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
            ################################ Testing TopK Data Processing ################################
            dataset.test_topk = dataset.test_topk.batch(100)
            dataset.test_topk.init_iter = dataset.iter.make_initializer(dataset.test_topk)
            return dataset

        self.dataset = datasetProcess(self, Dataset(self.path, num_neg=self.num_neg))
        self.uid, self.iid, self.label = self.dataset.iter.get_next()
        self.num_user, self.num_item = self.dataset.num_user, self.dataset.num_item

    def model(self):
        # self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        # self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.neg_item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_item_id')
        self.y = tf.placeholder("float", [None], 'rating')

        self.P = tf.Variable(tf.random_normal([self.num_user, self.num_factors], stddev=0.01))
        self.Q = tf.Variable(tf.random_normal([self.num_item, self.num_factors], stddev=0.01))

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.uid)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.iid)
        neg_item_latent_factor = tf.nn.embedding_lookup(self.Q, self.neg_item_id)

        self.pred_y = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1)
        self.pred_y_neg = tf.reduce_sum(tf.multiply(user_latent_factor, neg_item_latent_factor), 1)

        self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pred_y - self.pred_y_neg))) + self.reg_rate * (
                    tf.norm(self.P) + tf.norm(self.Q))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
