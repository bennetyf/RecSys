'''
This module is used to generate datasets objects for usage in tensorflow
'''

import tensorflow as tf
import RecUtils as util

############################################# Helper Functions #########################################################

def genCSVCols(num):
    return [[0]]*num

############################################# The Dataset Class ########################################################
class Dataset(object):
    '''
    Class description
    '''
    def __init__(self, path, num_neg=0):
        '''
        Constructor
        '''
        self.root_path = path
        self.num_neg = num_neg

        # Load train data
        csv_cols = genCSVCols(3)

        if num_neg > 0:
            # Load the data with combination of the negative samples
            self.train = self.load_dataset(path + 'train_neg_'+str(int(num_neg))+'.csv', csv_cols)
        else:
            self.train = self.load_dataset(path + 'train.csv',csv_cols)

        # Load test data
        self.test = self.load_dataset(path + 'test.csv',csv_cols)

        # Load the topK data
        self.test_topk = self.load_dataset(path + 'test_topk.csv', csv_cols)

        # Get the number of users and items
        self.num_user, self.num_item = util.getUINum(path + 'train.csv')

        # Dummy Iterator for this dataset
        self.iter = None

    # Parse the CSV file
    def parse_csv(self, csv_cols, value):
        # global _CSV_COLUMN_DEFAULTS
        return tf.decode_csv(value, record_defaults=csv_cols)
        # a, b, c = tf.decode_csv(value, record_defaults=csv_cols)
        # return tf.stack([a]), tf.stack([b]), tf.stack([c])

    # Construct the dataset generally
    def load_dataset(self, datafile, csv_cols):
        # Check whether the file exists
        assert tf.gfile.Exists(datafile)
        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(datafile)
        # Parse each input line as CSV line
        # _CSV_COLUMN_DEFAULTS =
        return dataset.map(lambda val: self.parse_csv(csv_cols=csv_cols,value=val), num_parallel_calls=5)
