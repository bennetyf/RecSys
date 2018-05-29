'''
doc
'''

import scipy.sparse as sp
import pandas as pd


class DataSet(object):
    def __init__(self, path):
        '''
        Constructor
        '''
        self.c_size = 10**6
        self.trainMatrix = self.load_training_data(path + 'training.csv',self.c_size)
        self.testList = self.load_testing_data(path + 'test.csv',self.c_size)
        self.negList = self.load_negative_data(path + 'negative.csv',self.c_size)

        # Test the reading results
        assert len(self.testList) == len(self.negList)

        self.n_user, self.n_item = self.trainMatrix.shape

    def load_training_data(self, path, c_size = 10**6):
        '''The training data is loaded into a sparse matrix'''
        data =  pd.DataFrame([])
        for df in pd.read_csv(path, header=None, chunksize=c_size):
            data = data.append(df.loc[:,:], ignore_index=True)

        # Convert pandas data frame into scipy sparse matrix
        row = data.loc[:,0].tolist()
        col = data.loc[:,1].tolist()
        element = data.loc[:,2].tolist()
        return sp.csr_matrix((element,(row,col)),shape=[max(row)+1,max(col)+1])

    def load_testing_data(self, path, c_size = 10**6):
        '''The testing data is loaded into a list'''
        testinglist=[]
        for df in pd.read_csv(path, header=None, chunksize=c_size):
            for i in range(df.shape[0]):
                testinglist.append(df.iloc[i,:].tolist())
        return  testinglist

    def load_negative_data(self, path, c_size = 10**6):
        '''The negative data is also loaded into a list'''
        neglist=[]
        for df in pd.read_csv(path, header=None, chunksize=c_size):
            for i in range(df.shape[0]):
                neg_local=[]
                for j in range(df.shape[1]-1):
                    # Skip the first item which is a tuple representing the testing target
                    neg_local.append(int(df.iloc[i,j+1]))
                neglist.append(neg_local)
        return neglist
