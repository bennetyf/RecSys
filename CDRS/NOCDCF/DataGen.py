'''
This program is used to preprocess the data for NOCDCF
'''
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

import Utils.RecUtils as ul
import numpy as np
import multiprocessing as mp


def mphelper(args):
    return ul.negSample(*args)

def CoNetDataPro(spath, flag, topk, negsa, domain1 = 'Amazon_Instant_Video', domain2 = 'Musical_Instruments'):
    # domainp1 = '/media/work/Workspace/PhD_Projects/RecSys/CDRS/CoNet/Data/domain3/'
    # domainp2 = '/media/work/Workspace/PhD_Projects/RecSys/CDRS/CoNet/Data/domain4/'
    domainp1 = '/share/scratch/fengyuan/Projects/RecSys/CDRS/NOCDCF/Data/InstantVideo/'
    domainp2 = '/share/scratch/fengyuan/Projects/RecSys/CDRS/NOCDCF/Data/MusicalInstrument/'

    if flag == True: # Recalculating from the original data
    ################ Load the original data, filter and splitting #####################
        print("Generating Data from Original Files")
        print("Loading Data for Domain {0}".format(domain1))
        data1 = ul.loadData(spath + domain1 + '_and_' + domain2 + '_ratings_1.csv',
                                     names=['uid','iid','ratings','time'])
        data1 = data1.drop(labels=['time'],axis=1)
        data1 = ul.mat2Bin(data1,threshold=2)
        data1 = ul.zeroPruning(data1)
        # data1 = ul.uiFilter(data1, opt='user', threshold=5)
        # data1 = ul.uiFilter(data1,opt='random', n_user=10**5, n_item=10**4)
        data1 = ul.uiFilter(data1,opt='item',threshold=5)
        data1 = ul.uiFilter(data1,opt='user',threshold=5)

        print("Loading Data for Domain {0}".format(domain2))
        data2 = ul.loadData(spath + domain1 + '_and_' + domain2 + '_ratings_2.csv',
                                  names=['uid', 'iid', 'ratings', 'time'])
        data2 = data2.drop(labels=['time'], axis=1)
        data2 = ul.mat2Bin(data2, threshold=2)
        data2 = ul.zeroPruning(data2)
        # data2 = ul.uiFilter(data2, opt='user', threshold=5)
        # data2 = ul.uiFilter(data2, opt='random', n_user=2*10**5, n_item=10**4)
        data2 = ul.uiFilter(data2, opt='item', threshold=5)
        data2 = ul.uiFilter(data2, opt='user', threshold=5)
        print("Data Loaded")

        tmp = data1.groupby(['uid'])[['iid']].count()
        print(tmp.min(axis=0))

        tmp = data2.groupby(['uid'])[['iid']].count()
        print(tmp.min(axis=0))

        # Generating the Shared Users
        # data1, data2, user_list = ul.filterBySharedUsers(data1, data2)

        # Turn into Number Coding
        data1, data2 = ul.id2Num(data1), ul.id2Num(data2)

        print("Unique Users in domain {0} is {1}".format(domain1, data1['uid'].unique().shape[0]))
        print("Unique Items in domain {0} is {1}".format(domain1, data1['iid'].unique().shape[0]))
        print("Unique Ratings in domain {0} is {1}".format(domain1, data1['ratings'].shape[0]))
        print("Density in domain {0} is {1}".format(domain1,
                                                    100*data1['ratings'].shape[0]/(
                                                            data1['uid'].unique().shape[0] * data1['iid'].unique().shape[0])))

        print("Unique Users in domain {0} is {1}".format(domain2, data2['uid'].unique().shape[0]))
        print("Unique Items in domain {0} is {1}".format(domain2, data2['iid'].unique().shape[0]))
        print("Unique Ratings in domain {0} is {1}".format(domain2, data2['ratings'].shape[0]))
        print("Density in domain {0} is {1}".format(domain2,
                                                    100*data2['ratings'].shape[0] / (
                                                        data2['uid'].unique().shape[0] * data2['iid'].unique().shape[0])))

        data1.to_csv(domainp1 + 'original.csv', index=False, header=False)
        data2.to_csv(domainp2 + 'original.csv', index=False, header=False)

        # Splitting Training and Testing Data
        train1, test1 = ul.dataSplit(data1, opt='leave-one-out')
        train2, test2 = ul.dataSplit(data2, opt='leave-one-out')
        # print(train1['uid'].unique().shape[0])
        # print(train2['uid'].unique().shape[0])

        # Make the training data of the same length in the two domains
        # n_ratings1 = train1.shape[0]
        # n_ratings2 = train2.shape[0]
        # if n_ratings1 == n_ratings2:
        #     pass
        # elif n_ratings1 < n_ratings2:
        #     idx = np.random.randint(n_ratings1,size=n_ratings2-n_ratings1).tolist()
        #     train1 = train1.append(train1.loc[idx,:],ignore_index=True)
        # else:
        #     idx = np.random.randint(n_ratings2, size=n_ratings1 - n_ratings2).tolist()
        #     train2 = train2.append(train2.loc[idx, :], ignore_index=True)
        # print(train1.shape,train2.shape)
        # print(train1.max(axis=0))
        # print(train2.max(axis=0))
        # print(train2)

        train1.to_csv(domainp1 + 'train.csv', index=False, header=False)
        test1.to_csv(domainp1 + 'test.csv', index=False, header=False)
        train2.to_csv(domainp2 + 'train.csv', index=False, header=False)
        test2.to_csv(domainp2 + 'test.csv', index=False, header=False)
    else:
        # Load the data from the disk
        print("Loading Data for Domain {0}".format(domain1))
        data1 = ul.loadData(domainp1 + 'original.csv', names=['uid', 'iid', 'ratings'])
        train1 = ul.loadData(domainp1 + 'train.csv', names=['uid', 'iid', 'ratings'])
        test1 = ul.loadData(domainp1 + 'test.csv', names=['uid', 'iid', 'ratings'])

        print("Loading Data for Domain {0}".format(domain2))
        data2 = ul.loadData(domainp2 + 'original.csv', names=['uid', 'iid', 'ratings'])
        train2 = ul.loadData(domainp2 + 'train.csv', names=['uid', 'iid', 'ratings'])
        test2 = ul.loadData(domainp2 + 'test.csv', names=['uid', 'iid', 'ratings'])
        print("Loading Completed")

    if topk:
        pool = mp.Pool(processes=6)
        # Generating the TopK Testing Data
        domainp3 = domainp1 + 'tmp/'
        domainp4 = domainp2 + 'tmp/'
        domainp5 = domainp1 + 'tmp2/'
        domainp6 = domainp2 + 'tmp2/'
        pool.map(mphelper, [(data1,test1, 99, 0, 10**5, 'direct', 'topk', True, domainp1, 'test_topk.csv'),
                            (data2,test2, 99, 0, 10**5, 'direct', 'topk', True, domainp2, 'test_topk.csv'),
                            (data1, train1, 1, 0, 10**4, 'direct', 'train', True, domainp3, 'train_neg_1.csv'),
                            (data2, train2, 1, 0, 10**4, 'direct', 'train', True, domainp4, 'train_neg_1.csv'),
                            (data1, train1, 4, 0, 10**4, 'direct', 'train', True, domainp5, 'train_neg_4.csv'),
                            (data2, train2, 4, 0, 10**4, 'direct', 'train', True, domainp6, 'train_neg_4.csv')
                            ])

        # pool.map(mphelper, [(data1, train1, 4, 0, 10**4, 'direct', 'train', True, domainp5, 'train_neg_4.csv'),
        #                     (data2, train2, 4, 0, 10**4, 'direct', 'train', True, domainp6, 'train_neg_4.csv')])
        pool.close()
        # print("Generating Negative Samples for TopK Testing, domain {0}".format(domain1))
        # ul.negSample(data1, test1, 99, neg_val=0, maxlines=10**6, method='direct', mod='topk',
        #              store=True,store_path=domainp1,fname='test_topk.csv')

        # print("Generating Negative Samples for TopK Testing, domain {0}".format(domain2))
        # ul.negSample(data2, test2, 99, neg_val=0, maxlines=10**6, method='direct', mod='topk',
        #              store=True, store_path=domainp2, fname='test_topk.csv')

    if negsa:
        pool = mp.Pool(processes=2)
        pool.map(mphelper, [(data1, train1, 4, 0, 10 ** 5, 'direct', 'train', True, domainp1, 'train_neg_4.csv'),
                            (data2, train2, 4, 0, 10 ** 5, 'direct', 'train', True, domainp2, 'train_neg_4.csv')])
        pool.close()
        # Generate Training Data with negative sampling
        # ul.negSample(data1, train1, 1, neg_val=0, maxlines=10**4, method='direct', mod='train',
        #              store=True, store_path=domainp1, fname='train_neg_1.csv')
        # ul.negSample(data2, train2, 1, neg_val=0, maxlines=10**4, method='direct', mod='train',
        #          store=True, store_path=domainp2, fname='train_neg_2.csv')
        # print(np.expand_dims(np.asarray(neg),axis=2))

# Merge the different domain data into one domain to compare with baseline single-domain algorithms
def Merge_Domains(data1, data2):
    least_iid = data1.max(axis=0)['iid'] + 1
    data2.loc[:,'iid'] = data2.loc[:,'iid'] + least_iid
    data1 = data1.append(data2,ignore_index=True)
    return data1.sort_values(by=['uid'])

###########################################################################################################
if __name__ == "__main__":
    # Generate the CoNet Data
    # CoNetDataPro('/media/work/Workspace/PhD_Projects/RecSys/CDRS/Data/Amazon/Shared_UID/',
    # CoNetDataPro('/share/scratch/fengyuan/Projects/RecSys/CDRS/Data/Amazon/Unique_UI/',
    #              flag=True,topk=False,negsa=False,
    #              # domain1='Books', domain2='Electronics')
    #              domain1='Amazon_Instant_Video', domain2='Musical_Instruments')

    domainp1 = '/share/scratch/fengyuan/Projects/RecSys/CDRS/CoNet/Data/books_small/'
    domainp2 = '/share/scratch/fengyuan/Projects/RecSys/CDRS/CoNet/Data/elec_small/'
    #
    # data1 = ul.loadData(domainp1 + 'train_neg_1.csv', names=['uid', 'iid', 'ratings'])
    # data2 = ul.loadData(domainp2 + 'train_neg_1.csv', names=['uid', 'iid', 'ratings'])
    # merged = Merge_Domains(data1, data2)
    # merged.to_csv(domainp1+'merged_neg_1.csv', index=False, header=False)
    #
    # data1 = ul.loadData(domainp1 + 'train_neg_4.csv', names=['uid', 'iid', 'ratings'])
    # data2 = ul.loadData(domainp2 + 'train_neg_4.csv', names=['uid', 'iid', 'ratings'])
    # merged = Merge_Domains(data1, data2)
    # merged.to_csv(domainp1+'merged_neg_4.csv', index=False, header=False)
    #
    # data1 = ul.loadData(domainp1 + 'test_topk.csv', names=['uid', 'iid', 'ratings'])
    # data2 = ul.loadData(domainp2 + 'test_topk.csv', names=['uid', 'iid', 'ratings'])
    # merged = Merge_Domains(data1, data2)
    # merged.to_csv(domainp1 + 'merged_topk.csv', index=False, header=False)
    data1 = ul.loadData(domainp1 + 'original.csv', names=['uid', 'iid', 'ratings'])
    data2 = ul.loadData(domainp2 + 'original.csv', names=['uid', 'iid', 'ratings'])
    merged = Merge_Domains(data1, data2)
    merged.to_csv(domainp1 + 'merged_original.csv', index=False, header=False)