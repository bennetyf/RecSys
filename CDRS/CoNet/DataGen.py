'''
This program is used to preprocess the data for CoNet
'''
import RecUtils as ul

def CoNetDataPro(spath, flag, topk, negsa, domain1 = 'Amazon_Instant_Video', domain2 = 'Musical_Instruments'):
    domainp1 = 'domain1/'
    domainp2 = 'domain2/'

    if flag == True: # Recalculating from the original data
    ################ Load the original data, filter and splitting #####################
        print("Loading Data for Domain {0}".format(domain1))
        data1,_ = ul.loadData(spath + domain1 + '_and_' + domain2 + '_ratings_1.csv',
                                     names=['uid','iid','ratings','time'])
        data1 = data1.drop(labels=['time'],axis=1)
        data1 = ul.mat2Bin(data1,threshold=3)
        data1 = ul.uiFilter(data1,opt='user',threshold=1)

        print("Loading Data for Domain {0}".format(domain2))
        data2, _ = ul.loadData(spath + domain1 + '_and_' + domain2 + '_ratings_2.csv',
                                  names=['uid', 'iid', 'ratings', 'time'])
        data2 = data2.drop(labels=['time'], axis=1)
        data2 = ul.mat2Bin(data2, threshold=3)
        data2 = ul.uiFilter(data2, opt='user', threshold=1)
        print("Data Loaded")

        # Generating the Shared Users
        data1, data2, _ = ul.filterBySharedUsers(data1, data2)

        # Turn into Number Coding
        data1, data2 = ul.id2Num(data1), ul.id2Num(data2)

        data1.to_csv(domainp1 + 'original.csv', index=False, header=False)
        data2.to_csv(domainp2 + 'original.csv', index=False, header=False)

        # Splitting Training and Testing Data
        train1, test1 = ul.dataSplit(data1, opt='leave-one-out')
        train1.to_csv(domainp1 + 'train.csv', index=False, header=False)
        test1.to_csv(domainp1 + 'test.csv', index=False, header=False)

        train2, test2 = ul.dataSplit(data2, opt='leave-one-out')
        train2.to_csv(domainp2 + 'train.csv', index=False, header=False)
        test2.to_csv(domainp2 + 'test.csv', index=False, header=False)
    else:
        # Load the data from the disk
        print("Loading Data for Domain {0}".format(domain1))
        data1, _ = ul.loadData(domainp1 + 'original.csv', names=['uid', 'iid', 'ratings'])
        train1, _ = ul.loadData(domainp1 + 'train.csv', names=['uid', 'iid', 'ratings'])
        test1, _ = ul.loadData(domainp1 + 'test.csv', names=['uid', 'iid', 'ratings'])

        print("Loading Data for Domain {0}".format(domain2))
        data2, _ = ul.loadData(domainp2 + 'original.csv', names=['uid', 'iid', 'ratings'])
        train2, _ = ul.loadData(domainp2 + 'train.csv', names=['uid', 'iid', 'ratings'])
        test2, _ = ul.loadData(domainp2 + 'test.csv', names=['uid', 'iid', 'ratings'])
        print("Loading Completed")

    if topk:
        # Generating the TopK Testing Data
        print("Generating Negative Samples for TopK Testing, domain {0}".format(domain1))
        ul.negSample(data1, test1, 99, neg_val=0, maxlines=10**7, method='direct', mod='topk',
                     store=True,store_path=domainp1,fname='test_topk.csv')

        print("Generating Negative Samples for TopK Testing, domain {0}".format(domain2))
        ul.negSample(data2, test2, 99, neg_val=0, maxlines=10**7, method='direct', mod='topk',
                     store=True, store_path=domainp2, fname='test_topk.csv')

    if negsa:
        # Generate Training Data with negative sampling
        ul.negSample(data1, train1, 1, neg_val=0, maxlines=10**7, method='direct', mod='train',
                     store=True, store_path=domainp1, fname='train_neg_1.csv')
        ul.negSample(data2, train2, 1, neg_val=0, maxlines=10 ** 7, method='direct', mod='train',
                 store=True, store_path=domainp2, fname='train_neg_1.csv')
        # print(np.expand_dims(np.asarray(neg),axis=2))







if __name__ == "__main__":
    # Generate the CoNet Data
    CoNetDataPro('/media/work/Workspace/PhD_Projects/CDRS/Data/Amazon/Shared_UID/',
                 flag=False,topk=True,negsa=False,
                 # domain1='Books', domain2='Electronics')
                 domain1='Amazon_Instant_Video',domain2='Musical_Instruments')