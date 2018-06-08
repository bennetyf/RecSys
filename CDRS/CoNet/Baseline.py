import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))
#
import pandas as pd
# #
# from surprise import Reader
# from surprise import Dataset
# from surprise import SVD
# from surprise import accuracy
#
import Utils.RecEval as evl

# train1 = pd.read_csv('Data/domain1/train.csv',names=['uid','iid','ratings'])
test1 = pd.read_csv('Data/domain2/test_topk.csv',names=['uid','iid','ratings'])
#
# reader = Reader(rating_scale=(0, 1.0))
#
# train1_data = Dataset.load_from_df(train1, reader)
# test1_data = Dataset.load_from_df(test1, reader)
#
# train1_data = train1_data.build_full_trainset()
# # test1_data = test1_data.build_full_trainset()
# # for i in train1_data.all_ratings():
# #     print(i)
#
# algo = SVD(n_factors=150)
#
# algo.fit(train1_data)
#
# hr = 0
# ndcg = 0
# n_batches = 0
#
# # print(test1.loc[0,'ratings'])
# # print(algo.predict(str(test1.loc[0,'uid']), str(test1.loc[0,'iid']), r_ui=test1.loc[0,'ratings']).est)
# # print(test1.shape[0]/100)
# for i in range(int(test1.shape[0]/100)):
#     preds=[]
#     items=[]
#     for j in range(100):
#         preds.append(algo.predict(str(test1.loc[100*i+j,'uid']), str(test1.loc[100*i+j,'iid']), r_ui=test1.loc[100*i+j,'ratings']+1).est)
#         items.append(test1.loc[100*i+j,'iid'])
#     _, hr_tmp, ndcg_tmp, _ = evl.evalTopK(preds,items,10)
#
#     hr += hr_tmp
#     ndcg += ndcg_tmp
#     n_batches += 1
#     if n_batches == 1:
#         print("local hr {0} and ndcg {1}".format(hr/n_batches,ndcg/n_batches))
#         print(preds)
#
# print("Average HR {0} and NDCG {1}".format(hr/n_batches,ndcg/n_batches))

# LibFM Results
predictions = pd.read_csv('Data/domain2/dom2_pred',header=None, names=['ratings'])
hr = 0
ndcg = 0
n_batches = 0

for i in range(int(predictions.shape[0]/100)):
    preds=[]
    items=[]
    for j in range(100):
        preds.append(predictions.loc[100*i+j,'ratings'])
        items.append(test1.loc[100*i+j,'iid'])
    _, hr_tmp, ndcg_tmp, _ = evl.evalTopK(preds,items,10)

    hr += hr_tmp
    ndcg += ndcg_tmp
    n_batches += 1

print("Average HR {0} and NDCG {1}".format(hr/n_batches,ndcg/n_batches))