import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl
import Utils.ModUtils as mod

import numpy as np

path_dict = {'ml1m-full':   'Data/ml1m-hr-full.mat',
             'ml1m-345':    'Data/ml1m-hr-345.mat',
             'ml1m-45':     'Data/ml1m-hr-45.mat',
             'ml1m-5':      'Data/ml1m-hr-5.mat',
             'ciao-full':   'Data/ciao-hr-full.mat',
             'ciao-345':    'Data/ciao-hr-345.mat',
             'ciao-45':     'Data/ciao-hr-45.mat',
             'ciao-5':      'Data/ciao-hr-5.mat',
             'douban-full': 'Data/douban-hr-full.mat',
             'douban-345':  'Data/douban-hr-345.mat',
             'douban-45':   'Data/douban-hr-45.mat',
             'douban-5':    'Data/douban-hr-5.mat',
             'filmtrust-full':  'Data/filmtrust-hr-full.mat',
             'filmtrust-34':   'Data/filmtrust-hr-34.mat',
             'filmtrust-4':     'Data/filmtrust-hr-4.mat',
             'flixster-full':   'Data/flixster-hr-full.mat',
             'flixster-345':    'Data/flixster-hr-345.mat',
             'flixster-45':     'Data/flixster-hr-45.mat',
             'flixster-5':      'Data/flixster-hr-5.mat',
             'flixster-old':    'Data/Flixster_Rank_HR.mat',
             'ymov-full':       'Data/ymov-hr-full.mat',
             'ymov-345':        'Data/ymov-hr-345.mat',
             'ymov-45':         'Data/ymov-hr-45.mat',
             'ymov-5':          'Data/ymov-hr-5.mat',
             'ymus-full':       'Data/ymus-hr-full.mat',
             'ymus-345':        'Data/ymus-hr-345.mat',
             'ymus-45':         'Data/ymus-hr-45.mat',
             'ymus-5':          'Data/ymus-hr-5.mat',
}

dataset = 'ciao-45'
path = path_dict[dataset]

date ='20180801'
filename='itempop.mat'

print('Loading Data From {0}'.format(path))
data = gtl.load_mat_as_matrix(path, opt='coo')
original_matrix, train_matrix, test_matrix = data['original'], data['train'], data['test']
print('Users:{0}, Items:{1}, Ratings:{2}'.format(original_matrix.shape[0], original_matrix.shape[1], original_matrix.nnz))

num_user, num_item = original_matrix.shape[0], original_matrix.shape[1]

topK=[5,10]
total_hr, total_ndcg = np.zeros(len(topK)), np.zeros(len(topK))


tr_mat = train_matrix.transpose()
item_pop_dict = {}
item_pop_list = []
for item, ratings in enumerate(tr_mat.data):
    item_pop_dict[item] = len(ratings)
    item_pop_list.append(len(ratings))
item_pop_arr = np.asarray(item_pop_list)

_, ranking_dict, test_dict = mtl.negdict_mat(original_matrix, test_matrix, num_neg=199, mod='others', random_state=10)

for user in ranking_dict:

    if len(test_dict[user]) == 0:
        continue

    iid = ranking_dict[user]  # The ranking item ids for user u
    rk = item_pop_arr[np.asarray(iid)]
    print(rk)
    hr, ndcg = evl.rankingMetrics(rk, iid, topK, test_dict[user], mod='hr')
    total_hr += hr
    total_ndcg += ndcg

avg_hr, avg_ndcg = total_hr / num_user, total_ndcg / num_user

for i in range(len(topK)):
    print('-' * 55)
    print("[HR@{0}] {1}".format(topK[i], avg_hr[i]))
    print("[nDCG@{0}] {1}".format(topK[i], avg_ndcg[i]))
print('=' * 55)

save_path = "Result/%s/ItemPop/%s/" % (dataset, date)
if not os.path.exists(save_path):
    os.makedirs(save_path)

gtl.array_to_mat(save_path+filename,HR5=avg_hr[0], NDCG5=avg_ndcg[0], HR10=avg_hr[1], NDCG10=avg_ndcg[1])
