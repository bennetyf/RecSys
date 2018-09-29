import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')
import pandas as pd

import Utils.RecUtils as rtl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

datapath_src = '/share/scratch/fengyuan/Projects/RecSys/Data/' + 'FilmTrust/ratings.txt'
datapath_save = '/share/scratch/fengyuan/Projects/RecSys/Data/' + 'Test.csv'
# datapath_mat = 'Data/gb-hr-5.mat'
########################################################################################################################
# df = pd.read_csv(datapath_src, names=['uid','iid','ratings','time'], sep='::', engine='python')

# df = pd.read_csv(datapath_src, names=['uid','iid','gid','rid','ratings','time'], sep=',', engine='python')

df = pd.read_csv(datapath_src, names=['iid','uid','ratings'], sep=' ', engine='python')

# df = df.loc[:,['uid','iid','ratings']]

df = df.reindex(columns=['uid', 'iid', 'ratings']).sort_values(by=['uid']).reset_index(drop=True)

filtered_df = df
# filtered_df = df.loc[df['ratings'].isin([4])]
# filtered_df = df[df['ratings'] >= 5]
# filtered_df = filtered_df[filtered_df.groupby('uid').uid.transform(len) > 1]
# filtered_df = rtl.id2Num(filtered_df)

# filtered_df['ratings'] = 1
# filtered_df = filtered_df.drop_duplicates()

# print(filtered_df.duplicated())

# filtered_df.to_csv(datapath_save, index=False, header=False)

num_user = filtered_df['uid'].unique().shape[0]
num_item = filtered_df['iid'].unique().shape[0]

print(num_user, num_item, filtered_df.shape[0])
print(max(filtered_df['uid']), max(filtered_df['iid']))

########################################################################################################################
print("Saving into Train and Test Matrices....")

# original_matrix \
#             = mtl.load_original_matrix(datafile=datapath_save, header=['uid', 'iid', 'ratings'], sep=',')
# original_matrix = mtl.matrix_to_binary(original_matrix,0)
# train_matrix, test_matrix \
#         = mtl.matrix_split(original_matrix, opt='ranking', mode='mat', n_item_per_user=1, random_state=10)
    # train_matrix, test_matrix \
    #     = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.2, random_state=10)

    # original_matrix = gtl.load_mat_as_matrix('Data/Ciao.mat', opt='coo')['rating']

# gtl.matrix_to_mat(datapath_mat, opt='coo', original=original_matrix, train=train_matrix, test=test_matrix)
