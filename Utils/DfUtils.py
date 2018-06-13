'''
This document is used to load the data as pandas dataframe and process it in the dataframe format
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_as_df(datafile, header=['uid','iid','ratings'], sep=',', opt='leave-one-out', test_size=0.2, seed=0):

    df = pd.read_csv(datafile, header=None, names=header, sep=sep, engine='python')

    num_users = df[header[0]].unique().shape[0]
    num_items = df[header[1]].unique().shape[0]

    if opt == 'leave-one-out':
        # Generate the test and training data
        test_df = df.drop_duplicates([header[0]], keep='last')
        train_df = df.drop(index=test_df.index)

        # Re-indexing the dataframe
        test_df = test_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
    else:
        # Randomly split the data into training and testing
        train_df, test_df = train_test_split(np.asarray(df), test_size=test_size, random_state=seed)
        train_df = pd.DataFrame(train_df, columns=[header[0], header[1], header[2]])
        test_df = pd.DataFrame(test_df, columns=[header[0], header[1], header[2]])

    return df, train_df, test_df, num_users, num_items

def negdict_df(original_df,test_df, num_neg=0):
    UID, IID, RATE = list(original_df)
    grouped_df = original_df.groupby([UID])
    num_items = original_df[IID].unique().shape[0]
    all_items = set(range(num_items))
    neg_dict, test_dict = {}, {}
    for uid, group in grouped_df:
        pos_item_list = group.loc[:,IID].tolist()
        neg_item_list = list(all_items - set(pos_item_list))
        neg_dict[uid] = neg_item_list
        if num_neg == -1:
            test_dict[uid] = neg_item_list + [test_df.loc[uid,IID]]
        else:
            neg_item_list = list(np.random.choice(neg_item_list, num_neg))
            test_dict[uid] = neg_item_list + [test_df.loc[uid,IID]]
    return neg_dict, test_dict

def negative_sample_df(original_df, neg_dict, train_df, num_neg = 0 , neg_val = 0):
    if num_neg == 0:
        return train_df
    UID, IID, RATE = list(original_df)
    res_user, res_item, res_rating = [], [], []
    for line in train_df.itertuples(index=False):
        u, i, r = line
        res_user.extend([u] * (num_neg + 1))  # extend is faster than a loop of append
        res_rating.append(r)
        res_rating.extend([neg_val] * num_neg)
        res_item.append(i)
        res_item.extend(list(np.random.choice(neg_dict[u], num_neg)))
    return pd.DataFrame({UID:res_user,IID:res_item,RATE:res_rating},columns=[UID,IID,RATE])

def data_upsample_df(df1, df2):
    length1, length2 = df1.shape[0], df2.shape[0]
    if length1 < length2:
        df1 = df1.append(df1.sample(n=length2-length1,axis=0),ignore_index=True)
    if length1 > length2:
        df2 = df2.append(df2.sample(n=length1-length2, axis=0),ignore_index=True)
    return df1, df2

def df_to_list(df):
    UID, IID, RATE = list(df)
    user_list, item_list, rating_list = df.loc[:,UID].tolist(), df.loc[:,IID].tolist(), df.loc[:,RATE].tolist()
    return user_list, item_list, rating_list

######################################################################################################################
# if __name__ == "__main__":
#     org_df1, tr_df1, ts_df1, num_users1, num_items1 = load_as_df('../CDRS/CoNet/Data/books_small/original.csv')
#     # org_df2, tr_df2, ts_df2, tdic_df2, num_users2, num_items2 = load_as_df('../CDRS/CoNet/Data/elec_small/original.csv')
#     # print(tdic_df1[100])
#     ui_dict1 = negdict_df(org_df1,ts_df1,num_neg=99)
#     # ui_dict2 = get_user_item_dict(org_df2)
#     # print(ui_dict1[12])
#     # tr_df1, tr_df2 = data_upsample(tr_df1,tr_df2)
#     # print(tr_df1.shape, tr_df2.shape)
#     tdict_df1_temp = negative_sample_df(org_df1,ui_dict1,tr_df1,ts_df1,neg_val=0)
#     # neg_df1.to_csv('tmp.csv',index=False, header=False)
#     print(len(tdict_df1_temp[0]), tdict_df1_temp[0])
#
#     # ul1, il1, rl1 = df_to_list(ts_df1)
#     # print(len(ul1))

