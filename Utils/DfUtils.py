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

    test_dict = {}
    for line in test_df.itertuples(index=False):
        test_dict[line[0]] = [line[1]]

    return df, train_df, test_df, test_dict, num_users, num_items


def get_user_item_dict(original_df):
    UID, IID, RATE = list(original_df)
    grouped_df = original_df.groupby([UID])
    item_dict={}
    for name, group in grouped_df:
        item_dict[name] = group.loc[:,IID].tolist()
    return item_dict


def negative_sample_df(original_df, user_item_dict, train_df=None, test_df=None, num_neg = 0 , neg_ratio = 0, neg_val = 0 , opt='train'):
    if neg_ratio == 0 and num_neg == 0:
        return
    UID, IID, RATE = list(original_df)

    num_users, num_items = original_df[UID].unique().shape[0], original_df[IID].unique().shape[0]
    # all_items = pd.DataFrame(list(original_df.groupby([IID]).groups.keys()), columns=[IID])
    all_items = set(range(num_items))
    if opt == 'train':
        res = []
        # grouped = train_df.groupby([UID])
        # user_list, item_list, rating_list = [],[],[]
        # for uid, group in grouped:
        for line in train_df.itertuples(index=False):
            uid , iid, rating = line
            # Append the row under consideration
            # res_df = res_df.append(train_df.loc[[i], :], ignore_index=True)

            items_for_uid = user_item_dict[uid]
            neg_items_for_uid = list(all_items - set(items_for_uid))
            # Drop those items that have interaction with the user of uid
            # neg_items_for_uid = all_items.drop(index=all_items.loc[all_items[IID].isin(items_for_uid), :].index)

            if neg_ratio != 0:
                num_negs_for_uid = int(neg_ratio * len(neg_items_for_uid))
            else:
                # num_negs_for_uid = num_neg * len(group)
                num_negs_for_uid = num_neg

            # Sample the negative item dataframe for uid
            neg_items_for_uid = list(np.random.choice(neg_items_for_uid, num_negs_for_uid))
            tmp = np.array([[uid] * num_negs_for_uid, neg_items_for_uid, [neg_val] * num_negs_for_uid]).transpose().tolist()
            res.append([uid, iid ,rating])
            for u, i, r in tmp:
                res.append([u,i,r])

            # user_list = user_list + [uid] * (len(group) + num_negs_for_uid)
            # item_list = item_list + group.loc[:,IID].tolist() + neg_items_for_uid
            # rating_list = rating_list + group.loc[:,RATE].tolist() + [neg_val] * num_negs_for_uid

        # res = np.array([user_list,item_list,rating_list]).transpose()
        return pd.DataFrame(res,columns=[UID,IID,RATE])

    if opt=='test':
        test_dict = {}
        for line in test_df.itertuples(index=False):
            uid, iid, _ = line
            items_for_uid = user_item_dict[uid]
            neg_item_list = list(all_items - set(items_for_uid))
            neg_item_list = list(np.random.choice(neg_item_list, num_neg))

            test_dict[uid] = neg_item_list + [iid]
        return test_dict


def data_upsample(df1, df2):
    length1, length2 = df1.shape[0], df2.shape[0]
    if length1 < length2:
        df1 = df1.append(df1.sample(n=length2-length1,axis=0),ignore_index=True)
    if length1 > length2:
        df2 = df2.append(df2.sample(n=length1-length2, axis=0), ignore_index=True)
    return df1, df2


def df_to_list(df):
    UID, IID, RATE = list(df)
    user_list, item_list, rating_list = df.loc[:,UID].tolist(), df.loc[:,IID].tolist(), df.loc[:,RATE].tolist()
    return user_list, item_list, rating_list

######################################################################################################################
if __name__ == "__main__":
    org_df1, tr_df1, ts_df1, tdic_df1, num_users1, num_items1 = load_as_df('../CDRS/CoNet/Data/books_small/original.csv')
    # org_df2, tr_df2, ts_df2, tdic_df2, num_users2, num_items2 = load_as_df('../CDRS/CoNet/Data/elec_small/original.csv')
    # print(tdic_df1[100])
    ui_dict1 = get_user_item_dict(org_df1)
    # ui_dict2 = get_user_item_dict(org_df2)
    # print(ui_dict1[12])
    # tr_df1, tr_df2 = data_upsample(tr_df1,tr_df2)
    # print(tr_df1.shape, tr_df2.shape)
    tdict_df1_temp = negative_sample(org_df1,ui_dict1,tr_df1,ts_df1,num_neg=99,neg_val=0,opt='test')
    # neg_df1.to_csv('tmp.csv',index=False, header=False)
    print(len(tdict_df1_temp[0]), tdict_df1_temp[0])

    # ul1, il1, rl1 = df_to_list(ts_df1)
    # print(len(ul1))

