'''
Load and process the data for two domains
Output sparse matrices
'''

import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from sklearn.model_selection import train_test_split

def load_as_matrix(datafile, header=['uid','iid','ratings'], sep=',', opt='leave-one-out', test_size=0.2, seed=0):
    df = pd.read_csv(datafile, names=header, sep=sep, engine='python')

    num_users = df[header[0]].unique().shape[0]
    num_items = df[header[1]].unique().shape[0]

    if opt == 'leave-one-out':
        # Generate the test and training data
        test = df.drop_duplicates([header[0]], keep='last')
        train = df.drop(index=test.index)

        # Re-indexing the dataframe
        test = test.reset_index(drop=True)
        train = train.reset_index(drop=True)
    else:
        # Randomly split the data into training and testing
        train, test = train_test_split(np.asarray(df), test_size=test_size, random_state=seed)
        train = pd.DataFrame(train, columns=[header[0], header[1], header[2]])
        test = pd.DataFrame(test, columns=[header[0], header[1], header[2]])

    row = df.loc[:, header[0]].tolist()
    col = df.loc[:, header[1]].tolist()
    element = df.loc[:, header[2]].tolist()
    original_matrix = csr_matrix((element, (row, col)), shape=(num_users, num_items))

    row = train.loc[:, header[0]].tolist()
    col = train.loc[:, header[1]].tolist()
    element = train.loc[:, header[2]].tolist()
    train_matrix = csr_matrix((element, (row, col)), shape=(num_users, num_items))

    row = test.loc[:, header[0]].tolist()
    col = test.loc[:, header[1]].tolist()
    element = test.loc[:, header[2]].tolist()
    test_matrix = csr_matrix((element, (row, col)), shape=(num_users, num_items))

    # test_dict={}
    # for u in range(num_users):
    #     test_dict[u] = list(test_matrix.getrow(u).nonzero()[1])

    return original_matrix, train_matrix, test_matrix, num_users, num_items

def negdict_mat(org_matrix, test_matrix, num_neg=0):
    num_users, num_items = org_matrix.shape
    all_items = set(range(num_items))
    neg_u_dict, test_dict = {},{}
    for u in range(num_users):
        neg_item_list = list(all_items - set(list(org_matrix.getrow(u).nonzero()[1])))
        neg_u_dict[u] = neg_item_list
        if num_neg == -1:
            test_dict[u] = neg_item_list + list(test_matrix.getrow(u).nonzero()[1])
        else:
            neg_item_list = list(np.random.choice(neg_item_list, num_neg))
            test_dict[u] = neg_item_list + list(test_matrix.getrow(u).nonzero()[1])
    return neg_u_dict, test_dict

# Input the negative dictionary and u,i,r lists, return the three of the negative sampled lists
def negative_sample_list(neg_dict, user_list, item_list, rating_list, num_neg=0, neg_val=0):
    if num_neg == 0:
        return user_list, item_list, rating_list
    res_user, res_item, res_rating = [], [], []
    for u, i, r in list(zip(user_list, item_list, rating_list)):
        res_user.extend([u] * (num_neg + 1))    # extend is faster than a loop of append
        res_rating.append(r)
        res_rating.extend([neg_val]*num_neg)
        res_item.append(i)
        res_item.extend(list(np.random.choice(neg_dict[u],num_neg)))
    return res_user, res_item, res_rating

def data_upsample_list(user_list, item_list, rating_list, num_ext = 0):
    user_array, item_array, rating_array = np.array(user_list), np.array(item_list), np.array(rating_list)
    idxs = list(np.random.choice(range(len(user_list)), num_ext))
    return list(np.append(user_array, user_array[idxs])), list(np.append(item_array, item_array[idxs])), list(np.append(rating_array, rating_array[idxs]))

def matrix_to_list(matrix):
    coo_mat = matrix.tocoo()
    return list(coo_mat.row), list(coo_mat.col), list(coo_mat.data)

########################################################################################################################

def negative_sample_mat(org_matrix, train_matrix=None, test_matrix=None, num_neg = 0 , neg_ratio = 0, neg_val = 0 , opt='train'):
    # Input matrices are all csr matrices
    if neg_ratio == 0 and num_neg == 0:
        return
    num_users, num_items = org_matrix.shape
    all_items = set(range(num_items))
    if opt == 'train':
        user_list, item_list, rating_list = [],[],[]
        for u in range(num_users):
            items_for_u = list(org_matrix.getrow(u).nonzero()[1])
            neg_item_list = list(all_items - set(items_for_u))
            items_for_u = list(train_matrix.getrow(u).nonzero()[1])

            if neg_ratio != 0:
                n_negs_for_u = int(neg_ratio * len(neg_item_list))
            else:
                n_negs_for_u = len(items_for_u) * num_neg

            neg_item_list = list(np.random.choice(neg_item_list,n_negs_for_u))

            # Regenerate the train matrix with extra negative samples as training data
            user_list = user_list + [u] * (len(items_for_u) + n_negs_for_u)
            item_list = item_list + items_for_u + neg_item_list
            rating_list = rating_list + [train_matrix[u,j] for j in items_for_u] + [neg_val] * n_negs_for_u
        return user_list, item_list, rating_list

    if opt == 'test':
        test_dict={}
        for u in range(num_users):
            items_for_u = list(org_matrix.getrow(u).nonzero()[1])
            neg_item_list = list(all_items - set(items_for_u))
            neg_item_list = list(np.random.choice(neg_item_list,num_neg))

            test_dict[u] = neg_item_list + list(test_matrix.getrow(u).nonzero()[1])
        return test_dict

# if __name__ == "__main__":
#     org_mat1, tr_mat1, ts_mat1, num_users1, num_items1 = load_as_matrix('Data/books_small/original.csv')
#     # org_mat2, tr_mat2, ts_mat2, tdic_mat2, num_users2, num_items2 = load_as_matrix('Data/elec_small/original.csv')
#     # tr_mat1 = negative_sample(org_mat1,tr_mat1,ts_mat1,num_neg=1,neg_val=0)
#     # print(tr_mat1.nnz)
#     userlist,_,_ = negative_sample_mat(org_mat1,tr_mat1,ts_mat1,num_neg=1,neg_val=0)
#     print(len(userlist))
