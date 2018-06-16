'''
Load and process the data for two domains
Output sparse matrices
'''
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# Load only the original matrix
def load_original_matrix(datafile, header=['uid','iid','ratings'], sep=','):
    df = pd.read_csv(datafile, names=header, sep=sep, engine='python')

    if len(header) > 3:
        df = df.drop(columns=[*header[3:]])

    num_users = df[header[0]].unique().shape[0]
    num_items = df[header[1]].unique().shape[0]

    row = df.loc[:, header[0]].tolist()
    col = df.loc[:, header[1]].tolist()
    element = df.loc[:, header[2]].tolist()

    if np.min(row) == 1:
        row = (np.array(row) - 1).tolist()
    if np.min(col) == 1:
        col = (np.array(col) - 1).tolist()

    original_matrix = csr_matrix((element, (row, col)), shape=(num_users, num_items))

    return original_matrix, num_users, num_items

# Split train and test data using matrix as input
def matrix_split(matrix, opt='ranking', mode='df', n_item_per_user=1, test_size=0.2, seed=None):
    num_user, num_item = matrix.shape
    if opt == 'ranking' and mode == 'df': # This is faster
        matrix = matrix.tocoo()
        row, col, data = matrix.row, matrix.col, matrix.data
        arr = np.column_stack((row,col,data))
        train = pd.DataFrame(arr,columns=['uid','iid','ratings'])

        test = pd.DataFrame([])
        for _ in range(n_item_per_user): # Sample N times randomly for each user
            tmp = train.sample(frac=1).drop_duplicates(['uid'], keep='last') # Shuffle the dataframe and drop the last one from the duplicated uids
            test = test.append(tmp)
            train = train.drop(index=tmp.index)

        test = test.sort_values(by=['uid'])
        train = train.sort_values(by=['uid'])

        row = train.loc[:, 'uid'].tolist()
        col = train.loc[:, 'iid'].tolist()
        element = train.loc[:, 'ratings'].tolist()
        train_matrix = csr_matrix((element, (row, col)), shape=matrix.shape)

        row = test.loc[:, 'uid'].tolist()
        col = test.loc[:, 'iid'].tolist()
        element = test.loc[:, 'ratings'].tolist()
        test_matrix = csr_matrix((element, (row, col)), shape=matrix.shape)

        return train_matrix, test_matrix

    elif opt == 'ranking' and mode =='mat': # Using matrix method has to loop all the users causing it to be very slow
        matrix = matrix.tocsr()
        test_row, test_col, test_data = [], [], []
        train_row, train_col, train_data = [], [], []
        for row in range(num_user):  # Randomly select n items for each user
            nonzeros = matrix.getrow(row).nonzero()[1]
            test_idx = list(np.random.choice(nonzeros, n_item_per_user)) # Randomly select n items from the nonzero values
            train_idx = [iid for iid in nonzeros if iid not in test_idx]

            test_row.extend([row]*n_item_per_user)
            test_col.extend(test_idx)
            test_data.extend(matrix[row, test_idx].data.tolist())

            train_row.extend([row] * len(train_idx))
            train_col.extend(train_idx)
            train_data.extend(matrix[row, train_idx].data.tolist())

        test_matrix = csr_matrix((test_data, (test_row, test_col)), shape=matrix.shape)
        train_matrix = csr_matrix((train_data, (train_row, train_col)), shape=matrix.shape)
        return train_matrix, test_matrix

    elif opt == 'prediction':
        matrix = matrix.tocoo()
        row, col, data = matrix.row, matrix.col, matrix.data
        arr = np.column_stack((row,col,data))

        train, test = train_test_split(arr, test_size=test_size, random_state=seed) # Split data randomly

        train_row, train_col, train_data = train[:,0],train[:,1],train[:,2]
        test_row, test_col, test_data = test[:,0],test[:,1],test[:,2]

        train_matrix = csr_matrix((train_data,(train_row,train_col)),shape=matrix.shape)
        test_matrix = csr_matrix((test_data,(test_row,test_col)),shape=matrix.shape)
        return train_matrix,test_matrix

    else:
        return [],[]

def matrix_cross_val(matrix, n_splits, seed = None):
    '''
    Cross validation on each user row for n_splits
    :param matrix: original matrix as input
    :param n_splits: n-fold cross-validation
    :param seed: the random state to control the shuffle
    :return:
    '''
    num_user, num_item = matrix.shape
    matrix = matrix.tocsr()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Generate different kFold splitting generators for each row
    kf_dict, pos_dict = {}, {}
    for u in range(num_user):
        pos_dict[u] = matrix.getrow(u).nonzero()[1]
        kf_dict[u] = kf.split(pos_dict[u])

    # For each train-test splitting, generate a pair of train-test sparse matrix
    for _ in range(n_splits):
        test_row, test_col, test_data = [], [], []
        train_row, train_col, train_data = [], [], []

        # Loop through all users to generate the train and test matrices
        for u in range(num_user):
            train_idx, test_idx = next(kf_dict[u])
            train_iid, test_iid = pos_dict[u][train_idx], pos_dict[u][test_idx]

            train_row.extend([u] * len(train_iid))
            train_col.extend(train_iid)
            train_data.extend(matrix[u, train_iid].data.tolist())

            test_row.extend([u] * len(test_iid))
            test_col.extend(test_iid)
            test_data.extend(matrix[u, test_iid].data.tolist())

        train_matrix = csr_matrix((train_data, (train_row, train_col)), shape=(num_user, num_item))
        test_matrix = csr_matrix((test_data, (test_row, test_col)), shape=(num_user, num_item))

        yield  train_matrix, test_matrix

# Get the negative samples for each user and generate the ranking list for testing
def negdict_mat(org_matrix, test_matrix, num_neg=0):
    test_matrix = test_matrix.tocsr()
    num_users, num_items = org_matrix.shape
    all_items = set(range(num_items))

    neg_dict, ranking_dict, test_dict = {},{},{}
    for u in range(num_users):
        # Negative Dict for User u
        neg_list_for_u = list(all_items - set(list(org_matrix.getrow(u).nonzero()[1])))
        neg_dict[u] = neg_list_for_u
        # Test Dict for User u
        test_nz_iid = list(test_matrix.getrow(u).nonzero()[1]) # Nonzero index of row u
        test_nz_data = list(test_matrix.getrow(u).data) # Nonzero data of row u
        test_dict[u] = dict(zip(test_nz_iid, test_nz_data))
        # Ranking Dict for User u
        if num_neg == -1:
            ranking_dict[u] = neg_list_for_u + test_nz_iid # Put the item ids to be ranked in the end of the ranking list
        else:
            ranking_dict[u] = list(np.random.choice(neg_list_for_u, num_neg)) + test_nz_iid

    return neg_dict, ranking_dict, test_dict

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

# Get all the explicit and implict ratings for a matrix except for those testing data in the exp_matrix
# This is particular useful for generating the training data for WRMF
def get_full_matrix(matrix, exp_matrix=None):
    num_user, num_item = matrix.shape
    if exp_matrix == None:
        arr = matrix.toarray()
        res_user, res_item, res_rating = [], [], []
        item_coo = list(range(num_item))
        for i in range(num_user):
            res_user.extend([i]*num_item)
            res_item.extend(item_coo)
            res_rating.extend(arr[i,:])
        return res_user, res_item, res_rating
    else:
        exp_matrix = exp_matrix.todok()
        exp_keys = exp_matrix.keys()
        res_user, res_item, res_rating = [],[],[]
        for uid, iid in np.ndindex(num_user,num_item): # Loop the entire matrix (very slow)
            if (uid,iid) in exp_keys:
                continue
            else:
                res_user.append(uid)
                res_item.append(iid)
                res_rating.append(matrix[uid,iid])
        return res_user, res_item, res_rating

# Upsample the data in list format
def data_upsample_list(user_list, item_list, rating_list, num_ext = 0):
    user_array, item_array, rating_array = np.array(user_list), np.array(item_list), np.array(rating_list)
    idxs = list(np.random.choice(range(len(user_list)), num_ext))
    return list(np.append(user_array, user_array[idxs])), list(np.append(item_array, item_array[idxs])), list(np.append(rating_array, rating_array[idxs]))

def matrix_to_list(matrix):
    coo_mat = matrix.tocoo()
    return list(coo_mat.row), list(coo_mat.col), list(coo_mat.data)

# Change the rating matrix into a binary one
def matrix_to_binary(matrix, threshold):
    matrix = matrix.tocsr()
    matrix.data = np.where(matrix.data > threshold, 1, 0)
    matrix.eliminate_zeros()
    return matrix

# Leave values greate than a threshold in the matrix
def matrix_theshold(matrix, threshold):
    matrix = matrix.tocsr()
    matrix.data = np.where(matrix.data > threshold, matrix.data, 0)
    matrix.eliminate_zeros()
    return matrix

def list_to_binary(ls,threshold):
    return [1 if ele > threshold else 0 for ele in ls]

def list_threshold(ls, threshold):
    return [ele if ele > threshold else 0 for ele in ls]

def list_zero_pruning(ls):
    return [ele for ele in ls if ele > 0]

########################################################################################################################
''' Deprecated Functions '''

def load_as_matrix(datafile, header=['uid','iid','ratings'], sep=',', opt='leave-one-out', test_size=0.2, seed=0):
    '''
    This function combines loading as matrix with splitting the test and training data
    If we want to process the original data before splitting or using a different method to split the original matrix,
    then this function will not be suitable.
    This function will be deprecated. Use load_original_matrix instead
    :param datafile:
    :param header:
    :param sep:
    :param opt:
    :param test_size:
    :param seed:
    :return:
    '''
    df = pd.read_csv(datafile, names=header, sep=sep, engine='python')

    if len(header) > 3:
        df = df.drop(columns=[*header[3:]])

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

    if np.min(row) == 1:
        row = (np.array(row) - 1).tolist()
    if np.min(col) == 1:
        col = (np.array(col) - 1).tolist()

    original_matrix = csr_matrix((element, (row, col)), shape=(num_users, num_items))

    row = train.loc[:, header[0]].tolist()
    col = train.loc[:, header[1]].tolist()
    element = train.loc[:, header[2]].tolist()
    if np.min(row) == 1:
        row = (np.array(row) - 1).tolist()
    if np.min(col) == 1:
        col = (np.array(col) - 1).tolist()
    train_matrix = csr_matrix((element, (row, col)), shape=(num_users, num_items))

    row = test.loc[:, header[0]].tolist()
    col = test.loc[:, header[1]].tolist()
    element = test.loc[:, header[2]].tolist()
    if np.min(row) == 1:
        row = (np.array(row) - 1).tolist()
    if np.min(col) == 1:
        col = (np.array(col) - 1).tolist()
    test_matrix = csr_matrix((element, (row, col)), shape=(num_users, num_items))

    return original_matrix, train_matrix, test_matrix, num_users, num_items

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

########################################################################################################################

# if __name__ == "__main__":
#     org_mat1, tr_mat1, ts_mat1, num_users1, num_items1 = load_as_matrix('Data/books_small/original.csv')
#     # org_mat2, tr_mat2, ts_mat2, tdic_mat2, num_users2, num_items2 = load_as_matrix('Data/elec_small/original.csv')
#     # tr_mat1 = negative_sample(org_mat1,tr_mat1,ts_mat1,num_neg=1,neg_val=0)
#     # print(tr_mat1.nnz)
#     userlist,_,_ = negative_sample_mat(org_mat1,tr_mat1,ts_mat1,num_neg=1,neg_val=0)
#     print(len(userlist))
