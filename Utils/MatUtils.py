'''
Load and process the data for two domains
Output sparse matrices
'''
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, KFold

# Load only the original matrix
def load_original_matrix(datafile, header=['uid','iid','ratings'], sep=','):
    '''
    Load the original dataset as sparse matrix
    :param datafile: The path of the datafile
    :param header: The header of the datafile
    :param sep: The separator of the data
    :return: csr sparse matrix and the number fo users and items
    '''

    df = pd.read_csv(datafile, names=header, sep=sep, engine='python')

    # Drop the other columns
    if len(header) > 3:
        df = df.drop(columns=[*header[3:]])

    # Note that do not use the following code to calculate the number of users and items
    # It is possible that some users may rate 0 items and some items have been rated 0 times
    # In this case, the number of unique users/items is different from the number of users/items
    # num_users, num_items = df[header[0]].unique().shape[0], df[header[1]].unique().shape[0]

    # Get the row, col and data of the pandas dataframe
    row = df.loc[:, header[0]].tolist()
    col = df.loc[:, header[1]].tolist()
    element = df.loc[:, header[2]].tolist()

    # Make the uid and iid start from 0
    if np.min(row) != 0:
        row = (np.array(row) - np.min(row)).tolist()
    if np.min(col) != 0:
        col = (np.array(col) - np.min(col)).tolist()

    num_users, num_items = np.max(row)+1, np.max(col)+1

    # Generate the matrix
    original_matrix = csr_matrix((element, (row, col)), shape=(num_users, num_items))
    return original_matrix.tolil()

# Split train and test data using matrix as input
def matrix_split(matrix, opt='ranking', mode='df', n_item_per_user=1, test_size=0.2, seed=None):
    num_user, num_item = matrix.shape

    if opt == 'ranking' and mode == 'df': # This is faster (Use pandas to select the n testing elements for each user)
        # Generate the pandas dataframe
        matrix = matrix.tocoo()
        arr = np.column_stack((matrix.row, matrix.col, matrix.data))
        train = pd.DataFrame(arr,columns=['uid','iid','ratings'])

        # Leave-N-out method
        test = pd.DataFrame([])
        for _ in range(n_item_per_user): # Sample N times randomly for each user
            tmp = train.sample(frac=1).drop_duplicates(['uid'], keep='last') # Shuffle the dataframe and drop the last one from the duplicated uids
            test = test.append(tmp)
            train = train.drop(index=tmp.index)

        # Sort by the user ids
        test = test.sort_values(by=['uid'])
        train = train.sort_values(by=['uid'])

        # Generate the train and test matrix
        row = train.loc[:, 'uid'].tolist()
        col = train.loc[:, 'iid'].tolist()
        element = train.loc[:, 'ratings'].tolist()
        train_matrix = csr_matrix((element, (row, col)), shape=matrix.shape)

        row = test.loc[:, 'uid'].tolist()
        col = test.loc[:, 'iid'].tolist()
        element = test.loc[:, 'ratings'].tolist()
        test_matrix = csr_matrix((element, (row, col)), shape=matrix.shape)
        return train_matrix.tolil(), test_matrix.tolil()

    elif opt == 'ranking' and mode =='mat': # Using matrix method has to loop all the users causing it to be very slow
        matrix = matrix.tolil()
        test_row, test_col, test_data = [], [], []
        train_row, train_col, train_data = [], [], []

        # Loop through the users
        for uid, (iids, ratings) in enumerate(zip(matrix.rows, matrix.data)): # Choose from index instead of from the iids directly
            test_idx = list(np.random.choice(range(len(iids)), n_item_per_user, replace=False)) # Randomly select n items from the nonzero values
            train_idx = [idx for idx in range(len(iids)) if idx not in test_idx]

            test_row.extend([uid] * len(test_idx))
            test_col.extend([iids[i] for i in test_idx])
            test_data.extend([ratings[i] for i in test_idx])

            train_row.extend([uid] * len(train_idx))
            train_col.extend([iids[i] for i in train_idx])
            train_data.extend([ratings[i] for i in train_idx])

        test_matrix = csr_matrix((test_data, (test_row, test_col)), shape=matrix.shape)
        train_matrix = csr_matrix((train_data, (train_row, train_col)), shape=matrix.shape)
        return train_matrix.tolil(), test_matrix.tolil()

    # Prediction (Randomly splitting the dataset)
    elif opt == 'prediction' and mode == 'all':
        matrix = matrix.tocoo()
        arr = np.column_stack((matrix.row, matrix.col, matrix.data))

        train, test = train_test_split(arr, test_size=test_size, random_state=seed) # Split data randomly

        train_matrix = csr_matrix((train[:,2],(train[:,0],train[:,1])),shape=matrix.shape)
        test_matrix = csr_matrix((test[:,2],(test[:,0],test[:,1])),shape=matrix.shape)
        return train_matrix.tolil(), test_matrix.tolil()

    elif opt == 'prediction' and mode == 'user':
        matrix = matrix.tolil()
        test_row, test_col, test_data = [], [], []
        train_row, train_col, train_data = [], [], []

        # Loop through the users
        for uid, (iids, ratings) in enumerate(zip(matrix.rows, matrix.data)):
            train_idx, test_idx = train_test_split(range(len(iids)), test_size=test_size, random_state=seed)  # Split data randomly

            test_row.extend([uid] * len(test_idx))
            test_col.extend([iids[i] for i in test_idx])
            test_data.extend([ratings[i] for i in test_idx])

            train_row.extend([uid] * len(train_idx))
            train_col.extend([iids[i] for i in train_idx])
            train_data.extend([ratings[i] for i in train_idx])

        test_matrix = csr_matrix((test_data, (test_row, test_col)), shape=matrix.shape)
        train_matrix = csr_matrix((train_data, (train_row, train_col)), shape=matrix.shape)
        return train_matrix.tolil(), test_matrix.tolil()

    else:
        return [],[]

def matrix_cross_validation(matrix, n_splits, seed = None):
    '''
    Cross validation on each user row for n_splits
    :param matrix: original matrix as input
    :param n_splits: n-fold cross-validation
    :param seed: the random state to control the shuffle
    :return:
    '''
    matrix = matrix.tolil()
    num_user, num_item = matrix.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Generate different kFold splitting generators for each row
    kf_dict = {}
    for uid, nonzeros in enumerate(matrix.rows):
        kf_dict[uid] = kf.split(nonzeros)

    # For each train-test splitting, generate a pair of train-test sparse matrix
    for _ in range(n_splits):
        test_row, test_col, test_data = [], [], []
        train_row, train_col, train_data = [], [], []

        # Loop through all users to generate the train and test matrices
        for uid, (iids, ratings) in enumerate(zip(matrix.rows, matrix.data)):
            train_idx, test_idx = next(kf_dict[uid])

            train_row.extend([uid] * len(train_idx))
            train_col.extend([iids[i] for i in train_idx])
            train_data.extend([ratings[i] for i in train_idx])

            test_row.extend([uid] * len(test_idx))
            test_col.extend([iids[i] for i in test_idx])
            test_data.extend([ratings[i] for i in test_idx])

        train_matrix = csr_matrix((train_data, (train_row, train_col)), shape=(num_user, num_item))
        test_matrix = csr_matrix((test_data, (test_row, test_col)), shape=(num_user, num_item))

        yield  train_matrix.tolil(), test_matrix.tolil()

# Generate the negative dictionaries
def negdict_mat(original_matrix, test_matrix, num_neg=0):
    '''
    Get the negative samples for each user
    Generate the ranking list for testing
    Generate the testing dictionary for ranking metrics calculation (a dictionary of iids and ratings)
    :param original_matrix: The input matrix
    :param test_matrix:
    :param num_neg:
    :return:
    '''
    original_matrix = original_matrix.tolil()
    test_matrix = test_matrix.tolil()

    num_users, num_items = original_matrix.shape

    neg_dict, ranking_dict, test_dict = {},{},{}
    for uid, (org_iids, test_iids, test_ratings) in enumerate(zip(original_matrix.rows, test_matrix.rows, test_matrix.data)):
        # Negative Dict for User u
        # neg_list_for_u = [iid for iid in range(num_items) if iid not in org_iids]
        neg_list_for_u = list(set(range(num_items))-set(org_iids)) #This is faster than list comprehension
        neg_dict[uid] = neg_list_for_u

        # Test Dict for User u
        test_dict[uid] = dict(zip(test_iids, test_ratings))

        # Ranking Dict for User u
        if num_neg == -1:
            ranking_dict[uid] = neg_list_for_u + test_iids # Put the item ids to be ranked in the end of the ranking list
        else:
            ranking_dict[uid] = list(np.random.choice(neg_list_for_u, num_neg)) + test_iids

    return neg_dict, ranking_dict, test_dict

# Input the negative dictionary and u,i,r lists, return the three of the negative sampled lists
def negative_sample_list(neg_dict, user_list, item_list, rating_list, num_neg=0, neg_val=0):
    '''
    Negative sampling: sample the user, item and rating lists with randomly picked negative samples
    Inputs and outputs must be lists, because duplicated elements are taken into consideration in this function
    (can not use sparse matrix)
    Return three lists
    :param neg_dict:
    :param user_list:
    :param item_list:
    :param rating_list:
    :param num_neg:
    :param neg_val:
    :return:
    '''
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
def get_full_matrix(matrix, exp_matrix = None, opt='fast'):
    '''
    Calculate the full matrix of a sparse matrix and return the three lists of uid, iid and ratings
    exp_matrix is the matrix to be excluded from the entire training matrix (usually the testing matrix)
    :param matrix:
    :param exp_matrix:
    :return:
    '''
    num_user, num_item = matrix.shape
    if exp_matrix == None:
        arr = matrix.toarray()
        res_user, res_item, res_rating = [], [], []
        item_coo = list(range(num_item))
        for uid, row in enumerate(map(list,arr)):
            res_user.extend([uid]*num_item)
            res_item.extend(item_coo)
            res_rating.extend(arr)
        return res_user, res_item, res_rating
    else:
        res_user, res_item, res_rating = [], [], []
        if opt == 'fast': # This way is much faster than the slow one
            exp_matrix = exp_matrix.tolil()
            matrix = matrix.tolil()
            for uid, (exp_iids, iids, ratings) in enumerate(zip(exp_matrix.rows, matrix.rows, matrix.data)):
                # tmp = [iid for iid in range(num_item) if iid not in exp_iids] # Using set is faster than list comprehension
                tmp = list(set(range(num_item))-set(exp_iids))
                res_user.extend([uid]*len(tmp))
                res_item.extend(tmp)
                res_rating.extend([ratings[iids.index(ele)] if ele in iids else 0 for ele in tmp])
        if opt == 'slow':
            exp_matrix = exp_matrix.todok()
            exp_keys = exp_matrix.keys()
            matrix = matrix.todok()
            for uid, iid in np.ndindex(num_user,num_item): # Loop the entire matrix (very slow)
                if (uid,iid) in exp_keys: # If the coordinate is in the testing matrix, then ignore this entry
                    continue
                else:
                    res_user.append(uid)
                    res_item.append(iid)
                    res_rating.append(matrix[uid,iid])
        return res_user, res_item, res_rating

# Upsample the data in list format
def data_upsample_list(user_list, item_list, rating_list, num_ext = 0):
    user_array, item_array, rating_array = np.array(user_list), np.array(item_list), np.array(rating_list)

    if len(user_list) < num_ext:
        idxs = list(np.random.choice(range(len(user_list)), num_ext, replace = True)) # Allow replacement(Important)
    else:
        idxs = list(np.random.choice(range(len(user_list)), num_ext, replace = False))

    return list(np.append(user_array, user_array[idxs])), list(np.append(item_array, item_array[idxs])), list(np.append(rating_array, rating_array[idxs]))

# Change a sparse matrix into lists of uid, iid and ratings
def matrix_to_list(matrix):
    coo_mat = matrix.tocoo()
    return list(coo_mat.row), list(coo_mat.col), list(coo_mat.data)

# Change the rating matrix into a binary one
def matrix_to_binary(matrix, threshold):
    matrix = matrix.tocsr()
    matrix.data = np.where(matrix.data > threshold, 1, 0)
    matrix.eliminate_zeros()
    return matrix.tolil()

# Leave values greate than a threshold in the matrix
def matrix_theshold(matrix, threshold):
    matrix = matrix.tocsr()
    matrix.data = np.where(matrix.data > threshold, matrix.data, 0)
    matrix.eliminate_zeros()
    return matrix.tolil()

def list_to_binary(ls,threshold):
    return [1 if ele > threshold else 0 for ele in ls]

def list_threshold(ls, threshold):
    return [ele if ele > threshold else 0 for ele in ls]

def list_zero_pruning(ls):
    return [ele for ele in ls if ele > 0]

def array_to_matrix(arr):
    mat = csr_matrix(arr)
    mat.eliminate_zeros()
    return mat.tolil()

# Convert a sparse matrix into rows and store it into a dictionary
# Actually, this is useless, because we can directly use toarray method and feed the array into the model
def matrix_to_vectors(matrix,opt='row'):
    num_user, num_item = matrix.shape
    vec_dict = {}
    if opt == 'row':
        # matrix = matrix.tolil()
        # for uid, (iids, ratings) in enumerate(zip(matrix.rows, matrix.data)): # Store the row vector for each user
        #     vec_dict[uid] = [ratings[iids.index(ele)] if ele in iids else 0 for ele in range(len(num_item))]
        matrix = matrix.tocsr()
        for uid in range(len(num_user)):
            vec_dict[uid] = matrix.getrow(uid).toarray().flatten().tolist()

    if opt == 'col':
        matrix = matrix.tocsc()
        for iid in range(len(num_item)):
            vec_dict[iid] = matrix.getcol(iid).getcol(iid).toarray().flatten().tolist()

    return vec_dict


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
