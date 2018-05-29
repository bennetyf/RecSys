'''
This document contains general utility functions for data pre-processing in recommender systems
'''
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import gc

########################################### Global Variables ###########################################################
# This program heavily uses pandas dataframe, so first we need to set the names of the columns as global parameters

_UID = 'uid'
_IID = 'iid'
_RATE = 'ratings'

########################################### System Operations ##########################################################
# Get the column names of the input pandas dataframe
def getColNames(data):
    global _UID,_IID,_RATE
    _UID, _IID, _RATE = data.columns.values

# Convert bytes into other units
def convertFromBytes(num, unit='MB'):
    divider = {
        "B":    1.0,
        "KB":   1024.0,
        "MB":   1024.0**2,
        "GB":   1024.0**3,
        "TB":   1024.0**4
    }.get(unit, 1024.0**2)
    return num/divider

# Return the size of a specific file on the disk
def fileSize(file_path):
    if os.path.isfile(file_path):
        return convertFromBytes(os.stat(file_path).st_size)

# Merge a list of files
def mergeFiles(target_file, filelist):
    # Always write into a new file
    if os.path.exists(target_file):
        os.remove(target_file)
    fout = open(target_file, "a")
    # now the rest:
    for i in range(len(filelist)):
        f = open(filelist[i])
        for line in f:
            fout.write(line)
        f.close()  # not really needed
    fout.close()

# Delete a list of files
def deleteFiles(filelist):
    for i in range(len(filelist)):
        os.remove(filelist[i])

########################################### Basic Data Operations ######################################################
# Load data as a pandas dataframe
def loadData(datafile, names = ['uid','iid','ratings'], chunksize=10**8):
    '''
    This function loads data as a pandas dataframe.
    If succeeded, return true, else return false for only loading a chunk of the data
    '''
    # Check whether the file exists
    assert os.path.exists(datafile)

    # If the datafile size is larger than 1GB
    if fileSize(datafile) > 1024:
        print("The file size is larger than 1GB, loading a chunk of it.")
        return pd.read_csv(datafile, header=None, names=names, engine='python', chunksize=chunksize), False
    else:
        # Read all the CSV file
        return pd.read_csv(datafile, header=None, names=names, engine='python'), True

# Convert the a list data into sparse matrix
def arr2Mat(data, matrix_type = 'dok'):
    n_user, n_item = np.max(data,0)[:2]

    if matrix_type == 'dok':
        mat = sp.dok_matrix((n_user+1,n_item+1),dtype=np.int32)
        # Read in all the elements
        for i in range(data.shape[0]):
            uid, iid, ratings = data[i,:]
            mat[uid, iid] = ratings

    elif matrix_type == 'csr':
        row,col,ratings = data[:, 0], data[:, 1], data[:, 2]
        mat = sp.csr_matrix((ratings,(row,col)),shape=[n_user+1, n_item+1])

    elif matrix_type == 'csc':
        row, col, ratings = data[:, 0], data[:, 1], data[:, 2]
        mat = sp.csc_matrix((ratings,(row,col)),shape=[n_user+1, n_item+1])

    else:
        return []

    return mat

########################################### Functions for UI Matrix ####################################################

# Convert original ratings into binary matrix
def mat2Bin(data, threshold=0):
    # Read the user data into the memory and map the ratings into binary
    # Construct the mapping dictionary
    map_dict = {
        0: {1:1, 2:1, 3:1, 4:1, 5:1},
        1: {1:0, 2:1, 3:1, 4:1, 5:1},
        2: {1:0, 2:0, 3:1, 4:1, 5:1},
        3: {1:0, 2:0, 3:0, 4:1, 5:1},
        4: {1:0, 2:0, 3:0, 4:0, 5:1},
        5: {1:0, 2:0, 3:0, 4:0, 5:0},
    }.get(threshold,{})
    data.loc[:, _RATE] = data.loc[:, _RATE].map(map_dict)
    return data

# Convert the non-numerical UIDs and IIDs to numbers
def id2Num(data):
    # Covert user ids into numbers
    uids = pd.DataFrame(list(data.groupby([_UID]).groups.keys()), columns=[_UID])
    uids = uids.reset_index().set_index([_UID]).to_dict()
    # Generate a mapping dictionary from index to column values
    uids = uids['index']
    data.loc[:,_UID] = data.loc[:,_UID].map(uids)

    # Covert item ids into numbers
    iids = pd.DataFrame(list(data.groupby([_IID]).groups.keys()), columns=[_IID])
    iids = iids.reset_index().set_index([_IID]).to_dict()
    # Generate a mapping dictionary from index to column values
    iids = iids['index']
    data.loc[:,_IID] = data.loc[:,_IID].map(iids)
    return data

# Filter out those users or items whose number of ratings are below a specific threshold
def uiFilter(data, opt = None, threshold = 0):
    if opt == 'user':
        # Filter the ratings data by the number of ratings in each user
        # (make sure there is no cold-start problem when training)
        n_ratings_per_user = data.groupby([_UID])[[_IID]].count()
        n_ratings_per_user.columns = ['number_of_ratings']
        filtered = n_ratings_per_user.loc[n_ratings_per_user['number_of_ratings'] > threshold]

        # Use the index to filter out the desired data
        data.index = data[_UID]
        data = data.loc[list(filtered.index.values)]
        data = data.reset_index(drop=True)

        # Change the code into numbers
        return data
    elif opt == 'item':
        # Filter the ratings data by the number of ratings in each item
        # (make sure there is no cold-start problem when training)
        n_ratings_per_item = data.groupby([_IID])[[_UID]].count()
        n_ratings_per_item.columns = ['number_of_ratings']
        filtered = n_ratings_per_item.loc[n_ratings_per_item['number_of_ratings'] > threshold]

        # Use the index to filter out the desired data
        data.index = data[_IID]
        data = data.loc[list(filtered.index.values)]
        data = data.reset_index(drop=True)

        # Change the code into numbers
        return data
    else:
        return data

# Split the data for one domain into training and test
def dataSplit(data, opt=None, train_ratio=None, seed=None):
    '''
    Split the input pandas dataframe into training and test datasets
    :param data: pandas dataframe
    :param opt: splitting methods
    :param train_ratio: in random splitting, specify the ratio of the training data
    :return: two pandas dataframe containing the training and testing data
    '''

    # Leave-one-out method
    if opt == 'leave-one-out':
        # Generate the test and training data
        test = data.drop_duplicates([_UID], keep='last')
        train = data.drop(index=test.index)

        # Re-indexing the dataframe
        test = test.reset_index(drop=True)
        train = train.reset_index(drop=True)
        return train, test

    # Randomly splitting
    if opt == 'random':
        # Randomly split the data into training and testing
        train, test = train_test_split(np.asarray(data), train_size=train_ratio, random_state=seed)
        return pd.DataFrame(train, columns=[_UID,_IID,_RATE]), pd.DataFrame(test, columns=[_UID,_IID,_RATE])

    else:
        print("DataSplit: Methods wrong!")
        return [],[]

# Negative sampling that generates a list of data containing the mixture of positive and negative samples
def negSample(source, target, num_neg = 0, maxlines = 10**7, neg_val = 0, method = 'direct', mod='train',
              store = False, store_path = None, fname = None):
    '''
    :param source:  The whole dataset(U-I matrix) in the dataframe format
    :param target:  The positive dataset to be interpolated
    :param num_neg: The number of negative samples to be generated
    :param store:   Flag to decide whether to store the generated data or not
    :param store_path: The path to store the generated data
    :return: The pandas dataframe
    '''

    # Sampling Methodology Flag
    option = {
        'direct':1,
        'trial-error':2
    }.get(method,0)

    mode = {
        'train':1,
        'topk':2
    }.get(mod,0)

    # Un-implemented Method
    if option == 0:
        print("[RecUtils.negSample]: Wrong Method")
        return pd.DataFrame([])

    # Using pandas dataframe to generate the negative samples systematically
    if option == 1:
        # Get the entire item list (including all the items)
        iids = pd.DataFrame(list(source.groupby([_IID]).groups.keys()), columns=[_IID])
        # Data Too Large
        flag = False
        if target.shape[0]*(num_neg+1) > maxlines:
            if not store:
                print("[RecUtils.negSample]:Data too large: Data has to be stored on disk.")
                return pd.DataFrame([])
            else:
                flag = True
        # Loop for each test user-item pair
        print("[RecUtils.negSample]: Using method {0}. Entering the Main Sampling Loop.".format(method))
        res, file_names, loop = pd.DataFrame([]), [], 0
        for i in range(target.shape[0]):
            if mode == 1: # For generating the training dataset
                res = res.append(target.loc[[i],:], ignore_index=True)
            # Get the target testing user-item pair
            uid, iid = tuple(target.loc[i, [_UID, _IID]].tolist())
            # Select the items that have been interacted by the test user
            item_interacted = source.loc[source[_UID] == uid, _IID].tolist()
            # Drop those items that have interaction with the test user
            iid_list = iids.drop(index=iids.loc[iids[_IID].isin(item_interacted), :].index)
            # Re-indexing (Prepare for random negative sampling)
            iid_list = iid_list.reset_index(drop=True)
            # Randomly select the negative items for user id of uid
            item_idx = np.random.randint(0, iid_list.shape[0] - 1, size = num_neg).tolist()
            item_neg = iid_list.loc[item_idx, :]
            item_neg.insert(0, column=_UID, value=[uid]*num_neg)
            item_neg.insert(2, column=_RATE, value=[neg_val]*num_neg)
            # Append the df into the result
            res = res.append(item_neg,ignore_index=True)
            if mode == 2: # For generating the topk testing dataset
                res = res.append(target.loc[[i], :], ignore_index=True)

            if flag:
                # The following is for generation of very large datasets
                batch = int(maxlines / (num_neg + 1))
                if (i+1) % batch == 0: # Write into file every number of maxlines
                    file_name = store_path + 'negsa_temp_'+ str(loop) +'.csv'
                    file_names.append(file_name)
                    res.to_csv(file_name, index=False, header=False)
                    res = pd.DataFrame([])
                    gc.collect()
                    loop += 1
                    print("[RecUtils.negSample]: Storing File : {0}".format(file_name))

        print("[RecUtils.negSample]: Using Method {0}. Leaving the Main Sampling Loop.".format(method))

        # The Last Piece of Data
        if flag:
            # Store the last piece of data
            file_name = store_path + 'negsa_temp_'+ str(loop) +'.csv'
            file_names.append(file_name)
            res.to_csv(file_name, index=False, header=False)

            # Merge all the files into one CSV file
            print("[RecUtils.negSample]: Using Method {0}. Merging Files.".format(method))
            mergeFiles(store_path + fname, file_names)
            deleteFiles(file_names)

            # Return something after processing
            return res

        # For Small Data
        else:
            if not store: # Do not store the generated data
                return res
            else:   # Store the generated data
                res.to_csv(store_path + fname, index=False, header=False)
                return res
    ######################################################################################################
    # Heuristically generate the data (using trial and error method)
    # Don't use this method, it is very slow indeed (Too many loops)
    if option == 2:
        # Find the number of items in all the data
        num_item = source.max(axis=0)[_IID]
        # Convert the target data into a sparse matrix
        spmat = arr2Mat(np.asarray(target), matrix_type='dok')
        # Data Too Large
        flag = False
        if target.shape[0] * (num_neg + 1) > maxlines:
            if not store:
                print("[RecUtils.negSample]: Data too large: Data has to be stored on disk.")
                return pd.DataFrame([])
            else:
                flag = True

        # Loop in the target sparse matrix
        print("[RecUtils.negSample]: Using Method {0}. Entering the Main Sampling Loop.".format(method))
        res, file_names, i, loop = pd.DataFrame([]), [], 0, 0
        for (uid, iid) in spmat.keys():
            res = res.append(target.loc[[i],:], ignore_index=True)
            for j in range(num_neg): # Generate num_neg of negative samples
                # Randomly choose an item number (This can work because the U-I matrix is sparse)
                trial = np.random.randint(num_item)
                # Make sure the generated U-I pair does not exist in the target dataset
                while (uid, trial) in spmat.keys():
                    trial = np.random.randint(num_item)
                # Append the dataframe
                res = res.append(pd.DataFrame([[uid, trial, neg_val]],columns=[_UID,_IID,_RATE]), ignore_index=True)
            i += 1

            if flag:
                # The following is for generation of very large datasets
                batch = int(maxlines / (num_neg + 1))
                if i % batch == 0:
                    file_name = store_path + 'negsa_temp_'+ str(loop) +'.csv'
                    file_names.append(file_name)
                    res.to_csv(file_name, index=False, header=False)
                    res = pd.DataFrame([])
                    gc.collect()
                    loop += 1
                    print("[RecUtils.negSample]: Storing File : {0}".format(file_name))

        print("[RecUtils.negSample]: Using Method {0}. Leaving the Main Sampling Loop.".format(method))

        # After leaving the loop
        if flag: # Large data
            # Store the last piece of data
            file_name = store_path + 'negsa_temp_' + str(loop) + '.csv'
            file_names.append(file_name)
            res.to_csv(file_name, index=False, header=False)

            # Merge all the files into one CSV file
            print("[RecUtils.negSample]: Using Method {0}. Merging Files.".format(method))
            mergeFiles(store_path + fname, file_names)
            deleteFiles(file_names)
            return res

        else: # Small data
            # Check whether we should save the file to hard drive or not
            if not store:
                return res
            else:
                res.to_csv(store_path + fname, index=False, header=False)
                return res

# This function is for filtering the data by shared users (also returns
def filterBySharedUsers(data1, data2):

    # Find the shared UIDs
    merged = data1.merge(data2, left_on=_UID, right_on=_UID, how='outer', indicator=True)
    shared_uid = merged.loc[merged['_merge'] == 'both', [_UID]]
    shared_uid = shared_uid.drop_duplicates([_UID], keep='last')
    shared_uid = shared_uid.reset_index(drop=True)

    # Extract the ratings of the shared UIDs in the two domains
    res1 = data1.loc[data1[_UID].isin(shared_uid[_UID].tolist())].sort_values(by=[_UID])
    res2 = data2.loc[data2[_UID].isin(shared_uid[_UID].tolist())].sort_values(by=[_UID])

    return res1, res2, shared_uid

# This function is used to get the number of users and items in the U-I matrix
def getUINum(filepath):
    assert os.path.exists(filepath)
    df, sig = loadData(filepath,names=[_UID,_IID,_RATE])
    # Make sure all the data is loaded into memory
    assert sig == True
    # Get the max values in each column
    return df.max(axis=0)[['uid', 'iid']] + 1

