# This program is used to preprocess the Amazon product review dataset
# import sys,os
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

import pandas as pd
# import Utils.GenUtils as gtl
import gzip

# Parse the compressed data file
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

# Read the file into a pandas dataframe
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# Read specific columns into a pandas dataframe
def getSlicedDF(path, chunksize = None, columns = ['reviewerID','asin','overall','unixReviewTime']):
    if chunksize is not None:
        # Read the raw data by chunks
        res = pd.DataFrame([])
        print("Raw Data is being read in chunks...\n")
        for d in pd.read_json(path,compression='gzip',lines=True,chunksize=chunksize):
            res = res.append(d.loc[:,columns])
        return res
    else:
        # Read the entire raw data file into memory (may cause memory overflow)
        print("Raw Data is being read in all...\n")
        i = 0
        df = {}
        for d in parse(path):
            df[i] = dict((k,d[k]) for k in columns)
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

# Store the UID, IID, Ratings, UnixTime tuples into a CSV file
def Raw2CSV(source_path, target_path):
    df = getSlicedDF(source_path, chunksize=10**6, columns = ['reviewerID','asin','overall','unixReviewTime'])
    df.to_csv(target_path, index=False, header=False)

# Calculate the statics in each category
def Stats(path, chuncksize = 10**6):
    # Read in the category file
    categorylist = pd.read_csv(path + 'Category.csv', names=['category'], engine='python')['category'].tolist()

    # Dictionary to generate the final statistics
    df = {}

    # Loop for each category
    for i in range(len(categorylist)):
        # Select the target category
        category = categorylist[i]

        # Read the data into the memory
        res = pd.DataFrame([])
        for ratings in pd.read_csv(path+'Category/'+category+'.csv', names=['uid', 'iid', 'rating', 'time'], chunksize=chuncksize):
            res = res.append(ratings.loc[:, ['uid', 'iid']], ignore_index=True)

        # Number of ratings per user and item
        n_ratings_per_user = res.groupby(['uid'])[['iid']].count()
        n_ratings_per_user.columns=['number_of_ratings']
        # n_ratings_per_user.to_csv(path+'Stats/user_'+category+'.csv',index=True, header=True)

        n_ratings_per_item = res.groupby(['iid'])[['uid']].count()
        n_ratings_per_item.columns = ['number_of_ratings']
        # n_ratings_per_item.to_csv(path+'Stats/item_'+category+'.csv',index=True, header=True)

        print("The number of distinct users: {0} and total ratings: {1} for category: {2}".format(n_ratings_per_user.shape[0], res.shape[0], category))
        print("The number of distinct items: {0} and total ratings: {1} for category: {2}".format(n_ratings_per_item.shape[0], res.shape[0], category))

        # Write into the key of corresponding genre
        df[category] = [res.shape[0], n_ratings_per_user.shape[0], n_ratings_per_item.shape[0]]

    # Write the final results into one csv file
    stat = pd.DataFrame.from_dict(df, orient='index', columns=['number_of_ratings','number_of_users','number_of_items'])
    print(stat)
    stat.to_csv(path + 'Stats/Stats_Final.csv')

# Calculate shared users between two different domains
def CrossDomainStats(spath, tpath, domain1, domain2, sel):
    if sel == 'user':
        # Summarize the shared users in the two nominated domains

        # Read the user data into the memory
        user_data1 = pd.read_csv(spath+'Stats/user_'+domain1+'.csv', names=['uid1', 'n_count'],skiprows=[0])
        user_data2 = pd.read_csv(spath+'Stats/user_'+domain2+'.csv', names=['uid2', 'n_count'],skiprows=[0])
        item_data1 = pd.read_csv(spath+'Stats/item_'+domain1+'.csv', names=['iid1', 'n_count'],skiprows=[0])
        item_data2 = pd.read_csv(spath+'Stats/item_'+domain2+'.csv', names=['iid2', 'n_count'],skiprows=[0])

        # Find the shared UIDs in the two domains
        user_merged = user_data1.merge(user_data2,left_on='uid1',right_on='uid2',how='outer',indicator=True)
        item_merged = item_data1.merge(item_data2,left_on='iid1',right_on='iid2',how='outer',indicator=True)

        shared_uid = user_merged.loc[user_merged['_merge'] == 'both', 'uid1'].tolist()
        shared_iid = item_merged.loc[item_merged['_merge'] == 'both', 'iid1'].tolist()
        print("Shared Number of UIDs and IIDs: {0} and {1}.".format(len(shared_uid),len(shared_iid)))
        print("Shared IIDs:{0}".format(shared_iid))

        unique_uid1 = user_merged.loc[user_merged['_merge'] == 'left_only',['uid1']]
        unique_uid2 = user_merged.loc[user_merged['_merge'] == 'right_only',['uid2']]
        unique_iid1 = item_merged.loc[item_merged['_merge'] == 'left_only',['iid1']]
        unique_iid2 = item_merged.loc[item_merged['_merge'] == 'right_only',['iid2']]

        # print(unique_uid2)

        # Extract the ratings of the unique UIDs in the two domains
        data1 = pd.read_csv(spath+'Category/'+domain1+'.csv', names=['uid', 'iid', 'rating', 'time'], skiprows=[0])
        data1 = data1.loc[data1['uid'].isin(unique_uid1.loc[:,'uid1'].tolist())]
        # if unique_iid1.shape[0] != 0:
        data1 = data1.loc[data1['iid'].isin(unique_iid1.loc[:,'iid1'].tolist())]

        data2 = pd.read_csv(spath+'Category/'+domain2+'.csv', names=['uid', 'iid', 'rating', 'time'], skiprows=[0])
        data2 = data2.loc[data2['uid'].isin(unique_uid2.loc[:,'uid2'].tolist())]
        # if unique_iid2.shape[0] != 0:
        data2 = data2.loc[data2['iid'].isin(unique_iid2.loc[:,'iid2'].tolist())]

        # Calculate the items in each of the domains
        num_uid1 = data1['uid'].unique().shape[0]
        num_iid1 = data1['iid'].unique().shape[0]
        num_uid2 = data2['uid'].unique().shape[0]
        num_iid2 = data2['iid'].unique().shape[0]
        num_ratings1 = data1.shape[0]
        num_ratings2 = data2.shape[0]

        # Write the shared ratings in the two domains into the target files
        print("The number of users in {0} is: {1}".format(domain1, num_uid1))
        print("The number of users in {0} is: {1}".format(domain2, num_uid2))
        print("The number of items in {0} is: {1}".format(domain1, num_iid1))
        print("The number of items in {0} is: {1}".format(domain2, num_iid2))
        print("The number of ratings in {0} is: {1}".format(domain1, num_ratings1))
        print("The number of ratings in {0} is: {1}".format(domain2, num_ratings2))


        # unique_uid1.to_csv(tpath + domain1 + '_and_' + domain2 + '_uid1.csv',index=False, header=False)
        # unique_uid2.to_csv(tpath + domain1 + '_and_' + domain2 + '_uid2.csv', index=False, header=False)
        # unique_iid1.to_csv(tpath + domain1 + '_and_' + domain2 + '_iid1.csv', index=False, header=False)
        # unique_iid2.to_csv(tpath + domain1 + '_and_' + domain2 + '_iid2.csv', index=False, header=False)
        # #
        # data1.to_csv(tpath + domain1 + '_and_' + domain2 + '_ratings_1.csv', index=False, header=False)
        # data2.to_csv(tpath + domain1 + '_and_' + domain2 + '_ratings_2.csv', index=False, header=False)

    elif sel == 'item':
        # Summarize the shared items in the two nominated domains

        return 0

    elif sel == 'none':
        # Summarize the part which both users and items do not overlap

        return 0


if __name__ == "__main__":
    # Change to CSV
    # Raw2CSV('/media/data/Workspace/DataSets/Amazon/Raw/Category/reviews_Amazon_Instant_Video.json.gz',
    #         '/media/data/Workspace/DataSets/Amazon/Raw/Ratings/Category/Amazon_Instant_Video.csv')

    # Testing
    # df = getSlicedDF('/media/data/Workspace/DataSets/Amazon/Complete/reviews_Musical_Instruments.json.gz',chunksize=10**5)
    # print(df.loc[:,:])

    # Statistics for all categories
    # Stats('/media/data/Workspace/DataSets/Amazon/Raw/Ratings/')

    # # Cross Domain Statistics
    CrossDomainStats(
                     # spath='/media/data/Workspace/DataSets/Amazon/Raw/Ratings/',
                     tpath = '/share/scratch/fengyuan/Projects/RecSys/CDRS/Data/Amazon/Unique_UI/',
                     # tpath='/media/data/Workspace/PhD_Projects/CDRS/Data/Amazon/Unique_UI/',
                     spath = '/share/scratch/fengyuan/Data/Amazon/Raw/Ratings/',
                     domain1='Amazon_Instant_Video', domain2='Musical_Instruments',
                     # domain1='Books', domain2='Electronics',
                     sel='user')