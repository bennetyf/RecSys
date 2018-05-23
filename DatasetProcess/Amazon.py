# This program is used to preprocess the Amazon product review dataset

import pandas as pd
import numpy as np
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
    # print(df.shape)
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
def CrossDomainStats(spath, tpath, domain1, domain2, sel, chunksize=10**6):
    if sel == 'user':
        # Summarize the shared users in the two nominated domains

        # Read the user data into the memory
        res1 = pd.DataFrame([])
        for df in pd.read_csv(spath+'Stats/user_'+domain1+'.csv', names=['uid1', 'iid'], chunksize=chunksize, skiprows=[0]):
            res1 = res1.append(df[['uid1']], ignore_index=True)
        # res1.columns=['uid1']
        res2 = pd.DataFrame([])
        for df in pd.read_csv(spath+'Stats/user_'+domain2+'.csv', names=['uid2', 'iid'], chunksize=chunksize, skiprows=[0]):
            res2 = res2.append(df[['uid2']], ignore_index=True)
        # res2.columns=['uid2']

        # Find the shared UIDs in the two domains
        merged = res1.merge(res2,left_on='uid1',right_on='uid2',how='outer',indicator=True)
        shared_uid = merged[merged['_merge'] == 'both'][['uid1']]
        # unique_uid1 = merged[merged['_merge'] == 'left_only']['uid1'].tolist()
        # unique_uid2 = merged[merged['_merge'] == 'right_only']['uid2'].tolist()
        # print(unique_uid2)

        # Extract the ratings of the shared UIDs in the two domains
        res1 = pd.DataFrame([])
        for ratings in pd.read_csv(spath+'Category/'+domain1+'.csv', names=['uid', 'iid', 'rating', 'time'], chunksize=chunksize, skiprows=[0]):
            res1 = res1.append(ratings.loc[ratings['uid'].isin(shared_uid['uid1'].tolist())], ignore_index=True)

        res2 = pd.DataFrame([])
        for ratings in pd.read_csv(spath+'Category/'+domain2+'.csv', names=['uid', 'iid', 'rating', 'time'], chunksize=chunksize, skiprows=[0]):
            res2 = res2.append(ratings.loc[ratings['uid'].isin(shared_uid['uid1'].tolist())], ignore_index=True)

        # Calculate the items in each of the domains
        iid1 = pd.DataFrame(list(res1.groupby(['iid']).groups.keys()))
        iid2 = pd.DataFrame(list(res2.groupby(['iid']).groups.keys()))
        # print(iid1)

        # Write the shared ratings in the two domains into the target files
        print("The number of shared users between {0} and {1} is: {2}".format(domain1,domain2,shared_uid.shape[0]))
        print("The number of items in {0} is: {1}".format(domain1, iid1.shape[0]))
        print("The number of items in {0} is: {1}".format(domain2, iid2.shape[0]))

        shared_uid.to_csv(tpath + domain1 + '_and_' + domain2 + '_uid.csv',index=False, header=False)
        iid1.to_csv(tpath + domain1 + '_and_' + domain2 + '_iid_1.csv',index=False, header=False)
        iid2.to_csv(tpath + domain1 + '_and_' + domain2 + '_iid_2.csv', index=False, header=False)
        res1.to_csv(tpath + domain1 + '_and_' + domain2 + '_ratings_1.csv', index=False, header=False)
        res2.to_csv(tpath + domain1 + '_and_' + domain2 + '_ratings_2.csv', index=False, header=False)

    elif sel == 'item':
        # Summarize the shared items in the two nominated domains
        return 0

    elif sel == 'none':
        # Summarize the part which both users and items do not overlap

        return 0



if __name__ == "__main__":
    # Change to CSV
    # Raw2CSV('/media/work/Workspace/DataSets/Amazon/Raw/Category/reviews_Amazon_Instant_Video.json.gz',
    #         '/media/work/Workspace/DataSets/Amazon/Raw/Ratings/Category/Amazon_Instant_Video.csv')

    # Testing
    # df = getSlicedDF('/media/work/Workspace/DataSets/Amazon/Complete/reviews_Musical_Instruments.json.gz',chunksize=10**5)
    # print(df.loc[:,:])

    # Statistics for all categories
    # Stats('/media/work/Workspace/DataSets/Amazon/Raw/Ratings/')

    # Cross Domain Statistics
    CrossDomainStats(spath='/media/work/Workspace/DataSets/Amazon/Raw/Ratings/',
                     tpath='/media/work/Workspace/PhD_Projects/CDRS/Data/Amazon/Shared_UID/',
                     domain1='Electronics', domain2='Movies_and_TV',
                     sel='user')