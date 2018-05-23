# This file is used to pre-process the MovieLens Datasets for CDRS

import math
import pandas as pd
import numpy as np

def GenreSplit_ML100K():
    # Read the item attributes
    genrelist = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    pre = ['iid','title','release_date','video_release_date','IMDb_URL']
    headers = pre + genrelist

    movies = pd.read_csv("/media/work/Workspace/DataSets/MovieLens/ml-100k/u.item", names=headers, sep='|', engine='python')
    # Keep the genre names only
    movies = movies[['iid']+genrelist]
    # print(movies.loc[0:4,['Action','Adventure']])

    # Read in the user-item ratings
    ratings = pd.read_csv("/media/work/Workspace/DataSets/MovieLens/ml-100k/u.data", names=['uid','iid','rating','time'], sep='\t', engine='python')
    # ratings = ratings[['uid','iid','rating']]

    # For each genre extract the relevent movie ids and then extract the relevant u-i pairs in rating matrix
    df=pd.DataFrame([])
    for i in range(len(genrelist)):
        genre = genrelist[i]
        tmp = movies.loc[movies[genre] == 1]

        # Store the classification results in a table
        tmp_df = pd.DataFrame([genre]+tmp['iid'].tolist()).T
        df = df.append(tmp_df)

        res = ratings.loc[ratings['iid'].isin(tmp['iid'].tolist())]
        res.to_csv('../Data/MovieLens/ml-100k/'+genre+'.csv',index=False,header=False)

    # Write the genres into a csv file
    genre = pd.DataFrame(genrelist, columns=['genre'])
    genre.to_csv('../Data/MovieLens/ml-100k/Genres.csv', index=False,header=False)

    # Write the genres classification results into one file
    df.to_csv('../Data/MovieLens/ml-100k/Genres_iid.csv', index=False,header=False)

def GenreSplit_ML1M():
    movies = pd.read_csv("/media/work/Workspace/DataSets/MovieLens/ml-1m/movies.dat", names=['iid','title','genres'], sep='::',engine='python')
    # print(movies.isnull().values.any())

    # Get the list of genres
    genrelist = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
    genres = pd.DataFrame(genrelist, columns=['genre'])

    # Tag each movie with a boolean list of genres
    movies = movies.join(movies.genres.str.get_dummies().astype(bool))
    movies.drop('genres', inplace=True, axis=1)

    # Read u-i ratings
    ratings = pd.read_csv("/media/work/Workspace/DataSets/MovieLens/ml-1m/ratings.dat", names=['uid','iid','rating','time'], sep='::',engine='python')

    # Split the u-i rating file
    df=pd.DataFrame([])
    for i in range(len(genres)):
        genre = genrelist[i]
        tmp = movies.loc[movies[genre]]

        # Store the classification results in a table
        tmp_df = pd.DataFrame([genre]+tmp['iid'].tolist()).T
        df = df.append(tmp_df)

        res = ratings.loc[ratings['iid'].isin(tmp['iid'].tolist())]
        res.to_csv('../Data/MovieLens/ml-1m/' + genre + '.csv', index=False, header=False)

    # Write the genres into a csv file
    # genres.to_csv('../Data/MovieLens/ml-1m/Genres.csv', index=False,header=False)

    # Write the genres classification results into one file
    # df.to_csv('../Data/MovieLens/ml-1m/Genres_iid.csv', index=False,header=False)

def GenreSplit_ML10M():
    movies = pd.read_csv("/media/work/Workspace/DataSets/MovieLens/ml-10m/movies.dat", names=['iid', 'title', 'genres'],sep='::', engine='python')
    # print(movies.isnull().values.any())

    # Get the list of genres
    genrelist = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
    genres = pd.DataFrame(genrelist, columns=['genre'])

    # Tag each movie with a boolean list of genres
    movies = movies.join(movies.genres.str.get_dummies().astype(bool))
    movies.drop('genres', inplace=True, axis=1)

    # Read u-i ratings
    ratings = pd.read_csv("/media/work/Workspace/DataSets/MovieLens/ml-10m/ratings.dat",
                          names=['uid', 'iid', 'rating', 'time'], sep='::', engine='python')

    # Split the u-i rating file
    df = pd.DataFrame([])
    for i in range(len(genres)):
        genre = genrelist[i]
        tmp = movies.loc[movies[genre]]

        # Store the classification results in a table
        tmp_df = pd.DataFrame([genre] + tmp['iid'].tolist()).T
        df = df.append(tmp_df)

        res = ratings.loc[ratings['iid'].isin(tmp['iid'].tolist())]
        res.to_csv('../Data/MovieLens/ml-10m/' + genre + '.csv', index=False, header=False)

    # Write the genres into a csv file
    genres.to_csv('../Data/MovieLens/ml-10m/Genres.csv', index=False,header=False)

    # Write the genres classification results into one file
    df.to_csv('../Data/MovieLens/ml-10m/Genres_iid.csv', index=False,header=False)

def GenreSplit_ML20M():
    movies = pd.read_csv("/media/work/Workspace/DataSets/MovieLens/ml-20m/movies.csv", names=['iid', 'title', 'genres'],sep=',', skiprows=1, engine='python')

    # Get the list of genres
    genrelist = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
    genres = pd.DataFrame(genrelist, columns=['genre'])
    # print(genres)

    # Tag each movie with a boolean list of genres
    movies = movies.join(movies.genres.str.get_dummies().astype(bool))
    movies.drop('genres', inplace=True, axis=1)

    # Read u-i ratings
    chunksize = 10**7
    for i in range(len(genrelist)):
        genre = genrelist[i]
        tmp = movies.loc[movies[genre]]
        res = pd.DataFrame([])
        for ratings in pd.read_csv("/media/work/Workspace/DataSets/MovieLens/ml-20m/ratings.csv",
                          names=['uid', 'iid', 'rating', 'time'], sep=',', engine='python', chunksize=chunksize):
            res = res.append(ratings.loc[ratings['iid'].isin(tmp['iid'].tolist())])
        res.to_csv('../Data/MovieLens/ml-20m/' + genre + '.csv', index=False, header=False)

    # Classify the genre-iid results in a table
    # df = pd.DataFrame([])
    # for i in range(len(genrelist)):
    #     genre = genrelist[i]
    #     tmp = movies.loc[movies[genre]]
    #
    #     # Store the classification results in a table
    #     tmp_df = pd.DataFrame([genre] + tmp['iid'].tolist()).T
    #     df = df.append(tmp_df)

    # Write the genres into a csv file
    genres.to_csv('../Data/MovieLens/ml-20m/Genres.csv', index=False,header=False)

    # Write the genres classification results into one file
    # df.to_csv('../Data/MovieLens/ml-20m/Genres_iid.csv', index=False,header=False)

def Stats(path,chuncksize = 10**6):
    genrelist = pd.read_csv(path+'Genres.csv', names=['genres'], engine='python')['genres'].tolist()

    # Dictionary to generate the final statistics
    df = {}

    # Loop for each genre
    for i in range(len(genrelist)):
        genre = genrelist[i]
        # print("Statistics for Genre: {0}".format(genre))

        res = pd.DataFrame([])
        for ratings in pd.read_csv(path+genre+'.csv',names=['uid','iid','rating','time'],chunksize=chuncksize):
            res = res.append(ratings.loc[:, ['uid', 'iid']], ignore_index=True)

        # Number of ratings per user and item
        n_ratings_per_user = res.groupby(['uid'])[['iid']].count()
        n_ratings_per_user.columns = ['number_of_ratings']
        n_ratings_per_user.to_csv(path + 'Stats/' + 'user_' + genre + '.csv', index=True, header=True)

        n_ratings_per_item = res.groupby(['iid'])[['uid']].count()
        n_ratings_per_item.columns = ['number_of_ratings']
        n_ratings_per_item.to_csv(path + 'Stats/' + 'item_' + genre + '.csv', index=True, header=True)

        # print("The number of distinct users: {0} for genre: {1}".format(n_ratings_per_user.shape[0],genre))
        # print("The number of distinct items: {0} for genre: {1}".format(n_ratings_per_item.shape[0],genre))

        # Write into the key of corresponding genre
        df[genre] = [n_ratings_per_user.shape[0],n_ratings_per_item.shape[0]]

    # Write the final results into one csv file
    stat = pd.DataFrame.from_dict(df, orient='index', columns=['number_of_users','number_of_items'])
    print(stat)
    stat.to_csv(path + 'Stats/Stats_Final.csv')

if __name__ == "__main__":
    # Memory Not Enough for the 20M dataset
    # GenreSplit_ML20M()
    # GenreSplit_ML100K()
    Stats('../Data/MovieLens/ml-20m/')