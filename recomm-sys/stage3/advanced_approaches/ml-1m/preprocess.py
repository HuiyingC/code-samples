# https://grouplens.org/datasets/movielens/1m/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



users_df = pd.read_csv('./users.dat', sep='::', names=['userId', 'gender', 'age', 'occupation', 'zipCode'])

users_df['userId'] = users_df['userId'].astype(np.int)
users_df['age'] = users_df['age'].astype(np.int)
users_df['occupation'] = users_df['occupation'].astype(np.int)
users_df['zipCode'] = users_df['zipCode'].astype(str)

movies_df = pd.read_csv('./movies.dat', sep='::', names=['movieId', 'title', 'genres'])

movies_df['movieId'] = movies_df['movieId'].astype(np.int)

ratings_df = pd.read_csv('./ratings.dat', sep='::', names=['userId', 'movieId', 'ratting', 'timestamp'])

ratings_df['userId'] = ratings_df['userId'].astype(np.int)
ratings_df['movieId'] = ratings_df['movieId'].astype(np.int)


print(users_df.head())
print('Total unique users:', users_df.userId.unique().shape[0],'\n')
print(movies_df.head())
print('Total unique movies:', movies_df.movieId.unique().shape[0],'\n')
print(ratings_df.head())
print('Each user has rated at least %d movies' %(ratings_df.groupby(by='userId', as_index=False).movieId.count().movieId.min()))


# seperate train/test by userIds
trainUserIds, testUserIds = train_test_split(users_df.userId.unique(), test_size=0.3, random_state=42)

train_ratings_df = ratings_df[ratings_df.userId.isin(trainUserIds)]
test_ratings_df = ratings_df[ratings_df.userId.isin(testUserIds)]

train_ratings_df = train_ratings_df.sort_values(by=['userId', 'timestamp'], ascending=[True, True])
test_ratings_df = test_ratings_df.sort_values(by=['userId', 'timestamp'], ascending=[True, True])

# prepare history movieIds
def mark_last_timestamp(df):
    last = df[['userId', 'movieId']].groupby(by='userId', as_index=False).tail(1).copy()
    last['last'] = 1
    df = pd.merge(df, last, how='left', on=['userId', 'movieId'])
    df.loc[~df['last'].isnull(), 'last'] = 1
    df.loc[df['last'].isnull(), 'last'] = 0
    return df

train_ratings_df = mark_last_timestamp(train_ratings_df)
test_ratings_df = mark_last_timestamp(test_ratings_df)

candidate_movie_ids = movies_df.movieId.values


def neg_sampling(candidates, filters, length):
    max_len = len(candidates)
    res = []
    for i in range(length):
        while (1):
            c = candidates[np.random.randint(0, max_len)]
            if c not in filters:
                res.append(str(c))
                filters.add(c)
                break
    return res



def get_hist_movie_ids(df, max_len=10):
    hist_movie_ids = list()
    neg_hist_movie_ids = list()
    for _, group in df.groupby(by='userId'):
        tmp_hist_movie_ids = list()
        for _, row in group.iterrows():
            # keep high rated movies
            if row['ratting'] >= 4 and row['last'] == 0:
                tmp_hist_movie_ids.append(str(int(row['movieId'])))
        # keep latest high rated movies
        tmp_hist_movie_ids.reverse()
        tmp_hist_movie_ids = tmp_hist_movie_ids[:max_len]
        # revert to timestamp order
        tmp_hist_movie_ids.reverse()
        tmp_neg_hist_movie_ids = neg_sampling(
            candidate_movie_ids, set(hist_movie_ids), len(tmp_hist_movie_ids))
        hist_movie_ids.append('|'.join(tmp_hist_movie_ids))
        neg_hist_movie_ids.append('|'.join(tmp_neg_hist_movie_ids))
    return hist_movie_ids, neg_hist_movie_ids

train_hist_movie_ids, train_neg_hist_movie_ids = get_hist_movie_ids(train_ratings_df)
test_hist_movie_ids, test_neg_hist_movie_ids = get_hist_movie_ids(test_ratings_df)

train_ratings_df = train_ratings_df[train_ratings_df['last'] == 1]
train_ratings_df['histHighRatedMovieIds'] = train_hist_movie_ids
train_ratings_df['negHistMovieIds'] = train_neg_hist_movie_ids

test_ratings_df = test_ratings_df[test_ratings_df['last'] == 1]
test_ratings_df['histHighRatedMovieIds'] = test_hist_movie_ids
test_ratings_df['negHistMovieIds'] = test_neg_hist_movie_ids

print(train_ratings_df.head())


# merge with other features
train_ratings_df = pd.merge(train_ratings_df, users_df, how='inner', on='userId')
test_ratings_df = pd.merge(test_ratings_df, users_df, how='inner', on='userId')
train_ratings_df = pd.merge(train_ratings_df, movies_df, how='inner', on='movieId')
test_ratings_df = pd.merge(test_ratings_df, movies_df, how='inner', on='movieId')

# create label
train_ratings_df['label'] = 0
train_ratings_df.loc[train_ratings_df['ratting'] >= 4, 'label'] = 1
test_ratings_df['label'] = 0
test_ratings_df.loc[test_ratings_df['ratting'] >= 4, 'label'] = 1

train_ratings_df.to_csv('./train.csv', index=False)
test_ratings_df.to_csv('./test.csv', index=False)

print(train_ratings_df.head())
