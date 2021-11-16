# Authors: Soumith Basina, Chakradhar Chokkaku
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

# the threshold of similarity for selecting the neighbourhood of a song
threshold = 0

# read the train.csv file and build a pivot table
train_df = pd.read_csv('train.csv')
piv = train_df.pivot(index=['customer_id'], columns=['song_id'])

del(train_df)

# Calculation of the Pearson Correlations among the songs
# normalise the scores in the pivot table and fill NaNs with 0
piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
piv_norm.fillna(0, inplace=True)

# calculate the cosine similarities in the normalised data
# used a sparse matrix to be efficient
piv_sparse = sp.sparse.csr_matrix(piv_norm.values)
song_sim_df = pd.DataFrame(cosine_similarity(piv_sparse.T), index = piv_norm.columns, columns = piv_norm.columns)

del(piv_sparse)

#cutting off all the values below threshold
song_sim_df = song_sim_df.where(song_sim_df > threshold, 0)

# taking the weighted average of the scores with similarities as weights
piv = piv.fillna(0)
predictions = piv.dot(song_sim_df)
sum_sim = piv.where(piv == 0, 1).dot(song_sim_df)
predictions = predictions.div(sum_sim)

# deleting useless variables to free some memory
del(piv)
del(piv_norm)
del(sum_sim)
del(song_sim_df)

# take the customer_id and song_id from each row and return the predicted value in predictions
test_df = pd.read_csv('test.csv')
test_df['score'] = test_df.apply(lambda row: predictions.loc[(row.customer_id, "score")][row.song_id], axis=1)

# replacing any na values in test_df with the mean score of the song
song_score_means = predictions.mean(axis=0)
# takes a slice of test_df in which the scores are NaN, replace them with the average rating of the song and update test_df
test_df.update(test_df.loc[test_df["score"].isnull()].apply(lambda row: row.fillna(song_score_means.loc["score"][row.song_id]), axis=1))

# drop the customer_id, song_id columns from the test_df and export the dataframe as a .csv file
test_df.drop(['customer_id', 'song_id'], inplace=True, axis=1)
test_df.to_csv(r'./submission.csv', index_label="test_row_id")
