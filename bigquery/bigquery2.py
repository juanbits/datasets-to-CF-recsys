import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import requests

header = ['user_id', 'item_id', 'rating']
df2 = pd.read_csv('bigquery-frutas-sin-items-no-vistos.csv', usecols=[0, 1, 2], names=header)

df = df2[['user_id', 'item_id']] # Get rid of unnecessary info

# map each item and user to a unique numeric value
data_user = df.user_id.astype("category")
data_item = df.item_id.astype("category")

stars = coo_matrix((np.ones(df.shape[0]),
                   (data_item.cat.codes.copy(),
                    data_user.cat.codes.copy())))

model = AlternatingLeastSquares(factors=50,
                                regularization=0.01,
                                dtype=np.float64,
                                iterations=50)

confidence = 40
model.fit(confidence * stars)

repos = dict(enumerate(data_item.cat.categories))
repo_ids = {r: i for i, r in repos.iteritems()}

#print [(repos[r], s) for r, s in model.similar_items(repo_ids['manzana'])]

def user_stars(user):
    repos = []
    repos = df.item_id.loc[df.user_id == str(user)]
    return repos

def user_items(u_stars):
    star_ids = [repo_ids[s] for s in u_stars if s in repo_ids]
    data = [confidence for _ in star_ids]
    rows = [0 for _ in star_ids]
    shape = (1, model.item_factors.shape[0])
    return coo_matrix((data, (rows, star_ids)), shape=shape).tocsr()

juan = user_items(user_stars("juan"))

def recommend(user_items):
    recs = model.recommend(userid=0, user_items=user_items, recalculate_user=True)
    return [(repos[r], s) for r, s in recs]

def explain(user_items, repo):
    _, recs, _ = model.explain(userid=0, user_items=user_items, itemid=repo_ids[repo])
    return [(repos[r], s) for r, s in recs]

print '----------ITEMS STARRED BY THE USER: juan'
print user_stars("juan")
print '----------ITEMS TO RECOMMEND TO THE USER: juan'
print recommend(juan)
print '----------EXPLAIN for: manzana'
print explain(juan, 'manzana')
