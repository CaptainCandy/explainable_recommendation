'''
Data pre process

@author:
Chong Chen (cstchenc@163.com)

@ created:
25/8/2017
@references:
'''
import os
import json
import pandas as pd
# import pickle
import numpy as np
import dill as pickle

dataset_name = "movies"
NARRE_DIR = '../data/%s' % dataset_name
TPS_DIR = '../data/%s_bert' % dataset_name
# TP_file = os.path.join(TPS_DIR, 'Musical_Instruments_5.json')
TP_file = os.path.join(NARRE_DIR, 'Movies_and_TV_5.json')
embed_file = os.path.join(TPS_DIR, 'reviews_embeddings')
embedding_size = 768

f = open(TP_file)
users_id = []
items_id = []
ratings = []
reviews = []
np.random.seed(2017)

null = 0
for line in f:
    js = json.loads(line)
    if str(js['reviewerID']) == 'unknown':
        print("reviewerID unknown")
        continue
    if str(js['asin']) == 'unknown':
        print("asin unknown")
        continue
    try:
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID']) + ',')
        items_id.append(str(js['asin']) + ',')
        ratings.append(str(js['overall']))
    except KeyError:
        null += 1
print("null:", null)
reviews_embeds = pickle.load(open(embed_file, "rb"))
data = pd.DataFrame({'user_id': pd.Series(users_id),
                     'item_id': pd.Series(items_id),
                     'ratings': pd.Series(ratings),
                     'reviews': pd.Series(reviews),
                     'reviews_embeds': pd.Series(reviews_embeds)})[['user_id', 'item_id', 'ratings', 'reviews', 'reviews_embeds']]


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
unique_uid = usercount.index
unique_sid = itemcount.index
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

# user2id_json = json.dumps(user2id)
# user2id_file = open('../data/%s/user2id.json' % dataset_name, 'w')
# user2id_file.write(user2id_json)
# user2id_file.close()
# item2id_json = json.dumps(item2id)
# item2id_file = open('../data/%s/item2id.json' % dataset_name, 'w')
# item2id_file.write(item2id_json)
# item2id_file.close()


def numerize(tp):
    tp['user_id'] = tp['user_id'].apply(lambda x: user2id[x])
    tp['item_id'] = tp['item_id'].apply(lambda x: item2id[x])
    return tp


data = numerize(data)
tp_rating = data[['user_id', 'item_id', 'ratings']]

n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train = tp_rating[~test_idx]

data2 = data[test_idx]
data = data[~test_idx]

n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]
tp_train.to_csv(os.path.join(TPS_DIR, '%s_train.csv' % dataset_name), index=False, header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, '%s_valid.csv' % dataset_name), index=False, header=None)
tp_test.to_csv(os.path.join(TPS_DIR, '%s_test.csv' % dataset_name), index=False, header=None)

user_reviews = {}
item_reviews = {}
user_rid = {}
item_rid = {}
user_reviews_embeddings = {}
item_reviews_embeddings = {}
for i in data.values:
    if i[0] in user_reviews:
        user_reviews[i[0]].append(i[3])
        user_rid[i[0]].append(i[1])
        user_reviews_embeddings[i[0]].append(i[4][0])
    else:
        user_rid[i[0]] = [i[1]]
        user_reviews[i[0]] = [i[3]]
        user_reviews_embeddings[i[0]] = [i[4][0]]
    if i[1] in item_reviews:
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
        item_reviews_embeddings[i[1]].append(i[4][0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]] = [i[0]]
        item_reviews_embeddings[i[1]] = [i[4][0]]

for i in data2.values:
    if i[0] in user_reviews:
        continue
    else:
        user_rid[i[0]] = [0]
        user_reviews[i[0]] = ['0']
        user_reviews_embeddings[i[0]] = np.ones(embedding_size)
    if i[1] in item_reviews:
        continue
    else:
        item_reviews[i[1]] = [0]
        item_rid[i[1]] = ['0']
        item_reviews_embeddings[i[1]] = np.ones(embedding_size)

pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))
pickle.dump(user_reviews_embeddings, open(os.path.join(TPS_DIR, 'user_reviews_embeddings'), 'wb'))
pickle.dump(item_reviews_embeddings, open(os.path.join(TPS_DIR, 'item_reviews_embeddings'), 'wb'))