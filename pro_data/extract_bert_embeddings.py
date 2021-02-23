# coding=utf-8

import sys
sys.path.append('C:\\Users\\ZJUSO\\Documents\\CaptainCandy\\bert-utils')
from extract_feature import BertVector

import numpy as np
import tensorflow as tf
import json
import dill as pickle
import os
from tqdm import tqdm

dataset_name = "movies"

f = open("../data/%s/Movies_and_TV_5.json" % dataset_name, "r")
f_w = open("../data/%s_bert/reviews_all" % dataset_name, "wb")

reviews_all = []
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
        reviews_all.append(js["reviewText"])
    except KeyError:
        null += 1
pickle.dump(reviews_all, f_w)
f.close()
f_w.close()
print("reviews_all saved. %s null reviews jumped. " % null)

reviews_all = []
reviews_embeddings = []
with open("../data/%s_bert/reviews_all" % dataset_name, "rb") as f:
    for line in f:
        reviews_all.append(line)
bv = BertVector()
for r in tqdm(reviews_all, ncols=80):
    embedding = bv.encode([r])
    reviews_embeddings.append(embedding)
print(len(reviews_embeddings))
pickle.dump(reviews_embeddings, open("../data/%s_bert/reviews_embeddings" % dataset_name, 'wb'))