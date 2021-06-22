# coding=utf-8
# 这个文件使用tensorflow v1的版本和一个bert_utils库写的，但是现在已经更新v2了，v1的contrib包都不支持了，而且v1的版本不支持实验室的3090显卡，所以虽然v1跑得更快但还是只能弃用

import sys
# sys.path.append('C:\\Users\\ZJUSO\\Documents\\CaptainCandy\\bert-utils')
from bert_utils.extract_feature import BertVector

import numpy as np
import tensorflow as tf
import json
import dill as pickle
import os
from tqdm import tqdm

dataset_name = "movies"

f = open("../data/%s/Movies_and_TV_5.json" % dataset_name, "r")
f_w = open("../data/%s_bert/reviews_all.txt" % dataset_name, "w")

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
        f_w.write(js["reviewText"])
        f_w.write("\n")
    except KeyError:
        null += 1
f.close()
f_w.close()
print("reviews_all saved. %s null reviews jumped. " % null)

reviews_all = []
reviews_embeddings = []
with open("../data/%s_bert/reviews_all.txt" % dataset_name, "r") as f:
    for line in f:
        reviews_all.append(line)
f.close()
bv = BertVector()
# 分batch扔进去的话还是很慢，居然还是size=1最快。。但都要100小时+
batch_size = 1
num = len(reviews_all)
for i in tqdm(range(num//batch_size+1), ncols=80):
    r_batch = reviews_all[batch_size*i:batch_size*i+batch_size]
    embeddings = bv.encode(r_batch)
    for embed in embeddings:
        reviews_embeddings.append(embed)
print(len(reviews_embeddings))
pickle.dump(reviews_embeddings, open("../data/%s_bert/reviews_embeddings" % dataset_name, 'wb'))
