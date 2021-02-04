import gensim

f = open("../data/GoogleNews-vectors-negative300.txt", "r")

# model = gensim.models.KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin", binary=True)

# model.wv.save_word2vec_format("../data/GoogleNews-vectors-negative300.txt")

#%%
import numpy as np

num = np.fromstring(f.readline(), dtype='float32')