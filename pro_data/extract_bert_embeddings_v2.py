import tensorflow as tf
import dill as pickle
import json
from tqdm import tqdm
from transformers import AutoConfig
from transformers import BertTokenizer, TFBertModel, BertConfig

dataset_name = "kindle"

# f = open("../data/%s/Movies_and_TV_5.json" % dataset_name, "r")
# f = open("../data/%s/Kindle_Store_5.json" % dataset_name, "r")
# f_w = open("../data/%s_bert/reviews_all.txt" % dataset_name, "w")

# null = 0
# for line in tqdm(f, ncols=80):
#     js = json.loads(line)
#     if str(js['reviewerID']) == 'unknown':
#         print("reviewerID unknown")
#         continue
#     if str(js['asin']) == 'unknown':
#         print("asin unknown")
#         continue
#     try:
#         f_w.write(js["reviewText"])
#         f_w.write("\n")
#     except KeyError:
#         null += 1
# f.close()
# f_w.close()
# print("reviews_all saved. %s null reviews jumped. " % null)

reviews_all = []
with open("../data/%s_bert/reviews_all.txt" % dataset_name, "r") as f:
    for line in f:
        # :-1是为了把最后的回车去掉
        l = "[CLS]" + str(line[:-1]) + "[SEP]"
        reviews_all.append(l)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased", output_hidden_states = True) # 如果想要获取到各个隐层值需要如此设置

# Convert token to vocabulary indices
token = []
for r in tqdm(reviews_all, ncols=80):
    tokenized_string = tokenizer.tokenize(r)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_string)
    token.append(tokens_ids)
pickle.dump(token, open("../data/%s_bert/reviews_token" % dataset_name, "wb"))
# token = pickle.load(open("../data/%s_bert/reviews_token" % dataset_name, "rb"))

print("extracting embeddings...")
outputs = []
for t in tqdm(token, ncols=80):
    out = model(tf.convert_to_tensor([t])) # encoded_layers, pooled_output
    # 2代表hidden_states, -2代表倒数第二层, 0代表输出的第一个句子
    outputs.append(out[2][-2][0])

reviews_embeddings = []
for token_vecs in tqdm(outputs, ncols=80):
# Calculate the average of all input token vectors.
    sentence_embedding = tf.math.reduce_mean(token_vecs, axis=0)
    reviews_embeddings.append(sentence_embedding)
print(len(reviews_embeddings))
pickle.dump(reviews_embeddings, open("../data/%s_bert/reviews_embeddings" % dataset_name, 'wb'))