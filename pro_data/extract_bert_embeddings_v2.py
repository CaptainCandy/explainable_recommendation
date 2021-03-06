'''
这个文件是在训练模型之间先利用BERT把评论文本的特征给抽取出来，这边使用了transformers开源库写的bert模型，因为官方版本只支持v1
'''
import tensorflow as tf
import dill as pickle
import json
from tqdm import tqdm
from transformers import AutoConfig
from transformers import BertTokenizer, TFBertModel, BertConfig

dataset_name = "movies"

f = open("../data2014/%s/Movies_and_TV_5.json" % dataset_name, "r")
# f = open("../data2014/%s/Kindle_Store_5.json" % dataset_name, "r")
# f = open("../data2014/%s/Toys_and_Games_5.json" % dataset_name, "r")
# f = open("../data2014/%s/Musical_Instruments_5.json" % dataset_name, "r")
# f = open("../data2014/%s/Digital_Music_5.json" % dataset_name, "r")
# f = open("../data2014/%s/Industrial_and_Scientific_5.json" % dataset_name, "r")
# f = open("../data2014/%s/Software_5.json" % dataset_name, "r")
# f = open("../data2014/%s/Luxury_Beauty_5.json" % dataset_name, "r")
# f = open("../data2014/%s/CDs_and_Vinyl_5.json" % dataset_name, "r")
f_w = open("../data2014/%s_bert/reviews_all" % dataset_name, "wb")

count = 0
null = 0
for line in tqdm(f, ncols=80):
    js = json.loads(line)
    if str(js['reviewerID']) == 'unknown':
        print("reviewerID unknown")
        continue
    if str(js['asin']) == 'unknown':
        print("asin unknown")
        continue
    try:
        pickle.dump(str(js["reviewText"]), f_w)
        count += 1
    except KeyError:
        null += 1
f.close()
f_w.close()
print("%s reviews_all saved. %s null reviews jumped. " % (count, null))

reviews_all = []
with open("../data2014/%s_bert/reviews_all" % dataset_name, "rb") as f:
    while True:
        try:
            line = pickle.load(f)
            reviews_all.append(line)
        except EOFError:
            break
    f.close()

# 定义切词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# 初始化BERT模型
model = TFBertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)  # 如果想要获取到各个隐层值需要如此设置
max_len = 512

# Convert token to vocabulary indices
print("converting tokens to vocab ids...")
token = []
for r in tqdm(reviews_all, ncols=80):
    tokenized_string = tokenizer.tokenize(r)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_string)
    token.append(tokens_ids)
# 存一下token可以在哪里出错的时候不用再转换一遍
pickle.dump(token, open("../data2014/%s_bert/reviews_token" % dataset_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
# token = pickle.load(open("../data2014/%s_bert/reviews_token" % dataset_name, "rb"))

print("extracting embeddings...")
# reviews_embeddings = []
r_f = open("../data2014/%s_bert/reviews_embeddings" % dataset_name, "wb")
num = len(token)
# batch_size = 128是经过测试后选择的，最大利用GPU
batch_size = 128
for i in tqdm(range(num//batch_size + 1), ncols=80):
    t_batch = token[batch_size*i:batch_size*i+batch_size]
    # padding tokens
    for idx, t in enumerate(t_batch):
        seq_len = len(t)
        if seq_len > max_len:
            t_batch[idx] = t[:max_len]
        elif seq_len < max_len:
            for j in range(max_len - seq_len):
                t.append(0)
    out = model(tf.convert_to_tensor(t_batch), training=False)
    # 2代表hidden_states, -2代表倒数第二层，经过一个腾讯大佬的测试-2层的表示效果要比-1层好
    second_last_layer = out[2][-2]
    for j in range(len(second_last_layer)):
        token_vecs = second_last_layer[j]
        sentence_embedding = tf.math.reduce_mean(token_vecs, axis=0)
        # reviews_embeddings.append(sentence_embedding)
        sentence_embedding = tf.make_ndarray(tf.make_tensor_proto(sentence_embedding))
        pickle.dump(sentence_embedding, r_f)
r_f.close()
print("done writing.")
# for t in tqdm(token, ncols=80):
#     if len(t) > max_len:
#         t = t[:max_len]
#     out = model(tf.convert_to_tensor([t])) # encoded_layers, pooled_output
#     # 2代表hidden_states, -2代表倒数第二层, 0代表输出的第一个句子
#     outputs.append(out[2][-2][0])
# pickle.dump(outputs, open("../data2014/%s_bert/outputs" % dataset_name, 'wb'))

# reviews_embeddings = []
# for token_vecs in tqdm(outputs, ncols=80):
# # Calculate the average of all input token vectors.
#     sentence_embedding = tf.math.reduce_mean(token_vecs, axis=0)
#     reviews_embeddings.append(sentence_embedding)
# print(len(reviews_embeddings))
# pickle.dump(reviews_embeddings, open("../data2014/%s_bert/reviews_embeddings" % dataset_name, 'wb'))
