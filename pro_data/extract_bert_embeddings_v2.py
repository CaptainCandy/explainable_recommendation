import tensorflow as tf
import dill as pickle
import json
from tqdm import tqdm
from transformers import AutoConfig
from transformers import BertTokenizer, TFBertModel, BertConfig

dataset_name = "music"

# f = open("../data/%s/Movies_and_TV_5.json" % dataset_name, "r")
# f = open("../data/%s/Kindle_Store_5.json" % dataset_name, "r")
# f = open("../data/%s/Toys_and_Games_5.json" % dataset_name, "r")
# f = open("../data/%s/Musical_Instruments_5.json" % dataset_name, "r")
f = open("../data/%s/Digital_Music_5.json" % dataset_name, "r")
f_w = open("../data/%s_bert/reviews_all" % dataset_name, "wb")

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
with open("../data/%s_bert/reviews_all" % dataset_name, "rb") as f:
    while True:
        try:
            line = pickle.load(f)
            reviews_all.append(line)
        except EOFError:
            break
    # for line in f:
    #     # :-1是为了把最后的回车去掉
    #     l = "[CLS]" + str(line[:-1]) + "[SEP]"
    #     reviews_all.append(l)
    f.close()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased", output_hidden_states = True) # 如果想要获取到各个隐层值需要如此设置
max_len = 512

# Convert token to vocabulary indices
print("converting tokens to vocab ids...")
token = []
for r in tqdm(reviews_all, ncols=80):
    tokenized_string = tokenizer.tokenize(r)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_string)
    token.append(tokens_ids)
# pickle.dump(token, open("../data/%s_bert/reviews_token" % dataset_name, "wb"))
# token = pickle.load(open("../data/%s_bert/reviews_token" % dataset_name, "rb"))

print("extracting embeddings...")
# reviews_embeddings = []
r_f = open("../data/%s_bert/reviews_embeddings" % dataset_name, "wb")
num = len(token)
# batch_size = 128是经过测试后选择的，最大利用GPU
batch_size = 64
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
    # 2代表hidden_states, -2代表倒数第二层
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
# pickle.dump(outputs, open("../data/%s_bert/outputs" % dataset_name, 'wb'))

# reviews_embeddings = []
# for token_vecs in tqdm(outputs, ncols=80):
# # Calculate the average of all input token vectors.
#     sentence_embedding = tf.math.reduce_mean(token_vecs, axis=0)
#     reviews_embeddings.append(sentence_embedding)
# print(len(reviews_embeddings))
# pickle.dump(reviews_embeddings, open("../data/%s_bert/reviews_embeddings" % dataset_name, 'wb'))