import json

dataset_name = "music"

# 将原数据的userid, itemid和review分别存在txt文件中，方便对齐
# 数据集名字要看着改
f = open('../data/%s/Digital_Music_5.json' % dataset_name)
f_user_id = open('../data/%s/user_id.txt' % dataset_name, "w")
f_item_id = open('../data/%s/item_id.txt' % dataset_name, "w")
f_review = open('../data/%s/reviews_all.txt' % dataset_name, "w")
for line in f:
    js = json.loads(line)
    if str(js['reviewerID']) == 'unknown':
        print("reviewerID unknown")
        continue
    if str(js['asin']) == 'unknown':
        print("asin unknown")
        continue
    f_user_id.write(js['reviewerID'])
    f_user_id.write("\n")
    f_item_id.write(js['asin'])
    f_item_id.write("\n")
    f_review.write(js['reviewText'])
    f_review.write("\n")
f_user_id.close()
f_item_id.close()
f_review.close()

# 将抽取好的bert_embedding和id一一对应
f_embedding = open("../data/%s/bert_embeddings.txt" % dataset_name)
for line in f_embedding:
    js = json.loads(line)
