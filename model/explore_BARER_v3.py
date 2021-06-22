# 探索一个用户和他的历史纪录
import json

f = open('../data2014/movies/Movies_and_TV_5.json', 'r')

jsf = open('../data2014/movies/meta_Movies_and_TV.json', 'r')
meta = {}
for line in jsf:
    js = json.loads(line)
    meta[js["asin"]] = js
jsf.close()

for l in f:
    js = json.loads(l)
    if js["reviewerID"] == "A152C8GYY25HAH":
        print(js["asin"], js["overall"], meta[js["asin"]]["title"], ",", js["reviewText"].replace("\n", ""))