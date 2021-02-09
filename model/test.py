import gensim
import numpy as np
import matplotlib.pyplot as plt

# %%
# f = open("../data/GoogleNews-vectors-negative300.txt", "r")

# model = gensim.models.KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin", binary=True)

# model.wv.save_word2vec_format("../data/GoogleNews-vectors-negative300.txt")

# %%
# with open("../data/instruments/Musical_Instruments_5.json", "r") as f:
#     lines = f.readlines()
#     # print(lines)
# with open("../data/instruments/Musical_Instruments_5_new.json", "w", encoding="utf-8") as f_new:
#     for line in lines:
#         if "reviewText" not in line:
#             continue
#         f_new.write(line)

# %%
criterion = np.load("./criterion_2021-02-08_22h47m03s.npz")
rmse_train, rmse_test, mae_train, mae_test = criterion["rmse_train"], criterion["rmse_test"], \
                                             criterion["mae_train"], criterion["mae_test"]
# 绘制曲线
print(np.min(mae_test))
print(np.min(rmse_test))
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(rmse_train)
plt.plot(rmse_test)
plt.legend(('Train rmse', 'Val rmse'))
plt.title("RMSE")
plt.subplot(122)
plt.plot(mae_train)
plt.plot(mae_test)
plt.title("MAE")
plt.legend(('Train mae', 'Val mae'))
plt.savefig('./results/%s.jpg' % "instruments_20210208")