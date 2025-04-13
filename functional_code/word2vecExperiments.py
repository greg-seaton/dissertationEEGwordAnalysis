import gensim.downloader as api
from tensorflow.keras import backend as K


print ("start")
# print(api.info())  # Lists all available models

# Load pretrained model (takes a while the first time)
model = api.load("glove-wiki-gigaword-100")
# model = api.load("glove-wiki-gigaword-50")
# model = api.load("word2vec-ruscorpora-50")  # Alternative model
# model = api.load("word2vec-ruscorpora-100")  # 100D, ~64MB

print ("loaded")

def cosine_similarity(y_true, y_pred):
    return K.sum(y_true * y_pred, axis=-1) / (K.sqrt(K.sum(y_true**2, axis=-1)) * K.sqrt(K.sum(y_pred**2, axis=-1)))

def cosine_similarityNorm(y_true, y_pred):
    y_true = y_true / K.sqrt(K.sum(y_true**2, axis=-1, keepdims=True))
    y_pred = y_pred / K.sqrt(K.sum(y_pred**2, axis=-1, keepdims=True))
    return K.sum(y_true * y_pred, axis=-1)

words=["food", "eat"]

while "n" not in words:
    words[0] = input ("word 0")
    words[1] = input ("word 1")

    print("cosine similarity =", cosine_similarity(model[words[0]], model[words[1]]))

# print ("my func", cosine_similarity(model[words[0]], model[words[1]]))
# print ("my func norm", cosine_similarityNorm(model[words[0]], model[words[1]]))

# print ("inbuilt", model.similarity(words[0], words[1]))

# similarity_food_eat = model.similarity("food", "eat")
# similarity_kettle_underpass = model.similarity("kettle", "underpass")