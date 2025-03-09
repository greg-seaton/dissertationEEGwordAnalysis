import gensim.downloader as api

print ("start")
# print(api.info())  # Lists all available models

# Load pretrained model (takes a while the first time)
model = api.load("glove-wiki-gigaword-100")
# model = api.load("glove-wiki-gigaword-50")
# model = api.load("word2vec-ruscorpora-50")  # Alternative model
# model = api.load("word2vec-ruscorpora-100")  # 100D, ~64MB

print ("loaded")

# Compute cosine similarity
similarity_food_eat = model.similarity("food", "eat")
similarity_kettle_underpass = model.similarity("kettle", "underpass")

print("Similarity between 'food' and 'eat':", similarity_food_eat)
print("Similarity between 'kettle' and 'underpass':", similarity_kettle_underpass)

print ("vector axe", model["axe"])

print ("vector food", model["food"])
print ("vector eat", model["eat"])

print ("vector kettle", model["kettle"])
print ("vector underpass", model["underpass"])