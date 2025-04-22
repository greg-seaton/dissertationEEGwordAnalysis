import gensim.downloader as api
import random
import numpy as np

#this code demonstrates that the avg cosineSimilaritiy between a random vector and a vector from the word embedding mode is ~0
#adds credability to eeg2vec results

print ("start")
model = api.load("glove-wiki-gigaword-100")
# print(api.info())  # Lists all available models

#generates a normalised random vector
def randomVector(length=100):
    vec = np.random.rand(length) 
    return vec / np.linalg.norm(vec)

#gets the vector of a random word fron the model
def randomWord():
    vec = model[random.choice(list(model.key_to_index.keys()))]
    return vec / np.linalg.norm(vec)

def cosineSimilarity(vec1, vec2):
    if (len(vec1)!=100 or len(vec2)!=100):
        print ("vector length error")

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

avgCS=0  #hold cosine probability
n=200

for i in range (n):
    avgCS=avgCS + cosineSimilarity(randomWord(),randomVector())

print(avgCS/n)

#print the average from n samples

