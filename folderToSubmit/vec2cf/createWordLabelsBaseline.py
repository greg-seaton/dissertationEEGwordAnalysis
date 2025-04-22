import os
import gensim.downloader as api
import numpy as np
import re


def load_glove_model(file_path):
    word_vectors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # First token is the word
            vector = np.array(values[1:], dtype=np.float32)  # Rest are vector values
            word_vectors[word] = vector
    return word_vectors

glove_model = load_glove_model("../../glove-wiki-gigaword-100")


folder = "data/"     ##change this to look at different runs

words_path = os.path.join(folder, "words_labels.npz")

words = np.load(words_path, allow_pickle=True)  
train_words = words["train_words"] #words
valid_words = words["valid_words"] #words
test_words = words["test_words"] #words

train_words = np.concatenate([train_words, valid_words], axis=0)

train_words = [re.sub(r'\d+$', '', word.lower()) for word in train_words] #remove the trailing numbers
test_words = [re.sub(r'\d+$', '', word.lower()) for word in test_words] #remove the trailing numbers

for i in range(20):
    print (train_words[i],test_words[i])

train_baseline = [glove_model[word] for word in train_words]
test_baseline = [glove_model[word] for word in test_words]

np.savez_compressed("testTrain_baseline.npz", 
    train_baseline=train_baseline,
    test_baseline=test_baseline)

