import numpy as np
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords

nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

def load_glove_model(file_path):
    word_vectors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors

NLPmodel = load_glove_model("../glove-wiki-gigaword-100")

FUNCTION_TAGS = {
    "DT", "CC", "IN", "PRP", "PRP$", "TO", "MD", "WP", "WRB", "EX"
}

content_vectors = []
function_vectors = []

vocab = list(NLPmodel.keys())

#tagging in batches for optimisation
BATCH_SIZE = 5000
EXPECTED_VECTOR_LENGTH = 100  # for glove-wiki-gigaword-100

for i in range(0, len(vocab), BATCH_SIZE):
    batch_words = vocab[i:i+BATCH_SIZE]
    tags = pos_tag(batch_words)

    for word, tag in tags:
        vec = NLPmodel.get(word)
        if vec is None or vec.shape[0] != EXPECTED_VECTOR_LENGTH:
            print ("altert, vector wrong shape",word)
            continue  # skip unexpected vector shapes (only one in the entire model applies here, doesnt matter)

        vec = vec / np.linalg.norm(vec)  # normalise for consistency, results will be used by cosine similarity anyway

        if tag in FUNCTION_TAGS:
            function_vectors.append(vec)
        else:
            content_vectors.append(vec)

avg_content = np.mean(content_vectors, axis=0)
avg_function = np.mean(function_vectors, axis=0)

print("Average content vector:", np.array(avg_content))
print("Average function vector:", np.array(avg_function))
