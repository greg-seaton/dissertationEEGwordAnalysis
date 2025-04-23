import numpy as np
import random
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
import pprint
from datetime import datetime


#loads the glove model and gets 10,000 random vectors from it
def load_glove_model(file_path="../../glove-wiki-gigaword-100", sample_size=10000):
    word_vectors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            word_vectors[word] = vector

    print(f"Total number of embeddings: {len(word_vectors)}")

    #set random seed for reproducibility
    random.seed(42)

    #randomly sample 10,000 words
    sampled_words = random.sample(list(word_vectors.keys()), sample_size)

    #convert sampled_words into embedding vectors
    word_vectors = np.array([word_vectors[word] for word in sampled_words])

    return word_vectors

sample_size=10000
word_vectors = load_glove_model(sample_size=sample_size)
print ("vectors retrieved")

#set up nested dictionary to store results
results = {
    "DBScan": {},
}

eps_values = [0.1,0.2,0.5,1,2,3]
min_samples_range = [3,7,15,50,125,250]
# distance_metric = "euclidean" #alternative cosine
distance_metric = "cosine"


for eps in eps_values:
    results["DBScan"][eps] = {}
    for min_samples in min_samples_range:

        print (f"Starting eps: {eps}, min_samples: {min_samples} at {datetime.now()}")

        #DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_metric, n_jobs=-1)
        labels = dbscan.fit_predict(word_vectors) # Use the numpy array X

        #get number of clusters
        n_clusters = len(set(labels)) - 1

        #get amount of noise [0,1]
        amount_noise = list(labels).count(-1)/sample_size

        #get evaluation metrics, needs at least 2 clusters
        silhouette = None
        db_score = None
        if n_clusters > 1:
            silhouette = silhouette_score(word_vectors, labels)
            db_score = davies_bouldin_score(word_vectors, labels)

        results["DBScan"][eps][min_samples] = {
            'n_clusters': n_clusters,
            'amount_noise': amount_noise,
            'silhouette_score': silhouette,
            'davies_bouldin_score': db_score,
        }

pprint.pprint(results)