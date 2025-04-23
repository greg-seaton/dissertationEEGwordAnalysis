import numpy as np
import random
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
import pprint
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances



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
print (word_vectors.shape)

#set up nested dictionary to store results
results = {
    "HDBScan": {},
}

distance_metric = "euclidean"
# distance_metric = "precomputed" #use this for cosine
min_cluster_sizes= [5, 10, 25, 50, 100, 250]

cosine_dist = cosine_distances(word_vectors).astype(np.float64)

word_vectors = word_vectors.astype(np.float64)

for min_cluster_size in min_cluster_sizes:
    print (f"Starting min_cluster_size: {min_cluster_size}, at {datetime.now()}")

    #HDBSCAN
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=-1, metric=distance_metric)
    # labels = hdbscan_model.fit_predict(cosine_dist) ##use this line for cosine
    labels = hdbscan_model.fit_predict(word_vectors) ## use this line for euclidean

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

    results["HDBScan"][min_cluster_size] = {
        'n_clusters': n_clusters,
        'amount_noise': amount_noise,
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_score,
    }

pprint.pprint(results)