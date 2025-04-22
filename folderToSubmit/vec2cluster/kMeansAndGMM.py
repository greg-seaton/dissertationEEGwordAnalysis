import numpy as np
import random
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pprint
from datetime import datetime


#loads the glove model and gets 10,000 random vectors from it
def load_glove_model(file_path="../../glove-wiki-gigaword-100"):
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
    sample_size=10000

    #randomly sample 10,000 words
    sampled_words = random.sample(list(word_vectors.keys()), sample_size)

    #convert sampled_words into embedding vectors
    word_vectors = [word_vectors[word] for word in sampled_words]

    return word_vectors

word_vectors = load_glove_model()
print ("vectors retrieved")

#set up nested dictionary to store results
results = {
    "KMeans": {},
    "GMM": {}
}

for n_clusters in range (3,15):
    print (f"Starting Cluters {n_clusters} and at {datetime.now()}")
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans_labels = kmeans.fit_predict(word_vectors)

    # GMM
    gmm = GaussianMixture(n_components=n_clusters, init_params='k-means++', covariance_type='full')
    gmm_labels = gmm.fit_predict(word_vectors)

    #Evaluate KMeans
    kmeans_silhouette = silhouette_score(word_vectors, kmeans_labels)
    kmeans_db_index = davies_bouldin_score(word_vectors, kmeans_labels)

    #Evaluate GMM
    gmm_silhouette = silhouette_score(word_vectors, gmm_labels)
    gmm_db_index = davies_bouldin_score(word_vectors, gmm_labels)

    results["KMeans"][n_clusters] = {
        "silhouette_score": kmeans_silhouette,
        "davies_bouldin_score": kmeans_db_index
    }

    results["GMM"][n_clusters] = {
        "silhouette_score": gmm_silhouette,
        "davies_bouldin_score": gmm_db_index
    }

pprint.pprint(results)