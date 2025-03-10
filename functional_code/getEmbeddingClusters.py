import gensim.downloader as api
import os
import re

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


# Load GloVe word embeddings
model = api.load("glove-wiki-gigaword-100")

# Define folder names for dataset
folder_names = ["content", "function"]
folders_path = "../dataSets/spectrogramDataHighGran1participant/1"
words = []

# Collect words from the dataset
for folder in folder_names:
    full_path = os.path.join(folders_path, folder)

    for word in os.listdir(full_path):
        word = re.sub(r"\d+$", "", word.lower())
        if word not in words:
            words.append(word)

# Get word embeddings for the words
word_vectors = []
for word in words:
    word_vectors.append(model[word])


##applying pca

pca = PCA(n_components=3)  # reduce to 3 principal components
word_vectors = pca.fit_transform(word_vectors)

for n_clusters in range (2,15):
    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, algorithm='lloyd', init='k-means++')
    kmeans_labels = kmeans.fit_predict(word_vectors)

    # GMM Clustering
    gmm = GaussianMixture(n_components=n_clusters, init_params='k-means++', covariance_type='full')
    gmm_labels = gmm.fit_predict(word_vectors)

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=n_clusters)
    dbscan_labels = dbscan.fit_predict(word_vectors)

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    agglo_labels = agglo.fit_predict(word_vectors)

    # Evaluate KMeans
    kmeans_inertia = kmeans.inertia_
    kmeans_silhouette = silhouette_score(word_vectors, kmeans_labels)
    kmeans_db_index = davies_bouldin_score(word_vectors, kmeans_labels)

    # Evaluate GMM
    gmm_silhouette = silhouette_score(word_vectors, gmm_labels)
    gmm_db_index = davies_bouldin_score(word_vectors, gmm_labels)

    # Evaluate DBSCAN
    # DBSCAN assigns -1 to noise, so we exclude noise from the metrics
    dbscan_labels_filtered = dbscan_labels[dbscan_labels != -1]
    word_vectors_filtered = np.array(word_vectors)[dbscan_labels != -1]
    dbscan_silhouette = silhouette_score(word_vectors_filtered, dbscan_labels_filtered) if len(dbscan_labels_filtered) > 0 else None
    dbscan_db_index = davies_bouldin_score(word_vectors_filtered, dbscan_labels_filtered) if len(dbscan_labels_filtered) > 0 else None

    # Evaluate Agglomerative Clustering
    agglo_silhouette = silhouette_score(word_vectors, agglo_labels)
    agglo_db_index = davies_bouldin_score(word_vectors, agglo_labels)

    # Print results
    print ("number of clsuters", n_clusters)
    print(f"KMeans Inertia: {kmeans_inertia}")
    print(f"KMeans Silhouette Score: {kmeans_silhouette}")
    print(f"KMeans Davies-Bouldin Index: {kmeans_db_index}")
    print()
    print(f"GMM Silhouette Score: {gmm_silhouette}")
    print(f"GMM Davies-Bouldin Index: {gmm_db_index}")
    print()
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette if dbscan_silhouette is not None else 'Noise present'}")
    print(f"DBSCAN Davies-Bouldin Index: {dbscan_db_index if dbscan_db_index is not None else 'Noise present'}")
    print()
    print(f"Agglomerative Clustering Silhouette Score: {agglo_silhouette}")
    print(f"Agglomerative Clustering Davies-Bouldin Index: {agglo_db_index}")



#db scan with 6-8 clusters appears to perform best
