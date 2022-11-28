from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pickle
import os


def create_centroid_map(num_features, min_word_count, context, num_clusters):
    os.makedirs(f'data/centroid', exist_ok=True)

    model = Word2Vec.load(f'data/models/word2vec_{num_features}features_{min_word_count}minwords_{context}context')

    word_vectors = model.wv.vectors

    # Initalize a k-means object and use it to extract centroids
    print('Extracting centroids')
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    print('Mapping centroids and saving map')
    word_centroid_map = dict(zip(model.wv.index_to_key, idx))

    # Save word centroid map
    with open(f'data/centroid/map_{num_clusters}clusters_{num_features}features_{min_word_count}minwords_{context}context.plk', 'wb') as f:
        pickle.dump(word_centroid_map, f)
