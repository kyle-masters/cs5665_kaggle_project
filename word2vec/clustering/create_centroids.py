import pickle
import numpy as np
from tqdm.auto import tqdm
import math
import os


def create_bag_of_centroids(wordlist, word_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_map:
            index = word_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


def create_centroids(train_size, test_size, file_size, num_features, min_word_count, context, num_clusters):
    os.makedirs(f'data/centroid', exist_ok=True)

    num_train = math.ceil(train_size / file_size)
    num_test = math.ceil(test_size / file_size)

    with open(f'data/centroid/map_{num_clusters}clusters_{num_features}features_{min_word_count}minwords_{context}context.plk', 'rb') as f:
        word_centroid_map = pickle.load(f)

    train_centroids = np.zeros((train_size, num_clusters), dtype="float32")

    print('Creating training centroids')
    for file_num in tqdm(range(num_train), position=0, desc='  files'):
        with open(f'data/clean_wordlists/{file_size}_rows/train/clean_n{file_num}.pkl', 'rb') as f:
            clean_reviews = pickle.load(f)
        for i, review in enumerate(tqdm(clean_reviews, leave=False, position=1, desc='     rows')):
            train_centroids[file_num*file_size+i] = create_bag_of_centroids(review, word_centroid_map)

    print('Saving training centroids')
    with open(f'data/centroid/train_{num_clusters}clusters_{num_features}features_{min_word_count}minwords_{context}context.plk', 'wb') as f:
        np.save(f, train_centroids)

    del train_centroids, clean_reviews

    test_centroids = np.zeros((test_size, num_clusters), dtype="float32")

    print('Creating test centroids')
    for file_num in tqdm(range(num_test), position=0, desc='  files'):
        with open(f'data/clean_wordlists/{file_size}_rows/test/clean_n{file_num}.pkl', 'rb') as f:
            clean_reviews = pickle.load(f)
        for i, review in enumerate(tqdm(clean_reviews, leave=False, position=1, desc='     rows')):
            test_centroids[file_num*file_size+i] = create_bag_of_centroids(review, word_centroid_map)

    print('Saving test centroids')
    with open(f'data/centroid/test_{num_clusters}clusters_{num_features}features_{min_word_count}minwords_{context}context.plk', 'wb') as f:
        np.save(f, test_centroids)

