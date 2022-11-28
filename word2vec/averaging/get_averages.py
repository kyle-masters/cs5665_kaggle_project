import numpy as np
from gensim.models import Word2Vec
import os
import math
from tqdm.auto import tqdm
import pickle


def get_average_features(wordlist, model, num_features):
    # Pre-initialize empty numpy array for speed
    features = np.zeros((num_features, ), dtype='float32')

    # Set for search speed
    index_to_keys = set(model.wv.index_to_key)

    # For each word, add to feature vector if in the model's vocab
    i = 0
    for word in wordlist:
        if word in index_to_keys:
            i += 1
            features = np.add(features, model[word])

    return np.divide(features, i)

def create_averages_list(train_size, test_size, file_size, num_features, min_word_count, context):
    os.makedirs(f'data/average', exist_ok=True)

    num_train = math.ceil(train_size / file_size)
    num_test = math.ceil(test_size / file_size)

    model = Word2Vec.load(f'data/models/word2vec_{num_features}features_{min_word_count}minwords_{context}context')

    train_averages = np.zeros((train_size, num_features), dtype='float32')

    print('Creating train averages...')
    for file_num in tqdm(range(num_train), position=0, desc='  files'):
        with open(f'data/clean_wordlists/{file_size}_rows/train/clean_n{file_num}.pkl', 'rb') as f:
            clean_reviews = pickle.load(f)
        for i, review in enumerate(tqdm(clean_reviews, leave=False, position=1, desc='     rows')):
            train_averages[file_num*file_size+i] = get_average_features(review, model, num_features)

    print('Saving training averages...')
    with open(f'data/average/train_{num_features}features_{min_word_count}minwords_{context}context.plk', 'wb') as f:
        np.save(f, train_averages)

    del train_averages, clean_reviews

    test_averages = np.zeros((test_size, num_features), dtype='float32')

    print('Creating test averages...')
    for file_num in tqdm(range(num_test), position=0, desc='  files'):
        with open(f'data/clean_wordlists/{file_size}_rows/test/clean_n{file_num}.pkl', 'rb') as f:
            clean_reviews = pickle.load(f)
        for i, review in enumerate(tqdm(clean_reviews, leave=False, position=1, desc='     rows')):
            test_averages[file_num*file_size+i] = get_average_features(review, model, num_features)

    print('Saving testing averages...')
    with open(f'data/average/test_{num_features}features_{min_word_count}minwords_{context}context.plk', 'wb') as f:
        np.save(f, test_averages)
