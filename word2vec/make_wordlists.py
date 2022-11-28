import pickle
import pandas as pd
from tqdm.auto import tqdm
import os
import math
from word2vec.build_model import review_to_wordlist


def make_wordlists(train_path, test_path, train_size, test_size, file_size):
    num_train = math.ceil(train_size / file_size)
    num_test = math.ceil(test_size / file_size)

    os.makedirs(f'data/clean_wordlists/{file_size}_rows/train', exist_ok=True)
    os.makedirs(f'data/clean_wordlists/{file_size}_rows/test', exist_ok=True)

    print('Reading training data')
    train = pd.read_csv(train_path, header=0)


    print('Create train files')
    for file_num in tqdm(range(num_train), position=0, desc='  files'):
        reviews = []
        for i in tqdm(range(file_size if file_num < num_train - 1 else train_size-(file_num*file_size)), leave=False, position=1, desc='     rows'):
            reviews.append(review_to_wordlist(train["review_text"][file_num*file_size+i]))
        with open(f'data/clean_wordlists/{file_size}_rows/train/clean_n{file_num}.pkl', 'wb') as f:
            pickle.dump(reviews, f)

    del reviews, train

    print('Reading test data')
    test = pd.read_csv(test_path, header=0)

    print('Create test files')
    for file_num in tqdm(range(num_test), position=0, desc='  Files'):
        reviews = []
        for i in tqdm(range(file_size if file_num < num_test - 1 else test_size-(file_num*file_size)), leave=False, position=1, desc='     rows'):
            reviews.append(review_to_wordlist(test["review_text"][file_num*file_size+i]))
        with open(f'data/clean_wordlists/{file_size}_rows/test/clean_n{file_num}.pkl', 'wb') as f:
            pickle.dump(reviews, f)
