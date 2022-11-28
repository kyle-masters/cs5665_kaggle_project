from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os


def predict(train_path, test_path, num_workers, num_features, min_word_count, context):
    os.makedirs(f'output', exist_ok=True)

    print('Reading training data...')
    train = pd.read_csv(train_path, header=0, usecols=['rating'])
    with open(f'data/average/train_{num_features}features_{min_word_count}minwords_{context}context.plk', 'rb') as f:
        train_averages = np.load(f)

    forest = RandomForestClassifier(n_jobs=num_workers)

    print('Fitting random forest to training data...')
    forest = forest.fit(train_averages, train['rating'])

    del train, train_averages

    print('Reading test data...')
    test = pd.read_csv(test_path, header=0, usecols=['review_id'])
    with open(f'data/average/test_{num_features}features_{min_word_count}minwords_{context}context.plk', 'rb') as f:
        test_averages = np.load(f)

    print('Predicting results and outputting')
    result = forest.predict(test_averages)
    output = pd.DataFrame(data={'review_id': test['review_id'], 'rating': result})
    output.to_csv(f'output/word2vec_averaging_{num_features}features_{min_word_count}minwords_{context}context.csv', index=False, quoting=3)
