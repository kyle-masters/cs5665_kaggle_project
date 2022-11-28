from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os

def predict(train_path, test_path, num_workers, num_features, min_word_count, context, num_clusters):
    os.makedirs(f'output', exist_ok=True)

    forest = RandomForestClassifier(n_jobs=num_workers)

    print('Reading training data')
    train = pd.read_csv(train_path, header=0)
    with open(f'data/centroid/train_{num_clusters}clusters_{num_features}features_{min_word_count}minwords_{context}context.plk', 'rb') as f:
        train_centroids = np.load(f)

    print('Fitting random forest to training data')
    forest = forest.fit(train_centroids, train['rating'])

    del train, train_centroids

    print('Reading test data')
    test = pd.read_csv(test_path, header=0)
    with open(f'data/centroid/test_{num_clusters}clusters_{num_features}features_{min_word_count}minwords_{context}context.plk', 'rb') as f:
        test_centroids = np.load(f)

    print('Predicting results and outputting')
    result = forest.predict(test_centroids)
    output = pd.DataFrame(data={'review_id': test['review_id'], 'rating': result})
    output.to_csv(f'output/word2vec_clustering_{num_clusters}clusters_{num_features}features_{min_word_count}minwords_{context}context.csv', index=False, quoting=3)
