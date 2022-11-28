from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import pandas as pd


def predict(train_path, test_path, num_workers, n_estimators):
    os.makedirs('data', exist_ok=True)

    # Read training data
    print('  Reading data...')
    train = pd.read_csv(train_path, header=0, usecols=['rating'])
    test = pd.read_csv(test_path, header=0, usecols=['review_id'])
    with open('data/train_features.pkl', 'rb') as f:
        train_data_features = pickle.load(f)
    with open('data/test_features.pkl', 'rb') as f:
        test_data_features = pickle.load(f)

    # fit random forest to train data, use to predict test results
    print('  Predicting results (this may take a while)...')
    forest = RandomForestClassifier(n_jobs=num_workers, n_estimators=n_estimators)
    forest = forest.fit(train_data_features, train["rating"])
    result = forest.predict(test_data_features)

    del train, train_data_features, test_data_features

    output = pd.DataFrame(data={"review_id": test["review_id"], "rating": result})
    output.to_csv("submission_bag.csv", index=False)
