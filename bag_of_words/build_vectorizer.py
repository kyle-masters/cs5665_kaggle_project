from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle


def extract_features(n_features):
    os.makedirs('data', exist_ok=True)

    with open('data/train_reviews.pkl', 'rb') as f:
        clean_train_reviews = pickle.load(f)

    # create transformer that uses bag of words
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=n_features)

    # fit vectorizer to train review text and extract features
    print('  Training vectorizer...')
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    del clean_train_reviews
    with open('data/train_features.pkl', 'wb') as f:
        pickle.dump(train_data_features, f)

    with open('data/test_reviews.pkl', 'rb') as f:
        clean_test_reviews = pickle.load(f)

    # extract features from test review text
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    del clean_test_reviews
    with open('data/test_features.pkl', 'wb') as f:
        pickle.dump(test_data_features, f)
