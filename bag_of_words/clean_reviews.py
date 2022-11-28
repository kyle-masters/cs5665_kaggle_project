import pandas as pd
from nltk.corpus import stopwords
import re
from tqdm.auto import tqdm
import pickle
import os

# Function to convert the text of a review to a meaningful string of words
def review_to_words(review_text):
    # remove spoiler tags
    clean_review_text = review_text.replace('(view spoiler)[', '').replace('(hide spoiler)]', '')

    # remove non-letters
    clean_review_text = re.sub("[\W_\d]", "", clean_review_text)

    # convert to lower case, split into individual words
    words = clean_review_text.lower().split()

    # create set from stopwords
    stops = set(stopwords.words("english"))

    # convert back to string and return
    return ' '.join([w for w in words if w not in stops])


def clean_reviews(train_path, test_path):
    os.makedirs('data', exist_ok=True)

    # Read training data
    print('  Reading training data...')
    train = pd.read_csv(train_path, header=0)

    # Clean training review texts and store as list of strings
    clean_train_reviews = []
    print('  Cleaning and parsing the training set reviews...')
    for i_review_text in tqdm(train['review_text'], position=0, desc='  rows'):
        clean_train_reviews.append(review_to_words(i_review_text))

    # Save training review texts and delete training variables
    with open('data/train_reviews.pkl', 'wb') as f:
        pickle.dump(clean_train_reviews, f)
    del train, clean_train_reviews

    # Read test data
    print('  Reading test data...')
    test = pd.read_csv(test_path, header=0)

    # Clean test review texts and store as strings
    clean_test_reviews = []
    print('  Cleaning and parsing the test set reviews...')
    for i_review_text in tqdm(test['review_text'], position=0, desc='  rows'):
        clean_test_reviews.append(review_to_words(i_review_text))

    # Save training review texts and delete training variables
    with open('data/test_reviews.pkl', 'wb') as f:
        pickle.dump(clean_test_reviews, f)
