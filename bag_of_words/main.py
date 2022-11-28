import nltk
import time
import local_utils
from bag_of_words import clean_reviews, build_vectorizer, predict
nltk.download('stopwords')

# Hyper-parameters
n_features = 2000
n_estimators = 100

num_workers = 16

train_path = '../kaggle_files/goodreads_train.csv'
test_path = '../kaggle_files/goodreads_test.csv'


def pred_bag_of_words(step1=True, step2=True, step3=True):
    start = time.time()

    if step1:
        local_utils.print_text('Cleaning reviews', border='=')
        start_step = time.time()
        clean_reviews.clean_reviews(train_path, test_path)
        local_utils.print_time('clean reviews', start_step)

    if step2:
        local_utils.print_text('Extracting features', border='=')
        start_step = time.time()
        build_vectorizer.extract_features(n_features)
        local_utils.print_time('extract features', start_step)

    if step3:
        local_utils.print_text('Making predictions', border='=')
        start_step = time.time()
        predict.predict(train_path, test_path, num_workers, n_estimators)
        local_utils.print_time('make predictions', start_step)

    local_utils.print_time('predict using bag of words', start)


if __name__ == '__main__':
    pred_bag_of_words()
