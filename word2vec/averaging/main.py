from word2vec import build_model, make_wordlists
from word2vec.averaging import get_averages, predict
import time

# Hyper-parameters
num_features = 300  # Word vector dimensionality
min_word_count = 40  # Minimum word count
context = 10  # Context window size

num_workers = 16  # Number of threads to run in parallel
file_size = 40000  # Number of rows to process at once, smaller will reduce memory and increase storage

train_path = '../../kaggle_files/goodreads_train.csv'  # Path to train file
test_path = '../../kaggle_files/goodreads_test.csv'  # Path to test file
train_size = 900000  # Rows in train file
test_size = 478033  # Rows in test file


def pred_word2vec_averaging(step1=True, step2=True, step3=True, step4=True):
    start = time.time()

    # Build model
    if step1:
        print('\n'
              '==============\n'
              'Building model\n'
              '==============')
        start_step = time.time()
        build_model.build_model(train_path, num_workers, num_features, min_word_count, context)
        print(f'Time taken to build model: {time.time() - start_step:.2f} seconds')

    # Clean and save wordlists
    if step2:
        print('\n'
              '================\n'
              'Making wordlists\n'
              '================')
        start_step = time.time()
        make_wordlists.make_wordlists(train_path, test_path, train_size, test_size, file_size)
        print(f'Time taken to make wordlists: {time.time() - start_step:.2f} seconds')

    # Create average feature vectors
    if step3:
        print('\n'
              '=======================\n'
              'Creating averages array\n'
              '=======================')
        start_step = time.time()
        get_averages.create_averages_list(train_size, test_size, file_size, num_features, min_word_count, context)
        print(f'Time taken to create averages array: {time.time() - start_step:.2f} seconds')

    # Make and save predictions
    if step4:
        print('\n'
              '==================\n'
              'Making predictions\n'
              '==================')
        start_step = time.time()
        predict.predict(train_path, test_path, num_workers, num_features, min_word_count, context)
        print(f'Time taken to make predictions: {time.time() - start_step:.2f} seconds')

    print(f'\n'
          f'Total time taken: {time.time() - start:.2f} seconds')


if __name__ == '__main__':
    pred_word2vec_averaging()
