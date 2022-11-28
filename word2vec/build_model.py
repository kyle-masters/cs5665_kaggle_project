import re
import nltk
from tqdm.auto import tqdm
from gensim.models import word2vec
import pandas as pd
import os

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Function to convert the text of a review to a list of words
def review_to_wordlist(review_text):
    # remove spoiler tags
    clean_review_text = review_text.replace('(view spoiler)[', '').replace('(hide spoiler)]', '')

    # remove non-letters
    clean_review_text = re.sub("[\W_\d]", "", clean_review_text)

    # convert to lower case, split into individual words
    words = clean_review_text.lower().split()

    return words


# Function to split a review into a list of sentences represented as lists of words
def review_to_sentences(review_text, tokenizer):
    # use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review_text.strip())

    # loop over each sentence, converting it to a list of words
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence))

    return sentences


def build_model(train_path, num_workers, num_features, min_word_count, context):
    os.makedirs(f'data/models', exist_ok=True)

    print('Reading training data')
    train = pd.read_csv(train_path, header=0)

    sentences = []

    print("Parsing sentences from training set")
    for review in tqdm(train["review_text"], position=0, desc='  reviews'):
        sentences += review_to_sentences(review, tokenizer)

    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                vector_size=num_features, min_count=min_word_count,
                window=context)

    model.init_sims(replace=True)
    model.save(f'data/models/word2vec_{num_features}features_{min_word_count}minwords_{context}context')
