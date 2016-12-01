import numpy as np
from gensim.models import Word2Vec
from sklearn.svm import SVC

from src.Corpus.ImdbCorpus import ImdbCorpus
from src.Corpus.ReviewPolarityCorpus import ReviewPolarityCorpus
from src.Corpus.SubjectivityCorpus import SubjectivityCorpus
from src.Tokenizers.AdvancedTokenizer import AdvancedTokenizer
from src.Tokenizers.SimpleTokenizer import SimpleTokenizer


def file_to_vector(files, model, number_of_features):
    '''
    This function is used to convert a list of files into a list of vector representations of the files.
    Each file consisting of tokens is converted to its equivalent vector representation by adding
    the word vectors for the tokens.
    :param files: list of files
    :param model: the word2vec model
    :param number_of_features: the number of features in the model
    :return: a list of files in their vector representation.
    '''
    converted_files = []

    for train_file in files:
        file_vector = np.zeros((number_of_features,),dtype="float32")
        for token in train_file:
            file_vector = np.add(file_vector, model[token])
        converted_files.append(file_vector)

    return converted_files

def evaluate(corpus, number_of_features, classifier):
    '''
    Function to measure the accuracy of a classifier on the imdb data set.
    :return: the accuracy calculated by the classifier
    '''
    model = Word2Vec(corpus, min_count=1, size=number_of_features, workers=4)

    X_train_files, y_train_labels = corpus.get_training_data()
    X_test_files, y_test_labels = corpus.get_test_data()

    X_train_data = file_to_vector(X_train_files, model, number_of_features)
    X_test_data = file_to_vector(X_test_files, model, number_of_features)

    classifier.fit(X_train_data, y_train_labels)
    score = classifier.score(X_test_data, y_test_labels)
    return score

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

classifier = SVC()
number_of_features = 100
# tokenizer = AdvancedTokenizer()
tokenizer = SimpleTokenizer()

corpus = ReviewPolarityCorpus(tokenizer)
# corpus = ImdbCorpus(tokenizer)
# corpus = SubjectivityCorpus(tokenizer)

# review_polarity_accuracy = evaluate_review_polarity()
imdb_accuracy = evaluate(corpus, number_of_features, classifier)
# subjectivity_accuracy = evaluate_subjectivity()
print(imdb_accuracy)

