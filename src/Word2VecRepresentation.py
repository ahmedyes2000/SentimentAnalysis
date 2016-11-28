from gensim.models import Word2Vec

from src.AdvancedTokenizer import AdvancedTokenizer
from src.SubjectivityCorpus import SubjectivityCorpus

import numpy as np
from sklearn.svm import SVC

tokenizer = AdvancedTokenizer()

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

def review_subjectivity():
    '''
    Function to measure the accuracy of a classifier on the review polarity data set.
    :return: the accuracy calculated by the classifier
    '''
    number_of_features = 100
    subjectivity_corpus = SubjectivityCorpus(tokenizer)

    model = Word2Vec(subjectivity_corpus, min_count=1, size=number_of_features)

    X_train_files, y_train_labels = subjectivity_corpus.get_training_data()
    X_test_files, y_test_labels = subjectivity_corpus.get_test_data()

    X_train_data = file_to_vector(X_train_files, model, number_of_features)
    X_test_data = file_to_vector(X_test_files, model, number_of_features)

    classifier = SVC()
    classifier.fit(X_train_data, y_train_labels)
    score = classifier.score(X_test_data, y_test_labels)
    return score

subjectivity_accuracy = review_subjectivity()
print(subjectivity_accuracy)

