from random import shuffle

import numpy as np
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
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
    model = Doc2Vec(min_count=1, window=10, size=number_of_features, sample=1e-4, negative=5, workers=7)

    sentences_list = corpus.to_array()
    model.build_vocab(sentences_list)

    Idx = list(range(len(sentences_list)))
    for epoch in range(10):
        shuffle(Idx)
        perm_sentences = [sentences_list[i] for i in Idx]
        model.train(perm_sentences)

    X_train_files, y_train_labels = corpus.get_training_documents(model)
    X_test_files, y_test_labels = corpus.get_testing_documents(model)

    classifier.fit(X_train_files, y_train_labels)
    score = classifier.score(X_test_files, y_test_labels)
    return score

def examine_model(corpus, number_of_features):
    model = Word2Vec(corpus, min_count=1, size=number_of_features, workers=4)
    print("Words similar to great:", model.similar_by_word("great", 10))
    print("words similar to bad:", model.similar_by_word("bad", 10))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

classifier = LogisticRegression()
number_of_features = 1000
# tokenizer = AdvancedTokenizer()
tokenizer = SimpleTokenizer()

# corpus = ReviewPolarityCorpus(tokenizer)
# corpus = ImdbCorpus(tokenizer)
corpus = SubjectivityCorpus(tokenizer)

accuracy = evaluate(corpus, number_of_features, classifier)
print(accuracy)

# examine_model(corpus, number_of_features)
