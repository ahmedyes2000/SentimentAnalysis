from random import shuffle

import os
import numpy as np
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.Analyzer import plot_accuracies
from src.Corpus import Corpus, ImdbCorpus, ReviewPolarityCorpus, SubjectivityCorpus
from src.Tokenizers import AdvancedTokenizer, BigramTokenizer, SimpleTokenizer

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def corpus_exists(corpus: Corpus, number_of_features):
    is_corpus_available = False
    model = None
    corpus_file = "./Corpus/PreGenerated/{0}_{1}_{2}".format(corpus.name, corpus.tokenizer.name, number_of_features)

    if os.path.exists(corpus_file):
        logging.log(logging.INFO, "{0} corpus exists".format(corpus.name))
        is_corpus_available = True
        model = Doc2Vec.load(corpus_file)

    return is_corpus_available, model


def save_model(model, corpus: Corpus, number_of_features):
    subjectivity_corpus_file = "./Corpus/PreGenerated/{0}_{1}_{2}".format(corpus.name, corpus.tokenizer.name,
                                                                          number_of_features)

    model.save(subjectivity_corpus_file)


def get_model(corpus: Corpus, number_of_features):
    exists, model = corpus_exists(corpus, number_of_features)

    if not exists:
        logging.log(logging.INFO, "Creating Doc2Vec model")
        model = Doc2Vec(min_count=1, window=10, size=number_of_features, sample=1e-4, negative=5, workers=7)

        logging.log(logging.INFO, "fetching sentences list")
        sentences_list = corpus.to_array()

        logging.log(logging.INFO, "building vocabulary")
        model.build_vocab(sentences_list)

        Idx = list(range(len(sentences_list)))
        for epoch in range(10):
            logging.log(logging.INFO, "running epoch %s" % epoch)
            shuffle(Idx)
            perm_sentences = [sentences_list[i] for i in Idx]
            model.train(perm_sentences)

        save_model(model, corpus, number_of_features)
    return model


def evaluate(model, corpus: Corpus, number_of_features, classifier):
    '''
    Function to measure the accuracy of a classifier on the imdb data set.
    :return: the accuracy calculated by the classifier
    '''
    X_train_files, y_train_labels = corpus.get_training_documents(model)
    X_test_files, y_test_labels = corpus.get_testing_documents(model)

    logging.log(logging.INFO, "Training classifier")
    classifier.fit(X_train_files, y_train_labels)

    logging.log(logging.INFO, "Evaluating score")
    score = classifier.score(X_test_files, y_test_labels)
    return score


def examine_model(corpus: Corpus, number_of_features):
    model = Word2Vec(corpus, min_count=1, size=number_of_features, workers=4)
    print("Words similar to great:", model.similar_by_word("great", 10))
    print("words similar to bad:", model.similar_by_word("bad", 10))


def run_experiment():
    # classifier = LogisticRegression()
    classifier = KNeighborsClassifier(n_neighbors=50)

    number_of_features = 100
    # tokenizer = SimpleTokenizer()
    tokenizer = AdvancedTokenizer()
    # tokenizer = BigramTokenizer()

    # corpus = ReviewPolarityCorpus(tokenizer)
    corpus = ImdbCorpus(tokenizer)
    # corpus = SubjectivityCorpus(tokenizer)

    model = get_model(corpus, number_of_features)

    accuracy = evaluate(model, corpus, number_of_features, classifier)
    print(accuracy)


def plot_results():
    hyper_parameters = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    review_accuracies = [0.6675, 0.6875, 0.7325, 0.725, 0.72, 0.71, 0.71, 0.6675, 0.715, 0.6875, 0.7175]
    imdb_accuracies = [0.7238, 0.7928, 0.81356, 0.8278, 0.83316, 0.84308, 0.84464, 0.84884, 0.8488, 0.85236, 0.85172]
    subjectivity_accuracies = [0.6325, 0.693, 0.71, 0.7275, 0.7425, 0.741, 0.75, 0.74, 0.7495, 0.747, 0.761]
    plot_accuracies("Doc2Vec", "K Nearest Neighbor", "Advanced Tokenizer", "# of Neighbors",
                     hyper_parameters, review_accuracies, imdb_accuracies, subjectivity_accuracies)

plot_results()
