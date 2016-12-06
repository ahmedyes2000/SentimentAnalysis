from random import shuffle

import os
import numpy as np
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.Analyzer import plot_accuracies, plot_word_embeddings
from src.Corpus import Corpus, ImdbCorpus, ReviewPolarityCorpus, SubjectivityCorpus
from src.Labels import Labels
from src.Tokenizers import AdvancedTokenizer, BigramTokenizer, SimpleTokenizer

from sklearn.manifold import TSNE

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def corpus_exists(corpus: Corpus, number_of_features):
    is_corpus_available = False
    model = None
    corpus_file = "./Corpus/PreGenerated/Doc2Vec/{0}_{1}_{2}".format(corpus.name, corpus.tokenizer.name, number_of_features)

    if os.path.exists(corpus_file):
        logging.log(logging.INFO, "{0} corpus exists".format(corpus.name))
        is_corpus_available = True
        model = Doc2Vec.load(corpus_file)

    return is_corpus_available, model


def save_model(model, corpus: Corpus, number_of_features):
    corpus_file = "./Corpus/PreGenerated/Doc2Vec/{0}_{1}_{2}".format(corpus.name, corpus.tokenizer.name,
                                                                          number_of_features)

    model.save(corpus_file)


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
    logging.log(logging.INFO, "Getting training documents")
    X_train_files, y_train_labels = corpus.get_training_documents(model)

    logging.log(logging.INFO, "Getting testing documents")
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
    # classifier = KNeighborsClassifier(n_neighbors=50)
    classifier = SVC()

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
    review_accuracies = [0.69, 0.6975, 0.77, 0.73, 0.7575, 0.7225, 0.7425, 0.7175, 0.72, 0.69, 0.7125]
    imdb_accuracies = [0.71748, 0.78584, 0.80204, 0.824, 0.82564, 0.83556, 0.8356, 0.84188, 0.84072, 0.8446, 0.8426]
    subjectivity_accuracies = [0.645, 0.645, 0.751, 0.7815, 0.793, 0.798, 0.8115, 0.8095, 0.8245, 0.812, 0.819]
    plot_accuracies("Doc2Vec", "K Nearest Neighbor", "Simple Tokenizer", "# of Neighbors",
                    hyper_parameters, review_accuracies, imdb_accuracies, subjectivity_accuracies)


def visualize():
    number_of_features = 100
    tokenizer = SimpleTokenizer()
    # tokenizer = AdvancedTokenizer()
    # tokenizer = BigramTokenizer()

    # corpus = ReviewPolarityCorpus(tokenizer)
    # corpus = ImdbCorpus(tokenizer)
    corpus = SubjectivityCorpus(tokenizer)

    model = get_model(corpus, number_of_features)

    logging.log(logging.INFO, "Getting training documents")
    X_train_files, y_train_labels = corpus.get_training_documents(model)

    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    X = model.fit_transform(X_train_files)

    plot_word_embeddings("Doc2Vec", corpus.name, "Positive", "Negative", X, y_train_labels)


visualize()
