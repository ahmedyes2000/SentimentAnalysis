import numpy as np
import os
from random import shuffle

from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.Analyzer import plot_word_embeddings
from src.Corpus import Corpus, ImdbCorpus, ReviewPolarityCorpus, SubjectivityCorpus
from src.Labels import Labels
from src.Tokenizers import AdvancedTokenizer, BigramTokenizer, SimpleTokenizer

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
        file_vector = np.zeros((number_of_features,), dtype="float32")
        for token in train_file:
            file_vector = np.add(file_vector, model[token])
        converted_files.append(file_vector)

    return np.array(converted_files)


def corpus_exists(corpus: Corpus, number_of_features):
    is_corpus_available = False
    model = None
    corpus_file = "./Corpus/PreGenerated/Word2Vec/{0}_{1}_{2}".format(corpus.name, corpus.tokenizer.name,
                                                                      number_of_features)

    if os.path.exists(corpus_file):
        logging.log(logging.INFO, "{0} corpus exists".format(corpus.name))
        is_corpus_available = True
        model = Word2Vec.load(corpus_file)

    return is_corpus_available, model


def get_model(corpus: Corpus, number_of_features):
    exists, model = corpus_exists(corpus, number_of_features)

    if not exists:
        logging.log(logging.INFO, "Creating Word2Vec model")
        model = Word2Vec(min_count=1, window=10, size=number_of_features, sample=1e-4, negative=5, workers=7)

        logging.log(logging.INFO, "fetching sentences list")
        sentences_list = []
        for sentence in corpus:
            sentences_list.append(sentence)

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


def save_model(model, corpus: Corpus, number_of_features):
    corpus_file = "./Corpus/PreGenerated/Word2Vec/{0}_{1}_{2}".format(corpus.name, corpus.tokenizer.name,
                                                                      number_of_features)

    model.save(corpus_file)


def kFoldCrossValidate(k, model, corpus: Corpus, number_of_features, classifier):
    logging.log(logging.INFO, "kFold Cross Validation {0}".format(k))
    kf = KFold(n_splits=k)
    scores = []

    logging.log(logging.INFO, "Getting training documents")
    X_train_files, y_train_labels = corpus.get_training_data()

    X_train_data = file_to_vector(X_train_files, model, number_of_features)
    y_train_labels_array = np.array(y_train_labels)

    fold = 0
    for train_index, test_index in kf.split(X_train_data):
        logging.log(logging.INFO, "Fold # {0}".format(fold))
        training_data, test_data = X_train_data[train_index], X_train_data[test_index]
        training_labels, test_labels = y_train_labels_array[train_index], y_train_labels_array[test_index]
        classifier.fit(training_data, training_labels)
        score = classifier.score(test_data, test_labels)
        scores.append(score)
        fold = fold + 1
    return scores


def evaluate(model, corpus: Corpus, number_of_features, classifier):
    '''
    Function to measure the accuracy of a classifier on the data set.
    :return: the accuracy calculated by the classifier
    '''
    logging.log(logging.INFO, "Getting training documents")
    X_train_files, y_train_labels = corpus.get_training_data()

    logging.log(logging.INFO, "Getting testing documents")
    X_test_files, y_test_labels = corpus.get_test_data()

    X_train_data = file_to_vector(X_train_files, model, number_of_features)
    X_test_data = file_to_vector(X_test_files, model, number_of_features)

    logging.log(logging.INFO, "Training classifier")
    classifier.fit(X_train_data, y_train_labels)

    logging.log(logging.INFO, "Evaluating score")
    score = classifier.score(X_test_data, y_test_labels)
    return score


def examine_model():
    number_of_features = 100
    # tokenizer = SimpleTokenizer()
    tokenizer = AdvancedTokenizer()
    # tokenizer = BigramTokenizer()

    # corpus = ReviewPolarityCorpus(tokenizer)
    corpus = ImdbCorpus(tokenizer)
    # corpus = SubjectivityCorpus(tokenizer)

    model = get_model(corpus, number_of_features)

    print("Words similar to great:", model.similar_by_word("great", 10))
    print("words similar to bad:", model.similar_by_word("bad", 10))

    words = []
    word_sentiment = []
    for word, score in model.similar_by_word("great", 10):
        words.append(model[word])
        word_sentiment.append(Labels.strong_pos)

    for word, score in model.similar_by_word("bad", 10):
        words.append(model[word])
        word_sentiment.append(Labels.strong_neg)

    tsne_model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    X = tsne_model.fit_transform(words)

    plot_word_embeddings("Word2Vec", corpus.name, "Positive", "Negative", X, word_sentiment)


def run_experiment():
    # classifier = LogisticRegression()
    for i in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        classifier = KNeighborsClassifier(n_neighbors=i)
        # classifier = SVC()

        number_of_features = 100
        # tokenizer = SimpleTokenizer()
        tokenizer = AdvancedTokenizer()
        # tokenizer = BigramTokenizer()

        # corpus = ReviewPolarityCorpus(tokenizer)
        corpus = ImdbCorpus(tokenizer)
        # corpus = SubjectivityCorpus(tokenizer)

        model = get_model(corpus, number_of_features)

        # accuracy = evaluate(model, corpus, number_of_features, classifier)
        accuracy = kFoldCrossValidate(10, model, corpus, number_of_features, classifier)
        print("Accuracy for {0} = {1}".format(i, accuracy))


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
    X_train_files, y_train_labels = corpus.get_training_data()
    X_train_data = file_to_vector(X_train_files, model, number_of_features)

    tsne_model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    X = tsne_model.fit_transform(X_train_data)

    plot_word_embeddings("Word2Vec", corpus.name, "Objective", "Subjective", X, y_train_labels)


# visualize()
# examine_model()
run_experiment()