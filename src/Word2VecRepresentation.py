import numpy as np
import os
from random import shuffle

from gensim.models import Word2Vec
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from src.Analyzer import plot_word_embeddings, plot_words
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


def kFoldCrossValidate(k, model, corpus: Corpus, X_train_data, y_train_labels_array, classifier):
    logging.log(logging.INFO, "kFold Cross Validation {0}".format(k))

    scores = cross_val_score(classifier, X_train_data, y_train_labels_array, cv=k)
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

        number_of_features = 100
        # tokenizer = SimpleTokenizer()
        tokenizer = AdvancedTokenizer()
        # tokenizer = BigramTokenizer()

        # corpus = ReviewPolarityCorpus(tokenizer)
        # corpus = ImdbCorpus(tokenizer)
        corpus = SubjectivityCorpus(tokenizer)

        model = get_model(corpus, number_of_features)

        logging.log(logging.INFO, "Getting training documents")
        X_train_files, y_train_labels = corpus.get_training_data()

        X_train_data = file_to_vector(X_train_files, model, number_of_features)
        y_train_labels_array = np.array(y_train_labels)

        for i in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]:
            # classifier = LogisticRegression(C=i)
            # classifier = KNeighborsClassifier(n_neighbors=i)
            # classifier = SVC()
            # classifier = AdaBoostClassifier()
            # classifier = BaggingClassifier(n_estimators=i)
            # classifier = DecisionTreeClassifier(max_depth=i)
            # classifier = RandomForestClassifier(max_depth=i)
            classifier = MultinomialNB(alpha=i)
            # scores = evaluate(model, corpus, number_of_features, classifier)
            scores = kFoldCrossValidate(10, model, corpus, X_train_data, y_train_labels_array, classifier)
            print("{0}, {1}".format(i, scores))

            # import pydotplus
            # dot_data = export_graphviz(classifier, out_file=None)
            # graph = pydotplus.graph_from_dot_data(dot_data)
            # graph.write_pdf("../results/Word2Vec/{0} - {1} - Decision Tree.pdf".format(corpus.name, tokenizer.name))


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

def visualize_words():
    number_of_features = 100
    # tokenizer = SimpleTokenizer()
    tokenizer = AdvancedTokenizer()
    # tokenizer = BigramTokenizer()

    corpus = ReviewPolarityCorpus(tokenizer)
    # corpus = ImdbCorpus(tokenizer)
    # corpus = SubjectivityCorpus(tokenizer)

    model = get_model(corpus, number_of_features)

    logging.log(logging.INFO, "building word vector array")
    words_np = []
    words_label = []
    word_limit = 100
    for word in ["great", "bad"]:
        similar_words = model.similar_by_word(word, 10)
        for similar_word in similar_words:
            words_np.append(model[similar_word[0]])
            words_label.append(similar_word[0])

    logging.log(logging.INFO, "dimensionality reduction")
    tsne_model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    X = tsne_model.fit_transform(words_np)

    logging.log(logging.INFO, "plotting")
    plot_words("Word2Vec", corpus.name, X, words_label, word_limit)

# visualize()
# examine_model()
run_experiment()
# visualize_words()