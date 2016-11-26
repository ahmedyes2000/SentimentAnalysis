from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.model_selection import KFold

from src.AdvancedTokenizer import AdvancedTokenizer
from src.Analyzer import printTable
from src.Tokenizer import Tokenizer
from src.Labels import Labels

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import os

# Path to dataset
PATH_TO_POLARITY_DATA = '../../Datasets/review_polarity/txt_sentoken/'
PATH_TO_IMDB_TEST_DATA = '../../Datasets/aclImdb/test/'
PATH_TO_IMDB_TRAIN_DATA = '../../Datasets/aclImdb/train/'

PATH_TO_SUBJECTIVITY_DATA_SUBJECTIVE = '../../Datasets/rotten_imdb/quote.tok.gt9.5000'
PATH_TO_SUBJECTIVITY_DATA_OBJECTIVE = '../../Datasets/rotten_imdb/plot.tok.gt9.5000'

POS_LABEL = 'pos'
NEG_LABEL = 'neg'
K = 5

def getSubjectivityDocuments(label, path):
    documents = []
    with open(path, "r", encoding='ISO-8859-1') as doc:
        content = doc.read()
        files = content.split('\n')
        for file in files:
            documents.append(file)
    return documents

def stream_subjectivity_documents(data_path, label):
    """Iterate over documents of the Subjectivity dataset.

    Documents are represented as strings.

    """

    if os.path.exists(data_path):
        with open(data_path, "r", encoding='ISO-8859-1') as doc:
            content = doc.read()
            files = content.split('\n')
            for file in files:
                yield file, label

def stream_documents(label, path, file_names):
    """Iterate over documents in the given path.

    Documents are represented as strings.

    """

    if os.path.exists(path):
        for file_name in file_names:
            with open(os.path.join(path, file_name), "r") as doc:
                content = doc.read()
                yield content, label


def kFoldCrossValidate(k, classifier, training_set: np.array):
    kf = KFold(n_splits=k)

    for train_index, test_index in kf.split(training_set):
        training_data, test_data = training_set[train_index], training_set[test_index]
        classifier.train_model(training_data.tolist())

def evaluateReviewPolarity(k, tokenizer: Tokenizer, alphas):
    count_vectorizer = CountVectorizer(tokenizer = tokenizer)

    pos_path = os.path.join(PATH_TO_POLARITY_DATA, POS_LABEL)
    neg_path = os.path.join(PATH_TO_POLARITY_DATA, NEG_LABEL)

    # stream documents
    pos_data_stream = stream_documents(Labels.strong_pos, pos_path, os.listdir(pos_path))
    neg_data_stream = stream_documents(Labels.strong_neg, neg_path, os.listdir(neg_path))

    X_pos_data, y_pos_labels = zip(*pos_data_stream)
    X_neg_data, y_neg_labels = zip(*neg_data_stream)

    X_pos_train_data = X_pos_data[:800]
    y_pos_train_labels = y_pos_labels[:800]

    X_neg_train_data = X_neg_data[:800]
    y_neg_train_labels = y_neg_labels[:800]

    X_pos_test_data = X_pos_data[800:]
    y_pos_test_labels = y_pos_labels[800:]

    X_neg_test_data = X_neg_data[800:]
    y_neg_test_labels = y_neg_labels[800:]

    # get vector counts
    X_train_counts = count_vectorizer.fit_transform(X_neg_train_data + X_pos_train_data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    accuracies = []
    for alpha in alphas:
        classifier = BernoulliNB(alpha=alpha)
        classifier.fit(X_train_tfidf, y_neg_train_labels + y_pos_train_labels)
        X_test_counts = count_vectorizer.transform(X_pos_test_data + X_neg_test_data)
        score = classifier.score(X_test_counts, y_pos_test_labels + y_neg_test_labels)
        accuracies.append(score)
    return accuracies

def evaluateIMDB(k, tokenizer: Tokenizer, alphas):
    count_vectorizer = CountVectorizer(tokenizer = tokenizer)

    train_pos_path = os.path.join(PATH_TO_IMDB_TRAIN_DATA, POS_LABEL)
    train_neg_path = os.path.join(PATH_TO_IMDB_TRAIN_DATA, NEG_LABEL)

    train_pos_data_stream = stream_documents(Labels.strong_pos, train_pos_path, os.listdir(train_pos_path))
    train_neg_data_stream = stream_documents(Labels.strong_neg, train_neg_path, os.listdir(train_neg_path))

    X_pos_train_data, y_pos_train_labels = zip(*train_pos_data_stream)
    X_neg_train_data, y_neg_train_labels = zip(*train_neg_data_stream)


    test_pos_path = os.path.join(PATH_TO_IMDB_TEST_DATA, POS_LABEL)
    test_neg_path = os.path.join(PATH_TO_IMDB_TEST_DATA, NEG_LABEL)

    test_pos_data_stream = stream_documents(Labels.strong_pos, test_pos_path, os.listdir(test_pos_path))
    test_neg_data_stream = stream_documents(Labels.strong_neg, test_neg_path, os.listdir(test_neg_path))

    X_pos_test_data, y_pos_test_labels = zip(*test_pos_data_stream)
    X_neg_test_data, y_neg_test_labels = zip(*test_neg_data_stream)

    # get vector counts
    X_train_counts = count_vectorizer.fit_transform(X_neg_train_data + X_pos_train_data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    accuracies = []
    for alpha in alphas:
        classifier = BernoulliNB(alpha=alpha)
        classifier.fit(X_train_tfidf, y_neg_train_labels + y_pos_train_labels)
        X_test_counts = count_vectorizer.transform(X_pos_test_data + X_neg_test_data)
        score = classifier.score(X_test_counts, y_pos_test_labels + y_neg_test_labels)
        accuracies.append(score)
    return accuracies

def evaluateSubjectivity(k, tokenizer: Tokenizer, alphas):
    count_vectorizer = CountVectorizer(tokenizer = tokenizer)
    objective_data_stream = stream_subjectivity_documents(PATH_TO_SUBJECTIVITY_DATA_OBJECTIVE, Labels.strong_pos)
    subjective_data_stream = stream_subjectivity_documents(PATH_TO_SUBJECTIVITY_DATA_SUBJECTIVE, Labels.strong_neg)

    X_objective_data, y_obj_labels = zip(*objective_data_stream)
    X_subjective_data, y_subj_labels = zip(*subjective_data_stream)

    X_objective_train_data = X_objective_data[:4000]
    y_obj_train_labels = y_obj_labels[:4000]

    X_subjective_train_data = X_subjective_data[:4000]
    y_subj_train_labels = y_subj_labels[:4000]

    X_objective_test_data = X_objective_data[4000:]
    y_obj_test_labels = y_obj_labels[4000:]

    X_subjective_test_data = X_subjective_data[4000:]
    y_subj_test_labels = y_subj_labels[4000:]

    # get vector counts
    X_train_counts = count_vectorizer.fit_transform(X_objective_train_data + X_subjective_train_data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    accuracies = []
    for alpha in alphas:
        classifier = BernoulliNB(alpha=alpha)
        classifier.fit(X_train_tfidf, y_obj_train_labels + y_subj_train_labels)
        X_test_counts = count_vectorizer.transform(X_objective_test_data + X_subjective_test_data)
        score = classifier.score(X_test_counts, y_obj_test_labels + y_subj_test_labels)
        accuracies.append(score)
    return accuracies

alphas = [1, 5, 10, 15, 20, 25, 30, 35]
tokenizer = AdvancedTokenizer()
review_polarity_accuracies = evaluateReviewPolarity(K, tokenizer, alphas)
imdb_accuracies = evaluateIMDB(K, tokenizer, alphas)
subjectivity_accuracies = evaluateSubjectivity(K, tokenizer, alphas)

# reviewPolarityAccuracies = [0.81, 0.83, 0.835, 0.8475, 0.8475, 0.85, 0.85, 0.845]
# imdbAccuracies = [0.82312, 0.83088, 0.83392, 0.83532, 0.83532, 0.83576, 0.83628, 0.83652]
# subjectivityAccuracies = [0.916083916, 0.919080919, 0.919080919, 0.918081918, 0.914585415, 0.911588412, 0.90959041, 0.908591409]
printTable("Bernoulli NB Classifier", "Advanced Tokenizer", alphas, review_polarity_accuracies, imdb_accuracies, subjectivity_accuracies)