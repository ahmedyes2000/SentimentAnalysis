import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from src.Labels import Labels
from src.Tokenizers.AdvancedTokenizer import AdvancedTokenizer
from src.Tokenizers.Tokenizer import Tokenizer

# Path to dataset
PATH_TO_POLARITY_DATA = '../../Datasets/review_polarity/txt_sentoken/'
PATH_TO_IMDB_TEST_DATA = '../../Datasets/aclImdb/test/'
PATH_TO_IMDB_TRAIN_DATA = '../../Datasets/aclImdb/train/'

PATH_TO_SUBJECTIVITY_DATA_SUBJECTIVE = '../../Datasets/rotten_imdb/quote.tok.gt9.5000'
PATH_TO_SUBJECTIVITY_DATA_OBJECTIVE = '../../Datasets/rotten_imdb/plot.tok.gt9.5000'


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

def kFoldCrossValidate(k, classifier, training_set: np.array):
    kf = KFold(n_splits=k)

    for train_index, test_index in kf.split(training_set):
        training_data, test_data = training_set[train_index], training_set[test_index]
        classifier.train_model(training_data.tolist())

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

    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X_train_tfidf, y_obj_train_labels + y_subj_train_labels)

    X_test_counts = count_vectorizer.transform(X_objective_test_data + X_subjective_test_data)
    score = classifier.score(X_test_counts, y_obj_test_labels + y_subj_test_labels)
    print(score)

tokenizer = AdvancedTokenizer()
evaluateSubjectivity(5, tokenizer, None)