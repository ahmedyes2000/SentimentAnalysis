import os

import gensim
from gensim.models.doc2vec import TaggedDocument

from src.Labels import Labels

import numpy as np


class ReviewPolarityCorpus(object):
    PATH_TO_POLARITY_DATA = '../../Datasets/review_polarity/txt_sentoken/'
    POS_LABEL = 'pos'
    NEG_LABEL = 'neg'

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        pos_path = os.path.join(self.PATH_TO_POLARITY_DATA, self.POS_LABEL)
        neg_path = os.path.join(self.PATH_TO_POLARITY_DATA, self.NEG_LABEL)

        pos_data_stream = self.stream_documents(Labels.strong_pos, pos_path, os.listdir(pos_path))
        neg_data_stream = self.stream_documents(Labels.strong_neg, neg_path, os.listdir(neg_path))

        self.X_pos_data, self.y_pos_labels = zip(*pos_data_stream)
        self.X_neg_data, self.y_neg_labels = zip(*neg_data_stream)

    def stream_polarity_documents(self, dir_name):
        """Iterate over documents of the review polarity data set.

        Documents are represented as strings.

        """

        if os.path.exists(dir_name):
            for file_name in os.listdir(dir_name):
                with open(os.path.join(dir_name, file_name), "r") as doc:
                    content = doc.read()
                    yield content

    def stream_documents(self, label, path, file_names):
        """Iterate over documents in the given path.

        Documents are represented as strings.

        """

        if os.path.exists(path):
            for file_name in file_names:
                with open(os.path.join(path, file_name), "r") as doc:
                    content = doc.read()
                    yield self.tokenizer(content), label

    def __iter__(self):
        dir_name = os.path.join(self.PATH_TO_POLARITY_DATA, self.POS_LABEL)
        for file in self.stream_polarity_documents(dir_name):
            yield self.tokenizer(file)

        dir_name = os.path.join(self.PATH_TO_POLARITY_DATA, self.NEG_LABEL)
        for file in self.stream_polarity_documents(dir_name):
            yield self.tokenizer(file)

    def get_training_data(self):

        X_pos_train_data = self.X_pos_data[:800]
        y_pos_train_labels = self.y_pos_labels[:800]

        X_neg_train_data = self.X_neg_data[:800]
        y_neg_train_labels = self.y_neg_labels[:800]

        return X_pos_train_data + X_neg_train_data, y_pos_train_labels + y_neg_train_labels

    def get_training_documents(self, model = gensim.models.Doc2Vec):
        number_of_training_documents = 1600
        mid_way = 800
        train_arrays = np.zeros((number_of_training_documents, model.vector_size))
        train_labels = np.zeros(number_of_training_documents)

        for i in range(mid_way):
            prefix_train_pos = 'strong_pos_' + str(i)
            prefix_train_neg = 'strong_neg_' + str(i)
            train_arrays[i] = model.docvecs[prefix_train_pos]
            train_arrays[mid_way + i] = model.docvecs[prefix_train_neg]
            train_labels[i] = Labels.strong_pos
            train_labels[mid_way + i] = Labels.strong_neg

        return train_arrays, train_labels

    def get_test_data(self):
        X_pos_test_data = self.X_pos_data[800:]
        y_pos_test_labels = self.y_pos_labels[800:]

        X_neg_test_data = self.X_neg_data[800:]
        y_neg_test_labels = self.y_neg_labels[800:]

        return X_pos_test_data + X_neg_test_data, y_pos_test_labels + y_neg_test_labels

    def get_testing_documents(self, model = gensim.models.Doc2Vec):
        number_of_testing_documents = 400
        mid_way = 200
        test_arrays = np.zeros((number_of_testing_documents, model.vector_size))
        test_labels = np.zeros(number_of_testing_documents)

        for i in range(mid_way):
            prefix_train_pos = 'strong_pos_' + str(800 + i)
            prefix_train_neg = 'strong_neg_' + str(800 + i)
            test_arrays[i] = model.docvecs[prefix_train_pos]
            test_arrays[mid_way + i] = model.docvecs[prefix_train_neg]
            test_labels[i] = Labels.strong_pos
            test_labels[mid_way + i] = Labels.strong_neg

        return test_arrays, test_labels

    def to_array(self):
        self.sentences = []

        dir_name = os.path.join(self.PATH_TO_POLARITY_DATA, self.POS_LABEL)
        for idx, file in enumerate(self.stream_polarity_documents(dir_name)):
            self.sentences.append(TaggedDocument(self.tokenizer(file), ['strong_pos_%s' % idx]))

        dir_name = os.path.join(self.PATH_TO_POLARITY_DATA, self.NEG_LABEL)
        for idx, file in enumerate(self.stream_polarity_documents(dir_name)):
            self.sentences.append(TaggedDocument(self.tokenizer(file), ['strong_neg_%s' % idx]))

        return self.sentences

    def sentences_perm(self):
        return np.random.permutation(self.sentences).tolist()