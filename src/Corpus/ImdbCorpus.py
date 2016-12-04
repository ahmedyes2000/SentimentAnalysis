import os

import gensim
from gensim.models.doc2vec import TaggedDocument

from src.Corpus.Corpus import Corpus
from src.Labels import Labels

import numpy as np


class ImdbCorpus(Corpus):
    PATH_TO_IMDB_TEST_DATA = '../../Datasets/aclImdb/test/'
    PATH_TO_IMDB_TRAIN_DATA = '../../Datasets/aclImdb/train/'

    POS_LABEL = 'pos'
    NEG_LABEL = 'neg'

    @property
    def name(self):
        return 'IMDB'

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def stream_imdb_documents(self, dir_name):
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
        dir_name = os.path.join(self.PATH_TO_IMDB_TRAIN_DATA, self.POS_LABEL)
        for file in self.stream_imdb_documents(dir_name):
            yield self.tokenizer(file)

        dir_name = os.path.join(self.PATH_TO_IMDB_TRAIN_DATA, self.NEG_LABEL)
        for file in self.stream_imdb_documents(dir_name):
            yield self.tokenizer(file)

        dir_name = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.POS_LABEL)
        for file in self.stream_imdb_documents(dir_name):
            yield self.tokenizer(file)

        dir_name = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.NEG_LABEL)
        for file in self.stream_imdb_documents(dir_name):
            yield self.tokenizer(file)

    def get_training_data(self):

        pos_path = os.path.join(self.PATH_TO_IMDB_TRAIN_DATA, self.POS_LABEL)
        neg_path = os.path.join(self.PATH_TO_IMDB_TRAIN_DATA, self.NEG_LABEL)

        pos_data_stream = self.stream_documents(Labels.strong_pos, pos_path, os.listdir(pos_path))
        neg_data_stream = self.stream_documents(Labels.strong_neg, neg_path, os.listdir(neg_path))

        X_pos_data, y_pos_labels = zip(*pos_data_stream)
        X_neg_data, y_neg_labels = zip(*neg_data_stream)

        return X_pos_data + X_neg_data, y_pos_labels + y_neg_labels

    def get_training_documents(self, model=gensim.models.Doc2Vec):
        pos_dir_name = os.path.join(self.PATH_TO_IMDB_TRAIN_DATA, self.POS_LABEL)
        neg_dir_name = os.path.join(self.PATH_TO_IMDB_TRAIN_DATA, self.NEG_LABEL)

        number_of_pos_training_documents = len(os.listdir(pos_dir_name))
        number_of_neg_training_documents = len(os.listdir(neg_dir_name))
        total_training_documents = number_of_pos_training_documents + number_of_neg_training_documents

        train_arrays = np.zeros((total_training_documents, model.vector_size))
        train_labels = np.zeros(total_training_documents)

        for i in range(number_of_pos_training_documents):
            prefix_test_pos = 'train_pos_' + str(i)
            train_arrays[i] = model.docvecs[prefix_test_pos]
            train_labels[i] = Labels.strong_pos

        for i in range(number_of_neg_training_documents):
            prefix_test_neg = 'train_neg_' + str(i)
            train_arrays[number_of_pos_training_documents + i] = model.docvecs[prefix_test_neg]
            train_labels[number_of_pos_training_documents + i] = Labels.strong_neg

        return train_arrays, train_labels

    def get_test_data(self):
        pos_path = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.POS_LABEL)
        neg_path = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.NEG_LABEL)

        pos_data_stream = self.stream_documents(Labels.strong_pos, pos_path, os.listdir(pos_path))
        neg_data_stream = self.stream_documents(Labels.strong_neg, neg_path, os.listdir(neg_path))

        X_pos_data, y_pos_labels = zip(*pos_data_stream)
        X_neg_data, y_neg_labels = zip(*neg_data_stream)

        return X_pos_data + X_neg_data, y_pos_labels + y_neg_labels

    def get_testing_documents(self, model=gensim.models.Doc2Vec):

        pos_dir_name = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.POS_LABEL)
        neg_dir_name = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.NEG_LABEL)

        number_of_pos_testing_documents = len(os.listdir(pos_dir_name))
        number_of_neg_testing_documents = len(os.listdir(neg_dir_name))
        total_testing_documents = number_of_pos_testing_documents + number_of_neg_testing_documents

        test_arrays = np.zeros((total_testing_documents, model.vector_size))
        test_labels = np.zeros(total_testing_documents)

        for i in range(number_of_pos_testing_documents):
            prefix_test_pos = 'test_pos_' + str(i)
            test_arrays[i] = model.docvecs[prefix_test_pos]
            test_labels[i] = Labels.strong_pos

        for i in range(number_of_neg_testing_documents):
            prefix_test_neg = 'test_neg_' + str(i)
            test_arrays[number_of_pos_testing_documents + i] = model.docvecs[prefix_test_neg]
            test_labels[number_of_pos_testing_documents + i] = Labels.strong_neg

        return test_arrays, test_labels

    def to_array(self):
        self.sentences = []

        dir_name = os.path.join(self.PATH_TO_IMDB_TRAIN_DATA, self.POS_LABEL)
        for idx, file in enumerate(self.stream_imdb_documents(dir_name)):
            self.sentences.append(TaggedDocument(self.tokenizer(file), ['train_pos_%s' % idx]))

        dir_name = os.path.join(self.PATH_TO_IMDB_TRAIN_DATA, self.NEG_LABEL)
        for idx, file in enumerate(self.stream_imdb_documents(dir_name)):
            self.sentences.append(TaggedDocument(self.tokenizer(file), ['train_neg_%s' % idx]))

        dir_name = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.POS_LABEL)
        for idx, file in enumerate(self.stream_imdb_documents(dir_name)):
            self.sentences.append(TaggedDocument(self.tokenizer(file), ['test_pos_%s' % idx]))

        dir_name = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.NEG_LABEL)
        for idx, file in enumerate(self.stream_imdb_documents(dir_name)):
            self.sentences.append(TaggedDocument(self.tokenizer(file), ['test_neg_%s' % idx]))

        return self.sentences

    def sentences_perm(self):
        return np.random.permutation(self.sentences).tolist()
