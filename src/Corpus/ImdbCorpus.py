import os

from src.Labels import Labels


class ImdbCorpus(object):
    PATH_TO_IMDB_TEST_DATA = '../../Datasets/aclImdb/test/'
    PATH_TO_IMDB_TRAIN_DATA = '../../Datasets/aclImdb/train/'

    POS_LABEL = 'pos'
    NEG_LABEL = 'neg'

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

    def get_test_data(self):
        pos_path = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.POS_LABEL)
        neg_path = os.path.join(self.PATH_TO_IMDB_TEST_DATA, self.NEG_LABEL)

        pos_data_stream = self.stream_documents(Labels.strong_pos, pos_path, os.listdir(pos_path))
        neg_data_stream = self.stream_documents(Labels.strong_neg, neg_path, os.listdir(neg_path))

        X_pos_data, y_pos_labels = zip(*pos_data_stream)
        X_neg_data, y_neg_labels = zip(*neg_data_stream)


        return X_pos_data + X_neg_data, y_pos_labels + y_neg_labels
