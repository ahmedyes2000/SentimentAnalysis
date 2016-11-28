import os

from src.Labels import Labels


class SubjectivityCorpus(object):
    PATH_TO_SUBJECTIVITY_DATA_SUBJECTIVE = '../../Datasets/rotten_imdb/quote.tok.gt9.5000'
    PATH_TO_SUBJECTIVITY_DATA_OBJECTIVE = '../../Datasets/rotten_imdb/plot.tok.gt9.5000'

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        objective_data_stream = self.stream_documents(self.PATH_TO_SUBJECTIVITY_DATA_OBJECTIVE, Labels.strong_pos)
        subjective_data_stream = self.stream_documents(self.PATH_TO_SUBJECTIVITY_DATA_SUBJECTIVE, Labels.strong_neg)

        self.X_objective_data, self.y_obj_labels = zip(*objective_data_stream)
        self.X_subjective_data, self.y_subj_labels = zip(*subjective_data_stream)

    def stream_subjectivity_documents(self, data_path):
        """Iterate over documents of the Subjectivity dataset.

        Documents are represented as strings.

        """

        if os.path.exists(data_path):
            with open(data_path, "r", encoding='ISO-8859-1') as doc:
                content = doc.read()
                files = content.split('\n')
                for file in files:
                    yield file

    def stream_documents(self, data_path, label):
        """Iterate over documents of the Subjectivity dataset.

        Documents are represented as strings.

        """

        if os.path.exists(data_path):
            with open(data_path, "r", encoding='ISO-8859-1') as doc:
                content = doc.read()
                files = content.split('\n')
                for file in files:
                    yield self.tokenizer(file), label

    def __iter__(self):
        dir_name = self.PATH_TO_SUBJECTIVITY_DATA_OBJECTIVE

        for file in self.stream_subjectivity_documents(dir_name):
            yield self.tokenizer(file)

        dir_name = self.PATH_TO_SUBJECTIVITY_DATA_SUBJECTIVE
        for file in self.stream_subjectivity_documents(dir_name):
            yield self.tokenizer(file)

    def get_training_data(self):

        X_objective_train_data = self.X_objective_data[:4000]
        y_obj_train_labels = self.y_obj_labels[:4000]

        X_subjective_train_data = self.X_subjective_data[:4000]
        y_subj_train_labels = self.y_subj_labels[:4000]

        return X_objective_train_data + X_subjective_train_data, y_obj_train_labels + y_subj_train_labels

    def get_test_data(self):
        X_objective_test_data = self.X_objective_data[4000:]
        y_obj_test_labels = self.y_obj_labels[4000:]

        X_subjective_test_data = self.X_subjective_data[4000:]
        y_subj_test_labels = self.y_subj_labels[4000:]

        return X_objective_test_data + X_subjective_test_data, y_obj_test_labels + y_subj_test_labels
