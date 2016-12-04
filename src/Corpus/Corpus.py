from abc import ABCMeta, abstractmethod

import gensim


class Corpus(metaclass=ABCMeta):

    @property
    def name(self):
        raise NotImplementedError("Abstract Corpus needs to be implemented")

    @abstractmethod
    def __init__(self, tokenizer):
        raise NotImplementedError("Abstract Corpus needs to be implemented")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError("Abstract Corpus needs to be implemented")

    @abstractmethod
    def get_training_data(self):
        raise NotImplementedError("Abstract Corpus needs to be implemented")

    @abstractmethod
    def get_training_documents(self, model = gensim.models.Doc2Vec):
        raise NotImplementedError("Abstract Corpus needs to be implemented")

    @abstractmethod
    def get_test_data(self):
        raise NotImplementedError("Abstract Corpus needs to be implemented")

    @abstractmethod
    def get_testing_documents(self, model = gensim.models.Doc2Vec):
        raise NotImplementedError("Abstract Corpus needs to be implemented")

    @abstractmethod
    def to_array(self):
        raise NotImplementedError("Abstract Corpus needs to be implemented")

    @abstractmethod
    def sentences_perm(self):
        raise NotImplementedError("Abstract Corpus needs to be implemented")
