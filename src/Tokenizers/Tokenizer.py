from abc import ABCMeta, abstractmethod

from src.Documents import Document


class Tokenizer(metaclass=ABCMeta):
    '''
    This is an abstract tokenizer.
    '''
    @abstractmethod
    def tokenize(self, document: Document):
        raise NotImplementedError("Abstract Tokenizer needs to be implemented")

    @property
    def name(self):
        raise NotImplementedError("Abstract Tokenizer needs to be implemented")

    @abstractmethod
    def __call__(self, doc):
        raise NotImplementedError("Abstract Tokenizer needs to be implemented")

    @abstractmethod
    def tokenize_content(self, content):
        raise NotImplementedError("Abstract Tokenizer needs to be implemented")
