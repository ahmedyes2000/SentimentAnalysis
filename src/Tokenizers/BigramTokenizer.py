from collections import defaultdict

import nltk

from src.Documents import Document
from src.Tokenizers.Tokenizer import Tokenizer


class BigramTokenizer(Tokenizer):
    @property
    def name(self):
        return 'BigramTokenizer'

    def __init__(self):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def __call__(self, doc):
        return self.tokenize_content(doc)

    def tokenize_content(self, content):
        tokens = self.tokenizer.tokenize(content)
        lowered_tokens = map(lambda t: t.lower(), tokens)
        bigrams = nltk.bigrams(lowered_tokens)
        return list(bigrams)

    def tokenize(self, document: Document):
        bow = defaultdict(float)
        content = document.getContent()
        bigrams = self.tokenize_content(content)
        for bigram in bigrams:
            bow[bigram] += 1.0
        return bow