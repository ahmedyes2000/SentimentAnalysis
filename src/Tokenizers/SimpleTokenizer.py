from collections import defaultdict

from src.Documents import Document
from src.Tokenizers.Tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    @property
    def name(self):
        return 'SimpleTokenizer'

    def __call__(self, doc):
        return self.tokenize_content(doc)

    def tokenize_content(self, content):
        tokens = content.split()
        lowered_tokens = map(lambda t: t.lower(), tokens)
        return list(lowered_tokens)


    def tokenize(self, document: Document):
        bow = defaultdict(float)
        content = document.getContent()
        lowered_tokens = self.tokenize_content(content)
        for token in lowered_tokens:
            bow[token] += 1.0
        return bow

