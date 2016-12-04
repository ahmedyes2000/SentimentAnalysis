import unittest

from collections import defaultdict

from src.Documents import SentenceDocument
from src.Tokenizers.BigramTokenizer import BigramTokenizer


class BigramTokenizerTests(unittest.TestCase):

    def setUp(self):
        self.tokenizer = BigramTokenizer()

    def test_tokenize_content(self):
        tokens = self.tokenizer("This is a test.")
        expected_tokens = [ ("this", "is"), ("is", "a"), ("a", "test"), ("test", ".") ]
        self.assertEqual(tokens, expected_tokens)

    def test_tokenize(self):
        document = SentenceDocument("This is a test.")
        tokens = self.tokenizer.tokenize(document)
        expected_tokens = defaultdict(float,{
            ("test", "."): 1.0,
            ("this", "is"): 1.0,
            ("is", "a"): 1.0,
            ("a", "test"): 1.0
        })
        self.assertEqual(tokens, expected_tokens)
