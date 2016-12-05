import unittest

from src.Analyzer import plot_word_embeddings
from src.Labels import Labels


class AnalyzerTests(unittest.TestCase):

    def test_plot_word_embeddings(self):
        plot_word_embeddings("Doc2Vec", "Test", "Objective", "Subjective", [[-3.99940408, -1.43488923],[ 3.51106635, -2.01347499], [-1.14695001, -2.10861514]],
                             [Labels.strong_pos, Labels.strong_pos, Labels.strong_neg])
