# Bag of Words Representation
## Naive Bayes Classifier

###Simple Tokenizer
| Alpha  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1  | 0.81  | 0.82312 | 0.916083916 |
| 5  | 0.83  | 0.83088 | 0.919080919 |
| 10  | 0.835  | 0.83392 | 0.919080919 |
| 15  | 0.8475  | 0.83532 | 0.918081918 |
| 20  | 0.8475  | 0.83532 | 0.914585415 |
| 25  | 0.85  | 0.83576 | 0.911588412 |
| 30  | 0.85  | 0.83628 | 0.90959041 |
| 35  | 0.845  | 0.83652 | 0.908591409 |

![Simple Tokenizer Accuracy]
(/results/Naive Bayes Classifier/Simple_Tokenizer_Accuracy.png)

###Advanced Tokenizer
| Alpha  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1  | 0.8025  | 0.83256 | 0.8961038961038961 |
| 5  | 0.8225  | 0.84004 | 0.9050949050949051 |
| 10  | 0.83  | 0.844 | 0.9050949050949051 |
| 15  | 0.835  | 0.84448 | 0.9015984015984015 |
| 20  | 0.8425  | 0.84556 | 0.8986013986013986 |
| 25  | 0.8375  | 0.84692 | 0.8966033966033966 |
| 30  | 0.8275  | 0.84772 | 0.8946053946053946 |
| 35  | 0.8075  | 0.84792 | 0.8931068931068931 |


![Advanced Tokenizer Accuracy]
(/results/Naive Bayes Classifier/Advanced_Tokenizer_Accuracy.png)

###Bigram Tokenizer
| Alpha  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1  | 0.8425  | 0.8624 | 0.9015984015984015 |
| 5  | 0.86  | 0.86788 | 0.8981018981018981 |
| 10  | 0.8525  | 0.86712 | 0.8921078921078921 |
| 15  | 0.8525  | 0.86568 | 0.8901098901098901 |
| 20  | 0.83  | 0.86372 | 0.8871128871128872 |
| 25  | 0.83  | 0.86164 | 0.8846153846153846 |
| 30  | 0.8175  | 0.86056 | 0.8826173826173827 |
| 35  | 0.8125  | 0.85956 | 0.8811188811188811 |

![Bigram Tokenizer Accuracy]
(/results/Naive Bayes Classifier/Bigram_Tokenizer_Accuracy.png)

## Multinomial NB Classifier
### Advanced Tokenizer
| Alpha  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1  | 0.815  | 0.83312 | 0.9045954045954046 |
| 5  | 0.7925  | 0.82372 | 0.8926073926073926 |
| 10  | 0.775  | 0.81264 | 0.8846153846153846 |
| 15  | 0.75  | 0.80308 | 0.8821178821178821 |
| 20  | 0.735  | 0.79548 | 0.8791208791208791 |
| 25  | 0.73  | 0.789 | 0.8766233766233766 |
| 30  | 0.725  | 0.78288 | 0.8736263736263736 |
| 35  | 0.7175  | 0.77832 | 0.8736263736263736 |
![Advanced Tokenizer Accuracy]
(/results/Multinomial NB Classifier/Advanced_Tokenizer_Accuracy.png)

## Bernoulli NB Classifier
###Advanced Tokenizer
| Alpha  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1  | 0.8125  | 0.82504 | 0.906093906093906 |
| 5  | 0.7575  | 0.8258 | 0.8911088911088911 |
| 10  | 0.72  | 0.82388 | 0.8786213786213786 |
| 15  | 0.69  | 0.82156 | 0.8646353646353646 |
| 20  | 0.6625  | 0.81916 | 0.8581418581418582 |
| 25  | 0.625  | 0.81588 | 0.8466533466533467 |
| 30  | 0.6025  | 0.8134 | 0.8386613386613386 |
| 35  | 0.59  | 0.81132 | 0.8266733266733267 |
![Advanced Tokenizer Accuracy]
(/results/Multinomial NB Classifier/Advanced_Tokenizer_Accuracy.png)

# Word2Vec Representation
## Logistic Regression
### Simple Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.8625 | 0.88428 | 0.906593406593 |

### Advanced Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.855 | 0.89076 | 0.875624375624 |

## SVC
### Simple Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.5025 | 0.5038 | 0.804195804196 |

### Advanced Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.5 | 0.5038| 0.877622377622 |

## k-Nearest Neighbour Classification
### Simple Tokenizer
| #Neighbours  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1 | 0.675 | 0.74268 | 0.86963036963 |
| 5 | 0.71 | 0.80024 | 0.903096903097 |
| 10 | 0.7425 | 0.81592 | 0.906593406593 |
| 15 | 0.73 | 0.82596 | 0.906093906094 |
| 20 | 0.7275 | 0.83016 | 0.904095904096 |
| 25 | 0.725 | 0.8328 | 0.905594405594 |
| 30 | 0.74 | 0.83512 | 0.907092907093 |
| 35 | 0.745 | 0.83584 | 0.903596403596 |
| 40 | 0.7425 | 0.83564 | 0.907092907093 |
| 45 | 0.735 | 0.8362 | 0.905094905095 |
| 50 | 0.7425 | 0.83524 | 0.905594405594 |
![Simple Tokenizer Accuracy using K-Nearest Neighbors and Word2Vec representation]
(/results/Word2Vec/K Nearest Neighbor/Simple_Tokenizer_Accuracy.png)

### Advanced Tokenizer
| #Neighbours  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1 | 0.7075 | 0.76756 | 0.82967032967 |
| 5 | .72 | 0.8202 | 0.8661338661338661 |
| 10 | 0.7325 | 0.8308 | 0.8621378621378621 |
| 15 | 0.715 | 0.8384 | 0.8706293706293706 |
| 20 | 0.73 | 0.84216 | 0.8701298701298701 |
| 25 | 0.7325 | 0.84268 | 0.8756243756243757 |
| 30 | 0.75 | 0.84436 | 0.8741258741258742 |
| 35 | 0.7225 | 0.84472 | 0.8741258741258742 |
| 40 | 0.735 | 0.84512 | 0.8736263736263736 |
| 45 | 0.72 | 0.84396 | 0.8746253746253746 |
| 50 | 0.7275 | 0.84536 | 0.8756243756243757 |
![Advanced Tokenizer Accuracy using K-Nearest Neighbors and Word2Vec representation]
(/results/Word2Vec/K Nearest Neighbor/Advanced_Tokenizer_Accuracy.png)

Top 10 Words similar to great:

| Word  | Similarity Score |
|---|:---:|
| wonderful | 0.8136018514633179 |
| good | 0.7258837819099426 |
| fantastic | 0.7196279764175415 |
| terrific | 0.6841671466827393 |
| excellent | 0.67384892702102660 |
| fine | 0.6657408475875854 |
| superb | 0.6327412128448486 |
| great, | 0.6117573380470276 |
| fabulous | 0.6105859279632568 |
| marvelous | 0.6080844402313232 |

Top 10 Words similar to bad:

| Word  | Similarity Score |
|---|:---:|
| terrible | 0.7424992322921753 |
| bad. | 0.7385110855102539 |
| bad, | 0.7382776141166687 |
| horrible | 0.7221196889877319 |
| lousy | 0.7122937440872192 |
| awful | 0.6978504657745361 |
| bad! | 0.6827770471572876 |
| good | 0.6407232284545898 |
| awful, | 0.633628785610199 |
| crappy | 0.6150558590888977 |

# Doc2Vec Representation
## Logistic Regression Classification
### Simple Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.865 | 0.87124 | 0.89 |
| 1000 | 0.845| 0 | 0.8885 |

### Advanced Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.8275 | 0.87452| 0.8625 |
| 1000 | 0.85| 0 | 0.8705 |

## SVC
### Simple Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.8475 | 0.8706 | 0.894 |

### Advanced Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.8 | 0.87564| 0.8495 |

## k-Nearest Neighbour Classification
### Simple Tokenizer
| #Neighbours  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1 | 0.69 | 0.71748 | 0.645 |
| 5 | 0.6975 | 0.78584 | 0.645 |
| 10 | 0.77| 0.80204 | 0.751 |
| 15 | 0.73| 0.824 | 0.7815 |
| 20 | 0.7575 |0.82564| 0.793 |
| 25 | 0.7225 | 0.83556| 0.798 |
| 30 | 0.7425 | 0.8356 | 0.8115 |
| 35 | 0.7175 | 0.84188 | 0.8095 |
| 40 | 0.72 | 0.84072 | 0.8245 |
| 45 | 0.69 | 0.8446 | 0.812 |
| 50 | 0.7125 | 0.8426 | 0.819 |
![Simple Tokenizer Accuracy using K-Nearest Neighbors and Word2Vec representation]
(/results/Doc2Vec/K Nearest Neighbor/Simple_Tokenizer_Accuracy.png)

### Advanced Tokenizer
| #Neighbours  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1 | 0.6675 | 0.7238 | 0.6325 |
| 5 |0.6875 | 0.7928 | 0.693 |
| 10 | 0.7325| 0.81356 | 0.71 |
| 15 | 0.725| 0.8278 | 0.7275 |
| 20 | 0.72 |0.83316| 0.7425 |
| 25 | 0.71 | 0.84308| 0.741 |
| 30 | 0.71 | 0.84464| 0.75 |
| 35 | 0.6675 | 0.84884 | 0.74 |
| 40 | 0.715 | 0.8488 | 0.7495 |
| 45 | 0.6875| 0.85236 | 0.747 |
| 50 | 0.7175 | 0.85172 | 0.761 |
![Advanced Tokenizer Accuracy using K-Nearest Neighbors and Word2Vec representation]
(/results/Doc2Vec/K Nearest Neighbor/Advanced_Tokenizer_Accuracy.png)
