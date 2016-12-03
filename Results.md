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
## Support Vector Classification
### Simple Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.5025 | 0.0 | 0.696303696304 |
| 1000 | 0.5 | 0.0 | 0.58991008991 |
| 10000 | 0.5275 | 0 | 0.555444555445 |

### Advanced Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.505 | 0.5038 | 0.521478521479 |
| 1000 | 0.5625 | 0.53692 | 0.509490509491 |
| 10000 | 0.62 | 0 | 0.502997002997 |

Words similar to great: [('time', 0.9997875690460205), ('film', 0.9997649192810059), ('love', 0.999764621257782), ('get', 0.9997416734695435), ('one', 0.9997367262840271), ('story', 0.999735951423645), ('movie', 0.9997340440750122), ('world', 0.9997314214706421), ('best', 0.9997295141220093), ('little', 0.9997293949127197)]
words similar to bad: [('movie', 0.9997878670692444), ('film', 0.9997793436050415), ('nt', 0.9997758865356445), ('much', 0.9997627139091492), ('like', 0.9997615814208984), ('make', 0.9997614026069641), ('would', 0.9997589588165283), ('world', 0.9997525215148926), ('good', 0.9997518062591553), ('one', 0.9997491836547852)]

# Doc2Vec Representation
## Logistic Regression Classification
### Advanced Tokenizer
| #Features  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 100 | 0.8275 | 0| 0.8625 |
| 1000 | 0| 0 | 0 |
| 10000 | 0 | 0 | 0 |
