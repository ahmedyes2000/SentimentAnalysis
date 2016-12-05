import os
import matplotlib.pyplot as plt

def plot_accuracies(representation_model_name, classifier_name, tokenizer_name,
                    parameter_name, hyper_parameters, review_polarity_accuracies, imdb_accuracies, subjectivity_accuracies):
    '''
    This function plots a graph of accuracies vs the hyper parameters for the 3 data sets
    :param representation_model_name: the model used to represent the document
    :param classifier_name: the classifier used to classify and obtain the accuracies
    :param tokenizer_name: the tokenizer that used to create the tokens
    :param parameter_name: the hyper parameter name
    :param hyper_parameters: the hyper parameter values
    :param review_polarity_accuracies: accuracies obtained for the review polarity data set
    :param imdb_accuracies: accuracies obtained for the IMDB data set
    :param subjectivity_accuracies:accuracies obtained for the subjectivity data set
    :return:
    '''
    results_dir = "../results/{0}/{1}".format(representation_model_name, classifier_name)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    line_rp, = plt.plot(hyper_parameters, review_polarity_accuracies, 'r', label='PL04')
    line_imdb, = plt.plot(hyper_parameters, imdb_accuracies, 'b', label='IMDB Dataset')
    line_subjectivity, = plt.plot(hyper_parameters, subjectivity_accuracies, 'g', label='Subjectivity')

    plt.legend(handles=[line_rp, line_imdb, line_subjectivity], bbox_to_anchor=(0., 1.02, 1., .102), loc=5,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.xlabel(parameter_name, fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    fig = plt.gcf()
    plt.show()
    plt.draw()
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fig.savefig(os.path.join(results_dir, "{0}_Accuracy.png".format(tokenizer_name.replace(" ", "_"))))


def print_table(classifier_name, tokenizerName, alphas, reviewPolarityAccuracies, imdbAccuracies, subjectivityAccuracies):
    print("###{0}".format(tokenizerName))
    print("| Alpha  | PL04 | IMDB Dataset | Subjectivity |")
    print("|---|:---:|:---:|:---:|")
    for i in range(len(alphas)):
        print("| {0}  | {1}  | {2} | {3} |".format(alphas[i], reviewPolarityAccuracies[i], imdbAccuracies[i], subjectivityAccuracies[i]))
    plot_accuracies(classifier_name, tokenizerName, alphas, reviewPolarityAccuracies, imdbAccuracies, subjectivityAccuracies)
