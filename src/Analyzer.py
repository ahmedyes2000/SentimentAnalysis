import os
import matplotlib.pyplot as plt

def plotAccuracies(classifier_name, tokenizerName, alphas, reviewPolarityAccuracies, imdbAccuracies, subjectivityAccuracies):
    results_dir = "../results/{0}".format(classifier_name)

    line_rp, = plt.plot(alphas, reviewPolarityAccuracies, 'r', label='PL04')
    line_imdb, = plt.plot(alphas, imdbAccuracies, 'b', label='IMDB Dataset')
    line_subjectivity, = plt.plot(alphas, subjectivityAccuracies, 'g', label='Subjectivity')

    plt.legend(handles=[line_rp, line_imdb, line_subjectivity], bbox_to_anchor=(0., 1.02, 1., .102), loc=5,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.xlabel('Pseduo counts', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    fig = plt.gcf()
    plt.show()
    plt.draw()
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fig.savefig(os.path.join(results_dir, "{1}_Accuracy.png".format(classifier_name, tokenizerName.replace(" ", "_"))))


def printTable(classifier_name, tokenizerName, alphas, reviewPolarityAccuracies, imdbAccuracies, subjectivityAccuracies):
    print("###{0}".format(tokenizerName))
    print("| Alpha  | PL04 | IMDB Dataset | Subjectivity |")
    print("|---|:---:|:---:|:---:|")
    for i in range(len(alphas)):
        print("| {0}  | {1}  | {2} | {3} |".format(alphas[i], reviewPolarityAccuracies[i], imdbAccuracies[i], subjectivityAccuracies[i]))
    plotAccuracies(classifier_name, tokenizerName, alphas, reviewPolarityAccuracies, imdbAccuracies, subjectivityAccuracies)
