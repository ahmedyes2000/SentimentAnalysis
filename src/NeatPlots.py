import numpy as np
import matplotlib.pyplot as plt
import os

def line_plots(**kwargs):
    print("Unpacking the stats...");
    results_dir = kwargs.pop('results_dir',"");
    data_set_name = kwargs.pop('data_set_name', '')
    title=kwargs.pop('title',"Accuracy Vs Classifiers");
    x_label=kwargs.pop('xlabel','Classifiers');
    y_label=kwargs.pop('ylabel','Accuracy values');
    x_ticks=['TM','NB','AdaBoost','Bagging','RForest','KNN','DT','LR'];
    x=range(len(x_ticks));
    plt.figure(figsize=[8,6],dpi=72);
    keys_=kwargs.keys();
    
    if 'ugrams' in keys_:
        ugrams=kwargs.pop('ugrams');
        plt.plot(x,ugrams,color='red',linestyle='-',linewidth=2.0,label='ugram_counts,lsa');
    if 'ugrams_pa' in keys_:
        ugrams_pa=kwargs.pop('ugrams_pa');
        plt.plot(x,ugrams_pa,color='green',linestyle='--',linewidth=2.0,label='ugram_pa,lsa');
    if 'word2vec' in keys_:
        w2vec=kwargs.pop('word2vec');
        plt.plot(x,w2vec,color='blue',linestyle='-.',linewidth=2.0,label='word2vec_avg,lsa');
    if 'bigrams' in keys_:
        bigrams=kwargs.pop('bigrams')
        plt.plot(x,bigrams,color='orange',linestyle='-',label='bigram_counts,lsa')
    if 'gensim_w2v' in keys_:
        gensim_w2v=kwargs.pop('gensim_w2v')
        plt.plot(x,gensim_w2v,color='black',linestyle='-.',label='gensim_w2v')
    if 'gensim_d2v' in keys_:
        gensim_w2v=kwargs.pop('gensim_d2v')
        plt.plot(x,gensim_d2v,color='brown',linestyle='--',label='gensim_doc2vec,lsa')
    
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=5, ncol=3, mode="expand", borderaxespad=0.)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Shrink current axis by 20%
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.ylim(0,1.0);
    plt.xlim(0,len(x));
    plt.axis('tight');
    plt.title(title, loc='center');
    plt.xlabel(x_label);
    plt.ylabel(y_label);
    plt.xticks(x,x_ticks);
    fig = plt.gcf()
    # fig.suptitle(title, fontsize=14, x=0.5, y=0.005)

    plt.show();
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fig.savefig(os.path.join(results_dir, "Accuracy Vs Classifiers - {0}.png".format(data_set_name)))
#subjectivity dataset statistics

# ['TM', 'NB', 'AdaBoost', 'Bagging', 'RForest', 'KNN', 'DT', 'LR'];
ugram_counts=[0.642,0.51,0.72,0.6,0.62,0.66,0.59,0.574]
ugrams_pa=[0.64,0.51,0.7,0.6,0.63,0.67,0.56,0.573]
word2vec=[0.58,0.48,0.537,0.53,0.515,0.54,0.5,0.51]
bigrams=[0.58,0.5,0.54,0.53,0.51,0.6,0.53,0.57]
gensim_w2v=[0.01, 0.01, 0.01, 0.01, 0.01, 0.907092907093, 0.01, 0.906593406593];
# gensim_d2v=[];

stats_dict={};
stats_dict['results_dir']='../results/'
stats_dict['data_set_name']='Subjectivity'
stats_dict['title']='Accuracy Vs Classifiers Subjectivity Analysis'
stats_dict['ugrams']=ugram_counts;
stats_dict['ugrams_pa']=ugrams_pa;
stats_dict['word2vec']=word2vec;
stats_dict['bigrams']=bigrams;
stats_dict['gensim_w2v']=gensim_w2v;
# stats_dict['gensim_d2v']=gensim_d2v;
line_plots(**stats_dict);         


#sentiment dataset statistics
ugram_counts=[0.642,0.51,0.72,0.6,0.62,0.66,0.59,0.574]
ugrams_pa=[0.64,0.51,0.75,0.6,0.63,0.67,0.56,0.573]
word2vec=[0.58,0.48,0.537,0.53,0.515,0.54,0.5,0.51]
bigrams=[0.58,0.5,0.54,0.53,0.51,0.6,0.53,0.57]
gensim_w2v=[0, 0, 0, 0, 0, 0.75, 0, 0.8625];
# gensim_d2v=[];

stats_dict={};
stats_dict['results_dir']='../results/'
stats_dict['data_set_name']='PL04'
stats_dict['title']='Accuracy Vs Classifiers Sentiment Analysis'
stats_dict['ugrams']=ugram_counts;
stats_dict['ugrams_pa']=ugrams_pa;
stats_dict['word2vec']=word2vec;
stats_dict['bigrams']=bigrams;
stats_dict['gensim_w2v']=gensim_w2v;
# stats_dict['gensim_d2v']=gensim_d2v;
line_plots(**stats_dict);         


