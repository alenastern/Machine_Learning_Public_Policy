import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import math
import numpy as np

def dist_table(data_frame):
    return data_frame.describe()

def dist_plot(data_frame):
    plt.rcParams['figure.figsize'] = 16, 12
    data.hist()
    plt.show()

def outliers_plot(data_frame):
    plt.rcParams['figure.figsize'] = 16, 12
    data_frame.plot(kind='box', subplots=True, layout=(4, math.ceil(len(data_frame.columns)/4)), sharex=False, sharey=False)
    plt.show()

def corr_table(data_frame):
    return data_frame.corr(method = 'pearson')

def plot_y_corr(corr_table, y):
    corr_y = corr_table[y]
    title = "Correlation with {}".format(y)
    corr_y.plot.bar(title = title)
    plt.xlabel("Variable")
    plt.show()

def plot_corr_matrix(corr_table):
    names = list(corr_table.index)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_table, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(names),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names,  rotation = 90)
    ax.set_yticklabels(names)
    
    plt.show()
    
    #https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

def plot_by_class(data_frame, y):
    plt.rcParams['figure.figsize'] = 5, 4
    grp = data.groupby(y)
    var = data_frame.columns
    
    for v in var:
        getattr(grp, v).hist(alpha=0.4)
        plt.title(v)
        plt.legend([0,1])
        plt.show()

