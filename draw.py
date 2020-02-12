#/https://stackoverflow.com/questions/3100985/plot-with-custom-text-for-x-axis-points
#https://stackoverflow.com/questions/47121997/plot-dictionary-of-dictionaries-in-one-barplot
from review import METRIC_NAMES
def draw(dist, metric):
    import matplotlib.pyplot as plt
    import numpy as np

    metric_index = 0
    for name in METRIC_NAMES:
        if metric == name:
            break
        metric_index += 1
 
    algos = [] 
    for algo_name, _ in dist.items():
        algos.append(algo_name)

    values = []
    for _ in algos:
        values.append(0)

    # zap algos and metric together.
    #first get metric values according to algos.
    for algo, rows in dist.items():
        for met, value in rows.items():
            if met != metric:
                continue
            for i,a in enumerate(algos):
                if a == algo:
                    values[i] = value
                    break
                    
        #print (rows[METRIC_NAMES[0]].key())
    # we are ready with algos and their corresponding metric values.
    x = []
    y = values
    i = 0
    for _ in algos:
        x.append(i)
        i += 1
    plt.xticks(x, algos)
    plt.plot(x, y)
    plt.xlabel(metric + " Score")
    plt.ylabel("Evaluated Algortihms")
    plt.title("using " + metric + " Predicting the Critical Temperature of a Superconductor")
    #plt.show() 
    path = '/Users/madgaikw/Desktop/mat/'
    plt.savefig(path + metric + '.png')
    pass
'''
def draw(dist):
    import matplotlib.pyplot as plt
    import pandas as pd
    pd.DataFrame(dist).plot(kind='bar')
    plt.show() 
'''
#dist = {'Decision Tree Regression': {'max_error': 99.0, 'explained_variance_score': 0.6638964541534156, 'mean_absolute_error': 12.790807127484436, 'mean_squared_error': 398.7790723759616, 'mean_squared_log_error': 0.7380404905042962, 'median_absolute_error': 7.16, 'r2_score': 0.6544541324325486}, 'Bayes Regression': {'max_error': 415.8932128902135, 'explained_variance_score': 0.581361363013618, 'mean_absolute_error': 15.734583659754117, 'mean_squared_error': 483.2261154538001, 'median_absolute_error': 12.062548980258176, 'r2_score': 0.5812799646158209}, 'Linear Regression': {'max_error': 421.96772145461773, 'explained_variance_score': 0.5805617190514802, 'mean_absolute_error': 15.726070842181436, 'mean_squared_error': 484.14894888166646, 'median_absolute_error': 12.032590672577495, 'r2_score': 0.580480320653683}, 'SVM Regression': {'max_error': 101.02626803503213, 'explained_variance_score': 0.7274347380091528, 'mean_absolute_error': 12.320126968626386, 'mean_squared_error': 316.4407621905829, 'median_absolute_error': 7.502504842346067, 'r2_score': 0.725801063096505}}

#for i in METRIC_NAMES:
#draw(dist, 'r2_score')


