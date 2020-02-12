import copy
import draw
def draw(dist, metric):
    import matplotlib.pyplot as plt
    import numpy as np
    METRIC_NAMES = ["max_error", "explained_variance_score", "mean_absolute_error", "mean_squared_error", "mean_squared_log_error", "median_absolute_error", "r2_score", "mean_poi    sson_deviance", "mean_gamma_deviance", "mean_tweedie_deviance"]
    metric_index = 0
    for name in METRIC_NAMES:
        if metric == name:
            break
        metric_index += 1

    algos = []
    print('XXXXXXXXXXXXXXXXXXX')
    print(dist) 
    for algo_name, _ in dist.items():
        algos.append(algo_name)

    values = []
    for _ in algos:
        values.append(0)

    # zap algos and metric together.
    #first get metric values according to algos.
    print('XXXXXXXXXXXXXXXXXXX')
    print(dist)
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
    plt.xlabel(metric + " Evaluated Algorithms")
    plt.ylabel("Score")
    plt.title("using " + metric + "\nPredicting the Critical Temperature \nof a Superconductor")
    #plt.show()
    path = '/Users/madgaikw/Desktop/mat/'
    plt.savefig(path + metric + '.png')

t_dict = {'RF': {'max_error': 97.85210517806547, 'explained_variance_score': 0.6638308683130739, 'mean_absolute_error': 14.10445955481363, 'mean_squared_error': 387.9762786499197, 'mean_squared_log_error': 0.7106648559251469, 'median_absolute_error': 8.373137724915281, 'r2_score': 0.663814856173583}, 'DecTree': {'max_error': 99.0, 'explained_variance_score': 0.6638964541534156, 'mean_absolute_error': 12.790807127484436, 'mean_squared_error': 398.7790723759616, 'mean_squared_log_error': 0.7380404905042962, 'median_absolute_error': 7.16, 'r2_score': 0.6544541324325486}, 'Bayes': {'max_error': 415.8932128902135, 'explained_variance_score': 0.581361363013618, 'mean_absolute_error': 15.734583659754117, 'mean_squared_error': 483.2261154538001, 'median_absolute_error': 12.062548980258176, 'r2_score': 0.5812799646158209}, 'Linear': {'max_error': 421.96772145461773, 'explained_variance_score': 0.5805617190514802, 'mean_absolute_error': 15.726070842181436, 'mean_squared_error': 484.14894888166646, 'median_absolute_error': 12.032590672577495, 'r2_score': 0.580480320653683}, 'DecPCA': {'max_error': 180.6, 'explained_variance_score': 0.5767013333593827, 'mean_absolute_error': 13.684549979014514, 'mean_squared_error': 491.635877290735, 'mean_squared_log_error': 0.6703157925388494, 'median_absolute_error': 6.300000000000001, 'r2_score': 0.5739928258182271}, 'SVM': {'max_error': 101.02626803503213, 'explained_variance_score': 0.7274347380091528, 'mean_absolute_error': 12.320126968626386, 'mean_squared_error': 316.4407621905829, 'median_absolute_error': 7.502504842346067, 'r2_score': 0.725801063096505}, 'XGboost': {'max_error': 99.22866439819336, 'explained_variance_score': 0.5766363033906383, 'mean_absolute_error': 18.276881892231692, 'mean_squared_error': 705.398066564147, 'mean_squared_log_error': 0.6292213773849121, 'median_absolute_error': 9.249156646728515, 'r2_score': 0.3887658511289418}, 'SVM-RBF': {'max_error': 176.03963778650416, 'explained_variance_score': 0.6267557889909896, 'mean_absolute_error': 11.558084166187866, 'mean_squared_error': 431.7465158387176, 'median_absolute_error': 5.784240744689654, 'r2_score': 0.6258875284105627}}


def filter_dict(value, t):
    out = {}
    inner = {}
    for num, i in enumerate(t):
        #print('num ' + str(num))
        #print(t[i])
        #print(type(t[i]))
        for num1, j in enumerate(t[i]):
            #print('num1 ' + str(num1))
            #print (j)
            #print(t[i][j])
            if j == value:
                inner[value] = t[i][j]
                out[i] = copy.deepcopy(inner)
                inner = {}
                #print(t[i][j])
    return out



max_error_dict = filter_dict('max_error', copy.deepcopy(t_dict))
#print(max_error_dict)
draw(max_error_dict, 'max_error')


explained_variance_score_dict = filter_dict('explained_variance_score', copy.deepcopy(t_dict))
#print(explained_variance_score_dict)
draw(explained_variance_score_dict, 'explained_variance_score')


mean_absolute_error_dict = filter_dict('mean_absolute_error', copy.deepcopy(t_dict))
#print(mean_absolute_error_dict)
draw(mean_absolute_error_dict, 'mean_absolute_error')

mean_squared_error_dict = filter_dict('mean_squared_error', copy.deepcopy(t_dict))
print(mean_squared_error_dict)
draw(mean_squared_error_dict, 'mean_squared_error')

mean_squared_log_error_dict = filter_dict('mean_squared_log_error', copy.deepcopy(t_dict))
#print(mean_squared_log_error_dict)
draw(mean_squared_log_error_dict, 'mean_squared_log_error')

median_absolute_error_dict = filter_dict('median_absolute_error', copy.deepcopy(t_dict))
#print(median_absolute_error_dict)
draw(median_absolute_error_dict, 'median_absolute_error')


r2_score_dict = filter_dict('r2_score', t_dict)
#print(r2_score_dict)
draw(r2_score_dict, 'r2_score')


