import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
import triedpy.triedsompy as SOM
import utils
import config

if __name__ == '__main__':


    CASE = 'All'

    if CASE.upper() == 'ALL':
        perfs_df_path = os.path.join(config.OUTPUT_PERF_PATH, config.ZALL_MODELS_FINAL_PERF_FILE_NAME_CSV)
        nb_classes = 7
    else:
        nb_classes = 7
        perfs_df_path = os.path.join(config.OUTPUT_PERF_PATH, config.ZSEL_MODELS_FINAL_PERF_FILE_NAME_CSV)


    df = pd.read_csv('C:\\Users\\DELL\\PycharmProjects\\PL2020\\output\\perfs\\Final_perf_df_ZALL.csv')

    values = df.values[:, 2:-2] / 100

    sc = StandardScaler()
    sc = sc.fit(values)
    std_values = values.copy() #sc.transform(values)
    # std_values = sc.transform(values)

    kmeans = KMeans(n_clusters=5, max_iter=500, n_init=10).fit(std_values)
    labels = kmeans.labels_

    acp = PCA()
    acp = acp.fit(std_values)
    x2d = acp.transform(std_values)

    plt.scatter(x2d[:, 0], x2d[:, 1], c=labels)

    for x, y, s in zip(x2d[:, 0], x2d[:, 1], df.values[:, 1]):
        plt.text(x=x+0.03, y=y+0.01, s=s, size='small')



    obsx2d = acp.transform((np.array([1 for _ in range(nb_classes)]).reshape(1, -1)))

    plt.scatter(obsx2d[:, 0], obsx2d[:, 1], c='grey')
    plt.text(x=obsx2d[:, 0] + 0.03, y=obsx2d[:, 1] + 0.01, s='Obs', size='small')
    plt.title('Projection des modèles sur les deux premiers axes (ACP)')
    plt.xlabel('Axe 1')
    plt.ylabel('Axe 2')
    plt.xlim((-0.55, 0.72))
    plt.grid(True)
    plt.show()


    plt.bar(np.arange(len(acp.explained_variance_ratio_)) + 1, acp.explained_variance_ratio_ * 100)
    plt.plot(np.arange(len(acp.explained_variance_ratio_)) + 1, np.cumsum(acp.explained_variance_ratio_ * 100),
             'r--o')
    plt.xticks(range(1, acp.n_components_ + 1), ['CP_{}'.format(i) for i in range(1, acp.n_components_ + 1)])
    plt.title('Variance expliquée (somme cumulée)')
    plt.xlabel('Composantes principales')
    plt.ylabel('% de variance expliquée')

    plt.show()