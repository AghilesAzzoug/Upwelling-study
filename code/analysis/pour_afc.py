#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

sys.path.append("..")
import triedpy.triedsompy as SOM
import utils
import config
import gaft
import UW3_triedctk as ctk

CASE = 'All'

if CASE.upper() == 'ALL':
    perfs_df_path = os.path.join(config.OUTPUT_PERF_PATH, config.ZALL_MODELS_FINAL_PERF_FILE_NAME_CSV)
    nb_classes = 7
else:
    nb_classes = 7
    perfs_df_path = os.path.join(config.OUTPUT_PERF_PATH, config.ZSEL_MODELS_FINAL_PERF_FILE_NAME_CSV)

perf_data = pd.read_csv('C:\\Users\\DELL\\PycharmProjects\\PL2020\\output\\perfs\\Final_perf_df_ZALL.csv')
TTperf = perf_data.values[:, 2:-2].astype(np.int)


def afaco(X, dual=True, Xs=None):
    F2V = CAj = F1sU = None
    m, p = np.shape(X)
    N = np.sum(X)
    Fij = X / N
    fip = np.sum(Fij, axis=1)
    fpj = np.sum(Fij, axis=0)
    F1a = Fij.T / fip
    F1a = F1a.T
    F1 = F1a / np.sqrt(fpj)

    sqrtfipfpj = np.sqrt(np.outer(fip, fpj))
    M = Fij / sqrtfipfpj
    T = np.dot(M.T, M)
    VAPT, VEPT = np.linalg.eig(T)
    # Ordonner selon les plus grandes valeurs propres
    idx = sorted(range(len(VAPT)), key=lambda k: VAPT[k], reverse=True)
    VAPT = VAPT[idx]
    VEPT = VEPT[:, idx]
    U = VEPT[:, 1:p]
    F1U = np.dot(F1, U)
    VAPT = VAPT[1:p]
    if dual:
        VBPTU = np.sqrt(VAPT) * U
        F2V = VBPTU.T / np.sqrt(fpj)
        F2V = F2V.T
    #
    # Contribution Absolue ligne
    # A    = (F1U**2).T*fip
    F1U2T = (F1U ** 2).T
    A = F1U2T * fip
    CAi = A.T / VAPT
    if dual:
        # Contribution Absolue colonne
        A = (F2V ** 2).T * fpj
        CAj = A.T / VAPT;
    #
    # Contribution Relative ligne
    d2pIG = np.sum((F1 - np.sqrt(fpj)) ** 2, axis=1)
    CRi = (F1U2T / d2pIG).T
    #    
    # Individus supplémentaires
    if Xs is not None:
        Fijs = Xs / N
        fips = np.sum(Fijs, axis=1)
        F1as = Fijs.T / fips
        F1as = F1as.T
        F1s = F1as / np.sqrt(fpj)
        F1sU = np.dot(F1s, U)
    #
    return VAPT, F1U, CAi, CRi, F2V, CAj, F1sU


AFCWITHOBS = False

# TTperf
# Out[38]: 
# array([[100,  89,  71,  84,  70,  58,  70],
#        [100,  67,  84,  57, 100,  21,  61],
#        [100,  61,  89,  60,  94,   8,  48],
#        [ 79,  64,  87,  52,  96,  21,  57],
#        [ 95,  70,  80,  76,  40,  29,  52],
#        [100,  69,  79,  71,  88,   4,  17],
#        [ 79,  75,  87,  47, 100,   0,   0],
#        ...
#        [100,  48,  58,   6,   2,   0,   0],
#        [ 84,  80,  32,   8,   0,   0,   0]])

Tp_ = TTperf  # Pourcentages des biens classés par classe (transformé ci après
# pour NIJ==3 en effectif après éventuel ajout des obs dans l'afc)
# On supprime les lignes dont la somme est 0 car l'afc n'aime pas ca.
# (en espérant que la suite du code reste cohérente !!!???)
som_ = np.sum(Tp_, axis=1)
Iok_ = np.where(som_ > 0)[0]  # Indice des modèles valides pour l'AFC
Tp_ = Tp_[Iok_]

Pobs_ = np.ones((1, nb_classes), dtype=int) * 100;  # perfs des OBS = 100% dans toutes les classes
#
if AFCWITHOBS:  # On ajoute Obs (si required)
    print("** do_afc: ajoute les OBS au tableau pour l'AFC **\n")
    Tp_ = np.concatenate((Tp_, Pobs_), axis=0);  # je mets les Obs A LA FIN #a

TTperf4afc = Tp_
# _________________________
# Faire l'AFC proprment dit
print('-- dimensions de la matrice pour afaco ... [{}]'.format(TTperf4afc.shape))
if AFCWITHOBS:
    VAPT, F1U, CAi, CRi, F2V, CAj, F1sU = afaco(TTperf4afc)
else:  # Les obs en supplémentaires
    VAPT, F1U, CAi, CRi, F2V, CAj, F1sU = afaco(TTperf4afc, Xs=Pobs_)
#
# Le retour principal de cette fonction est 
#  - VAPT ... l'inercie par axe
#  - F1U .... les valeurs des points dans les nouvels axes

print('-- dimensions de la matrice F1A de l\'AFC qui en resulte ... [{}]'.format(F1U.shape))
print('-- pourcentage d''inercie des valeurs propres ... [', end='')
[print(' {:.1%} '.format(VAPT[i] / np.sum(VAPT)), end='') for i in np.arange(len(VAPT))]
print(']')
# print(F2V.shape)#, CAi, CRi, F2V, CAj, F1sU)
# exit(1)
# plt.scatter(F1U[:, 0], F1U[:, 1])
# plt.show()

kmeans = KMeans(n_clusters=5, max_iter=500, n_init=10).fit(F1U)
labels = kmeans.labels_

plt.scatter(F1U[:, 0], F1U[:, 1], c=labels)

for x, y, s in zip(F1U[:, 0], F1U[:, 1], perf_data.values[:, 1]):
    plt.text(x=x + 0.01, y=y + 0.01, s=s, size='small')

# indiv supp.
plt.scatter(F1sU[:, 0], F1sU[:, 1], c='grey')
plt.text(x=F1sU[:, 0] + 0.012, y=F1sU[:, 1] + 0.01, s='Obs', size='small')
# variables

plt.scatter(F2V[:, 0], F2V[:, 1], c='red', marker='*')
for x, y, s in zip(F2V[:, 0], F2V[:, 1], [f'C_{i+1}' for i in range(nb_classes)]):
    plt.text(x=x - 0.01, y=y + 0.019, s=s, size='small')

plt.title('Projection des modèles sur les deux premiers axes (ACM)')
plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.xlim((-0.87, 0.4))
plt.grid(True)
plt.show()
