import numpy as np
import os
import sys
from sklearn.cluster import AgglomerativeClustering

import pandas as pd

sys.path.append("..")
import triedpy.triedsompy as SOM
import utils
import config
import UW3_triedctk as ctk
import mpi4py


CASE = 'All'
NB_CLASSES = 7
frlat, tolat, frlon, tolon = utils.get_zone_boundaries(case=CASE)

NB_CLASSES = 7

perf_df = pd.read_csv('C:\\Users\\DELL\\PycharmProjects\\PL2020\\output\\perfs\\Perf_df_ZALL.csv')
#print(perf_df.columns)
print(perf_df.head(5))
print(perf_df.values)

exit(1)
perf_df['perf_moyenne'] = perf_df.values[:, 1:].mean(axis=1)
perf_df = perf_df.sort_values(by='perf_moyenne', ascending=False).reset_index(drop=True)
perf_df.columns = ['Modèle'] + [f'perf{i+1}' for i in range(NB_CLASSES)] + ['perf_moyenne']
perf_df['perf_moyenne'] = perf_df['perf_moyenne'].astype(np.float64)
print(perf_df['Modèle'].values)

for index, model_name in enumerate(perf_df['Modèle'].values):
    print(model_name)
    file_name = f'thetao_Omon_{model_name}_historical_all-rxixpx_197901-200512_selectbox-forclim.nc'
    file_path = os.path.join(config.MODELS_DATA_PATH, file_name)
    print(file_path)
    data_label_base, temp, lon, lat, lev = utils.read_data(file_path)
    temp, lon, lat, ilat, ilon = utils.get_zone_obs(temp, lon, lat, size_reduction=CASE, frlon=frlon, tolon=tolon,
                                                    frlat=frlat, tolat=tolat)