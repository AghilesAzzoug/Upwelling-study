import numpy as np
import os
import sys
from sklearn.cluster import AgglomerativeClustering

import pandas as pd

sys.path.append("..")
import utils
import config
import UW3_triedctk as ctk



if __name__ == '__main__':
    # todo: get it from CMD
    CASE = 'All'
    SAVE_MODEL = True
    NB_CLASSES = 7
    frlat, tolat, frlon, tolon = utils.get_zone_boundaries(case=CASE)
    MODEL_NAME = 'thetao_Omon_GFDL-CM3_historical_all-rxixpx_197901-200512_selectbox-forclim.nc'

    # read the trained model file
    if CASE.upper() == 'ALL':
        obsSOM, true_labels = utils.loadSOM(save_dir=config.OUTPUT_TRAINED_MODELS_PATH,
                                            file_name=config.ZALL_SOM_3D_MODEL_NAME)
    elif CASE.upper() == 'SEL':
        obsSOM, true_labels = utils.loadSOM(save_dir=config.OUTPUT_TRAINED_MODELS_PATH,
                                            file_name=config.ZSEL_SOM_3D_MODEL_NAME)

    file_path = os.path.join(config.MODELS_DATA_PATH, MODEL_NAME)
    data_label_base, temp, lon, lat, lev = utils.read_data(file_path)
    temp, lon, lat, ilat, ilon = utils.get_zone_obs(temp, lon, lat, size_reduction=CASE, frlon=frlon, tolon=tolon,
                                                    frlat=frlat, tolat=tolat)

    agg_data = utils.aggregate_data(temp, case=CASE)

    all_model_data = agg_data.reshape(12, -1, order='A').T  # shape = 9900 (11*25*36), 12

    model_data = pd.DataFrame.from_records(all_model_data).dropna(axis=0)

    ocean_points_index = model_data.index.values  # ocean points index
    model_data = model_data.values  # train data

    model_labels = np.zeros(shape=(len(all_model_data)), dtype=int)

    hac = AgglomerativeClustering(n_clusters=NB_CLASSES)
    hac = hac.fit(obsSOM.codebook)
    ocean_predicted_labels = utils.get_reverse_classification(ctk.findbmus(sm=obsSOM, Data=model_data),
                                                              hac_labels=hac.labels_)
    model_labels[ocean_points_index] = ocean_predicted_labels.flatten() + 1  # à cause du 0 de la terre

    if CASE.upper() == 'ALL':
        model_labels_ = model_labels.reshape(11, 25, 36, order='A')
    elif CASE.upper() == 'SEL':
        model_labels_ = model_labels.reshape(11, 13, 12, order='A')

    utils.plot_levels_3D_SOM(model_labels_, nb_classes=NB_CLASSES, case=CASE,
                             figure_title=f'Model {data_label_base} (1979-2005), {NB_CLASSES} classes geographical representation',
                             save_file=True, save_dir=config.OUTPUT_FIGURES_PATH, file_name='')

    utils.plot_monthly_anomalies_3D_SOM(temperatures=model_data, labels=ocean_predicted_labels.flatten() + 1,
                                        figure_title=f'Model {data_label_base} (1979-2005). Monthly Mean by Class',
                                        save_file=True, save_dir=config.OUTPUT_FIGURES_PATH, file_name='')

    perf_vector = utils.get_projection_errors(true_labels=true_labels, pred_labels=model_labels_)
    print(f'Perfs. vector : {perf_vector}')