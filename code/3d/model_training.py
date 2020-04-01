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
    CASE = 'All' # All for full study zone and Sel for the restricted zone
    SAVE_MODEL = False
    NB_CLASSES = 7
    frlat, tolat, frlon, tolon = utils.get_zone_boundaries(case=CASE)
    file_path = os.path.join(config.OBS_DATA_PATH, config.USED_MODEL_FILE_NAME)
    data_label_base, temp, lon, lat, lev = utils.read_data(file_path)
    temp, lon, lat, ilat, ilon = utils.get_zone_obs(temp, lon, lat, size_reduction=CASE, frlon=frlon, tolon=tolon,
                                                    frlat=frlat, tolat=tolat)

    agg_data = utils.aggregate_data(temp, case=CASE)

    all_train_data = agg_data.reshape(12, -1, order='A').T  # shape = 9900 (11*25*36), 12

    train_data = pd.DataFrame.from_records(all_train_data).dropna(axis=0)

    ocean_points_index = train_data.index.values  # ocean points index
    train_data = train_data.values  # train data

    predicted_labels = np.zeros(shape=(len(all_train_data)), dtype=int)

    # print(np.count_nonzero(np.isnan(train_data)) / np.count_nonzero(~np.isnan(train_data)) * 100)  # 21.80 % de nan
    sMapO, eqO, etO = utils.do_ct_map_process(Dobs=train_data, name=None, mapsize=[11, 100], tseed=0, norm_method='var',
                                              initmethod='pca', neigh=None,
                                              varname=None, step1=[100, 7, 2], step2=[100, 2, 0.1], verbose='on',
                                              retqerrflg=False)

    hac = AgglomerativeClustering(n_clusters=NB_CLASSES)
    hac = hac.fit(sMapO.codebook)
    ocean_predicted_labels = utils.get_reverse_classification(ctk.findbmus(sm=sMapO, Data=train_data),
                                                              hac_labels=hac.labels_)
    predicted_labels[ocean_points_index] = ocean_predicted_labels.flatten() + 1  # à cause du 0 de la terre

    if CASE.upper() == 'ALL':
        predicted_labels_ = predicted_labels.reshape(11, 25, 36, order='A')
        if SAVE_MODEL:
            utils.saveSOM(som_object=sMapO, true_labels=predicted_labels, save_dir=config.OUTPUT_TRAINED_MODELS_PATH,
                          file_name=config.ZALL_SOM_3D_MODEL_NAME)

    elif CASE.upper() == 'SEL':
        predicted_labels_ = predicted_labels.reshape(11, 13, 12, order='A')
        if SAVE_MODEL:
            utils.saveSOM(som_object=sMapO, true_labels=predicted_labels, save_dir=config.OUTPUT_TRAINED_MODELS_PATH,
                          file_name=config.ZSEL_SOM_3D_MODEL_NAME)

    # classification plots
    utils.plot_levels_3D_SOM(predicted_labels_, nb_classes=NB_CLASSES, case=CASE,
                             figure_title=f'Observations (1979-2005), {NB_CLASSES} classes geographical representation',
                             save_file=True, save_dir=config.OUTPUT_FIGURES_PATH, file_name='')

    # monthly anomalies plot
    utils.plot_monthly_anomalies_3D_SOM(temperatures=train_data, labels=ocean_predicted_labels.flatten() + 1,
                                        figure_title='Observation (1979-2005). Monthly Mean by Class', save_file=True,
                                        save_dir=config.OUTPUT_FIGURES_PATH, file_name='')

    # dendrogram plot
    utils.do_plot_ct_dendrogram(sMapO, nb_class=NB_CLASSES)