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


if __name__ == '__main__':
    # todo: get it from CMD
    CASE = 'Sel'
    NB_CLASSES = 7
    frlat, tolat, frlon, tolon = utils.get_zone_boundaries(case=CASE)

    # read the trained model file
    # output files (perfs) definition
    if CASE.upper() == 'ALL':
        obsSOM, true_labels = utils.loadSOM(save_dir=config.OUTPUT_TRAINED_MODELS_PATH,
                                            file_name=config.ZALL_SOM_3D_MODEL_NAME)
        perfs_output_file = config.ZALL_MODELS_PERF_FILE_NAME
        cumul_perfs_output_file = config.ZALL_MODELS_CUMUL_PERF_FILE_NAME
        final_perf_output_file = config.ZALL_MODELS_FINAL_PERF_FILE_NAME_CSV
        final_perf_html_output_file = config.ZALL_MODELS_FINAL_PERF_FILE_NAME_HTML
    elif CASE.upper() == 'SEL':
        obsSOM, true_labels = utils.loadSOM(save_dir=config.OUTPUT_TRAINED_MODELS_PATH,
                                            file_name=config.ZSEL_SOM_3D_MODEL_NAME)
        perfs_output_file = config.ZSEL_MODELS_PERF_FILE_NAME
        cumul_perfs_output_file = config.ZSEL_MODELS_CUMUL_PERF_FILE_NAME
        final_perf_output_file = config.ZSEL_MODELS_FINAL_PERF_FILE_NAME_CSV
        final_perf_html_output_file = config.ZSEL_MODELS_FINAL_PERF_FILE_NAME_HTML

    perf_rows = []
    df_index = []

    # Individual perfs !
    for index, model_name in enumerate(os.listdir(config.MODELS_DATA_PATH)):
        # if index == 2:
        #     break

        file_path = os.path.join(config.MODELS_DATA_PATH, model_name)
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

        perf_vector = utils.get_projection_errors(true_labels=true_labels, pred_labels=model_labels_)

        if config.VERBOSE:
            print(f'Perfs. vector for model {data_label_base} : {perf_vector}')
        perf_rows.append(list(perf_vector))
        df_index.append(data_label_base)


    perf_df = pd.DataFrame.from_records(data=perf_rows, index=df_index,
                                          columns=[f'perf_{i + 1}' for i in range(NB_CLASSES)])
    perf_df = perf_df.reset_index()


    perf_df['perf. moyenne'] = perf_df.values[:, 1:].mean(axis=1)
    perf_df = perf_df.sort_values(by='perf. moyenne', ascending=False).reset_index(drop=True)
    perf_df.columns = ['Modèle'] + [f'perf{i+1}' for i in range(NB_CLASSES)] + ['perf. moyenne']
    perf_df['perf. moyenne'] = perf_df['perf. moyenne'].astype(np.float64)

    perf_df.to_csv(os.path.join(config.OUTPUT_PERF_PATH, perfs_output_file))

    # perf_df = pd.read_csv(os.path.join(config.OUTPUT_PERF_PATH, perfs_output_file))  # tweak for index
    if config.VERBOSE:
        print('[+] Computing cumulative performances ...')

    # cumul perfs
    models_values_up_to_now = []  # all temperature tensor up to the current model
    cumul_perf_rows = []
    cumul_df_index = []
    for index, model_name in enumerate(perf_df['Modèle'].values):
        file_name = f'thetao_Omon_{model_name}_historical_all-rxixpx_197901-200512_selectbox-forclim.nc'
        file_path = os.path.join(config.MODELS_DATA_PATH, file_name)

        data_label_base, temp, lon, lat, lev = utils.read_data(file_path)
        temp, lon, lat, ilat, ilon = utils.get_zone_obs(temp, lon, lat, size_reduction=CASE, frlon=frlon, tolon=tolon,
                                                        frlat=frlat, tolat=tolat)


        models_values_up_to_now.append(temp)
        # moyenner tout les modèles jusqu'au modèle courant
        all_values = np.array(models_values_up_to_now)

        temp = np.mean(all_values, axis=0)


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

        perf_vector = utils.get_projection_errors(true_labels=true_labels, pred_labels=model_labels_)

        cumul_perf_rows.append(list(perf_vector))
        cumul_df_index.append(str(index))

    cumul_perf_df = pd.DataFrame.from_records(data=cumul_perf_rows, index=cumul_df_index,
                                          columns=[f'perf_{i + 1}' for i in range(NB_CLASSES)])
    cumul_perf_df.to_csv(os.path.join(config.OUTPUT_PERF_PATH, cumul_perfs_output_file))

    perf_df['perf. cumulée'] = cumul_perf_df.values.mean(axis=1)

    modeles = perf_df['Modèle']
    perf_df = perf_df.drop(columns=['Modèle'])
    perf_df = (perf_df.round(4) * 100)  # .astype(str) + ' %'
    perf_df['Modèle'] = modeles
    perf_df = perf_df[['Modèle'] + [f'perf{i+1}' for i in range(NB_CLASSES)] + ['perf. moyenne'] + ['perf. cumulée']]

    # saving final performances file !
    perf_df.to_csv(os.path.join(config.OUTPUT_PERF_PATH, final_perf_output_file))
    perf_df.to_html(os.path.join(config.OUTPUT_PERF_PATH, final_perf_html_output_file))

    if config.VERBOSE:
        print(f'[+] Eveything saved in {config.OUTPUT_PERF_PATH} directory')