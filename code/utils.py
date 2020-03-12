import numpy as np
import sys, os
from time import time
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
import matplotlib
import pandas as pd
import pickle

import UW3_triedctk as ctk
import netCDF4
# PARAMETRAGE DU CAS
from config import *

import triedpy.triedsompy as SOM

np.set_printoptions(threshold=sys.maxsize)


def read_data(data_path, verbose=VERBOSE):
    """
    Lis et formate les données (modèles ou observations)

    :param data_path: chemin du fichier (format netCDF4)
    :param verbose: verbosité de la fonction (obtenu par défaut à partir du fichier config.py)
    :type data_path: str
    :type verbose: bool
    :return: data_label_base (nom du modèle ou observation), temp(températures), lon(longitude), lat (latitude),
    lev (profondeur)
    """

    # récupérer le nom du fichier
    # TODO: à modifier
    if 'thetao_Omon' in os.path.basename(data_path):
        data_field_name = 'thetao'
        data_label_base = data_path.split('_')[2]
    else:
        data_label_base = data_path.split('_')[1]
        data_field_name = 'to'

    nc = netCDF4.Dataset(data_path)
    liste_var = nc.variables  # mois par mois de janvier 1979 à decembre 2005 (time, lev, lat, lon)

    temp_var = liste_var[data_field_name]  # 2005 - 1979 + 1 = 27 ; 27 * 12 = 324

    temp = temp_var[:]  # np.shape = (324, 11, 25, 36) = (Obs, Can, Lobs, Cobs)

    temp = temp.filled(np.nan)

    lat = liste_var['lat'][:]
    lon = liste_var['lon'][:]
    lev = liste_var['lev'][:]

    # SI masked_array alors on recupare uniquement la data, on neglige le mask
    if isinstance(np.ma.array(lat), np.ma.MaskedArray):
        lat = lat.data
    if isinstance(np.ma.array(lon), np.ma.MaskedArray):
        lon = lon.data
    if isinstance(np.ma.array(lev), np.ma.MaskedArray):
        lev = lev.data

    # Correction : conversion en np.array
    lon = np.array(lon)
    lat = np.array(lat)
    lev = np.array(lev)

    if np.max(lon) > 180:  # ATTENTION, SI UN > 180 on considere TOUS > 180 ...
        print("-- LONGITUDE [0:360] to [-180:180] simple conversion ...")
        lon -= 360

    if verbose:
        print("\nData ({}x{}): {}".format(len(lat), len(lon), data_label_base))
        print(" - Dir : {}".format(os.path.dirname(data_path)))
        print(" - File: {}".format(os.path.basename(data_path)))
        print(" - dimensions of data : {}".format(temp.shape))
        print(" - Lat : {} values from {} to {}".format(len(lat), lat[0], lat[-1]))
        print(" - Lon : {} values from {} to {}".format(len(lon), lon[0], lon[-1]))
        print(" - Lev : {} values from {} to {}".format(len(lev), lev[0], lev[-1]))

    return data_label_base, temp, lon, lat, lev


def get_zone_obs(temp, lon, lat, size_reduction='All', frlon=None, tolon=None, frlat=None, tolat=None, verbose=VERBOSE):
    """
    Data cropping for getting either the full study zone or the restricted zone

    :param temp: temperature data
    :param lon: longitude
    :param lat: latitude
    :param size_reduction: 'All' for the "full study zone", 'Sel' for the "restricted zone"
    :param frlon: minimal longitude
    :param tolon: maximal longitude
    :param frlat: minimal latitude
    :param tolat: maximal latitude

    :return: temperatures, latitudes, longitudes and depths
    """

    if frlon is None and tolon is None:
        frlon = np.min(lon)
        tolon = np.max(lon) + 1
    if frlat is None and tolat is None:
        if lat[0] > lat[1]:
            frlat = np.max(lat)
            tolat = np.min(lat) - 1
        else:
            frlat = np.min(lat)
            tolat = np.max(lat) + 1

    if verbose:
        print("\nCurrent geographical limits ('to' limit excluded):")
        print(" - size_reduction is '{}'".format(size_reduction))
        print(" - Lat : from {} to {}".format(frlat, tolat))
        print(" - Lon : from {} to {}".format(frlon, tolon))

    # selectionne les LON et LAT selon les limites definies dans ctConfig.py
    # le fait pour tout cas de SIZE_REDUCTION, lat et lon il ne devraient pas
    # changer dans le cas de SIZE_REDUCTION=='All'
    ilat = np.intersect1d(np.where(lat <= frlat), np.where(lat > tolat))
    ilon = np.intersect1d(np.where(lon >= frlon), np.where(lon < tolon))
    lat = lat[np.intersect1d(np.where(lat <= frlat), np.where(lat > tolat))]
    lon = lon[np.intersect1d(np.where(lon >= frlon), np.where(lon < tolon))]

    # Prendre d'entrée de jeu une zone delimitee
    temp = temp[:, :, ilat, :]
    temp = temp[:, :, :, ilon]

    if verbose:
        print("\nDefinitive data:")
        print(" - Dimensions of data : {}".format(temp.shape))
        print(" - Lat : {} values from {} to {}".format(len(lat), lat[0], lat[-1]))
        print(" - Lon : {} values from {} to {}".format(len(lon), lon[0], lon[-1]))

    return temp, lon, lat, ilat, ilon


def aggregate_data(temp, case='All'):
    """
    Agréger les données et calculer les anomalies

    :param temp: temperatures tensor (from utils.get_zone_obs)
    :param case: selection parameter, 'All' for the full study zone, 'Sel' for the restricted one

    :return: anomalies tensor
    """

    temp_ = temp.copy()
    if case.upper() == 'ALL':
        temp_ = np.reshape(temp_, newshape=(12, 27, 11, 25, 36), order='F')
    else:
        temp_ = np.reshape(temp_, newshape=(12, 27, 11, 13, 12), order='F')

    annual_means = np.nanmean(temp_, axis=0)  # (27, 11, 25, 36)

    annomalies = temp_ - annual_means  # (12, 27, 11, 25, 36)

    out_data = np.nanmean(annomalies, axis=1)  # (12, 11, 25, 36)

    return out_data


def do_ct_map_process(Dobs, name=None, mapsize=None, tseed=SEED, norm_method=None, initmethod=None, neigh=None,
                      varname=None, step1=[5, 5, 2], step2=[5, 2, 0.1], verbose=VERBOSE, retqerrflg=False):
    """
    Training of a SOM on temperature data

    :param Dobs: train data
    :param name: SOM name
    :param mapsize: size of the SOM [W, H] = W * H
    :param tseed: default seed (from config.py)
    :param norm_method: data normalization
    :param initmethod: initialization method for SOM weights ('random' or 'pca')
    :param neigh:
    :param varname: name of variables
    :param step1: parameters for the 1st training pass : [number of epochs, initialization radius, end of training radius]
    :param step2: parameters for the 2nt training pass : [number of epochs, initialization radius, end of training radius]
    :param verbose: verbosity (from config.py)
    :param retqerrflg:

    :return: the SOM model, quantification error, topological error
    """

    if verbose:
        print("Initializing random generator with seed={}".format(tseed))
        print("tseed=", tseed)
    np.random.seed(tseed)

    # Création de la structure de la carte
    if verbose:
        smap_verbose = 'on'
    else:
        smap_verbose = 'off'
    if name is None:
        name = 'sMapObs'

    if norm_method is None:
        norm_method = 'data'  # je n'utilise pas 'var' mais je fais centred Ã 
        # la place (ou pas) qui est Ã©quivalent, mais qui
        # me permetde garder la maitrise du codage

    if initmethod is None:
        initmethod = 'random'  # peut etre ['random', 'pca']

    if neigh is None:
        neigh = 'Guassian'  # peut etre ['Bubble', 'Guassian'] (sic !!!)

    # Initialisation de la SOM ________________________________________________
    sMapO = SOM.SOM(name, Dobs,
                    mapsize=mapsize,
                    norm_method=norm_method,
                    initmethod='random',
                    varname=varname)

    # Apprentissage de la carte _______________________________________________
    ttrain0 = time()

    # Entrainenemt de la SOM __________________________________________________
    eqO = sMapO.train(etape1=step1, etape2=step2, verbose=smap_verbose, retqerrflg=retqerrflg)

    if verbose:
        print("Training elapsed time {:.4f}s".format(time() - ttrain0))

    # + err topologique
    bmus2O = ctk.mbmus(sMapO, Data=None, narg=2)
    etO = ctk.errtopo(sMapO, bmus2O)  # dans le cas 'rect' uniquement

    if verbose:
        print("Two phases training executed:")
        print(" - Phase 1: {0[0]} epochs for radius varying from {0[1]} to {0[2]}".format(step1))
        print(" - Phase 2: {0[0]} epochs for radius varying from {0[1]} to {0[2]}".format(step2))

    return sMapO, eqO, etO


def get_reverse_classification(bmus, hac_labels):
    """
    Get reverse classification from BMUS and HAC labels

    :param bmus: BMUS for an SOM
    :param hac_labels: HAC classification labels for SOM neurons

    :return: classification for each initial data point, represented by its BMU
    """
    return np.array([hac_labels[bmu] for bmu in bmus])


def plot_levels_3D_SOM(labels, nb_classes=8, figure_title='3D SOM', save_file=True, save_dir=OUTPUT_FIGURES_PATH,
                       file_name=''):
    """
    Plot the 11 depths level from a SOM trained over 3D data

    :param labels: classification labels (after HAC)
    :param nb_classes: number of classes used (class 0 represents the earth)
    :param figure_title: figure title
    :param save_file: whether or not to save the file to png format
    :param save_dir: output directory (default value in config.py)
    :param file_name : file name for png output file
    """

    fig, axs = plt.subplots(3, 4)
    cmap = plt.get_cmap('jet', nb_classes + 1)  # + 1 because of the earth

    for index, level in enumerate(labels):
        classes = labels[index]

        f = axs[index // 4, index % 4].imshow(classes, cmap=cmap, interpolation='none', vmin=0, vmax=nb_classes)
        plt.colorbar(f, ax=axs[index // 4, index % 4], ticks=np.arange(0, nb_classes + 1), shrink=0.7)
        axs[index // 4, index % 4].set_title(f'{index * 20 + 10}m')

    fig.delaxes(axs[2][3])
    fig.suptitle(figure_title)

    if save_file:
        if file_name == '':
            # if the name is not specified, use the title
            file_name = '{}.png'.format('_'.join(figure_title.split(' ')))

        output_path = os.path.join(save_dir, file_name)
        plt.savefig(output_path)

    plt.show()


def plot_monthly_anomalies_3D_SOM(temperatures, labels, figure_title='Monthly anomalies', save_file=True,
                                  save_dir=OUTPUT_FIGURES_PATH, file_name=''):
    """
    Plot monthly anomalies for an SOM result in 3D case

    :param: temperatures: tensor of temperatures
    :param: labels: labels after applying HAC over SOM results
    :param: figure_title: figure title
    :param save_file: whether or not to save the file to png format
    :param save_dir: output directory (default value in config.py)
    :param file_name : file name for png output file
    """

    cmap = plt.get_cmap('jet', max(labels.flatten()) + 1)

    plt.figure(figsize=(17, 8))

    for label in list(set(list(labels))):
        values = temperatures[labels == label]
        monthy_anom = values.mean(axis=0)
        monthy_dev = values.std(axis=0)
        # plt.plot(monthy_anom, label=f'classe = {label}', color=cmap(label))
        plt.errorbar(list(range(12)), monthy_anom, monthy_dev, label=f'class = {label}', color=cmap(label))

    plt.title(figure_title)
    plt.legend(loc='best')
    plt.ylim(-5, 5)
    plt.grid(True)
    plt.xticks(range(12), ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
    plt.ylabel('Mean temperature anomaly [°C]')
    plt.xlabel('Month')

    if save_file:
        if file_name == '':
            # if the name is not specified, use the title
            file_name = '{}.png'.format('_'.join(figure_title.split(' ')))
        output_path = os.path.join(save_dir, file_name)
        plt.savefig(output_path)

    plt.show()


def saveSOM(som_object, true_labels, save_dir=OUTPUT_TRAINED_MODELS_PATH, file_name='', verbose=VERBOSE):
    """
    Save a SOM model and its labels in pickle format (version 4.0)

    :param som_object: the SOM model
    :param true_labels: true labels after HAC classification
    :param save_dir: output directory (default OUTPUT_TRAINED_MODELS_PATH in config.py)
    :param file_name: output file name (without .pkl extension)
    :param verbose: verbosity (default value in config.py)
    """
    full_path = os.path.join(save_dir, f'{file_name}.pkl')
    with open(full_path, 'wb') as file_handler:
        pickle.dump({'som': som_object, 'labels': true_labels}, file_handler)

    if verbose:
        print(f'[+] Saving SOM object and it\'s labels in path : {full_path}')


def loadSOM(save_dir, file_name, verbose=VERBOSE):
    """
    Load a SOM model and its labels in pickle format (version 4.0)

    :param save_dir: saving directory (default OUTPUT_TRAINED_MODELS_PATH in config.py)
    :param file_name: saved file name (without .pkl extension)
    :param verbose: verbosity (default value in config.py)
    :return: the SOM model and its label (tuple)
    """
    full_path = os.path.join(save_dir, f'{file_name}.pkl')
    with open(full_path, 'rb') as file_handler:
        if verbose:
            print(f'[+] Reading SOM object and it\'s labels from path : {full_path}')

        file_content = pickle.load(file_handler)
        return file_content['som'], file_content['labels']


def get_zone_boundaries(case='All'):
    """
    Get zone boundaries for either the full study zone or the restricted area
    :param case: 'All' for full zone, 'Sel' for the restricted one

    :return: starting latitude, ending latitude, starting longitude, ending longitude
    """
    if case.upper() == 'ALL':
        # A - Grande zone de l'upwelling (25x36) :
        #    Longitude : 45W Ã  10W (-44.5 Ã  -9.5)
        #    Latitude :  30N Ã  5N ( 29.5 Ã   4.5)
        frlat = 29.5
        tolat = 4.5
        frlon = -44.5
        tolon = -8.5
    elif case.upper() == 'SEL':
        # B - Sous-zone ciblant l'upwelling (13x12) :
        #    LON:  28W Ã  16W (-27.5 to -15.5)
        #    LAT : 23N Ã  10N ( 22.5 to  9.5)
        frlat = 22.5
        tolat = 9.5
        frlon = -27.5
        tolon = -15.5
    else:
        raise (
            Exception('"case" argument must me either "All" for the full study zone or "Sel" for the restricted one'))

    return frlat, tolat, frlon, tolon


def get_projection_errors(true_labels, pred_labels):
    """
    Compute a performance vector, one entry for each class, based on true labels

    :param: true_labels: true labels obtained by observation data
    :param: pred_labels: labels obtained by projecting model data on the trained SOM model (on observations)

    :return: an "nb_class"-length numpy.array containing performances for each class (ground is ignored)
    """

    true_labels = true_labels.flatten()
    pred_labels = pred_labels.flatten()

    nb_classes = max(true_labels)

    perf_vector = np.zeros(nb_classes, dtype=np.int)  # vecteur des performances !
    true_labels_count = np.zeros(nb_classes, dtype=np.int)

    for true_label, pred_label in zip(true_labels, pred_labels):
        true_labels_count[true_label - 1] += 1

        if true_label == 0 or pred_label == 0:
            continue
        if true_label == pred_label:
            perf_vector[true_label - 1] += 1

    return np.array([x/y for x, y in zip(perf_vector, true_labels_count)])
