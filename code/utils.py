import numpy as np
import sys, os
from time import time
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
import pandas as pd
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage

import UW3_triedctk as ctk
import netCDF4
# PARAMETRAGE DU CAS
from config import *

import triedpy.triedsompy as SOM

np.set_printoptions(threshold=sys.maxsize)

CMAP = ListedColormap(["brown", "blue", "lawngreen", "red", "purple", "pink", "yellow", "orange"])


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


def plot_levels_3D_SOM(labels, nb_classes=8, case='All', figure_title='3D SOM', save_file=True,
                       save_dir=OUTPUT_FIGURES_PATH,
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
    global CMAP

    fig, axs = plt.subplots(3, 4)

    if case.upper() == 'ALL':
        plt.setp(axs, xticks=[5, 20, 35], xticklabels=[-40, -25, -9], yticks=[0, 10, 20], yticklabels=[30, 20, 10])
    else:
        plt.setp(axs, xticks=[2, 6, 10], xticklabels=[-26, -22, -18], yticks=[0, 5, 10], yticklabels=[23, 18, 13])

    # cmap = plt.get_cmap('jet', nb_classes + 1)  # + 1 because of the earth

    for index, level in enumerate(labels):
        classes = labels[index]

        f = axs[index // 4, index % 4].imshow(classes, cmap=CMAP, interpolation='none', vmin=0, vmax=nb_classes)

        # plt.colorbar(f, ax=axs[index // 4, index % 4], ticks=np.arange(0, nb_classes + 1), shrink=0.7)
        axs[index // 4, index % 4].set_title(f'{index * 20 + 10}m')
    fig.delaxes(axs[2][3])
    plt.colorbar(f, ax=axs[2, 3], ticks=np.arange(0, nb_classes + 1), shrink=1)
    fig.suptitle(figure_title, y=1)
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
    global CMAP
    # cmap = plt.get_cmap('jet', max(labels.flatten()) + 1)

    plt.figure(figsize=(17, 8))

    for label in list(set(list(labels))):
        values = temperatures[labels == label]
        monthy_anom = values.mean(axis=0)
        monthy_dev = values.std(axis=0)
        # plt.plot(monthy_anom, label=f'classe = {label}', color=cmap(label))
        plt.errorbar(list(range(12)), monthy_anom, monthy_dev, label=f'class = {label}', color=CMAP(label))

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

    return np.array([x / y for x, y in zip(perf_vector, true_labels_count)])


def show_genetic_solution(model_values, solution_weights, case, nb_classes, som_model, true_labels, figure_prefix='GA_',
                          save_file=True,
                          save_dir=OUTPUT_FIGURES_PATH):
    temp = np.average(a=model_values, weights=solution_weights, axis=0)

    agg_data = aggregate_data(temp, case=case)

    all_model_data = agg_data.reshape(12, -1, order='A').T  # shape = 9900 (11*25*36), 12

    model_data = pd.DataFrame.from_records(all_model_data).dropna(axis=0)

    ocean_points_index = model_data.index.values  # ocean points index
    model_data = model_data.values  # train data

    model_labels = np.zeros(shape=(len(all_model_data)), dtype=int)

    hac = AgglomerativeClustering(n_clusters=nb_classes)
    hac = hac.fit(som_model.codebook)
    ocean_predicted_labels = get_reverse_classification(ctk.findbmus(sm=som_model, Data=model_data),
                                                        hac_labels=hac.labels_)
    model_labels[ocean_points_index] = ocean_predicted_labels.flatten() + 1  # à cause du 0 de la terre

    if case.upper() == 'ALL':
        model_labels_ = model_labels.reshape(11, 25, 36, order='A')
    elif case.upper() == 'SEL':
        model_labels_ = model_labels.reshape(11, 13, 12, order='A')

    perf_vector = get_projection_errors(true_labels=true_labels, pred_labels=model_labels_)
    print(f'{"*"*10} Genetic algorithm results {"*"*10}')
    print(f'\t\t[+] solution weights : {solution_weights}\n')
    print(f'\t\t[+] perf vector for each one of the {nb_classes}')

    plot_levels_3D_SOM(model_labels_, nb_classes=nb_classes,
                       figure_title=f'Genetic result {NB_CLASSES} classes geographical representation',
                       save_file=True, save_dir=config.OUTPUT_FIGURES_PATH, file_name='')

    plot_monthly_anomalies_3D_SOM(temperatures=model_data, labels=ocean_predicted_labels.flatten() + 1,
                                  figure_title=f'Genetic result (1979-2005). Monthly Mean by Class',
                                  save_file=True, save_dir=config.OUTPUT_FIGURES_PATH, file_name='')


def set_lonlat_ticks(lon, lat, fontsize=12, lostep=1, lastep=1, step=None, londecal=None, latdecal=None,
                     roundlabelok=True, lengthen=True, verbose=False):
    ''' Pour tracer les "ticks" et "ticklabels" des figures de type geographique,
        ou les axes ce sont les Latitudes et Longitudes
    '''
    if londecal is None:
        londecal = (lon[1] - lon[0]) / 2
        if lon[0] < lon[1]:
            londecal = -londecal
    if latdecal is None:
        latdecal = (lat[1] - lat[0]) / 2
        if lat[0] < lat[1]:
            latdecal = -latdecal
    if step is not None:
        # force la même valeur de pas dans les ticks x et y
        lostep = step
        lastep = step
    if verbose:
        print('londecal: {}\nlatdecal: {}'.format(londecal, latdecal))
    # ralonge les lon et les lat
    if verbose:
        print('LON-LAT:\n  {}\n  {}'.format(lon, lat))
    if lengthen:
        lon = np.concatenate((lon, [lon[-1] + (lon[1] - lon[0])]))
        lat = np.concatenate((lat, [lat[-1] + (lat[1] - lat[0])]))
        if verbose:
            print('LENGHTED LON-LAT:\n  {}\n  {}'.format(lon, lat))
    nLon = lon.shape[0]
    nLat = lat.shape[0]
    # current axis limits
    lax = plt.axis()
    # Ticks
    xticks = np.arange(londecal, nLon, lostep)
    yticks = np.arange(latdecal, nLat, lastep)
    # Ticklabels
    if 0:
        xticklabels = lon[np.arange(0, nLon, lostep)]
        yticklabels = lat[np.arange(0, nLat, lastep)]
    else:
        xticklabels = lon[np.arange(0, nLon, lostep)]
        yticklabels = lat[np.arange(0, nLat, lastep)]
        if lon[0] < lon[1]:
            xticklabels += londecal
        else:
            xticklabels -= londecal
        if lat[0] < lat[1]:
            yticklabels += latdecal
        else:
            yticklabels -= latdecal
    if verbose:
        print('Tiks:\n  {}\n  {}'.format(xticks, yticks))
        print('Labels:\n  {}\n  {}'.format(xticklabels, yticklabels))
    if roundlabelok:
        xticklabels = np.round(xticklabels).astype(int)
        yticklabels = np.round(yticklabels).astype(int)
        if verbose:
            print('Rounded Labels:\n  {}\n  {}'.format(xticklabels, yticklabels))
    #
    plt.xticks(xticks, xticklabels, fontsize=fontsize)
    plt.yticks(yticks, yticklabels, fontsize=fontsize)
    # set axis limits to previous value
    plt.axis(lax)


def do_plot_dendrogram(data, nclass=None, datalinkg=None, indnames=None, method='ward', metric='euclidean',
                       truncate_mode=None, title="dendrogram", titlefnsize=14, ytitle=0.98, xlabel=None, xlabelpad=10,
                       xlabelrotation=0, ylabel=None, ylabelpad=10, ylabelrotation=90, labelfnsize=10, labelrotation=0,
                       labelsize=10, labelha='center', labelva='top', dendro_linewidth=2, tickpad=2, axeshiftfactor=150,
                       figsize=(14, 6), wspace=0.0, hspace=0.2, top=0.92, bottom=0.12, left=0.05, right=0.99):
    """
    plot SOM dendrogram
    """

    if datalinkg is None:
        # Performs hierarchical/agglomerative clustering on the condensed distance matrix data
        datalinkg = linkage(data, method=method, metric=metric)
    #
    Ncell = data.shape[0]
    minref = np.min(data)
    maxref = np.max(data)
    #
    fig = plt.figure(figsize=figsize, facecolor='w')
    fignum = fig.number  # numero de figure en cours ...
    plt.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    #
    if nclass is None:
        # dendrogramme sans controle de color_threshold (on laisse par defaut ...)
        R_ = dendrogram(datalinkg, p=Ncell, truncate_mode=truncate_mode,
                        orientation='top', leaf_font_size=6, labels=indnames,
                        leaf_rotation=labelrotation)
    else:
        # calcule la limite de decoupage selon le nombre de classes ou clusters
        max_d = np.sum(datalinkg[[-nclass + 1, -nclass], 2]) / 2
        color_threshold = max_d

        with plt.rc_context({'lines.linewidth': dendro_linewidth}):  # Temporarily override the default line width
            R_ = dendrogram(datalinkg, p=Ncell, truncate_mode=truncate_mode,
                            color_threshold=color_threshold,
                            orientation='top', leaf_font_size=6, labels=indnames,
                            leaf_rotation=labelrotation)

        plt.axhline(y=max_d, c='k')

    plt.tick_params(axis='x', reset=True)
    plt.tick_params(axis='x', which='major', direction='inout', length=7, width=dendro_linewidth,
                    pad=tickpad, top=False, bottom=True,  # rotation_mode='anchor',
                    labelrotation=labelrotation, labelsize=labelsize)

    if indnames is None:
        L = np.narange(Ncell)
    else:
        L_ = np.array(indnames)
    plt.xticks((np.arange(Ncell) * 10) + 5, L_[R_['leaves']],
               horizontalalignment=labelha, verticalalignment=labelva)
    #
    plt.grid(axis='y')
    if xlabel is not None:
        plt.xlabel(xlabel, labelpad=xlabelpad, rotation=xlabelrotation, fontsize=labelfnsize)
    if ylabel is not None:
        plt.ylabel(ylabel, labelpad=ylabelpad, rotation=ylabelrotation, fontsize=labelfnsize)
    if axeshiftfactor is not None:
        lax = plt.axis()
        daxy = (lax[3] - lax[2]) / axeshiftfactor
        plt.axis([lax[0], lax[1], lax[2] - daxy, lax[3]])
    plt.title(title, fontsize=titlefnsize, y=ytitle)

    return R_


def do_plot_ct_dendrogram(sMapO, nb_class, datalinkg=None, method='ward', metric='euclidean', truncate_mode=None,
                          title="SOM MAP dendrogram", titlefnsize=14, ytitle=0.98, xlabel="neurons",
                          ylabel="inter class distance", labelfnsize=10, labelrotation=0, labelsize=10, figsize=(14, 6),
                          wspace=0.0, hspace=0.2, top=0.92, bottom=0.12, left=0.05, right=0.99):
    """
    Wrapper for plot SOM dendrogram
    """
    ncodbk = sMapO.codebook.shape[0]
    # print(sMapO.codebook)
    do_plot_dendrogram(sMapO.codebook, nclass=nb_class, datalinkg=datalinkg,
                       indnames=np.arange(ncodbk) + 1,
                       method=method, metric=metric,
                       truncate_mode=truncate_mode,
                       title=title, ytitle=ytitle, titlefnsize=titlefnsize,
                       xlabel=xlabel, ylabel=ylabel, labelfnsize=labelfnsize,
                       labelrotation=labelrotation, labelsize=labelsize,
                       figsize=figsize,
                       wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                       )
    plt.show()