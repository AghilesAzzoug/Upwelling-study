import numpy as np
import sys
import os
import random
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

sys.path.append("..")
import triedpy.triedsompy as SOM
import utils
import config
import gaft
import UW3_triedctk as ctk
from simanneal import Annealer

if config.DISABLE_WARNING:
    import warnings

    warnings.filterwarnings("ignore")

MODELS_DICT = dict()  # mapping model name (str) ==> model id (int)
MODELS_VALUES = list()  # future numpy.array of all models tensor values


def setupEnv():
    global MODELS_VALUES
    global MODELS_DICT
    """
    Setup the necessary variables to speed up the genetic algorithm execution
    Reading all models

    :return:
    """
    for index, model_name in enumerate(os.listdir(config.MODELS_DATA_PATH)):
        if config.VERBOSE:
            print(f'[!] reading model {model_name}')
        file_path = os.path.join(config.MODELS_DATA_PATH, model_name)
        data_label_base, temp, lon, lat, lev = utils.read_data(file_path)
        temp, lon, lat, ilat, ilon = utils.get_zone_obs(temp, lon, lat, size_reduction=CASE, frlon=frlon, tolon=tolon,
                                                        frlat=frlat, tolat=tolat)

        MODELS_DICT[data_label_base] = index
        MODELS_VALUES.append(temp)

    MODELS_VALUES = np.array(MODELS_VALUES)

    if config.VERBOSE:
        print('\n [!] Simulated Annealing environment correctly configured.')


class UpwellingModelsEvaluation(Annealer):

    def move(self):
        """swap a random number in a random location"""
        #initial_energy = self.energy()

        idx = random.randint(0, len(self.state) - 1)
        if self.state[idx] == 0:
            self.state[idx] = 1
        else:
            self.state[idx] = 0
        # b = random.randint(0, len(self.state) - 1)
        # self.state[a], self.state[b] = self.state[b], self.state[a]
        #return self.energy() - initial_energy

    def energy(self):
        """Calculate the quality of a solution"""
        global MODELS_VALUES
        global MODELS_DICT
        global CASE
        global true_labels

        solution_weights = self.state
        temp = np.average(a=MODELS_VALUES, weights=solution_weights, axis=0)

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
        # print("perf = " + str(float(np.mean(perf_vector))))
        print(f'Here .. {-float(np.mean(perf_vector))}')
        return -float(np.mean(perf_vector))


if __name__ == '__main__':
    """
        Cas binaire (combinaison de modèles sans poids)
    """

    # todo: get it from CMD
    CASE = 'All'
    NB_CLASSES = 7
    frlat, tolat, frlon, tolon = utils.get_zone_boundaries(case=CASE)

    # read the trained model file
    # output files (perfs) definition
    if CASE.upper() == 'ALL':
        obsSOM, true_labels = utils.loadSOM(save_dir=config.OUTPUT_TRAINED_MODELS_PATH,
                                            file_name=config.ZALL_SOM_3D_MODEL_NAME)
    elif CASE.upper() == 'SEL':
        obsSOM, true_labels = utils.loadSOM(save_dir=config.OUTPUT_TRAINED_MODELS_PATH,
                                            file_name=config.ZSEL_SOM_3D_MODEL_NAME)

    setupEnv()

    best_sol = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    #best_sol = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    ume = UpwellingModelsEvaluation(best_sol)

    # auto_schedule = ume.auto(minutes=1)
    # print(auto_schedule)
    # ume.set_schedule(auto_schedule)
    # ume.set_schedule(ume.auto(minutes=0.2))

    solutions, perfs = ume.anneal()

    print(solutions)
    print(perfs)