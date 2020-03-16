import numpy as np
import os
import sys
from sklearn.cluster import AgglomerativeClustering
import scipy
import pandas as pd

sys.path.append("..")
import triedpy.triedsompy as SOM
import utils
import config
import gaft
import UW3_triedctk as ctk

from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis


# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore


def softmax(x):
    return np.exp(x)/sum(np.exp(x))




if __name__ == '__main__':

    # print(help(gaft.components.BinaryIndividual))

    indv_template = BinaryIndividual(ranges=[(0, 1) for _ in range(config.NB_MODELS)], eps=0.001)

    population = Population(indv_template=indv_template, size=50)
    population.init()
    selection = RouletteWheelSelection()
    crossover = UniformCrossover(pc=0.8, pe=0.5)
    mutation = FlipBitMutation(pm=0.1)


    engine = GAEngine(population=population, selection=selection, crossover=crossover, mutation=mutation,
                      analysis=[FitnessStore])

    @engine.fitness_register
    def fitness(indv):
        x = indv.solution
        return sum(x)


    @engine.analysis_register
    class ConsoleOutput(OnTheFlyAnalysis):
        master_only = True
        interval = 1

        def register_step(self, g, population, engine):
            best_indv = population.best_indv(engine.fitness)
            msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.fmax)
            engine.logger.info(msg)



    engine.run(ng=100)