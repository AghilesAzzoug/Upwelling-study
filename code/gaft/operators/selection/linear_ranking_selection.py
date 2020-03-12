#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Linear Ranking Selection implementation. '''

from random import random
from itertools import accumulate
from bisect import bisect_right

from ...plugin_interfaces.operators.selection import Selection


class LinearRankingSelection(Selection):
    ''' Selection operator using Linear Ranking selection method.

    Reference: Baker J E. Adaptive selection methods for genetic
    algorithms[C]//Proceedings of an International Conference on Genetic
    Algorithms and their applications. 1985: 101-111.
    '''
    def __init__(self, pmin=0.1, pmax=0.9):
        # Selection probabilities for the worst and best individuals.
        self.pmin, self.pmax = pmin, pmax

    def select(self, population, fitness):
        ''' Select a pair of parent individuals using linear ranking method.

        :param population: Population where the selection operation occurs.
        :type population: :obj:`gaft.components.Population`

        :return: Selected parents (a father and a mother)
        :rtype: list of :obj:`gaft.components.IndividualBase`
        '''
        # Individual number.
        NP = len(population)

        # Add rank to all individuals in population.
        all_fits = population.all_fits(fitness)
        indvs = population.individuals
        sorted_indvs = sorted(indvs,
                              key=lambda indv: all_fits[indvs.index(indv)])

        # Assign selection probabilities linearly.
        # NOTE: Here the rank i belongs to {1, ..., N}
        p = lambda i: (self.pmin + (self.pmax - self.pmin)*(i-1)/(NP-1))
        probabilities = [self.pmin] + [p(i) for i in range(2, NP)] + [self.pmax]

        # Normalize probabilities.
        psum = sum(probabilities)
        wheel = list(accumulate([p/psum for p in probabilities]))

        # Select parents.
        father_idx = bisect_right(wheel, random())
        father = sorted_indvs[father_idx]
        mother_idx = (father_idx + 1) % len(wheel)
        mother = sorted_indvs[mother_idx]

        return father, mother

