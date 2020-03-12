#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Module for Genetic Algorithm crossover operator class '''

from ..metaclasses import CrossoverMeta


class Crossover(metaclass=CrossoverMeta):
    ''' Class for providing an interface to easily extend the behavior of crossover
    operation between two individuals for children breeding.

    Attributes:

        pc(:obj:`float`): The probability of crossover (usaully between 0.25 ~ 1.0)
    '''

    # The probability of crossover (usaully between 0.25 ~ 1.0)
    pc = 0.8

    def cross(self, father, mother):
        ''' Called when we need to cross parents to generate children.

        :param father: The parent individual to be crossed
        :type father: gaft.components.IndividualBase

        :param mother: The parent individual to be crossed
        :type mother: gaft.components.IndividualBase

        :return children: Two new children individuals
        :type children: tuple of gaft.components.IndividualBase
        '''
        raise NotImplementedError

