#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Definition of individual class with Probability vector encoding.
'''
import numpy as np
from .individual import IndividualBase


class ProbabilisticIndividual(IndividualBase):
    ''' Individual with probability encoding.

    :param ranges: value ranges for all entries in solution.
    :type ranges: tuple list

    :param eps: decrete precisions for binary encoding, default is 0.001.
    :type eps: float or float list (with the same length with ranges)
    '''
    def __init__(self, ranges, eps=0.001):
        super(self.__class__, self).__init__(ranges, eps)
        # Initialize it randomly.
        self.init()
        #self.chromsome = self._softmax(self.chromsome)
        self.solution = self._softmax(self.solution)

    def encode(self):
        ''' Encode solution to gene sequence
        '''
        return self.solution

    def decode(self):
        ''' Decode gene sequence to decimal solution
        '''
        return self.solution

    def _softmax(self, x):
        return np.exp(x) / sum(np.exp(x))

