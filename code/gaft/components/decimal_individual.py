#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Definition of individual class with decimal encoding.
'''

from .individual import IndividualBase
import numpy as np

class DecimalIndividual(IndividualBase):
    ''' Individual with decimal encoding.

    :param ranges: value ranges for all entries in solution.
    :type ranges: tuple list

    :param eps: decrete precisions for binary encoding, default is 0.001.
    :type eps: float or float list (with the same length with ranges)
    '''
    def __init__(self, ranges, eps=0.001):
        super(self.__class__, self).__init__(ranges, eps)
        # Initialize it randomly.
        self.init()
        # added by Aghiles
        solution_ = np.array(self.solution)
        solution_[solution_ < 0.5] = 0
        solution_[solution_ > 0.5] = 1
        self.solution = list(solution_)

    def encode(self):
        ''' Encode solution to gene sequence
        '''
        return self.solution

    def decode(self):
        ''' Decode gene sequence to decimal solution
        '''
        return self.solution

