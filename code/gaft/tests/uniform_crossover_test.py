#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Test case for built-in Uniform Crossover operator.
'''

import unittest

from gaft.components import BinaryIndividual
from gaft.operators.crossover.uniform_crossover import UniformCrossover

class UniformCrossoverTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff

    def test_cross(self):
        ''' Make sure individuals can be crossed correctly.
        '''
        father = BinaryIndividual(ranges=[(0, 1)]).init(solution=[0.398])
        mother = BinaryIndividual(ranges=[(0, 1)]).init(solution=[0.298])
        crossover = UniformCrossover(pc=1.0, pe=0.5)
        child1, child2 = crossover.cross(father, mother)

if '__main__' == __name__:
    suite = unittest.TestLoader().loadTestsFromTestCase(UniformCrossoverTest)
    unittest.TextTestRunner(verbosity=2).run(suite)

