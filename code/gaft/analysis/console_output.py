#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..plugin_interfaces.analysis import OnTheFlyAnalysis


class ConsoleOutput(OnTheFlyAnalysis):
    ''' Built-in on-the-fly analysis plugin class for outputing log on console.

    Attribute:

        interval(:obj:`int`): The analysis interval in evolution iteration, default 
                              value is 1 meaning analyze every step.

        master_only(:obj:`bool`): Flag for if the analysis plugin is only effective 
                                  in master process. Default is True.
    '''

    # Analysis interval.
    interval = 1

    # Only analyze in master process?
    master_only = True

    def setup(self, ng, engine):
        generation_info = 'Generation number: {}'.format(ng)
        population_info = 'Population number: {}'.format(engine.population.size)
        self.logger.info('{} {}'.format(generation_info, population_info))

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        ng_info = 'Generation: {}, '.format(g+1)
        fit_info = 'best fitness: {:.3f}, '.format(engine.ori_fmax)
        scaled_info = 'scaled fitness: {:.3f}'.format(engine.fmax)
        msg = ng_info + fit_info + scaled_info
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmax
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)

