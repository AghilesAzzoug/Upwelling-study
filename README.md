##### Case study for the evaluation of climate multimodel ensembles in the Senegalese-Mauritanian **upwelling** region: ocean temperature study. The main purpose was to find a subset of climate models that best represents the “upwelling” phenomenon and use it for forecasting ocean temperature.

## Context:
the code was written in the case of a long project of TRIED master at Paris-Saclay University. 

@authors : Aghiles Azzoug & Constantin Bône

@supervisors : Carlos Mejia, Sylvie Thiria, and Michel Crepon


## Using the code:
For each of the following code file given, speficy in the begging of the main code, the desired parameters (number of classes, study zone, etc.)

To: 
* train SOM models: run ``code/3d/model_training.py``
* project a single model on the trained SOM: run ``code/3d/clim_models_projection.py``
* evaluate all models with the cumulative method: run ``code/3d/models_evaluation.py``
* show PCA results on the performances file: run ``code/analysis/pca_analysis.py``
* show MCA results on the performances file: run ``code/analysis/mca_analysis.py``
* execute the normal (decimal) genetic algorithm: run ``code/metaheuristics/genetic_search.py``
* execute the binary genetic algorithm: run ``code/metaheuristics/binary_genetic.py``
* execute the simulated annealing algorithm: run ``code/metaheuristics/simulated_annealing.py``

## Python requirements:

You should use Python 3.x rather than 2.x version. We are currently at
Python 3.7.3 from the Anaconda distribution in a Linux CentOS 7 platform
(with Intel Xeon processors). Tests has been realized on Windows 10 platforms as well. The current code has not been tested under previous versions.

You will need general modules installed in your Python distribution, as :

    numpy, matplotlib, getopt, copy, os, sys, time, timeit, datetime

but also more specific ones like:

    joblib, localdef, mpl_toolkits, shutil, 
    ipdb, numexpr, pandas, pickle, scipy, scikit-learn, netCDF4

## External projects used:
The code includes a modified version of the SOMPY code for Self Organizing Map (or
Kohonen Topological Map) library (https://github.com/sevamoo/SOMPY), named TRIEDSompy, and can be found in ``code/triedpy``

The code also use two metaheuritics projects, both located in ``code`` directory:
* GAFT (A Genetic Algorithm Framework in Python), which can be found here : https://github.com/PytLab/gaft
* Simanneal (Python module for Simulated Annealing optimization), which can be found here : https://github.com/perrygeo/simanneal
    

