# -*- coding: ISO-8859-1 -*-

OBS_DATA_PATH = '../../data/Obs'  # Observations folder path
MODELS_DATA_PATH = '../../data/Model'  # Models folder path
VERBOSE = True  # Default verbosity
DISABLE_WARNING = True  # disable warning due to np.nan(s) in data
SEED = 0  # Default seed
NB_MODELS = 27  # Number of models

OUTPUT_FIGURES_PATH = '../../output/figures'  # output figures folder path
OUTPUT_TRAINED_MODELS_PATH = '../../output/trained_models'  # output trained models (SOMs)
OUTPUT_PERF_PATH = '../../output/perfs'
USED_MODEL_FILE_NAME = 'to_ORAS4regtoORCA1_1979-2005_selectbox.nc'  # name of the model used for training data

ZALL_SOM_3D_MODEL_NAME = 'Map_Zall_ORAS4regtoORCA1-1979-2005'  # SOM model file name for the full study zone
ZSEL_SOM_3D_MODEL_NAME = 'Map_Zsel_ORAS4regtoORCA1-1979-2005'  # SOM model file name for the restricted study zone

ZALL_MODELS_PERF_FILE_NAME = 'Perf_df_ZALL.csv'  # performance file for each model (full zone)
ZSEL_MODELS_PERF_FILE_NAME = 'Perf_df_ZSEL.csv'  # performance file for each model (restricted zone)

ZALL_MODELS_CUMUL_PERF_FILE_NAME = 'Cumul_perf_df_ZALL.csv'  # cumul perfs. file (full zone)
ZSEL_MODELS_CUMUL_PERF_FILE_NAME = 'Cumul_perf_df_ZSEL.csv'  # cumul perfs. file (restricted zone)

ZALL_MODELS_FINAL_PERF_FILE_NAME_CSV = 'Final_perf_df_ZALL.csv'  # final perfs. file (full zone)
ZSEL_MODELS_FINAL_PERF_FILE_NAME_CSV = 'Final_perf_df_ZSEL.csv'  # final perfs. file (full zone)

ZALL_MODELS_FINAL_PERF_FILE_NAME_HTML = 'Final_perf_df_ZALL.html'  # final perfs. file (full zone) HTML format
ZSEL_MODELS_FINAL_PERF_FILE_NAME_HTML = 'Final_perf_df_ZSEL.html'  # final perfs. file (full zone) HTML format
