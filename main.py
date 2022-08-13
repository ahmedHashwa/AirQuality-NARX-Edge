import os
from collections import namedtuple

import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

import hybrid_helper
import hybrid_preprocess

data_dir = "data"
saving_dir = 'results'


# %%  Processing Helpers


def chunk_filter(chunk):
    end_date_start = '01/01/'
    end_time_start = '01'
    return not (chunk['End Time'].iloc[0][:len(end_time_start)] == end_time_start and
                chunk['End Date'].iloc[0][:len(end_date_start)] == end_date_start and
                chunk['PM10'].iloc[0] is None)


def fix_data_china(dataframe: pd.DataFrame, path, name):
    # add index column
    dataframe['Datetime'] = \
        dataframe['day'].map(str) + '/' + dataframe['month'].map(str) + '/' + dataframe['year'].map(str) + ' ' + \
        dataframe['hour'].map(str) + ":00:00"
    dataframe['Index'] = pd.to_datetime(dataframe['Datetime'], format='%d/%m/%Y %H:%M:%S')

    dataframe.set_index('Index', inplace=True)
    dataframe.drop(columns=['year', 'day', 'month', 'hour', 'No'], inplace=True)
    dir_path = os.path.dirname(path)
    dataframe.corr().to_csv(os.path.join(dir_path, f'../{name}/data_corr_pre_process.csv'))
    return dataframe


# Press the green button in the gutter to run the script.
# %% Other Station
# Start preprocessing
from enum import Enum, auto


class ImputeMethod(Enum):
    ExtractedData = auto()
    RemovedInvalid = auto()
    ImputeMean = auto()
    ImputeTimeInterpolate = auto()
    ImputeLinearInterpolate = auto()
    ImputeLinearRegression = auto()
    ImputeRandomForest = auto()


skipped_column_names = ['Status/units', 'Unnamed', 'PM10', 'NOXasNO2', 'NV25', 'V25']
t_tuple = namedtuple('t_tuple', 'name func')


# %% Process data and prepare for ML

def set_datetime_index(df: DataFrame):
    dropped_column = 'Unnamed: 0'

    if dropped_column in df.columns:
        df.drop(columns=[dropped_column], inplace=True)
    if 'Datetime' in df.columns:
        df['Index'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M:%S')
        df.drop('Datetime', axis=1, inplace=True)
        df.set_index('Index', inplace=True)
        # noinspection PyUnresolvedReferences
        df.index = df.index.strftime('%d/%m/%Y %H:%M:%S')


# %% prepare data for usage in ML

def pre_process_data(df: DataFrame, ds_name):
    if 'China' not in ds_name:
        df = df.dropna()
        return df
    df = df[24:]
    df['pm2.5'].fillna(0, inplace=True)
    df = df.dropna()
    return df


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime

china_data_extracted = hybrid_preprocess.process_dir(f'{data_dir}/China/Data', chunk_size=10000, skip_rows=0,
                                                     skip_existing=True,
                                                     t_tuple=t_tuple(ImputeMethod.ExtractedData.name,
                                                                     lambda df, p, n: fix_data_china(df, p, n)))

datasets = [
    hybrid_helper.Dataset(name='China_Extracted', data=china_data_extracted,
                          feature_columns=['pm2.5', 'Iws', 'Ir'], target_columns=['pm2.5'],
                          include=True),

]
del china_data_extracted

for ds in datasets:
    if ds.include:
        set_datetime_index(ds.data)
    else:
        del ds.data

import hybrid_algorithms
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import fireTS.models

# import fbprophet

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import hybrid_metrics

evaluation_methods = {
    'MSE': mean_squared_error,
    'RMSE': hybrid_metrics.rmse,
    'NRMSE': hybrid_metrics.nrmse,
    'MAE': mean_absolute_error,
    'R2': r2_score,
    'IA': hybrid_metrics.index_agreement,
    'Pearson R': lambda observed, predicted: pearsonr(observed, predicted)[0]
}

session_start_datetime = datetime.now()

regressors = {}


def f_list(lst):
    return f'[{", ".join(f"{x:02d}" for x in lst)}]'


def key(run_index, **kwargs):
    from box import Box
    global algorithm_index
    global regressors
    regressor_name = kwargs["regressor_name"]

    if regressor_name not in regressors.keys():
        regressors[regressor_name] = algorithm_index
        algorithm_index = algorithm_index + 1
    regressor_name = f'{regressors[regressor_name]:02d}_{regressor_name}'

    if 'NARX' in regressor_name:
        kwargs["regressor_name"] = f'{regressor_name}_ao_{look_back:02d}_ed_{f_list(e_delay)}_eo_{f_list(e_order)}'
    else:
        kwargs['regressor_name'] = regressor_name

    kwargs['max_limit'] = max_limit
    kwargs['look_back'] = look_back if 'NARX' not in regressor_name and 'DAR' not in regressor_name else 0
    kwargs['n_estimators'] = n_estimators
    kwargs['session_start_datetime'] = session_start_datetime
    kwargs['dropout_rate'] = dropout_rate
    kwargs['n_lstm_nodes'] = n_lstm_nodes
    kwargs['activation'] = activation
    kwargs['n_dense_nodes'] = n_dense_nodes
    kwargs['batch_size'] = batch_size
    kwargs['evaluation_methods'] = evaluation_methods
    kwargs['index'] = run_index
    kwargs['dataset_name'] = ds.name
    kwargs['dataset'] = ds
    kwargs['results_dir'] = saving_dir

    kwargs['target_column'] = target_column
    kwargs['datetime'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    requested_key = Box(kwargs, frozen_box=True)
    print(
        f'Training and Predicting {requested_key.regressor_name} on '
        f'{requested_key.dataset_name} Data {ds.data.shape} '
        f'[trn {X_train.shape},{y_train.shape}] [tst {X_test.shape},{y_test.shape}] \n'
        f'targeting {target_column} iteration '
        f'NO: {requested_key.index} / {iterations_count} '
        f'on {requested_key.datetime}')
    return requested_key


algorithm_index = 1
batch_size = 72
epochs = 25
activation = 'relu'

enable_LSTM = True
enable_LSTM_dropout = False
enable_RF = False
enable_ET = True
enable_SVR = False
enable_GB = False
enable_XGB = False
enable_XGBRF = False
enable_XGBRF_DART = False

enable_NARX_LSTM = True
enable_NARX_LSTM_dropout = False
enable_NARX_RF = False
enable_NARX_ET = True
enable_NARX_SVR = False
enable_NARX_GB = False
enable_NARX_XGB = False
enable_NARX_XGBRF = False
enable_NARX_XGBRF_DART = False
enable_DAR_XGBRF = False

n_estimators = 100
regressor_parameters = {'n_estimators': n_estimators, 'n_jobs': -1, 'verbose': 3}
iterations_count = 10
n_lstm_nodes = 128
n_dense_nodes = 50
dropout_rate = 0.1
look_back = 24
n_subsequences = 4
scaler_method = hybrid_preprocess.ScaleMethod.NoScaler
lstm_scaler_method = hybrid_preprocess.ScaleMethod.StandardScaler
scale_target = False
lstm_scale_target = True
max_limit = 4279


# import tensorflow as tf

# Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU')


# %% Run Code
def fit_process(m, x, y):
    return m.fit(x, y, epochs=epochs, batch_size=batch_size)


def predict_process(m, x, _):
    return m.predict(x, batch_size=batch_size)


from sklearn.model_selection import KFold

kf = KFold(n_splits=iterations_count, random_state=None, shuffle=False)  #
i: int = 0
for ds in [d for d in datasets if d.include]:
    for target_column in ds.target_columns:
        # Prepare dataset to produce the data required for those algorithms (shifted data with certain column)
        features, prediction_target, num_features = \
            hybrid_preprocess.prepare_for_split(ds, look_back=look_back,
                                                target_column=target_column,
                                                pre_process_func=lambda x: pre_process_data(x, ds.name))
        # split the data using TimeSeriesSplit
        i = 0
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = prediction_target[train_index], prediction_target[test_index]
            i = i + 1
            # Normalize the data if required by the algorithms
            # Pass the data to the algorithm

            if enable_LSTM:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=hybrid_algorithms.LSTMModel(
                                                      n_lstm_nodes=n_lstm_nodes,
                                                      n_dense_nodes=n_dense_nodes,
                                                      dropout_rate=0,
                                                      activation=activation), num_features=num_features,
                                                  n_subsequences=n_subsequences, scale_target=lstm_scale_target,
                                                  fit_process=fit_process,
                                                  predict_process=predict_process,
                                                  scale_features_method=lstm_scaler_method,
                                                  reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape,
                                                  method_key=key(i, regressor_name='LSTM'))
            if enable_LSTM_dropout:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=hybrid_algorithms.LSTMModel(
                                                      n_lstm_nodes=n_lstm_nodes,
                                                      n_dense_nodes=n_dense_nodes,
                                                      dropout_rate=dropout_rate,
                                                      activation=activation), num_features=num_features,
                                                  n_subsequences=n_subsequences, scale_target=lstm_scale_target,
                                                  fit_process=fit_process,
                                                  predict_process=predict_process,
                                                  scale_features_method=lstm_scaler_method,
                                                  reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape,
                                                  method_key=key(i, regressor_name='LSTM_dropout'))
            if enable_RF:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=RandomForestRegressor(
                                                      **regressor_parameters), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='RandomForest'))
            if enable_ET:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=ExtraTreesRegressor(
                                                      **regressor_parameters), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='ExtraTrees'))
            if enable_XGB:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=XGBRegressor(
                                                      n_estimators=n_estimators, verbosity=3),
                                                  scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='XGB'))
            if enable_XGBRF:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=XGBRFRegressor(
                                                      n_estimators=n_estimators, verbosity=3),
                                                  scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='XGBRF'))
            if enable_XGBRF_DART:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=XGBRFRegressor(
                                                      n_estimators=n_estimators, verbosity=3,
                                                      booster='dart'), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='XGBRF_DART'))
            if enable_GB:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=GradientBoostingRegressor(
                                                      n_estimators=n_estimators,
                                                      verbose=3), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='GradientBoosting'))
            if enable_SVR:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=SVR(C=2.0, epsilon=0.1,
                                                                       kernel='rbf', gamma=0.5,
                                                                       tol=0.001, verbose=True,
                                                                       shrinking=True,
                                                                       max_iter=10000), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='SVR'))


# noinspection PyPep8Naming
def NARX_predict_process(reg, x, y):
    return reg.predict(x, y.reshape(-1), step=1)


# 1, 4, 6, 8, 12, 24, 48 / 1,6
exog_order_values = [1]
exog_delay_values = [1, 8, look_back]


def post_prediction_processing(results, exog_order, exog_delay):
    return results[max(max(exog_order) + max(exog_delay), look_back):]


# noinspection PyPep8Naming
def NARX_predictor(regressor, exog_order, exog_delay):
    res = fireTS.models.NARX(regressor,
                             auto_order=look_back,
                             exog_order=exog_order,
                             exog_delay=exog_delay)
    return res


# NARX Algorithms
# has to be run alone as graphics dependent algorithms has to be run consecutively
import itertools

for ds in [d for d in datasets if d.include]:
    for target_column in ds.target_columns:
        # Prepare dataset to produce the data required for those algorithms (shifted data with certain column)
        features, prediction_target, num_features = \
            hybrid_preprocess.prepare_for_split(ds, look_back=0,
                                                target_column=target_column,
                                                pre_process_func=lambda x: pre_process_data(x, ds.name))
        # split the data using TimeSeriesSplit
        i = 0
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = prediction_target[train_index], prediction_target[test_index]
            i = i + 1
            for e_order in [list(x) for x in list(itertools.product(exog_order_values, repeat=num_features))]:
                for e_delay in [list(x) for x in
                                list(itertools.product(exog_delay_values, repeat=num_features))]:
                    # Normalize the data if required by the algorithms
                    # Pass the data to the algorithm
                    if enable_NARX_LSTM:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(
                                                              hybrid_algorithms.LSTMModel(n_lstm_nodes=n_lstm_nodes,
                                                                                          n_dense_nodes=n_dense_nodes,
                                                                                          dropout_rate=0,
                                                                                          activation=activation),

                                                              exog_order=e_order,
                                                              exog_delay=e_delay),
                                                          scale_target=lstm_scale_target,
                                                          fit_process=fit_process, predict_process=NARX_predict_process,
                                                          scale_features_method=lstm_scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_LSTM'))
                    if enable_NARX_LSTM_dropout:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(
                                                              hybrid_algorithms.LSTMModel(n_lstm_nodes=n_lstm_nodes,
                                                                                          n_dense_nodes=n_dense_nodes,
                                                                                          dropout_rate=dropout_rate,
                                                                                          activation=activation),

                                                              exog_order=e_order,
                                                              exog_delay=e_delay),
                                                          scale_target=lstm_scale_target,
                                                          fit_process=fit_process, predict_process=NARX_predict_process,
                                                          scale_features_method=lstm_scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_LSTM_dropout'))
                    if enable_NARX_RF:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(
                                                              RandomForestRegressor(**regressor_parameters),
                                                              exog_order=e_order,
                                                              exog_delay=e_delay), scale_target=scale_target,
                                                          predict_process=NARX_predict_process,
                                                          scale_features_method=scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_RandomForest'))
                    if enable_NARX_ET:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(
                                                              ExtraTreesRegressor(**regressor_parameters),
                                                              exog_order=e_order,
                                                              exog_delay=e_delay), scale_target=scale_target,
                                                          predict_process=NARX_predict_process,
                                                          scale_features_method=scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_ExtraTrees'))
                    if enable_NARX_XGB:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(
                                                              XGBRegressor(n_estimators=100, n_jobs=-1,
                                                                           verbosity=3),

                                                              exog_order=e_order,
                                                              exog_delay=e_delay),
                                                          scale_target=scale_target,
                                                          predict_process=NARX_predict_process,
                                                          scale_features_method=scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_XGB'))
                    if enable_NARX_XGBRF:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(
                                                              XGBRFRegressor(n_estimators=100, n_jobs=-1,
                                                                             verbosity=3),

                                                              exog_order=e_order,
                                                              exog_delay=e_delay),
                                                          scale_target=scale_target,
                                                          predict_process=NARX_predict_process,
                                                          scale_features_method=scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_XGBRF'))
                    if enable_NARX_GB:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(GradientBoostingRegressor(
                                                              n_estimators=n_estimators, verbose=3),
                                                              exog_order=e_order, exog_delay=e_delay),
                                                          scale_target=scale_target,
                                                          predict_process=NARX_predict_process,
                                                          scale_features_method=scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_GradientBoosting'))
                    if enable_NARX_SVR:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(
                                                              SVR(C=2.0, epsilon=0.1, kernel='rbf',
                                                                  gamma=0.5,
                                                                  tol=0.001, verbose=True,
                                                                  shrinking=True,
                                                                  max_iter=10000),

                                                              exog_order=e_order,
                                                              exog_delay=e_delay),
                                                          scale_target=scale_target,
                                                          predict_process=NARX_predict_process,
                                                          scale_features_method=scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_SVR'))
                    if enable_NARX_XGBRF_DART:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=NARX_predictor(
                                                              XGBRFRegressor(n_estimators=n_estimators,
                                                                             verbosity=3,
                                                                             booster='dart'),

                                                              exog_order=e_order,
                                                              exog_delay=e_delay),
                                                          scale_target=scale_target,
                                                          predict_process=NARX_predict_process,
                                                          scale_features_method=scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='NARX_XGBRF_DART'))
                    if enable_DAR_XGBRF:
                        hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                          y_test=y_test,
                                                          predictor_object=fireTS.models.DirectAutoRegressor(
                                                              XGBRFRegressor(n_estimators=n_estimators,
                                                                             n_jobs=-1, verbosity=3),
                                                              auto_order=look_back,
                                                              exog_order=e_order,
                                                              exog_delay=e_delay,
                                                              pred_step=1), scale_target=scale_target,
                                                          predict_process=lambda reg, x, y: reg.predict(x,
                                                                                                        y.reshape(
                                                                                                            -1)),
                                                          scale_features_method=scaler_method,
                                                          post_prediction_callback=post_prediction_processing,
                                                          method_key=key(i, regressor_name='DAR_XGBRF'))

# %% Joining results
from os import listdir
from os.path import join
import pandas as pd

formatted_datetime = session_start_datetime.strftime("%d-%m-%Y_%H_%M_%S")

pd.set_option('display.max_columns', 500)
data_dir = f'{saving_dir}/{formatted_datetime}'
for ds in [d for d in datasets if d.include]:
    for i in range(1, iterations_count + 1):
        first_file = None
        for file in listdir(data_dir):
            file = join(data_dir, file)
            if f'{i:02d}_predictions' in file and f'{ds.name}' in file:
                # noinspection DuplicatedCode
                cols = list(pd.read_csv(file, nrows=1))
                if first_file is None:
                    first_file = pd.read_csv(file, index_col=0,
                                             usecols=[c for c in cols if 'aqi' not in c])
                    if 'NARX' not in file:
                        first_file = first_file.iloc[8:]
                        first_file.reset_index(inplace=True)
                        first_file.drop(columns=['index'], inplace=True)

                else:
                    read_file = pd.read_csv(file, index_col=0,
                                            usecols=[c for c in cols if 'aqi' not in c and 'Real' not in c])
                    if 'NARX' not in file:
                        read_file = read_file.iloc[8:]
                        read_file.reset_index(inplace=True)
                        read_file.drop(columns=['index'], inplace=True)
                    first_file = first_file.join(read_file)
        first_file.to_csv(join(data_dir, f'run_{ds.name}_i{i:02d}_data_{formatted_datetime}.csv'))

summary_columns = ["Regressor Name", "Training Period Seconds", "Prediction Period Seconds", "MSE", "RMSE", "NRMSE",
                   "MAE", "R2", "IA", "Pearson R"]

for ds in [d for d in datasets if d.include]:
    all_metrics_results = None
    for i in range(1, iterations_count + 1):
        for file in listdir(data_dir):
            file = join(data_dir, file)
            if f'{i:02d}_metrics' in file and f'{ds.name}' in file:
                cols = list(pd.read_csv(file, nrows=1))
                if all_metrics_results is None:
                    all_metrics_results = pd.read_csv(file, index_col=0)
                else:
                    metrics = pd.read_csv(file, index_col=0)
                    all_metrics_results = pd.concat([all_metrics_results, metrics])
    all_metrics_results.to_csv(join(data_dir, f'all_metrics_{ds.name}_data_{formatted_datetime}.csv'))
    grouped_results_mean = all_metrics_results[summary_columns].groupby('Regressor Name').mean()
    grouped_results_mean.to_csv(join(data_dir, f'all_metrics_mean_{ds.name}_data_{formatted_datetime}.csv'))

    grouped_results_max = all_metrics_results[summary_columns].groupby('Regressor Name').max()
    grouped_results_max.to_csv(join(data_dir, f'all_metrics_max_{ds.name}_data_{formatted_datetime}.csv'))
    grouped_results_min = all_metrics_results[summary_columns].groupby('Regressor Name').min()
    grouped_results_min.to_csv(join(data_dir, f'all_metrics_min_{ds.name}_data_{formatted_datetime}.csv'))
    print("Printing Mean Results: ")
    print(grouped_results_mean)
    # print("Printing Max Results: ")
    # print(grouped_results_max)
    # print("Printing Min Results: ")
    # print(grouped_results_min)
