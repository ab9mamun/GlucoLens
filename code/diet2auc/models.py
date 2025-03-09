import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from . import datadriver as dd
import pickle
from xgboost import XGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor



def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    nrmse = rmse / y_true.mean()
    return mse, mae, r2, rmse, nrmse


def get_trained_model(model_type, X_train_scaled, y_train, seed=42):
    # Model training
    if model_type == 'ridge':
        model_name = 'Ridge'
        hyper_txt = 'alpha=1.0'
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

    elif model_type == 'randomforest':
        model_name = 'RandomForest'
        hyper_txt = 'n_estimators=10'
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X_train_scaled, y_train)

    elif model_type == 'mlpregressor':
        model_name = 'MLPRegressor'
        hyper_txt = 'hidden_layer_sizes=80_40_20_20_20_20_10_5'
        model = MLPRegressor(hidden_layer_sizes=(80, 40, 20, 20, 20, 20, 10, 5), random_state=seed)

        try:
            model.fit(X_train_scaled, y_train)
        except ConvergenceWarning:
            logging.warning(
                'ConvergenceWarning: The model did not converge. Try increasing the number of iterations or the number of hidden layers.')
    else:
        print('Invalid model. Exiting.')
        return None, None, None

    return model_name, hyper_txt, model

def get_train_and_test_data(basepath, outputpath, seed=42):
    # Load data
    data = dd.create_or_load_dummies(basepath, outputpath)
    X = dd.drop_noninput_columns(data)
    y = dd.get_label_column(data)

    X, y = shuffle(X, y, random_state=seed)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, train_size, test_size

def train_and_evaluate(basepath, outputpath, model_type, seed=42):
    # Load data
    X_train_scaled, X_test_scaled, y_train, y_test, train_size, test_size = get_train_and_test_data(basepath, outputpath, seed)
    # Model training
    model_name, hyper_txt, model = get_trained_model(model_type, X_train_scaled, y_train, seed)
    if model_name is None:
        return

    # Metrics on the train set
    y_pred_train = model.predict(X_train_scaled)
    mse_train, mae_train, r2_train, rmse_train, nrmse_train = get_metrics(y_train, y_pred_train)

    # Model evaluation on the test set
    y_pred = model.predict(X_test_scaled)
    mse, mae, r2, rmse, nrmse = get_metrics(y_test, y_pred)

    # Save results
    header_txt = 'model,hyperparams,seed,train_size,test_size,train_mse,traim_mae,train_r2,train_rmse,train_nrmse,test_mse,test_mae,test_r2,test_rmse,test_nrmse\n'
    result_txt = f'{model_name},{hyper_txt},{seed},{train_size},{test_size},{mse_train},{mae_train},{r2_train},{rmse_train},{nrmse_train},{mse},{mae},{r2},{rmse},{nrmse}\n'

    save_results(outputpath, header_txt, result_txt, 'results.csv')

def save_results(outputpath, header_txt, result_txt, filename='results.csv'):
    if not os.path.exists(f'{outputpath}/{filename}'):
        with open(f'{outputpath}/{filename}', 'w') as file:
            file.write(header_txt+result_txt)
    else:
        with open(f'{outputpath}/{filename}', 'a') as file:
            file.write(result_txt)



def diet2auc_ultra(basepath, outputpath):
    df = dd.load_dataset_with_respective_auc(basepath)
    df = df.drop(columns=['user_id', 'phase', 'day', 'auc', 'baseline_activity'])


    # with self-reported activity and macronutrients
    print('Now working with self-reported activity and macronutrients')
    df1 = df.drop(columns=['sitting_total', 'standing_total', 'stepping_total',
                           'sitting_at_work', 'standing_at_work', 'stepping_at_work',
                           'glycemic_load'])

    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    execute_three_tasks(df1, outputpath, 'self_reported_activity_macronutrients')


    # with activPAL data and macronutrients
    print('Now working with activPAL features and macronutrients')
    df1 = df.drop(columns=['recent_activity', 'glycemic_load'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose

    execute_three_tasks(df1, outputpath, 'activpal_macronutrients')

    # with self-reported activity and glycemic load
    print('Now working with self-reported activity and glycemic load')
    df1 = df.drop(columns=['Net Carbs(g)','Protein (g)','Fiber (g)','Total Fat (g)', 'sitting_total', 'standing_total', 'stepping_total', 'sitting_at_work', 'standing_at_work', 'stepping_at_work'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    execute_three_tasks(df1, outputpath, 'self_reported_activity_glycemic_load')

    # with activPAL data and glycemic load
    print('Now working with activPAL features and glycemic load')
    df1 = df.drop(columns=['recent_activity', 'Net Carbs(g)','Protein (g)','Fiber (g)','Total Fat (g)'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    execute_three_tasks(df1, outputpath, 'activpal_glycemic_load')

    # with all features
    print('Now working with all valid features')
    df1 = df
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    execute_three_tasks(df1, outputpath, 'all_features')

def execute_three_tasks(df, outputpath, foldername):

    if not os.path.exists(f'{outputpath}/{foldername}'):
        os.makedirs(f'{outputpath}/{foldername}')
    df.to_csv(f'{outputpath}/{foldername}/dataset_{foldername}.csv', index=False)

    # # 1. respective_auc
    df1 = df.drop(columns=['absolute_auc', 'max_postprandial_gluc', 'postprandial_hyperglycemia_140'])
    df1 = df1.dropna()
    X = df1.drop(columns=['respective_auc']).values.astype(float)
    y = df1['respective_auc'].to_numpy().astype(float)
    run_regression(X,y, outputpath, foldername, 'respective_auc', df1.columns)

    # 2. absolute_auc
    df1 = df.drop(columns=['respective_auc', 'max_postprandial_gluc', 'postprandial_hyperglycemia_140', 'norm_auc'])
    df1 = df1.dropna()
    X = df1.drop(columns=['absolute_auc']).values.astype(float)
    y = df1['absolute_auc'].to_numpy().astype(float)
    run_regression(X, y, outputpath, foldername, 'absolute_auc', df1.columns)

    # # 3. max_glucose
    df1 = df.drop(columns=['respective_auc', 'absolute_auc', 'postprandial_hyperglycemia_140'])
    df1 = df1.dropna()
    X = df1.drop(columns=['max_postprandial_gluc']).values.astype(float)
    y = df1['max_postprandial_gluc'].to_numpy().astype(float)
    
    run_regression(X, y, outputpath, foldername, 'max_postprandial_gluc', df1.columns)
 

def run_regression(X, y, outputpath, foldername, task, colnames):
    all_hyper_params = {
        'ridge': [{'alpha': 1.0}, {'alpha': 0.1}, {'alpha': 0.01}],
        'lasso': [{'alpha': 1.0}, {'alpha': 0.1}, {'alpha': 0.01}],
        'randomforest': [{'n_estimators': 10, 'max_nodes':24},
                         {'n_estimators': 10, 'max_nodes': 48},
                         {'n_estimators': 10, 'max_nodes': 96},
                         {'n_estimators': 50, 'max_nodes': 24},
                         {'n_estimators': 50, 'max_nodes': 48},
                         {'n_estimators': 50, 'max_nodes': 96},
                         {'n_estimators': 100, 'max_nodes': 24},
                         {'n_estimators': 100, 'max_nodes': 48},
                         {'n_estimators': 100, 'max_nodes': 96}],
        'mlpregressor': [{'hidden_layer_sizes': (20, 10, 5)},
                        {'hidden_layer_sizes': (40, 20, 10, 5)},
                        {'hidden_layer_sizes': (60, 30, 15, 7)},
                        {'hidden_layer_sizes': (80, 40, 20, 10, 5)},
                        {'hidden_layer_sizes': (100, 50, 25, 12, 6)},
                        {'hidden_layer_sizes': (120, 60, 30, 15, 7)},
                        {'hidden_layer_sizes': (140, 70, 35, 17, 8)},
                        {'hidden_layer_sizes': (160, 80, 40, 20, 10)},
                        {'hidden_layer_sizes': (80, 40, 20, 20, 20, 20, 10, 5)},
                        {'hidden_layer_sizes': (100, 50, 25, 25, 25, 25, 12, 6)},
                        {'hidden_layer_sizes': (120, 60, 30, 30, 30, 30, 15, 7)},
                        {'hidden_layer_sizes': (140, 70, 35, 35, 35, 35, 17, 8)},
                        {'hidden_layer_sizes': (160, 80, 40, 40, 40, 40, 20, 10)}],
        'xgboost':[{'settings': 'default'}],#, {'n_estimators': 50}, {'n_estimators': 100}],
        'tabnet':[{'settings': 'default'}]
    }

    for model_type in ['tabnet', 'xgboost', 'ridge', 'randomforest', 'mlpregressor']:
        print('Now running experiments with model: ', model_type)
        hyper_params_array = all_hyper_params[model_type]
        for hyper_params in hyper_params_array:
            for seed in [0, 10, 42]:
                train_and_evaluate_v2(outputpath, foldername, task, colnames,  hyper_params, X, y, model_type, seed=seed)


def train_and_evaluate_v2(outputpath, foldername, task, colnames, hyper_params, X, y, model_type, seed=42):
    # Load data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, train_size, test_size, scaler = get_train_and_test_data_v2(X, y, seed)
    # Model training
    if model_type == 'tabnet':
        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
    model_name, hyper_txt, model = get_trained_model_v2(model_type, hyper_params, X_train_scaled, y_train, seed)
    if model_name is None:
        return

    # Metrics on the train set
    y_pred_train = model.predict(X_train_scaled)
    if model_type == 'tabnet':
        y_pred_train = np.array(y_pred_train).reshape(-1,)
    mse_train, mae_train, r2_train, rmse_train, nrmse_train = get_metrics(y_train, y_pred_train)

    # Model evaluation on the test set
    y_pred = model.predict(X_test_scaled)
    if model_type == 'tabnet':
        y_pred = np.array(y_pred).reshape(-1,)
        y_test = np.array(y_test).reshape(-1,)

    mse, mae, r2, rmse, nrmse = get_metrics(y_test, y_pred)

    # Save results
    header_txt = 'model,hyperparams,seed,train_size,test_size,train_mse,traim_mae,train_r2,train_rmse,train_nrmse,test_mse,test_mae,test_r2,test_rmse,test_nrmse\n'
    result_txt = f'{model_name},{hyper_txt},{seed},{train_size},{test_size},{mse_train},{mae_train},{r2_train},{rmse_train},{nrmse_train},{mse},{mae},{r2},{rmse},{nrmse}\n'

    full_output_path = os.path.join(outputpath, foldername, model_name, task)
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)

    save_results(full_output_path, header_txt, result_txt, 'results.csv')

    # Save the train and test results
    train_results = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1), columns=colnames)
    train_results['predicted'] = y_pred_train

    test_results = pd.DataFrame(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1), columns=colnames)
    test_results['predicted'] = y_pred

    train_results.to_csv(f'{full_output_path}/train_results_{model_name}_{hyper_txt}_{seed}.csv', index=False)
    test_results.to_csv(f'{full_output_path}/test_results_{model_name}_{hyper_txt}_{seed}.csv', index=False)

    scaler_path = f'{full_output_path}/scaler_{seed}.pkl'
    if not os.path.exists(scaler_path):
        with open(f'{full_output_path}/scaler_{seed}.pkl', 'wb') as file:
            pickle.dump(scaler, file)


def get_train_and_test_data_v2(X, y, seed=42):
    X, y = shuffle(X, y, random_state=seed)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, train_size, test_size, scaler


def get_trained_model_v2(model_type, hyper_params, X_train_scaled, y_train, seed=42):
    # Model training
    if model_type == 'ridge':
        model_name = 'Ridge'
        alpha = hyper_params['alpha']
        hyper_txt = f'alpha_{alpha}'
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

    elif model_type == 'lasso':
        # Lasso is creating problem. Need to fix it.
        model_name = 'Lasso'
        alpha = hyper_params['alpha']
        hyper_txt = f'alpha_{alpha}'
        model = Lasso(alpha=alpha)
        try:
            model.fit(X_train_scaled, y_train)
        except ConvergenceWarning:
            logging.warning('ConvergenceWarning: The model did not converge. Try increasing the number of iterations or the alpha value.')

    elif model_type == 'randomforest':
        model_name = 'RandomForest'
        n_estimators = hyper_params['n_estimators']
        max_nodes = hyper_params['max_nodes']
        hyper_txt = f'n_estimators_{n_estimators}_max_nodes_{max_nodes}'
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(X_train_scaled, y_train)

    elif model_type == 'mlpregressor':
        model_name = 'MLPRegressor'
        hidden_layer_sizes = hyper_params['hidden_layer_sizes']
        hyper_txt = 'hidden_layer_sizes_' + '_'.join([str(x) for x in hidden_layer_sizes])
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=seed)

        try:
            model.fit(X_train_scaled, y_train)
        except ConvergenceWarning:
            logging.warning('ConvergenceWarning: The model did not converge. Try increasing the number of iterations or the number of hidden layers.')

    elif model_type == 'xgboost':
        model_name = 'XGBoost'
        hyper_txt = 'settings_default'
        model = XGBRegressor()
        model.fit(X_train_scaled, y_train)

    elif model_type == 'tabnet':
        model_name = 'TabNet'
        hyper_txt = 'settings_default'
        model = TabNetRegressor()
        model.fit(X_train_scaled, y_train,
                  max_epochs=100,
                  patience=10,
                  batch_size=12,
                    virtual_batch_size=4)

    else:
        print('Invalid model. Exiting.')
        return None, None, None



    return model_name, hyper_txt, model

def simulate(basepath, output_folder, model_type, test_param):
    if model_type == 'RandomForest':
        pass
    else:
        print('Other models are not supported now. Exiting.')
        return

    # Load model if available
    model_path = f'{output_folder}/simulate/model.pkl'
    scaler_path = f'{output_folder}/simulate/scaler.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            best_model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            best_scaler = pickle.load(file)

    else:
        # Load data
        if not os.path.exists(f'{output_folder}/simulate/'):
            os.makedirs(f'{output_folder}/simulate/')


        df = dd.load_dataset_with_respective_auc(basepath)
        df = df.drop(columns=['user_id', 'phase', 'day', 'auc', 'baseline_activity'])
        print('Now working with activPAL features and macronutrients')
        df1 = df.drop(columns=['recent_activity', 'glycemic_load'])

        df1 = df1.drop(columns=['respective_auc', 'max_postprandial_gluc', 'postprandial_hyperglycemia_140'])
        df1 = df1.dropna()
        X = df1.drop(columns=['absolute_auc']).values.astype(float)
        y = df1['absolute_auc'].to_numpy().astype(float)
        best_nrmse = np.inf
        for seed in [0, 10, 42]:
            nrmse, model, scaler = train_and_evaluate_v3( X, y, 'randomforest' , seed=seed)
            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_model = model
                best_scaler = scaler
        # Save model
        with open(model_path, 'wb') as file:
            pickle.dump(best_model, file)
        with open(scaler_path, 'wb') as file:
            pickle.dump(best_scaler, file)
        with open(f'{output_folder}/simulate/nrmse.txt', 'w') as file:
            file.write(f'Best NRMSE: {nrmse}\nModel: {model_type}\nSeed: {seed}\nHyper_parsm: n_estimators=50\n')

    # Load the test data
    # sample test_param: "Fasting_Glucose:80.5;Recent_CGM:89.99;Lunch_Time:12.25;BMI:32.23;Calories:648.66;Calories_from_Fat:221.84;Total_Fat_(g):24.94;Saturated_Fat_(g):7.34;Trans_Fat_(g):0.13;Cholesterol_(mg):66.3;Sodium_(mg):1072.0;Total_Carbs_(g):79.0;Fiber_(g):5.0;Sugars_(g):12.0;Net_Carbs_(g):74.0;Protein_(g):28.0;Today's_sitting_duration_(s):10000.0;Today's_standing_duration_(s):6300.0;Today's_stepping_duration_(s):1680.0;Sitting_duration_at_work_(s):8255.0;Standing_duration_at_work_(s):5000.0;Stepping_duration_at_work_(s):1130.0;Work_start_time:8.25;Work_from_home:false;Day_of_week:Monday;"
    fasting_gluc = float(test_param.split(';')[0].split(':')[1])
    recent_cgm = float(test_param.split(';')[1].split(':')[1])
    lunch_time = float(test_param.split(';')[2].split(':')[1])
    bmi = float(test_param.split(';')[3].split(':')[1])
    calories = float(test_param.split(';')[4].split(':')[1])
    calories_from_fat = float(test_param.split(';')[5].split(':')[1])
    total_fat = float(test_param.split(';')[6].split(':')[1])
    saturated_fat = float(test_param.split(';')[7].split(':')[1])
    trans_fat = float(test_param.split(';')[8].split(':')[1])
    cholesterol = float(test_param.split(';')[9].split(':')[1])
    sodium = float(test_param.split(';')[10].split(':')[1])
    total_carbs = float(test_param.split(';')[11].split(':')[1])
    fiber = float(test_param.split(';')[12].split(':')[1])
    sugars = float(test_param.split(';')[13].split(':')[1])
    net_carbs = float(test_param.split(';')[14].split(':')[1])
    protein = float(test_param.split(';')[15].split(':')[1])
    sitting_duration = float(test_param.split(';')[16].split(':')[1])
    standing_duration = float(test_param.split(';')[17].split(':')[1])
    stepping_duration = float(test_param.split(';')[18].split(':')[1])
    sitting_at_work = float(test_param.split(';')[19].split(':')[1])
    standing_at_work = float(test_param.split(';')[20].split(':')[1])
    stepping_at_work = float(test_param.split(';')[21].split(':')[1])
    work_start_time = float(test_param.split(';')[22].split(':')[1])
    work_from_home = test_param.split(';')[23].split(':')[1]
    day_of_week = test_param.split(';')[24].split(':')[1]

    if work_from_home == 'true':
        work_from_home = 1
    elif work_from_home == 'false':
        work_from_home = 0
    else:
        print('Invalid work_from_home value. Exiting.')
        return
    is_monday = 1 if day_of_week == 'Monday' else 0
    is_tuesday = 1 if day_of_week == 'Tuesday' else 0
    is_wednesday = 1 if day_of_week == 'Wednesday' else 0
    is_thursday = 1 if day_of_week == 'Thursday' else 0
    is_friday = 1 if day_of_week == 'Friday' else 0

    X = np.array([fasting_gluc, recent_cgm, lunch_time, work_from_home, bmi, calories, calories_from_fat, total_fat,
                 saturated_fat, trans_fat, cholesterol, sodium, total_carbs, fiber, sugars, net_carbs, protein,
                 is_friday, is_monday, is_thursday, is_tuesday, is_wednesday, sitting_duration,standing_duration,
                 stepping_duration, sitting_at_work, standing_at_work, stepping_at_work, work_start_time]).reshape(1, -1)

    X_scaled = best_scaler.transform(X)
    y_pred = best_model.predict(X_scaled)
    with open(f'{output_folder}/simulate/prediction.txt', 'w') as file:
        file.write(f'Predicted AUC: {y_pred[0]}\n')

    print("START_PREDICTION:", y_pred[0], "END_PREDICTION")




def train_and_evaluate_v3(X, y, model_type, seed=42):
    # Load data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, train_size, test_size, scaler = get_train_and_test_data_v2(X, y, seed)
    # Model training
    model_name, hyper_txt, model = get_trained_model_v2(model_type, {'n_estimators': 50}, X_train_scaled, y_train, seed)
    if model_name is None:
        return

    # Metrics on the train set
    y_pred_train = model.predict(X_train_scaled)
    mse_train, mae_train, r2_train, rmse_train, nrmse_train = get_metrics(y_train, y_pred_train)

    # Model evaluation on the test set
    y_pred = model.predict(X_test_scaled)
    mse, mae, r2, rmse, nrmse = get_metrics(y_test, y_pred)

    # Save results
    return nrmse, model, scaler
