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
from . import self_supervised_file
from imblearn.over_sampling import SMOTE

from llm_helper import call_llm_api


def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    nrmse = rmse / y_true.mean()
    return mse, mae, r2, rmse, nrmse


def augment_with_SMOTE(X, y, seed):
    """
    Augment dataset using Synthetic Minority Over-sampling Technique (SMOTE)
    to generate 10 times more data.

    Parameters:
        X (np.ndarray): Original feature set of shape (n_samples, n_features)
        y (np.ndarray): Original labels of shape (n_samples,) or (n_samples, n_classes)

    Returns:
        X_augmented (np.ndarray): Augmented feature set of shape (10 * n_samples, n_features)
        y_augmented (np.ndarray): Augmented labels of shape (10 * n_samples,) or (10 * n_samples, n_classes)
    """
    smote = SMOTE(sampling_strategy=10.0, random_state=seed)  # Generate 10 times more data

    X_augmented, y_augmented = smote.fit_resample(X, y)

    return X_augmented, y_augmented


def augment_with_GaussianNoise(X, y, seed, multiply_factor=5, noise_std=0.05):
    """
    Augments the dataset by adding Gaussian noise.

    Parameters:
        X (np.ndarray): Feature matrix (n_samples, n_features).
        y (np.ndarray): Target array (n_samples,).
        seed (int): Random seed for reproducibility.
        multiply_factor (int): Number of times to augment the dataset.
        noise_std (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Augmented feature matrix.
        np.ndarray: Augmented target array.
    """
    np.random.seed(seed)  # Set seed for reproducibility

    # Initialize augmented datasets
    X_augmented = [X]  # Start with original X
    y_augmented = [y]  # Start with original y

    # Generate augmented data
    for _ in range(multiply_factor - 1):  # Add (multiply_factor - 1) new sets
        noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
        X_augmented.append(X + noise)
        y_augmented.append(y)

    # Concatenate original and augmented data
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.hstack(y_augmented)

    return X_augmented, y_augmented

def save_results(outputpath, header_txt, result_txt, filename='results.csv'):
    if not os.path.exists(f'{outputpath}/{filename}'):
        with open(f'{outputpath}/{filename}', 'w') as file:
            file.write(header_txt+result_txt)
    else:
        with open(f'{outputpath}/{filename}', 'a') as file:
            file.write(result_txt)



def diet2auc_complete(basepath, outputpath):
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
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, outputpath, 'self_reported_activity_macronutrients')


    # with activPAL data and macronutrients
    print('Now working with activPAL features and macronutrients')
    df1 = df.drop(columns=['recent_activity', 'glycemic_load'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    # 4. postprandial_hyperglycemia

    execute_four_tasks(df1, outputpath, 'activpal_macronutrients')

    # with self-reported activity and glycemic load
    print('Now working with self-reported activity and glycemic load')
    df1 = df.drop(columns=['Net Carbs(g)','Protein (g)','Fiber (g)','Total Fat (g)', 'sitting_total', 'standing_total', 'stepping_total', 'sitting_at_work', 'standing_at_work', 'stepping_at_work'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, outputpath, 'self_reported_activity_glycemic_load')

    # with activPAL data and glycemic load
    print('Now working with activPAL features and glycemic load')
    df1 = df.drop(columns=['recent_activity', 'Net Carbs(g)','Protein (g)','Fiber (g)','Total Fat (g)'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, outputpath, 'activpal_glycemic_load')

    # with all features
    print('Now working with all valid features')
    df1 = df
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, outputpath, 'all_features')
    #pass

def execute_four_tasks(df, outputpath, foldername):

    if not os.path.exists(f'{outputpath}/{foldername}'):
        os.makedirs(f'{outputpath}/{foldername}')
    df.to_csv(f'{outputpath}/{foldername}/dataset_{foldername}.csv', index=False)

    # 1. respective_auc
    # df1 = df.drop(columns=['absolute_auc', 'max_postprandial_gluc', 'postprandial_hyperglycemia_140'])
    # df1 = df1.dropna()
    # X = df1.drop(columns=['respective_auc']).values.astype(float)
    # y = df1['respective_auc'].to_numpy().astype(float)
    # run_regression(X,y, outputpath, foldername, 'respective_auc', df1.columns)

    # 2. absolute_auc
    df1 = df.drop(columns=['respective_auc', 'max_postprandial_gluc', 'postprandial_hyperglycemia_140', 'norm_auc'])
    df1 = df1.dropna()
    X = df1.drop(columns=['absolute_auc']).values.astype(float)
    y = df1['absolute_auc'].to_numpy().astype(float)

    run_regression(X, y, outputpath, foldername, 'absolute_auc', df1.columns)

    # 3. max_glucose
    # df1 = df.drop(columns=['respective_auc', 'absolute_auc', 'postprandial_hyperglycemia_140'])
    # df1 = df1.dropna()
    # X = df1.drop(columns=['max_postprandial_gluc']).values.astype(float)
    # y = df1['max_postprandial_gluc'].to_numpy().astype(float)
    #
    # run_regression(X, y, outputpath, foldername, 'max_postprandial_gluc', df1.columns)
    # #
    # # 4. postprandial_hyperglycemia
    # df1 = df.drop(columns=['respective_auc', 'absolute_auc', 'max_postprandial_gluc'])
    # df1 = df1.dropna()
    # X = df1.drop(columns=['postprandial_hyperglycemia_140']).values.astype(float)
    # y = df1['postprandial_hyperglycemia_140'].to_numpy().astype(int)
    # run_classification(X, y, outputpath, foldername, 'postprandial_hyperglycemia_140', df1.columns)


def run_regression(X, y, outputpath, foldername, task, colnames):
    all_hyper_params = {
        'self_supervised': [{'settings': 'default'}],
        'ridge': [{'alpha': 1.0}, {'alpha': 0.1}, {'alpha': 0.01}],
        'lasso': [{'alpha': 1.0}, {'alpha': 0.1}, {'alpha': 0.01}],

        'randomforest': [{'n_estimators': 10},
                         {'n_estimators': 50},
                         {'n_estimators': 100}],


        # the following parameters for random forest are for the discussion section (experimenting with different feature sets)
        # 'randomforest': [{'n_estimators': 10, 'max_nodes':24},
        #                  {'n_estimators': 10, 'max_nodes': 48},
        #                  {'n_estimators': 10, 'max_nodes': 96},
        #                  {'n_estimators': 50, 'max_nodes': 24},
        #                  {'n_estimators': 50, 'max_nodes': 48},
        #                  {'n_estimators': 50, 'max_nodes': 96},
        #                  {'n_estimators': 100, 'max_nodes': 24},
        #                  {'n_estimators': 100, 'max_nodes': 48},
        #                  {'n_estimators': 100, 'max_nodes': 96}],
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

    for model_type in ['randomforest']:#, 'mlpregressor']:
        print('Now running experiments with model: ', model_type)
        hyper_params_array = all_hyper_params[model_type]
        for hyper_params in hyper_params_array:
            for seed in [0, 10, 42]:
                train_and_evaluate_v2(outputpath, foldername, task, colnames,  hyper_params, X, y, model_type, seed=seed)

def run_classification(X, y, outputpath, foldername, task, colnames):
    pass


def train_and_evaluate_v2(outputpath, foldername, task, colnames, hyper_params, X, y, model_type, seed=42):

    # Load data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, train_size, test_size, scaler = get_train_and_test_data_v2(X, y, seed)

    # Augment the training data with SMOTE if the model is not self-supervised
    print(f"======================= Training for model {model_type} and seed {seed}============================")
    print("Size of training data before augmentation:", X_train_scaled.shape)
    print("Size of test data:", X_test_scaled.shape)
    print("Size of training labels:", y_train.shape)
    print("Size of test labels:", y_test.shape)
    if model_type != 'self_supervised':
        X_train_scaled_augmented, y_train_augmented = augment_with_GaussianNoise(X_train_scaled, y_train, seed)

    else:
        X_train_scaled_augmented, y_train_augmented = X_train_scaled, y_train

    print("Size of training data after Gaussian augmentation:", X_train_scaled_augmented.shape)
    #return


    # Model training
    if model_type == 'tabnet':
        y_train_augmented = np.array(y_train_augmented).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)

    if model_type == 'self_supervised':
        model_name, hyper_txt, model, y_pred_train, y_train, y_pred, y_test = self_supervised_file.perform(X_train_scaled_augmented, y_train_augmented, X_test_scaled, y_test)


    else:
        model_name, hyper_txt, model = get_trained_model_v2(model_type, hyper_params, X_train_scaled_augmented, y_train_augmented, seed)
    if model_name is None:
        return

    # Metrics on the train set
    if model_type != 'self_supervised':
        y_pred_train_augmented = model.predict(X_train_scaled_augmented)
        y_pred_train = model.predict(X_train_scaled)


    if model_type == 'tabnet':
        y_pred_train_augmented = np.array(y_pred_train_augmented).reshape(-1,)


    mse_train, mae_train, r2_train, rmse_train, nrmse_train = get_metrics(y_train_augmented, y_pred_train_augmented)

    # Model evaluation on the test set
    if model_type != 'self_supervised':
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
        #max_nodes = hyper_params['max_nodes']
        hyper_txt = f'n_estimators_{n_estimators}'#_max_nodes_{max_nodes}'
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


def self_supervised(basepath, output_folder):
    # Load data
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
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, output_folder, 'self_reported_activity_macronutrients')

    # with activPAL data and macronutrients
    print('Now working with activPAL features and macronutrients')
    df1 = df.drop(columns=['recent_activity', 'glycemic_load'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, output_folder, 'activpal_macronutrients')

    # with self-reported activity and glycemic load
    print('Now working with self-reported activity and glycemic load')
    df1 = df.drop(columns=['Net Carbs(g)', 'Protein (g)', 'Fiber (g)', 'Total Fat (g)', 'sitting_total',
                           'standing_total', 'stepping_total', 'sitting_at_work', 'standing_at_work', 'stepping_at_work'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, output_folder, 'self_reported_activity_glycemic_load')

    # with activPAL data and glycemic load
    print('Now working with activPAL features and glycemic load')
    df1 = df.drop(columns=['recent_activity', 'Net Carbs(g)', 'Protein (g)', 'Fiber (g)', 'Total Fat (g)'])
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, output_folder, 'activpal_glycemic_load')

    # with all features
    print('Now working with all valid features')
    df1 = df
    # 1. respective_auc
    # 2. absolute_auc
    # 3. max_glucose
    # 4. postprandial_hyperglycemia
    execute_four_tasks(df1, output_folder, 'all_features')

def diet2auc_llm(basepath, output_folder):

    # Load data
    df = dd.load_dataset_with_respective_auc(basepath)
    df = df.drop(columns=['user_id', 'phase', 'day', 'auc', 'baseline_activity'])

    # Implement LLM-based prediction logic here
    # ...

    # use OpenAI API, Llama3.1, Claude, Gemini, DeepSeek, Mistrial AI

    # create a prompt for the LLM

    df1 = df.drop(columns=['respective_auc', 'max_postprandial_gluc', 'postprandial_hyperglycemia_140', 'norm_auc'])
    df1 = df1.dropna()
    df2 = df1.drop(columns=['absolute_auc'])
    X = df2.values.astype(float)
    y = df1['absolute_auc'].to_numpy().astype(float)

    # create a header for feature values, real_output, and predictions by all LLMs as columns
    header = "fasting_glucose,recent_cgm,lunch_time,work_at_home,recent_activity,bmi,Calories,Calories From Fat,Total Fat (g),Saturated Fat (g),Trans Fat (g),Cholesterol (mg),Sodium (mg),Total Carbs (g),Fiber (g),Sugars (g),Net Carbs(g),Protein (g),is_Friday,is_Monday,is_Thursday,is_Tuesday,is_Wednesday,sitting_total,standing_total,stepping_total,sitting_at_work,standing_at_work,stepping_at_work,work_start_time,glycemic_load,real_output,gpt-3.5-turbo,gpt-4,gemini-2.0-flash-001,deepseek-chat,mistral-large-2411,claude-opus-4-20250514,grok-3\n"
    output_file = f'{output_folder}/diet2auc_llm_predictions.csv'
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write(header)

    for i in range(len(df2)):
        zero_shot_prompt = f"Instruction: The goal is to predict the 3-hour postprandial AUC based on the following features: {df2.columns.tolist()}. fasting_glucose and recent_cgm are given in mg/dL. lunch_time and work_start time are in time represented by an hour value (e.g.,7.75 means 7:45 AM, 13.50 means 1:30 PM). recent_activity score is calculated by taking the average percentage of time spent in walking activity in the previous days of the same phase and adding with 0.5 times the average percentage of time spent in standing activity in the previous days of the same phase. sitting, standing, and stepping features are in seconds for the specific day before lunch. Predict the 3-hour postprandial AUC for the given features. Give me just the number enclosed within the <Prediction></Prediction> tags. Input: {df2.iloc[0].values.tolist()}. Output: "

        #print(zero_shot_prompt)

        # Call the LLM API with the prompt
        prediction_gpt_3_5 = extract_prediction(call_llm_api(basepath, zero_shot_prompt, model='gpt-3.5-turbo'))

        prediction_gpt_4 = extract_prediction(call_llm_api(basepath, zero_shot_prompt, model='gpt-4'))

        prediction_gemini = extract_prediction(call_llm_api(basepath, zero_shot_prompt, model='gemini-2.0-flash-001'))

        #prediction_deepseek = extract_prediction(call_llm_api(basepath, zero_shot_prompt, model='deepseek-reasoner'))

        prediction_deepseek = extract_prediction(call_llm_api(basepath, zero_shot_prompt, model='deepseek-chat'))

        prediction_mistral = extract_prediction(call_llm_api(basepath, zero_shot_prompt, model='mistral-large-2411'))

        prediction_claude = extract_prediction(call_llm_api(basepath, zero_shot_prompt, model='claude-opus-4-20250514'))

        #prediction_llama = extract_prediction(extract_predictioncall_llm_api(basepath, zero_shot_prompt, model='llama-3.1'))

        prediction_grok = extract_prediction(call_llm_api(basepath, zero_shot_prompt, model='grok-3'))

        # append predictions to the file
        with open(output_file, 'a') as f:
            row = ','.join(map(str, df2.iloc[i].values.tolist())) + f',{y[i]},{prediction_gpt_3_5},{prediction_gpt_4},{prediction_gemini},{prediction_deepseek},{prediction_mistral},{prediction_claude},{prediction_grok}\n'
            f.write(row)

    # calculate the RMSE, MSE, MAE, R2, and NRMSE for each model and saved in a CSV file

    df_predictions = pd.read_csv(output_file)
    models = ['gpt-3.5-turbo', 'gpt-4', 'gemini-2.0-flash-001', 'deepseek-chat', 'mistral-large-2411',
              'claude-opus-4-20250514', 'grok-3']

    preds2metrics(df_predictions, output_folder, models)

def extract_prediction(response):
    """
    Extracts the prediction from the LLM response.
    The prediction is expected to be enclosed within <Prediction></Prediction> tags.
    """
    if '<Prediction>' in response and '</Prediction>' in response:
        start = response.index('<Prediction>') + len('<Prediction>')
        end = response.index('</Prediction>')
        return float(response[start:end].strip())
    else:
        #print("No valid prediction found in the response.")
        return -1

def preds2metrics(df_predictions, output_folder, models):
    metrics = ['RMSE', 'MSE', 'MAE', 'R2', 'NRMSE']

    results = []

    for model in models:
        y_true = df_predictions['real_output'].values
        y_pred = df_predictions[model].values

        # Apply mask to ignore anomalies
        mask = (y_pred >= 120) & (y_pred <= 700)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        # Calculate metrics with filtered values
        mse = mean_squared_error(y_true_filtered, y_pred_filtered)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
        r2 = r2_score(y_true_filtered, y_pred_filtered)
        nrmse = rmse / (np.max(y_true_filtered) - np.min(y_true_filtered))

        results.append([model, rmse, mse, mae, r2, nrmse])

    # Save the results to a CSV file
    results_df = pd.DataFrame(results, columns=['Model'] + metrics)
    results_df.to_csv(f'{output_folder}/diet2auc_llm_metrics.csv', index=False)

def calculate_llm_metrics(basepath, output_folder):
    # Load the predictions file
    df_predictions = pd.read_csv(f'{output_folder}/diet2auc_llm_predictions.csv')

    # Define the models to evaluate
    models = ['gpt-3.5-turbo', 'gpt-4', 'gemini-2.0-flash-001', 'deepseek-chat',
              'claude-opus-4-20250514', 'grok-3']

    # Calculate metrics for each model
    preds2metrics(df_predictions, output_folder, models)



def diet2auc_hybrid(basepath, output_folder, only_claude=False, augmented=False):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from pytorch_tabnet.tab_model import TabNetRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Load dataset
    if only_claude:
        df = pd.read_csv(f"{basepath}/diet2auc_with_llm_predictions_only_claude.csv")
    else:
        df = pd.read_csv(f"{basepath}/diet2auc_with_llm_predictions.csv")  # Replace with the path to your dataset

    # Clip specified columns
    if only_claude:
        columns_to_clip = ["claude-opus-4-20250514"]
    else:
        columns_to_clip = [
            "gpt-3.5-turbo", "gpt-4", "gemini-2.0-flash-001",
            "deepseek-chat", "claude-opus-4-20250514", "grok-3"
        ]
    df[columns_to_clip] = df[columns_to_clip].clip(lower=120, upper=700)

    # Separate features and target
    X = df.drop(columns=['real_output'])
    y = df['real_output']

    # Initialize models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0),
        'TabNet': TabNetRegressor(seed=42, verbose=0)
    }

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, model in models.items():
        fold_rmse, fold_mae, fold_nrmse = [], [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # TabNet requires NumPy arrays
            if model_name == "TabNet":
                X_train, X_test = X_train.values, X_test.values
                y_train, y_test = y_train.values, y_test.values

                # reshape
                y_train = y_train.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)

            if augmented:
                # Augment the training data with Gaussian noise
                noise = np.random.normal(0, 0.01, X_train.shape)
                X_train = np.concatenate([X_train,  X_train + noise], axis=0)
                y_train = np.concatenate([y_train, y_train], axis=0)
                # more noise
                noise = np.random.normal(0, 0.02, X_train.shape)
                X_train = np.concatenate([X_train,  X_train + noise], axis=0)
                y_train = np.concatenate([y_train, y_train], axis=0)
                # more noise
                noise = np.random.normal(0, 0.03, X_train.shape)
                X_train = np.concatenate([X_train,  X_train + noise], axis=0)
                y_train = np.concatenate([y_train, y_train], axis=0)



            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if model_name == "TabNet":
                y_pred = y_pred.reshape(-1,)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            nrmse = rmse / (np.max(y_test) - np.min(y_test))

            fold_rmse.append(rmse)
            fold_mae.append(mae)
            fold_nrmse.append(nrmse)

        # Store average metrics for the model
        results.append([
            model_name,
            np.mean(fold_rmse),
            np.mean(fold_mae),
            np.mean(fold_nrmse)
        ])

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'NRMSE'])
    if only_claude:
        if augmented:
            results_df.to_csv(f"{output_folder}/model_metrics_only_claude_augmented.csv", index=False)
        else:
            results_df.to_csv(f"{output_folder}/model_metrics_only_claude.csv", index=False)
    else:
        results_df.to_csv(f"{output_folder}/model_metrics.csv", index=False)


