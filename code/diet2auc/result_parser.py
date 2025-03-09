import os
import pandas as pd
import numpy as np

def parse_regression_results(basepath, output_folder):
    # parse the regression results and extract meaningful information.

    # read the regression results
    header = 'outcome,feature_set,model,hyper_params,train_rmse,train_nrmse,test_rmse,test_nrmse\n'
    txt_model = ''
    txt_feature_set = ''
    txt_outcome = ''

    for outcome in ['absolute_auc','respective_auc', 'max_postprandial_gluc']:
        lowest_nrmse_per_outcome = np.inf
        best_hyper_params_per_outcome = None
        best_model_per_outcome = None
        best_feature_set_per_outcome = None
        for feature_set in ['activpal_glycemic_load','activpal_macronutrients','self_reported_activity_glycemic_load',
                            'self_reported_activity_macronutrients','all_features']:

            lowest_nrmse_per_feature_set = np.inf
            best_hyper_params_per_feature_set = None
            best_model_per_feature_set = None

            for model in ['TabNet', 'XGBoost', 'Ridge','RandomForest','MLPRegressor']:
                full_output_path = f'{output_folder}/{feature_set}/{model}/{outcome}/results.csv'
                if not os.path.exists(full_output_path):
                    continue
                df = pd.read_csv(full_output_path)

                # get the best hyperparameters
                hyper_params = df['hyperparams'].unique()
                lowest_nrmse = np.inf

                best_hyper_params = None
                for hyper_param in hyper_params:
                    hyper_df = df[df['hyperparams'] == hyper_param]
                    this_nrmse = hyper_df['test_nrmse'].values.mean()
                    if this_nrmse < lowest_nrmse:
                        best_hyper_params = hyper_param
                        lowest_nrmse = this_nrmse

                    if lowest_nrmse < lowest_nrmse_per_feature_set:
                        best_hyper_params_per_feature_set = hyper_param
                        lowest_nrmse_per_feature_set = lowest_nrmse
                        best_model_per_feature_set = model

                    if this_nrmse < lowest_nrmse_per_outcome:
                        best_hyper_params_per_outcome = hyper_param
                        lowest_nrmse_per_outcome = this_nrmse
                        best_model_per_outcome = model
                        best_feature_set_per_outcome = feature_set

                hyper_df = df[df['hyperparams'] == best_hyper_params]
                train_rmse = hyper_df['train_rmse'].values.mean()
                train_nrmse = hyper_df['train_nrmse'].values.mean()
                test_rmse = hyper_df['test_rmse'].values.mean()
                test_nrmse = hyper_df['test_nrmse'].values.mean()

                txt_model += f'{outcome},{feature_set},{model},{best_hyper_params},{train_rmse},{train_nrmse},{test_rmse},{test_nrmse}\n'

            best_feature_result_path = f'{output_folder}/{feature_set}/{best_model_per_feature_set}/{outcome}/results.csv'
            best_feature_df = pd.read_csv(best_feature_result_path)
            fildf = best_feature_df[best_feature_df['hyperparams'] == best_hyper_params_per_feature_set]
            train_rmse = fildf['train_rmse'].values.mean()
            train_nrmse = fildf['train_nrmse'].values.mean()
            test_rmse = fildf['test_rmse'].values.mean()
            test_nrmse = fildf['test_nrmse'].values.mean()

            txt_feature_set += f'{outcome},{feature_set},{best_model_per_feature_set},{best_hyper_params_per_feature_set},{train_rmse},{train_nrmse},{test_rmse},{test_nrmse}\n'

        best_outcome_result_path = f'{output_folder}/{best_feature_set_per_outcome}/{best_model_per_outcome}/{outcome}/results.csv'
        best_outcome_df = pd.read_csv(best_outcome_result_path)
        fildf = best_outcome_df[best_outcome_df['hyperparams'] == best_hyper_params_per_outcome]
        train_rmse = fildf['train_rmse'].values.mean()
        train_nrmse = fildf['train_nrmse'].values.mean()
        test_rmse = fildf['test_rmse'].values.mean()
        test_nrmse = fildf['test_nrmse'].values.mean()


        txt_outcome += f'{outcome},{best_feature_set_per_outcome},{best_model_per_outcome},{best_hyper_params_per_outcome},{train_rmse},{train_nrmse},{test_rmse},{test_nrmse}\n'



    with open(f'{output_folder}/regression_summary_per_model.csv', 'w') as file:
        file.write(header+txt_model)

    with open(f'{output_folder}/regression_summary_per_feature_set.csv', 'w') as file:
        file.write(header+txt_feature_set)

    with open(f'{output_folder}/regression_summary_per_outcome.csv', 'w') as file:
        file.write(header+txt_outcome)


def parse_mlp_results(basepath, output_folder):
    # at first find the list of all hyperparameters
    full_output_path = f'{output_folder}/activpal_glycemic_load/MLPRegressor/absolute_auc/results.csv'
    df = pd.read_csv(full_output_path)
    hyper_params = df['hyperparams'].unique()

    header = 'hyper_params,outcome,feature_set,train_rmse,train_nrmse,test_rmse,test_nrmse\n'
    txt = ''
    for hyper_params in hyper_params:
        for outcome in ['absolute_auc','respective_auc', 'max_postprandial_gluc']:
            best_nrmse = np.inf
            best_feature_set = None
            best_train_rmse = None
            best_train_nrmse = None
            best_test_rmse = None
            for feature_set in ['activpal_glycemic_load','activpal_macronutrients','self_reported_activity_glycemic_load',
                                'self_reported_activity_macronutrients','all_features']:
                full_output_path = f'{output_folder}/{feature_set}/MLPRegressor/{outcome}/results.csv'

                if not os.path.exists(full_output_path):
                    continue
                df = pd.read_csv(full_output_path)
                hyper_df = df[df['hyperparams'] == hyper_params]
                train_rmse = hyper_df['train_rmse'].values.mean()
                train_nrmse = hyper_df['train_nrmse'].values.mean()
                test_rmse = hyper_df['test_rmse'].values.mean()
                test_nrmse = hyper_df['test_nrmse'].values.mean()

                if test_nrmse < best_nrmse:
                    best_nrmse = test_nrmse
                    best_feature_set = feature_set
                    best_train_rmse = train_rmse
                    best_train_nrmse = train_nrmse
                    best_test_rmse = test_rmse

            txt += f'{hyper_params},{outcome},{best_feature_set},{best_train_rmse},{best_train_nrmse},{best_test_rmse},{best_nrmse}\n'


    with open(f'{output_folder}/regression_summary_MLP_hyperparams.csv', 'w') as file:
        file.write(header+txt)

def parse_sensor_macro_results(basepath, output_folder):
    # at first find the list of all hyperparameters
    header = 'outcome,feature_set,model,hyper,train_rmse,train_nrmse,test_rmse,test_nrmse\n'
    txt = ''

    for outcome in ['absolute_auc','respective_auc', 'max_postprandial_gluc']:
        for feature_set in ['activpal_macronutrients','all_features']:
            for model in ['Ridge','RandomForest','MLPRegressor']:
                best_nrmse = np.inf
                best_feature_set = None
                best_train_rmse = None
                best_train_nrmse = None
                best_test_rmse = None
                best_hyper = None

                full_output_path = f'{output_folder}/{feature_set}/{model}/{outcome}/results.csv'
                df = pd.read_csv(full_output_path)
                hyper_params = df['hyperparams'].unique()
                for hyper in hyper_params:
                    hyper_df = df[df['hyperparams'] == hyper]
                    train_rmse = hyper_df['train_rmse'].values.mean()
                    train_nrmse = hyper_df['train_nrmse'].values.mean()
                    test_rmse = hyper_df['test_rmse'].values.mean()
                    test_nrmse = hyper_df['test_nrmse'].values.mean()

                    if test_nrmse < best_nrmse:
                        best_nrmse = test_nrmse
                        best_feature_set = feature_set
                        best_train_rmse = train_rmse
                        best_train_nrmse = train_nrmse
                        best_test_rmse = test_rmse
                        best_hyper = hyper

                txt += f'{outcome},{best_feature_set},{model},{best_hyper},{best_train_rmse},{best_train_nrmse},{best_test_rmse},{best_nrmse}\n'

    with open(f'{output_folder}/regression_summary_each_model_hyperparams.csv', 'w') as file:
        file.write(header+txt)



def measure_acceptable_prediction_rate(basepath, outputpath):
    # This function will measure the acceptable prediction rate for the models
    # The acceptable prediction rate is defined as the percentage of predictions that are within 20% of the actual value
    # This will be calculated for the test set of each model
    # The results will be stored in the output folder

    header = 'outcome,feature_set,model,hyper_params_5,hyperparams_10,hyperparams_15,hyperparams_20,acceptable_5,acceptable_10,acceptable_15_acceptable_20\n'
    txt = ''
    for outcome in ['absolute_auc','respective_auc', 'max_postprandial_gluc']:
        for feature_set in ['activpal_glycemic_load','activpal_macronutrients','self_reported_activity_glycemic_load',
                            'self_reported_activity_macronutrients','all_features']:
            for model in ['RandomForest','xgboost']:
                full_output_path = f'{outputpath}/{feature_set}/{model}/{outcome}/results.csv'
                if not os.path.exists(full_output_path):
                    continue
                df = pd.read_csv(full_output_path)

                # get the best hyperparameters
                hyper_params = df['hyperparams'].unique()
                best_acceptable_prediction_rates = {5:0, 10:0, 15:0, 20:0}
                best_hyper_params = {5:None, 10:None, 15:None, 20:None}
                for hyper_param in hyper_params:
                    prediction_df = pd.DataFrame()
                    for seed in [0, 10, 42]:
                        prediction_csv_path = f'{outputpath}/{feature_set}/{model}/{outcome}/test_results_{model}_{hyper_param}_{seed}.csv'
                        prediction_df = pd.concat([prediction_df, pd.read_csv(prediction_csv_path)])
                    acceptable_prediction_rates = {5:0, 10:0, 15:0, 20:0}
                    for tolerance in [5, 10, 15, 20]:
                        prediction_df[f'acceptable_{tolerance}'] = np.abs(prediction_df['absolute_auc'] - prediction_df['predicted']) < (tolerance/100.0) * prediction_df['absolute_auc']
                        acceptable_prediction_rates[tolerance] = prediction_df[f'acceptable_{tolerance}'].sum() / prediction_df.shape[0]

                        if acceptable_prediction_rates[tolerance] > best_acceptable_prediction_rates[tolerance]:
                            best_acceptable_prediction_rates[tolerance] = acceptable_prediction_rates[tolerance]
                            best_hyper_params[tolerance] = hyper_param

                txt += f'{outcome},{feature_set},{model},{best_hyper_params[5]},{best_hyper_params[10]},{best_hyper_params[15]},{best_hyper_params[20]},{best_acceptable_prediction_rates[5]},{best_acceptable_prediction_rates[10]},{best_acceptable_prediction_rates[15]},{best_acceptable_prediction_rates[20]}\n'


    with open(f'{outputpath}/acceptable_prediction_rate.csv', 'w') as file:
        file.write(header+txt)

