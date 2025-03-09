
from dataprocessor import foodlog
from dataprocessor import cgmprocessor
from diet2auc import get_features, models, correlations, result_parser
import os
import argparse
import warnings
def main(args):
    """
        How will the tasks be handled?
       1. The allusers list will have all the users regardless of the phase and task. The cases for missing data will be handled in the respective functions.
       It is the responsibility of the function to handle the missing data.
       2. The task will be passed as an argument to the main function.
       3. The main function will call the respective function based on the task.
       4. The respective function will handle the task.
    """
    #warnings.filterwarnings("error")
    basepath = '../data/' # replace with the absolute path to the data folder for making it compatible with calling from other files.

    print(args)
    args = parser.parse_args()
    
    allusers = ['Participant1', 'Participant2', 'Participant3', 'Participant4', 'Participant5', 'Participant6',
                'Participant7', 'Participant8', 'Participant9', 'Participant10']

    task = args.task
    test_param = args.test_param
    exp_name = args.exp_name
    seed = args.seed
    model = args.model
    corr_column = args.corr_column
    hyper_threshold = args.hyper_threshold

    output_folder = os.path.join(basepath, 'output', exp_name)
    os.makedirs(output_folder, exist_ok=True)

    if task is None:
        print('No task specified. Exiting.')
        return

    print(f'Running task: {task}, for experiment: {exp_name}')
    if task == 'create_dataset':
        # Create the dataset
        # This will be a CSV file with the following columns:
        # user_id, phase, day, fasting_glucose, recent_cgm, day_of_week, lunch_time, work_at_home, baseline_activity, recent_activity, bmi, auc
        # The dataset will be stored in the output folder
        txt = get_features.get_all_features(basepath, allusers)
        #print(txt)
        with open(f'{output_folder}/dataset.csv', 'w') as file:
            file.write(txt)

    elif task == 'add_activpal':
        # Add the activPAL data to the dataset
        # This will be a CSV file with the following additional columns:
        # sitting_total, standing_total, stepping_total, sitting_at_work, standing_at_work, stepping_at_work, work_start_time
        # The dataset will be stored in the output folder
        output_df = get_features.add_activPAL(basepath, output_folder)
        output_df.to_csv(f'{output_folder}/dataset_with_activPAL.csv', index=False)
        #pass

    elif task == 'add_gload':
        # Add the glucose load data to the dataset
        # This will be a CSV file with the following additional column:
        # glycemic_load
        # The dataset will be stored in the output folder
        output_df = get_features.add_glycemic_load(output_folder)
        output_df.to_csv(f'{output_folder}/dataset_with_gload.csv', index=False)
        #pass
    elif task == 'add_respective_auc':
        # Add the respective AUC values to the dataset
        # This will be a CSV file with the following additional column: respective_auc
        # The dataset will be stored in the output folder

        output_df = get_features.add_respective_auc(basepath, output_folder, hyper_threshold=hyper_threshold)
        output_df.to_csv(f'{output_folder}/dataset_with_respective_auc.csv', index=False)


    elif task == 'diet2auc':
        # Run the diet2auc model
        # This will use the dataset created in the previous step
        # The model will be trained and tested
        # The results will be stored in the output folder
        models.train_and_evaluate(basepath, output_folder, model, seed)

    elif task == 'diet2auc_ultra':
        # Run the diet2auc model
        # This will use the dataset created in the previous step
        # The model will be trained and tested
        # The results will be stored in the output folder
        models.diet2auc_ultra(basepath, output_folder)

    elif task == 'parse_regression_results':
        # This will use the dataset created in the previous step
        # The results will be stored in the output folder
        result_parser.parse_regression_results(basepath, output_folder)

    elif task == 'parse_mlp_results':
        # The results will be stored in the output folder
        result_parser.parse_mlp_results(basepath, output_folder)
    elif task == 'parse_sensor_macro_results':
        # The results will be stored in the output folder
        result_parser.parse_sensor_macro_results(basepath, output_folder)
    elif task == 'test':
        print(f'Received param: {test_param}')

    elif task == 'simulate':
        print('Note: The simulator task is designed to be called from the simulator app because it expects all the '+
              'variables in a specific format, but it can be tested without the app too.')
        models.simulate(basepath, output_folder, 'RandomForest', test_param)

    elif task == 'plot_1_auc':
        get_features.plot_beautiful_auc(basepath, output_folder, user=args.plot_user, corrected_date=args.plot_date, lunch_time=args.plot_lunch_time, phase= args.plot_phase)
    elif task == 'correlation':
        correlations.pearson_correlation(basepath, output_folder, allusers, corr_column)

    elif task == 'acceptable_prediction_rate':
        result_parser.measure_acceptable_prediction_rate(basepath, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WorkWell Processing.')

    parser.add_argument('--dataset',
                        default='workwell',
                        type=str)
    parser.add_argument('--output_folder',
                        default='output',
                        help='The folder where the output will be stored',
                        type=str)
    parser.add_argument('--exp_name',
                        default='default_exp_name',
                        help='A unique name for the experiment. If not unique, the existing experiment may be overwritten.',
                        type=str)
    parser.add_argument('--task',
                        default='None',
                        help='Choose from diet2auc, diet2auc_ultra, parse_regression_results, parse_mlp_results, simulate, plot_1_auc, correlation',
                        type=str)
    parser.add_argument('--seed',
                        default=42,
                        help='Seed for random number generation',
                        type=int)

    parser.add_argument('--model',
                        default='None',
                        help='Type of model to use. Choose from ridge, randomforest, mlpregressor',
                        type=str)

    parser.add_argument('--corr_column',
                        default='all',
                        help='Columns to see the correlation for',
                        type=str)

    parser.add_argument('--hyper_threshold',
                        default=140,
                        help='Threshold for hyperglycemia (default: 140)',
                        type=int)

    parser.add_argument('--test_param',
                        default='None',
                        help='Test param from external application',
                        type=str)

    parser.add_argument('--plot_user',
                        default='None',
                        help='User ID for the plot',
                        type=str)
    
    parser.add_argument('--plot_date',
                        default='None',
                        help='Date for the plot',
                        type=str)
    
    parser.add_argument('--plot_lunch_time',
                        default='None',
                        help='lunch time for the plot',
                        type=float)
    
    parser.add_argument('--plot_phase',
                        default='None',
                        help='phase for the plot',
                        type=str)

    main(args=parser.parse_args())
    
