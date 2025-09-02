from . import  datadriver as dd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def init_output_dir(output_folder):
    """
    Create the output directory if it does not exist
    :param output_folder:
    :return:
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder+'/csvs/'):
        os.makedirs(output_folder+'/csvs/')
    if not os.path.exists(output_folder+'/plots/'):
        os.makedirs(output_folder+'/plots/')

    return output_folder


def calculate_and_save(X, corr_column, output_folder, filename):
    """
    Calculate the correlation and save it to a CSV file and plot it.
    If corr_column has a value other than 'all', only the correlation with that column will be calculated and plotted.
    :param X:
    :param corr_column:
    :param output_folder:
    :param filename:
    :return:
    """

    corr = X.corr()
    small_plot = False
    if corr_column != 'all':
        drop_columns = [col for col in corr.columns if col != corr_column]
        corr = corr.drop(columns=drop_columns)
        corr = corr.drop(index=corr_column)  # drop the row with the column name as it will always be 1
        small_plot = True

    corr.to_csv(f'{output_folder}/csvs/{filename}.csv')
    plot_correlation(corr, output_folder, filename, small_plot=small_plot)


def pearson_correlation(basepath, output_folder, allusers, corr_column):
    """

    :param basepath:
    :param output_folder:
    :param allusers:
    :param corr_column:
    :return:
    """

    data = dd.load_dataset_with_respective_auc(basepath)
    data = data.drop(columns=['auc'])
    maxpbg = data.pop('max_postprandial_gluc')
    data['MaxBGL'] = maxpbg

    hyperglycemia = data.pop('postprandial_hyperglycemia_140')
    data['MaxBGL>140'] = hyperglycemia

    maxpbg = data.pop('respective_auc')
    data['Incremental AUC'] = maxpbg

    label_col = data.pop('absolute_auc')  # remove the label column
    data['AUC'] = label_col  # add it back as the last column


    X = data.drop(columns=['phase', 'day', 'user_id'])
    # y = dd.get_label_column(data)

    correlation_output_folder = init_output_dir(f'{output_folder}/correlation_plots_{corr_column}')
    calculate_and_save(X, corr_column, correlation_output_folder, 'pearson_correlation')

    for user in allusers:
        user_data = data[data['user_id'] == user]
        user_X = user_data.drop(columns=['phase', 'day', 'user_id'])
        try:
            calculate_and_save(user_X, corr_column, correlation_output_folder, f'pearson_correlation_{user}')
        except RuntimeWarning as e:
            print(f'Error for user {user}: {e}')
            user_X.to_csv(f'{correlation_output_folder}/data_causing_error_{user}.csv')


def plot_correlation(corr, output_folder, filename, small_plot=False):
    if small_plot:
        plt.figure(figsize=(18, 15))
    else:
        plt.figure(figsize=(18, 15))

    sns.heatmap(corr*10, annot=True, cmap='coolwarm', fmt=".0f", vmin=-10, vmax=10)
    plt.savefig(f'{output_folder}/plots/{filename}.png')
    plt.close()
