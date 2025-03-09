import os
import pandas as pd

def create_or_load_dummies(basepath, outputpath, force=False):
    if os.path.exists(f'{outputpath}/dataset_v2_with_dummies.csv') and not force:
        return pd.read_csv(f'{outputpath}/dataset_v2_with_dummies.csv')


    data = pd.read_csv(f'{basepath}dataset_nonan_final_v2.csv')
    cat_cols = pd.get_dummies(data['day_of_week'], prefix='is').astype(int)
    data = pd.concat([data, cat_cols], axis=1)
    data.drop(columns=['day_of_week'], inplace=True)
    data['work_at_home'] = data['work_at_home'].astype(int)

    data.to_csv(f'{outputpath}/dataset_v2_with_dummies.csv', index=False)
    return data


def load_dataset_with_respective_auc(basepath):
    if not os.path.exists(f'{basepath}/dataset_with_respective_auc_norm.csv'):
        print(f'Dataset with respective AUC not found at the path {basepath}/dataset_with_respective_auc_norm.csv.  Exiting.')
    return pd.read_csv(f'{basepath}/dataset_with_respective_auc_norm.csv')


def drop_unused_input_columns(data):
    return data.drop(columns=['user_id', 'phase', 'day'])

def drop_label_column(data):
    return data.drop(columns=['auc'])

def drop_noninput_columns(data):
    return data.drop(columns=['user_id', 'phase', 'day', 'auc'])

def get_label_column(data):
    return data['auc']