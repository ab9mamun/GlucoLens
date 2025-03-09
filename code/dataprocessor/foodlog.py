import pandas as pd
import numpy as np
import external.sortutil as sortutil

import os

def merge_lunch_times(basepath, phase, allusers):
    print('WARNING-- Some manual processing may be required after running this function "merge_lunch_times", as times can be in many different formats, such as 12:05pm 12:05 PM, etc.')
    phasepath = f'{basepath}foodlogs/{phase}/'
    folders = os.listdir(phasepath)
    folders.sort(key=sortutil.natural_keys)
    txt = 'PID,Day #,Date,Day,Meal type,Time\n'
    for folder in folders:
        if '_excels' not in folder:
            continue
        if phase == 'baseline':
            userid = folder.split('_')[0].split('-')[1]
        elif phase == 'condition1':
            userid = folder.split('_')[1]
        elif phase == 'condition2':
            userid = folder.split('_')[1]
        print(userid)
        if userid in allusers:
            files = os.listdir(f'{phasepath}{folder}/')
            files.sort(key=sortutil.natural_keys)
            # print(files)
            for file in files:
                if not file.endswith('.xlsx'):
                    continue
                print(file)
                df = pd.read_excel(f'{phasepath}{folder}/{file}')
                # print(df)
                fildf = df[df['Meal type'] == 'Lunch']
                if (fildf.shape[0] < 1):
                    continue
                else:
                    date = fildf['Date'].to_numpy()[0]
                    lunchtime = fildf['Time'].to_numpy()[0]
                    day = fildf['Day'].to_numpy()[0]
                    dayseq = fildf['Day #'].to_numpy()[0]
                    txt = txt + f'{userid},{dayseq},{date},{day},Lunch,{lunchtime}\n'

    with open(f'{basepath}all_lunch_times_{phase}.csv', 'w') as file:
        file.write(txt)

    print('WARNING-- Some manual processing may be required after running this function "merge_lunch_times", as times can be in many different formats, such as 12:05pm 12:05 PM, etc.')
