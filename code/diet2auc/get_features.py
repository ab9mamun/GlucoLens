import pandas as pd
import numpy as np
import os
from datetime import datetime, date as date_func
from dataprocessor import cgmprocessor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def get_all_features(basepath, allusers):
    """
    Parse the raw data files and return the features for each user

    :return:
    """
    txt = 'user_id,phase,day,fasting_glucose,recent_cgm,day_of_week,lunch_time,work_at_home,baseline_activity,recent_activity,bmi,auc\n'
    workdf = pd.read_excel(f'{basepath}WorkLogs.xlsx', sheet_name='baseline')
    workdf = workdf._append(pd.read_excel(f'{basepath}WorkLogs.xlsx', sheet_name='condition1'))
    workdf = workdf._append(pd.read_excel(f'{basepath}WorkLogs.xlsx', sheet_name='condition2'))

    workdf.to_csv(f'{basepath}worklogs.csv', index=False)
    workdf = pd.read_csv(f'{basepath}worklogs.csv')
    dates = workdf['Date'].to_numpy().astype(str)
    # print(dates[0])
    for i in range(len(dates)):
        if '-' not in dates[i]:
            continue
        temp = str(dates[i]).split(':')[0].split('T')[0]
        yr, mon, day = temp.split('-')
        dates[i] = f'{mon}-{day}-{yr}'

    workdf['Date'] = dates

    for user in allusers:
        fildf1 = workdf[workdf['Participant_ID'] == user]
        for i, phase in enumerate(['baseline', 'condition1', 'condition2']):
            # print(f'Now processing: User {user}, phase {phase}')
            fildf = fildf1[fildf1['Phase'] == phase]
            alldays = fildf['Date'].unique()
            # print(alldays)
            for j in range(len(alldays)):
                day = alldays[j]

                # Get the fasting glucose
                fasting_glucose = get_fasting_glucose(basepath, user, phase, day)
                # Get the recent cgm
                recent_cgm = get_recent_cgm(basepath, user, phase, day)
                # Get the day of the week
                day_of_week = get_day_of_week(day)
                # Get the lunch time
                lunch_time = get_lunch_time(basepath, user, phase, day)
                # Get the work at home status
                work_at_home = get_work_at_home(fildf[fildf['Date'] == day])
                # Get the baseline activity level
                baseline_activity = get_baseline_activity(basepath, user)
                # Get the recent activity level
                recent_activity = get_recent_activity(basepath, user, phase, day)
                # Get the BMI
                bmi = get_bmi(basepath, user)
                # Get the postprandial AUC
                auc = get_postprandial_AUC(basepath, user, phase, day)

                txt = txt + f'{user},{phase},{day},{fasting_glucose},{recent_cgm},{day_of_week},{lunch_time},{work_at_home},{baseline_activity},{recent_activity},{bmi},{auc}\n'

    return txt


def get_fasting_glucose(basepath, participant, phase, date):
    """
    Get the fasting glucose for each user
    Here, fasting glucose is defined as the minimum glucose reading between 6 AM and 10 AM

    :return:
    """
    df = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{participant}_{phase}_cleaned_cgm.csv')
    dates = df['day'].to_numpy().astype(str)
    dates2 = ['' for i in range(len(dates))]
    flag = False
    for i in range(len(dates)):
        if '/' not in dates[i]:
            continue
        flag = True
        # print(dates[i])
        temp = dates[i].split(':')[0].split('T')[0]
        mon, day, yr = temp.split('/')
        # print(mon, day, yr)
        if len(mon) == 1:
            mon = f'0{mon}'
        if len(day) == 1:
            day = f'0{day}'

        dates2[i] = f'{str(mon)}-{str(day)}-{str(yr)}'
        # print(dates[i])
    if flag:
        df['day'] = dates2

    fildf = df[df['day'] == date]
   

    hours = fildf['hour'].to_numpy()
    gluc = fildf['corrected_glucose'].to_numpy()
    candidates = [gluc[i] for i, hour in enumerate(hours) if hour >= 6 and hour <= 10]
    if len(candidates) < 1:
        return None
    return min(candidates)


def get_recent_cgm(basepath, participant, phase, date):
    """
    Get the CGM reading from midnight to 8 AM for each user

    :return:
    """
    df = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{participant}_{phase}_cleaned_cgm.csv')
    dates = df['day'].to_numpy().astype(str)
    dates2 = ['' for i in range(len(dates))]
    flag = False
    for i in range(len(dates)):
        if '/' not in dates[i]:
            continue

        flag = True
        # print(dates[i])
        temp = dates[i].split(':')[0].split('T')[0]
        mon, day, yr = temp.split('/')
        # print(mon, day, yr)
        if len(mon) == 1:
            mon = f'0{mon}'
        if len(day) == 1:
            day = f'0{day}'

        dates2[i] = f'{str(mon)}-{str(day)}-{str(yr)}'
        # print(dates[i])
    if flag:
        df['day'] = dates2

    fildf = df[df['day'] == date]

    hours = fildf['hour'].to_numpy()
    gluc = fildf['corrected_glucose'].to_numpy()
    candidates = [gluc[i] for i, hour in enumerate(hours) if hour >= 0 and hour <= 8]
    if len(candidates) < 1:
        return None
    return np.mean(candidates)


def get_day_of_week(date):
    """
    Get the day of the week for each user

    :return:
    """
    from datetime import datetime
    thisdate = datetime.strptime(date, '%m-%d-%Y')
    return thisdate.strftime('%A')


def get_lunch_time(basepath, participant, phase, date):
    """
    Get the lunch time for each user

    :return:
    """
    lunchtimes = pd.read_csv(f'{basepath}lunch_time/all_lunch_times_{phase}.csv')
    dates = lunchtimes['Date'].to_numpy().astype(str)
    for i in range(len(dates)):
        if '-' not in dates[i]:
            continue
        temp = str(dates[i]).split(':')[0].split('T')[0]
        yr, mon, day = temp.split('-')
        dates[i] = f'{mon}-{day}-{yr}'

    lunchtimes['Date'] = dates

    fildf = lunchtimes[lunchtimes['PID'] == participant]
    fildf = fildf[fildf['Date'] == date]
    if fildf.shape[0] < 1:
        return None
    lunch_time = fildf['Time'].to_numpy()[0]
    if lunch_time != lunch_time:  # Check for NaN
        return None
    vals = lunch_time.split(':')
    hh = vals[0].strip()
    mm = vals[1].split()[0]
    timefloat = int(hh) + int(mm) / 60.0
    if 'PM' in lunch_time and timefloat < 12:
        timefloat += 12
    return timefloat


def get_work_at_home(fildf):
    """
    Get the work at home status for each user

    :return:
    """

    fildf2 = fildf[fildf['Got to work by'] == 'Work from home']
    if fildf2.shape[0] > 0:
        return True
    return False


def get_baseline_activity(basepath, participant):
    """
    Get the baseline activity level for each user

    :return:
    """
    wdf = pd.read_csv(f'{basepath}worklogs.csv')
    fildf = wdf[wdf['Participant_ID'] == participant]
    fildf = fildf[fildf['Phase'] == 'baseline']
    fildf = fildf.fillna(-1)

    walking = fildf['Walking'].to_numpy()
    try:
        walking = walking.astype(int)
    except RuntimeWarning:
        print(f'Warning in participant {participant}, phase baseline')

    standing = fildf['Standing'].to_numpy()
    try:
        standing = standing.astype(int)
    except RuntimeWarning:
        print(f'Warning in participant {participant}, phase baseline')
    index1 = np.argwhere(walking == -1)
    index2 = np.argwhere(standing == -1)

    if len(index1) > 0 or len(index2) > 0:
        index = np.union1d(index1, index2)
        walking2 = np.delete(walking, index)
        standing2 = np.delete(standing, index)
    else:
        walking2 = walking
        standing2 = standing

    if len(walking2) < 1 or len(standing2) < 1:
        return 0
    return np.mean(walking2 + 0.5 * standing2)


def get_recent_activity(basepath, participant, phase, date):
    """
    Get the recent activity level for each user

    :return:
    """
    wdf = pd.read_csv(f'{basepath}worklogs.csv')
    fildf = wdf[wdf['Participant_ID'] == participant]
    fildf = fildf[fildf['Phase'] == phase]
    fildf = fildf.fillna(-1)
    standing = fildf['Standing'].to_numpy()
    try:
        standing = standing.astype(int)
    except RuntimeWarning:
        print(f'Warning in participant {participant}, phase {phase}')
        print(standing)

    walking = fildf['Walking'].to_numpy()
    try:
        walking = walking.astype(int)
    except RuntimeWarning:
        print(f'Warning in participant {participant}, phase {phase}')
        print(walking)

    if len(standing) < 1 or len(walking) < 1:
        return 0

    dates = fildf['Date'].to_numpy()
    for i in range(len(dates)):
        datesi_corrected = dates[i].split(':')[0].split('T')[0]
        yr, mon, day = datesi_corrected.split('-')
        datesi_corrected = f'{mon}-{day}-{yr}'
        if datesi_corrected == date:
            if len(walking[:i]) < 1 or len(standing[:i]) < 1:
                return 0

            index1 = np.argwhere(walking[:i] == -1)
            index2 = np.argwhere(standing[:i] == -1)
            if len(index1) > 0 or len(index2) > 0:
                index = np.union1d(index1, index2)
                try:
                    walking2 = np.delete(walking[:i], index)
                    standing2 = np.delete(standing[:i], index)
                except IndexError:
                    print(f'IndexError in participant {participant}, phase {phase}')
                    print(walking)
                    print(standing)
                    print(i)
                    print(i)
                    return 0
            else:
                walking2 = walking[:i]
                standing2 = standing[:i]

            if len(walking2) < 1 or len(standing2) < 1:
                return 0
            return np.mean(walking2 + 0.5 * standing2)
    return 0

def get_bmi(basepath, participant):
    """
    Get the BMI for each user

    :return:
    """
    bmidf = pd.read_excel(f'{basepath}WW_BMI.xlsx')
    simple_pid = int(participant.split('WW')[1])
    fildf = bmidf[bmidf['PID'] == simple_pid]
    # print(fildf)
    if fildf.shape[0] < 1:
        return None
    bmi = fildf['Baseline'].to_numpy()[0]
    return bmi



def get_postprandial_AUC(basepath, participant, phase, date):
    """
    Get the postprandial AUC for each user

    :return:
    """
    df = pd.read_csv(f'{basepath}auc/auc_{phase}.csv')

    fildf = df[df['PID'] == participant]
    if fildf.shape[0] < 1:
        return None

    dates = fildf['Date'].to_numpy().astype(str)
    dates2 = ['' for i in range(len(dates))]
    flag = False
    for i in range(len(dates)):
        if '/' not in dates[i]:
            continue

        flag = True
        # print(dates[i])
        temp = dates[i].split(':')[0].split('T')[0]
        mon, day, yr = temp.split('/')
        # print(mon, day, yr)
        if len(mon) == 1:
            mon = f'0{mon}'
        if len(day) == 1:
            day = f'0{day}'
        if len(yr) == 2:
            yr = f'20{yr}'

        dates2[i] = f'{str(mon)}-{str(day)}-{str(yr)}'
        # print(dates[i], dates2[i])
    if flag:
        # print(fildf.shape, len(dates2))
        fildf = fildf.drop('Date', axis=1)
        fildf['Date'] = dates2

    # print(fildf['Date'][:5])
    # print(date)
    fildf = fildf[fildf['Date'] == date]
    if fildf.shape[0] < 1:
        return None

    auc_arr = fildf['Postprandial_AUC (3hrs)'].to_numpy()
    return auc_arr[0]


def get_ms_access_time_v1(year, month, day):
    """
    Get the time of the MS Access file for the given date
    :param year:
    :param month:
    :param day:
    :return:
    """
    date_now = date_func(year, month, day)
    date_0 = date_func(1899, 12, 30)
    return (date_now - date_0).days


def get_ms_access_time_v2(year, month, day, hr, min, sec):
    """
    Get the time of the MS Access file for the given date
    :param year:
    :param month:
    :param day:
    :return:
    """
    days = get_ms_access_time_v1(year, month, day)
    return days + hr / 24.0 + min / (24 * 60.0) + sec / (24 * 60 * 60.0)


def get_activity_before_lunch(basepath, participant, phase, date, lunch_time):
    """
    This function extracts the activity data from the activPAL data for the user before lunch on a given day.
    This function is called from the add_activPAL function. The curious readers are encouraged to look at the
    add_activPAL function to understand the context of this function.
    @param basepath:
    @param participant:

    :return:
    """
    # these are the column  names in the activPAL data file. There is an additional delimiter at the end of every row
    # of the event file, that is why we have an additional column name at the end.
    event_colnames = ['Time', 'Time(approx)', 'Data Count', 'Event Type', 'Duration (s)',
                      'Waking Day', 'Cumulative Step Count', 'Activity Score (MET.h)',
                      'AbsSumDiffX', 'AbsSumDiffY', 'AbsSumDiffZ', 'Time Upright (s)',
                      'Time Upside Down (s)', 'Time Back Lying (s)', 'Time Front Lying (s)',
                      'Time Left Lying (s)', 'Time Right Lying (s)', '_blank_']

    # read the worklogs file. We will need this for the work start and end times
    # at first filter the worklogs based on user id and phase
    workdf = pd.read_csv(f'{basepath}worklogs.csv')
    fildf = workdf[workdf['Participant_ID'] == participant]
    fildf = fildf[fildf['Phase'] == phase]
    phase_alias = {'baseline': 'Baseline', 'condition1': 'Condition 1', 'condition2': 'Condition 2'}

    # convert the date to the format that is used in worklogs file. For example, 3/4/21 to 2021-03-04
    month, day, year = date.split('/')
    if len(month) == 1:
        month_txt = f'0{month}'
    else:
        month_txt = month
    if len(day) == 1:
        day_txt = f'0{day}'
    else:
        day_txt = day

    date = f'20{year}-{month_txt}-{day_txt}'
    month = int(month)
    day = int(day)
    year = 2000 + int(year)

    # now we have the date in a format compatible with the work logs. Let's filter the work logs based on the date
    fildf = fildf[fildf['Date'] == date]

    # if the worklogs file does not have the data for the given date, return None
    if fildf.shape[0] < 1:
        return None, None, None, None, None, None, None

    # get the day and lunch time in ms access format (days since 1899-12-30).
    # This is done because the activPAL data is in this format

    lunch_day_ms_access = get_ms_access_time_v1(int(year), int(month), int(day))
    lunch_time_ms_access = lunch_day_ms_access + lunch_time / 24.0

    # get the work start and end times
    work_start = fildf['Time arrived'].to_numpy()[0]
    work_end = fildf['Time left'].to_numpy()[0]

    if work_start == work_start:
        work_start_hr = int(work_start.split(':')[0])
        work_start_min = int(work_start.split(':')[1])
        work_start_sec = int(work_start.split(':')[2])

        work_start_time = work_start_hr + work_start_min / 60.0 + work_start_sec / 3600.0
        work_start_ms_access_time = get_ms_access_time_v2(int(year), int(month), int(day), work_start_hr,
                                                          work_start_min, work_start_sec)
    else:
        work_start_time = None
        work_start_ms_access_time = None

    if work_end == work_end:
        work_end_hr = int(work_end.split(':')[0])
        work_end_min = int(work_end.split(':')[1])
        work_end_sec = int(work_end.split(':')[2])
        work_end_ms_access_time = get_ms_access_time_v2(int(year), int(month), int(day), work_end_hr, work_end_min,
                                                        work_end_sec)
    else:
        work_end_ms_access_time = None


    activ_pal_folder = f'{basepath}/activpal/{phase_alias[phase]}/{participant}/'

    if work_start_time is None:
        print(f'Work start time is None for {participant}, {phase}, {date}')

    if not os.path.exists(activ_pal_folder):
        return None, None, None, None, None, None, work_start_time
    found = False
    for file in os.listdir(activ_pal_folder):
        if 'EventsEx.csv' in file:
            found = True

            if participant == 'P4' and phase == 'condition2':
                skiprows = 15
            else:
                skiprows = 2
            event_df = pd.read_csv(os.path.join(activ_pal_folder, file), sep=';', skiprows=skiprows,
                                   names=event_colnames, header=None, low_memory=False)
            break
    if not found:
        return None, None, None, None, None, None, work_start_time


    event_df['next_time'] = event_df['Time'].shift(-1)
    event_df['next_time'] = event_df['next_time'].fillna(0)
    #event_df.to_csv('event_df.csv', index=False)

    filevent = event_df[event_df['Waking Day'] == 1]
    filevent = filevent[filevent['next_time'] >= lunch_day_ms_access]  # Starting from the waking time of that day until lunch
    filevent = filevent[filevent['Time'] <= lunch_time_ms_access]
    if filevent.shape[0] < 1:
        return None, None, None, None, None, None, work_start_time

    time_arr = filevent['Time'].to_numpy()
    duration_arr = filevent['Duration (s)'].to_numpy()
    event_type_arr = filevent['Event Type'].to_numpy()

    sitting_total, standing_total, stepping_total = extract_activity_from_arrays(time_arr, duration_arr, event_type_arr,
                                                                     lunch_day_ms_access, lunch_time_ms_access)

    filevent = event_df[event_df['Waking Day'] == 1]
    filevent = filevent[filevent['next_time'] >= work_start_ms_access_time]
    filevent = filevent[filevent['Time'] <= lunch_time_ms_access]
    if filevent.shape[0] < 1:
        return sitting_total, standing_total, stepping_total, None, None, None, work_start_time

    time_arr = filevent['Time'].to_numpy()
    duration_arr = filevent['Duration (s)'].to_numpy()
    event_type_arr = filevent['Event Type'].to_numpy()

    sitting_at_work, standing_at_work, stepping_at_work = extract_activity_from_arrays(time_arr, duration_arr,
                                                                           event_type_arr, work_start_ms_access_time,
                                                                           lunch_time_ms_access)

    return sitting_total, standing_total, stepping_total, sitting_at_work, standing_at_work, stepping_at_work, work_start_time


def extract_activity_from_arrays(time_arr, duration_arr, event_type_arr, start_ms_access_time, end_ms_access):

    sitting = 0
    standing = 0
    stepping = 0
    n = len(time_arr)
    slot_time = 0
    for i in range(n):
        if (time_arr[i] + duration_arr[i]/86400.0 < start_ms_access_time) or (time_arr[i] > end_ms_access):
            continue
        if time_arr[i] < start_ms_access_time:
            if time_arr[i] + duration_arr[i]/86400.0 > end_ms_access:
                slot_time = (end_ms_access - start_ms_access_time)*86400.0
            else:
                slot_time = duration_arr[i] - (start_ms_access_time - time_arr[i])*86400.0
        elif time_arr[i] + duration_arr[i]/86400.0 > end_ms_access:
            slot_time = (end_ms_access - time_arr[i])*86400.0
        else:
            slot_time = duration_arr[i]

        if event_type_arr[i] == 0:
            sitting += slot_time
        elif event_type_arr[i] == 1:
            standing += slot_time
        elif event_type_arr[i] == 2:
            stepping += slot_time

    return sitting, standing, stepping

def add_activPAL(basepath, output_folder):
    """
    Add the activPAL data to the dataset. This will be a CSV with the following additional columns:
    sitting_total, standing_total, stepping_total, sitting_at_work, standing_at_work, stepping_at_work, work_start_time
    The first three are the total duration of three activities before lunch of the day.
    The next three are the total duration of the three activities at work before lunch.
    The last one is the time at which the user started working on that day.

    :param basepath:
    :param output_folder:
    :return:
    """

    # at first read the dataset_v2_with_dummies.csv where we have the previous features and the absolute 3-hr AUC.
    currentdf = pd.read_csv(f'{output_folder}/dataset_v2_with_dummies.csv')

    # extract the users, phases, dates, and lunch times from the dataset
    user_arr = currentdf['user_id'].to_numpy()
    phase_arr = currentdf['phase'].to_numpy()
    days_arr = currentdf['day'].to_numpy()
    lunch_arr = currentdf['lunch_time'].to_numpy()

    # create the new columns for the activPAL data. Data will be populated in these arrays within the next for loop.
    sitting_arr = []
    standing_arr = []
    stepping_arr = []
    sitting_at_work_arr = []
    standing_at_work_arr = []
    stepping_at_work_arr = []
    work_start_time_arr = []

    # iterate over all the users and extract the activPAL data for each entry of the current dataset for the purpose
    # of adding the activPAL data to the dataset.
    for i in range(len(user_arr)):
        # from the current row of the dataset, take the user, phase, date, and lunch_time.
        user = user_arr[i]
        phase = phase_arr[i]
        day = days_arr[i]
        lunch_time = lunch_arr[i]

        # find the correct activPAL data file for the combination and extract the activity before lunch.
        sitting_total, standing_total, stepping_total, sitting_at_work, standing_at_work, stepping_at_work, work_start_time = get_activity_before_lunch(basepath, user, phase, day, lunch_time)

        # append the extracted data to the arrays created before the loop.
        sitting_arr.append(sitting_total)
        standing_arr.append(standing_total)
        stepping_arr.append(stepping_total)
        sitting_at_work_arr.append(sitting_at_work)
        standing_at_work_arr.append(standing_at_work)
        stepping_at_work_arr.append(stepping_at_work)
        work_start_time_arr.append(work_start_time)

    # at this point, the arrays are ready to be appended as pandas columns to the current dataset.

    currentdf['sitting_total'] = sitting_arr
    currentdf['standing_total'] = standing_arr
    currentdf['stepping_total'] = stepping_arr
    currentdf['sitting_at_work'] = sitting_at_work_arr
    currentdf['standing_at_work'] = standing_at_work_arr
    currentdf['stepping_at_work'] = stepping_at_work_arr
    currentdf['work_start_time'] = work_start_time_arr

    # return the modified dataframe so that it can be saved to a CSV file in the calling function (this case, main.py)
    return currentdf


def calculate_glycemic_load(net_carbs, total_fat, total_protein, total_fiber):
    """
    Calculate the glycemic load based on the macronutrients

    :param net_carbs:
    :param total_fat:
    :param total_protein:
    :param total_fiber:
    :return:
    """
    # formula for glycemic load
    # GL = 19.27 + 0.39 * net carbs - 0.21 * total fat - 0.01 * (total protein)^2 - 0.01 * (total fiber)^2

    return 19.27 + 0.39 * net_carbs - 0.21 * total_fat - 0.01 * (total_protein**2) - 0.01 * (total_fiber**2)
def add_glycemic_load(output_folder):
    """
    Add the glycemic load data to the dataset. This will be a CSV with the following additional columns:
    gload

    :param output_folder:
    :return: the modified dataframe with the glycemic load data added
    """
    activ_df = pd.read_csv(f'{output_folder}/dataset_with_activPAL.csv')

    # extract the users, phases, dates, and lunch times from the dataset
    net_carbs_arr = activ_df['Net Carbs(g)'].to_numpy()
    total_fat_arr = activ_df['Total Fat (g)'].to_numpy()
    total_protein_arr = activ_df['Protein (g)'].to_numpy()
    total_fiber_arr = activ_df['Fiber (g)'].to_numpy()

    n = len(net_carbs_arr)
    # output array
    gload_arr = []
    for i in range(n):
        # get the query parameters
        net_carbs = net_carbs_arr[i]
        total_fat = total_fat_arr[i]
        total_protein = total_protein_arr[i]
        total_fiber = total_fiber_arr[i]

        # get the glycemic load for the current user, phase, and day based on the macronutrients
        gload = calculate_glycemic_load(net_carbs, total_fat, total_protein, total_fiber)
        gload_arr.append(gload)

    # add the glycemic load data to the dataframe
    activ_df['glycemic_load'] = gload_arr
    return activ_df

def add_respective_auc(basepath, output_folder, hyper_threshold=140):
    """
    Add the respective AUC values to the dataset. This will be a CSV file with the following additional_column:
    respective_auc

    :param basepath:
    :param output_folder:
    :return: the modified dataframe with the respective AUC values added
    """
    activ_df = pd.read_csv(f'{output_folder}/dataset_with_gload.csv')

    # extract the users, phases, dates, and lunch times from the dataset
    user_arr = activ_df['user_id'].to_numpy()
    phase_arr = activ_df['phase'].to_numpy()
    days_arr = activ_df['day'].to_numpy()
    lunch_time_arr = activ_df['lunch_time'].to_numpy()

    # create the new columns for the respective AUC data. Data will be populated in these arrays within the next for loop.
    respective_auc_arr = []
    abs_auc_arr = []
    max_gluc_arr = []
    hyperglycemia_arr = []

    # iterate over all the users and extract the respective AUC data for each entry of the current dataset for the purpose
    # of adding the respective AUC data to the dataset.
    for i in range(len(user_arr)):
        # from the current row of the dataset, take the user, phase, date, and lunch_time.
        user = user_arr[i]
        phase = phase_arr[i]
        day = days_arr[i]
        lunch_time = lunch_time_arr[i]

        # find the correct AUC data file for the combination and extract the respective AUC.
        abs_auc, res_auc, max_gluc, hyperglycemia = get_absolute_and_respective_auc(basepath, user, phase, day, lunch_time)

        # append the extracted data to the arrays created before the loop.
        abs_auc_arr.append(abs_auc)
        respective_auc_arr.append(res_auc)
        max_gluc_arr.append(max_gluc)
        hyperglycemia_arr.append(hyperglycemia)

    # at this point, the arrays are ready to be appended as pandas columns to the current dataset.

    activ_df['absolute_auc'] = abs_auc_arr
    activ_df['respective_auc'] = respective_auc_arr
    activ_df['max_postprandial_gluc'] = max_gluc_arr
    activ_df[f'postprandial_hyperglycemia_{hyper_threshold}'] = hyperglycemia_arr

    # return the modified dataframe so that it can be saved to a CSV file in the calling function (this case, main.py)
    return activ_df

def get_absolute_and_respective_auc(basepath, participant, phase, date, lunch_time):
    """
    Get the respective AUC for each user

    :return:
    """
    cgm_path = f'{basepath}cgm/{phase}_cleaned/{participant}_{phase}_cleaned_cgm.csv'
    cgm_df = pd.read_csv(cgm_path)
    gluc = cgm_df['corrected_glucose'].to_numpy()
    timestamps = cgm_df['Device Timestamp'].to_numpy()

    lunch_hour = int(lunch_time)
    lunch_min = int((lunch_time - lunch_hour) * 60)
    month, day, year = date.split('/')
    if len(month) == 1:
        month_txt = f'0{month}'
    else:
        month_txt = month
    if len(day) == 1:
        day_txt = f'0{day}'
    else:
        day_txt = day
    corrected_date = f'20{year}-{month_txt}-{day_txt}'
    lunch_timestamp = cgmprocessor.string_to_stamp(corrected_date + ' ' + str(lunch_hour) + ':'+str(lunch_min)+':00')

    timestamps_again = []
    for i in range(len(timestamps)):
        timestamps_again.append(cgmprocessor.string_to_stamp(timestamps[i]))

    absolute_auc = cgmprocessor.calculate_AUC(timestamps_again, gluc, lunch_timestamp, False, 3, False)
    respective_auc = cgmprocessor.calculate_AUC(timestamps_again, gluc, lunch_timestamp, True, 3, False)
    hyper_threshold = 140
    max_gluc = cgmprocessor.calculate_max_postprandial_glucose(timestamps_again, gluc, lunch_timestamp, 3, False)


    if max_gluc is None:
        hyperglycemia = None
    else:
        try:
            hyperglycemia = int(max_gluc >= hyper_threshold)
        except TypeError:
            print(f'TypeError in participant {participant}, phase {phase}, date {date}, max_gluc {max_gluc}, type {type(max_gluc)}')
            hyperglycemia = None

    return absolute_auc, respective_auc, max_gluc, hyperglycemia




def plot_beautiful_auc(basepath, output_folder, user, corrected_date, lunch_time, phase):
    cgm_path = f'{basepath}cgm/{phase}_cleaned/{user}_{phase}_cleaned_cgm.csv'

    cgm_df = pd.read_csv(cgm_path)
    gluc = cgm_df['corrected_glucose'].to_numpy()
    timestamps = cgm_df['Device Timestamp'].to_numpy()
    lunch_hour = int(lunch_time)
    lunch_min = int((lunch_time - lunch_hour) * 60)

    lunch_timestamp = cgmprocessor.string_to_stamp(corrected_date + ' ' + str(lunch_hour) + ':'+str(lunch_min)+':00')
    timestamps_again = []
    for i in range(len(timestamps)):
        timestamps_again.append(cgmprocessor.string_to_stamp(timestamps[i]))

    absolute_auc = cgmprocessor.calculate_AUC(timestamps_again, gluc, lunch_timestamp, False, 3, False)
    print(f'Absolute AUC: {absolute_auc}')
    respective_auc, time_to_consider, gluc_to_consider, time_just_before, time_just_after, gluc_just_before, gluc_just_after, baseline_gluc  = cgmprocessor.calculate_AUC(timestamps_again, gluc, lunch_timestamp, True, 3, True, True)


    gluc_to_consider = [gluc_just_before] + gluc_to_consider + [gluc_just_after]
    print(f'Respective AUC: {respective_auc}')
    print(time_to_consider, gluc_to_consider)

    print(f'Max glucose: {max(gluc_to_consider)}')

    # remove the offset of time.
    time_to_consider = [(time_t - lunch_timestamp)/3600.0 for time_t in time_to_consider]
    time_just_after = (time_just_after - lunch_timestamp)/3600.0
    time_to_consider = [(time_just_before-lunch_timestamp)/3600.0] + time_to_consider + [time_just_after]
    lunch_timestamp = 0 # (lunch_timestamp - time_just_before)/3600.0

    print('After removing the offset')
    print(time_to_consider, gluc_to_consider)
    print(lunch_timestamp)

    # Create the plot
    fig, ax = plt.subplots()

    # Customize line plot
    ax.plot(time_to_consider, gluc_to_consider, label='Continuous blood glucose', color='black', linewidth=3)

    # Global plot settings for thick borders and bold fonts
    plt.rcParams['axes.linewidth'] = 3  # Thicker border
    plt.rcParams["font.weight"] = "bold"  # Bold fonts globally
    plt.rcParams["axes.labelweight"] = "bold"  # Bold axes labels

    # Set axis labels with bold fonts and custom font size
    ax.set_xlabel('Time since lunch (h)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Blood glucose (mg/dL)', fontsize=16, fontweight='bold')

    # Customize ticks
    ax.tick_params(axis='x', width=3, labelsize=16)  # Thicker x-axis ticks
    ax.tick_params(axis='y', width=3, labelsize=16)  # Thicker y-axis ticks

    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # Set y-axis limits
    ax.set_ylim([0, 180])

    # Add vertical and horizontal lines with labels
    ax.axvline(x=lunch_timestamp, color='red', linestyle='--', label='Lunch time', linewidth=2)
    ax.axvline(x=lunch_timestamp + 3, color='green', linestyle='--', label='Lunch + 3hrs', linewidth=2)
    ax.axhline(y=baseline_gluc, color='black', linestyle='--', label='Baseline glucose', linewidth=2)

    # Add legend
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=1)

    # Save the plot and close
    plt.savefig(f'{output_folder}/CGM_plot.png', bbox_inches='tight')  # Save with tight bounding box
    plt.close()


def plot_mlp_results(basepath, output_folder):
    # Data from the table
    categories = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    auc_values = [74, 29, 27, 22, 21, 19, 18, 18, 19, 18, 18, 18, 17]
    maxbgl_values = [50, 24, 24, 21, 21, 19, 18.3, 18.0, 19, 18, 19, 19, 17.5]

    # Divide each value by 100
    auc_values = [x / 100 for x in auc_values]
    maxbgl_values = [x / 100 for x in maxbgl_values]

    # Bar width
    bar_width = 0.25

    # X positions for bars
    x = np.arange(len(categories))

    # Plot bars with patterns
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_ylim(0, 1)  # Set y-axis limits to 0-1 for NRMSE
    ax.bar(x - bar_width, auc_values, width=bar_width, label='AUC NRMSE', hatch='//', facecolor='#1e81b0', edgecolor='black', linewidth=1.5)
    ax.bar(x + bar_width, maxbgl_values, width=bar_width, label='MaxBGL NRMSE', hatch='\\\\', facecolor='#e28743', edgecolor='black', linewidth=1.5)

    # Add labels and title
    ax.set_xlabel('MLP Variation', fontsize=14, fontweight='bold')
    ax.set_ylabel('NRMSE', fontsize=14, fontweight='bold')
    ax.set_title('NRMSE metrics for predictions of AUC and MaxBGL', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12)

    # Customize tick fonts
    ax.tick_params(axis='y', labelsize=14)  # Increase y-axis tick font size
    ax.tick_params(axis='x', labelsize=12)  # Maintain x-axis tick font size

    # Customize borders
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # Set spine thickness

    # Customize gridlines
    #ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_folder}/mlp_results_plot.png", bbox_inches='tight')  # Save with tight bounding box
    plt.close()



