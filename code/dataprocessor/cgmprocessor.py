import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def isnan(val):
    return val!=val
def string_to_stamp(string_):
    """
    This function converts a string to a timestamp. Several formats are supported.
    :param string_: the timestamp in string format
    :return: the timestamp in the format of datetime.datetime.timestamp()
    """
    try:
        if '/' in string_:
            if 'm' in string_.lower():
                return datetime.datetime.strptime(string_, '%m/%d/%y %I:%M:%S %p').timestamp()
            else:
                second_present = len(string_.split(':')) > 2
                if second_present:
                    return datetime.datetime.strptime(string_, '%m/%d/%y %H:%M:%S').timestamp()
                else:
                    return datetime.datetime.strptime(string_, '%m/%d/%Y %H:%M').timestamp()
        elif '-' in string_:
            if 'm' in string_.lower():
                return datetime.datetime.strptime(string_, '%m-%d-%Y %I:%M:%S %p').timestamp()
            else:
                second_present = len(string_.split(':')) > 2
                if second_present:
                    return datetime.datetime.strptime(string_, '%Y-%m-%d %H:%M:%S').timestamp()
                else:
                    return datetime.datetime.strptime(string_, '%m-%d-%Y %H:%M').timestamp()
    except ValueError:
        print('Value error for:', string_)
        return np.nan


def area_from_array(arr, dt, starttime, endtime, firsttime, lasttime, baseline_gluc, time_just_before, time_just_after,
                    gluc_just_before, gluc_just_after, respective=True):
    """
    This function calculates the area under the curve for a given array of values.
    :param arr: the array of values (e.g. CGM readings)
    :param dt: time difference between two consecutive values (e.g. for the CGM readings of our dataset, this is 15 minutes)
    :param starttime: the time when our area curve starts (e.g. the time of the meal)
    :param endtime: the time when our area curve stops (e.g. the time of the meal + 1 hour or 2 hours, etc.)
    :param firsttime: the time of the first reading in the array
    :param lasttime: the time of the last reading in the array
    :return: the area under the curve according to the trapezoidal rule
    """
    if len(arr) == 0:
        return 0
    sum_ = 0
    for i in range(len(arr) - 1):
        h1 = arr[i]
        h2 = arr[i + 1]
        sum_ = sum_ + dt * (h1 + h2) / 2

    #calculate starttime_gluc and endtime_gluc with linear interpolation
    if firsttime == time_just_before:
        starttime_gluc = arr[0]
    else:
        starttime_gluc = gluc_just_before + (arr[0]-gluc_just_before) * (starttime - time_just_before)/(firsttime - time_just_before)

    if lasttime == time_just_after:
        endtime_gluc = arr[-1]
    else:
        try:
            endtime_gluc = gluc_just_after + (arr[-1]-gluc_just_after) * (time_just_after - endtime)/(time_just_after - lasttime)
            #print(f'NoError: endtime:{endtime}, lasttime:{lasttime}, timejustbefore:{time_just_before}, glucjustafter:{gluc_just_after}, arr[-1]:{arr[-1]}')
        except TypeError:
            print(f'TypeError: endtime:{endtime}, lasttime:{lasttime}, timejustbefore:{time_just_before}, glucjustafter:{gluc_just_after}, arr[-1]:{arr[-1]}')
            endtime_gluc = arr[-1]


    # residue is the area of two small trapezoids at the beginning and end of the curve. This is necessary because of
    # a precise calculation. As the lunch time may not necessarily coincide with the time of the first reading.

    residue = (firsttime - starttime) / 3600 * (starttime_gluc + arr[0])/2.0 + (endtime - lasttime) / 3600 * (endtime_gluc + arr[-1])/2.0
    if respective:
        return sum_ + residue - baseline_gluc * (endtime - starttime) / 3600
    else:
        return sum_ + residue

def max_from_array(arr, starttime, endtime, firsttime, lasttime, time_just_before, time_just_after, gluc_just_before, gluc_just_after):
    if len(arr) == 0:
        return None

    # calculate starttime_gluc and endtime_gluc with linear interpolation
    if firsttime == time_just_before:
        starttime_gluc = arr[0]
    else:
        starttime_gluc = gluc_just_before + (arr[0]-gluc_just_before) * (starttime - time_just_before)/(firsttime - time_just_before)

    if lasttime == time_just_after:
        endtime_gluc = arr[-1]
    else:
        try:
            endtime_gluc = gluc_just_after + (arr[-1]-gluc_just_after) * (time_just_after - endtime)/(time_just_after - lasttime)

        except TypeError:
            print(f'TypeError: endtime:{endtime}, lasttime:{lasttime}, timejustbefore:{time_just_before}, glucjustafter:{gluc_just_after}, arr[-1]:{arr[-1]}')
            endtime_gluc = arr[-1]

    max_ = max(arr)
    return max(max_, starttime_gluc, endtime_gluc)


def calculate_AUC(timestamps, gluc, lunchtimestamp, respective=True, hours=2, return_arrays=False, return_others=False):
    """
    This function calculates the postprandial area under the curve from CGM readings.

    To find the AUC with respect to the CGM readings just before the meal, we can just subtract the area under the
    horizontal line from the actual area under the curve.


    :param timestamps: the timestamps of the CGM readings
    :param gluc: The CGM readings
    :param lunchtimestamp: Timestamp when lunch started
    :return:
    """
    if lunchtimestamp == '--missing--':
        return None
    time_to_consider = []
    gluc_to_consider = []

    time_just_before = None
    gluc_just_before = None
    time_just_after = None
    gluc_just_after = None
    baseline_gluc = None

    scan_status = 0 # 0: before lunch, 1: during or after lunch

    for i in range(len(timestamps)):
        if timestamps[i] >= lunchtimestamp and timestamps[i] <= lunchtimestamp + 3600*hours:
            time_to_consider.append(timestamps[i])
            gluc_to_consider.append(gluc[i])
            if scan_status == 0:
                time_just_before = timestamps[max(i-1, 0)]
                gluc_just_before = gluc[max(i-1, 0)]
                scan_status = 1

                #Now let's calculate the baseline glucose. We will take the average of the two readings just before lunch.
                basepoint1 = gluc[max(i-1, 0)]
                basepoint2 = gluc[max(i-2, 0)]
                baseline_gluc = (basepoint1 + basepoint2) / 2.0

        else:
            if scan_status == 1:
                time_just_after = timestamps[i]
                gluc_just_after = gluc[i]
                break # no need to scan the rest of the array.



    if len(time_to_consider) == 0 or time_just_after is None or time_just_before is None or gluc_just_after is None or gluc_just_before is None or baseline_gluc is None:
        if return_arrays:
            return None, [], []
        else:
            return None
    else:
        auc = area_from_array(gluc_to_consider, 15 / 60.0, lunchtimestamp, lunchtimestamp + 3600*hours, time_to_consider[0],
                              time_to_consider[-1], baseline_gluc, time_just_before, time_just_after, gluc_just_before,
                              gluc_just_after,  respective)
        if return_arrays:
            if return_others:
                return auc, time_to_consider, gluc_to_consider, time_just_before, time_just_after, gluc_just_before, gluc_just_after, baseline_gluc
            else:
                return auc, time_to_consider, gluc_to_consider
        else:
            return auc

def calculate_max_postprandial_glucose(timestamps, gluc, lunchtimestamp, hours=3, return_arrays=False):
    """
    This function calculates the maximum postprandial glucose for a given lunch time.
    :return:
    """
    if lunchtimestamp == '--missing--':
        return None
    time_to_consider = []
    gluc_to_consider = []

    time_just_before = None
    gluc_just_before = None
    time_just_after = None
    gluc_just_after = None
    baseline_gluc = None

    scan_status = 0  # 0: before lunch, 1: during or after lunch

    for i in range(len(timestamps)):
        if timestamps[i] >= lunchtimestamp and timestamps[i] <= lunchtimestamp + 3600 * hours:
            time_to_consider.append(timestamps[i])
            gluc_to_consider.append(gluc[i])
            if scan_status == 0:
                time_just_before = timestamps[min(i - 1, 0)]
                gluc_just_before = gluc[min(i - 1, 0)]
                scan_status = 1

                # Now let's calculate the baseline glucose. We will take the average of the two readings just before lunch.
                basepoint1 = gluc[min(i - 1, 0)]
                basepoint2 = gluc[min(i - 2, 0)]
                baseline_gluc = (basepoint1 + basepoint2) / 2.0

        else:
            if scan_status == 1:
                time_just_after = timestamps[i]
                gluc_just_after = gluc[i]
                break  # no need to scan the rest of the array.

    if len(time_to_consider) == 0 or time_just_after is None or time_just_before is None or gluc_just_after is None or gluc_just_before is None or baseline_gluc is None:
        if return_arrays:
            return None, [], []
        else:
            return None
    else:
        max_gluc = max_from_array(gluc_to_consider, lunchtimestamp, lunchtimestamp + 3600 * hours,
                              time_to_consider[0], time_to_consider[-1], time_just_before, time_just_after,
                                  gluc_just_before, gluc_just_after)
        if return_arrays:
            return max_gluc, time_to_consider, gluc_to_consider
        else:
            return max_gluc

def plot_AUC(timestamps, gluc, lunchtimestamp, participant, phase, hours=2):
    """
    This function plots the CGM readings and calculates the AUC for a given lunch time.

    :param timestamps:
    :param gluc:
    :param lunchtimestamp:
    :param participant:
    :param phase:
    :param hours:
    :return:
    """

    if lunchtimestamp == '--missing--':
        return ' '

    auc, time_to_consider, gluc_to_consider = calculate_AUC(timestamps, gluc, lunchtimestamp, True, hours, return_arrays=True)
    time_to_consider = [str(datetime.datetime.fromtimestamp(stamp)) for stamp in time_to_consider]

    plt.figure()
    plt.plot(time_to_consider, gluc_to_consider)
    plt.xlabel('Timestamp')
    plt.xticks(rotation=90, ha='right')
    plt.ylabel('Glucose')
    title = f'Lunch time_ {lunchtimestamp}, participant_ {participant}, phase_{phase}, auc_{auc}'
    plt.title(title)
    plt.savefig('data/'+title+'.png')
    return auc

def process_all_one_participant(basepath, participant, phase, lunchdf):
    filename = f'{basepath}cgm/{phase}_cleaned/{participant}_{phase}_cleaned_cgm.csv'
    if not os.path.exists(filename):
        print(f'File {filename} does not exist')
        return ''
    glucdf = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{participant}_{phase}_cleaned_cgm.csv')
    gluc = glucdf.corrected_glucose.to_numpy()
    timestamps = glucdf['Device Timestamp'].to_numpy()

    timestamps_again = []
    for i in range(len(timestamps)):
        timestamps_again.append(string_to_stamp(timestamps[i]))


    lunchfiltered = lunchdf[lunchdf.PID == participant]
    if lunchfiltered.shape[0] < 1:
        print(f'Lunch times missing for user {participant}, phase {phase}')
        return ''

    lunchdays = lunchfiltered.Date.to_numpy()
    lunchtimes = lunchfiltered.Time.to_numpy().astype(str)
    lunchdayseqs = lunchfiltered['Day #'].to_numpy()
    lunchdays = [day.split('T')[0] for day in lunchdays]

    lunchtimestamps = []

    for i in range(len(lunchdays)):
        if lunchtimes[i] == 'nan':
            lunchtimestamps.append('--missing--')
            continue
        timestamp = string_to_stamp(lunchdays[i] + ' ' + lunchtimes[i])
        lunchtimestamps.append(timestamp)

    txt2 = ''
    for i in range(len(lunchdays)):
        auc = plot_AUC(timestamps_again, gluc, lunchtimestamps[i], participant, phase)
        txt2 = txt2+ f'{participant},{lunchdayseqs[i]},{lunchdays[i]},{auc}\n'
    return txt2


def plot_all_one_participant(basepath, participant, phase, lunchdf):
    """
    This function plots the AUC for each lunch time for a given participant.
    :param basepath:
    :param participant:
    :param phase:
    :param lunchdf:
    :return:
    """
    glucdf = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{participant}_{phase}_cleaned_cgm.csv')
    gluc = glucdf.corrected_glucose.to_numpy()
    timestamps = glucdf['Device Timestamp'].to_numpy()

    timestamps_again = []
    for i in range(len(timestamps)):
        timestamps_again.append(string_to_stamp(timestamps[i]))

    lunchfiltered = lunchdf[lunchdf.PID == participant]
    if lunchfiltered.shape[0] < 1:
        print(f'Lunch times missing for user {participant}, phase {phase}')
        return ''
    lunchdays = lunchfiltered.Date.to_numpy()
    lunchtimes = lunchfiltered.Time.to_numpy().astype(str)
    lunchdayseqs = lunchfiltered['Day #'].to_numpy()
    lunchdays = [day.split('T')[0] for day in lunchdays]

    lunchtimestamps = []
    for i in range(len(lunchdays)):
        if lunchtimes[i] == 'nan':
            lunchtimestamps.append('--missing--')
            continue
        timestamp = string_to_stamp(lunchdays[i] + ' ' + lunchtimes[i])
        lunchtimestamps.append(timestamp)

    txt2 = ''
    for i in range(len(lunchdays)):
        auc = plot_AUC(timestamps_again, gluc, lunchtimestamps[i], participant, phase)
        txt2 = txt2+ f'{participant},{lunchdayseqs[i]},{lunchdays[i]},{auc}\n'
    return txt2

def process_all_timeinrange_participant(basepath, participant, phase):
    """
    This function calculates the time in range for a given participant.
    :param basepath:
    :param participant:
    :param phase:
    :return:
    """
    glucdf = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{participant}_{phase}_cleaned_cgm.csv')
    gluc = glucdf.corrected_glucose.to_numpy()
    txt2 = ''
    days = glucdf['day'].unique()
    count = 0
    for day in days:

        dailycount  =0
        fildf = glucdf[glucdf['day'] == day]
        filgluc = fildf.corrected_glucose.to_numpy()
        for g in filgluc:
            if g >= 60 and g <= 160:
                count+=1
                dailycount +=1
        daily_percentage = dailycount/len(filgluc)*100.0
        txt2 = txt2+ f'{participant},{phase},{day},{daily_percentage}\n'
    total_percentage = count/len(gluc)*100.0
    txt2 = txt2 + f'{participant},{phase},AVERAGE,{total_percentage}\n'
    return txt2


def process_all_auc(basepath, phase, allusers):
    """
    This function calculates the postprandial AUC for each participant and saves to a CSV file.
    :param basepath:
    :param phase:
    :param allusers:
    :return:
    """
    lunchdf = pd.read_csv(f'{basepath}/all_lunch_times_{phase}.csv')
    txt = 'PID,Day,Date,Postprandial_AUC\n'
    for participant in allusers:
        txt = txt + process_all_one_participant(basepath, participant, phase, lunchdf)
    print(txt)
    with open(f'{basepath}allauc_{phase}.csv', 'w') as file:
        file.write(txt)



def process_all_time_in_range(basepath, phase, allusers):
    """
    This function calculates the time in range for each participant and saves to a CSV file.
    :param basepath:
    :param phase:
    :param allusers:
    :return:
    """
    txt = 'PID,Phase,Date,Time_in_Range\n'
    for participant in allusers:
        txt = txt + process_all_timeinrange_participant(basepath, participant, phase)
    print(txt)
    with open(f'{basepath}timeinrange_{phase}.csv', 'w') as file:
        file.write(txt)

def plot_all_auc(basepath, phase, allusers):
    """
    This function plots the postprandial AUC for each participant and saves to a CSV file.
    :param basepath:
    :param phase:
    :param allusers:
    :return:
    """
    lunchdf = pd.read_csv(f'{basepath}/all_lunch_times_{phase}.csv')
    txt = 'PID,Day,Date,Postprandial_AUC\n'
    for participant in allusers:
        txt = txt + plot_all_one_participant(basepath, participant, phase, lunchdf)
    print(txt)
    #with open(f'{basepath}allauc_{phase}.csv', 'w') as file:
    #    file.write(txt)


def plot_all_cgm(basepath, phase, allusers):
    """
    This function plots the CGM readings for each participant and saves to a PNG file.
    :param basepath:
    :param phase:
    :param allusers:
    :return:
    """
    for user in allusers:
        glucdf = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{user}_{phase}_cleaned_cgm.csv')
        all_days = glucdf['day'].unique()
        for day in all_days:
            fildf = glucdf[glucdf['day'] == day]
            gluc = fildf['corrected_glucose'].to_numpy()
            hours_raw = fildf['hour'].to_numpy()
            hours = [int(h) for h in hours_raw]
            minutes = [int((h - math.floor(h)) * 60) for h in hours_raw]
            timetxt = [f"{str(h).rjust(2, '0')}:{str(m).rjust(2, '0')}" for h, m in zip(hours, minutes)]
            fig, ax = plt.subplots(figsize=(20, 10))
            #ax.set_xlim(40, 200)
            ax.set_ylim(40, 200)
            ax.yaxis.set_major_locator(plt.MultipleLocator(5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
            ax.plot(timetxt, gluc)
            ax.grid(which='major', axis='y', linestyle='-', linewidth='0.5', color='red')
            ax.grid(which='minor', axis='y', linestyle=':', linewidth='0.5', color='black')
            ax.grid(True, axis='x')

            ax.set_xlabel('Time')

            ax.set_xticks(np.arange(0, len(timetxt)))
            ax.set_xticklabels(timetxt, rotation=90, ha='center')
            ax.set_ylabel('Glucose')
            title = f'Participant_ {user}, phase_{phase}, date_{day}'.replace('/', '_')
            ax.set_title(title)
            plt.savefig(f'{basepath}/fasting_glucose/{title}.png')


def day_by_day_avg_for_each_phase_and_day(basepath, phase, allusers):
    """
    This function calculates the average glucose for each day for each participant and saves to a CSV file.
    :param basepath:
    :param phase:
    :param allusers:
    :return:
    """
    txt = 'PID,Phase,Day,Average Glucose,Standard Dev\n'
    for user in allusers:
        glucdf = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{user}_{phase}_cleaned_cgm.csv')
        all_days = glucdf['day'].unique()
        for day in all_days:
            fildf = glucdf[glucdf['day'] == day]
            gluc = fildf['corrected_glucose'].to_numpy()

            # Skip days with less than 72 readings or 18 hours
            #if len(gluc) < 92:
            #    continue
            avg = np.mean(gluc)
            std = np.std(gluc)
            txt = txt + f'{user},{phase},{day},{avg},{std}\n'
    with open(f'{basepath}day_by_day_avg_{phase}_no_restriction.csv', 'w') as file:
        file.write(txt)

def daily_avg_for_each_phase(basepath, all_phases, allusers):
    """
    This function calculates the daily average glucose for each participant and saves to a CSV file.
    :param basepath:
    :param all_phases:
    :param allusers:
    :return:
    """
    txt = 'PID,Phase,Total days,Phase Average,Phase Standard Dev,Avg of daily avg,Std of daily avg\n'
    for user in allusers:
        for phase in all_phases:
            disqualified_count = 0
            allgluc = np.array([]).astype(int)
            glucdf = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{user}_{phase}_cleaned_cgm.csv')
            all_days = glucdf['day'].unique()
            daily_avgs = []
            for day in all_days:
                fildf = glucdf[glucdf['day'] == day]
                gluc = fildf['corrected_glucose'].to_numpy()
                if len(gluc) < 92:
                    disqualified_count += 1
                    continue
                allgluc = np.concatenate((allgluc, gluc))
                daily_avgs.append(np.mean(gluc))


            total_days = len(all_days) - disqualified_count
            avg = np.mean(allgluc)
            std = np.std(allgluc)
            avgavg = np.mean(daily_avgs)
            avgstd = np.std(daily_avgs)
            txt = txt + f'{user},{phase},{total_days},{avg},{std},{avgavg},{avgstd}\n'
    with open(f'{basepath}daily_avg_all_phases.csv', 'w') as file:
        file.write(txt)


def daily_glucose_at7_930_11(basepath, all_phases, allusers):
    """
    This function calculates the glucose readings at 7 AM, 9:30 AM, and 11 AM for each participant and saves to a CSV file.
    :param basepath:
    :param all_phases:
    :param allusers:
    :return:
    """
    txt = 'PID,Phase,Day,Daily_avg,Daily_std,7am,9am,9:30am,11am,Total data available for the day (hrs)\n'
    for user in allusers:
        for phase in all_phases:
            glucdf = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{user}_{phase}_cleaned_cgm.csv')
            all_days = glucdf['day'].unique()
            for day in all_days:
                fildf = glucdf[glucdf['day'] == day]
                gluc = fildf['corrected_glucose'].to_numpy()
                fildf = fildf[fildf['hour'] >= 7]
                fildf = fildf[fildf['hour'] <= 11.25]
                fil7df = fildf[fildf['hour'] <= 7.25]
                fil9df = fildf[fildf['hour'] >= 9]
                fil9df = fil9df[fil9df['hour'] <= 9.25]
                fil930df = fildf[fildf['hour'] >= 9.5]
                fil930df = fil930df[fil930df['hour'] <= 9.75]
                fil11df = fildf[fildf['hour'] >= 11]

                gluc7 = fil7df['corrected_glucose'].to_numpy()
                gluc9 = fil9df['corrected_glucose'].to_numpy()
                gluc930 = fil930df['corrected_glucose'].to_numpy()
                gluc11 = fil11df['corrected_glucose'].to_numpy()

                if len(gluc7) == 0 or len(gluc9)==0 or len(gluc930) == 0 or len(gluc11) == 0:
                    continue

                avg = np.mean(gluc)
                std = np.std(gluc)
                duration = len(gluc)/4
                txt += f'{user},{phase},{day},{avg},{std},{gluc7[0]},{gluc9[0]},{gluc930[0]},{gluc11[0]},{duration}\n'

    with open(f'{basepath}daily_avg_and_glucose_at7_930_11.csv', 'w') as file:
        file.write(txt)


def phase_avg_glucose_at7_930_11(basepath, all_phases, allusers):
    """
    This function calculates the average glucose readings at 7 AM, 9:30 AM, and 11 AM for each phase and saves to a CSV file.
    :param basepath:
    :param all_phases:
    :param allusers:
    :return:
    """
    txt = 'PID,Phase,Total days,Daily_avg,Daily_std,7am avg,9am avg,9:30am avg,11am avg,Avg daily available data (hrs),7am std,9am std,9:30am std,11am std,Available (hrs) std\n'
    for user in allusers:
        for phase in all_phases:
            glucdf = pd.read_csv(f'{basepath}cgm/{phase}_cleaned/{user}_{phase}_cleaned_cgm.csv')
            all_days = glucdf['day'].unique()
            gluc7_all = np.array([]).astype(int)
            gluc9_all = np.array([]).astype(int)
            gluc930_all = np.array([]).astype(int)
            gluc11_all = np.array([]).astype(int)
            gluc_all = np.array([]).astype(int)
            duration_all = np.array([]).astype(int)
            disqualified_count = 0
            for day in all_days:
                fildf = glucdf[glucdf['day'] == day]
                gluc = fildf['corrected_glucose'].to_numpy()
                fildf = fildf[fildf['hour'] >= 7]
                fildf = fildf[fildf['hour'] <= 11.25]
                fil7df = fildf[fildf['hour'] <= 7.25]
                fil9df = fildf[fildf['hour'] >= 9]
                fil9df = fil9df[fil9df['hour'] <= 9.25]
                fil930df = fildf[fildf['hour'] >= 9.5]
                fil930df = fil930df[fil930df['hour'] <= 9.75]
                fil11df = fildf[fildf['hour'] >= 11]

                gluc7 = fil7df['corrected_glucose'].to_numpy()
                gluc9 = fil9df['corrected_glucose'].to_numpy()
                gluc930 = fil930df['corrected_glucose'].to_numpy()
                gluc11 = fil11df['corrected_glucose'].to_numpy()

                if len(gluc7) == 0 or len(gluc9)==0 or len(gluc930) == 0 or len(gluc11) == 0:
                    disqualified_count+=1
                    continue

                gluc_all = np.concatenate((gluc_all, gluc))
                gluc7_all = np.concatenate((gluc7_all, gluc7[0:1]))
                gluc9_all = np.concatenate((gluc9_all, gluc9[0:1]))
                gluc930_all = np.concatenate((gluc930_all, gluc930[0:1]))
                gluc11_all = np.concatenate((gluc11_all, gluc11[0:1]))
                duration_all = np.concatenate((duration_all, np.array([len(gluc)/4])))

            avg = np.mean(gluc_all)
            std = np.std(gluc_all)
            duration_avg = np.mean(duration_all)
            duration_std = np.std(duration_all)
            gluc7_avg = np.mean(gluc7_all)
            gluc9_avg = np.mean(gluc9_all)
            gluc930_avg = np.mean(gluc930_all)
            gluc11_avg = np.mean(gluc11_all)
            gluc7_std = np.std(gluc7_all)
            gluc9_std = np.std(gluc9_all)
            gluc930_std = np.std(gluc930_all)
            gluc11_std = np.std(gluc11_all)
            total_days = len(all_days) - disqualified_count

            txt += f'{user},{phase},{total_days},{avg},{std},{gluc7_avg},{gluc9_avg},{gluc930_avg},{gluc11_avg},{duration_avg},{gluc7_std},{gluc9_std},{gluc930_std},{gluc11_std},{duration_std}\n'

    with open(f'{basepath}each_phase_daily_avg_and_glucose_at7_9_930_11.csv', 'w') as file:
        file.write(txt)