#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import os
import glob

# Here the script refers to a folder where a csv file is expected to be located.

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

# The script searches in the folder for any csv files and save them as a variable f.

for f in csv_files:
    file_path = f

df_raw = pd.read_csv(f, sep=';')
vehicle_raw = df_raw.columns[2][:11]
vehicle_raw = vehicle_raw.replace('.', '')
date_raw = df_raw.columns[0]
vehicle_short = (vehicle_raw[0] + '' + vehicle_raw[-3:]).replace('.', '')
plt.rcParams["figure.figsize"] = (20, 10)


# The function below is used for preprocessing of the raw csv dataset.

def table(df, vehicle, veh, spont):
    df[[vehicle, 'STATUS_0']] = df[veh].str.split('.', n=1, expand=True)
    df.drop([veh, spont, vehicle], inplace=True, axis=1)
    df.columns = ['date', 'time', 'value', 'unit']
    df = df[['date', 'time', 'unit', 'value']]
    stat_zero_df = df.query('unit == "STATUS_0"')
    stat_zero_df = stat_zero_df.rename(columns={'value': 'status_0'})
    stat_zero_df.drop('unit', inplace=True, axis=1)
    stat_zero_df = stat_zero_df.reset_index(drop=True)
    hw_df = df.query('unit == "HW"')
    hw_df = hw_df.rename(columns={'value': 'hw'})
    hw_df.drop('unit', inplace=True, axis=1)
    hw_df = hw_df.reset_index(drop=True)
    rw_df = df.query('unit == "RW"')
    rw_df = rw_df.rename(columns={'value': 'rw'})
    rw_df.drop('unit', inplace=True, axis=1)
    rw_df = rw_df.reset_index(drop=True)
    speed_df = df.query('unit == "SPEED"')
    speed_df = speed_df.rename(columns={'value': 'speed'})
    speed_df.drop('unit', inplace=True, axis=1)
    speed_df = speed_df.reset_index(drop=True)
    stat1_df = df.query('unit == "STATUS_1"')
    stat1_df = stat1_df.rename(columns={'value': 'status_1'})
    stat1_df.drop('unit', inplace=True, axis=1)
    stat1_df = stat1_df.reset_index(drop=True)
    height_df = df.query('unit == "HSML"')
    height_df = height_df.rename(columns={'value': 'height'})
    height_df.drop('unit', inplace=True, axis=1)
    height_df = height_df.reset_index(drop=True)
    stat_zero_df['y_coor'] = hw_df['hw']
    stat_zero_df['x_coor'] = rw_df['rw']
    stat_zero_df['speed'] = speed_df['speed']
    stat_zero_df['status_1'] = stat1_df['status_1']
    stat_zero_df['height'] = height_df['height']
    df = stat_zero_df
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
    df['time'] = df['time'].dt.strftime('%H:%M:%S')
    df = df[['date', 'time', 'y_coor', 'x_coor', 'speed', 'height', 'status_0', 'status_1']]
    df = df.dropna()
    df['y_coor'] = df['y_coor'].astype(int)
    df['x_coor'] = df['x_coor'].astype(int)
    df['height'] = df['height'].astype(int)
    df['status_0'] = df['status_0'].astype(int)
    return df


df_clean = table(df_raw, df_raw.columns[2][:11], df_raw.columns[2], df_raw.columns[4])


# This function plots a graph with movements of a dump truck during the day.

def movements_plot(df):
    df_plot = df.plot(x='x_coor', y='y_coor', kind='scatter', s=250, figsize=(15, 9), alpha=0.01, fontsize=12,
                      grid=True)
    plt.title('Movements of {} on {}'.format(dump_truck, date), fontsize=20)
    df_plot.set_xlabel('X', fontsize=18)
    df_plot.set_ylabel('Y', fontsize=18)
    return plt.savefig('{}_movements_plot.png'.format(vehicle_short), facecolor='white', bbox_inches='tight')


dump_truck = vehicle_raw
date = date_raw
# movements_plot(df_clean)

df_zero_speed = df_clean.query('speed == 0 and (status_0 == 1 or status_0 == 5)')


# Here we get rid of the data that connect with the workshop places. We also exclude information during breaks
# and shift changes.

def drop_coor(df):
    workshop = df[(df['y_coor'] >= 5267413) &
                  (df['y_coor'] <= 5267610) &
                  (df['x_coor'] >= -108510) &
                  (df['x_coor'] <= -108251)].index

    df_without_workshop = df.drop(workshop)

    time = df_without_workshop['time']

    breaks = df_without_workshop[(time >= '00:09:30') &
                                 (time <= '00:10:00') &
                                 (time >= '00:17:30') &
                                 (time <= '00:18:00') &
                                 (time >= '00:01:30') &
                                 (time <= '00:03:00')].index

    df_without_breaks = df_without_workshop.drop(breaks)

    shifts = df_without_breaks[(time >= '00:05:45') &
                               (time <= '00:06:00') &
                               (time >= '00:13:45') &
                               (time <= '00:14:00') &
                               (time >= '00:21:45') &
                               (time <= '00:22:00')].index

    df_pure = df_without_breaks.drop(shifts)
    df_pure.reset_index(drop=True, inplace=True)
    df_pure['time'] = pd.to_datetime(df_pure['time'], format='%H:%M:%S')

    return df_pure


pure_df = drop_coor(df_zero_speed)

# Here we check if there is still any information in the dataset after clearing in the previous function.

if pure_df.shape[0] == 0:
    movements_plot(df_clean)

    with open("{}_idling_result.txt".format(vehicle_short), "a") as f:
        print('Looks like {} did not make any move on {}, so no further analysis can be made.'.format(vehicle_raw,
                                                                                                    date_raw), file=f)

else:
    movements_plot(df_clean)

    def group_stat1(df):
        status0_1 = df.query('status_0 == 1')
        status0_1 = status0_1.groupby('y_coor').agg(lambda x: list(x))
        status0_1 = status0_1.drop(['date', 'speed', 'height', 'status_1'], axis=1)
        status0_1.reset_index(inplace=True)
        status0_1 = status0_1[['y_coor', 'x_coor', 'time', 'status_0']]

        # here we group by Y coordinate and take the mean value of X coordinate

        status0_1['x_coor'] = status0_1['x_coor'].apply(np.mean).astype(int)
        return status0_1


    status0_1 = group_stat1(pure_df)


    # here we make a sum of all stop entries with status_0 = 1

    def stops_counter1(row):
        ones = 0
        for value in row:
            if value == 1:
                ones += 1
        return ones


    status0_1['status0_1'] = status0_1['status_0'].apply(stops_counter1)
    status0_1 = status0_1.drop(['status_0'], axis=1)


    def group_stat5(df):
        status0_5 = df.query('status_0 == 5')
        status0_5 = status0_5.groupby('y_coor').agg(lambda x: list(x))
        status0_5 = status0_5.drop(['date', 'speed', 'height', 'status_1'], axis=1)
        status0_5.reset_index(inplace=True)
        status0_5 = status0_5[['y_coor', 'x_coor', 'time', 'status_0']]

        # here we group by Y coordinate and take mean value of X coordinate

        status0_5['x_coor'] = status0_5['x_coor'].apply(np.mean).astype(int)
        return status0_5


    status0_5 = group_stat5(pure_df)


    # here we make a sum of all stop entries with status_0 = 5

    def stops_counter5(row):
        fives = 0
        for value in row:
            if value == 5:
                fives += 1
        return fives


    status0_5['status0_5'] = status0_5['status_0'].apply(stops_counter5)
    status0_5 = status0_5.drop(['status_0'], axis=1)


    # here we concatinate two tables with status 1 and 5 into one.
    def concat(df1, df5):
        joined_df = df1.append(df5)
        joined_df = joined_df.fillna(0)
        joined_df['status0_1'] = joined_df['status0_1'].astype(int)
        joined_df['status0_5'] = joined_df['status0_5'].astype(int)
        joined_df['total'] = joined_df['status0_1'] + joined_df['status0_5']
        return joined_df


    joined_df = concat(status0_1, status0_5)


    # here we group by coordinte Y, take mean X coordinate and make a total sum of all entries stops
    # (with status 1 and 5). We will exclude idling time from this step.

    def total_stops_counter(df):
        stops_counter = df.drop(['time'], axis=1)
        stops_counter = stops_counter.groupby('y_coor').agg(
            {'x_coor': 'mean', 'status0_1': 'sum', 'status0_5': 'sum', 'total': 'sum'})
        stops_counter['x_coor'] = stops_counter['x_coor'].astype(int)
        stops_counter = stops_counter.reset_index()
        return stops_counter


    stops_counter = total_stops_counter(joined_df)


    def stops_plot(df):
        df = df.set_index(['y_coor', 'x_coor'])
        df_sorted = df.sort_values(by='total', ascending=False).head(10)
        df_sorted.drop('total', inplace=True, axis=1)
        new_df_plot = df_sorted.plot(kind='bar',
                                     figsize=(12, 6),
                                     grid=True, fontsize=10)
        plt.title('Most frequent stop locations for {} on {} according to statuses'.format(dump_truck, date),
                  fontsize=18)
        new_df_plot.set_xlabel('Y and X coordinates', fontsize=16)
        new_df_plot.set_ylabel('Number of stops', fontsize=16)
        new_df_plot.set_xticklabels(new_df_plot.get_xticklabels(), rotation=30)
        plt.legend(prop={'size': 15})
        return plt.savefig('{}_stops_plot.png'.format(vehicle_short), facecolor='white', bbox_inches='tight')


    dump_truck = vehicle_raw
    date = date_raw
    stops_plot(stops_counter)

    joined_df['time'] = joined_df['time'].apply(np.diff)


    # here we calculate a sum of all idling entries regarding every unique pair of coordinates.

    # this function checks only those timedelta idling entries which equal to 2 seconds in a value list regarding every
    # unique pair of coordinates. Then it saves all found idling entries into a new list to a column ['idling']
    # regarding every unique pair of coordinates.

    def total_idle(row):
        idle = []
        d1 = timedelta(days=0, hours=0, minutes=0, seconds=2)
        for value in row:
            if value == d1:
                idle.append(value)
        return idle


    joined_df['idling'] = joined_df['time'].apply(total_idle)
    joined_df['idling'] = joined_df['idling'].apply(np.sum)
    joined_df = joined_df.drop(['time'], axis=1)
    joined_df = joined_df[['y_coor', 'x_coor', 'idling', 'status0_1', 'status0_5', 'total']]
    joined_df.drop(joined_df[joined_df['idling'] == 0].index, inplace=True)

    # now we sort our dataframe regarding y_coor in ascending

    joined_df = joined_df.sort_values(by='y_coor')
    joined_df.reset_index(drop=True, inplace=True)

    # here divide current dataset into 2, regarding status_0 in order to properly calculate idling time for
    # locations with status0_1 and status0_5

    # status_0 = 1

    idle_stat1 = joined_df.query('status0_5 == 0')
    idle_stat1 = idle_stat1.drop(['status0_5', 'total'], axis=1)
    idle_stat1.reset_index(drop=True, inplace=True)

    # Here we check if there are any entries in the table with when dump truck was full and idled

    if idle_stat1.shape[0] == 0:
        # here we divide current dataset into 2 regarding status_0 in order to properly calculate idling time for
        # locations only with status0_5. If this condition works, the dataframe consists only of idling time
        # with status 0 = 5

        # status_0 = 5

        idle_stat5 = joined_df.query('status0_1 == 0')
        idle_stat5 = idle_stat5.drop(['status0_1', 'total'], axis=1)
        idle_stat5.reset_index(drop=True, inplace=True)


        # here we create a dictionary of several datasets. These datasets consist of adjacent coordinates. Difference
        # between them is not greater than 5.

        # status_0 = 5

        def dict_idle(df):
            dict_stat_raw = {}
            k = 0
            while k <= len(df):
                i = 0
                while i <= len(df):
                    try:
                        if df.loc[i + 1, 'y_coor'] - df.loc[i, 'y_coor'] <= 5:
                            dict_stat_raw[k] = pd.DataFrame(df.loc[:i + 1])
                            i += 1
                        else:
                            dict_stat_raw[k] = pd.DataFrame(df.loc[:i])
                            break
                    except:
                        dict_stat_raw[k] = pd.DataFrame(df.loc[:i])
                        break

                k += 1
                df = df.loc[i + 1:]
                df = df.reset_index(drop=True)
            return dict_stat_raw


        dict_stat5_raw = dict_idle(idle_stat5)


        # here we divide dataframes from the previous dictionary into blocks of 5 rows, where it is necessary.

        # status_0 = 5

        def coor_blocks(df, status):
            new_y_coor_df = {}
            for name, df in df.items():
                if df.shape[0] > 5:
                    i = 4
                    while i <= len(df):
                        row_df = df.loc[i - 4:i]
                        new_y_coor_df[i] = pd.DataFrame(row_df)
                        i += 5
                else:
                    new_y_coor_df[name] = pd.DataFrame(df)

            # Here we reset index numeration, so every block starts with 0 index and ends with 4.
            # Also we take 1 median X and Y coordinates from every block, but take a sum of all idling entries and stops.

            final_y_df = {}
            for name, df in new_y_coor_df.items():
                df.reset_index(drop=True, inplace=True)
                df['y_coor'] = df['y_coor'].median().astype(int)
                df['x_coor'] = df['x_coor'].median().astype(int)
                df['idling'] = df['idling'].sum()
                df[status] = df[status].sum()
                df = df.loc[[0]]
                final_y_df[name] = pd.DataFrame(df)
                conc_df = pd.concat(final_y_df.values(), ignore_index=True)

            return conc_df


        check_stat5 = coor_blocks(dict_stat5_raw, 'status0_5')
        check_stat5 = check_stat5.sort_values(by='idling', ascending=False)
        check_stat5.reset_index(drop=True, inplace=True)


        # final dataset with both statuses
        def final_df(df5):
            final_joined_df = df5
            final_joined_df['status0_5'] = final_joined_df['status0_5'].fillna(0).astype(int)
            final_joined_df['total'] = final_joined_df['status0_5']
            final_joined_df = final_joined_df.sort_values(by='idling', ascending=False)
            final_joined_df.reset_index(drop=True, inplace=True)
            final_joined_df['Y,X'] = final_joined_df['y_coor'].apply(str) + ', ' + final_joined_df['x_coor'].apply(str)
            final_joined_df.drop(['y_coor', 'x_coor'], inplace=True, axis=1)
            final_joined_df = final_joined_df[['Y,X', 'idling', 'status0_5', 'total']]
            return final_joined_df


        final_joined_df = final_df(check_stat5)


        def total_idle_plot(df):
            colors = ['green' if x == 0 else 'red' for x in df['status0_5']]
            ax = sns.barplot(x=df['Y,X'], y=df['idling'].astype('timedelta64[s]'), data=df,
                             palette=colors)
            ax.set_title('Idling time of {} during {}'.format(dump_truck, date), fontsize=20)
            plt.xticks(rotation=65, fontsize=12)
            plt.yticks(fontsize=12)
            ax.set_ylabel('Idling time [sec]', fontsize=17)
            ax.set_xlabel('Y and X coordinates', fontsize=17)
            ax.set_xlim(-0.5, 14.5)
            leg1 = ax.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=12),
                              Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=12)],
                             ['empty/standing', 'loaded/standing'], loc='upper right')
            ax.add_artist(leg1)
            plt.rc('legend', fontsize=17)
            return plt.savefig('{}_idling_time.png'.format(vehicle_short), facecolor='white', bbox_inches='tight')


        dump_truck = vehicle_raw
        date = date_raw

        total_idle_plot(final_joined_df)


        def idling_report(df):
            with open("{}_idling_result.txt".format(vehicle_short), "a") as f:
                print('Overall idling time on {} -'.format(date_raw), df['idling'].sum(), file=f)
                print('Total idling time with status "loaded/standing" -', df['idling'].sum(), file=f)
                print('Total idling time with status "empty/standing" -', 0, file=f)


        idling_report(final_joined_df)

        main_ramp = pd.read_pickle(r'main_ramp.pkl')
        small_ramp = pd.read_pickle(r'small_ramp.pkl')
        small_road1 = pd.read_pickle(r'small_road1.pkl')
        small_road2 = pd.read_pickle(r'small_road2.pkl')


        def coor_split(df):
            df[['y_coor', 'x_coor']] = df['Y,X'].str.split(',', expand=True)
            df.drop(['Y,X'], inplace=True, axis=1)
            df = df[['y_coor', 'x_coor', 'idling', 'status0_5', 'total']]
            df['y_coor'] = df['y_coor'].astype(int)
            df['x_coor'] = df['x_coor'].astype(int)
            return df


        final_joined_df = coor_split(final_joined_df)


        def stops_map_plot(df5):
            fig, sx = plt.subplots(figsize=(12, 8))
            st5 = plt.scatter(df5['x_coor'], df5['y_coor'], s=300, c=df5['total'], cmap='Reds')
            sx.plot(main_ramp['X'], main_ramp['Y'], linewidth=3, color='tab:blue')
            sx.plot(small_ramp['X'], small_ramp['Y'], linewidth=3, color='tab:blue')
            sx.plot(small_road1['X'], small_road1['Y'], linewidth=3, color='tab:blue')
            sx.plot(small_road2['X'], small_road2['Y'], linewidth=3, color='tab:blue')
            plt.title('Idling locations of {} on {} '.format(dump_truck, date), fontsize=25)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.rc('axes', labelsize=15)
            plt.rc('xtick', labelsize=12)
            plt.rc('ytick', labelsize=12)
            plt.grid(True)

            cbaxes5 = fig.add_axes([0.93, 0.1, 0.03, 0.8])
            cb5 = plt.colorbar(st5, cax=cbaxes5)
            cb5.set_label('Number of stops')

            leg1 = sx.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=12),
                              Line2D([0], [0], color='tab:blue')], ['empty/standing', 'loaded/standing', 'main ramp'],
                             loc='lower center')

            sx.add_artist(leg1)
            plt.rc('legend', fontsize=10)

            return plt.savefig('{}_idling_map.png'.format(vehicle_short), facecolor='white', bbox_inches='tight')


        dump_truck = vehicle_raw
        date = date_raw
        stops_map_plot(final_joined_df)

    else:

        # If there is information about idling time with both statues we create a dictionary of several datasets.
        # These datasets consist of adjacent coordinates. Difference between them is not greater than 5.

        # status_0 = 1

        def dict_idle(df):

            dict_stat_raw = {}
            k = 0
            while k <= len(df):
                i = 0
                while i <= len(df):
                    try:
                        if df.loc[i + 1, 'y_coor'] - df.loc[i, 'y_coor'] <= 5:
                            dict_stat_raw[k] = pd.DataFrame(df.loc[:i + 1])
                            i += 1
                        else:
                            dict_stat_raw[k] = pd.DataFrame(df.loc[:i])
                            break
                    except:
                        dict_stat_raw[k] = pd.DataFrame(df.loc[:i])
                        break

                k += 1
                df = df.loc[i + 1:]
                df = df.reset_index(drop=True)

            return dict_stat_raw


        dict_stat1_raw = dict_idle(idle_stat1)


        # here we divide dataframes from the previous dictionary into blocks of 5 rows, where it is necessary.

        # status_0 = 1

        def coor_blocks(df, status):
            new_y_coor_df = {}
            for name, df in df.items():
                if df.shape[0] > 5:
                    i = 4
                    while i <= len(df):
                        row_df = df.loc[i - 4:i]
                        new_y_coor_df[i] = pd.DataFrame(row_df)
                        i += 5
                else:
                    new_y_coor_df[name] = pd.DataFrame(df)

            # Here we reset index numeration, so every block starts with 0 index and ends with 4.
            # Also we take 1 median X and Y coordinates from every block, but take a sum of all idling entries and stops.

            final_y_df = {}
            for name, df in new_y_coor_df.items():
                df.reset_index(drop=True, inplace=True)
                df['y_coor'] = df['y_coor'].median().astype(int)
                df['x_coor'] = df['x_coor'].median().astype(int)
                df['idling'] = df['idling'].sum()
                df[status] = df[status].sum()
                df = df.loc[[0]]
                final_y_df[name] = pd.DataFrame(df)
                conc_df = pd.concat(final_y_df.values(), ignore_index=True)
            return conc_df


        check_stat1 = coor_blocks(dict_stat1_raw, 'status0_1')
        check_stat1 = check_stat1.sort_values(by='idling', ascending=False)
        check_stat1.reset_index(drop=True, inplace=True)

        # here divide current dataset into 2 regarding status_0 in order to properly calculate idling time for
        # locations with status0_1 and status0_5

        # status_0 = 5

        idle_stat5 = joined_df.query('status0_1 == 0')
        idle_stat5 = idle_stat5.drop(['status0_1', 'total'], axis=1)
        idle_stat5.reset_index(drop=True, inplace=True)

        # here we create a dictionary of several datasets. These datasets consist of adjacent coordinates.
        # Difference between them is not greater than 5.

        # status_0 = 5

        dict_stat5_raw = dict_idle(idle_stat5)

        check_stat5 = coor_blocks(dict_stat5_raw, 'status0_5')
        check_stat5 = check_stat5.sort_values(by='idling', ascending=False)
        check_stat5.reset_index(drop=True, inplace=True)


        # final dataset with both statuses
        def final_df(df1, df5):
            final_joined_df = df1.append(df5)
            final_joined_df['status0_1'] = final_joined_df['status0_1'].fillna(0).astype(int)
            final_joined_df['status0_5'] = final_joined_df['status0_5'].fillna(0).astype(int)
            final_joined_df['total'] = final_joined_df['status0_1'] + final_joined_df['status0_5']
            final_joined_df = final_joined_df.sort_values(by='idling', ascending=False)
            final_joined_df.reset_index(drop=True, inplace=True)
            final_joined_df['Y,X'] = final_joined_df['y_coor'].apply(str) + ', ' + final_joined_df['x_coor'].apply(str)
            final_joined_df.drop(['y_coor', 'x_coor'], inplace=True, axis=1)
            final_joined_df = final_joined_df[['Y,X', 'idling', 'status0_1', 'status0_5', 'total']]
            return final_joined_df


        final_joined_df = final_df(check_stat1, check_stat5)


        def total_idle_plot(df):
            colors = ['red' if x == 0 else 'green' for x in df['status0_1']]
            kx = sns.barplot(x=df['Y,X'], y=df['idling'].astype('timedelta64[s]'), data=df,
                             palette=colors)
            kx.set_title('Idling time of {} during {}'.format(dump_truck, date), fontsize=20)
            plt.xticks(rotation=65, fontsize=12)
            plt.yticks(fontsize=12)
            kx.set_ylabel('Idling time [sec]', fontsize=17)
            kx.set_xlabel('Y and X coordinates', fontsize=17)
            kx.set_xlim(-0.5, 14.5)
            leg1 = kx.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=12),
                              Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=12)],
                             ['empty/standing', 'loaded/standing'], loc='upper right')
            kx.add_artist(leg1)
            plt.rc('legend', fontsize=17)
            return plt.savefig('{}_idling_time.png'.format(vehicle_short), facecolor='white', bbox_inches='tight')


        dump_truck = vehicle_raw
        date = date_raw

        total_idle_plot(final_joined_df)

        main_ramp = pd.read_pickle(r'main_ramp.pkl')
        small_ramp = pd.read_pickle(r'small_ramp.pkl')
        small_road1 = pd.read_pickle(r'small_road1.pkl')
        small_road2 = pd.read_pickle(r'small_road2.pkl')


        def coor_split(df):
            df[['y_coor', 'x_coor']] = df['Y,X'].str.split(',', expand=True)
            df.drop(['Y,X'], inplace=True, axis=1)
            df = df[['y_coor', 'x_coor', 'idling', 'status0_1', 'status0_5', 'total']]
            df['y_coor'] = df['y_coor'].astype(int)
            df['x_coor'] = df['x_coor'].astype(int)
            st1 = df.query('status0_5 == 0')
            st1 = st1.drop(['status0_5'], axis=1)
            st5 = df.query('status0_1 == 0')
            st5 = st5.drop(['status0_1'], axis=1)
            return df, st1, st5


        final_joined_df, stat_1, stat_5 = coor_split(final_joined_df)


        def stops_map_plot(df1, df5):
            fig, fx = plt.subplots(figsize=(12, 8))
            st1 = plt.scatter(df1['x_coor'], df1['y_coor'], s=300, c=df1['total'], cmap='Greens')
            st5 = plt.scatter(df5['x_coor'], df5['y_coor'], s=300, c=df5['total'], cmap='Reds')
            fx.plot(main_ramp['X'], main_ramp['Y'], linewidth=3, color='tab:blue')
            fx.plot(small_ramp['X'], small_ramp['Y'], linewidth=3, color='tab:blue')
            fx.plot(small_road1['X'], small_road1['Y'], linewidth=3, color='tab:blue')
            fx.plot(small_road2['X'], small_road2['Y'], linewidth=3, color='tab:blue')
            plt.title('Idling locations of {} on {} '.format(dump_truck, date), fontsize=25)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.rc('axes', labelsize=15)
            plt.rc('xtick', labelsize=12)
            plt.rc('ytick', labelsize=12)
            plt.grid(True)

            cbaxes1 = fig.add_axes([-0.001, 0.1, 0.03, 0.8])
            cbaxes5 = fig.add_axes([0.93, 0.1, 0.03, 0.8])
            cb1 = plt.colorbar(st1, cax=cbaxes1)
            cb5 = plt.colorbar(st5, cax=cbaxes5)
            cbaxes1.yaxis.set_ticks_position('left')
            cb1.set_label('Number of stops')
            cb5.set_label('Number of stops')

            leg1 = fx.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=12),
                              Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=12),
                              Line2D([0], [0], color='tab:blue')], ['empty/standing', 'loaded/standing', 'main ramp'],
                             loc='lower center')

            fx.add_artist(leg1)
            plt.rc('legend', fontsize=10)

            return plt.savefig('{}_idling_map.png'.format(vehicle_short), facecolor='white', bbox_inches='tight')


        dump_truck = vehicle_raw
        date = date_raw

        stops_map_plot(stat_1, stat_5)


        # This is a final report about total idling time, overall idling time with statuses 'empty standing' and
        # 'loaded standing'

        def result(df):
            with open("{}_idling_result.txt".format(vehicle_short), "a") as f:
                print('Overall idling time on {} -'.format(date_raw), df['idling'].sum(), file=f)

                if (df['status0_1'] == 0).any() and (df['status0_5'] == 0).any():
                    print('Total idling time with status "empty/standing" -',
                          df.query('status0_1 !=0')['idling'].sum(), file=f)
                    print('Total idling time with status "loaded/standing" -',
                          df.query('status0_5 !=0')['idling'].sum(), file=f)
                elif (df['status0_1'] == 0).all():
                    print('Total idling time with status "loaded/standing" -',
                          df.query('status0_5 !=0')['idling'].sum(), file=f)
                else:
                    print('Total idling time with status "empty/standing" -',
                          df.query('status0_1 !=0')['idling'].sum(), file=f)


        result(final_joined_df)
