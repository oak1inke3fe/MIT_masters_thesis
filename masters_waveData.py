# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:56:44 2023

@author: oaklin keefe

This file is used to pad and/or restrict the wave dataset to the correct start/end dates of the spring/fall start and end dates 

Input file location:
    folder where wave data is stored; here: /code_pipeline/
INPUT files:
    BBASIT_Spring_waves.mat
    BBASIT_Fall_waves.mat
    date_combinedAnalysis.csv
    windDir_withBadFlags_110to155_within15degRequirement_combinedAnalysis.csv

Output file location:
    /code_pipeline/Level2/
OUTPUT files:
    wave_despiked_spring.csv
    wave_despiked_fall.csv
    waveData_despiked_combinedAnalysis.csv
    figures
        
"""

#%% IMPORTS
# import os
# import pyrsktools
import numpy as np 
# import regex as re
import pandas as pd
from mat4py import loadmat
# import pyrsktools # Import the library
print('done with imports')

import matplotlib.pyplot as plt

#%%

file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/'
file_save_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
#%%

file_spring = "BBASIT_Spring_waves.mat"
data_spring = loadmat(file_path+file_spring)
print(data_spring.keys())

file_fall = "BBASIT_Fall_waves.mat"
data_fall = loadmat(file_path+file_fall)
print(data_fall.keys())

#%%
sigH_spring = np.array(data_spring['Hsig'])
T_period_spring = np.array(data_spring['T'])
Cp_spring = np.array(data_spring['c'])
dn_arr_spring = np.array(data_spring['mday'])
print('done spring')

sigH_fall = np.array(data_fall['Hsig'])
T_period_fall = np.array(data_fall['T'])
Cp_fall = np.array(data_fall['c'])
dn_arr_fall = np.array(data_fall['mday'])
print('done fall')


#%%
wave_df_spring = pd.DataFrame()
wave_df_spring['index_arr'] = np.arange(len(dn_arr_spring))
wave_df_spring['date_time'] = dn_arr_spring
wave_df_spring['sigH'] = sigH_spring
wave_df_spring['T_period'] = T_period_spring
wave_df_spring['Cp'] = Cp_spring
print('done spring wave df')

wave_df_fall = pd.DataFrame()
wave_df_fall['index_arr'] = np.arange(len(dn_arr_fall))
wave_df_fall['date_time'] = dn_arr_fall
wave_df_fall['sigH'] = sigH_fall
wave_df_fall['T_period'] = T_period_fall
wave_df_fall['Cp'] = Cp_fall
print('done fall wave df')


#%%
import datetime as dt
python_datetime_spring = []
for i in range(len(dn_arr_spring)):
    python_datetime_spring_i = dt.datetime.fromordinal(int(np.array(wave_df_spring['date_time'][i]))) + dt.timedelta(days=np.array(wave_df_spring['date_time'][i])%1) - dt.timedelta(days = 366)
    python_datetime_spring.append(python_datetime_spring_i)
wave_df_spring['date_time'] = python_datetime_spring
print('done spring datetime')


python_datetime_fall = []
for i in range(len(dn_arr_fall)):
    python_datetime_fall_i = dt.datetime.fromordinal(int(np.array(wave_df_fall['date_time'][i]))) + dt.timedelta(days=np.array(wave_df_fall['date_time'][i])%1) - dt.timedelta(days = 366)
    python_datetime_fall.append(python_datetime_fall_i)
wave_df_fall['date_time'] = python_datetime_fall
print('done fall datetime')

#%%
print(wave_df_spring.head(5))
print('above, spring head 5')

print(wave_df_spring.tail(5))
print('above, spring tail 5')



wave_df_spring.index = wave_df_spring['date_time']
del wave_df_spring['date_time']
# Resample to 20-minute data starting on the hour
wave_df_spring_resampled = wave_df_spring.resample('20T').mean()

# Display the original and resampled data
print("Original SPRING Data:")
print(wave_df_spring.head(9))

print("\nResampled SPRING Data:")
print(wave_df_spring_resampled.head(9))

#%%
print(wave_df_fall.head(5))
print('above, fall head 5')

print(wave_df_fall.tail(5))
print('above, fall tail 5')



wave_df_fall.index = wave_df_fall['date_time']
del wave_df_fall['date_time']
# Resample to 20-minute data starting on the hour
wave_df_fall_resampled = wave_df_fall.resample('20T').mean()

# Display the original and resampled data
print("Original FALL Data:")
print(wave_df_fall.head(9))

print("\nResampled FALL Data:")
print(wave_df_fall_resampled.head(9))
#%% interpolate to fill the NaNs

wave_df_spring_interpolated = wave_df_spring_resampled.interpolate(method='linear')
print('done spring interpolation')

print("Resampled SPRING Data:")
print(wave_df_spring_resampled.head(9))

print("\nInterpolated SPRING Data:")
print(wave_df_spring_interpolated.head(9))

#%%
wave_df_fall_interpolated = wave_df_fall_resampled.interpolate(method='linear')
print('done fall interpolation')

print("Resampled FALL Data:")
print(wave_df_fall_resampled.head(9))

print("\nInterpolated FALL Data:")
print(wave_df_fall_interpolated.head(9))
#%%
print(wave_df_spring_interpolated.head(5))
print('above, spring head 5')

'''
- we see that it starts at 4-15-2022 08:00:00, and goes in 20 min intervals
- we need to start it at 4-15-2022 00:00:00, so we will extend the start by 8*3 NaN entries (3, 20-min entries/hour)
'''


print(wave_df_spring_interpolated.tail(5))
print('above, spring tail 5')

'''
- we see that it ends at 5-31-2022 18:20:00, 
- we need to end it at 6-08-2022 23:40:00, so we will extend it by 8*24*3+17  NaN entries (plus 17 to get to june 1 then plus 8*24*3 (8 days w/24 hours and 3 20-min entries/hour) to get to end date)
'''
#%%
a_start = np.empty(8*3,)
a_end = np.empty(8*3*24+17-1,)
a_start[:] = np.nan
a_end[:] = np.nan
df_extension_start = pd.DataFrame({'index_arr': a_start,
                             'sigH': a_start,
                             'T_period': a_start,
                             'Cp': a_start,})
df_extension_end = pd.DataFrame({'index_arr': a_end,
                             'sigH': a_end,
                             'T_period': a_end,
                             'Cp': a_end,})
wave_df_spring_fullStart = pd.concat([df_extension_start, wave_df_spring_interpolated],axis=0)
wave_df_spring_full = pd.concat([wave_df_spring_fullStart, df_extension_end], axis=0)

print("Un-extended SPRING Data:")
print(wave_df_spring_interpolated.head(9))

print("\nExtended SPRING Data:")
print(wave_df_spring_full.head(9))

#%%
print(wave_df_fall_interpolated.head(5))
print('above, fall head 5')

'''
- we see that it starts at 9-22-2022 00:10:00, and goes in 20 min intervals

'''


print(wave_df_fall_interpolated.tail(5))
print('above, fall tail 5')

'''
- we see that it ends at 11-19-2022 00:00:00, 
- we need to end it at 11-21-2022 23:40:00, so we will extend it by 2*24*3  NaN entries (2*24*3 (2 days w/24 hours and 3 20-min entries/hour) to get to end date)
'''
#%%
# a_start = np.empty(8*3,)
a_end = np.empty(2*24*3-1,)
# a_start[:] = np.nan
a_end[:] = np.nan
# df_extension_start = pd.DataFrame({'index_arr': a_start,
#                              'sigH': a_start,
#                              'T_period': a_start,
#                              'Cp': a_start,})
df_extension_end = pd.DataFrame({'index_arr': a_end,
                             'sigH': a_end,
                             'T_period': a_end,
                             'Cp': a_end,})
# wave_df_fall_fullStart = pd.concat([df_extension_start, wave_df_fall_interpolated],axis=0)
wave_df_fall_full = pd.concat([wave_df_fall_interpolated, df_extension_end], axis=0)

print("Un-extended FALL Data:")
print(wave_df_fall_interpolated.tail(9))

print("\nExtended FALL Data:")
print(wave_df_fall_full.tail(9))
#%%
wave_df_combined = pd.concat([wave_df_spring_full, wave_df_fall_full], axis=0)
wave_df_combined['new_index_arr'] = np.arange(len(wave_df_combined))
wave_df_combined.set_index('new_index_arr', inplace=True)

#%%
from hampel import hampel
break_index = 3959
column_arr = ['index_arr', 'sigH', 'T_period', 'Cp', ]

#create a spring df, and make sure the indexing starts at zero by resetting the index
wave_df_good_spring = wave_df_combined[:break_index+1]
wave_df_good_spring = wave_df_good_spring.reset_index(drop = True)


wave_df_arr_spring = [wave_df_good_spring, ]
print('done with creating spring dataframes')

wave_df_despiked_spring = pd.DataFrame()

wave_despike_arr_spring = [wave_df_despiked_spring, ]

for i in range(len(wave_df_arr_spring)):
# for sonic in sonics_df_arr:
    for column_name in column_arr:
    
        my_array = wave_df_arr_spring[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 10
        n = 3
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        wave_despike_arr_spring[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('sonic '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done SPRING hampel')



#create a fall df, and make sure the indexing starts at zero by resetting the index
wave_df_good_fall = wave_df_combined[break_index+1:]
wave_df_good_fall = wave_df_good_fall.reset_index(drop = True)


wave_df_arr_fall = [wave_df_good_fall, ]
print('done with creating fall dataframes')

wave_df_despiked_fall = pd.DataFrame()

wave_despike_arr_fall = [wave_df_despiked_fall, ]

for i in range(len(wave_df_arr_fall)):
# for sonic in sonics_df_arr:
    for column_name in column_arr:
    
        my_array = wave_df_arr_fall[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 10
        n = 3
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        wave_despike_arr_fall[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('sonic '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done FALL hampel')


#%%
# save new despiked values/dataframes to new files separated by spring/fall deployment
wave_df_despiked_spring.to_csv(file_save_path + 'wave_despiked_spring.csv')
wave_df_despiked_fall.to_csv(file_save_path + 'wave_despiked_fall.csv')

print('done saving despiked to .csv')
#%%
file_path = file_save_path
wave_df_combined_despiked = pd.concat([wave_df_despiked_spring, wave_df_despiked_fall], axis=0)
dates_df = pd.read_csv(file_path + 'date_combinedAnalysis.csv')

wave_df_combined_despiked['new_datetime'] = dates_df['datetime']
print(wave_df_combined_despiked.head(9))


#%%

wave_df_combined_despiked['new_index_arr'] = np.arange(len(wave_df_combined_despiked))
wave_df_combined_despiked.set_index('new_index_arr', inplace=True)

wave_df_combined_despiked.to_csv(file_save_path +'waveData_despiked_combinedAnalysis.csv')
print('done. Saved to .csv')

#%%
# get rid of bad wind directions
windDir_file = "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv"
windDir_df = pd.read_csv(file_path + windDir_file)
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

windDir_index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, windDir_index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],windDir_index_array)

windDir_df[mask_goodWindDir] = np.nan
wave_df_combined_despiked[mask_goodWindDir] = np.nan
#%%
#some test plots
import matplotlib.dates as mdates

fig, axs = plt.subplots(3, sharex = True, sharey=False)
fig.suptitle('Wave Data Time Series Fall')
axs[0].plot(wave_df_combined_despiked['sigH'], label = 'sigH')
plt.ylabel('$H_{sig}$ [m]')
axs[1].plot(wave_df_combined_despiked['T_period'], label = 'T')
plt.ylabel('T [s]')
axs[2].plot(wave_df_combined_despiked['Cp'], label = '$c_p$')
plt.ylabel('$c_p$ [m/s]')
axs[2].tick_params(axis = 'x', rotation=90)


#%%
break_index = 3959
date_df = pd.read_csv(file_path+'date_combinedAnalysis.csv')
dates_arr = np.array(pd.to_datetime(date_df['datetime']))
axis_font_size = 8
title_font_size = 12
# SPRING
s=2
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize=(7,4))
fig.suptitle('Wave Data: Spring Deployment Period', fontsize=title_font_size)
fig.tight_layout()
fig.subplots_adjust(top=0.875)
ax1.hlines(y=1,xmin=dates_arr[0],xmax=dates_arr[3959],color='white',linestyles='--')
ax1.scatter(dates_arr[:break_index+1], wave_df_combined_despiked['sigH'][:break_index+1], s=s, color = 'navy', label = '$H_{sig}$')
# ax1.hlines(y=1,xmin=dates_arr[0],xmax=dates_arr[3959],color='k',linestyles='--', label = '1m')
# ax1.hlines(y=1.5,xmin=dates_arr[1000],xmax=dates_arr[3959],color='k',linestyles='--')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=axis_font_size)
# ax1.legend(prop={'size': 6}, loc='upper right')
ax1.set_title('$H_{sig}$ [m]', fontsize=axis_font_size)
ax1.tick_params(axis='y', labelsize=axis_font_size)
ax1.set_ylim([0,2])

ax2.scatter(dates_arr[:break_index+1], wave_df_combined_despiked['Cp'][:break_index+1], s=s, color = 'dimgray', label = '$c_{p}$')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=axis_font_size)
# ax2.legend(prop={'size': 6}, loc='upper right')
ax2.set_title('$c_{p}$ [m/s]', fontsize=axis_font_size)
ax2.tick_params(axis='y', labelsize=axis_font_size)
ax2.set_ylim([0,12])

ax3.scatter(dates_arr[:break_index+1], wave_df_combined_despiked['T_period'][:break_index+1], s=s, color = 'darkslategray', label = '$T$')
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax3.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=axis_font_size)
# ax3.legend(prop={'size': 6}, loc='upper right')
ax3.set_title('$T$ [s]', fontsize=axis_font_size)
ax3.tick_params(axis='y', labelsize=axis_font_size)
ax3.set_ylim([0,10])

ax1.set_ylabel('$H_{sig}$ [m]', fontsize=axis_font_size)
ax2.set_ylabel('$c_{p}$ [m/s]', fontsize=axis_font_size)
ax3.set_ylabel('$T$ [s]', fontsize=axis_font_size)


plt.show()


plot_savePath = file_save_path
fig.savefig(plot_savePath + "timeseries_WaveData_Spring.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_WaveData_Spring.pdf")

#%%

# FALL

fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize=(7,4))
fig.suptitle('Wave Data: Fall Deployment Period', fontsize=title_font_size)
fig.tight_layout()
fig.subplots_adjust(top=0.875)
ax1.hlines(y=1,xmin=dates_arr[break_index+1],xmax=dates_arr[-1],color='white',linestyles='--')
ax1.scatter(dates_arr[break_index+1:], wave_df_combined_despiked['sigH'][break_index+1:], s=s, color = 'navy', label = '$H_{sig}$')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=axis_font_size)
ax1.set_title('$H_{sig}$ [m]', fontsize=axis_font_size)
ax1.tick_params(axis='y', labelsize=axis_font_size)
ax1.set_ylim([0,2])
# ax1.hlines(y=1.5,xmin=dates_arr[break_index+1],xmax=dates_arr[-1],color='k',linestyles='--')
# ax1.legend(prop={'size': 6}, loc='upper right')

ax2.scatter(dates_arr[break_index+1:], wave_df_combined_despiked['Cp'][break_index+1:], s=s, color = 'dimgray', label = '$c_{p}$')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=axis_font_size)
ax2.set_title('$c_{p}$ [m/s]', fontsize=axis_font_size)
ax2.tick_params(axis='y', labelsize=axis_font_size)
ax2.set_ylim([0,12])
# ax2.legend(prop={'size': 6}, loc='upper right')

ax3.scatter(dates_arr[break_index+1:], wave_df_combined_despiked['T_period'][break_index+1:], s=s, color = 'darkslategray', label = '$T$')
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax3.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=axis_font_size)
ax3.set_title('$T$ [s]', fontsize=axis_font_size)
ax3.tick_params(axis='y', labelsize=axis_font_size)
ax3.set_ylim([0,10])
# ax3.legend(prop={'size': 6}, loc='upper right')

ax1.set_ylabel('$H_{sig}$ [m]', fontsize=axis_font_size)
ax2.set_ylabel('$c_{p}$ [m/s]', fontsize=axis_font_size)
ax3.set_ylabel('$T$ [s]', fontsize=axis_font_size)


plt.show()


plot_savePath = file_save_path
fig.savefig(plot_savePath + "timeseries_WaveData_Fall.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_WaveData_Fall.pdf")







