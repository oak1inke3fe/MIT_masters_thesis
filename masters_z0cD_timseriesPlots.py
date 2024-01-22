#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:32:10 2023

@author: oaklin keefe


This file is used to make timeseries plots of 

INPUT files:
    z0_combinedAnalysis.csv
    dragCoefficient_combinedAnalysis.csv
    windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv
    date_combinedAnalysis.csv
    
OUTPUT files:
    Only figures:
        timeseries_z0cD_Spring.png
        timeseries_z0cD_Fall.png
        

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


print('done with imports')
#%%

file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

z0_file = "z0_combinedAnalysis.csv"
z0_df = pd.read_csv(file_path+z0_file)
z0_df = z0_df.drop(['Unnamed: 0'], axis=1)

cd_file = 'dragCoefficient_combinedAnalysis.csv'
cd_df = pd.read_csv(file_path + cd_file)
cd_df = cd_df.drop(['Unnamed: 0'], axis=1)

windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv")
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

break_index = 3959

print('done')


#%% First, get rid of bad wind directions

plt.figure()
plt.scatter(windDir_df.index, windDir_df['alpha_s4'], color='gray',label='before')
plt.title('Wind direction BEFORE mask')
plt.xlabel('index')
plt.ylabel('Wind direction [deg]')
plt.legend(loc='lower right')
plt.show()

windDir_index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, windDir_index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],windDir_index_array)

windDir_df[mask_goodWindDir] = np.nan

z0_df[mask_goodWindDir] = np.nan
cd_df[mask_goodWindDir] = np.nan


print('done with setting up  good wind direction only dataframes')

# plt.figure()
plt.scatter(windDir_df.index, windDir_df['alpha_s4'], color = 'green',label = 'after')
plt.title('Wind direction AFTER mask')
plt.xlabel('index')
plt.ylabel('Wind direction [deg]')
plt.legend(loc='lower right')
# plt.xlim(4000,5000)

#%%
plt.plot(z0_df['z0_s1'], label = 's1')
# plt.plot(z0_df['z0_s2'], label = 's2')
# plt.plot(z0_df['z0_s3'], label = 's3')
# plt.plot(z0_df['z0_s4'], label = 's4')
plt.title('z0')
plt.legend()

#%%
plt.plot(cd_df['cd_s1'], label = 's1')
# plt.plot(cd_df['cd_s2'], label = 's2')
# plt.plot(cd_df['cd_s3'], label = 's3')
# plt.plot(cd_df['cd_s4'], label = 's4')
plt.title('$c_D$')
plt.legend()
#%% Then, pLot the timeseries

date_df = pd.read_csv(file_path+'date_combinedAnalysis.csv')
dates_arr = np.array(pd.to_datetime(date_df['datetime']))

# SPRING
fig, ax1 = plt.subplots(figsize=(8,3))
fig.suptitle('Roghness Length ($z_0$) and Drag Coefficient ($c_D$): Spring')
ax2 = ax1.twinx()
ax1.scatter(dates_arr[:break_index+1], z0_df['z0_s1'][:break_index+1], s=10, color = 'black', label = '$z_{0}$ [m]')
ax2.scatter(dates_arr[:break_index+1], cd_df['cd_s1'][:break_index+1], color = 'forestgreen', edgecolor = 'darkgreen', s=10, label = 'Wind Direction [º]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax1.set_ylabel('Speed ($\overline{u}$) [m/s]', color='gray')
ax2.set_ylabel('Direction [º]', color='forestgreen')
ax2.tick_params(colors='forestgreen', which='both')
ax2.tick_params(colors='black', which='both')

plt.show()

fig.savefig(plot_savePath + "timeseries_z0cD_Spring.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_z0cD_Spring.pdf")



# FALL
fig, ax1 = plt.subplots(figsize=(8,3))
fig.suptitle('Roghness Length ($z_0$) and Drag Coefficient ($c_D$): Fall')
ax2 = ax1.twinx()
ax1.scatter(dates_arr[break_index+1:], z0_df['z0_s1'][break_index+1:], s=10, color = 'black', label = '$z_{0}$ [m]')
ax2.scatter(dates_arr[break_index+1:], cd_df['cd_s1'][break_index+1:], color = 'darkorange', edgecolor = 'darkorange', s=10, label = 'Wind Direction [º]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax1.set_ylabel('Speed ($\overline{u}$) [m/s]', color='black')
ax2.set_ylabel('Direction [º]', color='darkorange')
ax2.tick_params(colors='darkorange', which='both')
ax2.tick_params(colors='black', which='both')

plt.show()

fig.savefig(plot_savePath + "timeseries_z0cD_Fall.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_z0cD_Fall.pdf")
