#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:32:10 2023

@author: oaklin keefe


This file is used to make timeseries plots of wind direction and wind speed for both deployments. 

NOTE: only sonic 3 is used for simplicity, and since we excluded sonic 4 from the majority of Thesis analysis.

Input file location:
    /code_pipeline/Level2/
INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv
    date_combinedAnalysis.csv

Output file location:
    /code_pipeline/Level2/    
OUTPUT files:
    Only figures:
        

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


print('done with imports')
#%%

file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
plot_savePath = file_path

sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)
sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)

windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv")
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

break_index = 3959

print('done')


#%% First, get rid of bad wind directions first

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

sonic3_df[mask_goodWindDir] = np.nan


print('done with setting up  good wind direction only dataframes')

# plt.figure()
plt.scatter(windDir_df.index, windDir_df['alpha_s4'], color = 'green',label = 'after')
plt.title('Wind direction AFTER mask')
plt.xlabel('index')
plt.ylabel('Wind direction [deg]')
plt.legend(loc='lower right')
# plt.xlim(4000,5000)

#%% Then, pLot the timeseries

date_df = pd.read_csv(file_path+'date_combinedAnalysis.csv')
dates_arr = np.array(pd.to_datetime(date_df['datetime']))

# SPRING
fig, ax1 = plt.subplots(figsize=(7,2))
fig.suptitle('Wind Speed and Direction', fontsize=12)
ax1.plot(dates_arr[:break_index+1], sonic3_df['Ubar'][:break_index+1], color = 'Navy', label = '$\overline{u}_{s3}$')
ax1.plot(dates_arr[:break_index+1], sonic2_df['Ubar'][:break_index+1], color = 'royalblue', label = '$\overline{u}_{s2}$')
ax1.plot(dates_arr[:break_index+1], sonic1_df['Ubar'][:break_index+1], color = 'lightskyblue', label = '$\overline{u}_{s1}$')
# plt.legend(bbox_to_anchor=[0.5, 0.5],loc='upper right', prop={'size': 6.75})
plt.legend(bbox_to_anchor=[0.94, 1.03], loc='upper right', prop={'size': 6.75})
# plt.legend(loc='upper right', fontsize=7)
ax1.set_ylim(0,18)
ax2 = ax1.twinx()
ax2.scatter(dates_arr[:break_index+1], windDir_df['alpha_s3'][:break_index+1], color = 'gray', edgecolor = 'dimgray', s=5)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=8)
ax1.set_ylabel('Speed ($\overline{u}$) [m/s]', color='navy', fontsize=8)
ax2.set_ylabel('Direction [º]', color='gray', fontsize=8)
ax2.tick_params(colors='gray', which='both')



plt.show()

fig.savefig(plot_savePath + "timeseries_UbarWindDir_Spring.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_UbarWindDir_Spring.pdf")


#%%
# FALL
fig, ax1 = plt.subplots(figsize=(7,2))
# fig.suptitle('Wind Speed and Direction: Fall', fontsize=12)
ax1.plot(dates_arr[break_index+1:], sonic3_df['Ubar'][break_index+1:], color = 'Navy', label = '$\overline{u}_{s3}$ [m/s]')
ax1.plot(dates_arr[break_index+1:], sonic2_df['Ubar'][break_index+1:], color = 'royalblue', label = '$\overline{u}_{s2}$ [m/s]')
ax1.plot(dates_arr[break_index+1:], sonic1_df['Ubar'][break_index+1:], color = 'lightskyblue', label = '$\overline{u}_{s1}$ [m/s]')
# plt.legend(loc='upper right', prop={'size': 4})
ax1.set_ylim(0,18)
ax2 = ax1.twinx()
ax2.scatter(dates_arr[break_index+1:], windDir_df['alpha_s3'][break_index+1:], color = 'gray', edgecolor = 'dimgray', s=5)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=8)
ax1.set_ylabel('Speed ($\overline{u}$) [m/s]', color='navy', fontsize=8)
ax2.set_ylabel('Direction [º]', color='gray', fontsize=8)
ax2.tick_params(colors='gray', which='both')


plt.show()


fig.savefig(plot_savePath + "timeseries_UbarWindDir_Fall.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_UbarWindDir_Fall.pdf")