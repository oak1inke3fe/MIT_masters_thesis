#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:27:10 2023

@author: oaklin keefe


This file is used to make timeseries plots of production, dissipation rate, and buoyancy (and wave-coherent pressure-work)

Input file location:
    /code_pipeline/Level2/
INPUT files:
    prodTerm_combinedAnalysis.csv
    epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv
    buoy_terms_combinedAnalysis.csv
    pw_combinedAnalysis.csv
    windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv
    date_combinedAnalysis.csv
    
Output file location:
    /code_pipeline/Level2/
OUTPUT files:
    Only figures:
        timeseries_PBEps_Spring.png
        timeseries_PBEps_Fall.png
        timeseries_PBEpsPW_Spring.png
        timeseries_PBEpsPW_Fall.png
        

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

prod_file = "prodTerm_combinedAnalysis.csv"
prod_df = pd.read_csv(file_path+prod_file)
prod_df = prod_df.drop(['Unnamed: 0'], axis=1)

eps_file = 'epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv'
eps_df = pd.read_csv(file_path + eps_file)
eps_df = eps_df.drop(['Unnamed: 0'], axis=1)

buoy_file = 'buoy_terms_combinedAnalysis.csv'
buoy_df = pd.read_csv(file_path + buoy_file)
buoy_df = buoy_df.drop(['Unnamed: 0'], axis=1)

pw_file = 'pw_combinedAnalysis.csv'
pw_df = pd.read_csv(file_path + pw_file)
pw_df = pw_df.drop(['Unnamed: 0'], axis=1)

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

prod_df[mask_goodWindDir] = np.nan
eps_df[mask_goodWindDir] = np.nan
buoy_df[mask_goodWindDir] = np.nan
pw_df[mask_goodWindDir] = np.nan

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
s=5
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(7,5))
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.suptitle('Production ($P$), Buoyancy ($B$), and Dissipation Rate ($\epsilon$) [$m^2/s^3$]: Spring')
ax1.scatter(dates_arr[:break_index+1], prod_df['prod_II'][:break_index+1], s=s, color = 'mediumblue', label = '$P$')
ax1.scatter(dates_arr[:break_index+1], buoy_df['buoy_II'][:break_index+1], s=s, color = 'black', label = '$B$')
ax1.scatter(dates_arr[:break_index+1], (eps_df['epsU_sonic3_MAD'][:break_index+1]+eps_df['epsU_sonic2_MAD'][:break_index+1])/2, s=5, color = 'gray', label = '$\epsilon$')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax1.legend(fontsize=7,loc='lower left')
ax1.set_yscale('log')
ax1.set_title('Level II')
ax1.set_ylim([10**-6,1])
ax2.scatter(dates_arr[:break_index+1], prod_df['prod_I'][:break_index+1], s=s, color = 'mediumblue', label = '$P$')
ax2.scatter(dates_arr[:break_index+1], buoy_df['buoy_I'][:break_index+1], s=s, color = 'black', label = '$B$')
ax2.scatter(dates_arr[:break_index+1], (eps_df['epsU_sonic2_MAD'][:break_index+1]+eps_df['epsU_sonic1_MAD'][:break_index+1])/2, s=s, color = 'gray', label = '$\epsilon$')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
# ax2.legend(fontsize=7)
ax2.set_yscale('log')
ax2.set_title('Level I')
ax2.set_ylim([10**-6,1])
ax1.set_ylabel('$P, \;B, \; \epsilon$ [$m^2/s^3$]')
ax2.set_ylabel('$P, \;B, \; \epsilon$  [$m^2/s^3$]')


plt.show()

fig.savefig(plot_savePath + "timeseries_PBEps_Spring.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_PBEps_Spring.pdf")

#%%

# FALL
s=5
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(7,5))
fig.suptitle('Production ($P$), Buoyancy ($B$), and Dissipation Rate ($\epsilon$) [$m^2/s^3$]: Fall')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
ax1.scatter(dates_arr[break_index+1:], prod_df['prod_II'][break_index+1:], s=s, color = 'mediumblue', label = '$P$')
ax1.scatter(dates_arr[break_index+1:], buoy_df['buoy_II'][break_index+1:], s=s, color = 'black', label = '$B$')
ax1.scatter(dates_arr[break_index+1:], (eps_df['epsU_sonic3_MAD'][break_index+1:]+eps_df['epsU_sonic2_MAD'][break_index+1:])/2, s=s, color = 'gray', label = '$\epsilon$')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax1.legend(fontsize=7, loc='lower left')
ax1.set_yscale('log')
ax1.set_title('Level II')
ax1.set_ylim([10**-6,1])
ax2.scatter(dates_arr[break_index+1:], prod_df['prod_I'][break_index+1:], s=s, color = 'mediumblue', label = '$P$')
ax2.scatter(dates_arr[break_index+1:], buoy_df['buoy_I'][break_index+1:], s=s, color = 'black', label = '$B$')
ax2.scatter(dates_arr[break_index+1:], (eps_df['epsU_sonic2_MAD'][break_index+1:]+eps_df['epsU_sonic1_MAD'][break_index+1:])/2, s=s, color = 'gray', label = '$\epsilon$')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
# ax2.legend(fontsize=7)
ax2.set_yscale('log')
ax2.set_title('Level I')
ax2.set_ylim([10**-6,1])
ax1.set_ylabel('$P, \;B, \; \epsilon$ [$m^2/s^3$]')
ax2.set_ylabel('$P, \;B, \; \epsilon$  [$m^2/s^3$]')


plt.show()

fig.savefig(plot_savePath + "timeseries_PBEps_Fall.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_PBEps_Fall.pdf")


#%% NOW INCLUDE PW
# SPRING
s=5
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(7,5))
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.suptitle('Production ($P$), Buoyancy ($B$), and Dissipation Rate ($\epsilon$) [$m^2/s^3$]: Spring')
ax1.scatter(dates_arr[:break_index+1], prod_df['prod_II'][:break_index+1], s=s, color = 'mediumblue', label = '$P$')
ax1.scatter(dates_arr[:break_index+1], buoy_df['buoy_II'][:break_index+1], s=s, color = 'black', label = '$B$')
ax1.scatter(dates_arr[:break_index+1], (eps_df['epsU_sonic3_MAD'][:break_index+1]+eps_df['epsU_sonic2_MAD'][:break_index+1])/2, s=5, color = 'gray', label = '$\epsilon$')
ax1.scatter(dates_arr[:break_index+1], pw_df['d_dz_pw_theory_II'][:break_index+1], s=s, color = 'cornflowerblue', label = '$T_{\widetilde{pw}}$')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax1.legend(fontsize=7,loc='lower left')
ax1.set_yscale('log')
ax1.set_title('Level II')
ax1.set_ylim([10**-6,1])
ax2.scatter(dates_arr[:break_index+1], prod_df['prod_I'][:break_index+1], s=s, color = 'mediumblue', label = '$P$')
ax2.scatter(dates_arr[:break_index+1], buoy_df['buoy_I'][:break_index+1], s=s, color = 'black', label = '$B$')
ax2.scatter(dates_arr[:break_index+1], (eps_df['epsU_sonic2_MAD'][:break_index+1]+eps_df['epsU_sonic1_MAD'][:break_index+1])/2, s=s, color = 'gray', label = '$\epsilon$')
ax2.scatter(dates_arr[:break_index+1], pw_df['d_dz_pw_theory_I'][:break_index+1], s=s, color = 'cornflowerblue', label = '$T_{\widetilde{pw}}$')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
# ax2.legend(fontsize=7)
ax2.set_yscale('log')
ax2.set_title('Level I')
ax2.set_ylim([10**-6,1])
ax1.set_ylabel('$P, \;B, \; \epsilon$ [$m^2/s^3$]')
ax2.set_ylabel('$P, \;B, \; \epsilon$  [$m^2/s^3$]')


plt.show()

fig.savefig(plot_savePath + "timeseries_PBEpsPW_Spring.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_PBEpsPW_Spring.pdf")

#%%

# FALL
s=5
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(7,5))
fig.suptitle('Production ($P$), Buoyancy ($B$), and Dissipation Rate ($\epsilon$) [$m^2/s^3$]: Fall')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
ax1.scatter(dates_arr[break_index+1:], prod_df['prod_II'][break_index+1:], s=s, color = 'mediumblue', label = '$P$')
ax1.scatter(dates_arr[break_index+1:], buoy_df['buoy_II'][break_index+1:], s=s, color = 'black', label = '$B$')
ax1.scatter(dates_arr[break_index+1:], (eps_df['epsU_sonic3_MAD'][break_index+1:]+eps_df['epsU_sonic2_MAD'][break_index+1:])/2, s=s, color = 'gray', label = '$\epsilon$')
ax1.scatter(dates_arr[break_index+1:], pw_df['d_dz_pw_theory_II'][break_index+1:], s=s, color = 'cornflowerblue', label = '$T_{\widetilde{pw}}$')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax1.legend(fontsize=7, loc='lower left')
ax1.set_yscale('log')
ax1.set_title('Level II')
ax1.set_ylim([10**-6,1])
ax2.scatter(dates_arr[break_index+1:], prod_df['prod_I'][break_index+1:], s=s, color = 'mediumblue', label = '$P$')
ax2.scatter(dates_arr[break_index+1:], buoy_df['buoy_I'][break_index+1:], s=s, color = 'black', label = '$B$')
ax2.scatter(dates_arr[break_index+1:], (eps_df['epsU_sonic2_MAD'][break_index+1:]+eps_df['epsU_sonic1_MAD'][break_index+1:])/2, s=s, color = 'gray', label = '$\epsilon$')
ax2.scatter(dates_arr[break_index+1:], pw_df['d_dz_pw_theory_I'][break_index+1:], s=s, color = 'cornflowerblue', label = '$T_{\widetilde{pw}}$')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
# ax2.legend(fontsize=7)
ax2.set_yscale('log')
ax2.set_title('Level I')
ax2.set_ylim([10**-6,1])
ax1.set_ylabel('$P, \;B, \; \epsilon$ [$m^2/s^3$]')
ax2.set_ylabel('$P, \;B, \; \epsilon$  [$m^2/s^3$]')


plt.show()

fig.savefig(plot_savePath + "timeseries_PBEpsPW_Fall.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_PBEpsPW_Fall.pdf")








