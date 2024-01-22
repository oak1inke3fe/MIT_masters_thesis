#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:07:04 2023

@author: oaklin keefe
This file is used to calculate ustar, (u*) also known as friction velocity.

INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    
OUTPUT files:
    usr_combinedAnalysis.csv
    a plot of u* versus wind speed
        
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('done with imports')

#%%
# file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'

date_df = pd.read_csv(file_path + 'date_combinedAnalysis.csv')

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic1_df = sonic1_df.drop(['new_index'], axis=1)
# print(sonic1_df.columns)


sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic2_df = sonic2_df.drop(['new_index'], axis=1)


sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)
sonic3_df = sonic3_df.drop(['new_index'], axis=1)


sonic_file4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)
sonic4_df = sonic4_df.drop(['new_index'], axis=1)

windDir_file = "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv"
windDir_df = pd.read_csv(file_path + windDir_file)

print('done reading in sonics')
#%%

usr_s1 = ((sonic1_df['UpWp_bar'])**2+(sonic1_df['VpWp_bar'])**2)**(1/4)

usr_s2 = ((sonic2_df['UpWp_bar'])**2+(sonic2_df['VpWp_bar'])**2)**(1/4)

usr_s3 = ((sonic3_df['UpWp_bar'])**2+(sonic3_df['VpWp_bar'])**2)**(1/4)

usr_s4 = ((sonic4_df['UpWp_bar'])**2+(sonic4_df['VpWp_bar'])**2)**(1/4)

usr_s1_noV = (-1*sonic1_df['UpWp_bar'])**(1/2)
usr_s2_noV = (-1*sonic2_df['UpWp_bar'])**(1/2)
usr_s3_noV = (-1*sonic3_df['UpWp_bar'])**(1/2)
usr_s4_noV = (-1*sonic4_df['UpWp_bar'])**(1/2)




USTAR_df = pd.DataFrame()
USTAR_df['usr_s1'] = np.array(usr_s1)
USTAR_df['usr_s2'] = np.array(usr_s2)
USTAR_df['usr_s3'] = np.array(usr_s3)
USTAR_df['usr_s4'] = np.array(usr_s4)
USTAR_df['usr_s1_noV'] = np.array(usr_s1_noV)
USTAR_df['usr_s2_noV'] = np.array(usr_s2_noV)
USTAR_df['usr_s3_noV'] = np.array(usr_s3_noV)
USTAR_df['usr_s4_noV'] = np.array(usr_s4_noV)
USTAR_df.to_csv(file_path + 'usr_combinedAnalysis.csv')

plt.figure()
plt.plot(usr_s1, label = "u*_{s1} = $(<u'w'>^{2} + <v'w'>^{2})^{1/4}$")
plt.legend()
plt.title('U*')


print('done with calculaitng ustar and plotting it.')

#%% Mask the DFs to only keep the good wind directions
windDir_index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, windDir_index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],windDir_index_array)

windDir_df[mask_goodWindDir] = np.nan

sonic1_df[mask_goodWindDir] = np.nan
sonic2_df[mask_goodWindDir] = np.nan
sonic3_df[mask_goodWindDir] = np.nan
sonic4_df[mask_goodWindDir] = np.nan

USTAR_df[mask_goodWindDir] = np.nan


print('done with setting up good wind direction only dataframes')
#%%
plt.figure(figsize=(10,5))
plt.scatter(sonic3_df['Ubar'],usr_s3,s=10,color = 'darkorange', edgecolor='red', label = 's3')
plt.scatter(sonic2_df['Ubar'],usr_s2,s=10,color = 'lime', edgecolor='olive', label = 's2')
plt.scatter(sonic1_df['Ubar'],usr_s1,s=10,color = 'blue', edgecolor='navy', label = 's1')
# plt.scatter(sonic4_df['Ubar'],usr_s4,s=10,color = 'gray', edgecolor='black', label = 's4')
plt.legend()
plt.title('Wind Speed ($\overline{u}$) versus Friction Velocity ($u_*$)')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$u_*$ [$ms^{-1}$]')
plt.ylim(0,1)
plt.xlim(2,17)

plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/plots/'
plt.savefig(plot_savePath + "scatter_uStar_v_windSpeed.png",dpi=300)
plt.savefig(plot_savePath + "scatter_uStar_v_windSpeed.pdf")
print('done with plot')

#%%
# Find extraneous points:
extraneous_s4 = np.where(usr_s4 >= 1)
extraneous_s3 = np.where(usr_s3 >= 1)
extraneous_s2 = np.where(usr_s2 >= 1)
extraneous_s1 = np.where(usr_s1 >= 1)

#%% 
# plot best fit line using numpy.polyfit
idx_s1 = np.isfinite(sonic1_df['Ubar']) & np.isfinite(usr_s1)
idx_s2 = np.isfinite(sonic2_df['Ubar']) & np.isfinite(usr_s2)
idx_s3 = np.isfinite(sonic3_df['Ubar']) & np.isfinite(usr_s3)

bf_curve_s1_input = np.polyfit(np.array(sonic1_df['Ubar'][idx_s1]), np.array(usr_s1[idx_s1]), 2)
bf_curve_s1 = np.poly1d(bf_curve_s1_input)
bf_x_s1 = np.linspace(np.min(np.array(sonic1_df['Ubar'][idx_s1])), np.max(np.array(sonic1_df['Ubar'][idx_s1])),50)
bf_y_s1 = bf_curve_s1(bf_x_s1)

bf_curve_s2_input = np.polyfit(np.array(sonic2_df['Ubar'][idx_s2]), np.array(usr_s2[idx_s2]), 2)
bf_curve_s2 = np.poly1d(bf_curve_s2_input)
bf_x_s2 = np.linspace(np.min(np.array(sonic2_df['Ubar'][idx_s2])), np.max(np.array(sonic2_df['Ubar'][idx_s2])),50)
bf_y_s2 = bf_curve_s2(bf_x_s2)

bf_curve_s3_input = np.polyfit(np.array(sonic3_df['Ubar'][idx_s3]), np.array(usr_s3[idx_s3]), 2)
bf_curve_s3 = np.poly1d(bf_curve_s3_input)
bf_x_s3 = np.linspace(np.min(np.array(sonic3_df['Ubar'][idx_s3])), np.max(np.array(sonic3_df['Ubar'][idx_s3])),50)
bf_y_s3 = bf_curve_s3(bf_x_s3)

plt.figure(figsize=(10,5))
plt.scatter(sonic3_df['Ubar'],usr_s3,s=10,color = 'darkorange', edgecolor='red', label = 's3')
plt.scatter(sonic2_df['Ubar'],usr_s2,s=10,color = 'lime', edgecolor='olive', label = 's2')
plt.scatter(sonic1_df['Ubar'],usr_s1,s=10,color = 'blue', edgecolor='navy', label = 's1')
plt.plot(bf_x_s1, bf_y_s1, color = 'k', label= 'best fit')
plt.plot(bf_x_s2, bf_y_s2, color = 'k',)
plt.plot(bf_x_s3, bf_y_s3, color = 'k')
# plt.scatter(sonic4_df['Ubar'],usr_s4,s=10,color = 'gray', edgecolor='black', label = 's4')
plt.legend()
plt.title('Wind Speed ($\overline{u}$) versus Friction Velocity ($u_*$)')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$u_*$ [$ms^{-1}$]')