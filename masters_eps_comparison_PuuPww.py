# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:57:53 2023

@author: oaklin keefe


This file is used to compare the epsilon estimates from inertial dissipation method using the PSD of Puu versus Pww.

For a comparison, we use a Pearson r-correlation

Input file location:
    /code_pipeline/Level2/
INPUT files:
    epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv
    epsW_terms_combinedAnalysis_MAD_k_UoverZbar.csv
    
Output file location:
    /code_pipeline/Level2/    
OUTPUT files:
    n/a; you may add lines to save files/figures if needed

"""


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hampel import hampel
print('done with imports')
#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'

eps_UoverZ_Puu = pd.read_csv(file_path + "epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv")
eps_UoverZ_Pww = pd.read_csv(file_path + "epsW_terms_combinedAnalysis_MAD_k_UoverZbar.csv")


#%%
sonic1_df = pd.DataFrame()
sonic1_df['UoverZ_Pww'] = eps_UoverZ_Pww['epsW_sonic1_MAD']
sonic1_df['UoverZ_Puu'] = eps_UoverZ_Puu['epsU_sonic1_MAD']
r_s1 = sonic1_df.corr()
print(r_s1)

sonic2_df = pd.DataFrame()
sonic2_df['UoverZ_Pww'] = eps_UoverZ_Pww['epsW_sonic2_MAD']
sonic2_df['UoverZ_Puu'] = eps_UoverZ_Puu['epsU_sonic2_MAD']
r_s2 = sonic2_df.corr()
print(r_s2)

sonic3_df = pd.DataFrame()
sonic3_df['UoverZ_Pww'] = eps_UoverZ_Pww['epsW_sonic3_MAD']
sonic3_df['UoverZ_Puu'] = eps_UoverZ_Puu['epsU_sonic3_MAD']
r_s3 = sonic3_df.corr()
print(r_s3)

sonic4_df = pd.DataFrame()
sonic4_df['UoverZ_Pww'] = eps_UoverZ_Pww['epsW_sonic4_MAD']
sonic4_df['UoverZ_Puu'] = eps_UoverZ_Puu['epsU_sonic4_MAD']
r_s4 = sonic4_df.corr()
print(r_s4)
#%%
'''    
After an initial correlation test, we now will despike the input data and then do another correlation test
'''
# despike the data using a hampel rolling mean filter
sonic_arr = ['1','2','3','4']

eps_UoverZ_Pww_despiked = pd.DataFrame()
for sonic in sonic_arr:

    L_array = eps_UoverZ_Pww['epsW_sonic'+sonic+'_MAD']
    
    # Just outlier detection
    input_array = L_array
    window_size = 10
    n = 3
    
    L_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
    L_despiked_1times = L_outlier_in_Ts
    
    # plt.figure()
    # plt.plot(L_despiked_once)

    input_array2 = L_despiked_1times
    L_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)

    eps_UoverZ_Pww_despiked['epsW_sonic'+sonic+'_MAD'] = L_outlier_in_Ts2
    print("UoverZ: "+str(sonic))
    # L_despiked_2times = L_outlier_in_Ts2

eps_UoverZ_Puu_despiked = pd.DataFrame()
for sonic in sonic_arr:

    L_array = eps_UoverZ_Puu['epsU_sonic'+sonic+'_MAD']
    
    # Just outlier detection
    input_array = L_array
    window_size = 10
    n = 3
    
    L_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
    L_despiked_1times = L_outlier_in_Ts
    
    # plt.figure()
    # plt.plot(L_despiked_once)

    input_array2 = L_despiked_1times
    L_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)

    eps_UoverZ_Puu_despiked['epsU_sonic'+sonic+'_MAD'] = L_outlier_in_Ts2
    print("UoverZ: "+str(sonic))
    # L_despiked_2times = L_outlier_in_Ts2

# perform new correlations; label as mod_ to indicate modified
sonic1_df_despiked = pd.DataFrame()
sonic1_df_despiked['mod_Pww'] = eps_UoverZ_Pww_despiked['epsW_sonic1_MAD']
sonic1_df_despiked['mod_Puu'] = eps_UoverZ_Puu_despiked['epsU_sonic1_MAD']
r_s1_despiked = sonic1_df_despiked.corr()
print(r_s1_despiked)

sonic2_df_despiked = pd.DataFrame()
sonic2_df_despiked['mod_Pww'] = eps_UoverZ_Pww_despiked['epsW_sonic2_MAD']
sonic2_df_despiked['mod_Puu'] = eps_UoverZ_Puu_despiked['epsU_sonic2_MAD']
r_s2_despiked = sonic2_df_despiked.corr()
print(r_s2_despiked)

sonic3_df_despiked = pd.DataFrame()
sonic3_df_despiked['mod_Pww'] = eps_UoverZ_Pww_despiked['epsW_sonic3_MAD']
sonic3_df_despiked['mod_Puu'] = eps_UoverZ_Puu_despiked['epsU_sonic3_MAD']
r_s3_despiked = sonic3_df_despiked.corr()
print(r_s3_despiked)

sonic4_df_despiked = pd.DataFrame()
sonic4_df_despiked['mod_Pww'] = eps_UoverZ_Pww_despiked['epsW_sonic4_MAD']
sonic4_df_despiked['mod_Puu'] = eps_UoverZ_Puu_despiked['epsU_sonic4_MAD']
r_s4_despiked = sonic4_df_despiked.corr()
print(r_s4_despiked)

#%% Do a "level" analysis of 95th and 99th percentile correlations
LI_despiked = pd.DataFrame()
LI_despiked['mod_Pww'] = np.array((sonic1_df_despiked['mod_Pww']+sonic2_df_despiked['mod_Pww'])/2)
LI_despiked['mod_Puu'] = np.array((sonic1_df_despiked['mod_Puu']+sonic2_df_despiked['mod_Puu'])/2)

LII_despiked = pd.DataFrame()
LII_despiked['mod_Pww'] = np.array((sonic2_df_despiked['mod_Pww']+sonic3_df_despiked['mod_Pww'])/2)
LII_despiked['mod_Puu'] = np.array((sonic2_df_despiked['mod_Puu']+sonic3_df_despiked['mod_Puu'])/2)
print('done with making despiked (mod) LI and LII dataframes')


L_I_mod_Pww_arr = np.array(LI_despiked['mod_Pww'])
percentile_95_I_mod_Pww = np.nanpercentile(np.abs(L_I_mod_Pww_arr), 95)
percentile_99_I_mod_Pww = np.nanpercentile(np.abs(L_I_mod_Pww_arr), 99)
L_I_mod_Pww_newArr_95 = np.where(np.abs(L_I_mod_Pww_arr) > percentile_95_I_mod_Pww, np.nan, L_I_mod_Pww_arr)
L_I_mod_Pww_newArr_99 = np.where(np.abs(L_I_mod_Pww_arr) > percentile_99_I_mod_Pww, np.nan, L_I_mod_Pww_arr)

L_I_mod_Puu_arr = np.array(LI_despiked['mod_Puu'])
percentile_95_I_mod_Puu = np.nanpercentile(np.abs(L_I_mod_Puu_arr), 95)
percentile_99_I_mod_Puu = np.nanpercentile(np.abs(L_I_mod_Puu_arr), 99)
L_I_mod_Puu_newArr_95 = np.where(np.abs(L_I_mod_Puu_arr) > percentile_95_I_mod_Puu, np.nan, L_I_mod_Puu_arr)
L_I_mod_Puu_newArr_99 = np.where(np.abs(L_I_mod_Puu_arr) > percentile_99_I_mod_Puu, np.nan, L_I_mod_Puu_arr)


L_II_mod_Pww_arr = np.array(LII_despiked['mod_Pww'])
percentile_95_II_mod_Pww = np.nanpercentile(np.abs(L_II_mod_Pww_arr), 95)
percentile_99_II_mod_Pww = np.nanpercentile(np.abs(L_II_mod_Pww_arr), 99)
L_II_mod_Pww_newArr_95 = np.where(np.abs(L_II_mod_Pww_arr) > percentile_95_II_mod_Pww, np.nan, L_II_mod_Pww_arr)
L_II_mod_Pww_newArr_99 = np.where(np.abs(L_II_mod_Pww_arr) > percentile_99_II_mod_Pww, np.nan, L_II_mod_Pww_arr)

L_II_mod_Puu_arr = np.array(LII_despiked['mod_Puu'])
percentile_95_II_mod_Puu = np.nanpercentile(np.abs(L_II_mod_Puu_arr), 95)
percentile_99_II_mod_Puu = np.nanpercentile(np.abs(L_II_mod_Puu_arr), 99)
L_II_mod_Puu_newArr_95 = np.where(np.abs(L_II_mod_Puu_arr) > percentile_95_II_mod_Puu, np.nan, L_II_mod_Puu_arr)
L_II_mod_Puu_newArr_99 = np.where(np.abs(L_II_mod_Puu_arr) > percentile_99_II_mod_Puu, np.nan, L_II_mod_Puu_arr)


perc99_LI_df = pd.DataFrame()
perc99_LI_df['mod_Pww'] = L_I_mod_Pww_newArr_99
perc99_LI_df['mod_Puu'] = L_I_mod_Puu_newArr_99
r_LI_99 = perc99_LI_df.corr()
print(r_LI_99)
r_LI_99_modComparison_str = round(r_LI_99['mod_Pww'][1],3)

perc99_LII_df = pd.DataFrame()
perc99_LII_df['mod_Pww'] = L_II_mod_Pww_newArr_99
perc99_LII_df['mod_Puu'] = L_II_mod_Puu_newArr_99
r_LII_99 = perc99_LII_df.corr()
print(r_LII_99)
r_LII_99_modComparison_str = round(r_LII_99['mod_Pww'][1],3)


perc95_LI_df = pd.DataFrame()
perc95_LI_df['mod_Pww'] = L_I_mod_Pww_newArr_95
perc95_LI_df['mod_Puu'] = L_I_mod_Puu_newArr_95
r_LI_95 = perc95_LI_df.corr()
print(r_LI_95)
r_LI_95_modComparison_str = round(r_LI_95['mod_Pww'][1],3)

perc95_LII_df = pd.DataFrame()
perc95_LII_df['mod_Pww'] = L_II_mod_Pww_newArr_95
perc95_LII_df['mod_Puu'] = L_II_mod_Puu_newArr_95
r_LII_95 = perc95_LII_df.corr()
print(r_LII_95)
r_LII_95_modComparison_str = round(r_LII_95['mod_Pww'][1],3)


#%% plot all values
plt.figure()
plt.scatter((eps_UoverZ_Pww_despiked['eps_sonic2_MAD']+eps_UoverZ_Pww_despiked['eps_sonic3_MAD'])/2,(eps_UoverZ_Puu_despiked['eps_sonic2_MAD'][27:]+eps_UoverZ_Puu_despiked['eps_sonic3_MAD'][27:])/2, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter((eps_UoverZ_Pww_despiked['eps_sonic1_MAD']+eps_UoverZ_Pww_despiked['eps_sonic2_MAD'])/2,(eps_UoverZ_Puu_despiked['eps_sonic1_MAD'][27:]+eps_UoverZ_Puu_despiked['eps_sonic2_MAD'][27:])/2, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates (DC) ($\epsilon$ [$m^{2}s^{-3}$])', fontsize=12)
plt.xlabel('Modeled ISR $\epsilon_{Pww}$')
plt.ylabel('Modeled ISR $\epsilon_{Puu}$')
plt.axis('square')
plt.xlim(-0.1,1.0)
plt.ylim(-0.1,1.0)
plt.legend(loc='upper left')

#%% plot 99-th percentilse
plt.figure()
plt.scatter(L_II_mod_Pww_newArr_99, L_II_mod_Puu_newArr_99, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter(L_I_mod_Pww_newArr_99, L_I_mod_Puu_newArr_99, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates ($\epsilon$ [$m^{2}s^{-3}$]); 99%-ile', fontsize=12)
plt.xlabel('$\epsilon_{Pww}$')
plt.ylabel('$\epsilon_{Puu}$')
plt.axis('square')
plt.xlim(-0.01,0.3)
plt.ylim(-0.01,0.3)
plt.legend(loc='lower right')
ax = plt.gca() 
plt.text(.05, .9, "Pearson's r L II ={:.3f}".format(r_LII_99_modComparison_str), transform=ax.transAxes)
plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_LI_99_modComparison_str), transform=ax.transAxes)

#%% plot 95-th percentilse
plt.figure()
plt.scatter(L_II_mod_Pww_newArr_95, L_II_mod_Puu_newArr_95, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter(L_I_mod_Pww_newArr_95, L_I_mod_Puu_newArr_95, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates ($\epsilon$ [$m^{2}s^{-3}$]); 95%-ile', fontsize=12)
plt.xlabel('$\epsilon_{Pww}$')
plt.ylabel('$\epsilon_{Puu}$')
plt.axis('square')
plt.xlim(-0.01,0.3)
plt.ylim(-0.01,0.3)
plt.legend(loc='lower right')
ax = plt.gca() 
plt.text(.05, .9, "Pearson's r L II ={:.3f}".format(r_LII_95_modComparison_str), transform=ax.transAxes)
plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_LI_95_modComparison_str), transform=ax.transAxes)
