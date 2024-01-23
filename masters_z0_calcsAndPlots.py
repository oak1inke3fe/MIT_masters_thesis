#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:45:24 2023

@author: oaklin keefe
This file is used to calculate roughness length, (z0) 
It also plots comparisons to Ubar, Ustar, and compares Ubar with Ustar only

Input file location:
    /code_pipeline/Level2/
INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv

Output file location:
    /code_pipeline/Level2/'    
OUTPUT files:
    z0_combinedAnalysis.csv
    
    Figures:
        scatter_z0_v_Ubar.png
        scatterBIN_z0_v_Ubar.png
        scatter_z0_v_uStar.png
        scatterBIN_z0_v_uStar.png
        scatter_Ubar_v_uStar.png
        scatterBIN_Ubar_v_uStar.png
        
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import binsreg
print('done with imports')

#%%

file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
plot_save_path = file_path

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)

sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)

sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)

sonic_file4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)

windDir_includingBadDirs_file = "windDir_IncludingBad_wS4rotation_combinedAnalysis.csv"
windDir_includingBad_df = pd.read_csv(file_path + windDir_includingBadDirs_file)

windDir_file = "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv"
windDir_df = pd.read_csv(file_path + windDir_file)

date_file = "date_combinedAnalysis.csv"
date_df = pd.read_csv(file_path + date_file)
print(date_df.columns)

zL_file = "ZoverL_combinedAnalysis.csv"
zL_df = pd.read_csv(file_path + zL_file)
zL_df.head(15)

z_file = "z_airSide_combinedAnalysis.csv"
z_df = pd.read_csv(file_path + z_file)
z_df.columns

plt.figure()
plt.plot(z_df['z_sonic4'])
plt.title('Z sonic 4; combined analysis')

usr_file = "usr_combinedAnalysis.csv"
usr_df = pd.read_csv(file_path + usr_file)


break_index = 3959



#%%
z0_s1 = z_df['z_sonic1']*np.exp(-1*sonic1_df['Ubar']*0.4/usr_df['usr_s1'])
z0_s2 = z_df['z_sonic2']*np.exp(-1*sonic2_df['Ubar']*0.4/usr_df['usr_s2'])
z0_s3 = z_df['z_sonic3']*np.exp(-1*sonic3_df['Ubar']*0.4/usr_df['usr_s3'])
z0_s4 = z_df['z_sonic4']*np.exp(-1*sonic4_df['Ubar']*0.4/usr_df['usr_s4'])

z0_df = pd.DataFrame()
z0_df['z0_s1'] = np.array(z0_s1)
z0_df['z0_s2'] = np.array(z0_s2)
z0_df['z0_s3'] = np.array(z0_s3)
z0_df['z0_s4'] = np.array(z0_s4)

z0_df.to_csv(file_path + "z0_combinedAnalysis.csv")

#%% Mask the DFs to only keep the good wind directions
windDir_index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, windDir_index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],windDir_index_array)

windDir_df[mask_goodWindDir] = np.nan

sonic1_df[mask_goodWindDir] = np.nan
sonic2_df[mask_goodWindDir] = np.nan
sonic3_df[mask_goodWindDir] = np.nan
sonic4_df[mask_goodWindDir] = np.nan

zL_df[mask_goodWindDir] = np.nan

z_df[mask_goodWindDir] = np.nan

usr_df[mask_goodWindDir] = np.nan

z0_df[mask_goodWindDir] = np.nan

print('done with setting up good wind direction only dataframes')

#%% Mask the DFs to only keep the near neutral stabilities

# zL_index_array = np.arange(len(zL_df))
# zL_df['new_index_arr'] = np.where((np.abs(zL_df['zL_I_dc'])<=0.5)&(np.abs(zL_df['zL_II_dc'])<=0.5), np.nan, zL_index_array)
# mask_neutral_zL = np.isin(zL_df['new_index_arr'],zL_index_array)

# zL_df[mask_neutral_zL] = np.nan

# windDir_df[mask_neutral_zL] = np.nan

# sonic1_df[mask_neutral_zL] = np.nan
# sonic2_df[mask_neutral_zL] = np.nan
# sonic3_df[mask_neutral_zL] = np.nan
# sonic4_df[mask_neutral_zL] = np.nan

# z_df[mask_neutral_zL] = np.nan

# usr_df[mask_neutral_zL] = np.nan

# print('done with setting up near-neautral stability with good wind directions dataframes')

#%% plot z0 versus wind speed (Ubar)
plt.figure(figsize=(10,5))
plt.scatter(sonic3_df['Ubar'],z0_df['z0_s3'],s=10,color = 'darkorange', edgecolor='red', label = 's3')
plt.scatter(sonic2_df['Ubar'],z0_df['z0_s2'],s=10,color = 'lime', edgecolor='olive', label = 's2')
plt.scatter(sonic1_df['Ubar'],z0_df['z0_s1'],s=10,color = 'blue', edgecolor='navy', label = 's1')

plt.legend()
plt.title('Wind Speed ($\overline{u}$) versus Roughness Length ($z_0$)')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$z_0$ [$m$]')
plt.xlim(2,17)
plt.ylim(0,0.005)

plt.savefig(plot_save_path + "scatter_z0_v_Ubar.png",dpi=300)
plt.savefig(plot_save_path + "scatter_z0_v_Ubar.pdf")
print('done with plot')

#%%
# Find extraneous points:
extraneous_s4 = np.where(z0_df['z0_s4'] >= 0.01)
print(extraneous_s4)
extraneous_s3 = np.where(z0_df['z0_s3'] >= 0.01)
print(extraneous_s3)
extraneous_s2 = np.where(z0_df['z0_s2'] >= 0.01)
print(extraneous_s2)
extraneous_s1 = np.where(z0_df['z0_s1'] >= 0.01)
print(extraneous_s1)

#%%
full_df1 = pd.DataFrame()
full_df1['Ubar'] = np.array(sonic1_df['Ubar'])
full_df1['usr'] = np.array(usr_df['usr_s1'])
full_df1['z0'] = np.array(z0_df['z0_s1'])

full_df2 = pd.DataFrame()
full_df2['Ubar'] = np.array(sonic2_df['Ubar'])
full_df2['usr'] = np.array(usr_df['usr_s2'])
full_df2['z0'] = np.array(z0_df['z0_s2'])

full_df3 = pd.DataFrame()
full_df3['Ubar'] = np.array(sonic3_df['Ubar'])
full_df3['usr'] = np.array(usr_df['usr_s3'])
full_df3['z0'] = np.array(z0_df['z0_s3'])

full_df4 = pd.DataFrame()
full_df4['Ubar'] = np.array(sonic4_df['Ubar'])
full_df4['usr'] = np.array(usr_df['usr_s4'])
full_df4['z0'] = np.array(z0_df['z0_s4'])



def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg
df_binEstimate_full_1_Xubar = binscatter(x='Ubar', y='z0', data= full_df1, ci=(3,3), randcut=1)
df_binEstimate_full_2_Xubar = binscatter(x='Ubar', y='z0', data= full_df2, ci=(3,3), randcut=1)
df_binEstimate_full_3_Xubar = binscatter(x='Ubar', y='z0', data= full_df3, ci=(3,3), randcut=1)
df_binEstimate_full_4_Xubar = binscatter(x='Ubar', y='z0', data= full_df4, ci=(3,3), randcut=1)
print('done with binning data')

#%% plot BINNED z0 versus wind speed (Ubar)

plt.figure(figsize=(10,5))
sns.scatterplot(x='Ubar', y='z0', data=df_binEstimate_full_3_Xubar, color = 'darkorange', label = "binned s3")
plt.errorbar('Ubar', 'z0', yerr='ci', data=df_binEstimate_full_3_Xubar, color = 'red', ls='', lw=2, alpha=0.2, label = 's3 error')
sns.scatterplot(x='Ubar', y='z0', data=df_binEstimate_full_2_Xubar, color = 'olive', label = "binned s3")
plt.errorbar('Ubar', 'z0', yerr='ci', data=df_binEstimate_full_2_Xubar, color = 'green', ls='', lw=2, alpha=0.2, label = 's2 error')
sns.scatterplot(x='Ubar', y='z0', data=df_binEstimate_full_1_Xubar, color = 'navy', label = "binned s3")
plt.errorbar('Ubar', 'z0', yerr='ci', data=df_binEstimate_full_1_Xubar, color = 'blue', ls='', lw=2, alpha=0.2, label = 's1 error')

plt.legend()
plt.title('Binned Wind Speed ($\overline{u}$) versus Roughness Length ($z_0$)')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$z_0$ [$m$]')
plt.xlim(2,17)
plt.ylim(0,0.005)

plt.savefig(plot_save_path + "scatterBIN_z0_v_Ubar.png",dpi=300)
plt.savefig(plot_save_path + "scatterBIN_z0_v_Ubar.pdf")
print('done with plot')

#%% plot z0 versus U*
plt.figure(figsize=(10,5))
plt.scatter(usr_df['usr_s3'],z0_df['z0_s3'],s=10,color = 'darkorange', edgecolor='red', label = 's3')
plt.scatter(usr_df['usr_s2'],z0_df['z0_s2'],s=10,color = 'lime', edgecolor='olive', label = 's2')
plt.scatter(usr_df['usr_s1'],z0_df['z0_s1'],s=10,color = 'blue', edgecolor='navy', label = 's1')

plt.legend()
plt.title('Friction Velocity ($u_*$) versus Roughness Length ($z_0$)')
plt.xlabel('$u_*$ [$ms^{-1}$]')
plt.ylabel('$z_0$ [$m$]')
plt.xlim(0,1)
plt.ylim(0,0.01)

plt.savefig(plot_save_path + "scatter_z0_v_uStar.png",dpi=300)
plt.savefig(plot_save_path + "scatter_z0_v_uStar.pdf")
print('done with plot')

#%%
# Bin the data
def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg
df_binEstimate_full_1_Xusr = binscatter(x='usr', y='z0', data= full_df1, ci=(3,3), randcut=1)
df_binEstimate_full_2_Xusr = binscatter(x='usr', y='z0', data= full_df2, ci=(3,3), randcut=1)
df_binEstimate_full_3_Xusr = binscatter(x='usr', y='z0', data= full_df3, ci=(3,3), randcut=1)
df_binEstimate_full_4_Xusr = binscatter(x='usr', y='z0', data= full_df4, ci=(3,3), randcut=1)
print('done with binning data')

#%% plot BINNED z0 versus u*
import seaborn as sns
plt.figure(figsize=(10,5))
sns.scatterplot(x='usr', y='z0', data=df_binEstimate_full_3_Xusr, color = 'darkorange', label = "binned s3")
plt.errorbar('usr', 'z0', yerr='ci', data=df_binEstimate_full_3_Xusr, color = 'red', ls='', lw=2, alpha=0.2, label = 's3 error')
sns.scatterplot(x='usr', y='z0', data=df_binEstimate_full_2_Xusr, color = 'olive', label = "binned s3")
plt.errorbar('usr', 'z0', yerr='ci', data=df_binEstimate_full_2_Xusr, color = 'green', ls='', lw=2, alpha=0.2, label = 's2 error')
sns.scatterplot(x='usr', y='z0', data=df_binEstimate_full_1_Xusr, color = 'navy', label = "binned s3")
plt.errorbar('usr', 'z0', yerr='ci', data=df_binEstimate_full_1_Xusr, color = 'blue', ls='', lw=2, alpha=0.2, label = 's1 error')

plt.legend()
plt.title('Binned Friction Velocity ($u_*$) versus Roughness Length ($z_0$)')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$z_0$ [$m$]')
plt.xlim(0,1.05)
plt.ylim(-0.0001,0.005)

plt.savefig(plot_save_path + "scatterBIN_z0_v_uStar.png",dpi=300)
plt.savefig(plot_save_path + "scatterBIN_z0_v_uStar.pdf")
print('done with plot')

#%% plot wind speed (Ubar) versus U*
plt.figure(figsize=(10,5))
plt.scatter(sonic3_df['Ubar'],usr_df['usr_s3'],s=10,color = 'darkorange', edgecolor='red', label = 's3')
plt.scatter(sonic2_df['Ubar'],usr_df['usr_s2'],s=10,color = 'lime', edgecolor='olive', label = 's2')
plt.scatter(sonic1_df['Ubar'],usr_df['usr_s1'],s=10,color = 'blue', edgecolor='navy', label = 's1')
plt.legend()
plt.title('Wind Speed ($\overline{u}$) versus Friction Velocity ($u_*$)')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$u_*$ [$ms^{-1}$]')


plt.savefig(plot_save_path + "scatter_Ubar_v_uStar.png",dpi=300)
plt.savefig(plot_save_path + "scatter_Ubar_v_uStar.pdf")
print('done with plot')

#%%
# Bin the data
def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg
df_binEstimate_full_1_X_Ubar_Ustar = binscatter(x='Ubar', y='usr', data= full_df1, ci=(3,3), randcut=1)
df_binEstimate_full_2_X_Ubar_Ustar = binscatter(x='Ubar', y='usr', data= full_df2, ci=(3,3), randcut=1)
df_binEstimate_full_3_X_Ubar_Ustar = binscatter(x='Ubar', y='usr', data= full_df3, ci=(3,3), randcut=1)
df_binEstimate_full_4_X_Ubar_Ustar = binscatter(x='Ubar', y='usr', data= full_df4, ci=(3,3), randcut=1)
print('done with binning data')

#%% plot BINNED z0 versus u*
import seaborn as sns
plt.figure(figsize=(10,5))
sns.scatterplot(x='Ubar', y='usr', data=df_binEstimate_full_3_X_Ubar_Ustar, color = 'darkorange', label = "binned s3")
plt.errorbar('Ubar', 'usr', yerr='ci', data=df_binEstimate_full_3_X_Ubar_Ustar, color = 'red', ls='', lw=2, alpha=0.2, label = 's3 error')
sns.scatterplot(x='Ubar', y='usr', data=df_binEstimate_full_2_X_Ubar_Ustar, color = 'olive', label = "binned s3")
plt.errorbar('Ubar', 'usr', yerr='ci', data=df_binEstimate_full_2_X_Ubar_Ustar, color = 'green', ls='', lw=2, alpha=0.2, label = 's2 error')
sns.scatterplot(x='Ubar', y='usr', data=df_binEstimate_full_1_X_Ubar_Ustar, color = 'navy', label = "binned s3")
plt.errorbar('Ubar', 'usr', yerr='ci', data=df_binEstimate_full_1_X_Ubar_Ustar, color = 'blue', ls='', lw=2, alpha=0.2, label = 's1 error')

plt.legend()
plt.title('Binned Wind Speed ($\overline{u}$) versus Friction Velocity ($u_*$)')
plt.xlabel('$u_*$ [$ms^{-1}$]')
plt.ylabel('$\overline{u}$ [$ms^{-1}$]')
# plt.xlim(0,1.05)
# plt.ylim(-0.0001,0.005)

plt.savefig(plot_save_path + "scatterBIN_Ubar_v_uStar.png",dpi=300)
plt.savefig(plot_save_path + "scatterBIN_Ubar_v_uStar.pdf")
print('done with plot')
