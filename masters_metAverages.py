#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:17:58 2022

@author: oaklin keefe


This file is used to calculate the 20min file averages of the port5 meteorological variables:
    temp, relative humidity, air pressure, shortwave radiation, longwave radiation

Input file location:
    /code_pipeline/Level1_align-interp/
INPUT files:
    port5/sonic 5 files

Output file location:
    /code_pipeline/Level2/
OUTPUT files:
    metAvg_CombinedAnalysis.csv
    
"""

#%% IMPORTS
import os
import numpy as np
import pandas as pd
import natsort

print('done with imports')
#%%
filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-interp/"
met_df_test = pd.read_csv(filepath + 'mNode_Port5_20220510_000000_1.csv')
print(met_df_test.columns)
#%%
t1_avg = []
t2_avg = []
rh1_avg = []
rh2_avg = []
p_air_avg = []
sw_dn_avg = [] #shortwave radiation (SW)
lw_dn_avg = [] #longwave radiation (IR)

for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-6]

        if filename.startswith('mNode_Port5'):
            met_df = pd.read_csv(file, index_col=None)
            t1 = met_df['T1'].mean()            
            t2 = met_df['T2'].mean()
            rh1 = met_df['RH1'].mean()
            rh2 = met_df['RH2'].mean()
            p_air = met_df['p_air'].mean()
            sw_dn = met_df['SW'].mean()     
            lw_dn= met_df['IRt'].mean()
            
            
            t1_avg.append(t1)
            t2_avg.append(t2)
            rh1_avg.append(rh1)
            rh2_avg.append(rh2)
            p_air_avg.append(p_air)
            sw_dn_avg.append(sw_dn)
            lw_dn_avg.append(lw_dn)
            print(filename_only)


#%%
    
t1_avg = np.array(t1_avg)
t2_avg = np.array(t2_avg)
rh1_avg = np.array(rh1_avg)
rh2_avg = np.array(rh2_avg)
p_air_avg = np.array(p_air_avg)
sw_dn_avg = np.array(sw_dn_avg)
lw_dn_avg = np.array(lw_dn_avg)
print('done with turning lists into arrays')
#%%
met_avg_df = pd.DataFrame()

met_avg_df['rh1']=rh1_avg
met_avg_df['rh2']=rh2_avg
met_avg_df['sw_dn']=sw_dn_avg
met_avg_df['lw_dn']=lw_dn_avg
met_avg_df['t1 [C]']=t1_avg
met_avg_df['t2 [C]']=t2_avg
met_avg_df['t1 [K]']=t1_avg+273.15      #convert to Kelvin
met_avg_df['t2 [K]']=t2_avg+273.15      #convert to Kelvin
met_avg_df['p_air [mb]']=p_air_avg
met_avg_df['p_air [Pa]']=p_air_avg/100  #convert to Pascals


print('done with creating dataframe of all averaged met variables')
#%% save this file where the other files we are pulling from will come from
path_save = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
met_avg_df.to_csv(path_save+"metAvg_CombinedAnalysis.csv")
print('file saved')