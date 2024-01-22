#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:20:34 2023

@author: oaklin keefe


This file is used to pad and/or restrict the PW dataset to the correct start/end dates of the spring/fall start and end dates

INPUT files:
    rhoAvg_CombinedAnalysis.csv
    date_combinedAnalysis.csv
    pw4oak.csv
    
OUTPUT files:
    pw_combinedAnalysis.csv

"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('done with imports')


#%%

file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
rho_df = pd.read_csv(file_path + 'rhoAvg_CombinedAnalysis.csv')
date_df = pd.read_csv(file_path + 'date_combinedAnalysis.csv')
spring_start = date_df['datetime'][0]
spring_end = date_df['datetime'][3959]
fall_start = date_df['datetime'][3960]
fall_end = date_df['datetime'][8279]
#%%
pw_file = "pw4oak.csv"
pw_df = pd.read_csv(file_path + pw_file)

pw_df = pw_df.drop(pw_df.index[0:216])
pw_df = pw_df.reset_index(drop=True)
print(pw_df)

print(pw_df.loc[3815])
print(pw_df.loc[4392])
#%%
pw_df = pw_df.drop(pw_df.index[3816:4392])
pw_df = pw_df.reset_index(drop=True)
print(pw_df)

print(date_df.loc[3600])
print(pw_df.loc[3600])
#%%
# 3599 + 2 full days (72*2 nan lines)
pw_Jun4thru6_pad_df = pd.DataFrame()
pw_Jun4thru6_pad_df['Year'] = np.full(72*2, np.nan)
pw_Jun4thru6_pad_df['Month'] = np.full(72*2, np.nan)
pw_Jun4thru6_pad_df['Day'] = np.full(72*2, np.nan)
pw_Jun4thru6_pad_df['Hour'] = np.full(72*2, np.nan)
pw_Jun4thru6_pad_df['Minute'] = np.full(72*2, np.nan)
pw_Jun4thru6_pad_df['PW boom-1 [m^3/s^3]'] = np.full(72*2, np.nan)
pw_Jun4thru6_pad_df['PW boom-2 [m^3/s^3]'] = np.full(72*2, np.nan)
pw_Jun4thru6_pad_df['PW boom-3 [m^3/s^3]'] = np.full(72*2, np.nan)

#%%
# Specify the index where you want to insert df2 into df1
insert_index = 3600

# Split df1 into two parts at the specified index
pw_df1_part1 = pw_df.loc[:insert_index - 1]
pw_df1_part2 = pw_df.loc[insert_index:]

# Concatenate df1_part1, df2, and df1_part2
expand1_pw_df = pd.concat([pw_df1_part1, pw_Jun4thru6_pad_df, pw_df1_part2], ignore_index=True)

# Optional: Reset index after concatenation
expand1_pw_df = expand1_pw_df.reset_index(drop=True)

# Print the resulting DataFrame
print(expand1_pw_df)

print(date_df.loc[3960])
print(expand1_pw_df.loc[3960])

#%%
print(date_df.loc[8279])
print(expand1_pw_df.loc[8279])
#%%
expand1_pw_df = expand1_pw_df.drop(expand1_pw_df.index[8280:])
expand1_pw_df = expand1_pw_df.reset_index(drop=True)
print(expand1_pw_df)

#%%
dz_LI_spring = 2.695  #sonic 2- sonic 1: spring APRIL 2022 deployment
dz_LII_spring = 2.795 #sonic 3- sonic 2: spring APRIL 2022 deployment
dz_LIII_spring = 2.415 #sonic 4- sonic 3: spring APRIL 2022 deployment
dz_LI_fall = 1.8161  #sonic 2- sonic 1: FALL SEPT 2022 deployment
dz_LII_fall = 3.2131 #sonic 3- sonic 2: FALL SEPT 2022 deployment
dz_LIII_fall = 2.468 #sonic 4- sonic 3: FALL SEPT 2022 deployment
break_index = 3959
# pw = (pw_boom2-pw_boom1)/(dzI)
# pw_dI_spring = np.array(rho_df['rho_bar_1_dry'][:break_index+1])*(np.array(expand1_pw_df['PW boom-2 [m^3/s^3]'][:break_index+1])-np.array(expand1_pw_df['PW boom-1 [m^3/s^3]'][:break_index+1]))/dz_LI_spring
# pw_dII_spring = np.array(rho_df['rho_bar_2_dry'][:break_index+1])*(np.array(expand1_pw_df['PW boom-3 [m^3/s^3]'][:break_index+1])-np.array(expand1_pw_df['PW boom-2 [m^3/s^3]'][:break_index+1]))/dz_LII_spring
# pw_dI_fall = np.array(rho_df['rho_bar_1_dry'][break_index+1:])*(np.array(expand1_pw_df['PW boom-2 [m^3/s^3]'][break_index+1:])-np.array(expand1_pw_df['PW boom-1 [m^3/s^3]'][break_index+1:]))/dz_LI_fall
# pw_dII_fall = np.array(rho_df['rho_bar_2_dry'][break_index+1:])*(np.array(expand1_pw_df['PW boom-3 [m^3/s^3]'][break_index+1:])-np.array(expand1_pw_df['PW boom-2 [m^3/s^3]'][break_index+1:]))/dz_LII_fall

pw_dI_spring = (np.array(expand1_pw_df['PW boom-2 [m^3/s^3]'][:break_index+1])-np.array(expand1_pw_df['PW boom-1 [m^3/s^3]'][:break_index+1]))/dz_LI_spring
pw_dII_spring = (np.array(expand1_pw_df['PW boom-3 [m^3/s^3]'][:break_index+1])-np.array(expand1_pw_df['PW boom-2 [m^3/s^3]'][:break_index+1]))/dz_LII_spring
pw_dI_fall = (np.array(expand1_pw_df['PW boom-2 [m^3/s^3]'][break_index+1:])-np.array(expand1_pw_df['PW boom-1 [m^3/s^3]'][break_index+1:]))/dz_LI_fall
pw_dII_fall = (np.array(expand1_pw_df['PW boom-3 [m^3/s^3]'][break_index+1:])-np.array(expand1_pw_df['PW boom-2 [m^3/s^3]'][break_index+1:]))/dz_LII_fall

#%%
pw_dI = np.concatenate([pw_dI_spring,pw_dI_fall],axis=0)
pw_dII = np.concatenate([pw_dII_spring,pw_dII_fall],axis=0)
#%%
expand1_pw_df['d_dz_pw_theory_I']=pw_dI
expand1_pw_df['d_dz_pw_theory_II']=pw_dII
expand1_pw_df['datetime']=date_df['datetime']
expand1_pw_df.to_csv(file_path + 'pw_combinedAnalysis.csv')
expand1_pw_df.columns
#%%
plt.figure()
plt.plot(pw_df['PW boom-1 [m^3/s^3]'])
# plt.xlim(2000,2500)