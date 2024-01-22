# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:25:22 2023

@author: oaklin keefe



This file is used to calculate rate of TKE production using the direct covariance observational method.

Two different "methods", just manual ways of calculating the direct covariance, are given.
Both produce the same answer.

INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    zAvg_fromCTD_allSpring.csv
    zAvg_fromCTD_allFall.csv

    
OUTPUT files:
    Ubar_CombinedAnalysis.csv
    UpWp_bar_CombinedAnalysis.csv
    UpWp_bar_Ubar_combinedAnalysis.csv
    prodTerm_combinedAnalysis.csv
    
    
"""


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


print('done with imports')
#%%

# file_path = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level4/"
# file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'

file_sonic1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_mean_df = pd.read_csv(file_path+file_sonic1)
sonic1_mean_df = sonic1_mean_df.drop(['new_index'], axis=1)


file_sonic2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_mean_df = pd.read_csv(file_path+file_sonic2)
sonic2_mean_df = sonic2_mean_df.drop(['new_index'], axis=1)


file_sonic3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_mean_df = pd.read_csv(file_path+file_sonic3)
sonic3_mean_df = sonic3_mean_df.drop(['new_index'], axis=1)

file_sonic4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_mean_df = pd.read_csv(file_path+file_sonic4)
sonic4_mean_df = sonic4_mean_df.drop(['new_index'], axis=1)

#%% Plots of wind speed during each deployment
plt.figure()
plt.plot(sonic1_mean_df['Ubar'], label = 's1')
plt.plot(sonic2_mean_df['Ubar'], label = 's2')
plt.plot(sonic3_mean_df['Ubar'], label = 's3')
plt.plot(sonic4_mean_df['Ubar'], label = 's4')
plt.legend()
plt.title('Ubar Spring')
plt.ylabel('Ubar m/s')
plt.xlabel('"time"')
plt.xlim(0,3959)

plt.figure()
plt.plot(sonic1_mean_df['Ubar'], label = 's1')
plt.plot(sonic2_mean_df['Ubar'], label = 's2')
plt.plot(sonic3_mean_df['Ubar'], label = 's3')
plt.plot(sonic4_mean_df['Ubar'], label = 's4')
plt.legend()
plt.title('Ubar Fall')
plt.ylabel('Ubar m/s')
plt.xlabel('"time"')
plt.xlim(3960,8279)
#%% Method 1 of calculating production:
dz_LI_spring = 2.695  #sonic 2- sonic 1: spring APRIL 2022 deployment
dz_LII_spring = 2.795 #sonic 3- sonic 2: spring APRIL 2022 deployment
dz_LIII_spring = 2.415 #sonic 4- sonic 3: spring APRIL 2022 deployment
dz_LI_fall = 1.8161  #sonic 2- sonic 1: FALL SEPT 2022 deployment
dz_LII_fall = 3.2131 #sonic 3- sonic 2: FALL SEPT 2022 deployment
dz_LIII_fall = 2.468 #sonic 4- sonic 3: FALL SEPT 2022 deployment

break_index = 3959 #index is 3959, full length is 3960


prod_LI_new_spring = -1*((np.array(sonic1_mean_df['UpWp_bar'][:break_index+1])+np.array(sonic2_mean_df['UpWp_bar'][:break_index+1]))/2)*((np.array(sonic2_mean_df['Ubar'][:break_index+1])-np.array(sonic1_mean_df['Ubar'][:break_index+1]))/dz_LI_spring)
prod_LI_new_fall = -1*((np.array(sonic1_mean_df['UpWp_bar'][break_index+1:])+np.array(sonic2_mean_df['UpWp_bar'][break_index+1:]))/2)*((np.array(sonic2_mean_df['Ubar'][break_index+1:])-np.array(sonic1_mean_df['Ubar'][break_index+1:]))/dz_LI_fall)
prod_LII_new_spring = -1*((np.array(sonic2_mean_df['UpWp_bar'][:break_index+1])+np.array(sonic3_mean_df['UpWp_bar'][:break_index+1]))/2)*((np.array(sonic3_mean_df['Ubar'][:break_index+1])-np.array(sonic2_mean_df['Ubar'][:break_index+1]))/dz_LII_spring)
prod_LII_new_fall = -1*((np.array(sonic2_mean_df['UpWp_bar'][break_index+1:])+np.array(sonic3_mean_df['UpWp_bar'][break_index+1:]))/2)*((np.array(sonic3_mean_df['Ubar'][break_index+1:])-np.array(sonic2_mean_df['Ubar'][break_index+1:]))/dz_LII_fall)
prod_LIII_new_spring = -1*((np.array(sonic3_mean_df['UpWp_bar'][:break_index+1])+np.array(sonic4_mean_df['UpWp_bar'][:break_index+1]))/2)*((np.array(sonic4_mean_df['Ubar'][:break_index+1])-np.array(sonic3_mean_df['Ubar'][:break_index+1]))/dz_LIII_spring)
prod_LIII_new_fall = -1*((np.array(sonic3_mean_df['UpWp_bar'][break_index+1:])+np.array(sonic4_mean_df['UpWp_bar'][break_index+1:]))/2)*((np.array(sonic4_mean_df['Ubar'][break_index+1:])-np.array(sonic3_mean_df['Ubar'][break_index+1:]))/dz_LIII_fall)



prod_LI_new = np.append(prod_LI_new_spring, prod_LI_new_fall)
prod_LII_new = np.append(prod_LII_new_spring, prod_LII_new_fall)
prod_LIII_new = np.append(prod_LIII_new_spring, prod_LIII_new_fall)
prod_newDF = pd.DataFrame()
prod_newDF['prod_I'] = prod_LI_new
prod_newDF['prod_II'] = prod_LII_new
prod_newDF['prod_III'] = prod_LIII_new

prod_newDF.to_csv(file_path+"prodTerm_combinedAnalysis.csv")

# plot result
plt.figure()
# plt.plot(sonic4_mean_df['Ubar']/10, label = '<u>/10', color = 'black')
plt.plot(prod_newDF['prod_III'], label = 'III', color = 'green')
plt.plot(prod_newDF['prod_II'], label = 'II', color = 'darkorange')
plt.plot(prod_newDF['prod_I'], label = 'I', color = 'blue')
# plt.vlines(x=break_index, ymin=-1, ymax=1, color = 'k')
plt.legend()
plt.ylim(-0.5,0.5)
plt.title('Production $m^2s^{-3}$ Combined')


#%% Method 2 of calculating production. 
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
s1_turbTerms_Df = pd.read_csv(file_path + "despiked_s1_turbulenceTerms_andMore_combined.csv")
s1_turbTerms_Df = s1_turbTerms_Df.drop(['new_index'], axis=1)
s2_turbTerms_Df = pd.read_csv(file_path + "despiked_s2_turbulenceTerms_andMore_combined.csv")
s2_turbTerms_Df = s2_turbTerms_Df.drop(['new_index'], axis=1)
s3_turbTerms_Df = pd.read_csv(file_path + "despiked_s3_turbulenceTerms_andMore_combined.csv")
s3_turbTerms_Df = s3_turbTerms_Df.drop(['new_index'], axis=1)
s4_turbTerms_Df = pd.read_csv(file_path + "despiked_s4_turbulenceTerms_andMore_combined.csv")
s4_turbTerms_Df = s4_turbTerms_Df.drop(['new_index'], axis=1)

print(s1_turbTerms_Df.columns)
print(s1_turbTerms_Df.head())
#%%
Ubar_df = pd.DataFrame()
Ubar_df['Ubar_s1'] = s1_turbTerms_Df['Ubar']
Ubar_df['Ubar_s2'] = s2_turbTerms_Df['Ubar']
Ubar_df['Ubar_s3'] = s3_turbTerms_Df['Ubar']
Ubar_df['Ubar_s4'] = s4_turbTerms_Df['Ubar']

Ubar_df.to_csv(file_path + 'Ubar_combinedAnalysis.csv')

print('done with Ubar')

#%%
UpWp_bar_df = pd.DataFrame()
UpWp_bar_df['UpWp_bar_s1'] = s1_turbTerms_Df['UpWp_bar']
UpWp_bar_df['UpWp_bar_s2'] = s2_turbTerms_Df['UpWp_bar']
UpWp_bar_df['UpWp_bar_s3'] = s3_turbTerms_Df['UpWp_bar']
UpWp_bar_df['UpWp_bar_s4'] = s4_turbTerms_Df['UpWp_bar']

UpWp_bar_df.to_csv(file_path + 'UpWp_bar_combinedAnalysis.csv')

plt.figure()
UpWp_bar_df.plot()

print('done with UpWp_bar')

#%%

zAvg_df_spring = pd.read_csv(file_path+"zAvg_fromCTD_allSpring.csv") #20 min file averages from CTD for all the spring
zAvg_df_spring = zAvg_df_spring.drop('Unnamed: 0', axis=1)
zAvg_df_fall = pd.read_csv(file_path+"zAvg_fromCTD_allFall.csv") #20 min file averages from CTD for all the fall
zAvg_df_fall = zAvg_df_fall.drop('Unnamed: 0', axis=1)

print(zAvg_df_spring.columns)
# print(z_df_spring.head())
print(zAvg_df_fall.columns)
# print(z_df_fall.head())

plt.figure()
zAvg_df_spring.plot()
plt.title('Spring z-heights (average)')

plt.figure()
zAvg_df_fall.plot()
plt.title('Fall z-heights (average)')


z_df = pd.concat([zAvg_df_spring, zAvg_df_fall], axis=0)
print(z_df.columns)
print(z_df.index)
z_df['combined_index'] = np.arange(len(z_df))

z_df = z_df.set_index('combined_index')
print(z_df.index)
print(z_df.columns)
#%%

z1 = z_df['z_s1_avg']
z2 = z_df['z_s2_avg']
z3 = z_df['z_s3_avg']
z4 = z_df['z_s4_avg']

Ubar_s1 = Ubar_df['Ubar_s1']
Ubar_s2 = Ubar_df['Ubar_s2']
Ubar_s3 = Ubar_df['Ubar_s3']
Ubar_s4 = Ubar_df['Ubar_s4']


UpWp_bar_s1 = UpWp_bar_df['UpWp_bar_s1']
UpWp_bar_s2 = UpWp_bar_df['UpWp_bar_s2']
UpWp_bar_s3 = UpWp_bar_df['UpWp_bar_s3']
UpWp_bar_s4 = UpWp_bar_df['UpWp_bar_s4']


date_arr = np.array(z_df['date'])


UpWp_bar_Ubar = pd.DataFrame()
UpWp_bar_Ubar['date'] = date_arr
UpWp_bar_Ubar['UpWp_bar_Ubar_sonic1'] = UpWp_bar_s1*Ubar_s1
UpWp_bar_Ubar['UpWp_bar_Ubar_sonic2'] = UpWp_bar_s2*Ubar_s2
UpWp_bar_Ubar['UpWp_bar_Ubar_sonic3'] = UpWp_bar_s3*Ubar_s3
UpWp_bar_Ubar['UpWp_bar_Ubar_sonic4'] = UpWp_bar_s4*Ubar_s4
UpWp_bar_Ubar['z_s1'] = z1
UpWp_bar_Ubar['z_s2'] = z2
UpWp_bar_Ubar['z_s3'] = z3
UpWp_bar_Ubar['z_s4'] = z4

UpWp_bar_Ubar.to_csv(file_path+"UpWp_bar_Ubar_combinedAnalysis.csv")
print('done making UpWp_bar_Ubar_combined file')
#%%
dz_LI_spring = 2.695  #sonic 2- sonic 1: spring APRIL 2022 deployment
dz_LII_spring = 2.795 #sonic 3- sonic 2: spring APRIL 2022 deployment
dz_LIII_spring = 2.415 #sonic 4- sonic 3: spring APRIL 2022 deployment
dz_LI_fall = 1.8161  #sonic 2- sonic 1: FALL SEPT 2022 deployment
dz_LII_fall = 3.2131 #sonic 3- sonic 2: FALL SEPT 2022 deployment
dz_LIII_fall = 2.468 #sonic 4- sonic 3: FALL SEPT 2022 deployment

dUbar_dz_LI_arr = []
for i in range(0, break_index+1):
    dUbar_dz_LI_iSpring = (np.array(Ubar_s2)[i]-np.array(Ubar_s1)[i])/dz_LI_spring
    dUbar_dz_LI_arr.append(dUbar_dz_LI_iSpring)
print(len(dUbar_dz_LI_arr))   

for i in range(break_index+1, len(date_arr)):
    dUbar_dz_LI_iFall = (np.array(Ubar_s2)[i]-np.array(Ubar_s1)[i])/dz_LI_fall
    dUbar_dz_LI_arr.append(dUbar_dz_LI_iFall)
print(len(dUbar_dz_LI_arr)) 

  
dUbar_dz_LII_arr = []
for i in range(0, break_index+1):
    dUbar_dz_LII_iSpring = (np.array(Ubar_s3)[i]-np.array(Ubar_s2)[i])/dz_LII_spring
    dUbar_dz_LII_arr.append(dUbar_dz_LII_iSpring)
print(len(dUbar_dz_LII_arr))   

for i in range(break_index+1, len(date_arr)):
    dUbar_dz_LII_iFall = (np.array(Ubar_s3)[i]-np.array(Ubar_s2)[i])/dz_LII_fall
    dUbar_dz_LII_arr.append(dUbar_dz_LII_iFall)
print(len(dUbar_dz_LII_arr)) 


dUbar_dz_LIII_arr = []
for i in range(0, break_index+1):
    dUbar_dz_LIII_iSpring = (np.array(Ubar_s4)[i]-np.array(Ubar_s3)[i])/dz_LIII_spring
    dUbar_dz_LIII_arr.append(dUbar_dz_LIII_iSpring)
print(len(dUbar_dz_LIII_arr))   

for i in range(break_index+1, len(date_arr)):
    dUbar_dz_LIII_iFall = (np.array(Ubar_s4)[i]-np.array(Ubar_s3)[i])/dz_LIII_fall
    dUbar_dz_LIII_arr.append(dUbar_dz_LIII_iFall)
print(len(dUbar_dz_LIII_arr))  
 
print('done with d/dz combined')

#%%
prod_I = -1*((np.array(UpWp_bar_s1)+np.array(UpWp_bar_s2))/2)*dUbar_dz_LI_arr
prod_II = -1*((np.array(UpWp_bar_s2)+np.array(UpWp_bar_s3))/2)*dUbar_dz_LII_arr
prod_III = -1*((np.array(UpWp_bar_s3)+np.array(UpWp_bar_s4))/2)*dUbar_dz_LIII_arr 

prod_df = pd.DataFrame()
prod_df['date'] = date_arr
prod_df['prod_I'] = prod_I
prod_df['prod_II'] = prod_II
prod_df['prod_III'] = prod_III        

prod_df.to_csv(file_path+'prodTerm_combinedAnalysis.csv')
print('done with production')

