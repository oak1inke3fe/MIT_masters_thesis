

# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:42:19 2023

@author: oaklin keefe

This file is used to calculate the virtual potential temperature.

Input file location:
    /code_pipeline/Level2/
INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    parosAvg_combinedAnalysis.csv

Output file location:
    /code_pipeline/Level2/
OUTPUT files:
    thetaV_combinedAnalysis.csv
        
        
"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('done with imports')

#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic1_df = sonic1_df.drop(['new_index'], axis=1)

sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic2_df = sonic2_df.drop(['new_index'], axis=1)

sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)
sonic3_df = sonic3_df.drop(['new_index'], axis=1)

sonic_file4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)
sonic4_df = sonic4_df.drop(['new_index'], axis=1)

paros_df = pd.read_csv(file_path+"parosAvg_combinedAnalysis.csv")
paros_df = paros_df.drop(['Unnamed: 0'], axis=1)

#%%
p0 = 1000 #hPa (1mb = 1 hPa) #this is a reference pressure
R_over_cp = 0.286
Tv_df = pd.DataFrame()
Tv_s1 = sonic1_df['Tbar']*(p0/paros_df['p1_avg [mb]'])**R_over_cp
Tv_s2 = sonic2_df['Tbar']*(p0/paros_df['p2_avg [mb]'])**R_over_cp
Tv_s3 = sonic3_df['Tbar']*(p0/paros_df['p3_avg [mb]'])**R_over_cp
Tv_s4 = sonic4_df['Tbar']*(p0/paros_df['p3_avg [mb]'])**R_over_cp


Tv_df = pd.DataFrame()
Tv_df['thetaV_sonic1'] = np.array(Tv_s1)
Tv_df['thetaV_sonic2'] = np.array(Tv_s2)
Tv_df['thetaV_sonic3'] = np.array(Tv_s3)
Tv_df['thetaV_sonic4'] = np.array(Tv_s4)

plt.figure()
plt.plot(Tv_s4, label = 's4')
plt.plot(Tv_s3, label = 's3')
plt.plot(Tv_s2, label = 's2')
plt.plot(Tv_s1, label = 's1')
plt.legend()
plt.title('theta_V by sonic')


Tv_df.to_csv(file_path+"thetaV_combinedAnalysis.csv")