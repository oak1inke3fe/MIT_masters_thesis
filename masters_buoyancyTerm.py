#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:37:44 2023

@author: oaklin keefe


This file is used to caculate the buoyancy term in the TKE budget equation given initial inputs of the variables
included in the term.

Input file location:
    code_pipeline/Level2/
These initial variables (<w'T'>, <T>, density (rho)  are found in the INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    z_air_side_combinedAnalysis.csv
    rhoAvg_CombinedAnalysis.csv
We also set:
    g= 9.81 [m/s^2] #gravitational acceleration
    cp = 1004.67 [J/kg/K] #specific heat of dry air at constant pressure 
    
Output file location:
    code_pipeline/Level2/
The OUTPUT file is one files with all the buoyancy terms per sonic combined into one dataframe (saved as a .csv):
    buoy_terms_combinedAnalysis.csv

"""


#%%

import pandas as pd

print('done with imports')
#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'

sonic1_df = pd.read_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df = pd.read_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df = pd.read_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df = pd.read_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_combined.csv')

#%% Take variables from inputs files that we need

#make sure T is in Kelvin here
Tbar_s1 = sonic1_df['Tbar']
Tbar_s2 = sonic2_df['Tbar']
Tbar_s3 = sonic3_df['Tbar']
Tbar_s4 = sonic4_df['Tbar']

WpTp_bar_s1 = sonic1_df['WpTp_bar']
WpTp_bar_s2 = sonic2_df['WpTp_bar']
WpTp_bar_s3 = sonic3_df['WpTp_bar']
WpTp_bar_s4 = sonic4_df['WpTp_bar']

rho_df = pd.read_csv(file_path+"rhoAvg_CombinedAnalysis.csv")
rho_1 = rho_df['rho_bar_1']
rho_2 = rho_df['rho_bar_2']
rho_3 = rho_df['rho_bar_3']

g = 9.81
cp = 1004.67

#%%
#calculate buoyancy: g/<T>*<w'T'>
#calculate buoyancy flux: <w't'>*rho*cp

buoy_1 = g/(Tbar_s1)*WpTp_bar_s1
buoy_2 = g/(Tbar_s2)*WpTp_bar_s2
buoy_3 = g/(Tbar_s3)*WpTp_bar_s3
buoy_4 = g/(Tbar_s4)*WpTp_bar_s4
buoyFlux_1 = WpTp_bar_s1*rho_1*cp
buoyFlux_2 = WpTp_bar_s2*rho_2*cp
buoyFlux_3 = WpTp_bar_s3*rho_3*cp
buoyFlux_4 = WpTp_bar_s4*rho_3*cp
 

#%%
# add new terms to a dataframe to save it.
buoy_df = pd.DataFrame()
buoy_df['buoy_1'] = buoy_1
buoy_df['buoy_2'] = buoy_2
buoy_df['buoy_3'] = buoy_3
buoy_df['buoy_4'] = buoy_4
buoy_df['buoy_I'] = (buoy_1+buoy_2)/2     # for level I in between sonics 1 and 2
buoy_df['buoy_II'] = (buoy_2+buoy_3)/2    # for level II in between sonics 2 and 3
buoy_df['buoy_III'] = (buoy_3+buoy_4)/2   # for level III in between sonics 3 and 4
buoy_df['buoyFlux_1'] =  buoyFlux_1
buoy_df['buoyFlux_2'] =  buoyFlux_2
buoy_df['buoyFlux_3'] =  buoyFlux_3
buoy_df['buoyFlux_4'] =  buoyFlux_4
buoy_df['buoyFlux_I'] = (buoyFlux_1+buoyFlux_2)/2    # for level I in between sonics 1 and 2
buoy_df['buoyFlux_II'] = (buoyFlux_2+buoyFlux_3)/2   # for level II in between sonics 2 and 3
buoy_df['buoyFlux_III'] = (buoyFlux_3+buoyFlux_4)/2  # for level III in between sonics 3 and 4

print('done with buoyancy')


buoy_df.to_csv(file_path+"buoy_terms_combinedAnalysis.csv")

print('done saving to .csv')

