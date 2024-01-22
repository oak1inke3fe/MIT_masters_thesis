#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:45:16 2023

@author: oak


This file is used to calculate the 20min file averages of rho (density) based on 20min file averages of temperature observations from the met/port5 sensor,
and 20min file averages of pressure observations from the high-res paros p sensor.

INPUT files:
    metAvg_CombinedAnalysis.csv
    parosAvg_combinedAnalysis.csv
    R = 287.053 #[Pa K^-1 m^3 kg^-1] # R is the ideal gas constant
    
OUTPUT files:
    rhoAvg_CombinedAnalysis.csv
    
"""
#%% IMPORTS
import pandas as pd
import numpy as np
print('done with imports')

#%%
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
met_df = pd.read_csv(file_path+"metAvg_CombinedAnalysis.csv")
p_df = pd.read_csv(file_path+"parosAvg_combinedAnalysis.csv")


#%%

# p from paros
# R_dry is the ideal gas constant for dry air
# R_moist is the gas constant for moist air
R_dry = 287.053 #[Pa K^-1 m^3 kg^-1]
R_waterVapor = 461 #[Pa K^-1 m^3 kg^-1]
RH = met_df['rh1'] #relative humidity as a percentage i.e. 87.95 = 87.95%
es = 611 * np.exp(17.27*(np.array(met_df['t1 [C]']))/(np.array(met_df['t1 [C]'])+237.3)) # saturation vapor pressure (emperical Magnus–Tetens formula)
Ws = 0.622*es/(p_df['p1_avg [Pa]']-es)# saturation mixing ratio
epsilon = RH/100*Ws
R_moist = R_dry*(1-epsilon)+(R_waterVapor*epsilon)
# T from RHT sensors
# rho = p/(R*T)

T_1 = met_df['t1 [K]']
T_2 = met_df['t2 [K]']

rho_bar_1_dry = p_df['p1_avg [Pa]']/(R_dry*T_1)
rho_bar_2_dry = p_df['p2_avg [Pa]']/(R_dry*T_1)
rho_bar_3_dry = p_df['p3_avg [Pa]']/(R_dry*T_2)
rho_bar_1_moist = p_df['p1_avg [Pa]']/(R_moist*T_1)
rho_bar_2_moist = p_df['p2_avg [Pa]']/(R_moist*T_1)
rho_bar_3_moist = p_df['p3_avg [Pa]']/(R_moist*T_2)

rho_df = pd.DataFrame()
rho_df['rho_bar_1_dry'] = rho_bar_1_dry
rho_df['rho_bar_2_dry'] = rho_bar_2_dry
rho_df['rho_bar_3_dry'] = rho_bar_3_dry
rho_df['rho_bar_1_moist'] = rho_bar_1_moist
rho_df['rho_bar_2_moist'] = rho_bar_2_moist
rho_df['rho_bar_3_moist'] = rho_bar_3_moist

rho_df.to_csv(file_path + 'rhoAvg_CombinedAnalysis.csv')


print('done saving rho avg. file')
