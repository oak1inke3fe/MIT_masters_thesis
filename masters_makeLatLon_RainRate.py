#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:31:02 2023

@author: oaklin keefe



This file is used to ccreate an empty df for rainrate that is used in COARE, and to create a file full of the 1 lat,lon coordinate
location of the tower, which is also used in COARE.

Input file location:
    code_pipeline/Level2/
INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv (this is used to just get the length of the dataset)
    break_index = 3959 (this is the index of the last point in the spring deployment; used for breaking the dataset into spring and fall)

Output file location:
    code_pipeline/Level2/    
OUTPUT files:
    rain_rate_allSpring.csv
    rain_rate_allFall.csv
    rain_rate_combinedAnalysis.csv
    
    LatLon_allSpring.csv
    LatLon_allFall.csv
    LatLon_combinedAnalysis.csv


"""

#%% IMPORTS
import numpy as np
import pandas as pd
print('done with imports')

#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
path_save = file_path
sonic1_df = pd.read_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
break_index = 3959

#%% Code for Rain Rate blank (zeros) file
rain_df_spring = pd.DataFrame()
rain_rate_spring = np.zeros(len(sonic1_df[:break_index+1]))
rain_df_spring['rain_rate'] = rain_rate_spring
rain_df_spring.to_csv(path_save+'rain_rate_allSpring.csv')

rain_df_fall = pd.DataFrame()
rain_rate_fall = np.zeros(len(sonic1_df[break_index+1:]))
rain_df_fall['rain_rate'] = rain_rate_fall
rain_df_fall.to_csv(path_save+'rain_rate_allFall.csv')

rain_df_combined = pd.DataFrame()
rain_rate_combined = np.zeros(len(sonic1_df))
rain_df_combined['rain_rate'] = rain_rate_combined
rain_df_combined.to_csv(path_save+'rain_rate_combinedAnalysis.csv')

#%% Code for Lat/Lon repeated file
LatLon_df_spring = pd.DataFrame()
lat_spring = np.ones(len(sonic1_df[:break_index+1]))*41.57770 #degN
lon_spring = np.ones(len(sonic1_df[:break_index+1]))*-70.74611 #degE (+70.74611 for degW)
LatLon_df_spring['lat'] = lat_spring
LatLon_df_spring['lon'] = lon_spring
LatLon_df_spring.to_csv(path_save+'LatLon_allSpring.csv')

LatLon_df_fall = pd.DataFrame()
lat_fall = np.ones(len(sonic1_df[break_index+1:]))*41.57770 #degN
lon_fall = np.ones(len(sonic1_df[break_index+1:]))*-70.74611 #degE (+70.74611 for degW)
LatLon_df_fall['lat'] = lat_fall
LatLon_df_fall['lon'] = lon_fall
LatLon_df_fall.to_csv(path_save+'LatLon_allFall.csv')

LatLon_df_combined = pd.DataFrame()
lat_combined = np.ones(len(sonic1_df))*41.57770 #degN
lon_combined = np.ones(len(sonic1_df))*-70.74611 #degE (+70.74611 for degW)
LatLon_df_combined['lat'] = lat_combined
LatLon_df_combined['lon'] = lon_combined
LatLon_df_combined.to_csv(path_save+'LatLon_combinedAnalysis.csv')