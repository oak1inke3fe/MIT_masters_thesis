#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:13:19 2023

@author: oaklin keefe

NOTE: this file needs to be run on the remote desktop.

This file is used to calculate the 20min file averages of the high resolution paros pressure sensors.

INPUT files:
    port6/ paros files from /Level1_align-despike-interp/ folder
    
OUTPUT files:
    parosAvg_combinedAnalysis.csv
    
"""
#%% IMPORTS

import os
import pandas as pd
import numpy as np
import natsort

print('done with imports')
#%%
# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level2_analysis\port6/"
filepath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-despike-interp/'
# save_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
save_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'

paros1_Pa_mean = []
paros1_mb_mean = []
paros2_Pa_mean = []
paros2_mb_mean = []
paros3_Pa_mean = []
paros3_mb_mean = []

test_file = pd.read_csv(filepath + "L1_mNode_Port6_20220509_000000_1.csv", header = 0)
print(test_file.head(10))
p_mb_test = np.nanmean(test_file['p'])
p_Pa_test = np.nanmean(test_file['p']*100)
print(p_mb_test)
print(p_Pa_test)
#%%
for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        if filename.startswith("L1"):
            paros_df = pd.read_csv(file,header = 0)
            p_mb = np.nanmean(paros_df['p'])
            p_Pa = np.nanmean(paros_df['p']*100) #convert to Pascals from millibars (hectopascals)
            paros1_Pa_mean.append(p_Pa)
            paros1_mb_mean.append(p_mb)
            print(filename)
        elif filename.startswith("L2"):
            paros_df = pd.read_csv(file,header = 0)
            p_mb = np.nanmean(paros_df['p'])
            p_Pa = np.nanmean(paros_df['p']*100) #convert to Pascals from millibars (hectopascals)
            paros2_Pa_mean.append(p_Pa)
            paros2_mb_mean.append(p_mb)
            print(filename)
        elif filename.startswith("L3"):
            paros_df = pd.read_csv(file,header = 0)
            p_mb = np.nanmean(paros_df['p'])
            p_Pa = np.nanmean(paros_df['p']*100) #convert to Pascals from millibars (hectopascals)
            paros3_Pa_mean.append(p_Pa)
            paros3_mb_mean.append(p_mb)
            print(filename)
        else:
            print('not port 6')
            
#%%
paros_df = pd.DataFrame()
paros_df['p1_avg [Pa]'] = paros1_Pa_mean
paros_df['p2_avg [Pa]'] = paros2_Pa_mean
paros_df['p3_avg [Pa]'] = paros3_Pa_mean
paros_df['p1_avg [mb]'] = paros1_mb_mean
paros_df['p2_avg [mb]'] = paros2_mb_mean
paros_df['p3_avg [mb]'] = paros3_mb_mean

paros_df.to_csv(save_path+"parosAvg_combinedAnalysis.csv")
print('done saving file')