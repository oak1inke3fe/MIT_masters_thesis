# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:38:26 2023

@author: oak
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:54:35 2022

@author: oaklin keefe


This is Level 0 pipeline for the LiDAR recored on Port 7: 
In this file, we are taking quality controlled Level00 data, making sure there is enough 'good' 
data, and then interpolating to the correct sensor sampling frequency. Edited files are saved 
to the Level1_align-interp folder.

Input file location:
    code_pipeline/Level1_errorLinesRemoved
INPUT files:
    .txt files per 20 min period per instrument-port 

Output file location:
    code_pipeline/Level1_align-interp
OUPUT files:
    .csv files per 20 min period per instrument-port 
    wind has been aligned to mean wind direction (if applicable)
    files have been interpolated to all be the same size 
    
    

"""
#%%
import numpy as np
import pandas as pd
import os
import natsort
import datetime

print('done with imports')
#%%
### function start
#######################################################################################
# Function for interpolating the lidar sensor (freq = 20 Hz)
def interp_lidar(df_lidar):    
    lidar_xnew = np.arange(0, 20*60*20)   # this will be the number of points per file based
    s7_df_interp= df_lidar.reindex(lidar_xnew).interpolate(limit_direction='both')
    return s7_df_interp
#######################################################################################
### function end
# returns: s7_df_interp
print('done with interp_lidar function')

#%%

filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"

filename_generalDate = []
filename_port1 = []
filename_port2 = []
filename_port3 = []
filename_port4 = []
filename_port5 = []
filename_port6 = []
filename_port7 = []
for root, dirnames, filenames in os.walk(filepath): 
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        fiilename_general_name_only = filename[12:-4]
        if filename.startswith("mNode_Port1"):
            filename_port1.append(filename_only)
            filename_generalDate.append(fiilename_general_name_only)
        if filename.startswith("mNode_Port2"):
            filename_port2.append(filename_only)
        if filename.startswith("mNode_Port3"):
            filename_port3.append(filename_only)
        if filename.startswith("mNode_Port4"):
            filename_port4.append(filename_only)
        if filename.startswith("mNode_Port5"):
            filename_port5.append(filename_only)
        if filename.startswith("mNode_Port6"):
            filename_port6.append(filename_only)
        if filename.startswith("mNode_Port7"):
            filename_port7.append(filename_only)
        else:
            continue
print('port 1 length = '+ str(len(filename_port1)))
print('port 2 length = '+ str(len(filename_port2)))
print('port 3 length = '+ str(len(filename_port3)))
print('port 4 length = '+ str(len(filename_port4)))
print('port 5 length = '+ str(len(filename_port5)))
print('port 6 length = '+ str(len(filename_port6)))
print('port 7 length = '+ str(len(filename_port7)))
#%% THIS CODE ALIGNS, AND INTERPOLATES THE RAW DATA W/REMOVED ERR LINES


start=datetime.datetime.now()

len_dfOutput = []
port7_singleFile_nans_sum = []


filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"

path_save = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_align-interp/"
for root, dirnames, filenames in os.walk(filepath): 
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]

        port7_singleFile_nans = []
        if filename.startswith('mNode_Port7'):
            filename_only = filename[:-4]
            
            s7_df = pd.read_csv(file,index_col=None, header = None)
            s7_df.columns =['range','amplitude','quality']            
            if s7_df['range'].isna().sum()<(0.50*(1*60*20)): #make sure at least 50% of 20 minutes (at 1Hz frequency because of wave dropout) is recorded
                s7_df_interp = interp_lidar(s7_df) #interpolate to Lidar's sampling frequency
                port7_singleFile_nans_i=0
                port7_singleFile_nans.append(port7_singleFile_nans_i)
                
            else:
                s7_df_interp = pd.DataFrame(np.nan, index=[0,1], columns=['range','amplitude','quality'])
                port7_singleFile_nans_i = 1
                port7_singleFile_nans.append(port7_singleFile_nans_i)
            port7_singleFile_nans_sum_i = sum(port7_singleFile_nans)
            port7_singleFile_nans_sum.append(port7_singleFile_nans_i)
            s7_df_interp.to_csv(path_save+str(filename_only)+'_1.csv') #save as csv
            print('Port 7 ran for file: '+ filename)
            len_file = len(s7_df_interp)
            len_dfOutput.append(len_file)
        
        else:

            continue

end = datetime.datetime.now()
print('done')
print(start)
print(end)

# import winsound
# duration = 3000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)


print('done with level0_p7 fully')
