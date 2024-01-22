# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:36:45 2023

@author: oak
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:54:35 2022

@author: oaklin keefe

NOTE: this file needs to be run on the remote desktop.

This is Level 0 pipeline: taking quality controlled Level00 data, making sure there is enough
'good' data, aligning it to the wind (for applicable sensors) then despiking it (aka getting 
rid of the outliers), and finally interpolating it to the correct sensor sampling frequency. 
Edited files are saved to the Level 1 folder, in their respective "port" sub-folder as .csv 
files.                                                                                          

INPUT files:
    .txt files per 20 min period per port from Level1_errorLinesRemoved and sub-port folder
OUPUT files:
    .csv files per 20 min period per port into LEVEL_1 folder
    wind has been aligned to mean wind direction
    files have been despiked and interpolated to all be the same size
    all units the same as the input raw units
    
    
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
# Function for interpolating the paros sensor (freq = 16 Hz)
def interp_paros(df_paros):
    paros_xnew = np.arange(0, (16*60*20))   # this will be the number of points per file based
    df_paros_interp = df_paros.reindex(paros_xnew).interpolate(limit_direction='both')
    return df_paros_interp
#######################################################################################
### function end
# returns: df_paros_interp
print('done with interp_paros function')

#%%
### function start
#######################################################################################
def despikeThis(input_df,n_std):
    n = input_df.shape[1]
    output_df = pd.DataFrame()
    for i in range(0,n):
        elements_input = input_df.iloc[:,i]
        elements = elements_input
        mean = np.mean(elements)
        sd = np.std(elements)
        extremes = np.abs(elements-mean)>(n_std*sd)
        elements[extremes]=np.NaN
        despiked = np.array(elements)
        colname = input_df.columns[i]
        output_df[str(colname)]=despiked

    return output_df
#######################################################################################
### function end
# returns: output_df
print('done with despike_this function')
#%%
# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filename_generalDate = []
filename_port1 = []
filename_port2 = []
filename_port3 = []
filename_port4 = []
filename_port5 = []
filename_port6 = []
filename_port7 = []
for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
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
#%% THIS CODE ALIGNS, INTERPOLATES, THEN DESPIKES THE RAW DATA W/REMOVED ERR LINES
# filepath= r"E:\ASIT-research\BB-ASIT\test_Level1_errorLinesRemoved"
# filepath= r"E:\ASIT-research\BB-ASIT\Level1_errorLinesRemoved"

start=datetime.datetime.now()


# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
# path_save = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
path_save = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
    
        
        if filename.startswith('mNode_Port6'):
            # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port6/"
            s6_df = pd.read_csv(file,index_col=None, header = None) #read into a df
            s6_df.columns =['sensor','p'] #rename columns            
            s6_df= s6_df[s6_df['sensor'] != 0] #get rid of any rows where the sensor is 0 because this is an error row
            s6_df_1 = s6_df[s6_df['sensor'] == 1] # make a df just for sensor 1
            s6_df_2 = s6_df[s6_df['sensor'] == 2] # make a df just for sensor 2
            s6_df_3 = s6_df[s6_df['sensor'] == 3] # make a df just for sensor 3
            df_despiked_1 = pd.DataFrame()
            df_despiked_2 = pd.DataFrame()
            df_despiked_3 = pd.DataFrame()
            if len(s6_df_1)>= (0.75*(16*60*20)): #making sure there is at least 75% of a complete file before interpolating
                s6_df_1.loc[((s6_df_1['p']>=2000)|(s6_df_1['p']<=100)) , 'p'] = np.nan #PUT RESTRICTION ON REASONABLE PRESSURE OBSERVATIONS (hPa)
                # don't worry about the "warning" it gives when doing this ^
                df_despiked_1 = despikeThis(s6_df_1,5)
                df_paros_interp = interp_paros(df_despiked_1) #interpolate to proper frequency
                s6_df_1_interp = df_paros_interp #rename                
            else: #if not enough points, make a df of NaNs that is the size of a properly interpolated df
                s6_df_1_interp = pd.DataFrame(np.nan, index=[0,1], columns=['sensor','p']) 
            s6_df_1_interp.to_csv(path_save+"L1_"+str(filename_only)+'_1.csv') #save as csv
            print('done with paros 1 '+filename)
            if len(s6_df_2)>= (0.75*(16*60*20)): #making sure there is at least 75% of a complete file before interpolating
                s6_df_2.loc[((s6_df_2['p']>=2000)|(s6_df_2['p']<=100)) , 'p'] = np.nan    #PUT RESTRICTION ON REASONABLE PRESSURE OBSERVATIONS (hPa) 
                # don't worry about the "warning" it gives when doing this ^
                df_despiked_2 = despikeThis(s6_df_2,5)
                df_paros_interp = interp_paros(df_despiked_2) #interpolate to proper frequency
                s6_df_2_interp = df_paros_interp #rename                
            else: #if not enough points, make a df of NaNs that is the size of a properly interpolated df
                s6_df_2_interp = pd.DataFrame(np.nan, index=[0,1], columns=['sensor','p'])
            s6_df_2_interp.to_csv(path_save+"L2_"+str(filename_only)+'_1.csv') #save as csv
            print('done with paros 2 '+filename)
            if len(s6_df_3)>= (0.75*(16*60*20)): #making sure there is at least 75% of a complete file before interpolating
                s6_df_3.loc[((s6_df_3['p']>=2000)|(s6_df_3['p']<=100)) , 'p'] = np.nan  #PUT RESTRICTION ON REASONABLE PRESSURE OBSERVATIONS (hPa)
                # don't worry about the "warning" it gives when doing this ^
                df_despiked_3 = despikeThis(s6_df_3,5)
                df_paros_interp = interp_paros(df_despiked_3) #interpolate to proper frequency
                s6_df_3_interp = df_paros_interp #rename                
            else: #if not enough points, make a df of NaNs that is the size of a properly interpolated df
                s6_df_3_interp = pd.DataFrame(np.nan, index=[0,1], columns=['sensor','p'])
            s6_df_3_interp.to_csv(path_save+"L3_"+str(filename_only)+'_1.csv') #save as csv
            print('done with paros '+filename)

        
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

print('done with level0_p6 fully')
