# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:13:37 2023

@author: oak
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:44:23 2023

@author: oak


This is Level 0 pipeline: taking quality controlled Level00 data, making sure there is enough
'good' data, aligning it to the wind (for applicable sensors) then despiking it (aka getting 
rid of the outliers), and finally interpolating it to the correct sensor sampling frequency. 
Edited files are saved to the Level 1 folder, in their respective "port" sub-folder as .csv 
files.                                                                                          

Input:
    .txt files per 20 min period per port from Level1_errorLinesRemoved and sub-port folder
Output:
    .csv files per 20 min period per port into LEVEL_1 folder
    wind has been aligned to mean wind direction
    files have been despiked and interpolated to all be the same size
    all units the same as the input raw units
    
    
"""
#%%
import numpy as np
import pandas as pd
# from pandas import rolling_median
import os
import matplotlib.pyplot as plt
import natsort
# import statistics
import time
import datetime
import math
from hampel import hampel
# from scipy import interpolate
# import re
# import scipy.signal as signal
# import pickle5 as pickle
# os.chdir(r'E:\mNode_test2folders\test')
print('done with imports')

#%%

### function start
#######################################################################################
# Function for aligning the U,V,W coordinaes to the mean wind direction
def alignwind(wind_df):
    # try: 
    wind_df = wind_df.replace('NAN', np.nan)
    wind_df['u'] = wind_df['u'].astype(float)
    wind_df['v'] = wind_df['v'].astype(float)
    wind_df['w'] = wind_df['w'].astype(float)
    Ub = wind_df['u'].mean()
    Vb = wind_df['v'].mean()
    Wb = wind_df['w'].mean()
    Sb = math.sqrt((Ub**2)+(Vb**2))
    beta = math.atan2(Wb,Sb)
    beta_arr = np.ones(len(wind_df))*(beta*180/math.pi)
    alpha = math.atan2(Vb,Ub)
    alpha_arr = np.ones(len(wind_df))*(alpha*180/math.pi)
    x1 = wind_df.index
    x = np.array(x1)
    Ur = wind_df['u']*math.cos(alpha)*math.cos(beta)+wind_df['v']*math.sin(alpha)*math.cos(beta)+wind_df['w']*math.sin(beta)
    Ur_arr = np.array(Ur)
    Vr = wind_df['u']*(-1)*math.sin(alpha)+wind_df['v']*math.cos(alpha)
    Vr_arr = np.array(Vr)
    Wr = wind_df['u']*(-1)*math.cos(alpha)*math.sin(beta)+wind_df['v']*(-1)*math.sin(alpha)*math.sin(beta)+wind_df['w']*math.cos(beta)     
    Wr_arr = np.array(Wr)
    T_arr = np.array(wind_df['T'])
    u_arr = np.array(wind_df['u'])
    v_arr = np.array(wind_df['v'])
    w_arr = np.array(wind_df['w'])

    df_aligned = pd.DataFrame({'base_index':x,'Ur':Ur_arr,'Vr':Vr_arr,'Wr':Wr_arr,'T':T_arr,'u':u_arr,'v':v_arr,'w':w_arr,'alpha':alpha_arr,'beta':beta_arr})

    return df_aligned
#######################################################################################
### function end
# returns: df_aligned (index, Ur, Vr, Wr, T, u, v, w, alpha, beta)
print('done with alignwind function')


#%%
### function start
#######################################################################################
# Function for interpolating the RMY sensor (freq = 32 Hz)
def interp_sonics123(df_sonics123):
    sonics123_xnew = np.arange(0, (32*60*20))   # this will be the number of points per file based
    df_align_interp= df_sonics123.reindex(sonics123_xnew).interpolate(limit_direction='both')
    return df_align_interp
#######################################################################################
### function end
# returns: df_align_interp
print('done with interp_sonics123 simple function')
#%%
### function start
#######################################################################################
# Function for interpolating the Gill sensor (freq = 20 Hz)
def interp_sonics4(df_sonics4):
    sonics4_xnew = np.arange(0, (20*60*20))   # this will be the number of points per file based
    df_align_interp_s4= df_sonics4.reindex(sonics4_xnew).interpolate(limit_direction='both')
    return df_align_interp_s4
#######################################################################################
### function end
# returns: df_align_interp_s4
print('done with interp_sonics4 function')
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
# Function for interpolating the sonics to the same frequency as the pressure heads (downsample to 16 Hz)
def interp_sonics2paros(df_despiked_sonics):
    sonic2paros_xnew = np.arange(0, 19200)   # this will be the number of points per file based
    df_sonic2paros_interp= df_despiked_sonics.reindex(sonic2paros_xnew).interpolate(limit_direction='both')
    return df_sonic2paros_interp
#######################################################################################
### function end
# returns: df_sonic2paros_interp
print('done with interp_sonic2paros function')
#%%
### function start
#######################################################################################
# Function for interpolating the sonics to the same frequency as the pressure heads (downsample to 16 Hz)
def interp_paros2met(df_paros):
    paros2met_xnew = np.arange(0, 1*60*20)   # this will be the number of points per file based
    df_paros2met_interp= df_paros.reindex(paros2met_xnew).interpolate(limit_direction='both')
    return df_paros2met_interp
#######################################################################################
### function end
# returns: df_sonic2paros_interp
print('done with interp_paros2met function')
#%%
### function start
#######################################################################################
# Function for interpolating the met sensor (freq = 1 Hz)
def interp_met(df_met):    
    met_xnew = np.arange(0, (1*60*20))   # this will be the number of points per file based
    s5_df_met_interp= df_met.reindex(met_xnew).interpolate(limit_direction='both')
    return s5_df_met_interp
#######################################################################################
### function end
# returns: s5_df_met_interp
print('done with interp_met function')
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
            if not filename.endswith("00.txt"):
                print(str(filename))
            else:
                continue
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
#%% THIS CODE ALIGNS, INTERPOLATES, THEN DESPIKES THE RAW DATA W/REMOVED ERR LINES
# filepath= r"E:\ASIT-research\BB-ASIT\test_Level1_errorLinesRemoved"
# filepath= r"E:\ASIT-research\BB-ASIT\Level1_errorLinesRemoved"

start=datetime.datetime.now()

Ubar_arr = []
Tbar_arr = []
UpWp_bar_arr = []
VpWp_bar_arr = []
WpTp_bar_arr = []
WpEp_bar_arr = []
Umedian_arr = []
Tmedian_arr = []
# U_horiz_arr = []
# U_streamwise_arr = []



# sonic_arr = ['1','2','3','4']
sonic_arr = ['4']

# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
# path_save = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp_errorLinesRemoved/"
path_save = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp_errorLinesRemoved/"
for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        for sonic in sonic_arr:
            sonic_int = int(sonic)
            if filename.startswith("mNode_Port"+sonic):
                step1 = 'True'
                print("step 1 = " + str(step1))
                sonic_df = pd.read_csv(file)
                if int(sonic) == 4:
                    fs = 20
                else:
                    fs = 32
                if len(sonic_df)>= (0.75*(fs*60*20)): #making sure there is at least 75% of a complete file before interpolating
                    step2 = 'True'
                    print("step 2 = " + str(step2))
                    print(filename+ " has sufficient data to run")
                    if int(sonic) == 4:
                        sonic_df.columns =['chk_1','chk_2','u', 'v', 'w', 'T', 'err_code']
                        # sonic_df = pd.to_numeric(sonic_df, errors='coerce')
                        err_code = np.array(pd.to_numeric(sonic_df['chk_1'], errors='coerce'))
                        err_code_new = np.where(err_code!=0, err_code, np.nan) #from gill manual
                        if np.isnan(err_code_new).sum() <= (0.1*len(err_code)):
                            step3 = 'True; sonic 4'
                            print("step 3 = " + str(step3))
                            mask_noErrorCode = np.isin(err_code_new,err_code)                        
                            sonic_df = sonic_df[mask_noErrorCode]
                        else: #if more than 10% of the file is bad, make it all nan
                            step3 = 'False'
                            print("step 3 = " + str(step3))
                            print(filename+ " more than 10% of this file is error")
                            sonic_df = pd.DataFrame(np.nan, index=[0,1], columns=['chk_1','chk_2','u', 'v', 'w', 'T', 'err_code'])
                            
                        
                    else:
                        sonic_df.columns =['u', 'v', 'w', 'T', 'err_code','chk_sum'] #set column names to the variable
                        # sonic_df = pd.to_numeric(sonic_df, errors='coerce')
                        err_code = np.array(pd.to_numeric(sonic_df['chk_sum'], errors='coerce'))
                        err_code_new = np.where(err_code==0, err_code, 1.0) #from RMY manual
                        if np.sum(err_code_new) <= (0.1*len(err_code)): #if more than 10% of the file is bad, make it all nan
                            step3 = 'True'
                            print("step 3 = " + str(step3))
                            mask_noErrorCode = np.isin(err_code_new,err_code)
                            sonic_df = sonic_df[mask_noErrorCode]
                        else:
                            step3 = 'False'
                            print("step 3 = " + str(step3))
                            print(filename+ " more than 10% of this file is error")
                            sonic_df = pd.DataFrame(np.nan, index=[0,1], columns=['u', 'v', 'w', 'T', 'err_code','chk_sum'])
                            
                    sonic_df = sonic_df[['u', 'v', 'w', 'T',]] 
                    
                    if len(sonic_df)>3: # won't do this part of the code if the file has more than 10% error lines
                        step4 = 'True'
                        print("step 4 = " + str(step4))
                        sonic_df['u']=sonic_df['u'].astype(float) 
                        sonic_df['v']=sonic_df['v'].astype(float)
                        sonic_df['w']=sonic_df['w'].astype(float)
                        if int(sonic) <= 2: #do this for the "upsidedown sonics
                            step5 = 'True, sonics 1 or 2'
                            print("step 5 = " + str(step5))
                            sonic_df['u']=-1*sonic_df['u']
                            sonic_df['w']=-1*sonic_df['w']
                        else:
                            step5 = 'False; sonic 3 or 4'
                            print("step 5 = " + str(step5))
                            sonic_df['u']=sonic_df['u']
                            sonic_df['w']=sonic_df['w']
                        df_aligned = alignwind(sonic_df)
                        df_aligned['Ur'][np.abs(df_aligned['Ur']) >=55 ] = np.nan
                        df_aligned['Vr'][np.abs(df_aligned['Vr']) >=20 ] = np.nan
                        df_aligned['Wr'][np.abs(df_aligned['Wr']) >=20 ] = np.nan
                        df_aligned['U_horiz'] = np.sqrt((np.array(df_aligned['Ur'])**2)+(np.array(df_aligned['Vr'])**2))
                        df_aligned['U_streamwise'] = np.sqrt((np.array(df_aligned['Ur'])**2)+(np.array(df_aligned['Vr'])**2)+(np.array(df_aligned['Wr'])**2))
                        if int(sonic) == 4:
                            step6 = 'True; sonic 4'
                            print("step 6 = " + str(step6))
                            df_interp = interp_sonics4(df_aligned)
                        else:
                            step6 = 'False; sonics 1-3'
                            print("step 6 = " + str(step6))
                            df_interp = interp_sonics123(df_aligned)
                        # Ur_s1 = df_s1_interp['Ur']
                        # Ur_s1_outlier_in_Ts = hampel(Ur_s1, window_size=10, n=3, imputation=True) # Outlier Imputation with rolling median
                        # df_s1_interp['Ur'] = Ur_s1_outlier_in_Ts
                        # Vr_s1 = df_s1_interp['Vr']
                        # Vr_s1_outlier_in_Ts = hampel(Vr_s1, window_size=10, n=3, imputation=True) # Outlier Imputation with rolling median
                        # df_s1_interp['Vr'] = Vr_s1_outlier_in_Ts
                        # Wr_s1 = df_s1_interp['Wr']
                        # Wr_s1_outlier_in_Ts = hampel(Wr_s1, window_size=10, n=3, imputation=True) # Outlier Imputation with rolling median
                        # df_s1_interp['Wr'] = Wr_s1_outlier_in_Ts
                        
                        # U_horiz = np.sqrt((np.array(df_interp['Ur'])**2)+(np.array(df_interp['Vr'])**2))
                        # U_streamwise = np.sqrt((np.array(df_interp['Ur'])**2)+(np.array(df_interp['Vr'])**2)+(np.array(df_interp['Wr'])**2))
                        Up = df_interp['Ur']-df_interp['Ur'].mean()
                        Vp = df_interp['Vr']-df_interp['Vr'].mean()
                        Wp = df_interp['Wr']-df_interp['Wr'].mean()
                        Tp = (df_interp['T']+273.15)-(df_interp['T']+273.15).mean()
                        TKEp = 0.5*((Up**2)+(Vp**2)+(Wp**2))                
                        Tbar = (df_interp['T']+273.15).mean()
                        UpWp_bar = np.mean(Up*Wp)
                        VpWp_bar = np.mean(Vp*Wp)
                        WpTp_bar = np.mean(Wp*Tp)
                        WpEp_bar = np.mean(Wp*TKEp)
                        Ubar = df_interp['Ur'].mean()
                        Umedian = df_interp['Ur'].median()
                        Tmedian = (df_interp['T']+273.15).median()
                        df_interp.to_csv(path_save+filename_only+"_1.csv")
                    else:
                        step4 = 'False'
                        print("step 4 = " + str(step4))
                        df_interp = pd.DataFrame(np.nan, index=[0,1], columns=['base_index','Ur','Vr','Wr','T','u','v','w','alpha','beta','U_horiz','U_streamwise'])
                        Tbar = np.nan
                        UpWp_bar = np.nan
                        VpWp_bar = np.nan
                        WpTp_bar = np.nan
                        WpEp_bar = np.nan
                        Ubar = np.nan
                        Umedian = np.nan
                        Tmedian = np.nan
                   
                else:
                    step2 = 'False'
                    print("step 2 = " + str(step2))
                    print(filename+ " does NOT have sufficient data to run")
                    df_interp = pd.DataFrame(np.nan, index=[0,1], columns=['base_index','Ur','Vr','Wr','T','u','v','w','alpha','beta','U_horiz','U_streamwise'])
                    Tbar = np.nan
                    UpWp_bar = np.nan
                    VpWp_bar = np.nan
                    WpTp_bar = np.nan
                    WpEp_bar = np.nan
                    Ubar = np.nan
                    Umedian = np.nan
                    Tmedian = np.nan
                    # U_horiz = np.nan
                    # U_streamwise = np.nan
                Tbar_arr.append(Tbar)
                UpWp_bar_arr.append(UpWp_bar)
                VpWp_bar_arr.append(VpWp_bar)
                WpTp_bar_arr.append(WpTp_bar)
                WpEp_bar_arr.append(WpEp_bar)
                Ubar_arr.append(Ubar)
                Umedian_arr.append(Umedian)
                Tmedian_arr.append(Tmedian)
                # U_horiz_arr.append(U_horiz)
                # U_streamwise_arr.append(U_streamwise)
                df_interp.to_csv(path_save+filename_only+"_1.csv")
                
                print('Port '+sonic+' ran: '+filename)
                ## BELOW ARE PLOTTING LINES IF YOU WANT TO CHECK OUTPUTS AS THE CODE RUNS
                ## CAN ADD BELOW OTHER PORTS TO CHECK THEIR OUTPUTS TOO
            #     plt.plot(df_aligned['Ur'], label="Ur")
            #     plt.plot(df_aligned['Vr'], label="Vr")
            #     plt.plot(df_aligned['Wr'], label="Wr")                         
            #     # plt.ylim(-4,10)
            #     plt.legend(loc='upper left',prop={'size': 6})
            #     plt.title(str(filename))
            #     plt.draw()
            #     plt.pause(0.0001)
            #     plt.clf()
            else:
                step1= False
                continue
#%%
mean_df = pd.DataFrame()
mean_df['Ubar'] = Ubar_arr
mean_df['Tbar'] = Tbar_arr
mean_df['UpWp_bar'] = UpWp_bar_arr
mean_df['VpWp_bar'] = VpWp_bar_arr
mean_df['WpTp_bar'] = WpTp_bar_arr
mean_df['WpEp_bar'] = WpEp_bar_arr
mean_df['Umedian'] = Umedian_arr
mean_df['Tmedian'] = Tmedian_arr

# file_save_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level4/"
mean_df.to_csv(file_save_path+"meanQuantities_sonic"+sonic+".csv")

end=datetime.datetime.now()
print('done')
print(start)
print(end)

import winsound
duration = 3000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#%%
