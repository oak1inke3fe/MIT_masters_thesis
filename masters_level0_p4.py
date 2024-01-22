# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:32:42 2023

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
import math

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
# filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_errorLinesRemoved/"
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

Ubar_s4_arr = []
Tbar_s4_arr = []
UpWp_bar_s4_arr = []
VpWp_bar_s4_arr = []
WpTp_bar_s4_arr = []
WpEp_bar_s4_arr = []
Umedian_s4_arr = []
Tmedian_s4_arr = []
U_horiz_bar_s4_arr = []
U_streamwise_bar_s4_arr = []
TKE_bar_s4_arr = []


# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
# filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_errorLinesRemoved/"
# path_save = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
# path_save = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
path_save = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-despike-interp/"
for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]

            
        if filename.startswith("mNode_Port4"):
            s4_df = pd.read_csv(file)
            if len(s4_df)>= (0.75*(20*60*20)):
                s4_df =pd.read_csv(file, index_col=None, header = None)
                s4_df.columns =['chk_1','chk_2','u', 'v', 'w', 'T', 'err_code']
                s4_df = s4_df[['u', 'v', 'w', 'T',]]                         
                s4_df['u']=s4_df['u'].astype(float) 
                s4_df['v']=s4_df['v'].astype(float) 
                df_s4_aligned = alignwind(s4_df)
                df_s4_aligned['Ur'][np.abs(df_s4_aligned['Ur']) >=55 ] = np.nan
                df_s4_aligned['Vr'][np.abs(df_s4_aligned['Vr']) >=20 ] = np.nan
                df_s4_aligned['Wr'][np.abs(df_s4_aligned['Wr']) >=20 ] = np.nan
                df_s4_interp = interp_sonics4(df_s4_aligned)
                U_horiz_s4 = np.sqrt((np.array(df_s4_interp['Ur'])**2)+(np.array(df_s4_interp['Vr'])**2))
                U_streamwise_s4 = np.sqrt((np.array(df_s4_interp['Ur'])**2)+(np.array(df_s4_interp['Vr'])**2)+(np.array(df_s4_interp['Wr'])**2))
                Up_s4 = df_s4_interp['Ur']-df_s4_interp['Ur'].mean()
                Vp_s4 = df_s4_interp['Vr']-df_s4_interp['Vr'].mean()
                Wp_s4 = df_s4_interp['Wr']-df_s4_interp['Wr'].mean()
                Tp_s4 = (df_s4_interp['T']+273.15)-(df_s4_interp['T']+273.15).mean()
                TKEp_s4 = 0.5*((Up_s4**2)+(Vp_s4**2)+(Wp_s4**2))
                Tbar_s4 = (df_s4_interp['T']+273.15).mean()
                UpWp_bar_s4 = np.nanmean(Up_s4*Wp_s4)
                VpWp_bar_s4 = np.nanmean(Vp_s4*Wp_s4)
                WpTp_bar_s4 = np.nanmean(Tp_s4*Wp_s4)
                WpEp_bar_s4 = np.nanmean(TKEp_s4*Wp_s4)
                Ubar_s4 = df_s4_interp['Ur'].mean()
                Umedian_s4 = df_s4_interp['Ur'].median()
                Tmedian_s4 = (df_s4_interp['T']+273.15).median()
                TKE_bar_s4 = np.nanmean(TKEp_s4)
                U_horiz_bar_s4 = np.nanmean(U_horiz_s4)
                U_streamwise_bar_s4 = np.nanmean(U_streamwise_s4)
            else:
                df_s4_interp = pd.DataFrame(np.nan, index=[0,1], columns=['base_index','Ur','Vr','Wr','T','u','v','w','alpha','beta'])
                Tbar_s4 = np.nan
                UpWp_bar_s4 = np.nan
                VpWp_bar_s4 = np.nan
                WpTp_bar_s4 = np.nan
                WpEp_bar_s4 = np.nan
                Ubar_s4 = np.nan
                Umedian_s4 = np.nan
                Tmedian_s4 = np.nan
                TKE_bar_s4 = np.nan
                U_horiz_bar_s4 = np.nan
                U_streamwise_bar_s4 = np.nan
                
            Tbar_s4_arr.append(Tbar_s4)
            UpWp_bar_s4_arr.append(UpWp_bar_s4)
            VpWp_bar_s4_arr.append(VpWp_bar_s4)
            WpTp_bar_s4_arr.append(WpTp_bar_s4)
            WpEp_bar_s4_arr.append(WpEp_bar_s4)
            Ubar_s4_arr.append(Ubar_s4)
            Umedian_s4_arr.append(Umedian_s4)
            Tmedian_s4_arr.append(Tmedian_s4)
            TKE_bar_s4_arr.append(TKE_bar_s4)
            U_horiz_bar_s4_arr.append(U_horiz_bar_s4)
            U_streamwise_bar_s4_arr.append(U_streamwise_bar_s4)
            df_s4_interp.to_csv(path_save+filename_only+"_1.csv")
            print('Port 4 ran: '+filename)
        
        
        else:
            # print("file doesn't start with mNode_Port 1-7")
            continue

end = datetime.datetime.now()
print('done')
print(start)
print(end)

# import winsound
# duration = 3000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)




#%%

# path_save_L4 = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level4/"
path_save_L4 = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/"

Ubar_s4_arr = np.array(Ubar_s4_arr)
U_horiz_s4_arr = np.array(U_horiz_bar_s4_arr)
U_streamwise_s4_arr = np.array(U_streamwise_bar_s4_arr)
Umedian_s4_arr = np.array(Umedian_s4_arr)
Tbar_s4_arr = np.array(Tbar_s4_arr)
Tmedian_s4_arr = np.array(Tmedian_s4_arr)
UpWp_bar_s4_arr = np.array(UpWp_bar_s4_arr)
VpWp_bar_s4_arr = np.array(VpWp_bar_s4_arr)
WpTp_bar_s4_arr = np.array(WpTp_bar_s4_arr)
WpEp_bar_s4_arr = np.array(WpEp_bar_s4_arr)
TKE_bar_s4_arr = np.array(TKE_bar_s4_arr)


combined_s4_df = pd.DataFrame()
combined_s4_df['Ubar_s4'] = Ubar_s4_arr
combined_s4_df['U_horiz_s4'] = U_horiz_s4_arr
combined_s4_df['U_streamwise_s4'] = U_streamwise_s4_arr
combined_s4_df['Umedian_s4'] = Umedian_s4_arr
combined_s4_df['Tbar_s4'] = Tbar_s4_arr
combined_s4_df['Tmedian_s4'] = Tmedian_s4_arr
combined_s4_df['UpWp_bar_s4'] = UpWp_bar_s4_arr
combined_s4_df['VpWp_bar_s4'] = VpWp_bar_s4_arr
combined_s4_df['WpTp_bar_s4'] = WpTp_bar_s4_arr
combined_s4_df['WpEp_bar_s4'] = WpEp_bar_s4_arr
combined_s4_df['TKE_bar_s4'] = TKE_bar_s4_arr

combined_s4_df.to_csv(path_save_L4 + "s4_turbulenceTerms_andMore_combined.csv")


print('done fully with level0_p4')
