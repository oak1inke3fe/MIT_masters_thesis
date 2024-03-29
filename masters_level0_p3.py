# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:31:46 2023

@author: oak
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:54:35 2022

@author: oaklin keefe

Part 1:
This is Level 0 pipeline for port 3 (sonic anemometer #3): taking quality controlled Level00 data, 
making sure there is enough 'good' data, aligning it to the wind (for applicable sensors) then despiking 
it (aka getting rid of the outliers), and finally interpolating it to the correct sensor sampling frequency. 
Edited files are saved to the Level1_align-interp folder as .csv files.                                                                                          

Input file location:
    code_pipeline/Level1_errorLinesRemoved
INPUT files:
    .txt files per 20 min period per instrument-port 

Output file location:
    Part 1: code_pipeline/Level1_align-interp
OUPUT files:
    .csv files per 20 min period per instrument-port 
    wind has been aligned to mean wind direction
    files have been interpolated to all be the same size
    all units the same as the input raw units

    
Part 2:
We take the edited (aligned, interpolated) mean data that we have saved as lists for each variable, and then we assign
them to a data frame. In this data frame, each line is the mean of a 20 minute period.
The dataframe is then saved to the Level2 folder as a .csv file.

Input file location:
    this code
INPUT files:
    lists from part 1

Output file location:
    code_pipeline/Level2
OUPUT files:
    s3_turbulenceTerms_andMore_combined.csv
    
    
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
#%% THIS CODE ALIGNS, AND INTERPOLATES THE RAW DATA W/REMOVED ERR LINES
'''
PART 1
'''

start=datetime.datetime.now()


Ubar_s3_arr = []
Tbar_s3_arr = []
UpWp_bar_s3_arr = []
VpWp_bar_s3_arr = []
WpTp_bar_s3_arr = []
WpEp_bar_s3_arr = []
Umedian_s3_arr = []
Tmedian_s3_arr = []
TKE_bar_s3_arr = []
U_horiz_bar_s3_arr = []
U_streamwise_bar_s3_arr = []


filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_errorLinesRemoved/"

path_save = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-interp/"
for root, dirnames, filenames in os.walk(filepath): 
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        
     
        if filename.startswith("mNode_Port3"):
            s3_df = pd.read_csv(file)
            if len(s3_df)>= (0.75*(32*60*20)):
                s3_df.columns =['u', 'v', 'w', 'T', 'err_code','chk_sum'] #set column names to the variable
                s3_df = s3_df[['u', 'v', 'w', 'T',]]            
                s3_df['u']=s3_df['u'].astype(float) 
                s3_df['v']=s3_df['v'].astype(float) 
                df_s3_aligned = alignwind(s3_df)
                df_s3_aligned['Ur'][np.abs(df_s3_aligned['Ur']) >=55 ] = np.nan
                df_s3_aligned['Vr'][np.abs(df_s3_aligned['Vr']) >=20 ] = np.nan
                df_s3_aligned['Wr'][np.abs(df_s3_aligned['Wr']) >=20 ] = np.nan
                df_s3_interp = interp_sonics123(df_s3_aligned)
                U_horiz_s3 = np.sqrt((np.array(df_s3_interp['Ur'])**2)+(np.array(df_s3_interp['Vr'])**2))
                U_streamwise_s3 = np.sqrt((np.array(df_s3_interp['Ur'])**2)+(np.array(df_s3_interp['Vr'])**2)+(np.array(df_s3_interp['Wr'])**2))
                Up_s3 = df_s3_interp['Ur']-df_s3_interp['Ur'].mean()
                Vp_s3 = df_s3_interp['Vr']-df_s3_interp['Vr'].mean()
                Wp_s3 = df_s3_interp['Wr']-df_s3_interp['Wr'].mean()
                Tp_s3 = (df_s3_interp['T']+273.15)-(df_s3_interp['T']+273.15).mean()
                TKEp_s3 = 0.5*((Up_s3**2)+(Vp_s3**2)+(Wp_s3**2))
                Tbar_s3 = (df_s3_interp['T']+273.15).mean()
                UpWp_bar_s3 = np.nanmean(Up_s3*Wp_s3)
                VpWp_bar_s3 = np.nanmean(Vp_s3*Wp_s3)
                WpTp_bar_s3 = np.nanmean(Tp_s3*Wp_s3)
                WpEp_bar_s3 = np.nanmean(TKEp_s3*Wp_s3)
                Ubar_s3 = df_s3_interp['Ur'].mean()
                Umedian_s3 = df_s3_interp['Ur'].median()
                Tmedian_s3 = (df_s3_interp['T']+273.15).median()
                TKE_bar_s3 = np.nanmean(TKEp_s3)
                U_horiz_bar_s3 = np.nanmean(U_horiz_s3)
                U_streamwise_bar_s3 = np.nanmean(U_streamwise_s3)
            else:
                df_s3_interp = pd.DataFrame(np.nan, index=[0,1], columns=['base_index','Ur','Vr','Wr','T','u','v','w','alpha','beta'])
                Tbar_s3 = np.nan
                UpWp_bar_s3 = np.nan
                VpWp_bar_s3 = np.nan
                WpTp_bar_s3 = np.nan
                WpEp_bar_s3 = np.nan
                Ubar_s3 = np.nan
                Umedian_s3 = np.nan
                Tmedian_s3 = np.nan
                TKE_bar_s3 = np.nan
                U_horiz_bar_s3 = np.nan
                U_streamwise_bar_s3 = np.nan
                
            Tbar_s3_arr.append(Tbar_s3)
            UpWp_bar_s3_arr.append(UpWp_bar_s3)
            VpWp_bar_s3_arr.append(VpWp_bar_s3)
            WpTp_bar_s3_arr.append(WpTp_bar_s3)
            WpEp_bar_s3_arr.append(WpEp_bar_s3)
            Ubar_s3_arr.append(Ubar_s3)
            Umedian_s3_arr.append(Umedian_s3)
            Tmedian_s3_arr.append(Tmedian_s3)
            TKE_bar_s3_arr.append(TKE_bar_s3)
            U_horiz_bar_s3_arr.append(U_horiz_bar_s3)
            U_streamwise_bar_s3_arr.append(U_streamwise_bar_s3)
            df_s3_interp.to_csv(path_save+filename_only+"_1.csv")
            print('Port 3 ran: '+filename)

        
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
'''
PART 2
'''

path_save_L2 = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/"

Ubar_s3_arr = np.array(Ubar_s3_arr)
U_horiz_s3_arr = np.array(U_horiz_bar_s3_arr)
U_streamwise_s3_arr = np.array(U_streamwise_bar_s3_arr)
Umedian_s3_arr = np.array(Umedian_s3_arr)
Tbar_s3_arr = np.array(Tbar_s3_arr)
Tmedian_s3_arr = np.array(Tmedian_s3_arr)
UpWp_bar_s3_arr = np.array(UpWp_bar_s3_arr)
VpWp_bar_s3_arr = np.array(VpWp_bar_s3_arr)
WpTp_bar_s3_arr = np.array(WpTp_bar_s3_arr)
WpEp_bar_s3_arr = np.array(WpEp_bar_s3_arr)
TKE_bar_s3_arr = np.array(TKE_bar_s3_arr)


combined_s3_df = pd.DataFrame()
combined_s3_df['Ubar_s3'] = Ubar_s3_arr
combined_s3_df['U_horiz_s3'] = U_horiz_s3_arr
combined_s3_df['U_streamwise_s3'] = U_streamwise_s3_arr
combined_s3_df['Umedian_s3'] = Umedian_s3_arr
combined_s3_df['Tbar_s3'] = Tbar_s3_arr
combined_s3_df['Tmedian_s3'] = Tmedian_s3_arr
combined_s3_df['UpWp_bar_s3'] = UpWp_bar_s3_arr
combined_s3_df['VpWp_bar_s3'] = VpWp_bar_s3_arr
combined_s3_df['WpTp_bar_s3'] = WpTp_bar_s3_arr
combined_s3_df['WpEp_bar_s3'] = WpEp_bar_s3_arr
combined_s3_df['TKE_bar_s3'] = TKE_bar_s3_arr

combined_s3_df.to_csv(path_save_L2 + "s3_turbulenceTerms_andMore_combined.csv")


print('done fully with level0_p3')

