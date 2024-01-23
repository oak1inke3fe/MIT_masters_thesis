# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:34:00 2023

@author: oak
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:54:35 2022

@author: oaklin keefe


This is Level 0 pipeline for the meteorological sensors recored on Port 5: taking quality controlled Level00 data, 
making sure there is enough 'good' data and interpolating it to the correct sensor sampling frequency. 
Edited files are saved to the Level1_align-interp folder                                                                     

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

filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_errorLinesRemoved/"
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

#%%
s5_test_file = "mNode_Port5_20221104_000000.txt"
s5_df = pd.read_csv(filepath + s5_test_file, index_col=None, header = None)
s5_df.columns =['date','YYYY','MM','DD','time',
                'hh','mm','ss','yearDay', 
                'bat_volt','pannel_T', 'T1', 'T2','TIR',
                'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                'fix', 'GPS', 'Nsat'] 
#%% THIS CODE ALIGNS, AND INTERPOLATES THE RAW DATA W/REMOVED ERR LINES


start=datetime.datetime.now()



filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_errorLinesRemoved/"

path_save = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-interp/"
for root, dirnames, filenames in os.walk(filepath): 
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        
        
        ### NOTE WE NEED TO DO THIS MONTHLY BECAUSE THE FILE FORMAT IS DIFFERENT FROM SPRING TO FALL OBSERVATION FILES
        
        if filename.startswith('mNode_Port5_202204'):
            # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
            # EX lines of data:
            # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
            # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
            s5_df = pd.read_csv(file, index_col=None, header = None)
            s5_df.columns =['yearDay', 
                            'bat_volt','pannel_T', 'T1', 'T2','TIR',
                            'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                            'fix', 'GPS', 'Nsat'] 
            if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating            
                sigma_sb = 5.67*(10**(-8))                
                IRt = s5_df['IR'] + sigma_sb*(s5_df['TIR']+273.15)**4
                s5_df['IRt'] = IRt
                s5_df_met_interp = interp_met(s5_df)
            else:
                s5_df_met_interp = pd.DataFrame(np.nan, index=[0,1], columns=['yearDay', 
                                                                    'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
                                                                    'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                                                                    'fix', 'GPS', 'Nsat', 'IRt'])
            s5_df_met_interp.to_csv(path_save+str(filename_only)+'_1.csv')
            print('Port 5 ran: '+filename)
        if filename.startswith('mNode_Port5_202205'):
            # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
            # EX lines of data:
            # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
            # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
            s5_df = pd.read_csv(file, index_col=None, header = None)
            s5_df.columns =['yearDay', 
                            'bat_volt','pannel_T', 'T1', 'T2','TIR',
                            'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                            'fix', 'GPS', 'Nsat'] 
            if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating                
                sigma_sb = 5.67*(10**(-8))                
                IRt = s5_df['IR'] + sigma_sb*(s5_df['TIR']+273.15)**4
                s5_df['IRt'] = IRt
                s5_df_met_interp = interp_met(s5_df)
            else:
                s5_df_met_interp = pd.DataFrame(np.nan, index=[0,1], columns=['yearDay', 
                                                                    'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
                                                                    'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                                                                    'fix', 'GPS', 'Nsat', 'IRt'])
            s5_df_met_interp.to_csv(path_save+str(filename_only)+'_1.csv')
            print('Port 5 ran: '+filename)
        if filename.startswith('mNode_Port5_202206'):
            # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
            # EX lines of data:
            # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
            # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
            s5_df = pd.read_csv(file, index_col=None, header = None)
            s5_df.columns =['yearDay', 
                            'bat_volt','pannel_T', 'T1', 'T2','TIR',
                            'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                            'fix', 'GPS', 'Nsat'] 
            if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating            
                sigma_sb = 5.67*(10**(-8))                
                IRt = s5_df['IR'] + sigma_sb*(s5_df['TIR']+273.15)**4
                s5_df['IRt'] = IRt
                s5_df_met_interp = interp_met(s5_df)
            else:
                s5_df_met_interp = pd.DataFrame(np.nan, index=[0,1], columns=['yearDay', 
                                                                    'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
                                                                    'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                                                                    'fix', 'GPS', 'Nsat', 'IRt'])
            s5_df_met_interp.to_csv(path_save+str(filename_only)+'_1.csv')
            print('Port 5 ran: '+filename)
            
            
            
            
            
        if filename.startswith('mNode_Port5_202209'):

            s5_df = pd.read_csv(file, index_col=None, header = None)
            s5_df.columns =['date','YYYY','MM','DD','time',
                            'hh','mm','ss','yearDay', 
                            'bat_volt','pannel_T', 'T1', 'T2','TIR',
                            'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                            'fix', 'GPS', 'Nsat'] 
            if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating                
                sigma_sb = 5.67*(10**(-8))                
                IRt = s5_df['IR'] + sigma_sb*(s5_df['TIR']+273.15)**4
                s5_df['IRt'] = IRt
                s5_df_met_interp = interp_met(s5_df)
                
            else:
                s5_df_met_interp = pd.DataFrame(np.nan, index=[0,1], columns=['date','YYYY','MM','DD','time',
                                                                    'hh','mm','ss','yearDay', 
                                                                    'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
                                                                    'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                                                                    'fix', 'GPS', 'Nsat', 'IRt'])
            s5_df_met_interp.to_csv(path_save+str(filename_only)+'_1.csv')
            print('Port 5 ran: '+filename)
        if filename.startswith('mNode_Port5_202210'):
            s5_df = pd.read_csv(file, index_col=None, header = None)
            s5_df.columns =['date','YYYY','MM','DD','time',
                            'hh','mm','ss','yearDay', 
                            'bat_volt','pannel_T', 'T1', 'T2','TIR',
                            'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                            'fix', 'GPS', 'Nsat'] 
            if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating                
                sigma_sb = 5.67*(10**(-8))                
                IRt = s5_df['IR'] + sigma_sb*(s5_df['TIR']+273.15)**4
                s5_df['IRt'] = IRt
                s5_df_met_interp = interp_met(s5_df)
                
            else:
                s5_df_met_interp = pd.DataFrame(np.nan, index=[0,1], columns=['date','YYYY','MM','DD','time',
                                                                    'hh','mm','ss','yearDay', 
                                                                    'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
                                                                    'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                                                                    'fix', 'GPS', 'Nsat', 'IRt'])
            s5_df_met_interp.to_csv(path_save+str(filename_only)+'_1.csv')
            print('Port 5 ran: '+filename)
        if filename.startswith('mNode_Port5_202211'):
            s5_df = pd.read_csv(file, index_col=None, header = None)
            s5_df.columns =['date','YYYY','MM','DD','time',
                            'hh','mm','ss','yearDay', 
                            'bat_volt','pannel_T', 'T1', 'T2','TIR',
                            'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                            'fix', 'GPS', 'Nsat'] 
            if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating                
                sigma_sb = 5.67*(10**(-8))                
                IRt = s5_df['IR'] + sigma_sb*(s5_df['TIR']+273.15)**4
                s5_df['IRt'] = IRt
                s5_df_met_interp = interp_met(s5_df)
                
            else:
                s5_df_met_interp = pd.DataFrame(np.nan, index=[0,1], columns=['date','YYYY','MM','DD','time',
                                                                    'hh','mm','ss','yearDay', 
                                                                    'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
                                                                    'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
                                                                    'fix', 'GPS', 'Nsat', 'IRt'])
            s5_df_met_interp.to_csv(path_save+str(filename_only)+'_1.csv')
            print('Port 5 ran: '+filename)
                
        
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


print('done with running level0_p5')






