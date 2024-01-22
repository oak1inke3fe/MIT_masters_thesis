# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:25:16 2023

@author: oak

NOtE: this code needs to be run on the remote desktop (or computer that has the CTD data).

This file is used to calculate the height of each sonic above the sea surface given the pressure of the CTD sensor, and knowing
the distance between the CTD sensor and all of the sonics.


INPUT files:
    CTD_data_spring.mat  (file with ctd observations of spring deployment)
    Fall_Seabird_CTD.mat (file with ctd observations of fall deployment)

    
The OUTPUT file is one files with all the buoyancy terms per sonic combined into one dataframe (saved as a .csv):
    ctd20mAvg_allSpring.csv     (file taking average of the 20min period and then concatenating them together; spring deployment, ctd obs)
    ctd20mAvg_allFall.csv       (file taking average of the 20min period and then concatenating them together; fall deployment, ctd obs)
    zAvg_fromCTD_allSpring.csv  (file filled with the full spring sonic/other sensor heights average (1 value that represents the entrie time period's average))
    zAvg_fromCTD_allFall.csv    (file filled with the full fall sonic/other sensor heights average (1 value that represents the entrie time period's average))
    z_airSide_allSpring.csv     (file taking average of the 20min period and then concatenating them together; spring deployment, sonic/other sensor heights)
    z_airSide_allFall.csv     (file taking average of the 20min period and then concatenating them together; fall deployment, sonic/other sensor heights)



"""

#%%
import pandas as pd
import datetime as dt
import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt

#%% This is only to do on the initial read in and don't have to do it again after done once
# file_spring = r"Z:\combined_analysis\OaklinCopyMNode\CTD_data_spring.mat"
file_spring = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/CTD_data_spring.mat'
data_spring = loadmat(file_spring)

depth = np.array(data_spring['press'])
salinity = np.array(data_spring['salt'])
sst = np.array(data_spring['temp'])


sst_1 = []
salt_1 = []
depth_1 = []

sst_2 = []
salt_2 = []
depth_2 = []

sst_3 = []
salt_3 = []
depth_3 = []
for i in range(len(sst)):
    sst_1i = sst[i,0]
    sst_1.append(sst_1i)
    
    salt_1i = salinity[i,0]
    salt_1.append(salt_1i)
    
    depth_1i = depth[i,0]
    depth_1.append(depth_1i)
    
    sst_2i = sst[i,0]
    sst_2.append(sst_2i)
    
    salt_2i = salinity[i,0]
    salt_2.append(salt_2i)
    
    depth_2i = depth[i,0]
    depth_2.append(depth_2i)
    
    sst_3i = sst[i,0]
    sst_3.append(sst_3i)
    
    salt_3i = salinity[i,0]
    salt_3.append(salt_3i)
    
    depth_3i = depth[i,0]
    depth_3.append(depth_3i)
print('done')



ctd_df_spring = pd.DataFrame()
ctd_df_spring['salnity'] = np.array(salt_1).flatten()
ctd_df_spring['sst'] = np.array(sst_1).flatten()
ctd_df_spring['depth'] = np.array(depth_1).flatten()

save_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/'
ctd_df_spring.to_csv(save_path+"ctd_spring.csv")


print('done')

plt.figure()
plt.plot(depth_3)
plt.plot(depth_2)
plt.plot(depth_1)
plt.title('Spring depth')
# plt.xlim(1900,2100)
#%%
# file_fall = r"Z:\Fall_Deployment\OaklinCopyRBRquartz\RAW_dontTouch\Fall_Seabird_CTD.mat"
file_fall = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/Fall_Deployment/OaklinCopyRBRquartz/RAW_dontTouch/Fall_Seabird_CTD.mat'
data_fall = loadmat(file_fall)

depth = np.array(data_fall['depth'])
salinity = np.array(data_fall['salt'])
sst = np.array(data_fall['temp'])


sst_1 = []
salt_1 = []
depth_1 = []

sst_2 = []
salt_2 = []
depth_2 = []

sst_3 = []
salt_3 = []
depth_3 = []
for i in range(len(sst)):
    sst_1i = sst[i,0]
    sst_1.append(sst_1i)
    
    salt_1i = salinity[i,0]
    salt_1.append(salt_1i)
    
    depth_1i = depth[i,0]
    depth_1.append(depth_1i)
    
    sst_2i = sst[i,0]
    sst_2.append(sst_2i)
    
    salt_2i = salinity[i,0]
    salt_2.append(salt_2i)
    
    depth_2i = depth[i,0]
    depth_2.append(depth_2i)
    
    sst_3i = sst[i,0]
    sst_3.append(sst_3i)
    
    salt_3i = salinity[i,0]
    salt_3.append(salt_3i)
    
    depth_3i = depth[i,0]
    depth_3.append(depth_3i)
print('done')



ctd_df_fall = pd.DataFrame()
ctd_df_fall['salnity'] = np.array(salt_1).flatten()
ctd_df_fall['sst'] = np.array(sst_1).flatten()
ctd_df_fall['depth'] = np.array(depth_1).flatten()

ctd_df_fall.to_csv(save_path+"ctd_fall.csv")

print('done')

plt.figure()
plt.plot(depth_3)
plt.plot(depth_2)
plt.plot(depth_1)
plt.title('Fall depth')
# plt.xlim(1900,2100)
#%%
# file_path_spring = r"Z:\combined_analysis\OaklinCopyMNode/"
file_path_spring = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/'
file_spring = "ctd_spring.csv"
ctd_df_spring = pd.read_csv(file_path_spring+file_spring)
ctd_df_spring = ctd_df_spring.drop('Unnamed: 0', axis=1)
print(ctd_df_spring.columns)

file_path_fall = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/'
file_fall = "ctd_fall.csv"
ctd_df_fall = pd.read_csv(file_path_fall+file_fall)
ctd_df_fall = ctd_df_fall.drop('Unnamed: 0', axis=1)
print(ctd_df_fall.columns)
#%%
dn_arr_spring = np.array(data_spring['dn'])
ctd_df_spring['date_time'] = dn_arr_spring
print(ctd_df_spring.columns)

dn_arr_fall = np.array(data_fall['dn'])
ctd_df_fall['date_time'] = dn_arr_fall
print(ctd_df_fall.columns)
#%%
ctd_df_spring['date_time'] = dn_arr_spring
python_datetime_spring = []
for i in range(len(ctd_df_spring)):
    python_datetime_spring_i = dt.datetime.fromordinal(int(np.array(ctd_df_spring['date_time'][i]))) + dt.timedelta(days=np.array(ctd_df_spring['date_time'][i])%1) - dt.timedelta(days = 366)
    python_datetime_spring.append(python_datetime_spring_i)
ctd_df_spring['date_time'] = python_datetime_spring

ctd_df_fall['date_time'] = dn_arr_fall
python_datetime_fall = []
for i in range(len(ctd_df_fall)):
    python_datetime_fall_i = dt.datetime.fromordinal(int(np.array(ctd_df_fall['date_time'][i]))) + dt.timedelta(days=np.array(ctd_df_fall['date_time'][i])%1) - dt.timedelta(days = 366)
    python_datetime_fall.append(python_datetime_fall_i)
ctd_df_fall['date_time'] = python_datetime_fall

#%%
salinity_20MinAvg_spring = []
sst_20MinAvg_spring = []
depth_20MinAvg_spring = []
time_20MinStart_spring = []

for i in range(1,len(ctd_df_spring)-1,2):
    salt_20m_spring_i = (ctd_df_spring['salnity'][i]+ctd_df_spring['salnity'][i+1])/2
    salinity_20MinAvg_spring.append(salt_20m_spring_i)
    sst_20m_spring_i = (ctd_df_spring['sst'][i]+ctd_df_spring['sst'][i+1])/2
    sst_20MinAvg_spring.append(sst_20m_spring_i)
    depth_20m_spring_i = (ctd_df_spring['depth'][i]+ctd_df_spring['depth'][i+1])/2
    depth_20MinAvg_spring.append(depth_20m_spring_i)
    time_spring_i = ctd_df_spring['date_time'][i]
    time_20MinStart_spring.append(time_spring_i)
    
ctd_20minAvg_df_spring = pd.DataFrame()
ctd_20minAvg_df_spring['date_time'] = time_20MinStart_spring
ctd_20minAvg_df_spring['salinity'] = salinity_20MinAvg_spring
ctd_20minAvg_df_spring['sst'] = sst_20MinAvg_spring
ctd_20minAvg_df_spring['depth'] = depth_20MinAvg_spring

spring_start_index = 8
spring_stop_index = 3967

ctd_20minAvg_df_spring = ctd_20minAvg_df_spring[spring_start_index : spring_stop_index + 1]
path_save = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
ctd_20minAvg_df_spring.to_csv(path_save+'ctd20mAvg_allSpring.csv')


salinity_20MinAvg_fall = []
sst_20MinAvg_fall = []
depth_20MinAvg_fall = []
time_20MinStart_fall = []

for i in range(0,len(ctd_df_fall)-1,2):
    salt_20m_fall_i = (ctd_df_fall['salnity'][i]+ctd_df_fall['salnity'][i+1])/2
    salinity_20MinAvg_fall.append(salt_20m_fall_i)
    sst_20m_fall_i = (ctd_df_fall['sst'][i]+ctd_df_fall['sst'][i+1])/2
    sst_20MinAvg_fall.append(sst_20m_fall_i)
    depth_20m_fall_i = (ctd_df_fall['depth'][i]+ctd_df_fall['depth'][i+1])/2
    depth_20MinAvg_fall.append(depth_20m_fall_i)
    time_fall_i = ctd_df_fall['date_time'][i]
    time_20MinStart_fall.append(time_fall_i)
    
ctd_20minAvg_df_fall = pd.DataFrame()
ctd_20minAvg_df_fall['date_time'] = time_20MinStart_fall
ctd_20minAvg_df_fall['salinity'] = salinity_20MinAvg_fall
ctd_20minAvg_df_fall['sst'] = sst_20MinAvg_fall
ctd_20minAvg_df_fall['depth'] = depth_20MinAvg_fall

fall_start_index = 45
fall_stop_index = 4364
ctd_20minAvg_df_fall = ctd_20minAvg_df_fall[fall_start_index : fall_stop_index + 1]

ctd_20minAvg_df_fall.to_csv(path_save+'ctd20mAvg_allFall.csv')

#%%
# SPRING
# for getting zu, zt, zq from CTD depth: (these are used in COARE)
ctd_avgDepth_spring = ctd_20minAvg_df_spring['depth'].mean()
s1_ctd_diff_spring = 15.015 - 9.91
s2_ctd_diff_spring = 17.71  - 9.91
s3_ctd_diff_spring = 20.505 - 9.91
s4_ctd_diff_spring = 22.92  - 9.91

met1_ctd_diff_spring = 16.33  - 9.91
met2_ctd_diff_spring = 18.885 - 9.91

paros1_ctd_diff_spring = 15.335 - 9.91
paros2_ctd_diff_spring = 17.39  - 9.91
paros3_ctd_diff_spring = 20.22  - 9.91

z_s1_spring = s1_ctd_diff_spring - ctd_avgDepth_spring
z_s2_spring = s2_ctd_diff_spring - ctd_avgDepth_spring
z_s3_spring = s3_ctd_diff_spring - ctd_avgDepth_spring
z_s4_spring = s4_ctd_diff_spring - ctd_avgDepth_spring

z_met1_spring = met1_ctd_diff_spring - ctd_avgDepth_spring
z_met2_spring = met2_ctd_diff_spring - ctd_avgDepth_spring

z_paros1_spring = paros1_ctd_diff_spring - ctd_avgDepth_spring
z_paros2_spring = paros2_ctd_diff_spring - ctd_avgDepth_spring
z_paros3_spring = paros3_ctd_diff_spring - ctd_avgDepth_spring


zAvg_df_spring = pd.DataFrame()
zAvg_df_spring['date'] = time_20MinStart_spring[spring_start_index : spring_stop_index + 1]
zAvg_df_spring['z_s1_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_s1_spring
zAvg_df_spring['z_s2_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_s2_spring
zAvg_df_spring['z_s3_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_s3_spring
zAvg_df_spring['z_s4_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_s4_spring
zAvg_df_spring['z_met1_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_met1_spring
zAvg_df_spring['z_met2_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_met2_spring
zAvg_df_spring['z_paros1_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_paros1_spring
zAvg_df_spring['z_paros2_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_paros2_spring
zAvg_df_spring['z_paros3_avg'] = np.ones(len(ctd_20minAvg_df_spring))*z_paros3_spring

zAvg_df_spring.to_csv(path_save+"zAvg_fromCTD_allSpring.csv")

#%%

# 20min depth
z_sonic1_spring = []
z_sonic2_spring = []
z_sonic3_spring = []
z_sonic4_spring = []

z_met1_spring = []
z_met2_spring = []

z_paros1_spring = []
z_paros2_spring = []
z_paros3_spring = []


for i in range (8,len(ctd_20minAvg_df_spring)+8): #start at 8 because that is the entry in the matlab file corresponding to our first spring index
    
    z_sonic1_spring_i = s1_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    z_sonic1_spring_i = s1_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    z_sonic2_spring_i = s2_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    z_sonic3_spring_i = s3_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    z_sonic4_spring_i = s4_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    
    z_met1_spring_i = met1_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    z_met2_spring_i = met2_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    
    z_paros1_spring_i = paros1_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    z_paros2_spring_i = paros2_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    z_paros3_spring_i = paros3_ctd_diff_spring - ctd_20minAvg_df_spring['depth'][i]
    
    
    z_sonic1_spring.append(z_sonic1_spring_i)
    z_sonic2_spring.append(z_sonic2_spring_i)
    z_sonic3_spring.append(z_sonic3_spring_i)
    z_sonic4_spring.append(z_sonic4_spring_i)

    z_met1_spring.append(z_met1_spring_i)
    z_met2_spring.append(z_met2_spring_i)

    z_paros1_spring.append(z_paros1_spring_i)
    z_paros2_spring.append(z_paros2_spring_i)
    z_paros3_spring.append(z_paros3_spring_i)
    
z_df_20m_spring = pd.DataFrame()
z_df_20m_spring['z_sonic1']=z_sonic1_spring
z_df_20m_spring['z_sonic2']=z_sonic2_spring
z_df_20m_spring['z_sonic3']=z_sonic3_spring
z_df_20m_spring['z_sonic4']=z_sonic4_spring
z_df_20m_spring['z_met1']=z_met1_spring
z_df_20m_spring['z_met2']=z_met2_spring
z_df_20m_spring['z_paros1']=z_paros1_spring
z_df_20m_spring['z_paros2']=z_paros2_spring
z_df_20m_spring['z_paros3']=z_paros3_spring
z_df_20m_spring.to_csv(path_save+'z_airSide_allSpring.csv')


#%%
## FALL average depth
# for getting zu, zt, zq from CTD depth:
ctd_avgDepth_fall = ctd_20minAvg_df_fall['depth'].mean()
s1_ctd_diff_fall = 15.40764- 9.23544
s2_ctd_diff_fall = 17.23644- 9.23544
s3_ctd_diff_fall = 20.45208- 9.23544
s4_ctd_diff_fall = 22.92- 9.23544

met1_ctd_diff_fall = 16.32204 - 9.23544
met2_ctd_diff_fall = 18.77568 - 9.23544

paros1_ctd_diff_fall = 15.621 - 9.23544
paros2_ctd_diff_fall = 17.43456 - 9.23544
paros3_ctd_diff_fall = 20.25396 - 9.23544

z_s1_avg_fall = s1_ctd_diff_fall - ctd_avgDepth_fall
z_s2_avg_fall = s2_ctd_diff_fall - ctd_avgDepth_fall
z_s3_avg_fall = s3_ctd_diff_fall - ctd_avgDepth_fall
z_s4_avg_fall = s4_ctd_diff_fall - ctd_avgDepth_fall

z_met1_avg_fall = met1_ctd_diff_fall - ctd_avgDepth_fall
z_met2_avg_fall = met2_ctd_diff_fall - ctd_avgDepth_fall

z_paros1_avg_fall = paros1_ctd_diff_fall - ctd_avgDepth_fall
z_paros2_avg_fall = paros2_ctd_diff_fall - ctd_avgDepth_fall
z_paros3_avg_fall = paros3_ctd_diff_fall - ctd_avgDepth_fall


zAvg_df_fall = pd.DataFrame()
zAvg_df_fall['date'] = time_20MinStart_fall[fall_start_index : fall_stop_index + 1]
zAvg_df_fall['z_s1_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_s1_avg_fall
zAvg_df_fall['z_s2_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_s2_avg_fall
zAvg_df_fall['z_s3_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_s3_avg_fall
zAvg_df_fall['z_s4_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_s4_avg_fall
zAvg_df_fall['z_met1_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_met1_avg_fall
zAvg_df_fall['z_met2_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_met2_avg_fall
zAvg_df_fall['z_paros1_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_paros1_avg_fall
zAvg_df_fall['z_paros2_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_paros2_avg_fall
zAvg_df_fall['z_paros3_avg'] = np.ones(len(ctd_20minAvg_df_fall))*z_paros3_avg_fall

zAvg_df_fall.to_csv(path_save+"zAvg_fromCTD_allFall.csv")



#instantandeous depth
z_sonic1_fall = []
z_sonic2_fall = []
z_sonic3_fall = []
z_sonic4_fall = []

z_met1_fall = []
z_met2_fall = []

z_paros1_fall = []
z_paros2_fall = []
z_paros3_fall = []
for i in range (45, len(ctd_20minAvg_df_fall)+45): #start at 45 because that is the entry in the matlab file corresponding to our first fall index

    z_sonic1_fall_i = s1_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    z_sonic2_fall_i = s2_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    z_sonic3_fall_i = s3_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    z_sonic4_fall_i = s4_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    
    z_met1_fall_i = met1_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    z_met2_fall_i = met2_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    
    z_paros1_fall_i = paros1_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    z_paros2_fall_i = paros2_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    z_paros3_fall_i = paros3_ctd_diff_fall - ctd_20minAvg_df_fall['depth'][i]
    
    
    z_sonic1_fall.append(z_sonic1_fall_i)
    z_sonic2_fall.append(z_sonic2_fall_i)
    z_sonic3_fall.append(z_sonic3_fall_i)
    z_sonic4_fall.append(z_sonic4_fall_i)

    z_met1_fall.append(z_met1_fall_i)
    z_met2_fall.append(z_met2_fall_i)

    z_paros1_fall.append(z_paros1_fall_i)
    z_paros2_fall.append(z_paros2_fall_i)
    z_paros3_fall.append(z_paros3_fall_i)
    
z_df_20m_Fall = pd.DataFrame()
z_df_20m_Fall['z_sonic1']=z_sonic1_fall
z_df_20m_Fall['z_sonic2']=z_sonic2_fall
z_df_20m_Fall['z_sonic3']=z_sonic3_fall
z_df_20m_Fall['z_sonic4']=z_sonic4_fall
z_df_20m_Fall['z_met1']=z_met1_fall
z_df_20m_Fall['z_met2']=z_met2_fall
z_df_20m_Fall['z_paros1']=z_paros1_fall
z_df_20m_Fall['z_paros2']=z_paros2_fall
z_df_20m_Fall['z_paros3']=z_paros3_fall
z_df_20m_Fall.to_csv(path_save+'z_airSide_allFall.csv')


print('done with saving to .csv')
print('done with code')

#%%
filepath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
ctd_df_spring = pd.read_csv(filepath + 'ctd20mAvg_allSpring.csv')
print('spring head')
print(ctd_df_spring.head(5))
print('tail')
print(ctd_df_spring.tail(5))
ctd_df_fall = pd.read_csv(filepath + 'ctd20mAvg_allFall.csv')
print('fall head')
print(ctd_df_fall.head(5))
print('tail')
print(ctd_df_fall.tail(5))
#%%
print('average spring ctd depth: '+str(np.nanmean(ctd_df_spring['depth'])+9.91)) #spring uses CTD #1
print('average fall ctd depth: '+str(np.nanmean(ctd_df_fall['depth'])+9.235)) #Fall uses CTD #2... they are similar because we moved up the CTDs 
# in the second deployment to get closer to the seasurface


print('average spring s3 height: '+str(20.505-9.91-np.nanmean(ctd_df_spring['depth'])))
print('average fall s3 height: '+str(20.45208-9.235-np.nanmean(ctd_df_fall['depth'])))


