#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:05:49 2023

@author: oaklin keefe


This file is used to determine "bad wind directions" where the wind direcion may cause turbulence to be formed from interaction with the tower or
other tower components and that is unrelated to the flow if the tower was not present.

Input file location:
    part 1: /code_pipeline/Level1_align-interp/
    part 2: /code_pipeline/Level2/
INPUT files:
    part 1:
        sonics files (ports 1-4) from Level1_align-interp files 
    part 2:
        alpha_combinedAnalysis.csv
        despiked_s1_turbulenceTerms_andMore_combined.csv
        despiked_s2_turbulenceTerms_andMore_combined.csv
        despiked_s3_turbulenceTerms_andMore_combined.csv
        despiked_s4_turbulenceTerms_andMore_combined.csv
        date_combinedAnalysis.csv
    
    
We also set:
    base_index= 3959 as the last point in the spring deployment to separate spring and fall datasets so the hampel filter is not 
    corrupted by data that is not in consecutive time sequence.

Output file location:
    part 1 and part 2:
        /code_pipeline/Level2/
OUTPUT files:
    part 1:
        alpha_combinedAnalysis.csv (this is a file with all the wind directions between -180,+180 for the full spring/fall deployment. 0 degrees is
                                coming from the E, +/-180 is coming from the W, +90 is coming from the N, -90 is coming from the S)
        beta_combinedAnalysis.csv
    part 2:
        windDir_withBadFlags_combinedAnalysis.csv (this has teh adjusted alpha such that 0 degrees is N, 90 E, 180 S, 270 W; it also has binary flags
                                               for when a wind direction is blowing through or near the tower)
        windDir_IncludingBad_wS4rotation_combinedAnalysis.csv
        windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv
        WindRose_spring.png
        WindRose_fall.png
        WindRose_combinedAnalysis.png
        windRose_DAYLIGHTcombinedAnalysis.png
        windRose_DaylightMay.png
        windRose_DaylightOct.png
    
    
"""



#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes

print('done with imports')


#%% For first time running

'''
PART 1: only need to run this once to create "alpha_combinedAnalysis.csv" and "beta_combinedAnalysis.csv" files;

can comment out after first run
'''

# alpha_s1 = []
# alpha_s2 = []
# alpha_s3 = []
# alpha_s4 = []

# beta_s1 = []
# beta_s2 = []
# beta_s3 = []
# beta_s4 = []

# # filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
# filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-interp/"
# for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
#     for filename in natsort.natsorted(filenames):
#         file = os.path.join(root, filename)
#         filename_only = filename[:-6]
        
#         if filename.startswith("mNode_Port1"):
#             df = pd.read_csv(file)
#             # #alpha = np.nanmean(df['alpha']) #we don't need to do this... see next line
#             alpha = df['alpha'][1] #first rotation 
#             alpha_s1.append(alpha)
#             beta = df['beta'][1] #second rotation
#             beta_s1.append(beta)
#             print(filename_only)
#         if filename.startswith("mNode_Port2"):
#             df = pd.read_csv(file)
#             alpha = df['alpha'][1] 
#             alpha_s2.append(alpha)
#             beta = df['beta'][1] 
#             beta_s2.append(beta)
#             print(filename_only)
#         if filename.startswith("mNode_Port3"):
#             df = pd.read_csv(file)
#             alpha = df['alpha'][1] 
#             alpha_s3.append(alpha)
#             beta = df['beta'][1] 
#             beta_s3.append(beta)
#             print(filename_only)
#         if filename.startswith("mNode_Port4"):
#             df = pd.read_csv(file)
#             alpha = df['alpha'][1] 
#             alpha_s4.append(alpha)
#             beta = df['beta'][1] 
#             beta_s4.append(beta)
#             print(filename_only)
#         else:
#             continue


# alpha_df = pd.DataFrame()
# alpha_df['alpha_s1'] = alpha_s1
# alpha_df['alpha_s2'] = alpha_s2
# alpha_df['alpha_s3'] = alpha_s3
# alpha_df['alpha_s4'] = alpha_s4

# beta_df = pd.DataFrame()
# beta_df['beta_s1'] = beta_s1
# beta_df['beta_s2'] = beta_s2
# beta_df['beta_s3'] = beta_s3
# beta_df['beta_s4'] = beta_s4


# file_save_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
# alpha_df.to_csv(file_save_path + "alpha_combinedAnalysis.csv")
# beta_df.to_csv(file_save_path + "beta_combinedAnalysis.csv")

# print('done')


#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'

plot_save_path = file_path

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)

sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)

sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)

sonic_file4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)


alpha_df = pd.read_csv(file_path+"alpha_combinedAnalysis.csv")
beta_df = pd.read_csv(file_path+"beta_combinedAnalysis.csv")

time_arr = np.arange(0,len(alpha_df))

date_df = pd.read_csv(file_path + "date_combinedAnalysis.csv")
print(date_df.columns)
print(date_df['datetime'][10])


alpha_df['time'] = time_arr
alpha_df['datetime'] = date_df['datetime']
alpha_df['MM'] = date_df['MM']
alpha_df['hh'] = date_df['hh']

beta_df['time'] = time_arr
beta_df['datetime'] = date_df['datetime']
beta_df['MM'] = date_df['MM']
beta_df['hh'] = date_df['hh']

print(alpha_df.columns)
#%%
#This step to see if the sonics were facing the same way

fig = plt.figure()
plt.scatter(alpha_df['time'], alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(alpha_df['time'], alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(alpha_df['time'], alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.title('Combined Analysis (ALPHA)')
plt.legend()

fig = plt.figure()
plt.scatter(beta_df['time'], beta_df['beta_s1'], color = 'r', label = 's1')
plt.scatter(beta_df['time'], beta_df['beta_s2'], color = 'orange', label = 's2')
plt.scatter(beta_df['time'], beta_df['beta_s3'], color = 'g', label = 's3')
plt.scatter(beta_df['time'], beta_df['beta_s4'], color = 'b', label = 's4')
plt.title('Combined Analysis (BETA)')
plt.legend()


fig = plt.figure()
plt.scatter(alpha_df['time'], alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(alpha_df['time'], alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(alpha_df['time'], alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.xlim(0,3959)
# plt.xlim(2140,2150)
plt.title('Spring Deployment')
plt.legend()
#here we see they are relatively aligned... off by about 5-10 degrees to eachother... I put in a 15º restriction later

fig = plt.figure()
plt.scatter(alpha_df['time'], alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(alpha_df['time'], alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(alpha_df['time'], alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.xlim(3960,8279)
plt.title('Fall Deployment')
plt.legend()
#here we see sonic 4 is aligned differently
#%%
#here we adjust to a 0-360 degree circle

alpha_s1 = np.array(alpha_df['alpha_s1'])
alpha_s2 = np.array(alpha_df['alpha_s2'])
alpha_s3 = np.array(alpha_df['alpha_s3'])
alpha_s4 = np.array(alpha_df['alpha_s4'])
adjusted_arr = [alpha_s1, alpha_s2, alpha_s3, alpha_s4]

for arr in adjusted_arr:
    for i in range(0,len(arr)):
        if arr[i] > 360:
            arr[i] = arr[i]-360
        elif arr[i] < 0:
            arr[i] = 360 + arr[i]
        else:
            arr[i] = arr[i]
print('done')

fig = plt.figure()
plt.scatter(alpha_df['time'], alpha_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], alpha_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], alpha_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], alpha_s4, color = 'b', label = 's4')
plt.title('Combined Analysis on 360º direction')
plt.legend()
#%%
#now we need to adjust them to true north

break_index = 3959 


"""
below, it is found that do 30º - alpha to match high frequency waves will give us the proper wind direction, 
which we then re-correct to 360º
"""
adjusted_a_s1_spring = 30- np.array(alpha_df['alpha_s1'][:break_index+1])
adjusted_a_s2_spring = 30- np.array(alpha_df['alpha_s2'][:break_index+1])
adjusted_a_s3_spring = 30- np.array(alpha_df['alpha_s3'][:break_index+1])
adjusted_a_s4_spring = 30- np.array(alpha_df['alpha_s4'][:break_index+1])

adjusted_a_s1_fall = 30- np.array(alpha_df['alpha_s1'][break_index+1:])
adjusted_a_s2_fall = 30- np.array(alpha_df['alpha_s2'][break_index+1:])
adjusted_a_s3_fall = 30- np.array(alpha_df['alpha_s3'][break_index+1:])
adjusted_a_s4_fall = 30- np.array(alpha_df['alpha_s4'][break_index+1:])

#combine the spring and fall deployments
adjusted_a_s1 = np.concatenate([adjusted_a_s1_spring, adjusted_a_s1_fall], axis = 0)
adjusted_a_s2 = np.concatenate([adjusted_a_s2_spring, adjusted_a_s2_fall], axis = 0)
adjusted_a_s3 = np.concatenate([adjusted_a_s3_spring, adjusted_a_s3_fall], axis = 0)
adjusted_a_s4 = np.concatenate([adjusted_a_s4_spring, adjusted_a_s4_fall], axis = 0)


fig = plt.figure()
plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
plt.hlines(y=360, xmin=0, xmax=len(alpha_df), color = 'k')
plt.hlines(y=0, xmin=0, xmax=len(alpha_df), color = 'k')
plt.legend()
plt.xlim(0, 3959)
plt.title('Spring Deployment')


fig = plt.figure()
plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
plt.hlines(y=360, xmin=0, xmax=len(alpha_df), color = 'k')
plt.hlines(y=0, xmin=0, xmax=len(alpha_df), color = 'k')
plt.legend()
plt.xlim(3959, 8279)
plt.title('Fall Deployment')

#%%
#here we adjust AGAIN to a 0-360 degree circle, with 0 degrees as wind coming from the north
adjusted_arr = [adjusted_a_s1, adjusted_a_s2, adjusted_a_s3, adjusted_a_s4]

for arr in adjusted_arr:
    for i in range(0,len(arr)):
        if arr[i] > 360:
            arr[i] = arr[i]-360
        elif arr[i] < 0:
            arr[i] = 360 + arr[i]
        else:
            arr[i] = arr[i]
print('done')

fig = plt.figure()
plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
plt.hlines(y=360, xmin=0, xmax=len(alpha_df), color = 'k')
plt.hlines(y=0, xmin=0, xmax=len(alpha_df), color = 'k')
plt.legend()
plt.xlim(0, 3959)
plt.title('Spring Deployment')


fig = plt.figure()
plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
plt.hlines(y=360, xmin=0, xmax=len(alpha_df), color = 'k')
plt.hlines(y=0, xmin=0, xmax=len(alpha_df), color = 'k')
plt.legend()
plt.xlim(3959, 8279)
plt.title('Fall Deployment')

#make a combined dataframe
adjusted_alpha_df = pd.DataFrame()
adjusted_alpha_df['alpha_s1'] = adjusted_a_s1
adjusted_alpha_df['alpha_s2'] = adjusted_a_s2
adjusted_alpha_df['alpha_s3'] = adjusted_a_s3
adjusted_alpha_df['alpha_s4'] = adjusted_a_s4
adjusted_alpha_df['time'] = alpha_df['time']
adjusted_alpha_df['date'] = alpha_df['datetime']
adjusted_alpha_df['MM'] = alpha_df['MM']
adjusted_alpha_df['hh'] = alpha_df['hh']
"""
#%%
#compare adjustment to period when we know the wind was consistently from the SW and
#check the output is from the SW
#here we will pick May 14 1800 [index 2142] where the wind was from the SSW-SW
plt.figure()
plt.plot(adjusted_a_s1, label ='s1', color = 'r')
plt.plot(adjusted_a_s2, label ='s2', color = 'orange')
plt.plot(adjusted_a_s3, label ='s3', color = 'g')
plt.plot(adjusted_a_s4, label ='s4', color = 'b')
plt.legend()
# plt.xlim(2140,2150)
# plt.title ('checking flow is from the SSW-SW')
plt.xlim(925,975)
plt.ylim(300,360)
plt.title ('checking difference between s4 and s1 directions \n at different points in time')


# plt.figure()
# plt.plot(alpha_df['alpha_s1'][:break_index+1], label ='s1', color = 'r')
# plt.plot(alpha_df['alpha_s2'][:break_index+1], label ='s2', color = 'orange')
# plt.plot(alpha_df['alpha_s3'][:break_index+1], label ='s3', color = 'g')
# plt.plot(alpha_df['alpha_s4'][:break_index+1], label ='s4', color = 'b')
# plt.legend()
# plt.xlim(2140,2150)
# plt.title ('checking flow is from the SSW-SW')

# fig = plt.figure()
# plt.scatter(alpha_df['time'], adjusted_alpha_df['alpha_s1'], color = 'r', label = 's1')
# plt.scatter(alpha_df['time'], adjusted_alpha_df['alpha_s2'], color = 'orange', label = 's2')
# plt.scatter(alpha_df['time'], adjusted_alpha_df['alpha_s3'], color = 'g', label = 's3')
# plt.scatter(alpha_df['time'], adjusted_alpha_df['alpha_s4'], color = 'b', label = 's4')
# plt.legend()
# plt.xlim(0,3959)
# # plt.ylim(300,350)
# plt.title('Spring Deployment Adjusted to 360')
#%%
#compare adjustment to period when we know the wind was consistently from the NE and
#check the output is from the NE
#here we will pick Oct 02 1400 [index 4650] where the wind was from the NE
fig = plt.figure()
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.legend()
plt.xlim(4650,4800)
# plt.ylim(300,350)
plt.title('Fall Deployment Adjusted to 360')
"""
#%%
#s4 is still off by 90 in the fall deployment, so we will rotate it to match the proper north alignment
adjusted_a_s1_fall_s4rotation = adjusted_a_s1_fall
adjusted_a_s2_fall_s4rotation = adjusted_a_s2_fall
adjusted_a_s3_fall_s4rotation = adjusted_a_s3_fall
adjusted_a_s4_fall_s4rotation = adjusted_a_s4_fall + 90

adjusted_arr = [adjusted_a_s1_fall_s4rotation, 
                adjusted_a_s2_fall_s4rotation, 
                adjusted_a_s3_fall_s4rotation, 
                adjusted_a_s4_fall_s4rotation]

for arr in adjusted_arr:
    for i in range(0,len(arr)):
        if arr[i] > 360:
            arr[i] = arr[i]-360
        elif arr[i] < 0:
            arr[i] = 360 + arr[i]
        else:
            arr[i] = arr[i]
print('done')

#combine un-adjusted spring data with newly adjusted fall data
adjusted_a_s1_wS4rotation = np.concatenate([np.array(adjusted_a_s1[:break_index+1]), adjusted_a_s1_fall_s4rotation], axis = 0)
adjusted_a_s2_wS4rotation = np.concatenate([np.array(adjusted_a_s2[:break_index+1]), adjusted_a_s2_fall_s4rotation], axis = 0)
adjusted_a_s3_wS4rotation = np.concatenate([np.array(adjusted_a_s3[:break_index+1]), adjusted_a_s3_fall_s4rotation], axis = 0)
adjusted_a_s4_wS4rotation = np.concatenate([np.array(adjusted_a_s4[:break_index+1]), adjusted_a_s4_fall_s4rotation], axis = 0)

#create a final combined dataset (prior to excluding "bad" wind directions)
adjusted_alpha_df_final = pd.DataFrame()
adjusted_alpha_df_final['alpha_s1'] = adjusted_a_s1_wS4rotation
adjusted_alpha_df_final['alpha_s2'] = adjusted_a_s2_wS4rotation
adjusted_alpha_df_final['alpha_s3'] = adjusted_a_s3_wS4rotation
adjusted_alpha_df_final['alpha_s4'] = adjusted_a_s4_wS4rotation
adjusted_alpha_df_final['time'] = alpha_df['time']
adjusted_alpha_df_final['date'] = alpha_df['datetime']
adjusted_alpha_df_final['MM'] = alpha_df['MM']
adjusted_alpha_df_final['hh'] = alpha_df['hh']
adjusted_alpha_df_final.to_csv(file_path+"windDir_IncludingBad_wS4rotation_combinedAnalysis.csv")

#plot to make sure they are aligned within 15 degrees of each other
fig = plt.figure()
plt.plot(adjusted_alpha_df_final['time'], adjusted_alpha_df_final['alpha_s1'], color = 'r', label = 's1')
plt.plot(adjusted_alpha_df_final['time'], adjusted_alpha_df_final['alpha_s2'], color = 'orange', label = 's2')
plt.plot(adjusted_alpha_df_final['time'], adjusted_alpha_df_final['alpha_s3'], color = 'g', label = 's3')
plt.plot(adjusted_alpha_df_final['time'], adjusted_alpha_df_final['alpha_s4'], color = 'b', label = 's4')
plt.legend()
plt.xlim(4650,4680) #this checks the fall period which we had to adjust sonic 4 to
plt.ylim(40,60)
plt.title('Fall Deployment Adjusted to 360')
#%%
# create a windrose to display the full conditions

fig, ax = plt.subplots(1,1)
ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s3'], sonic3_df['Ubar'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Wind Velocity [$ms^{-1}$]', fontsize=20)
ax.set_legend(bbox_to_anchor=(0.9, -0.1),fontsize=20)

plt.savefig(plot_save_path+ "windRoseAllWindDirections_combinedAnalysis.png", dpi=300)
plt.savefig(plot_save_path+ "windRoseAllWindDirections_combinedAnalysis.pdf")

#%% make windroses of daylight conditions (not necessary, but just for fun)
df_daylight = pd.DataFrame()

plt.figure()
plt.scatter(adjusted_alpha_df_final['alpha_s4'], sonic4_df['Ubar'], s=1)

index_array = np.arange(len(adjusted_alpha_df))
adjusted_alpha_df_final['new_index_arr'] = np.where((adjusted_alpha_df_final['hh']>=7)&(adjusted_alpha_df_final['hh']<=17), np.nan, index_array)
mask_Daylight = np.isin(adjusted_alpha_df_final['new_index_arr'],index_array)
adjusted_alpha_df_final[mask_Daylight] = np.nan
sonic4_df[mask_Daylight] = np.nan

plt.figure()
plt.scatter(adjusted_alpha_df_final['alpha_s4'], sonic4_df['Ubar'],s=1)

ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s4'], sonic4_df['Ubar'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('DAYLIGHT Wind Velocity [$ms^{-1}$] \n Combined Deployments', fontsize=20)
ax.set_legend(bbox_to_anchor=(0.9, -0.1),fontsize=20)

plt.savefig(plot_save_path+ "windRose_DAYLIGHTcombinedAnalysis.png", dpi=300)
plt.show()
plt.savefig(plot_save_path+ "windRose_DAYLIGHTcombinedAnalysis.pdf")

#%% daylight by month (just May and Oct.) to compare to Sailflow.com (WeatherFlow Tempest, Inc.)

may_daylight_df = pd.DataFrame()
may_daylight_df['index'] = np.arange(len(adjusted_alpha_df_final))
may_daylight_df['alpha_s4'] = np.array(adjusted_alpha_df_final['alpha_s4'])
may_daylight_df['Ubar_s4'] = np.array(sonic4_df['Ubar'])
plt.figure()
plt.scatter(may_daylight_df['alpha_s4'], may_daylight_df['Ubar_s4'],s=1)
plt.title('May daylight Before')

oct_daylight_df = pd.DataFrame()
oct_daylight_df['index'] = np.arange(len(adjusted_alpha_df_final))
oct_daylight_df['alpha_s4'] = np.array(adjusted_alpha_df_final['alpha_s4'])
oct_daylight_df['Ubar_s4'] = np.array(sonic4_df['Ubar'])
plt.figure()
plt.scatter(oct_daylight_df['alpha_s4'], oct_daylight_df['Ubar_s4'],s=1)
plt.title('Oct. daylight Before')

may_daylight_df['may_daylight']= np.where(adjusted_alpha_df_final['MM']==5, np.nan, index_array)
oct_daylight_df['oct_daylight']= np.where(adjusted_alpha_df_final['MM']==10, np.nan, index_array)
mask_mayDaylight = np.isin(may_daylight_df['may_daylight'],index_array)
mask_octDaylight = np.isin(oct_daylight_df['oct_daylight'],index_array)

may_daylight_df[mask_mayDaylight] = np.nan
plt.figure()
plt.scatter(may_daylight_df['alpha_s4'], may_daylight_df['Ubar_s4'],s=1)
plt.title('May daylight AFTER')

oct_daylight_df[mask_octDaylight] = np.nan
plt.figure()
plt.scatter(oct_daylight_df['alpha_s4'], oct_daylight_df['Ubar_s4'],s=1)
plt.title('Oct. daylight After')

# windrose of May 2022 daylight wind conditions
ax = WindroseAxes.from_ax()
ax.bar(may_daylight_df['alpha_s4'], may_daylight_df['Ubar_s4'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Daylight Wind Velocity [$ms^{-1}$] \n MAY', fontsize=20)
ax.set_legend(bbox_to_anchor=(0.9, -0.1),fontsize=20)

plt.savefig(plot_save_path+ "windRose_DaylightMay.png", dpi=300)
plt.show()
plt.savefig(plot_save_path+ "windRose_DaylightMay.pdf")

# windrose of October 2022 daylight wind conditions
ax = WindroseAxes.from_ax()
ax.bar(oct_daylight_df['alpha_s4'], oct_daylight_df['Ubar_s4'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Daylight Wind Velocity [$ms^{-1}$] \n OCTOBER', fontsize=20)
ax.set_legend(bbox_to_anchor=(0.9, -0.1),fontsize=20)

plt.savefig(plot_save_path+ "windRose_DaylightOct.png", dpi=300)
plt.show()
plt.savefig(plot_save_path+ "windRose_DaylightOct.pdf")



#%% Moving on to excluding bad wind directions
"""
"Bad" wind directions are where there was flow distortion by the tower creating "tower-turbulence".

The method below uses the law of sines to flag bad wind directions based on the dimensions of the triangular BB-ASIT tower structure
and the angle of the booms holding the sonic anemometers. It takes into account that the booms are pointing at roughly 305º with 0º as N.

"""
sonic_head_arr = [1.27, 1.285, 1.23]
sonic_min = np.min(sonic_head_arr)
b = 0.641
c = 0.641+sonic_min
A = 60
import math
a_len = math.sqrt(b**2+c**2-(2*b*c*math.cos(A*math.pi/180)))
angle_offset_rad = math.asin(b*math.sin(A*math.pi/180)/a_len)
angle_offset = angle_offset_rad*180/math.pi
print("angle offset = " + str(angle_offset))
angle_start= 125+angle_offset
angle_end = 125

print(angle_start)
print(angle_end)

# the resulting degrees to exclude are: 125-144
#%%
"""
Here, we have added a 15º pad to the angle_start and angle_end values to be sure we are excluding near-angles
that may be causing flow distortion
"""

angle_start_spring= 110
angle_end_spring = 160 

angle_start_fall = 110
angle_end_fall = 160 

print("SPRING: good angle start = " + str(angle_start_spring))
print("SPRING good angle end = " + str(angle_end_spring))
print("FALL: good angle start = " + str(angle_start_fall))
print("FALL good angle end = " + str(angle_end_fall))


#%% create flags of when wind direction was "good" or "bad"

adjusted_a_s4_copy_spring = np.array(adjusted_a_s4_wS4rotation[: break_index+1])
adjusted_a_s4_copy_fall = np.array(adjusted_a_s4_wS4rotation[break_index+1:])

good_flag_spring = np.ones(len(adjusted_a_s4_copy_spring), dtype = bool)
good_flag_fall = np.ones(len(adjusted_a_s4_copy_fall), dtype = bool)

good_flag_4_spring = np.where((adjusted_a_s4_copy_spring >= angle_start_spring)&(adjusted_a_s4_copy_spring <= angle_end_spring), 'False', good_flag_spring)
good_flag_4_fall = np.where((adjusted_a_s4_copy_fall >= angle_start_fall)&(adjusted_a_s4_copy_fall <= angle_end_fall), 'False', good_flag_fall)

good_flag_4 = np.concatenate([good_flag_4_spring, good_flag_4_fall], axis=0)

print('done')

adjusted_alpha_df_final['good_wind_dir'] = np.array(good_flag_4)
print('done adding good flag array to df')

#%% This is for flagging more bad wind directions if sonic 3 deviates from sonic 4 by more than 15 degrees.
"""
We compare to sonic 4 because it was unobstructed from the tower and should have the most accurate real wind direction. 

There are periods where alpha_s4 is NaN, and when this happens, we check that sonic 3 and sonic 2 are within 15 degrees of eachother,
or they get a bad flag.

"""
test_badWindDir_s1 = []
test_badWindDir_s2 = []
test_badWindDir_s3 = []
test_badWindDir_s4 = []
original_badWindDir_s4 = adjusted_alpha_df_final['alpha_s4']

import math
for i in range(len(adjusted_alpha_df_final)):
    if math.isnan(adjusted_alpha_df_final['alpha_s4'][i]) == False:
        if np.abs(adjusted_alpha_df_final['alpha_s4'][i] - adjusted_alpha_df_final['alpha_s3'][i]) > 15:
            test_badWindDir_s4_i = 'False'
        else:
            test_badWindDir_s4_i = 'True'
    else:
        if np.abs(adjusted_alpha_df_final['alpha_s2'][i] - adjusted_alpha_df_final['alpha_s3'][i]) > 15:
            test_badWindDir_s4_i = 'False'
        else:
            test_badWindDir_s4_i = 'True'
    test_badWindDir_s4.append(test_badWindDir_s4_i)

plt.figure()
plt.scatter(np.arange(len(test_badWindDir_s4)), test_badWindDir_s4)
plt.title('when s3 is within 15 deg of s4 or s2, if s4 is nan; combined analysis')


adjusted_alpha_df_final['windDir_within_15_deg'] = np.array(test_badWindDir_s4) #add a new column with the expanded bad wind direction restrictions
final_good_windDir = []
for i in range(len(adjusted_alpha_df_final)):
    if adjusted_alpha_df_final['windDir_within_15_deg'][i] == 'True':
        if adjusted_alpha_df_final['good_wind_dir'][i] == 'True':
            final_good_windDir_i = 'True'
        else:
            final_good_windDir_i = 'False'
    else:
        final_good_windDir_i = 'False'
    final_good_windDir.append(final_good_windDir_i)
adjusted_alpha_df_final['windDir_final'] = np.array(final_good_windDir)       



adjusted_alpha_df_final['time'] = alpha_df['time'] #add a time column
adjusted_alpha_df_final['datetime'] = alpha_df['datetime']       #add a datetime column
adjusted_alpha_df_final.to_csv(file_path+"windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv")

print('saved file of bad wind directions')


#%% Plot to see which points were excluded during each deployment
groups_good = adjusted_alpha_df_final.groupby('good_wind_dir')
plt.figure()
for name, group in groups_good:
    plt.plot(group['time'], group['alpha_s4'], marker='o', linestyle='', markersize=2, label=name)
plt.xlabel('time index')
plt.ylabel('wind direction (deg)')
plt.vlines(x=3959, ymin = 0, ymax=360, color = 'k', label = 'Spring | Fall')
# plt.xlim(2000,2200)
plt.legend(loc = 'lower left')
plt.title("Wind Dir. (Good wind directions = False)")
#%%
ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s4'][:break_index+1], sonic4_df['Ubar'][:break_index+1], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Wind Rose Spring Dataset')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))
fig.savefig(plot_save_path + 'WindRose_spring.png', dpi = 300)
fig.savefig(plot_save_path + 'WindRose_spring.pdf')

ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s4'][break_index+1:], sonic4_df['Ubar'][break_index+1:], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Wind Rose Fall Dataset')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))
fig.savefig(plot_save_path + 'WindRose_fall.png', dpi = 300)
fig.savefig(plot_save_path + 'WindRose_fall.pdf')

ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s3'], sonic3_df['Ubar'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Wind Rose Combined Datasets')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))
fig.savefig(plot_save_path + 'WindRose_combinedAnalysis.png', dpi = 300)
fig.savefig(plot_save_path + 'WindRose_combinedAnalysis.pdf')

print('figures saved')

