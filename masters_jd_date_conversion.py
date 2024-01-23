# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:48:32 2023

@author: oaklin keefe


This file is used to create a file of julien dates for COARE, and then a file of normal date formats to be used for other code
Input file location:
    /code_pipeline/Level1_errorLinesRemoved/
INPUT files:
    port1/sonic1 files 

Output file location:
    /code_pipeline/Level2/
OUTPUT files:
    jd_combinedAnalysis.csv
    date_combinedAnalysis.csv
    
"""


#%%
import numpy as np
import pandas as pd
import os
import natsort

print('done with imports')
#%%

filepath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_errorLinesRemoved/'
filename_generalDate = []
filename_port1 = []

for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        filename_general_name_only = filename[12:-4]
        if filename.startswith("mNode_Port1"):
            filename_port1.append(filename_only)
            filename_generalDate.append(filename_general_name_only)
            print(filename_general_name_only)

        else:
            continue
#%%
#below for getting decimal date in the COARE format
date_arr = []
time_arr = []
date_time = np.array(filename_generalDate)
for i in range(len(filename_generalDate)):
    (date_i,time_i) = date_time[i].split("_")
    date_arr.append(date_i)
    time_arr.append(time_i)
    # print(str(i))

date_arr = pd.to_datetime(date_arr)
day_of_year = date_arr.dayofyear


hour_arr = []
min_arr = []
sec_arr = []

jd_hour_arr = []
jd_min_arr = []
sec_arr = []
jd_decimal = []
for i in range(len(time_arr)):
    hour_i = time_arr[i][:2]    
    hour_arr.append(hour_i)
    min_i = time_arr[i][2:4]
    min_arr.append(min_i)
    sec_i = time_arr[i][4:]
    sec_arr.append(sec_i)

jd = []
for i in range(len(date_arr)):
    jd_decimal_i = int(hour_arr[i])*3600+int(min_arr[i])*60
    jd_decimal.append(jd_decimal_i)
    if int(hour_arr[i])*3600<10000:
        jd_i = str(day_of_year[i])+".0"+str((int(hour_arr[i])*3600+int(min_arr[i])*60+int(sec_arr[i])))
    else:
        jd_i = str(day_of_year[i])+"."+str((int(hour_arr[i])*3600+int(min_arr[i])*60+int(sec_arr[i])))
    jd_hour_i = int(hour_arr[i])*3600
    jd_min_i = int(min_arr[i])*60
    jd_hour_arr.append(jd_hour_i)
    jd_min_arr.append(jd_min_i)
    jd.append(jd_i)

jd_df = pd.DataFrame()
jd_df['jd'] = jd
jd_df['old_date'] = filename_generalDate

path_save_jd = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
jd_df.to_csv(path_save_jd + 'jd_combinedAnalysis.csv')
print('done')
print(jd_df.head(16))

#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
jd_df = pd.read_csv(file_path+"jd_combinedAnalysis.csv")
#%%
old_date_arr = np.array(jd_df['old_date'])
print(old_date_arr)
#%%
old_date_YYYYMMDD = []
old_date_time = []
for i in range(len(old_date_arr)):
    old_date_YYYYMMDD_i, old_date_time_i = old_date_arr[i].split('_')
    old_date_YYYYMMDD.append(old_date_YYYYMMDD_i)
    old_date_time.append(old_date_time_i)
#%%    
year_arr = [s[:4] for s in old_date_YYYYMMDD]
print(year_arr)
month_arr = [s[4:6] for s in old_date_YYYYMMDD]
print(month_arr)
day_arr = [s[6:] for s in old_date_YYYYMMDD]
print(day_arr)

hour_arr = [s[:2] for s in old_date_time]
print(hour_arr)
min_arr = [s[2:4] for s in old_date_time]
print(min_arr)
sec_arr = [s[4:] for s in old_date_time]
# print(sec_arr)
#%%
date_df = pd.DataFrame()
date_df['YYYY'] = year_arr
date_df['MM'] = month_arr
date_df['DD'] = day_arr
date_df['hh'] = hour_arr
date_df['mm'] = min_arr
date_df['ss'] = sec_arr

date_df['datetime'] = pd.to_datetime(date_df['YYYY'] + '-' +date_df['MM'] + '-' +date_df['DD']+'-'+date_df['hh']+ ':' +date_df['mm'] + ":"+date_df['ss'], format='%Y-%m-%d-%H:%M:%S')


print(date_df)
print(len(date_df))
date_df.to_csv(file_path+"date_combinedAnalysis.csv")
print('done')
