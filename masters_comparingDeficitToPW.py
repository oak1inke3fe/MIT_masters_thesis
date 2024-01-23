#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:47:11 2023

@author: oaklin keefe


This file is used to compare the dissipation deficit (P+B-Epsilon) to wave-coherent pressure-work

Input file location:
    code_pipeline/Level2
INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv
    pw_combinedAnalysis.csv
    prodTerm_combinedAnalysis.csv
    epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv
    buoy_terms_combinedAnalysis.csv
    rhoAvg_combinedAnalysis.csv
    z_airSide_allSpring.csv
    z_airSide_allFall.csv
    ZoverL_combinedAnalysis.csv
    usr_combinedAnalysis.csv
    
    
We also set:
    base_index= 3959 as the last point in the spring deployment to separate spring and fall datasets so the hampel filter is not 
    corrupted by data that is not in consecutive time sequence.

Output file location:
    code_pipeline/Level2
OUTPUT plots:
    mayStorm_despike31dissipationDeficit_comparisonWithPW.png
    mayStorm_despike32dissipationDeficit_comparisonWithPW.png
    mayStorm_AVGdissipationDeficit_comparisonWithPW.png

"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hampel import hampel

print('done with imports')
#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
plot_savePath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)
sonic_file4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)

windSpeed_df = pd.DataFrame()
windSpeed_df['Ubar_LI'] = (sonic1_df['Ubar']+sonic2_df['Ubar'])/2
windSpeed_df['Ubar_LII'] = (sonic2_df['Ubar']+sonic3_df['Ubar'])/2


windDir_file = "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv"
windDir_df = pd.read_csv(file_path + windDir_file)
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)


pw_df = pd.read_csv(file_path + 'pw_combinedAnalysis.csv')
pw_df = pw_df.drop(['Unnamed: 0'], axis=1)


prod_df = pd.read_csv(file_path+'prodTerm_combinedAnalysis.csv')
prod_df = prod_df.drop(['Unnamed: 0'], axis=1)


eps_df = pd.read_csv(file_path+"epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv")
eps_df = eps_df.drop(['Unnamed: 0'], axis=1)

buoy_df = pd.read_csv(file_path+'buoy_terms_combinedAnalysis.csv')
buoy_df = buoy_df.drop(['Unnamed: 0'], axis=1)

rho_df = pd.read_csv(file_path+'rhoAvg_combinedAnalysis.csv')
rho_df = rho_df.drop(['Unnamed: 0'], axis=1)

z_df_spring = pd.read_csv(file_path+'z_airSide_allSpring.csv')
z_df_spring = z_df_spring.drop(['Unnamed: 0'], axis=1)

z_df_fall = pd.read_csv(file_path+'z_airSide_allFall.csv')
z_df_fall = z_df_fall.drop(['Unnamed: 0'], axis=1)

z_df = pd.concat([z_df_spring, z_df_fall], axis=0)

zL_df = pd.read_csv(file_path + 'ZoverL_combinedAnalysis.csv')
zL_df = zL_df.drop(['Unnamed: 0'], axis=1)

usr_df = pd.read_csv(file_path + "usr_combinedAnalysis.csv")
usr_df = usr_df.drop(['Unnamed: 0'], axis=1)

break_index = 3959

print('done with setting up dataframes')
#%%
plt.figure()
plt.plot(-1*sonic1_df['UpWp_bar'], label = 's1')
plt.plot(-1*sonic2_df['UpWp_bar'], label = 's2')
plt.plot(-1*sonic3_df['UpWp_bar'], label = 's3')
plt.plot(-1*sonic4_df['UpWp_bar'], label = 's4')
# plt.plot(-1*sonic2_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 's2-s1 difference')
# plt.plot(-1*sonic3_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 's3-s1 difference')
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1500,2000)
plt.ylim(-0.2,0.7)
plt.legend()
plt.title("$-\overline{u'w'}$")

#%%
# mask to make when s4 out of bounds, it reads NaN
s4_index_array = np.arange(len(sonic4_df))
sonic4_df['new_index_arr'] = np.where((np.abs(sonic4_df['UpWp_bar'])<=np.abs(sonic3_df['UpWp_bar'])+0.05), np.nan, s4_index_array)
mask_s4 = np.isin(sonic4_df['new_index_arr'],s4_index_array)

sonic4_df[mask_s4] = np.nan

plt.figure()
plt.plot(-1*sonic1_df['UpWp_bar'], label = 's1')
plt.plot(-1*sonic2_df['UpWp_bar'], label = 's2')
plt.plot(-1*sonic3_df['UpWp_bar'], label = 's3')
plt.plot(-1*sonic4_df['UpWp_bar'], label = 's4')
# plt.plot(-1*sonic2_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 's2-s1 difference')
# plt.plot(-1*sonic3_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 's3-s1 difference')
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1500,2000)
plt.ylim(-0.2,0.7)
plt.legend()
plt.title("$-\overline{u'w'}$")


#%%
"""
For each 20m block, we have 4 estimates of <uw> (one from each sonic). 
take the median of those 4 measurements to reduce the noise (i.e., assume constant stress layer, and average more). 
to have a single lower-noise <uw> for each 20min block.
 
The integrated production from sonic1 to sonic3 would then be: ( - <uw>_ave )*(U3-U1)
"""

UpWp_bar_df = pd.DataFrame()
UpWp_bar_df['s1_UpWp_bar']= sonic1_df['UpWp_bar']
UpWp_bar_df['s2_UpWp_bar']= sonic2_df['UpWp_bar']
UpWp_bar_df['s3_UpWp_bar']= sonic3_df['UpWp_bar']
# UpWp_bar_df['s4_UpWp_bar']= sonic4_df['UpWp_bar']

UpWp_bar_avg = np.array(UpWp_bar_df.mean(axis=1, skipna=True))

plt.figure()
plt.plot(-1*sonic1_df['UpWp_bar'], label = 's1')
plt.plot(-1*sonic2_df['UpWp_bar'], label = 's2')
plt.plot(-1*sonic3_df['UpWp_bar'], label = 's3')
# plt.plot(-1*sonic4_df['UpWp_bar'], label = 's4') 
plt.plot(-1*UpWp_bar_avg, color = 'k', label = 'AVG')
# plt.plot(-1*sonic2_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 's2-s1 difference')
# plt.plot(-1*sonic3_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 's3-s1 difference')
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1500,2000)
plt.ylim(-0.2,0.7)
plt.legend()
plt.title("$-\overline{u'w'}$")

#%%
UpWp_spring_df = pd.DataFrame()
UpWp_spring_df['UpWp_bar'] = UpWp_bar_avg[:break_index+1]

UpWp_arr_spring = [UpWp_spring_df,]

UpWp_spring_despike = pd.DataFrame()
UpWp_despike_arr_spring = [UpWp_spring_despike,]

column_arr = ['UpWp_bar']

for i in range(len(UpWp_arr_spring)):
    for column_name in column_arr:
    
        my_array = UpWp_arr_spring[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 5
        n = 1
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        UpWp_despike_arr_spring[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel SPRING despike')

#%%
UpWp_fall_df = pd.DataFrame()
UpWp_fall_df['UpWp_bar'] = UpWp_bar_avg[break_index+1:]
UpWp_fall_df = UpWp_fall_df.reset_index(drop = True)



UpWp_arr_fall = [UpWp_fall_df,]

UpWp_fall_despike = pd.DataFrame()

UpWp_despike_arr_fall = [UpWp_fall_despike,]
column_arr = ['UpWp_bar']

for i in range(len(UpWp_arr_fall)):
    for column_name in column_arr:
    
        my_array = UpWp_arr_fall[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 5
        n = 1
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        UpWp_despike_arr_fall[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel FALL despike')
#%%
# combine new spring/fall despiked values/dataframes to one combined dataframe
UpWp_despiked_combined = pd.concat([UpWp_spring_despike,UpWp_fall_despike], axis = 0)
UpWp_despiked_combined['new_index'] = np.arange(0, len(UpWp_despiked_combined))
UpWp_despiked_combined = UpWp_despiked_combined.set_index('new_index')

print('done combining spring and fall')
#%%
plt.figure()
# plt.plot(-1*sonic1_df['UpWp_bar'], label = 's1')
# plt.plot(-1*sonic2_df['UpWp_bar'], label = 's2')
# plt.plot(-1*sonic3_df['UpWp_bar'], label = 's3')
# plt.plot(-1*sonic4_df['UpWp_bar'], label = 's4')
plt.plot(-1*UpWp_bar_avg, color = 'gray', label = ' AVG')
plt.plot(-1*UpWp_despiked_combined, color = 'k', label = 'despiked AVG')
# plt.plot(-1*sonic2_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 's2-s1 difference')
# plt.plot(-1*sonic3_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 's3-s1 difference')
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1500,2000)
plt.ylim(-0.2,0.7)
plt.legend()
plt.title("Despiked $-\overline{u'w'}$")

y_spring = np.vstack((eps_df['epsU_sonic1_MAD'][:break_index+1], eps_df['epsU_sonic3_MAD'][:break_index+1])).T
y_fall = np.vstack((eps_df['epsU_sonic1_MAD'][break_index+1:], eps_df['epsU_sonic3_MAD'][break_index+1:])).T
rho_eps_spring = np.array(rho_df['rho_bar_1_dry'][:break_index+1])*np.trapz(y=y_spring, x=None, dx=5.49)#do trapz for between sonics 1-3
rho_eps_fall = np.array(rho_df['rho_bar_1_dry'][break_index+1:])*np.trapz(y=y_fall, x=None, dx=5.0292)#do trapz for between sonics 1-3
rho_eps = np.concatenate((rho_eps_spring, rho_eps_fall), axis=0)

plt.figure()
# plt.plot(rho_df['rho_bar_1_dry']*(-1*np.array(UpWp_despiked_combined['UpWp_bar']))*(np.array(sonic3_df['Ubar']-sonic1_df['Ubar'])), color = 'gray', label = r"$\rho (-\overline{u'w'} \cdot \overline{u})$" )
# plt.plot(rho_df['rho_bar_1_dry']*(-1*np.array(sonic3_df['UpWp_bar'])*(np.array(sonic3_df['Ubar']-sonic1_df['Ubar']))), color = 'blue', label = r"$\rho (-\overline{u'w'} \cdot \overline{u})$" )
# plt.plot(rho_eps, color = 'brown', label = r"$\rho (\epsilon)$" )
plt.plot(rho_df['rho_bar_1_dry']*(-1*np.array(UpWp_despiked_combined['UpWp_bar']))*(np.array(sonic3_df['Ubar']-sonic1_df['Ubar']))-rho_eps, color = 'k', label = r"$\rho (-\overline{u'w'}_{AVG} \cdot \overline{u})-\rho(\epsilon)$")
plt.plot(rho_df['rho_bar_1_dry']*(-1*np.array(sonic3_df['UpWp_bar'])*(np.array(sonic3_df['Ubar']-sonic1_df['Ubar'])))-rho_eps, color = 'gray', label = r"$\rho (-\overline{u'w'}_{s3} \cdot \overline{u})-\rho(\epsilon)$")
plt.title(r"P and Eps")
plt.xlim(1500,2000)
plt.legend()
plt.ylim(-0.3,0.3)


#%%
Eps_spring_df = pd.DataFrame()
Eps_spring_df['Eps'] = rho_eps[:break_index+1]

Eps_arr_spring = [Eps_spring_df,]

Eps_spring_despike = pd.DataFrame()
Eps_despike_arr_spring = [Eps_spring_despike,]

column_arr = ['Eps']

for i in range(len(Eps_arr_spring)):
    for column_name in column_arr:
    
        my_array = Eps_arr_spring[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 5
        n = 1
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        Eps_despike_arr_spring[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel SPRING despike')

#%%
Eps_fall_df = pd.DataFrame()
Eps_fall_df['Eps'] = rho_eps[break_index+1:]
Eps_fall_df = Eps_fall_df.reset_index(drop = True)



Eps_arr_fall = [Eps_fall_df,]

Eps_fall_despike = pd.DataFrame()

Eps_despike_arr_fall = [Eps_fall_despike,]
column_arr = ['Eps']

for i in range(len(Eps_arr_fall)):
    for column_name in column_arr:
    
        my_array = Eps_arr_fall[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 5
        n = 1
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        Eps_despike_arr_fall[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel FALL despike')
#%%
# combine new spring/fall despiked values/dataframes to one combined dataframe
Eps_despiked_combined = pd.concat([Eps_spring_despike,Eps_fall_despike], axis = 0)
Eps_despiked_combined['new_index'] = np.arange(0, len(Eps_despiked_combined))
Eps_despiked_combined = Eps_despiked_combined.set_index('new_index')

print('done combining spring and fall')


#%%
plt.figure()
# plt.plot(rho_df['rho_bar_1_dry']*(-1*np.array(UpWp_despiked_combined['UpWp_bar']))*(np.array(sonic3_df['Ubar']-sonic1_df['Ubar'])), label = r"$\rho (-\overline{u'w'} \cdot \overline{u})$" )
plt.plot(rho_eps, color = 'k', label = r'$\rho(\epsilon)$')
plt.plot(Eps_despiked_combined['Eps'], color = 'gray', label = r'despiked $\rho(\epsilon)$')
plt.title(r"Eps and despiked Eps")
plt.xlim(1500,2000)
plt.legend()
plt.ylim(0,3)

#%%
rho_P_avg_13 = np.array(rho_df['rho_bar_1_dry']*(-1*np.array(UpWp_despiked_combined['UpWp_bar']))*(np.array(sonic3_df['Ubar']-sonic1_df['Ubar'])))
deficit_despike = (rho_P_avg_13)-np.array(Eps_despiked_combined['Eps'])

# deficit_minus_pw = deficit 
plt.figure(figsize=(8,3))
# plt.scatter(np.arange(len(deficit_despike)), rho_P_avg_13, s=10, color = 'b', label = r'$\rho \cdot P$')
# plt.plot(np.arange(len(deficit_despike)), rho_P_avg_13, color = 'b',)
# plt.scatter(np.arange(len(deficit_despike)), Eps_despiked_combined['Eps'], s=10, color = 'navy', label = r'$\rho \cdot \epsilon$')
# plt.plot(np.arange(len(deficit_despike)), Eps_despiked_combined['Eps'], color = 'navy',)
plt.scatter(np.arange(len(deficit_despike)), deficit_despike, s=10, color = 'gray', label = r'$\rho \cdot P -\rho \cdot \epsilon$')
plt.plot(np.arange(len(deficit_despike)), deficit_despike, color = 'gray',)
plt.scatter(np.arange(len(deficit_despike)), pw_df['PW boom-1 [m^3/s^3]'], s=10, color='red', label = 'PW')
plt.plot(np.arange(len(deficit_despike)), pw_df['PW boom-1 [m^3/s^3]'],color='red', )
plt.scatter(np.arange(len(deficit_despike)), deficit_despike+pw_df['PW boom-1 [m^3/s^3]'], s=10, color = 'black', label = r'$(\rho \cdot P -\rho \cdot \epsilon) - PW$')
plt.plot(np.arange(len(deficit_despike)), deficit_despike+pw_df['PW boom-1 [m^3/s^3]'], color = 'black',)
plt.hlines(y=0,xmin=0,xmax=3959,linestyles='--', color = 'k')
plt.legend()
plt.xlim(1500,2000)
plt.ylim(-0.3,0.3)
plt.ylabel("$[m^3/s^3]$")
plt.xlabel('May Storm Time Index')
plt.title('Deficit Despike with PW')


#%%
rho_P_avg_13 = np.array(rho_df['rho_bar_1_dry']*(-1*np.array(UpWp_despiked_combined['UpWp_bar']))*(np.array(sonic3_df['Ubar']-sonic1_df['Ubar'])))
y_spring = np.vstack((eps_df['epsU_sonic1_MAD'][:break_index+1], eps_df['epsU_sonic3_MAD'][:break_index+1])).T
y_fall = np.vstack((eps_df['epsU_sonic1_MAD'][break_index+1:], eps_df['epsU_sonic3_MAD'][break_index+1:])).T
rho_eps_spring = np.array(rho_df['rho_bar_1_dry'][:break_index+1])*np.trapz(y=y_spring, x=None, dx=5.49)#do trapz for between sonics 1-3
rho_eps_fall = np.array(rho_df['rho_bar_1_dry'][break_index+1:])*np.trapz(y=y_fall, x=None, dx=5.0292)#do trapz for between sonics 1-3
rho_eps = np.concatenate((rho_eps_spring, rho_eps_fall), axis=0)
# rho_eps_MAYstorm = rho_eps[storm_index_start:storm_index_stop]
deficit = (rho_P_avg_13)-np.array(rho_eps)
#%%
# deficit_minus_pw = deficit 
plt.figure(figsize=(8,3))
# plt.scatter(np.arange(len(deficit)), rho_P_avg_13, s=10, color = 'b', label = r'$\rho \cdot P$')
# plt.plot(np.arange(len(deficit)), rho_P_avg_13, color = 'b',)
# plt.scatter(np.arange(len(deficit)), rho_eps, s=10, color = 'navy', label = r'$\rho \cdot \epsilon$')
# plt.plot(np.arange(len(deficit)), rho_eps, color = 'navy',)
plt.scatter(np.arange(len(deficit)), deficit, s=10, color = 'gray', label = r'$\rho \cdot P -\rho \cdot \epsilon$')
plt.plot(np.arange(len(deficit)), deficit, color = 'gray',)
plt.scatter(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]'], s=10, color='red', label = 'PW')
plt.plot(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]'],color='red', )
plt.scatter(np.arange(len(deficit)), deficit+pw_df['PW boom-1 [m^3/s^3]'], s=10, color = 'black', label = r'$(\rho \cdot P -\rho \cdot \epsilon) - PW$')
plt.plot(np.arange(len(deficit)), deficit+pw_df['PW boom-1 [m^3/s^3]'], color = 'black',)
plt.hlines(y=0,xmin=0,xmax=3959,linestyles='--', color = 'k')
plt.legend(fontsize=9)
plt.xlim(1500,2000)
plt.ylim(-0.3,0.3)
plt.title('Deficit with PW')

#%%
rho_P_avg_13_spring_df = pd.DataFrame()
rho_P_avg_13_spring_df['rho_P'] = rho_P_avg_13[:break_index+1]

rho_P_avg_13_arr_spring = [rho_P_avg_13_spring_df,]

rho_P_avg_13_spring_despike = pd.DataFrame()

rho_P_avg_13_despike_arr_spring = [rho_P_avg_13_spring_despike,]
column_arr = ['rho_P']

for i in range(len(rho_P_avg_13_arr_spring)):
    for column_name in column_arr:
    
        my_array = rho_P_avg_13_arr_spring[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 5
        n = 2
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        rho_P_avg_13_despike_arr_spring[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel SPRING despike')

#%%
rho_P_avg_13_fall_df = pd.DataFrame()
rho_P_avg_13_fall_df['rho_P'] = rho_P_avg_13[break_index+1:]
rho_P_avg_13_fall_df = rho_P_avg_13_fall_df.reset_index(drop = True)



rho_P_avg_13_arr_fall = [rho_P_avg_13_fall_df,]

rho_P_avg_13_fall_despike = pd.DataFrame()

rho_P_avg_13_despike_arr_fall = [rho_P_avg_13_fall_despike,]
column_arr = ['rho_P']

for i in range(len(rho_P_avg_13_arr_fall)):
    for column_name in column_arr:
    
        my_array = rho_P_avg_13_arr_fall[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 5
        n = 2
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        rho_P_avg_13_despike_arr_fall[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel FALL despike')
#%%
# combine new spring/fall despiked values/dataframes to one combined dataframe
rho_P_avg_13_despiked_combined = pd.concat([rho_P_avg_13_spring_despike,rho_P_avg_13_fall_despike], axis = 0)
rho_P_avg_13_despiked_combined['new_index'] = np.arange(0, len(rho_P_avg_13_despiked_combined))
rho_P_avg_13_despiked_combined = rho_P_avg_13_despiked_combined.set_index('new_index')

print('done combining spring and fall')
#%%
deficit_despike = np.array(rho_P_avg_13_despiked_combined['rho_P'])-np.array(rho_eps)

# deficit_minus_pw = deficit 
plt.figure(figsize=(8,3))
# plt.scatter(np.arange(len(deficit_despike)), rho_P_avg_13_despiked_combined['rho_P'], s=10, color = 'b', label = r'$\rho \cdot P$')
# plt.plot(np.arange(len(deficit_despike)), rho_P_avg_13_despiked_combined['rho_P'], color = 'b',)
# plt.scatter(np.arange(len(deficit_despike)), rho_eps, s=10, color = 'navy', label = r'$\rho \cdot \epsilon$')
# plt.plot(np.arange(len(deficit_despike)), rho_eps, color = 'navy',)
plt.scatter(np.arange(len(deficit_despike)), deficit_despike, s=10, color = 'gray', label = r'$\rho \cdot P -\rho \cdot \epsilon$')
plt.plot(np.arange(len(deficit_despike)), deficit_despike, color = 'gray',)
plt.scatter(np.arange(len(deficit_despike)), pw_df['PW boom-1 [m^3/s^3]'], s=10, color='red', label = 'PW')
plt.plot(np.arange(len(deficit_despike)), pw_df['PW boom-1 [m^3/s^3]'],color='red', )
plt.scatter(np.arange(len(deficit_despike)), deficit_despike+pw_df['PW boom-1 [m^3/s^3]'], s=10, color = 'black', label = r'$(\rho \cdot P -\rho \cdot \epsilon) - PW$')
plt.plot(np.arange(len(deficit_despike)), deficit_despike+pw_df['PW boom-1 [m^3/s^3]'], color = 'black',)
plt.hlines(y=0,xmin=0,xmax=3959,linestyles='--', color = 'k')
plt.legend()
plt.xlim(1500,2000)
plt.ylim(-0.3,0.3)
plt.ylabel("$[m^3/s^3]$")
plt.xlabel('May Storm Time Index')
plt.title('Deficit Despike with PW')
plt.savefig(plot_savePath+'mayStorm_despike31dissipationDeficit_comparisonWithPW.png', dpi = 300)
plt.savefig(plot_savePath+'mayStorm_despike31dissipationDeficit_comparisonWithPW.pdf')

#%%
plt.figure()
# plt.plot((-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = 's1')
# plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = 's2')
plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = '2-1 difference')
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = '3-1 difference')
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = '3-2 difference')
plt.title("$-\overline{u'w'} (\overline{u})$")
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1500,2000)
plt.ylim(-2,10)
plt.ylim(-2,2)
plt.legend()

#%%
plt.figure()
plt.plot(eps_df['epsU_sonic1_MAD'])
plt.title('Dissipation ($\epsilon$)')
plt.xlim(0,break_index)
plt.ylim(0,0.5)
#%%
plt.figure()
# plt.plot((-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = 's1')
# plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = 's2')
plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = '2-1 difference')
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = '3-1 difference')
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = '3-2 difference')
plt.title("$-\overline{u'w'} (\overline{u})$")
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1700,1800)
plt.ylim(-2,10)
plt.ylim(-2,2)
plt.legend()

plt.figure()
plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = "2-1 Production as flux")
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = "3-1 Production as flux")
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = '3-2 difference')
plt.plot(eps_df['epsU_sonic1_MAD']*z_df['z_sonic1'], label = 'Dissipation as flux')
plt.title("Dissipation Flux ($\epsilon_1*z_1$) and Momentum Flux \n $(-\overline{u'w'}_2 (\overline{u}_2)+\overline{u'w'}_1 (\overline{u}_1)$)")
plt.xlim(0,break_index)
plt.ylim(-2,2)
plt.xlim(1500,2000)
plt.ylabel("$[m^3/s^3]$")
plt.xlabel('May Storm Time Index')
plt.legend()
# plt.savefig(plot_savePath + "timeseries_MAYstorm_PandEpsFlux.png", dpi = 300)
# plt.savefig(plot_savePath + "timeseries_MAYstorm_PandEpsFlux.pdf")

#%%
plt.figure()
plt.plot(np.arange(len(z_df)),z_df['z_sonic1'])
plt.title('height of sonic 1 ($z$)')
plt.xlim(0,break_index)
plt.xlim(1500,2000)
#%%
plt.figure()
plt.plot(rho_df['rho_bar_1_dry'], label = 'dry')
plt.plot(rho_df['rho_bar_1_moist'], label = 'moist')
plt.legend()
plt.title(r'air density ($\rho$)')
# plt.xlim(1500,2000)
#%%
plt.figure()
plt.plot(pw_df['PW boom-1 [m^3/s^3]'], color = 'r', label ='$T_{\widetilde{pw}}$')
plt.legend()
plt.title('$T_{\widetilde{pw}}$ flux')
plt.xlim(1500,2000)


#%%
# plt.figure()
# plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar']), label = r"s3 $(-\overline{u'w'})(\overline{u})$")
# plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = r"s2 $(-\overline{u'w'})(\overline{u})$")
# plt.title()

#average this for levels I and II, then do trapZ for sonics 1-3
rho_P_12 = rho_df['rho_bar_1_dry']*((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']))
rho_P_23 = rho_df['rho_bar_2_dry']*((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']))
rho_P_13 = rho_df['rho_bar_2_dry']*((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']))
rho_P_avg = (rho_P_12 + rho_P_23+ rho_P_13) /3


#%%
plt.figure()
plt.plot(rho_P_12, label='rho_P_1-2')
plt.plot(rho_P_23, label='rho_P_2-3')
plt.plot(rho_P_13, label='rho_P_1-3')
plt.plot(rho_P_avg, label='rho_P_avg')
plt.legend()
plt.ylim(-5,5)
plt.xlim(1500,2000)
plt.title(r"$\rho \cdot (-\overline{u'w'})(\overline{u}) \; [Wm^{-2}]$")

#%%
rho_P_12_spring = pd.DataFrame()
rho_P_12_spring['rho_P'] = rho_P_12[:break_index+1]
rho_P_23_spring = pd.DataFrame()
rho_P_23_spring['rho_P'] = rho_P_23[:break_index+1]
rho_P_13_spring = pd.DataFrame()
rho_P_13_spring['rho_P'] = rho_P_13[:break_index+1]


rho_P_arr_spring = [rho_P_12_spring,
                    rho_P_23_spring,
                    rho_P_13_spring]

rho_P_12_spring_despike = pd.DataFrame()
rho_P_23_spring_despike = pd.DataFrame()
rho_P_13_spring_despike = pd.DataFrame()

rho_P_despike_arr_spring = [rho_P_12_spring_despike,
                            rho_P_23_spring_despike,
                            rho_P_13_spring_despike]
column_arr = ['rho_P']

for i in range(len(rho_P_arr_spring)):
    for column_name in column_arr:
    
        my_array = rho_P_arr_spring[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 10
        n = 3
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        rho_P_despike_arr_spring[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel SPRING despike')

#%%
rho_P_12_fall = pd.DataFrame()
rho_P_12_fall['rho_P'] = rho_P_12[break_index+1:]
rho_P_12_fall = rho_P_12_fall.reset_index(drop = True)
rho_P_23_fall = pd.DataFrame()
rho_P_23_fall['rho_P'] = rho_P_23[break_index+1:]
rho_P_23_fall = rho_P_23_fall.reset_index(drop = True)
rho_P_13_fall = pd.DataFrame()
rho_P_13_fall['rho_P'] = rho_P_13[break_index+1:]
rho_P_13_fall = rho_P_13_fall.reset_index(drop = True)


rho_P_arr_fall = [rho_P_12_fall,
                  rho_P_23_fall,
                  rho_P_13_fall]

rho_P_12_fall_despike = pd.DataFrame()
rho_P_23_fall_despike = pd.DataFrame()
rho_P_13_fall_despike = pd.DataFrame()

rho_P_despike_arr_fall = [rho_P_12_fall_despike,
                            rho_P_23_fall_despike,
                            rho_P_13_fall_despike]
column_arr = ['rho_P']

for i in range(len(rho_P_arr_fall)):
    for column_name in column_arr:
    
        my_array = rho_P_arr_fall[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 10
        n = 3
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        rho_P_despike_arr_fall[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel FALL despike')
#%%
# combine new spring/fall despiked values/dataframes to one combined dataframe
rho_P_12_despiked_combined = pd.concat([rho_P_12_spring_despike,rho_P_12_fall_despike], axis = 0)
rho_P_12_despiked_combined['new_index'] = np.arange(0, len(rho_P_12_despiked_combined))
rho_P_12_despiked_combined = rho_P_12_despiked_combined.set_index('new_index')

rho_P_23_despiked_combined = pd.concat([rho_P_23_spring_despike,rho_P_23_fall_despike], axis = 0)
rho_P_23_despiked_combined['new_index'] = np.arange(0, len(rho_P_23_despiked_combined))
rho_P_23_despiked_combined = rho_P_23_despiked_combined.set_index('new_index')

rho_P_13_despiked_combined = pd.concat([rho_P_13_spring_despike,rho_P_13_fall_despike], axis = 0)
rho_P_13_despiked_combined['new_index'] = np.arange(0, len(rho_P_13_despiked_combined))
rho_P_13_despiked_combined = rho_P_13_despiked_combined.set_index('new_index')

print('done combining spring and fall')

#%%

storm_index_start = 1500
storm_index_stop = 2000

rho_P_12_MAYstorm = pd.DataFrame()
rho_P_12_MAYstorm['rho_P'] = rho_P_12[storm_index_start:storm_index_stop]
rho_P_12_MAYstorm = rho_P_12_MAYstorm.reset_index(drop = True)
rho_P_23_MAYstorm = pd.DataFrame()
rho_P_23_MAYstorm['rho_P'] = rho_P_23[storm_index_start:storm_index_stop]
rho_P_23_MAYstorm = rho_P_23_MAYstorm.reset_index(drop = True)
rho_P_13_MAYstorm = pd.DataFrame()
rho_P_13_MAYstorm['rho_P'] = rho_P_13[storm_index_start:storm_index_stop]
rho_P_13_MAYstorm = rho_P_13_MAYstorm.reset_index(drop = True)


rho_P_arr_MAYstorm = [rho_P_12_MAYstorm,
                      rho_P_23_MAYstorm,
                      rho_P_13_MAYstorm,]

rho_P_12_MAYstorm_despike = pd.DataFrame()
rho_P_23_MAYstorm_despike = pd.DataFrame()
rho_P_13_MAYstorm_despike = pd.DataFrame()

rho_P_despike_arr_MAYstorm = [rho_P_12_MAYstorm_despike,
                            rho_P_23_MAYstorm_despike,
                            rho_P_13_MAYstorm_despike]
column_arr = ['rho_P']

for i in range(len(rho_P_arr_MAYstorm)):
    for column_name in column_arr:
    
        my_array = rho_P_arr_MAYstorm[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 15
        n = 1
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        rho_P_despike_arr_MAYstorm[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('done with '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2

print('done hampel MAY storm despike')

#%%
y_spring = np.vstack((eps_df['epsU_sonic1_MAD'][:break_index+1], eps_df['epsU_sonic3_MAD'][:break_index+1])).T
y_fall = np.vstack((eps_df['epsU_sonic1_MAD'][break_index+1:], eps_df['epsU_sonic3_MAD'][break_index+1:])).T
rho_eps_spring = np.array(rho_df['rho_bar_1_dry'][:break_index+1])*np.trapz(y=y_spring, x=None, dx=5.49)#do trapz for between sonics 1-3
rho_eps_fall = np.array(rho_df['rho_bar_1_dry'][break_index+1:])*np.trapz(y=y_fall, x=None, dx=5.0292)#do trapz for between sonics 1-3

rho_eps = np.concatenate((rho_eps_spring, rho_eps_fall), axis=0)
rho_eps_MAYstorm = rho_eps[storm_index_start:storm_index_stop]
#%%
deficit = (np.array(rho_P_23))-np.array(rho_eps)
deficit_despike = np.array(rho_P_23_MAYstorm_despike['rho_P']) - rho_eps_MAYstorm
# deficit_minus_pw = deficit 
plt.figure(figsize=(8,3))
plt.scatter(np.arange(len(deficit_despike)), rho_P_23_MAYstorm_despike, s=10, color = 'b', label = r'$\rho \cdot P$')
plt.plot(np.arange(len(deficit_despike)), rho_P_23_MAYstorm_despike, color = 'b',)
plt.scatter(np.arange(len(deficit_despike)), rho_eps_MAYstorm, s=10, color = 'navy', label = r'$\rho \cdot \epsilon$')
plt.plot(np.arange(len(deficit_despike)), rho_eps_MAYstorm,  color = 'navy', )
plt.scatter(np.arange(len(deficit_despike)), deficit_despike, s=10, color = 'gray', label = r'$(\rho \cdot P)- (\rho \cdot \epsilon)$')
plt.plot(np.arange(len(deficit_despike)), deficit_despike,  color = 'gray',)
# plt.plot(-1*deficit/10, color = 'k', label = '-1*deficit/10')
plt.scatter(np.arange(len(deficit_despike)), pw_df['PW boom-1 [m^3/s^3]'][storm_index_start:storm_index_stop], s=5, color='darkorange', label = '-PW')
plt.plot(np.arange(len(deficit_despike)), pw_df['PW boom-1 [m^3/s^3]'][storm_index_start:storm_index_stop],color='darkorange',)
plt.scatter(np.arange(len(deficit_despike)), pw_df['PW boom-1 [m^3/s^3]'][storm_index_start:storm_index_stop]*-10, s=10, color='red', label = '-PW*10')
plt.plot(np.arange(len(deficit_despike)), pw_df['PW boom-1 [m^3/s^3]'][storm_index_start:storm_index_stop]*-10,color='red', )
# plt.scatter(np.arange(len(deficit)),(-1*deficit/10)-pw_df['PW boom-1 [m^3/s^3]'],color='green', label = 'deficit+PW')
plt.legend()
plt.ylim(-1,3.5)
plt.xlim(0,break_index)
plt.xlim(0,500)
plt.ylabel('$[m^3/s^3]$')
plt.xlabel('May Storm Index')
plt.title('Dissipation Flux Deficit versus Wave Coherent PW')
plt.savefig(plot_savePath+'mayStorm_despike32dissipationDeficit_comparisonWithPW.png', dpi = 300)
plt.savefig(plot_savePath+'mayStorm_despike32dissipationDeficit_comparisonWithPW.pdf')

#%%
deficit = (np.array(rho_P_avg))-np.array(rho_eps)
# deficit_minus_pw = deficit 
plt.figure(figsize=(8,3))
plt.scatter(np.arange(len(deficit)), rho_P_avg, s=10, color = 'b', label = r'$\rho \cdot P$')
plt.plot(np.arange(len(deficit)), rho_P_avg, color = 'b',)
plt.scatter(np.arange(len(deficit)), rho_eps, s=10, color = 'navy', label = r'$\rho \cdot \epsilon$')
plt.plot(np.arange(len(deficit)), rho_eps,  color = 'navy', )
plt.scatter(np.arange(len(deficit)), deficit, s=10, color = 'gray', label = r'$(\rho \cdot P)- (\rho \cdot \epsilon)$')
plt.plot(np.arange(len(deficit)), deficit,  color = 'gray',)
# plt.plot(-1*deficit/10, color = 'k', label = '-1*deficit/10')
plt.scatter(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]'], s=5, color='darkorange', label = '-PW')
plt.plot(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]'],color='darkorange',)
plt.scatter(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]']*-10, s=10, color='red', label = '-PW*10')
plt.plot(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]']*-10,color='red', )
# plt.scatter(np.arange(len(deficit)),(-1*deficit/10)-pw_df['PW boom-1 [m^3/s^3]'],color='green', label = 'deficit+PW')
plt.legend()
plt.ylim(-1,3.5)
plt.xlim(0,break_index)
plt.xlim(1500,2000)
plt.ylabel('$[m^3/s^3]$')
plt.xlabel('May Storm Index')
plt.title('Dissipation Flux Deficit versus Wave Coherent PW')
plt.savefig(plot_savePath+'mayStorm_AVGdissipationDeficit_comparisonWithPW.png', dpi = 300)
plt.savefig(plot_savePath+'mayStorm_AVGdissipationDeficit_comparisonWithPW.pdf')






