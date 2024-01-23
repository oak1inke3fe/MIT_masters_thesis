# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:39:58 2023

@author: oaklin keefe

This file is used to calculate monin obukhov mixing lengthscale, L, and then determine a stability parameter z/L (zeta)

Input file location:
    /code_pipeline/Level2/
INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    prodTerm_combinedAnalysis.csv
    z_air_side_combinedAnalysis.csv
    thetaV_combinedAnalysis.csv
    
Output file location:
    /code_pipeline/Level2/
OUTPUT files:
    ZoverL_combinedAnalysis.csv
    
"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hampel import hampel
# import seaborn as sns
print('done with imports')

#%%
g = -9.81
kappa = 0.4

file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic1_df = sonic1_df.drop(['new_index'], axis=1)
# print(sonic1_df.columns)


sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic2_df = sonic2_df.drop(['new_index'], axis=1)


sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)
sonic3_df = sonic3_df.drop(['new_index'], axis=1)


sonic_file4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)
sonic4_df = sonic4_df.drop(['new_index'], axis=1)

print('done reading in sonics')

#%%

prod_df = prod_df = pd.read_csv(file_path+'prodTerm_combinedAnalysis.csv')
prod_df = prod_df.drop(['Unnamed: 0'], axis=1)

print('done reading in production terms')
#%%

# z_df_spring = pd.read_csv(file_path+'z_airSide_allSpring.csv')
# z_df_spring = z_df_spring.drop(['Unnamed: 0'], axis=1)
# plt.figure()
# plt.plot(z_df_spring['z_sonic1'])
# plt.title('Spring')

# z_df_fall = pd.read_csv(file_path+'z_airSide_allFall.csv')
# z_df_fall = z_df_fall.drop(['Unnamed: 0'], axis=1)
# plt.figure()
# plt.plot(z_df_fall['z_sonic1'])
# plt.title('Fall')

# z_df = pd.concat([z_df_spring, z_df_fall], axis=0)
# z_df['new_index'] = np.arange(0, len(z_df))
# z_df = z_df.set_index('new_index')

# # z_df.to_csv(file_path + 'z_air_side_combinedAnalysis.csv')

z_df = pd.read_csv(file_path + 'z_air_side_combinedAnalysis.csv')
z_df = z_df.drop(['new_index'], axis=1)

plt.figure()
plt.plot(z_df['z_sonic1'])
plt.title('Combined')


# file_dissipation = "epsU_terms_combinedAnalysis_MAD_k_UoverZ_Puu.csv"
# Eps_df = pd.read_csv(file_path+file_dissipation)
# Eps_df = Eps_df.drop(['Unnamed: 0'], axis=1)


# tke_transport_df = pd.read_csv(file_path + "tke_transport_allFall.csv")
# tke_transport_df = tke_transport_df.drop(['Unnamed: 0'], axis=1)

# met_df = pd.read_csv(file_path + "metAvg_allFall.csv")
# met_df = met_df.iloc[27:]
# met_df = met_df.reset_index()
# met_df = met_df.drop(['index'], axis=1)
# met_df = met_df.drop(['Unnamed: 0'], axis=1)

thetaV_df = pd.read_csv(file_path + "thetaV_combinedAnalysis.csv")
thetaV_df = thetaV_df.drop(['Unnamed: 0'], axis=1)

# windDir_df = pd.read_csv(file_path + "windDir_withBadFlags.csv")
# windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

# rho_df = pd.read_csv(file_path + 'rho_bar_allFall.csv' )
# rho_df = rho_df.iloc[27:]
# rho_df = rho_df.reset_index()
# rho_df = rho_df.drop(['index'], axis=1)
# rho_df = rho_df.drop(['Unnamed: 0'], axis=1)

#%%

plt.figure()
plt.plot(sonic4_df['Ubar'], label = 'sonic 4')
plt.plot(sonic3_df['Ubar'], label = 'sonic 3')
plt.plot(sonic2_df['Ubar'], label = 'sonic 2')
plt.plot(sonic1_df['Ubar'], label = 'sonic 1')
plt.legend()
# plt.xlim(1400,1800)
plt.title("<u> time series (despiked)")
#%%
plt.figure()
plt.plot(sonic4_df['UpWp_bar'], label = 'sonic 4')
plt.plot(sonic3_df['UpWp_bar'], label = 'sonic 3')
plt.plot(sonic2_df['UpWp_bar'], label = 'sonic 2')
plt.plot(sonic1_df['UpWp_bar'], label = 'sonic 1')
plt.legend()
plt.title("<u'w'> time series (despiked)")

plt.figure()
plt.plot(sonic4_df['VpWp_bar'], label = 'sonic 4')
plt.plot(sonic3_df['VpWp_bar'], label = 'sonic 3')
plt.plot(sonic2_df['VpWp_bar'], label = 'sonic 2')
plt.plot(sonic1_df['VpWp_bar'], label = 'sonic 1')
plt.legend()
plt.title("<v'w'> time series (despiked)")

plt.figure()
plt.plot(sonic4_df['WpEp_bar'], label = 'sonic 4')
plt.plot(sonic3_df['WpEp_bar'], label = 'sonic 3')
plt.plot(sonic2_df['WpEp_bar'], label = 'sonic 2')
plt.plot(sonic1_df['WpEp_bar'], label = 'sonic 1')
plt.legend()
plt.title("<w'E'> time series (despiked)")

plt.figure()
plt.plot(sonic4_df['WpTp_bar'], label = 'sonic 4')
plt.plot(sonic3_df['WpTp_bar'], label = 'sonic 3')
plt.plot(sonic2_df['WpTp_bar'], label = 'sonic 2')
plt.plot(sonic1_df['WpTp_bar'], label = 'sonic 1')
plt.legend()
plt.title("<w'T'> time series (despiked)")

plt.figure()
plt.plot(sonic4_df['Tbar'], label = 'sonic 4')
plt.plot(sonic3_df['Tbar'], label = 'sonic 3')
plt.plot(sonic2_df['Tbar'], label = 'sonic 2')
plt.plot(sonic1_df['Tbar'], label = 'sonic 1')
plt.legend()
plt.title("<T> time series (despiked)")



# plt.figure()
# plt.plot(rho_df['rho_bar_1'], label = '<rho> 1')
# plt.plot(rho_df['rho_bar_2'], label = '<rho> 2')
# plt.legend()
# plt.title("<rho> time series")
# plt.xlim(800,1600)

#plotting relative humidity to figure out when it was raining
# plt.figure()
# plt.plot(met_df['rh2'], label = 'rh2')
# plt.plot(met_df['rh1'], label = 'rh1')
# plt.legend()
# plt.title("RH time series (with spikes)")
# plt.ylim(90,101)
# plt.xlim(800,1600)



#%%

# usr_s1_withRho = (1/rho_df['rho_bar_1'])*((sonic1_df_despiked['UpWp_bar'])**2+(sonic1_df_despiked['VpWp_bar'])**2)**(1/4)
usr_s1 = ((sonic1_df['UpWp_bar'])**2+(sonic1_df['VpWp_bar'])**2)**(1/4)

usr_s2 = ((sonic2_df['UpWp_bar'])**2+(sonic2_df['VpWp_bar'])**2)**(1/4)

usr_s3 = ((sonic3_df['UpWp_bar'])**2+(sonic3_df['VpWp_bar'])**2)**(1/4)

usr_s4 = ((sonic4_df['UpWp_bar'])**2+(sonic4_df['VpWp_bar'])**2)**(1/4)

USTAR_df = pd.DataFrame()
USTAR_df['usr_s1'] = np.array(usr_s1)
USTAR_df['usr_s2'] = np.array(usr_s2)
USTAR_df['usr_s3'] = np.array(usr_s3)
USTAR_df['usr_s4'] = np.array(usr_s4)
# USTAR_df.to_csv(file_path + 'usr_combinedAnalysis.csv') #we already have another file where we do this

plt.figure()
plt.plot(usr_s1, label = "u*_{s1} = $(<u'w'>^{2} + <v'w'>^{2})^{1/4}$")
plt.legend()
plt.title('U*')

#%%

z_s1 = z_df['z_sonic1']
z_s2 = z_df['z_sonic2']
z_s3 = z_df['z_sonic3']
z_s4 = z_df['z_sonic4']



plt.figure()
plt.plot(z_s4, label = 'sonic 4')
plt.plot(z_s3, label = 'sonic 3')
plt.plot(z_s2, label = 'sonic 2')
plt.plot(z_s1, label = 'sonic 1')
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.title('timeseries of z [m]')
# plt.xlim(4000,7000)
plt.xlabel('time')
plt.ylabel('height (z) [m]')



#%%

usr_LI = np.array(usr_s1+usr_s2)/2
Tbar_LI = np.array(sonic1_df['Tbar']+sonic2_df['Tbar'])/2
WpTp_bar_LI = -1*(np.array(sonic1_df['WpTp_bar']+sonic2_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

usr_LII = np.array(usr_s2+usr_s3)/2
Tbar_LII = np.array(sonic2_df['Tbar']+sonic3_df['Tbar'])/2
WpTp_bar_LII = -1*(np.array(sonic2_df['WpTp_bar']+sonic3_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

usr_LIII = np.array(usr_s3+usr_s4)/2
Tbar_LIII = np.array(sonic3_df['Tbar']+sonic4_df['Tbar'])/2
WpTp_bar_LIII = -1*(np.array(sonic3_df['WpTp_bar']+sonic4_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

print('done getting ustar etc. to Levels')


#%%
plt.figure()
plt.plot(usr_s1, label = 's1')
plt.plot(usr_s2, label = 's2')
plt.plot(usr_s3, label = 's3')
plt.plot(usr_s4, label = 's4')
plt.legend()
plt.title('U_star intra-sonic comparison')

plt.figure()
plt.plot(sonic1_df['WpTp_bar'], label = 's1')
plt.plot(sonic2_df['WpTp_bar'], label = 's2')
plt.plot(sonic3_df['WpTp_bar'], label = 's3')
plt.plot(sonic4_df['WpTp_bar'], label = 's4')
plt.legend()
plt.ylim(-0.5,0.5)
plt.title('WpTp_bar intra-sonic comparison')

plt.figure()
plt.plot(thetaV_df['thetaV_sonic1'], label = 's1')
plt.plot(thetaV_df['thetaV_sonic2'], label = 's2')
plt.plot(thetaV_df['thetaV_sonic3'], label = 's3')
plt.plot(thetaV_df['thetaV_sonic4'], label = 's4')
plt.legend()
plt.title('Theta_V intra-sonic comparison')

print('done with plots of Ustar and WpTp')

#%%
## calculate L: L = -ustar^3*<Tv> / [g*kappa*<w'Tv'>]
'''
NOTE: because COARE v3.6 defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

L_1_dc_unspiked = -1*(np.array(usr_s1)**3)*np.array(thetaV_df['thetaV_sonic1'])/(g*kappa*(-1*np.array(sonic1_df['WpTp_bar'])))


L_2_dc_unspiked = -1*(np.array(usr_s2)**3)*np.array(thetaV_df['thetaV_sonic2'])/(g*kappa*(-1*np.array(sonic2_df['WpTp_bar'])))


L_3_dc_unspiked = -1*(np.array(usr_s3)**3)*np.array(thetaV_df['thetaV_sonic3'])/(g*kappa*(-1*np.array(sonic3_df['WpTp_bar'])))


L_4_dc_unspiked = -1*(np.array(usr_s4)**3)*np.array(thetaV_df['thetaV_sonic4'])/(g*kappa*(-1*np.array(sonic4_df['WpTp_bar'])))


L_dc_df_unspiked = pd.DataFrame()
L_dc_df_unspiked['L_sonic1'] = np.array(L_1_dc_unspiked)
L_dc_df_unspiked['L_sonic2'] = np.array(L_2_dc_unspiked)
L_dc_df_unspiked['L_sonic3'] = np.array(L_3_dc_unspiked)
L_dc_df_unspiked['L_sonic4'] = np.array(L_4_dc_unspiked)


plt.figure()
plt.plot(L_4_dc_unspiked, label = 'dc s4')
plt.legend()
plt.title('comparison of L: COARE vs. DC')




#%%
sonic_arr = ['1','2','3','4']

break_index = 3959
L_dc_df_unspiked_spring = L_dc_df_unspiked[:break_index+1]
L_dc_df_unspiked_spring = L_dc_df_unspiked_spring.reset_index(drop = True)
L_dc_df_unspiked_fall = L_dc_df_unspiked[break_index+1:]
L_dc_df_unspiked_fall = L_dc_df_unspiked_fall.reset_index(drop = True)

L_dc_df_spring = pd.DataFrame()
L_dc_df_fall = pd.DataFrame()

for sonic in sonic_arr:
    L_array = L_dc_df_unspiked_spring['L_sonic'+sonic]
    
    # Just outlier detection
    input_array = L_array
    window_size = 10
    n = 3
    
    L_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
    L_despiked_1times = L_outlier_in_Ts
    
    # plt.figure()
    # plt.plot(L_despiked_once)

    input_array2 = L_despiked_1times
    L_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)

    L_dc_df_spring['L_sonic'+sonic] = L_outlier_in_Ts2
    print("dc: "+str(sonic))
    # L_despiked_2times = L_outlier_in_Ts2
print('done with SPRING')
    
for sonic in sonic_arr:
    L_array = L_dc_df_unspiked_fall['L_sonic'+sonic]
    
    # Just outlier detection
    input_array = L_array
    window_size = 10
    n = 3
    
    L_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
    L_despiked_1times = L_outlier_in_Ts
    
    # plt.figure()
    # plt.plot(L_despiked_once)

    input_array2 = L_despiked_1times
    L_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)

    L_dc_df_fall['L_sonic'+sonic] = L_outlier_in_Ts2
    print("dc: "+str(sonic))
    # L_despiked_2times = L_outlier_in_Ts2
print('done with FALL')


L_dc_df = pd.concat([L_dc_df_spring,L_dc_df_fall], axis = 0)
L_dc_df['new_index'] = np.arange(0, len(L_dc_df))
L_dc_df = L_dc_df.set_index('new_index')

L_dc_df.to_csv(file_path+'L_dc_combinedAnalysis.csv')

#%%
plt.figure()
plt.plot(L_4_dc_unspiked, label = 'orig.', color = 'gray')
plt.plot(L_dc_df['L_sonic4'], label = 'despiked', color = 'black')
plt.legend()
plt.title('Sonic 4: comparison of L spiked and despiked')
#%%

L_1_dc = L_dc_df['L_sonic1']
L_2_dc = L_dc_df['L_sonic2']
L_3_dc = L_dc_df['L_sonic3']
L_4_dc = L_dc_df['L_sonic4']

r_L_dc = L_dc_df.corr()
print(r_L_dc)

#%%
break_index = 3959
dz_LI_spring = 2.695  #sonic 2- sonic 1: spring APRIL 2022 deployment
dz_LII_spring = 2.795 #sonic 3- sonic 2: spring APRIL 2022 deployment
dz_LIII_spring = 2.415 #sonic 4- sonic 3: spring APRIL 2022 deployment
dz_LI_fall = 1.8161  #sonic 2- sonic 1: FALL SEPT 2022 deployment
dz_LII_fall = 3.2131 #sonic 3- sonic 2: FALL SEPT 2022 deployment
dz_LIII_fall = 2.468 #sonic 4- sonic 3: FALL SEPT 2022 deployment

z_I_spring = np.array(np.array(z_s1)[break_index+1:]+(0.5*dz_LI_spring))
z_II_spring = np.array(np.array(z_s2)[break_index+1:]+(0.5*dz_LII_spring))
z_III_spring = np.array(np.array(z_s3)[break_index+1:]+(0.5*dz_LIII_spring))

z_I_fall = np.array(np.array(z_s1)[:break_index+1]+(0.5*dz_LI_fall))
z_II_fall = np.array(np.array(z_s2)[:break_index+1]+(0.5*dz_LII_fall))
z_III_fall = np.array(np.array(z_s3)[:break_index+1]+(0.5*dz_LIII_fall))

z_I = np.concatenate([z_I_spring, z_I_fall], axis = 0)
z_II = np.concatenate([z_II_spring, z_II_fall], axis = 0)
z_III = np.concatenate([z_III_spring, z_III_fall], axis = 0)
z_over_L_I_dc = np.array(z_I)/np.array(0.5*(L_1_dc+L_2_dc))
z_over_L_II_dc = np.array(z_II)/np.array(0.5*(L_2_dc+L_3_dc))
z_over_L_III_dc = np.array(z_III)/np.array(0.5*(L_3_dc+L_4_dc))
#%%
z_over_L_df = pd.DataFrame()
z_over_L_df['zL_I_dc'] = z_over_L_I_dc
z_over_L_df['zL_II_dc'] = z_over_L_II_dc
z_over_L_df['zL_III_dc'] = z_over_L_III_dc
z_over_L_df['zL_1_dc'] = np.array(z_s1/L_1_dc)
z_over_L_df['zL_2_dc'] = np.array(z_s1/L_2_dc)
z_over_L_df['zL_3_dc'] = np.array(z_s1/L_3_dc)
z_over_L_df['zL_4_dc'] = np.array(z_s1/L_4_dc)

z_over_L_df.to_csv(file_path + 'ZoverL_combinedAnalysis.csv')
print('saved to .csv')



#%%


plt.figure()
plt.plot(z_over_L_I_dc, label = 'I')
plt.plot(z_over_L_II_dc, label = 'II')
plt.plot(z_over_L_III_dc, label = 'III')
plt.legend()
plt.ylim(-30,5)
plt.title('z/L DC by level')



#%%
