# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:04:24 2023

@author: oaklin keefe

This file is used to plot our data against the universal function curves from Edson 1998.

INPUT files:
    prodTerm_combinedAnalysis.csv
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    buoy_terms_combinedAnalysis.csv
    z_airSide_allSpring.csv
    z_airSide_allFall.csv
    ZoverL_combinedAnalysis.csv
    usr_combinedAnalysis.csv
    rhoAvg_combinedAnalysis.csv
    windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv
    
    
OUTPUT files:
    Only figures
        

"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import binsreg
import seaborn as sns
import os
print('done with imports')

print(os.getcwd())
os.chdir('/Users/oaklinkeefe/Documents/GitHub/masters_thesis/')
print(os.getcwd())

# from binscatter.py import binscatterplot
# name = "binscatter"
# print('done importing binscatter')

g = -9.81
kappa = 0.40 # von Karman constant used by Edson (1998) Similarity Theory paper
# kappa =0.35 # Businger (1971) paper
print('done with setting gravity (g = -9.81) and von-karman (kappa = 4)')

#%%
# file_path = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/"
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'

sonic1_df = pd.read_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df = pd.read_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df = pd.read_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df = pd.read_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_combined.csv')

z_df_spring = pd.read_csv(file_path + "z_airSide_allSpring.csv")
z_df_spring = z_df_spring.drop(['Unnamed: 0'], axis=1)

z_df_fall = pd.read_csv(file_path + "z_airSide_allFall.csv")
z_df_fall = z_df_fall.drop(['Unnamed: 0'], axis=1)

z_df = pd.concat([z_df_spring, z_df_fall], axis = 0)
print('done with z concat')
#%%

Eps_df = pd.read_csv(file_path+"epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv")
Eps_df = Eps_df.drop(['Unnamed: 0'], axis=1)


tke_transport_df = pd.read_csv(file_path + "tke_transport_combinedAnalysis.csv")
tke_transport_df = tke_transport_df.drop(['Unnamed: 0'], axis=1)


windDir_file = "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv"
windDir_df = pd.read_csv(file_path + windDir_file)
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

zL_df = pd.read_csv(file_path+'ZoverL_combinedAnalysis.csv')
zL_df = zL_df.drop(['Unnamed: 0'], axis=1)

rho_df = pd.read_csv(file_path + 'rhoAvg_combinedAnalysis.csv' )
rho_df = rho_df.drop(['Unnamed: 0'], axis=1)

usr_df = pd.read_csv(file_path + 'usr_combinedAnalysis.csv')
usr_df = usr_df.drop(['Unnamed: 0'], axis=1)

buoy_df = pd.read_csv(file_path+'buoy_terms_combinedAnalysis.csv')
buoy_df = buoy_df.drop(['Unnamed: 0'], axis=1)

oct_storm = False

print('done with loading rest of files')
#%% Mask the DFs to only keep the good wind directions
all_windDirs = True
onshore = False
offshore = False

windDir_index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, windDir_index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],windDir_index_array)

windDir_df[mask_goodWindDir] = np.nan

sonic1_df[mask_goodWindDir] = np.nan
sonic2_df[mask_goodWindDir] = np.nan
sonic3_df[mask_goodWindDir] = np.nan
sonic4_df[mask_goodWindDir] = np.nan

zL_df[mask_goodWindDir] = np.nan

z_df[mask_goodWindDir] = np.nan

Eps_df[mask_goodWindDir] = np.nan

rho_df[mask_goodWindDir] = np.nan

buoy_df[mask_goodWindDir] = np.nan

tke_transport_df[mask_goodWindDir] = np.nan

print('done with setting up good wind direction only dataframes')




#%% Offshore setting
# all_windDirs = False
# onshore = False
# offshore = True

# windDir_df['offshore_index_arr'] = np.arange(len(windDir_df))
# windDir_df['new_offshore_index_arr'] = np.where((windDir_df['alpha_s4'] >= 270)&(windDir_df['alpha_s4'] <= 359), windDir_df['offshore_index_arr'], np.nan)

# mask_offshoreWindDir = np.isin(windDir_df['new_offshore_index_arr'],windDir_df['offshore_index_arr'])
# windDir_df = windDir_df[mask_offshoreWindDir]

# prod_df[mask_offshoreWindDir] = np.nan
# Eps_df[mask_offshoreWindDir] = np.nan
# sonic1_df[mask_offshoreWindDir] = np.nan
# sonic2_df[mask_offshoreWindDir] = np.nan
# sonic3_df[mask_offshoreWindDir] = np.nan
# sonic4_df[mask_offshoreWindDir] = np.nan
# tke_transport_df[mask_offshoreWindDir] = np.nan
# z_df[mask_offshoreWindDir] = np.nan
# zL_df[mask_offshoreWindDir] = np.nan
# wave_df[mask_offshoreWindDir] = np.nan

#%% On-shore setting
# all_windDirs = False
# onshore = True
# offshore = False

# windDir_df['onshore_index_arr'] = np.arange(len(windDir_df))
# windDir_df['new_onshore_index_arr'] = np.where((windDir_df['alpha_s4'] >= 197)&(windDir_df['alpha_s4'] <= 269), windDir_df['onshore_index_arr'], np.nan)

# mask_onshoreWindDir = np.isin(windDir_df['new_onshore_index_arr'],windDir_df['onshore_index_arr'])
# windDir_df = windDir_df[mask_onshoreWindDir]

# prod_df[mask_onshoreWindDir] = np.nan
# Eps_df[mask_onshoreWindDir] = np.nan
# sonic1_df[mask_onshoreWindDir] = np.nan
# sonic2_df[mask_onshoreWindDir] = np.nan
# sonic3_df[mask_onshoreWindDir] = np.nan
# sonic4_df[mask_onshoreWindDir] = np.nan
# tke_transport_df[mask_onshoreWindDir] = np.nan
# z_df[mask_onshoreWindDir] = np.nan
# zL_df[mask_onshoreWindDir] = np.nan
# wave_df[mask_onshoreWindDir] = np.nan

#%%
# figure titles based on wind directions
if all_windDirs == True:
    title_windDir = 'All wind directions; '
    wave_df_save_name = '_allGoodWindDirs_'
elif onshore == True:
    title_windDir = 'Onshore winds; '
    wave_df_save_name = '_onshoreWindDirs_'
elif offshore == True:
    title_windDir = 'Offshore winds; '
    wave_df_save_name = '_offshoreWindDirs_'
if oct_storm == True:
    oct_addition = "Oct 2-4: "
else:
    oct_addition = ''
#%%

usr_LI = np.array(usr_df['usr_s1']+usr_df['usr_s2'])/2
Tbar_LI = np.array(sonic1_df['Tbar']+sonic2_df['Tbar'])/2
WpTp_bar_LI = -1*(np.array(sonic1_df['WpTp_bar']+sonic2_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

usr_LII = np.array(usr_df['usr_s2']+usr_df['usr_s3'])/2
Tbar_LII = np.array(sonic2_df['Tbar']+sonic3_df['Tbar'])/2
WpTp_bar_LII = -1*(np.array(sonic2_df['WpTp_bar']+sonic3_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

usr_LIII = np.array(usr_df['usr_s3']+usr_df['usr_s4'])/2
Tbar_LIII = np.array(sonic3_df['Tbar']+sonic4_df['Tbar'])/2
WpTp_bar_LIII = -1*(np.array(sonic3_df['WpTp_bar']+sonic4_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

#%%
usr_dc_df = pd.DataFrame()
usr_dc_df['usr_s1'] = usr_df['usr_s1']
usr_dc_df['usr_s2'] = usr_df['usr_s2']
usr_dc_df['usr_s3'] = usr_df['usr_s3']
usr_dc_df['usr_s4'] = usr_df['usr_s4']
usr_dc_df['usr_LI'] = usr_LI
usr_dc_df['usr_LII'] = usr_LII
usr_dc_df['usr_LIII'] = usr_LIII
# usr_dc_df['usr_s1'] = usr_df['usr_s1_noV']
# usr_dc_df['usr_s2'] = usr_df['usr_s2_noV']
# usr_dc_df['usr_s3'] = usr_df['usr_s3_noV']
# usr_dc_df['usr_s4'] = usr_df['usr_s4_noV']
# usr_dc_df['usr_LI'] = np.array(usr_df['usr_s1_noV']+usr_df['usr_s2_noV'])/2
# usr_dc_df['usr_LII'] = np.array(usr_df['usr_s2_noV']+usr_df['usr_s3_noV'])/2
# usr_dc_df['usr_LIII'] = np.array(usr_df['usr_s3_noV']+usr_df['usr_s4_noV'])/2
usr_dc_df['usr_4_1'] = np.array(usr_df['usr_s4']+usr_df['usr_s1'])/2
usr_dc_df['usr_3_1'] = np.array(usr_df['usr_s3']+usr_df['usr_s1'])/2
usr_dc_df['usr_4_2'] = np.array(usr_df['usr_s4']+usr_df['usr_s2'])/2


#%%
dz_LI_spring = 2.695  #sonic 2- sonic 1: spring APRIL 2022 deployment
dz_LII_spring = 2.795 #sonic 3- sonic 2: spring APRIL 2022 deployment
dz_LIII_spring = 2.415 #sonic 4- sonic 3: spring APRIL 2022 deployment
dz_4_1_spring = 7.904 #sonic 4 - sonic 1: spring APRIL 2022 deployment
dz_3_1_spring = 5.490 #sonic 3 - sonic 1: spring APRIL 2022 deployment
dz_4_2_spring = 5.21 #sonic 4 - sonic 3: spring APRIL 2022 deployment

dz_LI_fall = 1.8161  #sonic 2- sonic 1: FALL SEPT 2022 deployment
dz_LII_fall = 3.2131 #sonic 3- sonic 2: FALL SEPT 2022 deployment
dz_LIII_fall = 2.468 #sonic 4- sonic 3: FALL SEPT 2022 deployment
dz_4_1_fall = 7.4972 #sonic 4 - sonic 1: FALL SEPT 2022 deployment
dz_3_1_fall = 5.0292 #sonic 3 - sonic 1: FALL SEPT 2022 deployment
dz_4_2_fall = 5.6811 #sonic 4 - sonic 2: FALL SEPT 2022 deployment

break_index = 3959 #index is 3959, full length is 3960

z_LI_spring = z_df_spring['z_sonic1']+(0.5*dz_LI_spring)
z_LII_spring  = z_df_spring['z_sonic2']+(0.5*dz_LII_spring)
z_LIII_spring  = z_df_spring['z_sonic3']+(0.5*dz_LIII_spring)
z_4_1_spring = z_df_spring['z_sonic1']+(0.5*dz_4_1_spring)
z_3_1_spring = z_df_spring['z_sonic1']+(0.5*dz_3_1_spring)
z_4_2_spring = z_df_spring['z_sonic2']+(0.5*dz_4_2_spring)

z_LI_fall = z_df_fall['z_sonic1']+(0.5*dz_LI_fall)
z_LII_fall = z_df_fall['z_sonic2']+(0.5*dz_LII_fall)
z_LIII_fall = z_df_fall['z_sonic3']+(0.5*dz_LIII_fall)
z_4_1_fall = z_df_fall['z_sonic1']+(0.5*dz_4_1_fall)
z_3_1_fall = z_df_fall['z_sonic1']+(0.5*dz_3_1_fall)
z_4_2_fall = z_df_fall['z_sonic2']+(0.5*dz_4_2_fall)

z_LI = np.concatenate([z_LI_spring, z_LI_fall], axis = 0)
z_LII = np.concatenate([z_LII_spring, z_LII_fall], axis = 0)
z_LIII = np.concatenate([z_LIII_spring, z_LIII_fall], axis = 0)
z_4_1 = np.concatenate([z_4_1_spring, z_4_1_fall], axis = 0)
z_3_1 = np.concatenate([z_3_1_spring, z_3_1_fall], axis = 0)
z_4_2 = np.concatenate([z_4_2_spring, z_4_2_fall], axis = 0)



dUbardz_LI_spring = np.array(sonic2_df['Ubar'][:break_index+1]-sonic1_df['Ubar'][:break_index+1])/dz_LI_spring 
dUbardz_LII_spring  = np.array(sonic3_df['Ubar'][:break_index+1]-sonic2_df['Ubar'][:break_index+1])/dz_LII_spring 
dUbardz_LIII_spring  = np.array(sonic4_df['Ubar'][:break_index+1]-sonic3_df['Ubar'][:break_index+1])/dz_LIII_spring 
dUbardz_4_1_spring = np.array(sonic4_df['Ubar'][:break_index+1]-sonic1_df['Ubar'][:break_index+1])/dz_4_1_spring 
dUbardz_3_1_spring = np.array(sonic3_df['Ubar'][:break_index+1]-sonic1_df['Ubar'][:break_index+1])/dz_3_1_spring 
dUbardz_4_2_spring = np.array(sonic4_df['Ubar'][:break_index+1]-sonic2_df['Ubar'][:break_index+1])/dz_4_2_spring 

dUbardz_LI_fall = np.array(sonic2_df['Ubar'][break_index+1:]-sonic1_df['Ubar'][break_index+1:])/dz_LI_fall
dUbardz_LII_fall = np.array(sonic3_df['Ubar'][break_index+1:]-sonic2_df['Ubar'][break_index+1:])/dz_LII_fall
dUbardz_LIII_fall = np.array(sonic4_df['Ubar'][break_index+1:]-sonic3_df['Ubar'][break_index+1:])/dz_LIII_fall
dUbardz_4_1_fall = np.array(sonic4_df['Ubar'][break_index+1:]-sonic1_df['Ubar'][break_index+1:])/dz_4_1_fall
dUbardz_3_1_fall = np.array(sonic3_df['Ubar'][break_index+1:]-sonic1_df['Ubar'][break_index+1:])/dz_3_1_fall
dUbardz_4_2_fall = np.array(sonic4_df['Ubar'][break_index+1:]-sonic2_df['Ubar'][break_index+1:])/dz_4_2_fall

dUbardz_LI = np.concatenate([dUbardz_LI_spring, dUbardz_LI_fall], axis = 0)
dUbardz_LII = np.concatenate([dUbardz_LII_spring, dUbardz_LII_fall], axis = 0)
dUbardz_LIII = np.concatenate([dUbardz_LIII_spring, dUbardz_LIII_fall], axis = 0)
dUbardz_4_1 = np.concatenate([dUbardz_4_1_spring, dUbardz_4_1_fall], axis = 0)
dUbardz_3_1 = np.concatenate([dUbardz_3_1_spring, dUbardz_3_1_fall], axis = 0)
dUbardz_4_2 = np.concatenate([dUbardz_4_2_spring, dUbardz_4_2_fall], axis = 0)

print('done with ustar, z/L, and dUbardz')

#%%
coare_zL_neg = np.arange(-4,0,0.005)
coare_zL_pos = np.arange(0,2,0.005)

coare_zL_combined = np.concatenate([coare_zL_neg,coare_zL_pos], axis=0)

print('done getting x-axis z/L array for functional forms to be plotted on a continuous nature')
#%%
######################################################################
######################################################################
"""
NOW WE ARE MOVING ON TO PHI_M CALCULATIONS
"""


phi_m_I_dc = kappa*np.array(z_LI)/(np.array(usr_LI))*(np.array(dUbardz_LI))
phi_m_II_dc = kappa*np.array(z_LII)/(np.array(usr_LII))*(np.array(dUbardz_LII))
phi_m_III_dc = kappa*np.array(z_LIII)/(np.array(usr_LIII))*(np.array(dUbardz_LIII))
phi_m_4_1_dc = kappa*np.array(z_4_1)/(np.array(usr_dc_df['usr_4_1']))*(np.array(dUbardz_4_1))
phi_m_3_1_dc = kappa*np.array(z_3_1)/(np.array(usr_dc_df['usr_3_1']))*(np.array(dUbardz_3_1))
phi_m_4_2_dc = kappa*np.array(z_4_2)/(np.array(usr_dc_df['usr_4_2']))*(np.array(dUbardz_4_2))

phi_m_dc_df = pd.DataFrame()
phi_m_dc_df['z/L I'] = zL_df['zL_I_dc']
phi_m_dc_df['z/L II'] = zL_df['zL_II_dc']
phi_m_dc_df['z/L III'] = zL_df['zL_III_dc']
phi_m_dc_df['z/L 4_1'] = (np.array(zL_df['zL_4_dc'])+np.array(zL_df['zL_1_dc']))/2
phi_m_dc_df['z/L 3_1'] = (np.array(zL_df['zL_3_dc'])+np.array(zL_df['zL_1_dc']))/2
phi_m_dc_df['z/L 4_2'] = (np.array(zL_df['zL_4_dc'])+np.array(zL_df['zL_2_dc']))/2
# phi_m_dc_df['z/L I'] = zL_df['z/L I coare']
# phi_m_dc_df['z/L II'] = zL_df['z/L II coare']
# phi_m_dc_df['z/L III'] = zL_df['z/L III coare']
phi_m_dc_df['phi_m I'] = phi_m_I_dc
phi_m_dc_df['phi_m II'] = phi_m_II_dc
phi_m_dc_df['phi_m III'] = phi_m_III_dc
phi_m_dc_df['phi_m 4_1'] = phi_m_4_1_dc
phi_m_dc_df['phi_m 3_1'] = phi_m_3_1_dc
phi_m_dc_df['phi_m 4_2'] = phi_m_4_2_dc

print('done with writing phi_m via D.C. method')
print('done at line 338')


#%%
phi_m_I_dc_df = pd.DataFrame()
phi_m_I_dc_df['z/L'] = zL_df['zL_I_dc']
# phi_m_I_dc_df['z/L'] = zL_df['z/L I coare']
phi_m_I_dc_df['phi_m'] = phi_m_I_dc
phi_m_I_dc_df.to_csv(file_path + "phiM_I_dc.csv")
#get rid of negative shear values
phi_m_I_dc_df['phi_m_pos'] = np.where(phi_m_I_dc_df['phi_m']>=0,phi_m_I_dc_df['phi_m'],np.nan)
mask_phi_m_I_dc_df = np.isin(phi_m_I_dc_df['phi_m_pos'], phi_m_I_dc_df['phi_m'])
phi_m_I_dc_df_final = phi_m_I_dc_df[mask_phi_m_I_dc_df]

phi_m_I_dc_df_final = phi_m_I_dc_df_final.sort_values(by='z/L')
phi_m_I_dc_neg = phi_m_I_dc_df_final.loc[phi_m_I_dc_df_final['z/L']<=0]
phi_m_I_dc_pos = phi_m_I_dc_df_final.loc[phi_m_I_dc_df_final['z/L']>=0]




phi_m_II_dc_df = pd.DataFrame()
phi_m_II_dc_df['z/L'] = zL_df['zL_II_dc']
# phi_m_II_dc_df['z/L'] = zL_df['z/L II coare']
phi_m_II_dc_df['phi_m'] = phi_m_II_dc
phi_m_II_dc_df.to_csv(file_path + "phiM_II_dc.csv")
#get rid of negative shear values
phi_m_II_dc_df['phi_m_pos'] = np.where(phi_m_II_dc_df['phi_m']>=0,phi_m_II_dc_df['phi_m'],np.nan)
mask_phi_m_II_dc_df = np.isin(phi_m_II_dc_df['phi_m_pos'], phi_m_II_dc_df['phi_m'])
phi_m_II_dc_df_final = phi_m_II_dc_df[mask_phi_m_II_dc_df]

phi_m_II_dc_df_final = phi_m_II_dc_df_final.sort_values(by='z/L')
phi_m_II_dc_neg = phi_m_II_dc_df_final.loc[phi_m_II_dc_df['z/L']<=0]
phi_m_II_dc_pos = phi_m_II_dc_df_final.loc[phi_m_II_dc_df['z/L']>=0]




phi_m_III_dc_df = pd.DataFrame()
phi_m_III_dc_df['z/L'] = zL_df['zL_III_dc']
# phi_m_III_dc_df['z/L'] = zL_df['z/L III coare']
phi_m_III_dc_df['phi_m'] = phi_m_III_dc
phi_m_III_dc_df.to_csv(file_path + "phiM_III_dc.csv")
#get rid of negative shear values
phi_m_III_dc_df['phi_m_pos'] = np.where(phi_m_III_dc_df['phi_m']>=0,phi_m_III_dc_df['phi_m'],np.nan)
mask_phi_m_III_dc_df = np.isin(phi_m_III_dc_df['phi_m_pos'], phi_m_III_dc_df['phi_m'])
phi_m_III_dc_df_final = phi_m_III_dc_df[mask_phi_m_III_dc_df]

phi_m_III_dc_df_final = phi_m_III_dc_df_final.sort_values(by='z/L')
phi_m_III_dc_neg = phi_m_III_dc_df_final.loc[phi_m_III_dc_df['z/L']<=0]
phi_m_III_dc_pos = phi_m_III_dc_df_final.loc[phi_m_III_dc_df['z/L']>=0]

print('done at line 391')
#%% #calculate phi_M for levels 4-1, 3-1, and 4-2
'''
phi_m_4_1_dc_df = pd.DataFrame()
phi_m_4_1_dc_df['z/L'] = (np.array(zL_df['zL_4_dc'])+np.array(zL_df['zL_1_dc']))/2
phi_m_4_1_dc_df['phi_m'] = phi_m_4_1_dc
phi_m_4_1_dc_df.to_csv(file_path + "phiM_4_1_dc.csv")
#get rid of negative shear values
phi_m_4_1_dc_df['phi_m_pos'] = np.where(phi_m_4_1_dc_df['phi_m']>=0,phi_m_4_1_dc_df['phi_m'],np.nan)
mask_phi_m_4_1_dc_df = np.isin(phi_m_4_1_dc_df['phi_m_pos'], phi_m_4_1_dc_df['phi_m'])
phi_m_4_1_dc_df_final = phi_m_4_1_dc_df[mask_phi_m_4_1_dc_df]
#separate into negative z/L and positive z/L
phi_m_4_1_dc_df_final = phi_m_4_1_dc_df_final.sort_values(by='z/L')
phi_m_4_1_dc_neg = phi_m_4_1_dc_df_final.loc[phi_m_4_1_dc_df_final['z/L']<=0]
phi_m_4_1_dc_pos = phi_m_4_1_dc_df_final.loc[phi_m_4_1_dc_df_final['z/L']>=0]



phi_m_3_1_dc_df = pd.DataFrame()
phi_m_3_1_dc_df['z/L'] = (np.array(zL_df['zL_3_dc'])+np.array(zL_df['zL_1_dc']))/2
phi_m_3_1_dc_df['phi_m'] = phi_m_3_1_dc
phi_m_3_1_dc_df.to_csv(file_path + "phiM_3_1_dc.csv")
#get rid of negative shear values
phi_m_3_1_dc_df['phi_m_pos'] = np.where(phi_m_3_1_dc_df['phi_m']>=0,phi_m_3_1_dc_df['phi_m'],np.nan)
mask_phi_m_3_1_dc_df = np.isin(phi_m_3_1_dc_df['phi_m_pos'], phi_m_3_1_dc_df['phi_m'])
phi_m_3_1_dc_df_final = phi_m_3_1_dc_df[mask_phi_m_3_1_dc_df]
#separate into negative z/L and positive z/L
phi_m_3_1_dc_df_final = phi_m_3_1_dc_df_final.sort_values(by='z/L')
phi_m_3_1_dc_neg = phi_m_3_1_dc_df_final.loc[phi_m_3_1_dc_df_final['z/L']<=0]
phi_m_3_1_dc_pos = phi_m_3_1_dc_df_final.loc[phi_m_3_1_dc_df_final['z/L']>=0]



phi_m_4_2_dc_df = pd.DataFrame()
phi_m_4_2_dc_df['z/L'] = (np.array(zL_df['zL_4_dc'])+np.array(zL_df['zL_2_dc']))/2
phi_m_4_2_dc_df['phi_m'] = phi_m_4_2_dc
phi_m_4_2_dc_df.to_csv(file_path + "phiM_4_2_dc.csv")
#get rid of negative shear values
phi_m_4_2_dc_df['phi_m_pos'] = np.where(phi_m_4_2_dc_df['phi_m']>=0,phi_m_4_2_dc_df['phi_m'],np.nan)
mask_phi_m_4_2_dc_df = np.isin(phi_m_4_2_dc_df['phi_m_pos'], phi_m_4_2_dc_df['phi_m'])
phi_m_4_2_dc_df_final = phi_m_4_2_dc_df[mask_phi_m_4_2_dc_df]
#separate into negative z/L and positive z/L
phi_m_4_2_dc_df_final = phi_m_4_2_dc_df_final.sort_values(by='z/L')
phi_m_4_2_dc_neg = phi_m_4_2_dc_df_final.loc[phi_m_4_2_dc_df_final['z/L']<=0]
phi_m_4_2_dc_pos = phi_m_4_2_dc_df_final.loc[phi_m_4_2_dc_df_final['z/L']>=0]
print('done at line 435')
'''
#%% COARE functional form for Phi_m

coare_zL_neg = np.arange(-4,0,0.005)
coare_zL_pos = np.arange(0,2,0.005)

eq34 = (1-15*coare_zL_neg)**(-1/3) #eq 34 for negative z/L
e = 6
eq41 = 1+e*coare_zL_pos #eq 41 for posative z/L


plt.figure()
plt.plot(coare_zL_neg,eq34, color = 'g', label = 'eq34')
plt.plot(coare_zL_pos,eq41, color = 'b', label = 'eq41; e=6')

plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()
plt.title('$\phi_m$ functional form')


print('done plotting Phi_M COARE functional form')
#%% Phi_m plots
plt.figure()
plt.scatter(phi_m_I_dc_df_final['z/L'],phi_m_I_dc_df_final['phi_m'], color = 'dodgerblue', edgecolor = 'navy', label ='$\phi_m(z/L)$ d.c. L I')
# plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_m(z/L)$')
plt.title('$\phi_m(z/L)$ Level I, Combined Analysis')
# plt.title(oct_addition+ title_windDir + "Level I: $\phi_m(z/L)$")
plt.xlim(-4.5,2.5)
# plt.ylim(-6,6)
plt.ylim(-0.5,3)
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_m_II_dc_df_final['z/L'],phi_m_II_dc_df_final['phi_m'], color = 'orange', edgecolor = 'red', label ='$\phi_m(z/L)$ d.c. L II')
# plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_m(z/L)$')
plt.title('$\phi_m(z/L)$ Level II, Combined Analysis')
# plt.title(oct_addition+ title_windDir + "Level II: $\phi_m(z/L)$")
plt.xlim(-4.5,2.5)
# plt.ylim(-6,6)
plt.ylim(-0.5,3)
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_m_III_dc_df_final['z/L'],phi_m_III_dc_df_final['phi_m'], color = 'seagreen', edgecolor = 'darkgreen', label ='$\phi_m(z/L)$ d.c. L III')
# plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_m(z/L)$')
plt.title('$\phi_m(z/L)$ Level III, Combined Analysis')
# plt.title(title_windDir + "Level III: $\phi_m(z/L)$")
plt.xlim(-4.5,2.5)
# plt.ylim(-6,6)
plt.ylim(-0.5,3)
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()

print('done at line 516')
#%% Phi_m plots levels 4-1, 3-1, 4-2
"""
ymin = -4
ymax = 4
xmin = -4
xmax = 2

plt.figure()
plt.scatter(phi_m_4_1_dc_df_final['z/L'],phi_m_4_1_dc_df_final['phi_m'], color = 'gold', edgecolor = 'black', label ='$\phi_m(z/L)$ d.c. 4-1')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_m(z/L)$')
plt.title('$\phi_m(z/L)$ Sonic 4 - Sonic 1, Combined Analysis')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_m_3_1_dc_df_final['z/L'],phi_m_3_1_dc_df_final['phi_m'], color = 'silver', edgecolor = 'black', label ='$\phi_m(z/L)$ d.c. 3-1')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_m(z/L)$')
plt.title('$\phi_m(z/L)$ Sonic 3 - Sonic 1, Combined Analysis')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_m_4_2_dc_df_final['z/L'],phi_m_4_2_dc_df_final['phi_m'], color = 'tan', edgecolor = 'black', label ='$\phi_m(z/L)$ d.c. 4-2')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_m(z/L)$')
plt.title('$\phi_m(z/L)$ Sonic 4 - Sonic 2, Combined Analysis')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.legend()
"""
#%%
# import binsreg
# binscatter function written by Matteo Courthoud in Towards Data Science May 9, 2022
def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg
# nbins_num = 100
df_binEstimate_phi_m_I_dc = binscatter(x='z/L', y='phi_m', data=phi_m_I_dc_df_final, ci=(3,3), randcut=1, )
df_binEstimate_phi_m_II_dc = binscatter(x='z/L', y='phi_m', data=phi_m_II_dc_df_final, ci=(3,3), randcut=1, )
df_binEstimate_phi_m_III_dc = binscatter(x='z/L', y='phi_m', data=phi_m_III_dc_df_final, ci=(3,3), randcut=1, )
# df_binEstimate_phi_m_4_1_dc = binscatter(x='z/L', y='phi_m', data=phi_m_4_1_dc_df_final, ci=(3,3), randcut=1, )
# df_binEstimate_phi_m_3_1_dc = binscatter(x='z/L', y='phi_m', data=phi_m_3_1_dc_df_final, ci=(3,3), randcut=1, )
# df_binEstimate_phi_m_4_2_dc = binscatter(x='z/L', y='phi_m', data=phi_m_4_2_dc_df_final, ci=(3,3), randcut=1, )

print('done with binning data')

#%% Phi_m binned scatterplot
plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc, color = 'dodgerblue', label = "binned $\phi_{M}(z/L)$: L I")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "Level I: $phi_{M}(z/L) (DC)$")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc, color = 'darkorange', label = "binned $\phi_{M}(z/L)$: L II")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "Level II: $phi_{M}(z/L) (DC)$")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc, color = 'blue', label = "binned $\phi_{M}(z/L)$: L III")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "Level III: $phi_{M}(z/L) (DC)$")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()

plt.figure(figsize=(6,5))
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc, color = 'dodgerblue', label = "L I")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc, color = 'navy', ls='', lw=2, alpha=0.2, label = 'L I error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc, color = 'darkorange', label = "L II")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc, color = 'red', ls='', lw=2, alpha=0.2, label = 'L II error')
# sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc, color = 'seagreen', label = "L III")
# plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc, color = 'k', ls='', lw=2, alpha=0.2, label = 'L III error')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE eq. 34, 41')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title("$\phi_{M}(\zeta)$")
plt.vlines(x=0,ymin=-0.5,ymax=3,linestyles='--',color='k',)
plt.xlim(-2,1)
plt.ylim(0,3)
plt.xlabel("$\zeta = z/L$")
plt.ylabel('$\phi_M(\zeta)$')
plt.legend()
plt.savefig(plot_savePath + "binnedScatterplot_phiM_Puu.png",dpi=300)
plt.savefig(plot_savePath + "binnedScatterplot_phiM_Puu.pdf")

plt.figure(figsize=(6,5))
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc, color = 'dodgerblue', label = "L I")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc, color = 'navy', ls='', lw=2, alpha=0.2, label = 'L I error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc, color = 'darkorange', label = "L II")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc, color = 'red', ls='', lw=2, alpha=0.2, label = 'L II error')
# sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc, color = 'seagreen', label = "L III")
# plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc, color = 'k', ls='', lw=2, alpha=0.2, label = 'L III error')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE eq. 34, 41')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title("$\phi_{M}(\zeta)$\n(Near-Neutral Conditions)")
plt.vlines(x=0,ymin=-0.5,ymax=3,linestyles='--',color='k',)
plt.xlim(-0.5,0.5)
plt.ylim(0,3)
plt.xlabel("$\zeta = z/L$")
plt.ylabel('$\phi_M(\zeta)$')
plt.legend()
plt.savefig(plot_savePath + "binnedZOOMScatterplot_phiM_Puu.png",dpi=300)
plt.savefig(plot_savePath + "binnedZOOMScatterplot_phiM_Puu.pdf")


#%%
# Calculate the difference between the observations and the functional form

coare_zL_neg_nearNeutral = np.arange(-0.5,0,0.001)
coare_zL_pos_nearNeutral = np.arange(0,0.5,0.001)

eq34 = (1-15*coare_zL_neg_nearNeutral)**(-1/3) #eq 34 for negative z/L
e = 6
eq41 = 1+e*coare_zL_pos_nearNeutral #eq 41 for posative z/L
coare_zL_combined = np.arange(-0.5,0.5,0.001)

zL_phiM_obs_arr_neg = []
phiM_obs_arr_neg = []
zL_phiM_obs_arr_pos = []
phiM_obs_arr_pos = []
for j in range(len(df_binEstimate_phi_m_I_dc)):
    for i in range(len(coare_zL_neg_nearNeutral)):
        if (coare_zL_neg_nearNeutral[i] >= df_binEstimate_phi_m_I_dc['z/L'][j]-0.0005) & (coare_zL_neg_nearNeutral[i] <= df_binEstimate_phi_m_I_dc['z/L'][j]+0.0005):
            zL_phiM_obs_arr_neg.append(coare_zL_neg_nearNeutral[i])
            phiM_obs_arr_neg.append(eq34[i])
            print(eq34[i])
        else:
            continue
    for i in range(len(coare_zL_pos_nearNeutral)):
        if (coare_zL_pos_nearNeutral[i] >= df_binEstimate_phi_m_I_dc['z/L'][j]-0.0005) & (coare_zL_pos_nearNeutral[i] <= df_binEstimate_phi_m_I_dc['z/L'][j]+0.0005):
            zL_phiM_obs_arr_pos.append(coare_zL_pos_nearNeutral[i])
            phiM_obs_arr_pos.append(eq41[i])
            print(eq41[i])
        else:
            continue
            # print('does not match')
    # zL_phiM_obs_arr.append(zL_phiM_obs_arr_i)
zL_phiM_obs_arr = np.concatenate([np.array(zL_phiM_obs_arr_neg), np.array(zL_phiM_obs_arr_pos)], axis=0)   
phiM_obs_arr = np.concatenate([np.array(phiM_obs_arr_neg), np.array(phiM_obs_arr_pos)], axis=0)   
print('done')


y_obs = np.ones(len(df_binEstimate_phi_m_I_dc))
y_coare = np.ones(len(np.array(zL_phiM_obs_arr)))*1.1

plt.figure()
plt.scatter(df_binEstimate_phi_m_I_dc['z/L'], y_obs, color = 'r')
plt.scatter(np.array(zL_phiM_obs_arr), y_coare, color = 'b')
plt.ylim(0,2)
#here we see the first point, outside the near-neutral -0.5 - +0.5 range is the "problematic point"
#we will exclude that from the next part of the analysis

phiM_coare_nearNeutral = phiM_obs_arr
phiM_DC_obs = df_binEstimate_phi_m_I_dc['phi_m'][1:]

avg_offset_phiM = np.mean(phiM_DC_obs-phiM_coare_nearNeutral)
print(avg_offset_phiM)
avg_percent_difference_phiM = np.mean((phiM_coare_nearNeutral-phiM_DC_obs)/phiM_coare_nearNeutral*100)
print(avg_percent_difference_phiM)
median_percent_difference_phiM = np.median((phiM_coare_nearNeutral-phiM_DC_obs)/phiM_coare_nearNeutral*100)
print(median_percent_difference_phiM)
#%% Phi_m binned scatterplot levels 4-1, 3-1, 4-2
'''
xmin = -2
xmax = 1
ymin = -0.5
ymax = 3

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_4_1_dc, color = 'mediumaquamarine', label = "binned $\phi_{M}(z/L)$: 4-1")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_4_1_dc, color = 'lightseagreen', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "sonic 4 - sonic 1: $phi_{M}(z/L) (DC)$")
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_3_1_dc, color = 'blue', label = "binned $\phi_{M}(z/L)$: 3-1")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_3_1_dc, color = 'mediumblue', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "sonic 3 - sonic 1: $phi_{M}(z/L) (DC)$")
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_4_2_dc, color = 'lime', label = "binned $\phi_{M}(z/L)$: 4-2")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_4_2_dc, color = 'seagreen', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "sonic 4 - sonic 2: $phi_{M}(z/L) (DC)$")
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.legend()
#%%
plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_4_1_dc, color = 'mediumaquamarine', label = "4-1")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_4_1_dc, color = 'lightseagreen', ls='', lw=2, alpha=0.2, label = '4-1 error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_3_1_dc, color = 'blue', label = "3-1")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_3_1_dc, color = 'mediumblue', ls='', lw=2, alpha=0.2, label = '3-1 error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_4_2_dc, color = 'lime', label = "4-2")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_4_2_dc, color = 'seagreen', ls='', lw=2, alpha=0.2, label = '4-2 error')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE eq. 34, 41')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(oct_addition+ title_windDir + "$\phi_{M}(z/L) (DC)$")
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.legend()
plt.savefig(plot_savePath + "binnedScatterplot_phiM_testDiffLevels_Puu.png",dpi=300)
plt.savefig(plot_savePath + "binnedScatterplot_phiM_testDiffLevels_Puu.pdf")

'''
#%%



######################################################################
######################################################################
"""
NOW WE ARE MOVING ON TO PHI_EPSILON
"""

# getting phi_epsilon
# eps_s1 = Eps_df['eps_sonic1']
eps_s1 = Eps_df['epsU_sonic1_MAD']
phi_eps_1_dc = np.array(eps_s1)*kappa*np.array(z_df['z_sonic1'])/np.array(usr_df['usr_s1']**3)

# eps_s2 = Eps_df['eps_sonic2']
eps_s2 = Eps_df['epsU_sonic2_MAD']
phi_eps_2_dc = np.array(eps_s2)*kappa*np.array(z_df['z_sonic2'])/np.array(usr_df['usr_s2']**3)

# eps_s3 = Eps_df['eps_sonic3']
eps_s3 = Eps_df['epsU_sonic3_MAD']
phi_eps_3_dc = np.array(eps_s3)*kappa*np.array(z_df['z_sonic3'])/np.array(usr_df['usr_s3']**3)

# eps_s4 = Eps_df['eps_sonic4']
eps_s4 = Eps_df['epsU_sonic4_MAD']
phi_eps_4_dc = np.array(eps_s4)*kappa*np.array(z_df['z_sonic4'])/np.array(usr_df['usr_s4']**3)

phi_eps_1_dc_df = pd.DataFrame()
# phi_eps_1_dc_df['z/L'] = (np.array(z_df['z_sonic1']))/np.array(L_dc_df['L_sonic1'])
phi_eps_1_dc_df['phi_eps'] = np.array(phi_eps_1_dc)

phi_eps_2_dc_df = pd.DataFrame()
# phi_eps_2_dc_df['z/L'] = (np.array(z_df['z_sonic2']))/np.array(L_dc_df['L_sonic2'])
phi_eps_2_dc_df['phi_eps'] = np.array(phi_eps_2_dc)

phi_eps_3_dc_df = pd.DataFrame()
# phi_eps_3_dc_df['z/L'] = (np.array(z_df['z_sonic3']))/np.array(L_dc_df['L_sonic3'])
phi_eps_3_dc_df['phi_eps'] = np.array(phi_eps_3_dc)

phi_eps_4_dc_df = pd.DataFrame()
# phi_eps_4_dc_df['z/L'] = (np.array(z_df['z_sonic4']))/np.array(L_dc_df['L_sonic4'])
phi_eps_4_dc_df['phi_eps'] = np.array(phi_eps_4_dc)

print('done with writing phi_eps via D.C. method')
print('done at line 408')


phi_eps_I_dc = (phi_eps_1_dc+phi_eps_2_dc)/2
phi_eps_II_dc = (phi_eps_2_dc+phi_eps_3_dc)/2
phi_eps_III_dc = (phi_eps_3_dc+phi_eps_4_dc)/2
phi_eps_4_1_dc = (phi_eps_4_dc+phi_eps_1_dc)/2
phi_eps_3_1_dc = (phi_eps_3_dc+phi_eps_1_dc)/2
phi_eps_4_2_dc = (phi_eps_4_dc+phi_eps_2_dc)/2

phi_eps_dc_df = pd.DataFrame()
phi_eps_dc_df['z/L I'] = zL_df['zL_I_dc']
phi_eps_dc_df['z/L II'] = zL_df['zL_II_dc']
phi_eps_dc_df['z/L III'] = zL_df['zL_III_dc']
phi_eps_dc_df['z/L 4_1'] = (np.array(zL_df['zL_4_dc'])+np.array(zL_df['zL_1_dc']))/2
phi_eps_dc_df['z/L 3_1'] = (np.array(zL_df['zL_3_dc'])+np.array(zL_df['zL_1_dc']))/2
phi_eps_dc_df['z/L 4_2'] = (np.array(zL_df['zL_4_dc'])+np.array(zL_df['zL_2_dc']))/2
phi_eps_dc_df['phi_eps I'] = phi_eps_I_dc
phi_eps_dc_df['phi_eps II'] = phi_eps_II_dc
phi_eps_dc_df['phi_eps III'] = phi_eps_III_dc
phi_eps_dc_df['phi_eps 4_1'] = phi_eps_4_1_dc
phi_eps_dc_df['phi_eps 3_1'] = phi_eps_3_1_dc
phi_eps_dc_df['phi_eps 4_2'] = phi_eps_4_2_dc

plt.figure()
plt.plot(phi_eps_1_dc, label = 'dc 1')
plt.plot(phi_eps_2_dc, label = 'dc 2')
plt.plot(phi_eps_3_dc, label = 'dc 3')
plt.plot(phi_eps_4_dc, label = 'dc 4')
plt.legend()
# plt.ylim(0,1000)
plt.title('Time Series $\phi_{\epsilon}$ DC')
print('done at line 635')

plt.figure()
plt.plot(phi_eps_I_dc, label = 'dc I')
plt.plot(phi_eps_II_dc, label = 'dc II')
plt.plot(phi_eps_III_dc, label = 'dc III')
plt.legend()
# plt.ylim(0,1000)
plt.title('Time Series $\phi_{\epsilon}$ DC')

plt.figure()
plt.plot(phi_eps_4_1_dc, label = 'dc 4_1')
plt.plot(phi_eps_3_1_dc, label = 'dc 3_1')
plt.plot(phi_eps_4_2_dc, label = 'dc 4_2')
plt.legend()
# plt.ylim(0,1000)
plt.title('Time Series $\phi_{\epsilon}$ DC')

print('done at line 881')


#%%
phi_eps_I_dc_df = pd.DataFrame()
phi_eps_I_dc_df['z/L'] = zL_df['zL_I_dc']
phi_eps_I_dc_df['phi_eps'] = phi_eps_I_dc
phi_eps_I_dc_df.to_csv(file_path + "phiEps_I_dc.csv")

phi_eps_I_dc_df_final = phi_eps_I_dc_df.sort_values(by='z/L')
phi_eps_I_dc_neg = phi_eps_I_dc_df_final.loc[phi_eps_I_dc_df_final['z/L']<=0]
phi_eps_I_dc_pos = phi_eps_I_dc_df_final.loc[phi_eps_I_dc_df_final['z/L']>=0]


phi_eps_II_dc_df = pd.DataFrame()
phi_eps_II_dc_df['z/L'] = zL_df['zL_II_dc']
phi_eps_II_dc_df['phi_eps'] = phi_eps_II_dc
phi_eps_II_dc_df.to_csv(file_path + "phiEps_II_dc.csv")

phi_eps_II_dc_df_final = phi_eps_II_dc_df.sort_values(by='z/L')
phi_eps_II_dc_neg = phi_eps_II_dc_df_final.loc[phi_eps_II_dc_df_final['z/L']<=0]
phi_eps_II_dc_pos = phi_eps_II_dc_df_final.loc[phi_eps_II_dc_df_final['z/L']>=0]


phi_eps_III_dc_df = pd.DataFrame()
phi_eps_III_dc_df['z/L'] = zL_df['zL_III_dc']
phi_eps_III_dc_df['phi_eps'] = phi_eps_III_dc
phi_eps_III_dc_df.to_csv(file_path + "phiEps_III_dc.csv")

phi_eps_III_dc_df_final = phi_eps_III_dc_df.sort_values(by='z/L')
phi_eps_III_dc_neg = phi_eps_III_dc_df_final.loc[phi_eps_III_dc_df_final['z/L']<=0]
phi_eps_III_dc_pos = phi_eps_III_dc_df_final.loc[phi_eps_III_dc_df_final['z/L']>=0]

print('done at line 917')

#%% For 4-1, 3-1, 4-2
'''
phi_eps_4_1_dc_df = pd.DataFrame()
phi_eps_4_1_dc_df['z/L'] = (np.array(zL_df['zL_4_dc'])+np.array(zL_df['zL_1_dc']))/2
phi_eps_4_1_dc_df['phi_eps'] = phi_eps_4_1_dc
phi_eps_4_1_dc_df.to_csv(file_path + "phiEps_4_1_dc.csv")
#sort by pos and neg z/L values
phi_eps_4_1_dc_df_final = phi_eps_4_1_dc_df.sort_values(by='z/L')
phi_eps_4_1_dc_neg = phi_eps_4_1_dc_df_final.loc[phi_eps_4_1_dc_df_final['z/L']<=0]
phi_eps_4_1_dc_pos = phi_eps_4_1_dc_df_final.loc[phi_eps_4_1_dc_df_final['z/L']>=0]


phi_eps_3_1_dc_df = pd.DataFrame()
phi_eps_3_1_dc_df['z/L'] = (np.array(zL_df['zL_3_dc'])+np.array(zL_df['zL_1_dc']))/2
phi_eps_3_1_dc_df['phi_eps'] = phi_eps_3_1_dc
phi_eps_3_1_dc_df.to_csv(file_path + "phiEps_3_1_dc.csv")
#sort by pos and neg z/L values
phi_eps_3_1_dc_df_final = phi_eps_3_1_dc_df.sort_values(by='z/L')
phi_eps_3_1_dc_neg = phi_eps_3_1_dc_df_final.loc[phi_eps_3_1_dc_df_final['z/L']<=0]
phi_eps_3_1_dc_pos = phi_eps_3_1_dc_df_final.loc[phi_eps_3_1_dc_df_final['z/L']>=0]


phi_eps_4_2_dc_df = pd.DataFrame()
phi_eps_4_2_dc_df['z/L'] = (np.array(zL_df['zL_4_dc'])+np.array(zL_df['zL_2_dc']))/2
phi_eps_4_2_dc_df['phi_eps'] = phi_eps_4_2_dc
phi_eps_4_2_dc_df.to_csv(file_path + "phiEps_4_2_dc.csv")
#sort by pos and neg z/L values
phi_eps_4_2_dc_df_final = phi_eps_4_2_dc_df.sort_values(by='z/L')
phi_eps_4_2_dc_neg = phi_eps_4_2_dc_df_final.loc[phi_eps_4_2_dc_df_final['z/L']<=0]
phi_eps_4_2_dc_pos = phi_eps_4_2_dc_df_final.loc[phi_eps_4_2_dc_df_final['z/L']>=0]
'''
#%% COARE functional form for Phi_epsilon
eq40 = ((1-coare_zL_neg)/(1-7*coare_zL_neg))-coare_zL_neg
eq40_me = ((1-coare_zL_neg)/(1-14*coare_zL_neg))-coare_zL_neg
eq42 = 1+(e-1)*coare_zL_pos

print('done writing phi_eps as equations 40 and 42 for each sonic')
print('done at line 957')

plt.figure()
plt.plot(coare_zL_neg, -1*eq40, color = 'g', label = '-1*eq40') # we multiply by negative 1 just here to match Edson 98 plots
plt.plot(coare_zL_neg, -1*eq40_me, color = 'r', label = '-1*eq40_me') # fooling around and proposing my own function!
plt.plot(coare_zL_pos, -1*eq42, color = 'b', label = '-1*eq42; e=6') # we multiply by negative 1 just here to match Edson 98 plots
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()
plt.title('$\phi_{\epsilon}$ functional form')

print('done at line 969')

#%% Phi_epsilon plots
plt.figure()
plt.scatter(phi_eps_I_dc_df_final['z/L'],phi_eps_I_dc_df_final['phi_eps'], color = 'dodgerblue', edgecolor = 'navy', label ='$\phi_{\epsilon}(z/L)$ d.c. L I')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_neg, eq40_me, color = 'r',linewidth=3, label = 'ME')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{\epsilon}(z/L)$')
plt.title('$\phi_{\epsilon}(z/L)$ Level I, Combined Analysis')
plt.xlim(-4,2)
# plt.ylim(-4,4)
plt.yscale('log')
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_eps_II_dc_df_final['z/L'],phi_eps_II_dc_df_final['phi_eps'], color = 'orange', edgecolor = 'red', label ='$\phi_{\epsilon}(z/L)$ d.c. L II')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{\epsilon}(z/L)$')
plt.title('$\phi_{\epsilon}(z/L)$ Level II, Combined Analysis')
plt.xlim(-4,2)
# plt.ylim(-4,4)
plt.yscale('log')
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_eps_III_dc_df_final['z/L'],phi_eps_III_dc_df_final['phi_eps'], color = 'seagreen', edgecolor = 'darkgreen', label ='$\phi_{\epsilon}(z/L)$ d.c. L III')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{\epsilon}(z/L)$')
plt.title('$\phi_{\epsilon}(z/L)$ Level III, Combined Analysis')
plt.xlim(-4,2)
# plt.ylim(-4,4)
plt.yscale('log')
plt.grid()
plt.legend()

#%% Phi_epsilon plots levels 4-1, 3-1, 4-2
'''
xmin = -4
xmax = 2
ymin = -4
ymax = 4

plt.figure()
plt.scatter(phi_eps_4_1_dc_df_final['z/L'],phi_eps_4_1_dc_df_final['phi_eps'], color = 'mediumaquamarine', edgecolor = 'k', label ='$\phi_{\epsilon}(z/L)$ d.c. 4-1')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_neg, eq40_me, color = 'r',linewidth=3, label = 'ME')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{\epsilon}(z/L)$')
plt.title('$\phi_{\epsilon}(z/L)$ sonics 4-1, Combined Analysis')
plt.xlim(xmin,xmax)
# plt.ylim(ymin,ymax)
plt.yscale('log')
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_eps_3_1_dc_df_final['z/L'],phi_eps_3_1_dc_df_final['phi_eps'], color = 'blue', edgecolor = 'k', label ='$\phi_{\epsilon}(z/L)$ d.c. 3-1')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_neg, eq40_me, color = 'r',linewidth=3, label = 'ME')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{\epsilon}(z/L)$')
plt.title('$\phi_{\epsilon}(z/L)$ sonics 3-1, Combined Analysis')
plt.xlim(xmin,xmax)
# plt.ylim(ymin,ymax)
plt.yscale('log')
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_eps_4_2_dc_df_final['z/L'],phi_eps_4_2_dc_df_final['phi_eps'], color = 'lime', edgecolor = 'k', label ='$\phi_{\epsilon}(z/L)$ d.c. 4-2')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_neg, eq40_me, color = 'r',linewidth=3, label = 'ME')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{\epsilon}(z/L)$')
plt.title('$\phi_{\epsilon}(z/L)$ sonics 4-2, Combined Analysis')
plt.xlim(xmin,xmax)
# plt.ylim(ymin,ymax)
plt.yscale('log')
plt.grid()
plt.legend()
'''
#%%
def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg
df_binEstimate_phi_eps_I_dc = binscatter(x='z/L', y='phi_eps', data=phi_eps_I_dc_df_final, ci=(3,3),randcut=1)
df_binEstimate_phi_eps_II_dc = binscatter(x='z/L', y='phi_eps', data=phi_eps_II_dc_df_final, ci=(3,3),randcut=1)
df_binEstimate_phi_eps_III_dc = binscatter(x='z/L', y='phi_eps', data=phi_eps_III_dc_df_final, ci=(3,3),randcut=1)
# df_binEstimate_phi_eps_4_1_dc = binscatter(x='z/L', y='phi_eps', data=phi_eps_4_1_dc_df_final, ci=(3,3),randcut=1)
# df_binEstimate_phi_eps_3_1_dc = binscatter(x='z/L', y='phi_eps', data=phi_eps_3_1_dc_df_final, ci=(3,3),randcut=1)
# df_binEstimate_phi_eps_4_2_dc = binscatter(x='z/L', y='phi_eps', data=phi_eps_4_2_dc_df_final, ci=(3,3),randcut=1)

#%% Phi_epsilon binned scatterplots
plt.figure()
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_I_dc, color = 'dodgerblue', label = "binned $\phi_{\epsilon}(z/L)$: L I")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_I_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=2, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=2)
# plt.plot(coare_zL_neg, eq40_me, color = 'yellow',linewidth=2, label = 'ME')
plt.title(title_windDir + "Level I: $phi_{\epsilon}(z/L) (DC)$")
plt.xlim(-4,2)
# plt.ylim(-5,4)
plt.yscale('log')
plt.legend()
# plt.gca().invert_yaxis()

plt.figure()
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_II_dc, color = 'orange', label = "binned $\phi_{\epsilon}(z/L)$: L II")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_II_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=2, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=2)
# plt.plot(coare_zL_neg, eq40_me, color = 'yellow',linewidth=2, label = 'ME')
plt.title(title_windDir + "Level II: $phi_{\epsilon}(z/L) (DC)$")
plt.xlim(-4,2)
# plt.ylim(-5,4)
plt.yscale('log')
plt.legend()
# plt.gca().invert_yaxis()

plt.figure()
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_III_dc, color = 'seagreen', label = "binned $\phi_{\epsilon}(z/L)$: L III")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_III_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=2, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=2)
# plt.plot(coare_zL_neg, eq40_me, color = 'yellow',linewidth=2, label = 'ME')
plt.title(title_windDir + "Level III: $phi_{\epsilon}(z/L) (DC)$")
plt.xlim(-4,2)
# plt.ylim(-5,4)
plt.yscale('log')
plt.legend()
# plt.gca().invert_yaxis()

plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'
plt.figure(figsize=(6,5))
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_I_dc, color = 'dodgerblue',edgecolor='navy', label = "L I")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_I_dc, color = 'navy', ls='', lw=2, alpha=0.2, label = 'L I error')
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_II_dc, color = 'darkorange', label = "L II")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_II_dc, color = 'red', ls='', lw=2, alpha=0.2, label = 'L II error')
# sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_III_dc, color = 'seagreen', label = "L III")
# plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_III_dc, color = 'k', ls='', lw=2, alpha=0.2, label = 'L III error')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'Edson et al. (1998) eq. 40, 42')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
# plt.plot(coare_zL_neg, eq40_me, color = 'blue',linewidth=2, label = 'My suggested new form')
plt.title("$\phi_{\epsilon}(\zeta)$")
plt.xlim(-2,2)
# plt.ylim(0,10)
plt.vlines(x=0,ymin=0,ymax=10,linestyles='--',color='k')
plt.xlabel('$\zeta = z/L$')
plt.ylabel('$\phi_{\epsilon}(\zeta)$')
plt.yscale('log')
plt.legend(loc = 'upper left')
plt.savefig(plot_savePath + "binnedScatterplot_phiEps_Puu.png",dpi=300)
plt.savefig(plot_savePath + "binnedScatterplot_phiEps_Puu.pdf")

plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'
plt.figure(figsize=(6,5))
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_I_dc, color = 'dodgerblue',edgecolor='navy', label = "L I")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_I_dc, color = 'navy', ls='', lw=2, alpha=0.2, label = 'L I error')
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_II_dc, color = 'darkorange', label = "L II")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_II_dc, color = 'red', ls='', lw=2, alpha=0.2, label = 'L II error')
# sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_III_dc, color = 'seagreen', label = "L III")
# plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_III_dc, color = 'k', ls='', lw=2, alpha=0.2, label = 'L III error')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'Edson et al. (1998) eq. 40, 42')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
# plt.plot(coare_zL_neg, eq40_me, color = 'blue',linewidth=2, label = 'My suggested new form')
plt.title("$\phi_{\epsilon}(\zeta)$\n(Near-Neutral Conditions)")
plt.xlim(-0.5,0.5)
# plt.ylim(0,10)
plt.vlines(x=0,ymin=0,ymax=10,linestyles='--',color='k')
plt.xlabel('$\zeta = z/L$')
plt.ylabel('$\phi_{\epsilon}(\zeta)$')
plt.yscale('log')
plt.legend(loc = 'upper left')
plt.savefig(plot_savePath + "binnedZOOMScatterplot_phiEps_Puu.png",dpi=300)
plt.savefig(plot_savePath + "binnedZOOMScatterplot_phiEps_Puu.pdf")

#%%
# Calculate the difference between the observations and the functional form

coare_zL_neg_nearNeutral = np.arange(-0.5,0,0.001)
coare_zL_pos_nearNeutral = np.arange(0,0.5,0.001)
e=6
eq40 = ((1-coare_zL_neg_nearNeutral)/(1-7*coare_zL_neg_nearNeutral))-coare_zL_neg_nearNeutral #eq 40 for negative z/L
# eq40_me = ((1-coare_zL_neg)/(1-14*coare_zL_neg))-coare_zL_neg
eq42 = 1+(e-1)*coare_zL_pos_nearNeutral #eq 42 for posative z/L

coare_zL_combined = np.arange(-0.5,0.5,0.001)
zL_phiEps_obs_arr_neg = []
phiEps_obs_arr_neg = []
zL_phiEps_obs_arr_pos = []
phiEps_obs_arr_pos = []
for j in range(len(df_binEstimate_phi_eps_I_dc)):
    for i in range(len(coare_zL_neg_nearNeutral)):
        if (coare_zL_neg_nearNeutral[i] >= df_binEstimate_phi_eps_I_dc['z/L'][j]-0.0005) & (coare_zL_neg_nearNeutral[i] <= df_binEstimate_phi_eps_I_dc['z/L'][j]+0.0005):
            zL_phiEps_obs_arr_neg.append(coare_zL_neg_nearNeutral[i])
            phiEps_obs_arr_neg.append(eq40[i])
            print(eq40[i])
        else:
            continue
    for i in range(len(coare_zL_pos_nearNeutral)):
        if (coare_zL_pos_nearNeutral[i] >= df_binEstimate_phi_eps_I_dc['z/L'][j]-0.0005) & (coare_zL_pos_nearNeutral[i] <= df_binEstimate_phi_eps_I_dc['z/L'][j]+0.0005):
            zL_phiEps_obs_arr_pos.append(coare_zL_pos_nearNeutral[i])
            phiEps_obs_arr_pos.append(eq42[i])
            print(eq42[i])
        else:
            continue
            # print('does not match')
    # zL_phiM_obs_arr.append(zL_phiM_obs_arr_i)
zL_phiEps_obs_arr = np.concatenate([np.array(zL_phiEps_obs_arr_neg), np.array(zL_phiEps_obs_arr_pos)], axis=0)   
phiEps_obs_arr = np.concatenate([np.array(phiEps_obs_arr_neg), np.array(phiEps_obs_arr_pos)], axis=0)   
print('done')
print(len(zL_phiEps_obs_arr))
print(len(df_binEstimate_phi_eps_I_dc))


y_obs = np.ones(len(df_binEstimate_phi_eps_I_dc))
y_coare = np.ones(len(np.array(zL_phiEps_obs_arr)))*1.1

plt.figure()
plt.scatter(df_binEstimate_phi_eps_I_dc['z/L'], y_obs, color = 'r')
plt.scatter(np.array(zL_phiEps_obs_arr), y_coare, color = 'b')
plt.ylim(0,2)
#here we see the first point, outside the near-neutral -0.5 - +0.5 range is the "problematic point"
#we will exclude that from the next part of the analysis

phiEps_coare_nearNeutral = phiEps_obs_arr
phiEps_DC_obs = df_binEstimate_phi_eps_I_dc['phi_eps'][1:]

avg_offset_phiEps = np.mean(phiEps_DC_obs-phiEps_coare_nearNeutral)
avg_percent_difference_phiEps = np.mean((phiEps_coare_nearNeutral-phiEps_DC_obs)/phiEps_coare_nearNeutral*100)
median_percent_difference_phiEps = np.median((phiEps_coare_nearNeutral-phiEps_DC_obs)/phiEps_coare_nearNeutral*100)
print(avg_offset_phiEps)
print(avg_percent_difference_phiEps)
print(median_percent_difference_phiEps)
#%% Phi_epsilon binned scatterplots levels 4-1, 3-1, 4-2
'''
plt.figure()
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_4_1_dc, color = 'mediumaquamarine', label = "binned $\phi_{\epsilon}(z/L)$: S 4-1")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_4_1_dc, color = 'lightseagreen', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=2, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=2)
# plt.plot(coare_zL_neg, eq40_me, color = 'yellow',linewidth=2, label = 'ME')
plt.title(title_windDir + "sonic 4 - sonic 1: $phi_{\epsilon}(z/L) (DC)$")
plt.xlim(xmin,xmax)
# plt.ylim(ymin,ymax)
plt.yscale('log')
plt.legend()
# plt.gca().invert_yaxis()

plt.figure()
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_3_1_dc, color = 'blue', label = "binned $\phi_{\epsilon}(z/L)$: S 3-1")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_3_1_dc, color = 'mediumblue', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=2, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=2)
# plt.plot(coare_zL_neg, eq40_me, color = 'yellow',linewidth=2, label = 'ME')
plt.title(title_windDir + "sonic 3 - sonic 1: $phi_{\epsilon}(z/L) (DC)$")
plt.xlim(xmin,xmax)
# plt.ylim(ymin,ymax)
plt.yscale('log')
plt.legend()
# plt.gca().invert_yaxis()

plt.figure()
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_4_2_dc, color = 'lime', label = "binned $\phi_{\epsilon}(z/L)$: S 4-2")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_4_2_dc, color = 'seagreen', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=2, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=2)
# plt.plot(coare_zL_neg, eq40_me, color = 'yellow',linewidth=2, label = 'ME')
plt.title(title_windDir + "sonic 4 - sonic 2: $phi_{\epsilon}(z/L) (DC)$")
plt.xlim(xmin,xmax)
# plt.ylim(ymin,ymax)
plt.yscale('log')
plt.legend()
# plt.gca().invert_yaxis()


plt.figure()
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_4_1_dc, color = 'mediumaquamarine', label = "L 4-1")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_4_1_dc, color = 'lightseagreen', ls='', lw=2, alpha=0.2, label = '4-1 error')
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_3_1_dc, color = 'blue', label = "L 3-1")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_3_1_dc, color = 'mediumblue', ls='', lw=2, alpha=0.2, label = '3-1 error')
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_4_2_dc, color = 'lime', label = "L 4-2")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_4_2_dc, color = 'seagreen', ls='', lw=2, alpha=0.2, label = '4-2 error')
plt.plot(coare_zL_neg, eq40, color = 'k',linewidth=3, label = 'Edson et al. (1998) eq. 40, 42')
plt.plot(coare_zL_pos, eq42, color = 'k',linewidth=3)
# plt.plot(coare_zL_neg, eq40_me, color = 'blue',linewidth=2, label = 'My suggested new form')
plt.title(oct_addition+ title_windDir + "$\phi_{\epsilon}(z/L) (DC)$")
plt.xlim(xmin,xmax)
# plt.ylim(ymin,ymax)
plt.ylabel('$\phi_\epsilon$')
plt.yscale('log')
plt.legend(loc = 'lower left',fontsize=7.5)
plt.savefig(plot_savePath + "binnedScatterplot_phiEps_testDiffLevels_Puu.png",dpi=300)
plt.savefig(plot_savePath + "binnedScatterplot_phiEps_testDiffLevels_Puu.pdf")
'''
#%%

######################################################################
######################################################################
"""
comparison of Phi_M and Phi_Eps
"""

plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'
plt.figure(figsize=(6,5))
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc, color = 'red', label = "$\phi_{M}$")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc, color = 'red', ls='', lw=2, alpha=0.2, label = '$\phi_{M}$ error')
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_I_dc, color = 'gray', label = "$\phi_{\epsilon}$")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_I_dc, color = 'gray', ls='', lw=2, alpha=0.2, label = '$\phi_{\epsilon}$ error')
plt.title("Level I: $\phi_{M}(\zeta)$ and $\phi_{\epsilon}(\zeta)$\n(Near-Neutral Conditions)")
plt.xlim(-0.5,0.5)
# plt.ylim(0,3)
plt.vlines(x=0,ymin=0,ymax=10,linestyles='--',color='k')
plt.xlabel('$\zeta = z/L$')
plt.ylabel('$\phi_{M}(\zeta)$, $\phi_{\epsilon}(\zeta)$')
plt.yscale('log')
plt.legend(loc = 'upper left')
plt.savefig(plot_savePath + "comparisonLI_phiM_phiEps_Puu.png",dpi=300)
plt.savefig(plot_savePath + "comparisonLI_phiM_phiEps_Puu.pdf")


plt.figure(figsize=(6,5))
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc, color = 'red', label = "$\phi_{M}$")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc, color = 'red', ls='', lw=2, alpha=0.2, label = '$\phi_{M}$ error')
sns.scatterplot(x='z/L', y='phi_eps', data=df_binEstimate_phi_eps_II_dc, color = 'gray', label = "$\phi_{\epsilon}$")
plt.errorbar('z/L', 'phi_eps', yerr='ci', data=df_binEstimate_phi_eps_II_dc, color = 'gray', ls='', lw=2, alpha=0.2, label = '$\phi_{\epsilon}$ error')
plt.title("Level II: $\phi_{M}(\zeta)$ and $\phi_{\epsilon}(\zeta)$\n(Near-Neutral Conditions)")
plt.xlim(-0.5,0.5)
# plt.ylim(0,3)
plt.vlines(x=0,ymin=0,ymax=10,linestyles='--',color='k')
plt.xlabel('$\zeta = z/L$')
plt.ylabel('$\phi_{M}(\zeta)$, $\phi_{\epsilon}(\zeta)$')
plt.yscale('log')
plt.legend(loc = 'upper left')
plt.savefig(plot_savePath + "comparisonLII_phiM_phiEps_Puu.png",dpi=300)
plt.savefig(plot_savePath + "comparisonLII_phiM_phiEps_Puu.pdf")

#%%

######################################################################
######################################################################
"""
NOW WE ARE MOVING ON TO BUOYANCY z/L
"""

blank_index = np.arange(0,len(zL_df))

zL_df['index_num'] = blank_index


phi_B_df = pd.DataFrame()
phi_B_df['z/L I index'] = zL_df['zL_I_dc']
phi_B_df['z/L I values'] = zL_df['zL_I_dc']
phi_B_df.set_index('z/L I index')
phi_B_df.sort_index(ascending=True)
eq_buoy = -1*coare_zL_neg

plt.figure()
plt.scatter(phi_B_df['z/L I index'],-1*phi_B_df['z/L I values'], label = 'dc buoy')
plt.plot(coare_zL_neg,eq_buoy, label = 'coare func form', color = 'k')
plt.legend()






#%%

######################################################################
######################################################################
"""
NOW WE ARE MOVING ON TO PHI_TKE
"""
# getting phi_tke

dWpEp_bar_dz_LI_spring = np.array(tke_transport_df['WpEp_bar_2'][:break_index+1]-tke_transport_df['WpEp_bar_1'][:break_index+1])/dz_LI_spring
dWpEp_bar_dz_LII_spring = np.array(tke_transport_df['WpEp_bar_3'][:break_index+1]-tke_transport_df['WpEp_bar_2'][:break_index+1])/dz_LII_spring
dWpEp_bar_dz_LIII_spring = np.array(tke_transport_df['WpEp_bar_4'][:break_index+1]-tke_transport_df['WpEp_bar_3'][:break_index+1])/dz_LIII_spring

dWpEp_bar_dz_LI_fall = np.array(tke_transport_df['WpEp_bar_2'][break_index+1:]-tke_transport_df['WpEp_bar_1'][break_index+1:])/dz_LI_fall
dWpEp_bar_dz_LII_fall = np.array(tke_transport_df['WpEp_bar_3'][break_index+1:]-tke_transport_df['WpEp_bar_2'][break_index+1:])/dz_LII_fall
dWpEp_bar_dz_LIII_fall = np.array(tke_transport_df['WpEp_bar_4'][break_index+1:]-tke_transport_df['WpEp_bar_3'][break_index+1:])/dz_LIII_fall

dWpEp_bar_dz_LI = np.concatenate([dWpEp_bar_dz_LI_spring, dWpEp_bar_dz_LI_fall], axis = 0)
dWpEp_bar_dz_LII = np.concatenate([dWpEp_bar_dz_LII_spring, dWpEp_bar_dz_LII_fall], axis = 0)
dWpEp_bar_dz_LIII = np.concatenate([dWpEp_bar_dz_LIII_spring, dWpEp_bar_dz_LIII_fall], axis = 0)

phi_tke_I_dc = kappa*np.array(z_LI)/(np.array(usr_LI)**3)*(np.array(dWpEp_bar_dz_LI))
phi_tke_II_dc = kappa*np.array(z_LII)/(np.array(usr_LII)**3)*(np.array(dWpEp_bar_dz_LII))
phi_tke_III_dc = kappa*np.array(z_LIII)/(np.array(usr_LIII)**3)*(np.array(dWpEp_bar_dz_LIII))

phi_tke_dc_df = pd.DataFrame()
phi_tke_dc_df['z/L I'] = zL_df['zL_I_dc']
phi_tke_dc_df['z/L II'] = zL_df['zL_II_dc']
phi_tke_dc_df['z/L III'] = zL_df['zL_III_dc']
phi_tke_dc_df['phi_tke I'] = phi_tke_I_dc
phi_tke_dc_df['phi_tke II'] = phi_tke_II_dc
phi_tke_dc_df['phi_tke III'] = phi_tke_III_dc



plt.figure()
plt.plot(phi_tke_dc_df['phi_tke I'], label='dc')
plt.legend()
# plt.ylim(-10,10)
plt.title(title_windDir + 'phi_tke LEVEL I')


print('done with writing phi_tke via D.C. method')
print('done at line 1267')

#%%
phi_tke_I_dc_df = pd.DataFrame()
phi_tke_I_dc_df['z/L'] = zL_df['zL_I_dc']
phi_tke_I_dc_df['phi_tke'] = -1*phi_tke_I_dc
# # can use to get rid of negative tke values
# phi_tke_I_dc_df['phi_tke_pos'] = np.where(phi_tke_I_dc_df['phi_tke']>=0,phi_tke_I_dc_df['phi_tke'],np.nan)
# mask_phi_tke_I_dc_df = np.isin(phi_tke_I_dc_df['phi_tke_pos'], phi_tke_I_dc_df['phi_tke'])
# phi_tke_I_dc_df_final = phi_tke_I_dc_df[mask_phi_tke_I_dc_df]

phi_tke_I_dc_df_final = phi_tke_I_dc_df.sort_values(by='z/L')
phi_tke_I_dc_neg = phi_tke_I_dc_df_final.loc[phi_tke_I_dc_df_final['z/L']<=0]
phi_tke_I_dc_pos = phi_tke_I_dc_df_final.loc[phi_tke_I_dc_df_final['z/L']>=0]


phi_tke_II_dc_df = pd.DataFrame()
phi_tke_II_dc_df['z/L'] = zL_df['zL_II_dc']
phi_tke_II_dc_df['phi_tke'] = -1*phi_tke_II_dc
# # can use to get rid of negative tke values
# phi_tke_II_dc_df['phi_tke_pos'] = np.where(phi_tke_II_dc_df['phi_tke']>=0,phi_tke_II_dc_df['phi_tke'],np.nan)
# mask_phi_tke_II_dc_df = np.isin(phi_tke_II_dc_df['phi_tke_pos'], phi_tke_II_dc_df['phi_tke'])
# phi_tke_II_dc_df_final = phi_tke_II_dc_df[mask_phi_tke_II_dc_df]


phi_tke_II_dc_df_final = phi_tke_II_dc_df.sort_values(by='z/L')
phi_tke_II_dc_neg = phi_tke_II_dc_df_final.loc[phi_tke_II_dc_df_final['z/L']<=0]
phi_tke_II_dc_pos = phi_tke_II_dc_df_final.loc[phi_tke_II_dc_df_final['z/L']>=0]


phi_tke_III_dc_df = pd.DataFrame()
phi_tke_III_dc_df['z/L'] = zL_df['zL_III_dc']
phi_tke_III_dc_df['phi_tke'] = -1*phi_tke_III_dc
# # can use to get rid of negative tke values
# phi_tke_III_dc_df['phi_tke_pos'] = np.where(phi_tke_III_dc_df['phi_tke']>=0,phi_tke_III_dc_df['phi_tke'],np.nan)
# mask_phi_tke_III_dc_df = np.isin(phi_tke_III_dc_df['phi_tke_pos'], phi_tke_III_dc_df['phi_tke'])
# phi_tke_III_dc_df_final = phi_tke_III_dc_df[mask_phi_tke_III_dc_df]


phi_tke_III_dc_df_final = phi_tke_III_dc_df.sort_values(by='z/L')
phi_tke_III_dc_neg = phi_tke_III_dc_df_final.loc[phi_tke_III_dc_df_final['z/L']<=0]
phi_tke_III_dc_pos = phi_tke_III_dc_df_final.loc[phi_tke_III_dc_df_final['z/L']>=0]


#%% COARE functional form for Phi_e (eq 38) and Phi_te (eq 39)
coare_zL_neg = np.arange(-4,0,0.005)
coare_zL_pos = np.arange(0,2,0.005)
eq38 = 2*(-1*coare_zL_neg)**(1/3)*(1-coare_zL_neg)**(2/3)
phi_energy = eq38
eq39 = kappa/3*(4*(-1*coare_zL_neg)**(4/3)*(1-coare_zL_neg)**(-1/3)+phi_energy)

print('done writing phi_tke as equations 38 and 39')
print('done at line 691')

#%%
plt.figure()
plt.plot(coare_zL_neg, -1*eq39, color = 'g', label = '-1*eq39')
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()
plt.title('$\phi_{te}$ functional form')

print('done at line 703')


#%% Phi_tke plots
plt.figure()
plt.scatter(phi_tke_I_dc_df_final['z/L'],phi_tke_I_dc_df_final['phi_tke'], color = 'dodgerblue', edgecolor = 'navy', label ='$\phi_{te}(z/L)$ d.c. L I')
plt.plot(coare_zL_neg, -1*eq39, color = 'k',linewidth=3, label = 'COARE functional form')
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{te}(z/L)$')
plt.title(title_windDir + "Level I: $\phi_{te}(z/L)$")
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_tke_II_dc_df_final['z/L'],phi_tke_II_dc_df_final['phi_tke'], color = 'orange', edgecolor = 'red', label ='$\phi_{te}(z/L)$ d.c. L II')
plt.plot(coare_zL_neg, -1*eq39, color = 'k',linewidth=3, label = 'COARE functional form')
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{te}(z/L)$')
plt.title(title_windDir + "Level II: $\phi_{te}(z/L)$")
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()

plt.figure()
plt.scatter(phi_tke_III_dc_df_final['z/L'],phi_tke_III_dc_df_final['phi_tke'], color = 'cyan', edgecolor = 'blue', label ='$\phi_{te}(z/L)$ d.c. L III')
plt.plot(coare_zL_neg, -1*eq39, color = 'k',linewidth=3, label = 'COARE functional form')
plt.xlabel("$z/L$")
plt.ylabel('$\phi_{te}(z/L)$')
plt.title(title_windDir + "Level III: $\phi_{te}(z/L)$")
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()



print('done at line 1507')


#%%
def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg

df_binEstimate_phi_tke_I_dc = binscatter(x='z/L', y='phi_tke', data=phi_tke_I_dc_df_final, ci=(3,3))
df_binEstimate_phi_tke_II_dc = binscatter(x='z/L', y='phi_tke', data=phi_tke_II_dc_df_final, ci=(3,3))
df_binEstimate_phi_tke_III_dc = binscatter(x='z/L', y='phi_tke', data=phi_tke_III_dc_df_final, ci=(3,3))


print('done at line 770')

#%% Phi_tke binned scatterplot
plt.figure()
sns.scatterplot(x='z/L', y='phi_tke', data=df_binEstimate_phi_tke_I_dc, color = 'dodgerblue', label = "binned $\phi_{te}(z/L)$: L I")
plt.errorbar('z/L', 'phi_tke', yerr='ci', data=df_binEstimate_phi_tke_I_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, -1*eq39, color = 'k',linewidth=2, label = 'COARE functional form')
plt.title(title_windDir + "Level I: $phi_{te}(z/L) (DC)$")
plt.xlim(-1.2,0.8)
# plt.ylim(-0.5,3)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_tke', data=df_binEstimate_phi_tke_II_dc, color = 'darkorange', label = "binned $\phi_{te}(z/L)$: L II")
plt.errorbar('z/L', 'phi_tke', yerr='ci', data=df_binEstimate_phi_tke_II_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, -1*eq39, color = 'k',linewidth=2, label = 'COARE functional form')
plt.title(title_windDir + "Level II: $phi_{te}(z/L) (DC)$")
plt.xlim(-1.2,0.8)
# plt.ylim(-0.5,3)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_tke', data=df_binEstimate_phi_tke_III_dc, color = 'blue', label = "binned $\phi_{te}(z/L)$: L III")
plt.errorbar('z/L', 'phi_tke', yerr='ci', data=df_binEstimate_phi_tke_III_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, -1*eq39, color = 'k',linewidth=2, label = 'COARE functional form')
plt.title(title_windDir + "Level III: $phi_{te}(z/L) (DC)$")
plt.xlim(-1.2,0.8)
# plt.ylim(-0.5,3)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_tke', data=df_binEstimate_phi_tke_II_dc, color = 'darkorange', label = "binned $\phi_{te}(z/L)$: L II")
plt.errorbar('z/L', 'phi_tke', yerr='ci', data=df_binEstimate_phi_tke_II_dc, color = 'orange', ls='', lw=2, alpha=0.2)
sns.scatterplot(x='z/L', y='phi_tke', data=df_binEstimate_phi_tke_I_dc, color = 'dodgerblue', label = "binned $\phi_{te}(z/L)$: L I")
plt.errorbar('z/L', 'phi_tke', yerr='ci', data=df_binEstimate_phi_tke_I_dc, color = 'dodgerblue', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, -1*eq39, color = 'k',linewidth=2, label = 'COARE functional form')
plt.title(title_windDir + "Level II: $phi_{te}(z/L) (DC)$")
plt.xlim(-1.2,0.8)
# plt.ylim(-0.5,3)
plt.legend()

print('done at line 800')
#%%
######################################################################
######################################################################
"""
Combine to a master DF --> we use this to determine Phi_Tp
"""
master_I_dc_df = pd.DataFrame()
master_I_dc_df['z/L'] = np.array(zL_df['zL_I_dc'])
master_I_dc_df['phi_m'] = np.array(phi_m_dc_df['phi_m I'])
master_I_dc_df['phi_eps'] = -1*np.array(phi_eps_dc_df['phi_eps I'])
master_I_dc_df['phi_tke'] = -1*np.array(phi_tke_dc_df['phi_tke I'])
master_I_dc_df['buoyancy'] = -1*np.array(zL_df['zL_I_dc'])
master_I_dc_df['phi_tp'] = -1*(np.array(master_I_dc_df['buoyancy']) + np.array(master_I_dc_df['phi_m']) + np.array(master_I_dc_df['phi_eps']) + np.array(master_I_dc_df['phi_tke']))


plt.figure()
plt.scatter(master_I_dc_df['z/L'],master_I_dc_df['phi_m'], label = 'phi_m')
plt.scatter(master_I_dc_df['z/L'],master_I_dc_df['phi_eps'], label = 'phi_eps')
plt.scatter(master_I_dc_df['z/L'],master_I_dc_df['phi_tke'], label = 'phi_tke')
plt.scatter(master_I_dc_df['z/L'],master_I_dc_df['buoyancy'], label = 'buoyancy')
plt.scatter(master_I_dc_df['z/L'],master_I_dc_df['phi_tp'], label = 'phi_tp')
plt.legend()
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.hlines(y=0, xmin=-4,xmax=2, color = 'k')
plt.vlines(x=0, ymin=-4,ymax=4, color = 'k')

master_II_dc_df = pd.DataFrame()
master_II_dc_df['z/L'] = np.array(zL_df['zL_II_dc'])
master_II_dc_df['phi_m'] = np.array(phi_m_dc_df['phi_m II'])
master_II_dc_df['phi_eps'] = -1*np.array(phi_eps_dc_df['phi_eps II'])
master_II_dc_df['phi_tke'] = -1*np.array(phi_tke_dc_df['phi_tke II'])
master_II_dc_df['buoyancy'] = -1*np.array(zL_df['zL_II_dc'])
master_II_dc_df['phi_tp'] = -1*(np.array(master_II_dc_df['buoyancy']) + np.array(master_II_dc_df['phi_m']) + np.array(master_II_dc_df['phi_eps']) + np.array(master_II_dc_df['phi_tke']))


plt.figure()
plt.scatter(master_II_dc_df['z/L'],master_II_dc_df['phi_m'], label = 'phi_m')
plt.scatter(master_II_dc_df['z/L'],master_II_dc_df['phi_eps'], label = 'phi_eps')
plt.scatter(master_II_dc_df['z/L'],master_I_dc_df['phi_tke'], label = 'phi_tke')
plt.scatter(master_II_dc_df['z/L'],master_II_dc_df['buoyancy'], label = 'buoyancy')
plt.scatter(master_II_dc_df['z/L'],master_II_dc_df['phi_tp'], label = 'phi_tp')
plt.legend()
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.hlines(y=0, xmin=-4,xmax=2, color = 'k')
plt.vlines(x=0, ymin=-4,ymax=4, color = 'k')



#%%
"""
NOW WE ARE MOVING ON TO PHI_pressure_transport (Phi_Tp)
"""
# edson 98 defines pressure as the residual od buoyancy, production, dissipation, and turbulent transport
# equation, wise, that means phi_tp (transport pressure) = eq34 - (eq39 + eq40 + z/L) 

coare_zL_neg = np.arange(-4,0,0.005) #also, buoyancy negative values
coare_zL_pos = np.arange(0,2,0.005)  #also, buoyancy positive values

eq34 = (1-15*coare_zL_neg)**(-1/3) #eq 34 for negative z/L Phi_M
e = 6
eq41 = 1+e*coare_zL_pos #eq 41 for posative z/L Phi_M

eq40 = ((1-coare_zL_neg)/(1-7*coare_zL_neg))-coare_zL_neg #for negative z/L Phi_Eps
eq42 = 1+(e-1)*coare_zL_pos #for positive z/L Phi_Eps

eq38 = 2*(-1*coare_zL_neg)**(1/3)*(1-coare_zL_neg)**(2/3)
phi_energy = eq38
eq39 = kappa/3*(4*(-1*coare_zL_neg)**(4/3)*(1-coare_zL_neg)**(-1/3)+phi_energy)


phi_tp_neg = eq34 + -1*coare_zL_neg - eq39 - eq40
phi_tp_pos = eq41 - coare_zL_pos -eq42

plt.figure()
plt.plot(coare_zL_neg, -1*phi_tp_neg, color = 'red', label = '-1*$\phi_{tp}$')
plt.plot(coare_zL_pos, -1*phi_tp_pos, color = 'red',)
plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()
plt.title('$\phi_{tp}$ functional form')

print('done at line 1665')


#%%
def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg

df_binEstimate_phi_tp_I_dc = binscatter(x='z/L', y='phi_tp', data=master_I_dc_df, ci=(3,3), randcut=1)
df_binEstimate_phi_tp_II_dc = binscatter(x='z/L', y='phi_tp', data=master_II_dc_df, ci=(3,3), randcut = 1)

#%% Phi_tke binned scatterplot
plt.figure()
sns.scatterplot(x='z/L', y='phi_tp', data=df_binEstimate_phi_tp_II_dc, color = 'orange', label = "binned $\phi_{tp}(z/L)$: L II")
plt.errorbar('z/L', 'phi_tp', yerr='ci', data=df_binEstimate_phi_tp_II_dc, color = 'orange', ls='', lw=2, alpha=0.2)
sns.scatterplot(x='z/L', y='phi_tp', data=df_binEstimate_phi_tp_I_dc, color = 'dodgerblue', label = "binned $\phi_{tp}(z/L)$: L I")
plt.errorbar('z/L', 'phi_tp', yerr='ci', data=df_binEstimate_phi_tp_I_dc, color = 'dodgerblue', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, -1*phi_tp_neg, color = 'k', label = 'COARE funcitonal form')
plt.title(title_windDir + "Level I: $phi_{tp}(z/L) (DC)$")
plt.xlim(-1.2,0.8)
plt.ylim(-0.5,3)
plt.legend(loc = 'upper right')
plt.grid()


#%%
# compare pressure transport to TKE turbulent transport (which we expect to be equal and opposite)
plt.figure(figsize=(6,5))
sns.scatterplot(x='z/L', y='phi_tke', data=df_binEstimate_phi_tke_I_dc, color = 'red', label = "$\phi_{T_{E}}$")
plt.errorbar('z/L', 'phi_tke', yerr='ci', data=df_binEstimate_phi_tke_I_dc, color = 'red', ls='', lw=2, alpha=0.2, label = "error $\phi_{T_{E}}$")
sns.scatterplot(x='z/L', y='phi_tp', data=df_binEstimate_phi_tp_I_dc, color = 'gray', label = "$\phi_{T_{P}}$")
plt.errorbar('z/L', 'phi_tp', yerr='ci', data=df_binEstimate_phi_tp_I_dc, color = 'gray', ls='', lw=2, alpha=0.2, label = "error $\phi_{T_{P}}$")
plt.xlabel("$\zeta = z/L$")
plt.ylabel('$\phi_{T_{E}}$, $\phi_{T_{P}}$')
plt.title("Level I: $\phi_{T_{E}}(\zeta)$ and $\phi_{T_{P}}(\zeta)$\nNear-Neutral Conditions")
plt.xlim(-0.5,0.5)
# plt.ylim(-0.5,3)
# plt.yscale('log')
plt.legend(loc = 'upper right')
plt.grid()
plt.savefig(plot_savePath + "binnedScatterplot_TransportTerms.png",dpi=300)
plt.savefig(plot_savePath + "binnedScatterplot_TransportTerms.pdf")
#%% Another way of checking if they are equal and opposite is to see if their sum is zero
transport_diff_LI = (phi_tke_I_dc_df_final['phi_tke']-master_I_dc_df['phi_tp'])/(kappa*np.array(z_LI)/(np.array(usr_LI)**3))
transport_diff_LII = (phi_tke_II_dc_df_final['phi_tke']-master_II_dc_df['phi_tp'])/(kappa*np.array(z_LII)/(np.array(usr_LII)**3))

plt.figure()
plt.scatter(np.arange(len(transport_diff_LII)), transport_diff_LII, s=5, color = 'darkorange', label = 'LII')
plt.scatter(np.arange(len(transport_diff_LI)), transport_diff_LI, s=5, color = 'dodgerblue', label = 'LI')
plt.title('$T_e - T_p$ $[m^2/s^3]$')
plt.legend(loc='lower right')
plt.ylabel('$T_e - T_p$ $[m^2/s^3]$')
plt.xlabel('time index')
plt.ylim(-1,1)


transport_df_LI = pd.DataFrame()
transport_df_LI['Te-Tp'] = transport_diff_LI
transport_df_LI['z/L'] = np.array(zL_df['zL_I_dc'])

transport_df_LII = pd.DataFrame()
transport_df_LII['Te-Tp'] = transport_diff_LII
transport_df_LII['z/L'] = np.array(zL_df['zL_II_dc'])

def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg

df_binEstimate_transportDiff_I_dc = binscatter(x='z/L', y='Te-Tp', data=transport_df_LI, ci=(3,3), randcut=1)
df_binEstimate_transportDiff_II_dc = binscatter(x='z/L', y='Te-Tp', data=transport_df_LII, ci=(3,3), randcut=1)


# transport terms difference binned scatterplot
plt.figure(figsize=(6,5))
sns.scatterplot(x='z/L', y='Te-Tp', data=df_binEstimate_transportDiff_II_dc, color = 'darkorange', label = "binned $T_e+T_p$: L II")
plt.errorbar('z/L', 'Te-Tp', yerr='ci', data=df_binEstimate_transportDiff_II_dc, color = 'darkorange', ls='', lw=2, alpha=0.2, label = 'L II error')
sns.scatterplot(x='z/L', y='Te-Tp', data=df_binEstimate_transportDiff_I_dc, color = 'dodgerblue', label = "binned $T_e+T_p$: L I")
plt.errorbar('z/L', 'Te-Tp', yerr='ci', data=df_binEstimate_transportDiff_I_dc, color = 'navy', ls='', lw=2, alpha=0.2, label = 'L I error')
# plt.plot(coare_zL_neg, -1*phi_tp, color = 'k', label = 'COARE funcitonal form')
plt.title(r"$T_e+T_p$")
plt.xlim(-1.2,0.8)
# plt.ylim(-0.5,3)
plt.legend(loc = 'upper left')
plt.grid()
plt.savefig(plot_savePath + "binnedScatterplot_TransportTermDifferences_TeMinusTp.png",dpi=300)
plt.savefig(plot_savePath + "binnedScatterplot_TransportTermDifferences_TeMinusTp.pdf")