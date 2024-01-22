# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:39:58 2023

@author: oaklin keefe

This file is used to calculate monin obukhov mixing lengthscale, L, and then determine a stability parameter z/L (zeta)

INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    prodTerm_combinedAnalysis.csv
    z_air_side_combinedAnalysis.csv
    thetaV_combinedAnalysis.csv
    
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

file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
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
################################################################################################################################
################################################################################################################################
# COARE THINGS COMMENTED OUT BELOW
"""
#%%
# add in coare to see how the L's compare
file_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
L_coare_wind = pd.DataFrame()
usr_coare_wind = pd.DataFrame()
sonic_arr = ['1','2','3','4']
for sonic_num in sonic_arr:
# sonic_num = str(1)
    file_name = 'coare_outputs_s'+sonic_num+'_Warm_UbarGreaterThan2ms.txt'
    A_hdr = 'usr\ttau\thsb\thlb\thbb\thsbb\thlwebb\ttsr\tqsr\tzo\tzot\tzoq\tCd\t'
    A_hdr += 'Ch\tCe\tL\tzeta\tdT_skinx\tdq_skinx\tdz_skin\tUrf\tTrf\tQrf\t'
    A_hdr += 'RHrf\tUrfN\tTrfN\tQrfN\tlw_net\tsw_net\tLe\trhoa\tUN\tU10\tU10N\t'
    A_hdr += 'Cdn_10\tChn_10\tCen_10\thrain\tQs\tEvap\tT10\tT10N\tQ10\tQ10N\tRH10\t'
    A_hdr += 'P10\trhoa10\tgust\twc_frac\tEdis\tdT_warm\tdz_warm\tdT_warm_to_skin\tdu_warm'
    coare_warm = np.genfromtxt(file_path + file_name, delimiter='\t')
    L_coare_wind['L_sonic'+sonic_num] = np.array(coare_warm[:,15])
    usr_coare_wind['usr_sonic'+sonic_num] = np.array(coare_warm[:,0])
    print('Wind stress only: did this with sonic '+sonic_num)
usr_coare_wind.to_csv(file_path+"usr_coare_allFall.csv")
# L_coare_wind_Cp = pd.DataFrame()
# usr_coare_wind_Cp = pd.DataFrame()
# sonic_arr = ['1','2','3','4']
# for sonic_num in sonic_arr:
# # sonic_num = str(1)
#     file_name = 'coare_outputs_s'+sonic_num+'_Warm_UbarGreaterThan2ms_withCp.txt'
#     A_hdr = 'usr\ttau\thsb\thlb\thbb\thsbb\thlwebb\ttsr\tqsr\tzo\tzot\tzoq\tCd\t'
#     A_hdr += 'Ch\tCe\tL\tzeta\tdT_skinx\tdq_skinx\tdz_skin\tUrf\tTrf\tQrf\t'
#     A_hdr += 'RHrf\tUrfN\tTrfN\tQrfN\tlw_net\tsw_net\tLe\trhoa\tUN\tU10\tU10N\t'
#     A_hdr += 'Cdn_10\tChn_10\tCen_10\thrain\tQs\tEvap\tT10\tT10N\tQ10\tQ10N\tRH10\t'
#     A_hdr += 'P10\trhoa10\tgust\twc_frac\tEdis\tdT_warm\tdz_warm\tdT_warm_to_skin\tdu_warm'
#     coare_warm = np.genfromtxt(file_path + file_name, delimiter='\t')
#     L_coare_wind_Cp['L_sonic'+sonic_num] = np.array(coare_warm[:,15])
#     usr_coare_wind_Cp['usr_sonic'+sonic_num] = np.array(coare_warm[:,0])
#     print('Cp and wind stress: did this with sonic '+sonic_num)

# L_coare_wind_sigH = pd.DataFrame()
# usr_coare_wind_sigH = pd.DataFrame()
# sonic_arr = ['1','2','3','4']
# for sonic_num in sonic_arr:
# # sonic_num = str(1)
#     file_name = 'coare_outputs_s'+sonic_num+'_Warm_UbarGreaterThan2ms_withSigH.txt'
#     A_hdr = 'usr\ttau\thsb\thlb\thbb\thsbb\thlwebb\ttsr\tqsr\tzo\tzot\tzoq\tCd\t'
#     A_hdr += 'Ch\tCe\tL\tzeta\tdT_skinx\tdq_skinx\tdz_skin\tUrf\tTrf\tQrf\t'
#     A_hdr += 'RHrf\tUrfN\tTrfN\tQrfN\tlw_net\tsw_net\tLe\trhoa\tUN\tU10\tU10N\t'
#     A_hdr += 'Cdn_10\tChn_10\tCen_10\thrain\tQs\tEvap\tT10\tT10N\tQ10\tQ10N\tRH10\t'
#     A_hdr += 'P10\trhoa10\tgust\twc_frac\tEdis\tdT_warm\tdz_warm\tdT_warm_to_skin\tdu_warm'
#     coare_warm = np.genfromtxt(file_path + file_name, delimiter='\t')
#     L_coare_wind_sigH['L_sonic'+sonic_num] = np.array(coare_warm[:,15])
#     usr_coare_wind_sigH['usr_sonic'+sonic_num] = np.array(coare_warm[:,0])
#     print('sigH and wind stress: did this with sonic '+sonic_num)

# L_coare_wind_Cp_sigH = pd.DataFrame()
# usr_coare_wind_Cp_sigH = pd.DataFrame()
# sonic_arr = ['1','2','3','4']
# for sonic_num in sonic_arr:
# # sonic_num = str(1)
#     file_name = 'coare_outputs_s'+sonic_num+'_Warm_UbarGreaterThan2ms_withCp_SigH.txt'
#     A_hdr = 'usr\ttau\thsb\thlb\thbb\thsbb\thlwebb\ttsr\tqsr\tzo\tzot\tzoq\tCd\t'
#     A_hdr += 'Ch\tCe\tL\tzeta\tdT_skinx\tdq_skinx\tdz_skin\tUrf\tTrf\tQrf\t'
#     A_hdr += 'RHrf\tUrfN\tTrfN\tQrfN\tlw_net\tsw_net\tLe\trhoa\tUN\tU10\tU10N\t'
#     A_hdr += 'Cdn_10\tChn_10\tCen_10\thrain\tQs\tEvap\tT10\tT10N\tQ10\tQ10N\tRH10\t'
#     A_hdr += 'P10\trhoa10\tgust\twc_frac\tEdis\tdT_warm\tdz_warm\tdT_warm_to_skin\tdu_warm'
#     coare_warm = np.genfromtxt(file_path + file_name, delimiter='\t')
#     L_coare_wind_Cp_sigH['L_sonic'+sonic_num] = np.array(coare_warm[:,15])
#     usr_coare_wind_Cp_sigH['usr_sonic'+sonic_num] = np.array(coare_warm[:,0])
#     print('Cp, sighH, and wind stress: did this with sonic '+sonic_num)

#%%
plt.figure()
plt.plot(L_coare_wind['L_sonic1'], label = 's1')
plt.plot(L_coare_wind['L_sonic2'], label = 's2')
plt.plot(L_coare_wind['L_sonic3'], label = 's3')
plt.plot(L_coare_wind['L_sonic4'], label = 's4')
plt.legend()
plt.title('COARE L by sonic')
plt.ylim(-10000,1000)

r_L_coare = L_coare_wind.corr()
print(r_L_coare)

#%% hampel filtering

from hampel import hampel

sonic_arr = ['1','2','3','4']
# coare_arrays = [L_coare_wind, L_coare_wind_Cp, L_coare_wind_sigH, L_coare_wind_Cp_sigH]
coare_arrays = [L_coare_wind,]

L_coare_wind_despiked = pd.DataFrame()
# L_coare_wind_Cp_despiked = pd.DataFrame()
# L_coare_wind_sigH_despiked = pd.DataFrame()
# L_coare_wind_Cp_sigH_despiked = pd.DataFrame()

# despiked_coare_arrays = [L_coare_wind_despiked, L_coare_wind_Cp_despiked, L_coare_wind_sigH_despiked, L_coare_wind_Cp_sigH_despiked]
despiked_coare_arrays = [L_coare_wind_despiked, ]

for i in range(len(coare_arrays)):
    for sonic in sonic_arr:
    
        L_array = coare_arrays[i]['L_sonic'+sonic]
        
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
        despiked_coare_arrays[i]['L_sonic'+sonic] = L_outlier_in_Ts2
        print("coare: "+str(sonic))
        # L_despiked_2times = L_outlier_in_Ts2
    # L_coare_despike = L_despiked_2times
#%%

plt.figure()
plt.plot(L_coare_wind_despiked['L_sonic4'], label = 's4')
plt.plot(L_coare_wind_despiked['L_sonic3'], label = 's3')
plt.plot(L_coare_wind_despiked['L_sonic2'], label = 's2')
plt.plot(L_coare_wind_despiked['L_sonic1'], label = 's1')
# plt.plot(L_coare_wind_Cp_despiked['L_sonic1'], label = 'Cp')
# plt.plot(L_coare_wind_sigH_despiked['L_sonic1'], label = 'sigH')
# plt.plot(L_coare_wind_Cp_sigH_despiked['L_sonic1'], label = 'all')
plt.legend()

r_L_coare_despiked = L_coare_wind_despiked.corr()
print(r_L_coare_despiked)

print('done with hampel filtering')
print('done at line 223')   




usr_s1_coare = usr_coare_wind['usr_sonic1']

sonic1_usr_df = pd.DataFrame()
sonic1_usr_df['usr_coare'] = np.array(usr_s1_coare)
sonic1_usr_df['usr_dc_old'] = np.array(usr_s1_old)
sonic1_usr_df['usr_dc_new'] = np.array(usr_s1)
r_usr_sonic1 = sonic1_usr_df.corr()
print(r_usr_sonic1)


usr_s1_coare = usr_coare_wind['usr_sonic1']
usr_s2_coare = usr_coare_wind['usr_sonic2']
usr_s3_coare = usr_coare_wind['usr_sonic3']
usr_s4_coare = usr_coare_wind['usr_sonic4']

"""




#%%

usr_LI = np.array(usr_s1+usr_s2)/2
# usr_LI_coare = np.array(usr_s1_coare+usr_s2_coare)/2
Tbar_LI = np.array(sonic1_df['Tbar']+sonic2_df['Tbar'])/2
WpTp_bar_LI = -1*(np.array(sonic1_df['WpTp_bar']+sonic2_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

usr_LII = np.array(usr_s2+usr_s3)/2
# usr_LII_coare = np.array(usr_s2_coare+usr_s3_coare)/2
Tbar_LII = np.array(sonic2_df['Tbar']+sonic3_df['Tbar'])/2
WpTp_bar_LII = -1*(np.array(sonic2_df['WpTp_bar']+sonic3_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

usr_LIII = np.array(usr_s3+usr_s4)/2
# usr_LIII_coare = np.array(usr_s3_coare+usr_s4_coare)/2
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
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

L_1_dc_unspiked = -1*(np.array(usr_s1)**3)*np.array(thetaV_df['thetaV_sonic1'])/(g*kappa*(-1*np.array(sonic1_df['WpTp_bar'])))
# L_1_coare = L_coare_wind_despiked['L_sonic1']

L_2_dc_unspiked = -1*(np.array(usr_s2)**3)*np.array(thetaV_df['thetaV_sonic2'])/(g*kappa*(-1*np.array(sonic2_df['WpTp_bar'])))
# L_2_coare = L_coare_wind_despiked['L_sonic2']

L_3_dc_unspiked = -1*(np.array(usr_s3)**3)*np.array(thetaV_df['thetaV_sonic3'])/(g*kappa*(-1*np.array(sonic3_df['WpTp_bar'])))
# L_3_coare = L_coare_wind_despiked['L_sonic3']

L_4_dc_unspiked = -1*(np.array(usr_s4)**3)*np.array(thetaV_df['thetaV_sonic4'])/(g*kappa*(-1*np.array(sonic4_df['WpTp_bar'])))
# L_4_coare = L_coare_wind_despiked['L_sonic4']

L_dc_df_unspiked = pd.DataFrame()
L_dc_df_unspiked['L_sonic1'] = np.array(L_1_dc_unspiked)
L_dc_df_unspiked['L_sonic2'] = np.array(L_2_dc_unspiked)
L_dc_df_unspiked['L_sonic3'] = np.array(L_3_dc_unspiked)
L_dc_df_unspiked['L_sonic4'] = np.array(L_4_dc_unspiked)


plt.figure()
# plt.plot(L_4_coare, label = 'coare s4 despiked')
plt.plot(L_4_dc_unspiked, label = 'dc s4')
plt.legend()
plt.title('comparison of L: COARE vs. DC')


# plt.figure()
# plt.plot(L_coare_df_despiked.index, label='coare despiked')
# plt.plot(L_dc_df_unspiked.index, label='dc with spikes')
# # plt.plot(test_df4.index, label = 'ubar')
# plt.legend()
# # plt.ylim(-100,100)
# # plt.xlim(1400,1500)
# plt.title('Comparison of indices after excluding bad wind dir.')
# print('done at line 198')

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
# L_dirCov_despike = L_despiked_2times

# L_dc_df['index_arr'] = L_coare_df.index
# L_dc_df = L_dc_df.set_index(L_dc_df['index_arr'])
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
# z_over_L_I_coare = np.array(z_I)/np.array(L_I_coare)
# z_over_L_II_coare = np.array(z_II)/np.array(L_II_coare)
# z_over_L_III_coare = np.array(z_III)/np.array(L_III_coare)


#%%
# zL_LI_df = pd.DataFrame()
# zL_LI_df['zL_coare'] = np.array(z_over_L_I_dc)
# zL_LI_df['zL_dc'] = np.array(z_over_L_I_coare)
# r_zL_LI = zL_LI_df.corr()
# print(r_zL_LI)

# zL_LII_df = pd.DataFrame()
# zL_LII_df['zL_coare'] = np.array(z_over_L_II_dc)
# zL_LII_df['zL_dc'] = np.array(z_over_L_II_coare)
# r_zL_LII = zL_LII_df.corr()
# print(r_zL_LII)

# zL_LIII_df = pd.DataFrame()
# zL_LIII_df['zL_coare'] = np.array(z_over_L_III_dc)
# zL_LIII_df['zL_dc'] = np.array(z_over_L_III_coare)
# r_zL_LIII = zL_LIII_df.corr()
# print(r_zL_LIII)

plt.figure()
plt.plot(z_over_L_I_dc, label = 'I')
plt.plot(z_over_L_II_dc, label = 'II')
plt.plot(z_over_L_III_dc, label = 'III')
plt.legend()
plt.ylim(-30,5)
plt.title('z/L DC by level')


# plt.figure()
# plt.plot(z_over_L_I_coare, label = 'I')
# plt.plot(z_over_L_II_coare, label = 'II')
# plt.plot(z_over_L_III_coare, label = 'III')
# plt.legend()
# plt.ylim(-30,10)
# plt.title('z/L COARE by level')
#%%

zL_I = pd.DataFrame()
zL_I['dc'] = z_over_L_I_dc
zL_I['coare'] = z_over_L_I_coare

# plt.figure()
# sns.boxplot(data=zL_I[["dc", "coare"]], orient="h")
# plt.xlim(-1,1)
# plt.title('z/L Level I')

plt.figure()
plt.scatter(zL_I.index, zL_I['dc'], label = 'dc', color = 'blue')
plt.scatter(zL_I.index, zL_I['coare'], label = 'coare', color = 'orange')
plt.hlines(y=0.25,xmin=0, xmax = 4395, color = 'k')
plt.hlines(y=-0.35,xmin=0, xmax = 4395, color = 'k')
plt.legend()
plt.ylim(-2,2)
plt.title('z/L comparison dc vs. coare: level I')

plt.figure()
plt.plot(z_over_L_I_dc, label = 'dc', color = 'blue')
plt.plot(z_over_L_I_coare, label = 'coare', color = 'orange')
plt.hlines(y=0.25,xmin=0, xmax = 4395, color = 'k')
plt.hlines(y=-0.35,xmin=0, xmax = 4395, color = 'k')
plt.legend()
plt.ylim(-2,2)
plt.title('z/L comparison dc vs. coare: level I')
#%%

zL_II = pd.DataFrame()
zL_II['dc'] = z_over_L_II_dc
zL_II['coare'] = z_over_L_II_coare

# plt.figure()
# sns.boxplot(data=zL_II[["dc", "coare"]], orient="h")
# plt.xlim(-1,1)
# plt.title('z/L Level II')

plt.figure()
plt.scatter(zL_II.index, zL_II['dc'], label = 'dc', color = 'blue')
plt.scatter(zL_II.index, zL_II['coare'], label = 'coare', color = 'orange')
plt.hlines(y=0.5,xmin=0, xmax = 4395, color = 'k')
plt.hlines(y=-0.75,xmin=0, xmax = 4395, color = 'k')
# plt.hlines(y=-0.5, xmin=0, xmax=4395, color = 'k')
plt.legend()
plt.ylim(-2,2)
# plt.xlim(2975,3050)
plt.title('z/L comparison dc vs. coare: level II')


plt.figure()
plt.plot(z_over_L_II_dc, label = 'dc', color = 'blue')
plt.plot(z_over_L_II_coare, label = 'coare', color = 'orange')
plt.hlines(y=0.5,xmin=0, xmax = 4395, color = 'k')
plt.hlines(y=-0.75,xmin=0, xmax = 4395, color = 'k')
# plt.hlines(y=-0.5, xmin=0, xmax=4395, color = 'k')
plt.legend()
plt.ylim(-2,2)
# plt.xlim(2975,3050)
plt.title('z/L comparison dc vs. coare: level II')
#%%
plt.figure()
plt.scatter(zL_II['dc'],zL_II['coare'], color = 'orange', edgecolor = 'red', label = 'L II')
plt.scatter(zL_I['dc'],zL_I['coare'], color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-2, 2], [-2, 2], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title(r"Buoyancy Parameter ($\zeta = z/L$ [-])", fontsize=16)
plt.xlabel('DC')
plt.ylabel('COARE')
plt.legend(loc='upper left')
plt.xlim(-5,3)
plt.ylim(-5,3)
# plt.axis('square')

#%%

zL_df = pd.DataFrame()
zL_df['z/L I coare'] = z_over_L_I_coare
zL_df['z/L II coare'] = z_over_L_II_coare
zL_df['z/L III coare'] = z_over_L_III_coare
zL_df['z/L I dc'] = z_over_L_I_dc
zL_df['z/L II dc'] = z_over_L_II_dc
zL_df['z/L III dc'] = z_over_L_III_dc
zL_df['z/L 1 dc'] = np.array(z_s1/L_1_dc)
zL_df['z/L 2 dc'] = np.array(z_s1/L_2_dc)
zL_df['z/L 3 dc'] = np.array(z_s1/L_3_dc)
zL_df['z/L 4 dc'] = np.array(z_s1/L_4_dc)
zL_df['z/L 1 coare'] = np.array(z_s1/L_1_coare)
zL_df['z/L 2 coare'] = np.array(z_s1/L_2_coare)
zL_df['z/L 3 coare'] = np.array(z_s1/L_3_coare)
zL_df['z/L 4 coare'] = np.array(z_s1/L_4_coare)

zL_df.to_csv(file_path + 'zL_allFall.csv')

#%%
zL_I_dc_arr = np.array(z_over_L_I_dc)
percentile_95_I_dc = np.nanpercentile(np.abs(zL_I_dc_arr), 95)
percentile_99_I_dc = np.nanpercentile(np.abs(zL_I_dc_arr), 99)
print(percentile_99_I_dc)
zL_I_dc_newArr_95 = np.where(np.abs(zL_I_dc_arr) > percentile_95_I_dc, np.nan, zL_I_dc_arr)
zL_I_dc_newArr_99 = np.where(np.abs(zL_I_dc_arr) > percentile_99_I_dc, np.nan, zL_I_dc_arr)

zL_I_coare_arr = np.array(z_over_L_I_coare)
percentile_95_I_coare = np.nanpercentile(np.abs(zL_I_coare_arr), 95)
percentile_99_I_coare = np.nanpercentile(np.abs(zL_I_coare_arr), 99)
print(percentile_95_I_coare)
zL_I_coare_newArr_95 = np.where(np.abs(zL_I_coare_arr) > percentile_95_I_coare, np.nan, zL_I_coare_arr)
zL_I_coare_newArr_99 = np.where(np.abs(zL_I_coare_arr) > percentile_99_I_coare, np.nan, zL_I_coare_arr)


plt.figure()
plt.plot(zL_I_dc_newArr_95, label = 'dc')
plt.plot(zL_I_coare_newArr_95, label = 'coare')
plt.legend()
plt.title('95% z/L Level I')

zL_95p_I_df = pd.DataFrame()
zL_95p_I_df['dc'] = zL_I_dc_newArr_95
zL_95p_I_df['coare'] = zL_I_coare_newArr_95

r_zL_95p_I = zL_95p_I_df.corr()
print(r_zL_95p_I)

plt.figure()
plt.plot(zL_I_dc_newArr_99, label = 'dc')
plt.plot(zL_I_coare_newArr_99, label = 'coare')
plt.legend()
plt.title('99% z/L Level I')

zL_99p_I_df = pd.DataFrame()
zL_99p_I_df['dc'] = zL_I_dc_newArr_99
zL_99p_I_df['coare'] = zL_I_coare_newArr_99

r_zL_99p_I = zL_99p_I_df.corr()
print(r_zL_99p_I)
#%%

zL_II_dc_arr = np.array(z_over_L_II_dc)
percentile_95_II_dc = np.nanpercentile(np.abs(zL_II_dc_arr), 95)
percentile_99_II_dc = np.nanpercentile(np.abs(zL_II_dc_arr), 99)
print(percentile_95_II_dc)
zL_II_dc_newArr_95 = np.where(np.abs(zL_II_dc_arr) > percentile_95_II_dc, np.nan, zL_II_dc_arr)
zL_II_dc_newArr_99 = np.where(np.abs(zL_II_dc_arr) > percentile_99_II_dc, np.nan, zL_II_dc_arr)

zL_II_coare_arr = np.array(z_over_L_II_coare)
percentile_95_II_coare = np.nanpercentile(zL_II_coare_arr, 95)
percentile_99_II_coare = np.nanpercentile(zL_II_coare_arr, 99)
print(percentile_95_II_coare)
zL_II_coare_newArr_95 = np.where(np.abs(zL_II_coare_arr) > percentile_95_II_coare, np.nan, zL_II_coare_arr)
zL_II_coare_newArr_99 = np.where(np.abs(zL_II_coare_arr) > percentile_99_II_coare, np.nan, zL_II_coare_arr)


plt.figure()
plt.plot(zL_II_dc_newArr_95, label = 'dc')
plt.plot(zL_II_coare_newArr_95, label = 'coare')
plt.legend()
plt.title('95% z/L Level II')


zL_95p_II_df = pd.DataFrame()
zL_95p_II_df['dc'] = zL_II_dc_newArr_95
zL_95p_II_df['coare'] = zL_II_coare_newArr_95

r_zL_95p_II = zL_95p_II_df.corr()
print(r_zL_95p_II)

plt.figure()
plt.plot(zL_II_dc_newArr_99, label = 'dc')
plt.plot(zL_II_coare_newArr_99, label = 'coare')
plt.legend()
plt.title('95% z/L Level II')


zL_99p_II_df = pd.DataFrame()
zL_99p_II_df['dc'] = zL_II_dc_newArr_99
zL_99p_II_df['coare'] = zL_II_coare_newArr_99

r_zL_99p_II = zL_99p_II_df.corr()
print(r_zL_99p_II)

#%%
sonic1_L_df = pd.DataFrame()
sonic1_L_df['L_coare'] = np.array(L_1_coare)
sonic1_L_df['L_dc'] = np.array(L_1_dc)
r_L_sonic1 = sonic1_L_df.corr()
print(r_L_sonic1)


sonic2_L_df = pd.DataFrame()
sonic2_L_df['L_coare'] = np.array(L_2_coare)
sonic2_L_df['L_dc'] = np.array(L_2_dc)
r_L_sonic2 = sonic2_L_df.corr()
print(r_L_sonic2)

sonic3_L_df = pd.DataFrame()
sonic3_L_df['L_coare'] = np.array(L_3_coare)
sonic3_L_df['L_dc'] = np.array(L_3_dc)
r_L_sonic3 = sonic3_L_df.corr()
print(r_L_sonic3)

sonic4_L_df = pd.DataFrame()
sonic4_L_df['L_coare'] = np.array(L_4_coare)
sonic4_L_df['L_dc'] = np.array(L_4_dc)
r_L_sonic4 = sonic4_L_df.corr()
print(r_L_sonic4)

plt.figure()
plt.plot(L_2_coare, label = 'coare s2 despiked')
plt.plot(L_2_dc, label = 'dc s2')
plt.legend()
plt.title('sonic 2 comparison of L: COARE vs. DC')
plt.ylim(-1000,1000)

plt.figure()
plt.scatter(L_1_coare, L_1_dc, label = 's1')
plt.scatter(L_2_coare, L_2_dc, label = 's2')
plt.scatter(L_3_coare, L_3_dc, label = 's3')
# plt.scatter(L_4_coare, L_4_dc, label = 's4')
plt.hlines(y=0,xmin=-4000,xmax=4000, color = 'k')
plt.vlines(x=0,ymin=-4000,ymax=4000, color = 'k')
plt.xlabel('coare')
plt.ylabel('dc')
plt.legend()
plt.title("L comparison coare v. dc")
plt.ylim(-100,100)
plt.xlim(-100,100)

#%%
plt.figure()
plt.plot(L_1_coare, label = 'coare s1 despiked')
plt.plot(L_1_dc, label = 'dc s1 despiked')
plt.legend()
plt.title('sonic 1: comparison of L: COARE vs. DC')
plt.xlabel('time')
plt.ylabel('MO-Length (L) [m]')

plt.figure()
plt.plot(L_2_coare, label = 'coare s2 despiked')
plt.plot(L_2_dc, label = 'dc s2 despiked')
plt.legend()
plt.title('sonic 2: comparison of L: COARE vs. DC')
plt.xlabel('time')
plt.ylabel('MO-Length (L) [m]')

plt.figure()
plt.plot(L_3_coare, label = 'coare s3 despiked')
plt.plot(L_3_dc, label = 'dc s3 despiked')
plt.legend()
plt.title('sonic 3: comparison of L: COARE vs. DC')
plt.xlabel('time')
plt.ylabel('MO-Length (L) [m]')

plt.figure()
plt.plot(L_4_coare, label = 'coare s4 despiked')
plt.plot(L_4_dc, label = 'dc s4 despiked')
plt.legend()
plt.title('sonic 4: comparison of L: COARE vs. DC')
plt.xlabel('time')
plt.ylabel('MO-Length (L) [m]')


#%%
plt.figure()
plt.plot(L_4_coare, label='coare')
plt.plot(L_4_dc, label='dc')
plt.legend()
plt.ylim(-1000,1000)
plt.title('Comparison of MO-Length (L) at SONIC 4')
print('done at line 198')
#%%
## getting z/L for the x-axis
L_I_dc = -1*(usr_LI**3)*Tbar_LI/(g*kappa*WpTp_bar_LI)
L_I_coare = (L_coare_wind_despiked['L_sonic1']+L_coare_wind_despiked['L_sonic2'])/2
# L_I_coare = np.array(L_coare_df['L_sonic1'])

L_II_dc = -1*(usr_LII**3)*Tbar_LII/(g*kappa*WpTp_bar_LII)
L_II_coare = (L_coare_wind_despiked['L_sonic2']+L_coare_wind_despiked['L_sonic3'])/2
# L_II_coare = np.array(L_coare_df['L_sonic2'])

L_III_dc = -1*(usr_LIII**3)*Tbar_LIII/(g*kappa*WpTp_bar_LIII)
L_III_coare = (L_coare_wind_despiked['L_sonic3']+L_coare_wind_despiked['L_sonic4'])/2
# L_III_coare = np.array(L_coare_df['L_sonic3'])
"""
**** make sure these match

""" 
plt.figure()
plt.plot(L_I_dc, label='dc', color = 'orange')
plt.plot(L_I_coare, label='coare', color = 'blue')
plt.legend()
plt.ylim(-5000,5000)
plt.title('Comparison of MO-Length (L) at LEVEL I')

plt.figure()
plt.plot(L_II_dc, label='dc', color = 'orange')
plt.plot(L_II_coare, label='coare', color = 'blue')
plt.legend()
plt.ylim(-5000,5000)
plt.title('Comparison of MO-Length (L) at LEVEL II')

plt.figure()
plt.plot(L_III_dc, label='dc', color = 'orange')
plt.plot(L_III_coare, label='coare', color = 'blue')
plt.legend()
plt.ylim(-5000,5000)
plt.title('Comparison of MO-Length (L) at LEVEL III')

plt.figure()
plt.scatter(L_I_coare, L_I_dc, label = 'LI')
plt.scatter(L_II_coare, L_II_dc, label = 'LII')

# plt.scatter(L_4_coare, L_4_dc, label = 's4')
plt.hlines(y=0,xmin=-4000,xmax=4000, color = 'k')
plt.vlines(x=0,ymin=-4000,ymax=4000, color = 'k')
plt.xlabel('coare')
plt.ylabel('dc')
plt.legend()
plt.title("L comparison coare v. dc")
plt.ylim(-500,500)
plt.xlim(-500,500)

#%%
from matplotlib.ticker import PercentFormatter
# data = np.array(wave_age_df['waveAge_usr4']).flatten()
# plt.figure()
# n = 55
# # plt.hist(wave_age_df['waveAge_usr4'],n,edgecolor='white')
# plt.hist(data, n, weights=np.ones(len(data)) / len(data), edgecolor='white')
data_a = L_I_coare
data_b = L_I_dc
plt.figure()
plt.hist(data_b, bins = 50, range = (-1000,1000),  weights=np.ones(len(data_b)) / len(data_b), color = 'gray',edgecolor = 'white', label = 'dc')
plt.hist(data_a, bins = 50, range = (-1000,1000),  weights=np.ones(len(data_a)) / len(data_a), color = 'black',edgecolor = 'white', label = 'coare')
plt.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('Histogram Comparison of L outputs: Level I')
#%%

# # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
# # sns.set(style="darkgrid")
# # df = sns.load_dataset("iris")
 
# # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
# f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
# # assigning a graph to each ax
# sns.boxplot(L_dc_df["L_sonic1"], ax=ax_box)
# # sns.histplot(data=L_dc_df, x="L_sonic1", weights=np.ones(len(L_dc_df)) / len(L_dc_df), binrange = (-1000,1000), ax=ax_hist)
# sns.histplot(data=L_dc_df, x="L_sonic1", bins = 500, weights=np.ones(len(L_dc_df)) / len(L_dc_df), ax=ax_hist)
# # Remove x axis name for the boxplot
# # ax_box.set(xlabel='')
# ax_hist.set_xlim(-100,100)
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# #%%
# lower_lim = -250
# upper_lim = 250
# plt.figure()
# sns.boxplot(L_dc_df["L_sonic1"])
# plt.ylim(lower_lim,upper_lim)
# plt.title('L_sonic1 d.c.')

# plt.figure()
# sns.boxplot(L_coare_wind_despiked["L_sonic1"])
# plt.ylim(lower_lim,upper_lim)
# plt.title('L_sonic1 coare')

# plt.figure()
# sns.boxplot(L_dc_df["L_sonic2"])
# plt.ylim(lower_lim,upper_lim)
# plt.title('L_sonic2 d.c.')

# plt.figure()
# sns.boxplot(L_coare_wind_despiked["L_sonic2"])
# plt.ylim(lower_lim,upper_lim)
# plt.title('L_sonic2 coare')

# plt.figure()
# sns.boxplot(L_dc_df["L_sonic3"])
# plt.ylim(lower_lim,upper_lim)
# plt.title('L_sonic3 d.c.')

# plt.figure()
# sns.boxplot(L_coare_wind_despiked["L_sonic3"])
# plt.ylim(lower_lim,upper_lim)
# plt.title('L_sonic3 coare')

#%%
data_a = L_I_coare
plt.figure()
plt.hist(data_a, bins = 50, range = (-1000,1000),  weights=np.ones(len(data_a)) / len(data_a), color = 'black',edgecolor = 'white', label = 'coare')
plt.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('Histogram of L CAORE: Level I')

data_b = L_I_dc
plt.figure()
plt.hist(data_b, bins = 50, range = (-1000,1000),  weights=np.ones(len(data_b)) / len(data_b), color = 'gray',edgecolor = 'white', label = 'dc')
plt.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('Histogram of L d.c.: Level I')

data_a = L_II_coare
plt.figure()
plt.hist(data_a, bins = 50, range = (-1000,1000),  weights=np.ones(len(data_a)) / len(data_a), color = 'black',edgecolor = 'white', label = 'coare')
plt.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('Histogram of L CAORE: Level II')

data_b = L_II_dc
plt.figure()
plt.hist(data_b, bins = 50, range = (-1000,1000),  weights=np.ones(len(data_b)) / len(data_b), color = 'gray',edgecolor = 'white', label = 'dc')
plt.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('Histogram of L d.c.: Level II')

#%%

