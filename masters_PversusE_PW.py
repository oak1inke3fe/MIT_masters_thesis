#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:36:06 2023

@author: oaklin keefe


This file is used to analyze the production, dissipation, buoyancy, and wave-coherent pw terms of the TKE budget equation.
There are options to perform this analysis with on-shore wind conditions as well as off-shore (see commented out sections of the code)

Input file location:
    /code_pipeline/Level2/
INPUT files:
    prodTerm_combinedAnalysis.csv
    epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv
    buoy_terms_combinedAnalysis.csv
    z_airSide_allSpring.csv
    z_airSide_allFall.csv
    ZoverL_combinedAnalysis.csv
    usr_combinedAnalysis.csv
    
 
Output file location:
    /code_pipeline/Level2/   
OUTPUT files:
    Only figures:
        

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import binsreg
import seaborn as sns

print('done with imports')
#%%
# file_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'

plot_savePath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)

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
# eps_df[eps_df['eps_sonic1'] > 1] = np.nan

buoy_df = pd.read_csv(file_path+'buoy_terms_combinedAnalysis.csv')
buoy_df = buoy_df.drop(['Unnamed: 0'], axis=1)

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
#%% Get rid of bad wind directions first
all_windDirs = True
onshore = False
offshore = False

# plt.figure()
# plt.scatter(windDir_df.index, windDir_df['alpha_s1'])
# plt.scatter(zL_df.index, zL_df['z/L I dc'])
# plt.yscale('log')

windDir_index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, windDir_index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],windDir_index_array)

windDir_df[mask_goodWindDir] = np.nan

prod_df[mask_goodWindDir] = np.nan
eps_df[mask_goodWindDir] = np.nan
buoy_df[mask_goodWindDir] = np.nan
z_df[mask_goodWindDir] = np.nan
zL_df[mask_goodWindDir] = np.nan
pw_df[mask_goodWindDir] = np.nan
sonic1_df[mask_goodWindDir] = np.nan
windSpeed_df[mask_goodWindDir] = np.nan

print('done with setting up  good wind direction only dataframes')

# plt.figure()
# plt.scatter(windDir_df.index, windDir_df['alpha_s1'])
# plt.scatter(zL_df.index, zL_df['z/L I dc'])

#%% set to near-neutral stability conditions

plt.figure()
# plt.scatter(zL_df.index, zL_df['zL_II_dc'], label = 'z/L level II')
plt.scatter(zL_df.index, zL_df['zL_I_dc'], label = 'z/L level I', color = 'gray')
plt.plot(prod_df['prod_I']*10, label = 'prod LI*10', color = 'black')
plt.legend()
plt.title('z/L and Prod BEFORE neutral mask')

zL_index_array = np.arange(len(zL_df))
zL_df['new_index_arr'] = np.where((np.abs(zL_df['zL_I_dc'])<=0.5)&(np.abs(zL_df['zL_II_dc'])<=0.5), np.nan, zL_index_array)
mask_neutral_zL = np.isin(zL_df['new_index_arr'],zL_index_array)

zL_df[mask_neutral_zL] = np.nan

eps_df[mask_neutral_zL] = np.nan
prod_df[mask_neutral_zL] = np.nan
buoy_df[mask_neutral_zL] = np.nan
z_df[mask_neutral_zL] = np.nan
windDir_df[mask_neutral_zL] = np.nan
pw_df[mask_neutral_zL] = np.nan
sonic1_df[mask_neutral_zL] = np.nan
windSpeed_df[mask_neutral_zL] = np.nan

plt.figure()
# plt.scatter(zL_df.index, zL_df['zL_II_dc'], label = 'z/L level II')
plt.scatter(zL_df.index, zL_df['zL_I_dc'], s=5, label = 'z/L level I', color = 'gray')
plt.plot(prod_df['prod_I']*10, label = 'prod LI*10', color = 'black')
plt.legend()
plt.title('z/L and Prod AFTER neutral mask')

# plt.figure()
# plt.scatter(zL_df.index, zL_df['z/L II dc'], label = 'level II')
# plt.scatter(zL_df.index, zL_df['z/L I dc'], label = 'level I')
# plt.legend()



print('done with setting up near-neutral dataframes')

#%% mask to s1 Ubar >=10m/s for when dissipaiton deficit and pw have larger magnitudes

plt.figure()
# plt.scatter(zL_df.index, zL_df['zL_II_dc'], label = 'z/L level II')
plt.scatter(sonic1_df.index, sonic1_df['Ubar'], label = 'Ubar s1', color = 'gray')
plt.legend()
plt.title('Ubar BEFORE 8m/s restriction mask')
plt.ylim(0,18)

Ubar_index_array = np.arange(len(sonic1_df))
sonic1_df['new_index_arr'] = np.where((sonic1_df['Ubar']>=10), np.nan, Ubar_index_array)
mask_confidentPW_byUbarRestriction = np.isin(sonic1_df['new_index_arr'],Ubar_index_array)

sonic1_df[mask_confidentPW_byUbarRestriction]=np.nan

zL_df[mask_confidentPW_byUbarRestriction] = np.nan
eps_df[mask_confidentPW_byUbarRestriction] = np.nan
prod_df[mask_confidentPW_byUbarRestriction] = np.nan
buoy_df[mask_confidentPW_byUbarRestriction] = np.nan
z_df[mask_confidentPW_byUbarRestriction] = np.nan
windDir_df[mask_confidentPW_byUbarRestriction] = np.nan
pw_df[mask_confidentPW_byUbarRestriction] = np.nan
windSpeed_df[mask_confidentPW_byUbarRestriction] = np.nan

plt.figure()
# plt.scatter(zL_df.index, zL_df['zL_II_dc'], label = 'z/L level II')
plt.scatter(sonic1_df.index, sonic1_df['Ubar'], label = 'Ubar s1', color = 'gray')
plt.legend()
plt.title('Ubar AFTER 10m/s restriction mask')
plt.ylim(0,18)
print('done with setting up Ubar restrictions to confident pw calcs')

#%% Offshore setting
# all_windDirs = False
# onshore = False
# offshore = True

# windDir_df['offshore_index_arr'] = np.arange(len(windDir_df))
# windDir_df['new_offshore_index_arr'] = np.where((windDir_df['alpha_s4'] >= 270)&(windDir_df['alpha_s4'] <= 359), windDir_df['offshore_index_arr'], np.nan)

# mask_offshoreWindDir = np.isin(windDir_df['new_offshore_index_arr'],windDir_df['offshore_index_arr'])
# windDir_df = windDir_df[mask_offshoreWindDir]

# zL_df = zL_df[mask_offshoreWindDir]
# eps_df = eps_df[mask_offshoreWindDir]
# eps_MAD_df = eps_MAD_df[mask_offshoreWindDir]
# eps_newMAD_df = eps_newMAD_df[mask_offshoreWindDir]
# prod_df = prod_df[mask_offshoreWindDir]
# buoy_df = buoy_df[mask_offshoreWindDir] 
# z_df = z_df[mask_offshoreWindDir]
# pw_df = pw_df[mask_offshoreWindDir]

#%% On-shore setting
# all_windDirs = False
# onshore = True
# offshore = False

# windDir_df['onshore_index_arr'] = np.arange(len(windDir_df))
# windDir_df['new_onshore_index_arr'] = np.where((windDir_df['alpha_s4'] >= 197)&(windDir_df['alpha_s4'] <= 269), windDir_df['onshore_index_arr'], np.nan)

# mask_onshoreWindDir = np.isin(windDir_df['new_onshore_index_arr'],windDir_df['onshore_index_arr'])
# windDir_df = windDir_df[mask_onshoreWindDir]

# zL_df = zL_df[mask_onshoreWindDir]
# eps_df = eps_df[mask_onshoreWindDir]
# eps_MAD_df = eps_MAD_df[mask_onshoreWindDir]
# eps_newMAD_df = eps_newMAD_df[mask_onshoreWindDir]
# prod_df = prod_df[mask_onshoreWindDir]
# buoy_df = buoy_df[mask_onshoreWindDir] 
# z_df = z_df[mask_onshoreWindDir]
# pw_df = pw_df[mask_onshoreWindDir]

#%%

eps_LIII = (eps_df['eps_sonic3']+eps_df['eps_sonic4'])/2
eps_LII = (eps_df['eps_sonic2']+eps_df['eps_sonic3'])/2
eps_LI = (eps_df['eps_sonic1']+eps_df['eps_sonic2'])/2
eps_df['eps_LIII'] = eps_LIII
eps_df['eps_LII'] = eps_LII
eps_df['eps_LI'] = eps_LI



buoy_LIII = buoy_df['buoy_III']
buoy_LII = buoy_df['buoy_II']
buoy_LI = buoy_df['buoy_I']



print('done with getting epsilon with LI,II,II and dfs and buoyancy dfs')

#%% Production minus dissipation (dissipation deficit if result is +)
p_minus_diss_I =  np.array(np.array(prod_df['prod_I'])-np.array(eps_df['eps_LI']))
p_minus_diss_II =  np.array(np.array(prod_df['prod_II'])-np.array(eps_df['eps_LII']))
p_minus_diss_III =  np.array(np.array(prod_df['prod_III'])-np.array(eps_df['eps_LIII']))
print('done doing production minus dissipation')

#%%
# figure titles based on wind directions
if all_windDirs == True:
    pw_timeseries_title = 'Neutral Conditions: P-$T_{\widetilde{pw}}$ vs. $T_{\widetilde{pw}}$ '
    pw_title = 'Neutral Conditions: P-$\epsilon$ vs. $T_{\widetilde{pw}}$ '
    prod_eps_title = 'Neutral Conditions: P vs. $\epsilon$ '
    prodBuoy_eps_title = 'Neutral P+B vs. $\epsilon$ '
elif onshore == True:
    pw_timeseries_title = 'Onshore Neutral Conditions: P-$T_{\widetilde{pw}}$ vs. $T_{\widetilde{pw}}$ '
    pw_title = 'Onshore Winds, Neutral Conditions: P-$\epsilon$ vs. $T_{\widetilde{pw}}$ '
    prod_eps_title = 'Onshore Winds, Neutral Conditions: P vs. $\epsilon$ '
    prodBuoy_eps_title = 'Onshore Winds, Neutral P+B vs. $\epsilon$ '
elif offshore == True:
    pw_timeseries_title = 'Offshore Neutral Conditions: P-$T_{\widetilde{pw}}$ vs. $T_{\widetilde{pw}}$ '
    pw_title = 'Offshore Winds, Neutral Conditions: P-$\epsilon$ vs. $T_{\widetilde{pw}}$ '
    prod_eps_title = 'Offshore Winds, Neutral Conditions: P vs. $\epsilon$ '
    prodBuoy_eps_title = 'Offshore Winds, Neutral P+B vs. $\epsilon$ '

#%%
#make some plots!

fig = plt.figure()
plt.scatter(np.arange(len(p_minus_diss_I)), p_minus_diss_I, color = 'dodgerblue', s=5, label = '$P-\epsilon$')
plt.plot(np.arange(len(p_minus_diss_I)), p_minus_diss_I, color = 'dodgerblue')
plt.scatter(np.arange(len(p_minus_diss_I)), np.array(pw_df['d_dz_pw_theory_I']), color = 'navy', s=5, label = '$T_{\widetilde{pw}}$')
plt.plot(np.arange(len(p_minus_diss_I)), np.array(pw_df['d_dz_pw_theory_I']), color = 'navy')
# plt.hlines(y=0, xmin=0,xmax=4395,color='k')
plt.legend()
plt.ylabel('$P- \epsilon, \;, T_{\widetilde{pw}} \quad [m^2/s^3]$')
plt.ylim(-0.05,0.1)
# plt.xlim(1600,1950)
plt.title("$P-\epsilon$ and $T_{\widetilde{pw}}$ \n Level I")

#%%

fig = plt.figure()
plt.scatter(np.arange(len(p_minus_diss_I)), p_minus_diss_I-np.array(pw_df['d_dz_pw_theory_I']), color = 'navy', s=5, label = '$P-\epsilon-T_{\widetilde{pw}}$')
plt.plot(np.arange(len(p_minus_diss_I)), p_minus_diss_I-np.array(pw_df['d_dz_pw_theory_I']), color = 'navy')
# plt.hlines(y=0, xmin=0,xmax=4395,color='k')
plt.legend()
plt.ylabel('$P- \epsilon -T_{\widetilde{pw}} \quad [m^2/s^3]$')
plt.ylim(-0.1,0.1)
# plt.xlim(1600,1950)
plt.title("$P-\epsilon-T_{\widetilde{pw}}$ \n Level I")
# plt.savefig(plot_savePath + "timeseries_MAYstorm_neutralPvEps_pwUbarRestriction.png", dpi = 300)
# plt.savefig(plot_savePath + "timeseries_MAYstorm_neutralPvEps_pwUbarRestriction.pdf")

#%%
fig = plt.figure()
plt.plot(p_minus_diss_II, color = 'darkorange', label = 'dissipation deficit')
plt.plot(np.array(pw_df['d_dz_pw_theory_II']), color = 'red', label = '$T_{\widetilde{pw}}$')
plt.legend()
plt.ylim(-0.1,0.1)
# plt.xlim(1600,1950)
plt.title("$P-\epsilon$ and $T_{\widetilde{pw}}$ \n Level II")


fig = plt.figure(figsize=(15,8))
plt.plot(p_minus_diss_II, color = 'darkorange', label = 'L II')
plt.plot(p_minus_diss_I, color = 'dodgerblue', label = 'L I')
plt.legend()
# plt.hlines(y=0, xmin=0,xmax=4395,color='k')
# plt.xlim(700,860)
plt.ylim(-0.2,0.2)
plt.xlabel('time')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')


plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis \n for $\overline{u}$ sonic1 >=10m/s')
plt.savefig(plot_savePath + "timeseries_neutralPvEps_pwUbarRestriction.png", dpi = 300)
plt.savefig(plot_savePath + "timeseries_neutralPvEps_pwUbarRestriction.pdf")
#%%
fig = plt.figure(figsize=(7,2))
# fig = plt.figure()
plt.plot(p_minus_diss_II, color = 'darkorange', label = 'L II')
plt.plot(p_minus_diss_I, color = 'dodgerblue', label = 'L I')
plt.legend()
plt.hlines(y=0, xmin=0,xmax=4395,color='k')
plt.xlim(1550,2050)
# plt.ylim(-0.1,0.20)
plt.ylim(-0.05,0.1)
plt.xlabel('May Storm')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')
plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis')

plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis \n for $\overline{u}$ sonic1 $\geq$ 10m/s')
plt.savefig(plot_savePath + "timeseries_MAYstorm_neutralPvEps_pwUbarRestriction.png", dpi = 300)
plt.savefig(plot_savePath + "timeseries_MAYstorm_neutralPvEps_pwUbarRestriction.pdf")


fig = plt.figure(figsize=(7,2))
plt.plot(p_minus_diss_II, color = 'darkorange', label = 'L II')
plt.plot(p_minus_diss_I, color = 'dodgerblue', label = 'L I')
plt.legend()
plt.hlines(y=0, xmin=0,xmax=8000,color='k')
plt.xlim(4550,4850)
plt.ylim(-0.1,0.2)
plt.xlabel('October Storm')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')
plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis')

plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis \n for $\overline{u}$ sonic1 $\geq$ 10m/s')
plt.savefig(plot_savePath + "timeseries_OCTstorm_neutralPvEps_pwUbarRestriction.png", dpi = 300)
plt.savefig(plot_savePath + "timeseries_OCTstorm_neutralPvEps_pwUbarRestriction.pdf")
#%%
plt.figure()
plt.scatter(zL_df['zL_II_dc'], p_minus_diss_II, color = 'orange', edgecolor = 'red', label = 'L II')
plt.scatter(zL_df['zL_I_dc'], p_minus_diss_I, color = 'dodgerblue', edgecolor = 'navy', label = 'L I')
plt.legend()
plt.xlim(-1,1)
plt.yscale('log')
plt.title("$P - \epsilon$ (Dissipation Deficit)")
plt.xlabel('z/L')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')

#%%
plt.figure()
plt.hist(p_minus_diss_I,40,color = 'dodgerblue', edgecolor = 'white', label = 'L I; bins=40')
plt.hist(p_minus_diss_II,37,color = 'darkorange', edgecolor = 'white', label = 'L II, bins = 37')
plt.xlabel('$P-\epsilon$')
plt.ylabel('occurance')
plt.title('Histogram of $P-\epsilon$')
# plt.xlim(-0.01,0.05)
# plt.ylim(0,50)
plt.vlines(x=0, ymin=0, ymax=400, color = 'k')
plt.legend()


#%%

prodPLUSbuoy= pd.DataFrame()
prodPLUSbuoy['P+B LI'] = np.array(prod_df['prod_I'])+np.array(buoy_LI)
prodPLUSbuoy['P+B LII'] = np.array(prod_df['prod_II'])+np.array(buoy_LII)
prodPLUSbuoy['P+B LIII'] = np.array(prod_df['prod_III'])+np.array(buoy_LIII)

PplusB_minus_eps_df = pd.DataFrame()
PplusB_minus_eps_df['LI'] = np.array(prodPLUSbuoy['P+B LI'])-np.array(eps_df['eps_LI'])
PplusB_minus_eps_df['LII'] = np.array(prodPLUSbuoy['P+B LII'])-np.array(eps_df['eps_LII'])

print('done with combining production and buoyancy')


#%%
pw_percentage_of_deficit = pw_df['d_dz_pw_theory_I']/PplusB_minus_eps_df['LI']*100
pw_percentage_of_deficit_mean = np.nanmean(pw_percentage_of_deficit)
print('mean percent = '+ str(pw_percentage_of_deficit_mean))
pw_percentage_of_deficit_median = np.nanmedian(pw_percentage_of_deficit)
print('meadin percent = '+ str(pw_percentage_of_deficit_median))
from scipy import stats
pw_percentage_of_deficit_mode = stats.mode(pw_percentage_of_deficit,nan_policy='omit')
print('mode percent = '+ str(pw_percentage_of_deficit_mode))

plt.figure()
plt.scatter(pw_percentage_of_deficit,pw_percentage_of_deficit)
# plt.ylim(0,100)
#%% P + B versus Eps (NO PW added, but PW Ubar restriction applied)
fig = plt.figure(figsize=(6,6))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-2,1], [10**-3,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-3,0.1], [10**-2,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(prodPLUSbuoy['P+B LII'], np.array(eps_df['eps_LII']), color = 'darkorange', edgecolor = 'red', label = 'level II')
plt.scatter(prodPLUSbuoy['P+B LI'], np.array(eps_df['eps_LI']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# plt.scatter(PplusB_minus_eps_df['LII'], np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P +B $ [$m^2s^{-3}$]', fontsize=12)
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]', fontsize=12)
plt.legend()
# plt.tight_layout()

plt.ylim(10**-2,1)
plt.xlim(10**-2,1)
plt.title('Neutral Conditions: $P+B$ vs. $\epsilon$ \n for $\overline{u}$ $\geq$ 10m/s', fontsize=16)
plt.savefig(plot_savePath + "scatter_Eps_vs_ProdPlusBuoy_pwUbarRestriction.png",dpi=300)
plt.savefig(plot_savePath + "scatter_Eps_vs_ProdPlusBuoy_pwUbarRestriction.pdf")
# plt.axis('square')

print('done plotting P+B vs. Eps-PW simple diss (new)')
#%% P + B versus Eps plus wave-coherent pw
fig = plt.figure(figsize=(6,6))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-1,1], [10**-2,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-3,0.1], [10**-2,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(prodPLUSbuoy['P+B LII'], (np.array(eps_df['eps_LII'])+np.array(pw_df['d_dz_pw_theory_II'])), color = 'darkorange', edgecolor = 'red', label = 'level II')
plt.scatter(prodPLUSbuoy['P+B LI'], (np.array(eps_df['eps_LI'])+np.array(pw_df['d_dz_pw_theory_I'])), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# plt.scatter(PplusB_minus_eps_df['LII'], np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P +B $ [$m^2s^{-3}$]',fontsize=12)
plt.ylabel('$\epsilon+ T_{\widetilde{pw}}$ [$m^2s^{-3}$]', fontsize=12)
plt.legend()
# plt.tight_layout()

plt.ylim(10**-2,1)
plt.xlim(10**-2,1)
plt.title('Neutral Conditions: $P+B$ vs. $\epsilon+T_{\widetilde{pw}}$ \n for $\overline{u}$ $\geq$ 10m/s', fontsize=16)
plt.savefig(plot_savePath + "scatter_EpsAndPW_vs_ProdPlusBuoy_pwUbarRestriction.png",dpi=300)
plt.savefig(plot_savePath + "scatter_EpsAndPW_vs_ProdPlusBuoy_pwUbarRestriction.pdf")
# plt.axis('square')

print('done plotting P+B vs. Eps-PW simple diss (new)')
#%% zoom out

fig = plt.figure(figsize = (6,6))
# fig= plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prodPLUSbuoy['P+B LIII'], eps_df['eps_LIII'], color = 'seagreen', edgecolor = 'darkgreen', label = 'level III')
plt.scatter(prodPLUSbuoy['P+B LII'], (np.array(eps_df['eps_LII'])+np.array(pw_df['d_dz_pw_theory_II'])), color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prodPLUSbuoy['P+B LI'], (np.array(eps_df['eps_LI'])+np.array(pw_df['d_dz_pw_theory_I'])), color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
# plt.xscale('log')
# plt.yscale('log')
plt.legend(loc = 'upper left')
plt.xlabel('$P\; +\; B$ [$m^2s^{-3}$]', fontsize=12)
plt.ylabel('$\epsilon \; + \; T_{\widetilde{pw}} $ [$m^2s^{-3}$]', fontsize=12)
plt.title('Neutral Conditions: P+B vs. $\epsilon + T_{\widetilde{pw}}$ \n for $\overline{u}$ $\geq$ 10m/s', fontsize=16)
ax = plt.gca() 
# plt.text(.05, .8, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes) #this is 0.359
# plt.text(.05, .75, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)  #this is 0.709
plt.axis('equal')
plt.tight_layout()

plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig(plot_savePath + "scatterZoomOut_EpsAndPW_vs_ProdPlusBuoy_pwUbarRestriction.png",dpi=300)
plt.savefig(plot_savePath + "scatterZoomOut_EpsAndPW_vs_ProdPlusBuoy_pwUbarRestriction.pdf")

#%% dissipation deficit (with buoyancy) versus wave-coherent pw
fig = plt.figure(figsize=(6,6))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-4,1], [10**-5,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-5,0.1], [10**-4,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(PplusB_minus_eps_df['LII'], np.array(pw_df['d_dz_pw_theory_II']), color = 'darkorange', edgecolor = 'red', label = 'level II')
plt.scatter(PplusB_minus_eps_df['LI'], np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# plt.scatter(PplusB_minus_eps_df['LII'], np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P +B - \epsilon$ [$m^2s^{-3}$]', fontsize=12)
plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]', fontsize=12)
plt.legend()
plt.tight_layout()
# plt.savefig(plot_savePath + "scatter_PW_vs_dissDeficit.png",dpi=300)
# plt.savefig(plot_savePath + "scatter_PW_vs_dissDeficit.pdf")
plt.title('Neutral Conditions: $P+B-\epsilon$ vs. $T_{\widetilde{pw}}$ \n for $\overline{u}$ $\geq$ 10m/s',fontsize=16)
plt.savefig(plot_savePath + "scatter_PW_vs_dissDeficit_pwUbarRestriction.png",dpi=300)
plt.savefig(plot_savePath + "scatter_PW_vs_dissDeficit_pwUbarRestriction.pdf")
# plt.axis('square')
# plt.xlim(-0.0001,0.01)
# plt.ylim(-0.01,0.01)


print('done with plotting dissipation deficit with buoyancy versus wave-coherent pw')





#%% NEUTRAL CONDITIONS

# scatterplot of Production versus Dissipation SPRING
fig = plt.figure(figsize=(6,6))
# fig = plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prod_df['prod_III'][:break_index+1], eps_df['eps_LIII'][:break_index+1], color = 'seagreen', edgecolor = 'darkgreen', label = 'level III')
plt.scatter(prod_df['prod_II'][:break_index+1], eps_df['eps_LII'][:break_index+1], color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prod_df['prod_I'][:break_index+1], eps_df['eps_LI'][:break_index+1], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
# plt.xlim(10**-5,1)
# plt.ylim(10**-5,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
# plt.title("SPRING: Neutral Conditions: $P$ vs. $\epsilon$")
plt.title("SPRING: Neutral Conditions: $P$ vs. $\epsilon$ \n for $\overline{u}$ $\geq$ 10m/s")
plt.legend(loc = 'lower right')
ax = plt.gca() 
# plt.text(.05, .9, "Pearson's r L III ={:.3f}".format(r_spring_III), transform=ax.transAxes)
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_spring_II), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_spring_I), transform=ax.transAxes)
plt.axis('equal')
plt.tight_layout()

plt.savefig(plot_savePath + "scatterplot_PvEps_spring_pwUbarRestriction.png",dpi=300)
plt.savefig(plot_savePath + "scatterplot_PvEps_spring_pwUbarRestriction.pdf")
print('done plotting P vs. Eps SPRING')



# scatterplot of Production versus Dissipation FALL
fig = plt.figure(figsize=(6,6))
# fig = plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prod_df['prod_III'][break_index+1:], eps_df['eps_LIII'][break_index+1:], color = 'seagreen', edgecolor = 'darkgreen', label = 'level III')
plt.scatter(prod_df['prod_II'][break_index+1:], eps_df['eps_LII'][break_index+1:], color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prod_df['prod_I'][break_index+1:], eps_df['eps_LI'][break_index+1:], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
# plt.xlim(10**-5,1)
# plt.ylim(10**-5,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
# plt.title("FALL: Neutral Conditions: $P$ vs. $\epsilon$")
plt.title("FALL: Neutral Conditions: $P$ vs. $\epsilon$ \n for $\overline{u}$ $\geq$ 10m/s")
plt.legend(loc = 'lower right')
ax = plt.gca() 
# plt.text(.05, .9, "Pearson's r L III ={:.3f}".format(r_fall_III), transform=ax.transAxes)
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_fall_II), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_fall_I), transform=ax.transAxes)
# ax.set_xlim(10**-5,1)
# ax.set_ylim(10**-5,1)
plt.axis('equal')
plt.tight_layout()
# plt.savefig(plot_savePath + "scatterplot_PvEps_fall.png",dpi=300)
# plt.savefig(plot_savePath + "scatterplot_PvEps_fall.pdf")
plt.savefig(plot_savePath + "scatterplot_PvEps_fall_pwUbarRestriction.png",dpi=300)
plt.savefig(plot_savePath + "scatterplot_PvEps_fall_pwUbarRestriction.pdf")
print('done plotting P vs. Eps FALL')


#%%

fig = plt.figure(figsize = (4,4))
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1",linewidth=3) #scale 1-to-1 line
plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10',linewidth=3) #scale line by power of 10
plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--',linewidth=3) #scale line by power of 10
# plt.scatter(prod_df['prod_III'], eps_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
plt.scatter(p_minus_diss_I, p_minus_diss_II, color = 'cyan', edgecolor = 'navy')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Level I $P-\epsilon$ [$m^2s^{-3}$]')
plt.ylabel('Level II $P-\epsilon$ [$m^2s^{-3}$]')
plt.title(r'$P-\epsilon$')
plt.legend(loc = 'upper left')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_str), transform=ax.transAxes)
plt.axis('square')

print('done plotting P vs. Eps simple diss (new)')


#%%
# # scatterplot of spring conditions
# fig = plt.figure(figsize = (6,6))
# # fig= plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# # plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# # plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# # plt.scatter(prodPLUSbuoy['P+B LIII'][:break_index+1], eps_df['eps_LIII'][:break_index+1], color = 'seagreen', edgecolor = 'darkgreen',label = 'level III')
# plt.scatter(prodPLUSbuoy['P+B LII'][:break_index+1], eps_df['eps_LII'][:break_index+1], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(prodPLUSbuoy['P+B LI'][:break_index+1], eps_df['eps_LI'][:break_index+1], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc = 'upper left')
# plt.xlabel('$P\; +\; B$ [$m^2s^{-3}$]')
# plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
# plt.title('SPRING: P+B vs. $\epsilon$')
# ax = plt.gca() 
# # plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
# # plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
# plt.axis('equal')
# plt.tight_layout()
# plt.savefig(plot_savePath + "scatterplot_PplusBvEps_Spring.png",dpi=300)
# plt.savefig(plot_savePath + "scatterplot_PplusBvEps_Spring.pdf")

# # scatterplot of fall conditions
# fig = plt.figure(figsize = (6,6))
# # fig= plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# # plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# # plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# # plt.scatter(prodPLUSbuoy['P+B LIII'][break_index+1:], eps_df['eps_LIII'][break_index+1:], color = 'seagreen', edgecolor = 'darkgreen',label = 'level III')
# plt.scatter(prodPLUSbuoy['P+B LII'][break_index+1:], eps_df['eps_LII'][break_index+1:], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(prodPLUSbuoy['P+B LI'][break_index+1:], eps_df['eps_LI'][break_index+1:], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc = 'upper left')
# plt.xlabel('$P\; +\; B$ [$m^2s^{-3}$]')
# plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
# plt.title('FALL: P+B vs. $\epsilon$')
# ax = plt.gca() 
# # plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
# # plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
# plt.axis('equal')
# plt.tight_layout()
# plt.savefig(plot_savePath + "scatterplot_PplusBvEps_Fall.png",dpi=300)
# plt.savefig(plot_savePath + "scatterplot_PplusBvEps_Fall.pdf")

#%%


Level_I_df = pd.DataFrame()
Level_I_df['P+B-Eps-PW'] = np.array(PplusB_minus_eps_df['LI']-pw_df['d_dz_pw_theory_I'])
Level_I_df['Ubar'] = np.array(windSpeed_df['Ubar_LI'])
Level_I_df['PW'] = np.array(pw_df['d_dz_pw_theory_I'])
Level_I_df['P+B-Eps'] = np.array(PplusB_minus_eps_df['LI'])
Level_I_df['windDir'] = np.array(windDir_df['alpha_s2'])


Level_II_df = pd.DataFrame()
Level_II_df['P+B-Eps-PW'] = np.array(PplusB_minus_eps_df['LII']-pw_df['d_dz_pw_theory_II'])
Level_II_df['Ubar'] = np.array(windSpeed_df['Ubar_LII'])
Level_II_df['PW'] = np.array(pw_df['d_dz_pw_theory_I'])
Level_II_df['P+B-Eps'] = np.array(PplusB_minus_eps_df['LII'])
Level_II_df['windDir'] = np.array(windDir_df['alpha_s3'])
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
df_binEstimate_LI_UbarVsDeficit = binscatter(x='Ubar', y='P+B-Eps-PW',w=['windDir'], data=Level_I_df, ci=(3,3),randcut=1)
df_binEstimate_LI_UbarVsDeficit = binscatter(x='Ubar', y='P+B-Eps-PW', data=Level_I_df, ci=(3,3),randcut=1)

df_binEstimate_LII_UbarVsDeficit = binscatter(x='Ubar', y='P+B-Eps-PW',w=['windDir'], data=Level_II_df, ci=(3,3),randcut=1)
df_binEstimate_LII_UbarVsDeficit = binscatter(x='Ubar', y='P+B-Eps-PW', data=Level_II_df, ci=(3,3),randcut=1)


df_binEstimate_LI_UbarVsDeficit_noPW = binscatter(x='Ubar', y='P+B-Eps',w=['windDir'], data=Level_I_df, ci=(3,3),randcut=1)
df_binEstimate_LI_UbarVsDeficit_noPW = binscatter(x='Ubar', y='P+B-Eps', data=Level_I_df, ci=(3,3),randcut=1)

df_binEstimate_LII_UbarVsDeficit_noPW = binscatter(x='Ubar', y='P+B-Eps',w=['windDir'], data=Level_II_df, ci=(3,3),randcut=1)
df_binEstimate_LII_UbarVsDeficit_noPW = binscatter(x='Ubar', y='P+B-Eps', data=Level_II_df, ci=(3,3),randcut=1)

df_binEstimate_LI_UbarVsPW = binscatter(x='Ubar', y='PW',w=['windDir'], data=Level_I_df, ci=(3,3),randcut=1)
df_binEstimate_LI_UbarVsPW = binscatter(x='Ubar', y='PW', data=Level_I_df, ci=(3,3),randcut=1)
#%%
# scatterplot of Production plus Buoyancy versus Dissipation minus PW versus wind speed: combined analysis
fig = plt.figure(figsize = (6,6))
sns.scatterplot(x='Ubar', y='P+B-Eps-PW', data=df_binEstimate_LII_UbarVsDeficit, color = 'darkorange', label = "binned LII")
plt.errorbar('Ubar', 'P+B-Eps-PW', yerr='ci', data=df_binEstimate_LII_UbarVsDeficit, color = 'coral', ls='', lw=2, alpha=0.2, label = 'LII errorbar')
sns.scatterplot(x='Ubar', y='P+B-Eps-PW', data=df_binEstimate_LI_UbarVsDeficit, color = 'dodgerblue', label = "binned LI")
plt.errorbar('Ubar', 'P+B-Eps-PW', yerr='ci', data=df_binEstimate_LI_UbarVsDeficit, color = 'navy', ls='', lw=2, alpha=0.2, label = 'LI errorbar')
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line

# plt.xscale('log')
# plt.yscale('log')
plt.legend(loc = 'upper left')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$P + B - (\epsilon + T_{\widetilde{pw}})$ [$m^2s^{-3}$]')
plt.title('$P+B-(\epsilon+T_{\widetilde{pw}})$ vs. Wind Speed ($\overline{u}$) \n (Binned)')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
# plt.axis('equal')
plt.hlines(y=0,xmin=2,xmax=16,color='k',linestyles='--')
plt.xlim(10,16)
plt.ylim(-0.2,0.2)
plt.tight_layout()
plt.savefig(plot_savePath + "scatterplotBIN_PWrestriction_UbarVSPplusBminusEpsMinusPW_combinedAnalysis.png",dpi=300)
plt.savefig(plot_savePath + "scatterplotBIN_PWrestriction_UbarVSPplusBvEpsMinusPW_combinedAnalysis.pdf")

#%%
# scatterplot of Production plus Buoyancy versus Dissipation minus PW versus wind speed AND PW vs Wind speed: combined analysis
fig = plt.figure(figsize = (6,6))
sns.scatterplot(x='Ubar', y='P+B-Eps-PW', data=df_binEstimate_LII_UbarVsDeficit, color = 'darkorange', label = "binned LII")
plt.errorbar('Ubar', 'P+B-Eps-PW', yerr='ci', data=df_binEstimate_LII_UbarVsDeficit, color = 'coral', ls='', lw=2, alpha=0.2, label = 'LII errorbar')
sns.scatterplot(x='Ubar', y='P+B-Eps-PW', data=df_binEstimate_LI_UbarVsDeficit, color = 'dodgerblue', label = "binned LI")
plt.errorbar('Ubar', 'P+B-Eps-PW', yerr='ci', data=df_binEstimate_LI_UbarVsDeficit, color = 'navy', ls='', lw=2, alpha=0.2, label = 'LI errorbar')
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
sns.scatterplot(x='Ubar', y='PW', data=df_binEstimate_LI_UbarVsPW, color = 'dimgray', label = "binned PW boom-1")
plt.errorbar('Ubar', 'PW', yerr='ci', data=df_binEstimate_LI_UbarVsPW, color = 'dimgray', ls='', lw=2, alpha=0.2, label = 'PW errorbar')


# plt.xscale('log')
# plt.yscale('log')
plt.legend(loc = 'upper left')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$P + B - (\epsilon + T_{\widetilde{pw}})$ [$m^2s^{-3}$]')
plt.title('$P+B-(\epsilon+T_{\widetilde{pw}})$ vs. Wind Speed ($\overline{u}$) \n (Binned)')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
# plt.axis('equal')
plt.hlines(y=0,xmin=2,xmax=16,color='k',linestyles='--')
plt.xlim(10,16)
plt.ylim(-0.2,0.2)
plt.tight_layout()
plt.savefig(plot_savePath + "scatterplotBIN_PWrestriction_UbarVSPplusBminusEpsMinusPW_withPWvsWindSpeed_combinedAnalysis.png",dpi=300)
plt.savefig(plot_savePath + "scatterplotBIN_PWrestriction_UbarVSPplusBminusEpsMinusPW_withPWvsWindSpeed_combinedAnalysis.pdf")

#%%
# scatterplot of Production plus Buoyancy versus Dissipation (no PW) versus wind speed AND PW vs Wind speed: combined analysis
fig = plt.figure(figsize = (6,6))
sns.scatterplot(x='Ubar', y='P+B-Eps', data=df_binEstimate_LII_UbarVsDeficit_noPW, color = 'darkorange', label = "binned LII")
plt.errorbar('Ubar', 'P+B-Eps', yerr='ci', data=df_binEstimate_LII_UbarVsDeficit_noPW, color = 'coral', ls='', lw=2, alpha=0.2, label = 'LII errorbar')
sns.scatterplot(x='Ubar', y='P+B-Eps', data=df_binEstimate_LI_UbarVsDeficit_noPW, color = 'dodgerblue', label = "binned LI")
plt.errorbar('Ubar', 'P+B-Eps', yerr='ci', data=df_binEstimate_LI_UbarVsDeficit_noPW, color = 'navy', ls='', lw=2, alpha=0.2, label = 'LI errorbar')
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
sns.scatterplot(x='Ubar', y='PW', data=df_binEstimate_LI_UbarVsPW, color = 'darkolivegreen', label = "binned PW boom-1")
plt.errorbar('Ubar', 'PW', yerr='ci', data=df_binEstimate_LI_UbarVsPW, color = 'darkolivegreen', ls='', lw=2, alpha=0.2, label = 'PW errorbar')


# plt.xscale('log')
# plt.yscale('log')
plt.legend(loc = 'upper left')
plt.xlabel('$\overline{u}$ [$ms^{-1}$]')
plt.ylabel('$P + B - \epsilon$, $T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
plt.title('$P+B-\epsilon$ and $T_{\widetilde{pw}}$ vs. Wind Speed ($\overline{u}$) \n (Binned)')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
# plt.axis('equal')
plt.hlines(y=0,xmin=2,xmax=16,color='k',linestyles='--')
plt.xlim(10,16)
plt.ylim(-0.2,0.2)
plt.tight_layout()
plt.savefig(plot_savePath + "scatterplotBIN_PWrestriction_UbarVSPplusBminusEps_withPWvsWindSpeed_combinedAnalysis.png",dpi=300)
plt.savefig(plot_savePath + "scatterplotBIN_PWrestriction_UbarVSPplusBminusEps_withPWvsWindSpeed_combinedAnalysis.pdf")

#%% Comparing deficit to z/L
# sort by z/L
zL_I_df = pd.DataFrame()
zL_I_df['zL_I'] = zL_df['zL_I_dc']
zL_I_df['prod_I'] = prod_df['prod_I']
zL_I_df['pb_I'] = prodPLUSbuoy['P+B LI']
zL_I_df['buoy_I'] = buoy_df['buoy_I']
zL_I_df['eps_I'] = eps_df['eps_LI']
zL_I_df.set_index('zL_I')
zL_I_df.sort_index(ascending=True)

zL_II_df = pd.DataFrame()
zL_II_df['zL_II'] = zL_df['zL_II_dc']
zL_II_df['prod_II'] = prod_df['prod_II']
zL_II_df['pb_II'] = prodPLUSbuoy['P+B LII']
zL_I_df['buoy_II'] = buoy_df['buoy_II']
zL_II_df['eps_II'] = eps_df['eps_LII']
zL_II_df.set_index('zL_II')
zL_II_df.sort_index(ascending=True)


plt.figure()
plt.scatter(zL_I_df['zL_I'], zL_I_df['eps_I'], color = 'darkorange', edgecolor = 'red', s=25, label = '$\epsilon$')
plt.scatter(zL_I_df['zL_I'], zL_I_df['prod_I'], color ='lightskyblue', edgecolor = 'navy', s = 15, label = 'P')
plt.scatter(zL_I_df['zL_I'], zL_I_df['buoy_I'], color ='green', edgecolor = 'darkgreen', s = 15, label = 'B')
plt.scatter(zL_I_df['zL_I'], zL_I_df['buoy_I']+zL_I_df['prod_I']-zL_I_df['eps_I'], color ='black', edgecolor = 'black', s = 15, label = 'deficit')
# plt.hlines(y=0,xmin=-7,xmax=1,color='k')
plt.legend(loc='upper left')
# plt.xlim(-7,0)
# plt.ylim(-0.02,0.02)

# plt.xlim(-1,1)

# plt.yscale('log')
# plt.xscale('log')
plt.title('Dissipation Rate and Production: Level I')
plt.xlabel('Stability (z/L)')
plt.ylabel('$\epsilon$, P, P+B [$m^2s^{-3}$]')

