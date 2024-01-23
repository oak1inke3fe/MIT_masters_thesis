#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:51:57 2023

@author: oaklin keefe


This file is used to analyze the production, dissipation, buoyancy, and wave-coherent pw terms of the TKE budget equation for specific high-wind events
with large dissipation deficits

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

# import scipy as sp
# import seaborn as sns

print('done with imports')
#%%

file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
plot_savePath = file_path

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic1_df['sum'] = np.ones(len(sonic1_df))

windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv")
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

date_df = pd.read_csv(file_path + "date_combinedAnalysis.csv")
date_df = date_df.drop(['Unnamed: 0'], axis=1)


print('done with setting up dataframes')


#%%
#add in epsilon variables per-level to dataframe 
eps_LIII = (eps_df['eps_sonic3']+eps_df['eps_sonic4'])/2
eps_LII = (eps_df['eps_sonic2']+eps_df['eps_sonic3'])/2
eps_LI = (eps_df['eps_sonic1']+eps_df['eps_sonic2'])/2
eps_df['eps_LIII'] = eps_LIII
eps_df['eps_LII'] = eps_LII
eps_df['eps_LI'] = eps_LI


#add in buoyancy per-level variables to dataframe
buoy_LIII = buoy_df['buoy_III']
buoy_LII = buoy_df['buoy_II']
buoy_LI = buoy_df['buoy_I']



print('done with getting epsilon with LI,II,II and dfs and buoyancy LI, II, III into dfs')


#%% Get rid of bad wind directions first
all_windDirs = True
onshore = False
offshore = False

plt.figure()
plt.scatter(windDir_df.index, windDir_df['alpha_s1'], color='gray',label='before')
plt.title('Wind direction BEFORE mask')
plt.xlabel('index')
plt.ylabel('Wind direction [deg]')
plt.legend(loc='lower right')
plt.show()

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
date_df[mask_goodWindDir] = np.nan

print('done with setting up  good wind direction only dataframes')

# plt.figure()
plt.scatter(windDir_df.index, windDir_df['alpha_s1'], color = 'green',label = 'after')
plt.title('Wind direction AFTER mask')
plt.xlabel('index')
plt.ylabel('Wind direction [deg]')
plt.legend(loc='lower right')

#%% set to near-neutral stability conditions

plt.figure()
plt.scatter(zL_df.index, zL_df['zL_II_dc'], label = 'z/L level II', color = 'k')
plt.scatter(zL_df.index, zL_df['zL_I_dc'], label = 'z/L level I', color = 'gray')
plt.title('z/L BEFORE neutral mask')
plt.xlabel('index')
plt.ylabel('z/L')
plt.legend(loc='lower right')
plt.show()

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
date_df[mask_neutral_zL] = np.nan


plt.figure()
plt.scatter(zL_df.index, zL_df['zL_II_dc'], label = 'z/L level II',color = 'green')
plt.scatter(zL_df.index, zL_df['zL_I_dc'], label = 'z/L level I', color = 'lightgreen')
plt.title('z/L AFTER neutral mask')
plt.xlabel('index')
plt.ylabel('z/L')
plt.legend(loc='lower right')
plt.show()


print('done with setting up near-neutral dataframes')

#%% mask to s1 Ubar >=8m/s for higher confidence comparison to pw

plt.figure()
plt.scatter(sonic1_df.index, sonic1_df['Ubar'], label = 'Ubar s1', color = 'gray')
plt.title('Ubar BEFORE 8m/s restriction mask')
plt.xlabel('index')
plt.ylabel('Ubar')
plt.legend(loc='lower right')
plt.xlim(1600,1950)
plt.show()

Ubar_index_array = np.arange(len(sonic1_df))
sonic1_df['new_index_arr'] = np.where(sonic1_df['Ubar']>=8, np.nan, Ubar_index_array)
mask_confidentPW_byUbarRestriction = np.isin(sonic1_df['new_index_arr'],Ubar_index_array)

sonic1_df[mask_confidentPW_byUbarRestriction]=np.nan

zL_df[mask_confidentPW_byUbarRestriction] = np.nan
eps_df[mask_confidentPW_byUbarRestriction] = np.nan
prod_df[mask_confidentPW_byUbarRestriction] = np.nan
buoy_df[mask_confidentPW_byUbarRestriction] = np.nan
z_df[mask_confidentPW_byUbarRestriction] = np.nan
windDir_df[mask_confidentPW_byUbarRestriction] = np.nan
pw_df[mask_confidentPW_byUbarRestriction] = np.nan
date_df[mask_confidentPW_byUbarRestriction] = np.nan

plt.figure()
plt.scatter(sonic1_df.index, sonic1_df['Ubar'], label = 'Ubar s1', color = 'g')
plt.title('Ubar AFTER 8m/s restriction mask')
plt.xlabel('index')
plt.ylabel('Ubar')
plt.legend(loc='lower right')
plt.show()

print('done with setting up Ubar restrictions to confident pw calcs')

#%%
plt.figure()
plt.scatter(sonic1_df.index, sonic1_df['Ubar'], label = 'Ubar s1', color = 'g')
plt.title('Ubar AFTER 8m/s restriction mask')
plt.xlabel('index')
plt.ylabel('Ubar')
plt.legend(loc='lower right')
plt.xlim(1600,1950) # spring
# plt.xlim(4790,4920) # fall
plt.hlines(y=8,xmin=0,xmax=7000,color='k')

spring_event_arr_sonic1 = np.array(sonic1_df['sum'][1609:1933]) # 8m/s restriction
spring_event_arr_sonic1_sum = np.nansum(spring_event_arr_sonic1) 
print(spring_event_arr_sonic1_sum)
print('wind direction range = ' + str(np.max(windDir_df['alpha_s3'][1609:1933]))+", "+ str(np.min(windDir_df['alpha_s3'][1609:1933])))

fall_event_arr_sonic1 = np.array(sonic1_df['sum'][4583:4807]) # 8m/s restriction
fall_event_arr_sonic1_sum = np.nansum(fall_event_arr_sonic1)
print(fall_event_arr_sonic1_sum)
print('wind direction range = ' + str(np.max(windDir_df['alpha_s3'][4583:4807]))+", "+ str(np.min(windDir_df['alpha_s3'][4583:4807])))
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
# Production minus dissipation (dissipation deficit if result is +)
p_minus_diss_I =  np.array(np.array(prod_df['prod_I'])-np.array(eps_df['eps_LI']))
p_minus_diss_II =  np.array(np.array(prod_df['prod_II'])-np.array(eps_df['eps_LII']))
p_minus_diss_III =  np.array(np.array(prod_df['prod_III'])-np.array(eps_df['eps_LIII']))
P_minus_eps_df = pd.DataFrame()
P_minus_eps_df['LI'] = np.array(p_minus_diss_I)
P_minus_eps_df['LII'] = np.array(p_minus_diss_II)
P_minus_eps_df['LIII'] = np.array(p_minus_diss_III)
print('done doing production minus dissipation and dataframe')

# Production plus buoyancy 
prodPLUSbuoy= pd.DataFrame()
prodPLUSbuoy['P+B LI'] = np.array(prod_df['prod_I'])+np.array(buoy_LI)
prodPLUSbuoy['P+B LII'] = np.array(prod_df['prod_II'])+np.array(buoy_LII)
prodPLUSbuoy['P+B LIII'] = np.array(prod_df['prod_III'])+np.array(buoy_LIII)

# Production plus buoyancy minus dissipation (dissipation deficit if result is +)
PplusB_minus_eps_df = pd.DataFrame()
PplusB_minus_eps_df['LI'] = np.array(prodPLUSbuoy['P+B LI'])-np.array(eps_df['eps_LI'])
PplusB_minus_eps_df['LII'] = np.array(prodPLUSbuoy['P+B LII'])-np.array(eps_df['eps_LII'])

print('done with combining production and buoyancy')

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
plt.plot(p_minus_diss_I, label = 'dissipation deficit')
plt.plot(np.array(pw_df['d_dz_pw_theory_I']), label = '$T_{\widetilde{pw}}$')
plt.legend()
plt.ylim(-0.1,0.1)
plt.title(pw_timeseries_title + "Level I")
plt.title(prod_eps_title + "Level I")

fig = plt.figure()
plt.plot(p_minus_diss_II, label = 'dissipation deficit')
plt.plot(np.array(pw_df['d_dz_pw_theory_II']), label = '$T_{\widetilde{pw}}$')
plt.legend()
plt.ylim(-0.1,0.1)
plt.title(pw_timeseries_title + "Level II")
plt.title(prod_eps_title + "Level II")

fig = plt.figure(figsize=(15,8))
plt.plot(p_minus_diss_II, color = 'darkorange', label = 'L II')
plt.plot(p_minus_diss_I, color = 'dodgerblue', label = 'L I')
plt.legend()
# plt.hlines(y=0, xmin=0,xmax=4395,color='k')
# plt.xlim(700,860)
plt.ylim(-0.2,0.2)
plt.xlabel('time index')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')
# plt.savefig(plot_savePath + "timeseries_neutralPvEps.png", dpi = 300)
# plt.savefig(plot_savePath + "timeseries_neutralPvEps.pdf")
plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis \n for $\overline{u}$ sonic1 >=8m/s')
plt.savefig(plot_savePath + "timeseries_neutralPvEps_pwUbarRestriction.png", dpi = 300)
plt.savefig(plot_savePath + "timeseries_neutralPvEps_pwUbarRestriction.pdf")

#%% Zoom in on period 1: may storm, indices [1609:1736+1] #this is when deficit at LI was primarily greater than deficit at LII 
fig = plt.figure(figsize=(15,8))
plt.plot(p_minus_diss_II, color = 'darkorange', label = 'L II')
plt.plot(p_minus_diss_I, color = 'dodgerblue', label = 'L I')
plt.legend()
plt.hlines(y=0, xmin=0,xmax=5000,color='k')
plt.vlines(x=1609, ymin=-1,ymax=1,color='k')
plt.vlines(x=1736, ymin=-1,ymax=1,color='k')
plt.ylim(-0.01,0.1)
plt.xlim(1600,1745)
plt.xlabel('time index')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')
# plt.savefig(plot_savePath + "timeseries_neutralPvEps.png", dpi = 300)
# plt.savefig(plot_savePath + "timeseries_neutralPvEps.pdf")
plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis \n for Period1: May Storm')

#%% Create new DF for period1
sonic1_df_period1 = sonic1_df[1609:1736+1]
sonic1_df_period1 = sonic1_df_period1.reset_index(drop=True)

zL_df_period1 = zL_df[1609:1736+1]
zL_df_period1 = zL_df_period1.reset_index(drop=True)

eps_df_period1 = eps_df[1609:1736+1]
eps_df_period1 = eps_df_period1.reset_index(drop=True)

prod_df_period1 = prod_df[1609:1736+1]
prod_df_period1 = prod_df_period1.reset_index(drop=True)

buoy_df_period1 = buoy_df[1609:1736+1]
buoy_df_period1 = buoy_df_period1.reset_index(drop=True)

z_df_period1 = z_df[1609:1736+1]
z_df_period1 = z_df_period1.reset_index(drop=True)

windDir_df_period1 = windDir_df[1609:1736+1]
windDir_df_period1 = windDir_df_period1.reset_index(drop=True)

pw_df_period1 = pw_df[1609:1736+1]
pw_df_period1 = pw_df_period1.reset_index(drop=True)

date_df_period1 = date_df[1609:1736+1]
date_df_period1 = date_df_period1.reset_index(drop=True)

P_minus_eps_df_period1 = P_minus_eps_df[1609:1736+1]
P_minus_eps_df_period1 = P_minus_eps_df_period1.reset_index(drop=True)

prodPLUSbuoy_period1 = prodPLUSbuoy[1609:1736+1]
prodPLUSbuoy_period1 = prodPLUSbuoy_period1.reset_index(drop=True)

PplusB_minus_eps_df_period1 = PplusB_minus_eps_df[1609:1736+1]
PplusB_minus_eps_df_period1 = PplusB_minus_eps_df_period1.reset_index(drop=True)

print(len(sonic1_df_period1))
#%% Zoom in on period 2: oct storm, indices [4591:4773+1] #this is when deficit at LI was primarily greater than deficit at LII 
fig = plt.figure(figsize=(15,8))
plt.plot(p_minus_diss_II, color = 'darkorange', label = 'L II')
plt.plot(p_minus_diss_I, color = 'dodgerblue', label = 'L I')
plt.legend()
plt.hlines(y=0, xmin=0,xmax=5000,color='k')
plt.vlines(x=4591, ymin=-1,ymax=1,color='k')
plt.vlines(x=4773, ymin=-1,ymax=1,color='k')
plt.ylim(-0.01,0.1)
plt.xlim(4580,4784)
plt.xlabel('time index')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')
# plt.savefig(plot_savePath + "timeseries_neutralPvEps.png", dpi = 300)
# plt.savefig(plot_savePath + "timeseries_neutralPvEps.pdf")
plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis \n for Period2: October Storm')

#%% Create new DF for period2
sonic1_df_period2 = sonic1_df[4591:4773+1]
sonic1_df_period2 = sonic1_df_period2.reset_index(drop=True)

zL_df_period2 = zL_df[4591:4773+1]
zL_df_period2 = zL_df_period2.reset_index(drop=True)

eps_df_period2 = eps_df[4591:4773+1]
eps_df_period2 = eps_df_period2.reset_index(drop=True)

prod_df_period2 = prod_df[4591:4773+1]
prod_df_period2 = prod_df_period2.reset_index(drop=True)

buoy_df_period2 = buoy_df[4591:4773+1]
buoy_df_period2 = buoy_df_period2.reset_index(drop=True)

z_df_period2 = z_df[4591:4773+1]
z_df_period2 = z_df_period2.reset_index(drop=True)

windDir_df_period2 = windDir_df[4591:4773+1]
windDir_df_period2 = windDir_df_period2.reset_index(drop=True)

pw_df_period2 = pw_df[4591:4773+1]
pw_df_period2 = pw_df_period2.reset_index(drop=True)

date_df_period2 = date_df[4591:4773+1]
date_df_period2 = date_df_period2.reset_index(drop=True)

P_minus_eps_df_period2 = P_minus_eps_df[4591:4773+1]
P_minus_eps_df_period2 = P_minus_eps_df_period2.reset_index(drop=True)

prodPLUSbuoy_period2 = prodPLUSbuoy[4591:4773+1]
prodPLUSbuoy_period2 = prodPLUSbuoy_period2.reset_index(drop=True)

PplusB_minus_eps_df_period2 = PplusB_minus_eps_df[4591:4773+1]
PplusB_minus_eps_df_period2 = PplusB_minus_eps_df_period2.reset_index(drop=True)

print(len(sonic1_df_period2))

#%%
fig = plt.figure(figsize=(15,8))
plt.plot(np.arange(len(windDir_df_period1)), windDir_df_period1['alpha_s4'], color = 'gray', label = 'period 1 (May)')
plt.plot(np.arange(len(windDir_df_period2)), windDir_df_period2['alpha_s4'], color = 'k', label = 'period 2 (Oct)')
plt.legend()
plt.xlabel('time index')
plt.ylabel('wind direction')
# plt.savefig(plot_savePath + "timeseries_neutralPvEps.png", dpi = 300)
# plt.savefig(plot_savePath + "timeseries_neutralPvEps.pdf")
plt.title('Neutral Conditions: Wind direction per period of high wind')
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
#%% dissipation deficit (without buoyancy) versus wave-coherent pw

# fig = plt.figure(figsize=(10,10))
# plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
# plt.scatter(p_minus_diss_I, np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# # plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('$P - \epsilon$ [$m^2s^{-3}$]')
# plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
# plt.title(pw_title)
# plt.legend()
# plt.axis('square')


# #same data as above, just flipping the order of plotting by level
# fig = plt.figure(figsize=(10,10))
# plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
# # plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
# plt.scatter(p_minus_diss_I, np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('$P - \epsilon$ [$m^2s^{-3}$]')
# plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
# plt.title(pw_title)
# plt.legend()
# plt.axis('square')
# print('done plotting P vs. Eps simple diss (new)')

# print('done with plotting dissipation deficit (without buoyancy) versus wave-coherent pw')

#%%

prodPLUSbuoy= pd.DataFrame()
prodPLUSbuoy['P+B LI'] = np.array(prod_df['prod_I'])+np.array(buoy_LI)
prodPLUSbuoy['P+B LII'] = np.array(prod_df['prod_II'])+np.array(buoy_LII)
prodPLUSbuoy['P+B LIII'] = np.array(prod_df['prod_III'])+np.array(buoy_LIII)

PplusB_minus_eps_df = pd.DataFrame()
PplusB_minus_eps_df['LI'] = np.array(prodPLUSbuoy['P+B LI'])-np.array(eps_df['eps_LI'])
PplusB_minus_eps_df['LII'] = np.array(prodPLUSbuoy['P+B LII'])-np.array(eps_df['eps_LII'])

print('done with combining production and buoyancy')
#%% dissipation deficit (with buoyancy) versus wave-coherent pw
fig = plt.figure(figsize=(6,6))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(PplusB_minus_eps_df['LII'], np.array(pw_df['d_dz_pw_theory_II']), color = 'darkorange', edgecolor = 'red', label = 'level II')
plt.scatter(PplusB_minus_eps_df['LI'], np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# plt.scatter(PplusB_minus_eps_df['LII'], np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P +B - \epsilon$ [$m^2s^{-3}$]')
plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
plt.legend()
plt.tight_layout()
plt.title('Neutral Conditions: $P+B-\epsilon$ vs. $T_{\widetilde{pw}}$')
plt.savefig(plot_savePath + "scatter_PW_vs_dissDeficit.png",dpi=300)
plt.savefig(plot_savePath + "scatter_PW_vs_dissDeficit.pdf")
# plt.title('Neutral Conditions: $P+B-\epsilon$ vs. $T_{\widetilde{pw}}$ \n for $\overline{u}$ sonic1 >=8m/s')
# plt.savefig(plot_savePath + "scatter_PW_vs_dissDeficit_pwUbarRestriction.png",dpi=300)
# plt.savefig(plot_savePath + "scatter_PW_vs_dissDeficit_pwUbarRestriction.pdf")
# plt.axis('square')
# plt.xlim(-0.0001,0.01)
# plt.ylim(-0.01,0.01)
print('done plotting P vs. Eps simple diss (new)')

print('done with plotting dissipation deficit with buoyancy versus wave-coherent pw')
#%%
# If we just want to examine the high wind event from Oct2-4, 2022, use the following mask:
# return to old version of code for this.

#%%
fig = plt.figure()
plt.plot(buoy_df['buoy_4'], label = 's4')
plt.plot(buoy_df['buoy_3'], label = 's3')
plt.plot(buoy_df['buoy_2'], label = 's2')
plt.plot(buoy_df['buoy_1'], label = 's1')
plt.legend()

print('done with plotting buoyancy')

#%% Correlation coefficients with Fixed ISR
PvEps_all_df = pd.DataFrame()
PvEps_all_df['P_I'] = np.array(prod_df['prod_I'])
PvEps_all_df['Eps_I'] = np.array(eps_df['eps_LI'])
PvEps_all_df['P_II'] = np.array(prod_df['prod_II'])
PvEps_all_df['Eps_II'] = np.array(eps_df['eps_LII'])
PvEps_all_df['P_III'] = np.array(prod_df['prod_III'])
PvEps_all_df['Eps_III'] = np.array(eps_df['eps_LIII'])

PvEps_I_df = pd.DataFrame()
PvEps_I_df['P_I'] = np.array(prod_df['prod_I'])
PvEps_I_df['Eps_I'] = np.array(eps_df['eps_LI'])

PvEps_II_df = pd.DataFrame()
PvEps_II_df['P_II'] = np.array(prod_df['prod_II'])
PvEps_II_df['Eps_II'] = np.array(eps_df['eps_LII'])

PvEps_III_df = pd.DataFrame()
PvEps_III_df['P_III'] = np.array(prod_df['prod_III'])
PvEps_III_df['Eps_III'] = np.array(eps_df['eps_LIII'])

break_index = 3959
PvEps_Spring_df = pd.DataFrame()
PvEps_Spring_df['P_I'] = np.array(prod_df['prod_I'][:break_index+1])
PvEps_Spring_df['Eps_I'] = np.array(eps_df['eps_LI'][:break_index+1])
PvEps_Spring_df['P_II'] = np.array(prod_df['prod_II'][:break_index+1])
PvEps_Spring_df['Eps_II'] = np.array(eps_df['eps_LII'][:break_index+1])
PvEps_Spring_df['P_III'] = np.array(prod_df['prod_III'][:break_index+1])
PvEps_Spring_df['Eps_III'] = np.array(eps_df['eps_LIII'][:break_index+1])
r_spring = PvEps_Spring_df.corr()
r_spring_I = r_spring['P_I'][1]
r_spring_I = round(r_spring_I, 3)
r_spring_II = r_spring['P_II'][3]
r_spring_II = round(r_spring_II, 3)
r_spring_III = r_spring['P_III'][5]
r_spring_III = round(r_spring_III, 3)

PvEps_Fall_df = pd.DataFrame()
PvEps_Fall_df['P_I'] = np.array(prod_df['prod_I'][break_index+1:])
PvEps_Fall_df['Eps_I'] = np.array(eps_df['eps_LI'][break_index+1:])
PvEps_Fall_df['P_II'] = np.array(prod_df['prod_II'][break_index+1:])
PvEps_Fall_df['Eps_II'] = np.array(eps_df['eps_LII'][break_index+1:])
PvEps_Fall_df['P_III'] = np.array(prod_df['prod_III'][break_index+1:])
PvEps_Fall_df['Eps_III'] = np.array(eps_df['eps_LIII'][break_index+1:])
r_fall = PvEps_Fall_df.corr()
r_fall_I = r_fall['P_I'][1]
r_fall_I = round(r_fall_I, 3)
r_fall_II = r_fall['P_II'][3]
r_fall_II = round(r_fall_II, 3)
r_fall_III = r_fall['P_III'][5]
r_fall_III = round(r_fall_III, 3)


r_I = PvEps_I_df.corr()
print(r_I)
r_I_str = r_I['P_I'][1]
r_I_str = round(r_I_str, 3)
print(r_I_str)

r_all = PvEps_all_df.corr()
print(r_all)
r_I_str = r_all['P_I'][1]
r_I_str = round(r_I_str, 3)
r_II_str = r_all['P_II'][3]
r_II_str = round(r_II_str, 3)
r_III_str = r_all['P_III'][5]
r_III_str = round(r_III_str, 3)
print(r_I_str)
print(r_II_str)
print(r_III_str)

#with buoyancy
PBvEps_all_df = pd.DataFrame()
PBvEps_all_df['PB_I'] = np.array(prodPLUSbuoy['P+B LI'])
PBvEps_all_df['Eps_I'] = np.array(eps_df['eps_LI'])
PBvEps_all_df['PB_II'] = np.array(prodPLUSbuoy['P+B LII'])
PBvEps_all_df['Eps_II'] = np.array(eps_df['eps_LII'])
PBvEps_all_df['PB_III'] = np.array(prodPLUSbuoy['P+B LIII'])
PBvEps_all_df['Eps_III'] = np.array(eps_df['eps_LIII'])

r_PB_all = PBvEps_all_df.corr()
print(r_PB_all)
r_I_PB_str = r_PB_all['PB_I'][1]
r_I_PB_str = round(r_I_PB_str, 3)
r_II_PB_str = r_PB_all['PB_II'][3]
r_II_PB_str = round(r_II_PB_str, 3)
r_III_PB_str = r_PB_all['PB_III'][5]
r_III_PB_str = round(r_III_PB_str, 3)
print(r_I_PB_str)
print(r_II_PB_str)
print(r_III_PB_str)



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
plt.xlim(10**-5,1)
plt.ylim(10**-5,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
# plt.title("SPRING: Neutral Conditions: $P$ vs. $\epsilon$")
plt.title("SPRING: Neutral Conditions: $P$ vs. $\epsilon$ \n for $\overline{u}$ sonic1 >=8m/s")
plt.legend(loc = 'lower right')
ax = plt.gca() 
# plt.text(.05, .9, "Pearson's r L III ={:.3f}".format(r_spring_III), transform=ax.transAxes)
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_spring_II), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_spring_I), transform=ax.transAxes)
plt.axis('equal')
plt.tight_layout()
# plt.savefig(plot_savePath + "scatterplot_PvEps_spring.png",dpi=300)
# plt.savefig(plot_savePath + "scatterplot_PvEps_spring.pdf")
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
plt.xlim(10**-5,1)
plt.ylim(10**-5,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
# plt.title("FALL: Neutral Conditions: $P$ vs. $\epsilon$")
plt.title("FALL: Neutral Conditions: $P$ vs. $\epsilon$ \n for $\overline{u}$ sonic1 >=8m/s")
plt.legend(loc = 'lower right')
ax = plt.gca() 
# plt.text(.05, .9, "Pearson's r L III ={:.3f}".format(r_fall_III), transform=ax.transAxes)
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_fall_II), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_fall_I), transform=ax.transAxes)
ax.set_xlim(10**-5,1)
ax.set_ylim(10**-5,1)
plt.axis('equal')
plt.tight_layout()
# plt.savefig(plot_savePath + "scatterplot_PvEps_fall.png",dpi=300)
# plt.savefig(plot_savePath + "scatterplot_PvEps_fall.pdf")
plt.savefig(plot_savePath + "scatterplot_PvEps_fall_pwUbarRestriction.png",dpi=300)
plt.savefig(plot_savePath + "scatterplot_PvEps_fall_pwUbarRestriction.pdf")
print('done plotting P vs. Eps FALL')
#%%
# Timeseries (split into 2 plots for each deployment)
import matplotlib.dates as mdates

date_df = pd.read_csv(file_path+'date_combinedAnalysis.csv')
dates_arr = np.array(pd.to_datetime(date_df['datetime']))


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,4))
ax1.plot(dates_arr[:break_index+1], p_minus_diss_I[:break_index+1], color = 'dodgerblue', label = 'L I')
ax1.plot(dates_arr[:break_index+1], p_minus_diss_II[:break_index+1], color = 'darkorange', label = 'L II')
ax1.legend(loc = 'lower left')
# ax1.set_title('Level I')
ax1.set_ylabel(r'$P-\epsilon$ [$m^{2}s^{-3}$]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax2.plot(dates_arr[break_index+1:], p_minus_diss_I[break_index+1:], color = 'dodgerblue', label = 'L I')
ax2.plot(dates_arr[break_index+1:], p_minus_diss_II[break_index+1:], color = 'darkorange', label = 'L II')
ax2.legend(loc = 'lower left')
ax2.set_ylabel(r'$P-\epsilon$ [$m^{2}s^{-3}$]')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
fig.suptitle(r'$P-\epsilon$', fontsize=16)
fig.savefig(plot_savePath + "timeseries_PvEps_combinedAnalysis.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_PvEps_combinedAnalysis.pdf")

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,4))
ax1.plot(dates_arr[:break_index+1], PplusB_minus_eps_df['LI'][:break_index+1], color = 'dodgerblue', label = 'L I')
ax1.plot(dates_arr[:break_index+1], PplusB_minus_eps_df['LII'][:break_index+1], color = 'darkorange', label = 'L II')
ax1.legend(loc = 'lower left')
# ax1.set_title('Level I')
ax1.set_ylabel(r'$P+B-\epsilon$ [$m^{2}s^{-3}$]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax2.plot(dates_arr[break_index+1:], PplusB_minus_eps_df['LI'][break_index+1:], color = 'dodgerblue', label = 'L I')
ax2.plot(dates_arr[break_index+1:], PplusB_minus_eps_df['LII'][break_index+1:], color = 'darkorange', label = 'L II')
ax2.legend(loc = 'lower left')
ax2.set_ylabel(r'$P+B-\epsilon$ [$m^{2}s^{-3}$]')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
fig.suptitle(r'$P+B-\epsilon$', fontsize=16)
fig.savefig(plot_savePath + "timeseries_PplusBvEps_combinedAnalysis.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_PplusBvEps_combinedAnalysis.pdf")

#%%
ymin = -0.7
ymax = 0.7
# same as above, but as scatter plots instead of lineplots
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,4))
ax1.scatter(dates_arr[:break_index+1], p_minus_diss_I[:break_index+1], color = 'dodgerblue', edgecolor='navy', label = 'L I', s=15)
ax1.scatter(dates_arr[:break_index+1], p_minus_diss_II[:break_index+1], color = 'darkorange', edgecolor='darkorange', label = 'L II', s=5)
ax1.legend(loc = 'lower left')
# ax1.set_title('Level I')
ax1.set_ylabel(r'$P-\epsilon$ [$m^{2}s^{-3}$]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.set_ylim(ymin,ymax)
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax2.scatter(dates_arr[break_index+1:], p_minus_diss_I[break_index+1:], color = 'dodgerblue', edgecolor='navy', label = 'L I', s=15)
ax2.scatter(dates_arr[break_index+1:], p_minus_diss_II[break_index+1:], color = 'darkorange', edgecolor='darkorange', label = 'L II', s=5)
ax2.legend(loc = 'lower left')
ax2.set_ylabel(r'$P-\epsilon$ [$m^{2}s^{-3}$]')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.set_ylim(ymin,ymax)
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
fig.suptitle(r'$P-\epsilon$', fontsize=16)
fig.savefig(plot_savePath + "timeseriesScatter_PvEps_combinedAnalysis.png",dpi=300)
fig.savefig(plot_savePath + "timeseriesScatter__PvEps_combinedAnalysis.pdf")

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,4))
ax1.scatter(dates_arr[:break_index+1], PplusB_minus_eps_df['LI'][:break_index+1], color = 'dodgerblue', edgecolor='navy', label = 'L I', s=15)
ax1.scatter(dates_arr[:break_index+1], PplusB_minus_eps_df['LII'][:break_index+1], color = 'darkorange', edgecolor='darkorange', label = 'L II', s=5)
ax1.legend(loc = 'lower left')
# ax1.set_title('Level I')
ax1.set_ylabel(r'$P+B-\epsilon$ [$m^{2}s^{-3}$]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.set_ylim(ymin,ymax)
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax2.scatter(dates_arr[break_index+1:], PplusB_minus_eps_df['LI'][break_index+1:], color = 'dodgerblue', edgecolor='navy', label = 'L I', s=15)
ax2.scatter(dates_arr[break_index+1:], PplusB_minus_eps_df['LII'][break_index+1:], color = 'darkorange', edgecolor='darkorange', label = 'L II', s=5)
ax2.legend(loc = 'lower left')
ax2.set_ylabel(r'$P+B-\epsilon$ [$m^{2}s^{-3}$]')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.set_ylim(ymin,ymax)
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
fig.suptitle(r'$P+B-\epsilon$', fontsize=16)
fig.savefig(plot_savePath + "timeseriesScatter__PplusBvEps_combinedAnalysis.png",dpi=300)
fig.savefig(plot_savePath + "timeseriesScatter__PplusBvEps_combinedAnalysis.pdf")


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
#%% Now we'll include buoyancy
# scatterplot of Production AND Buoyancy versus Dissipation: combined analysis
fig = plt.figure(figsize = (6,6))
# fig= plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prodPLUSbuoy['P+B LIII'], eps_df['eps_LIII'], color = 'seagreen', edgecolor = 'darkgreen', label = 'level III')
plt.scatter(prodPLUSbuoy['P+B LII'], eps_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prodPLUSbuoy['P+B LI'], eps_df['eps_LI'], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('$P\; +\; B$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
plt.title('P+B vs. $\epsilon$')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
plt.axis('equal')
plt.tight_layout()
plt.savefig(plot_savePath + "scatterplot_PplusBvEps_combinedAnalysis.png",dpi=300)
plt.savefig(plot_savePath + "scatterplot_PplusBvEps_combinedAnalysis.pdf")

# scatterplot of spring conditions
fig = plt.figure(figsize = (6,6))
# fig= plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prodPLUSbuoy['P+B LIII'][:break_index+1], eps_df['eps_LIII'][:break_index+1], color = 'seagreen', edgecolor = 'darkgreen',label = 'level III')
plt.scatter(prodPLUSbuoy['P+B LII'][:break_index+1], eps_df['eps_LII'][:break_index+1], color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prodPLUSbuoy['P+B LI'][:break_index+1], eps_df['eps_LI'][:break_index+1], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('$P\; +\; B$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
plt.title('SPRING: P+B vs. $\epsilon$')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
plt.axis('equal')
plt.tight_layout()
plt.savefig(plot_savePath + "scatterplot_PplusBvEps_Spring.png",dpi=300)
plt.savefig(plot_savePath + "scatterplot_PplusBvEps_Spring.pdf")

# scatterplot of fall conditions
fig = plt.figure(figsize = (6,6))
# fig= plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prodPLUSbuoy['P+B LIII'][break_index+1:], eps_df['eps_LIII'][break_index+1:], color = 'seagreen', edgecolor = 'darkgreen',label = 'level III')
plt.scatter(prodPLUSbuoy['P+B LII'][break_index+1:], eps_df['eps_LII'][break_index+1:], color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prodPLUSbuoy['P+B LI'][break_index+1:], eps_df['eps_LI'][break_index+1:], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('$P\; +\; B$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
plt.title('FALL: P+B vs. $\epsilon$')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
plt.axis('equal')
plt.tight_layout()
plt.savefig(plot_savePath + "scatterplot_PplusBvEps_Fall.png",dpi=300)
plt.savefig(plot_savePath + "scatterplot_PplusBvEps_Fall.pdf")



#%%
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
plt.scatter(zL_I_df['zL_I'], zL_I_df['buoy_I']+zL_I_df['prod_I']-zL_I_df['eps_I'], color ='black', edgecolor = 'black', s = 15, label = 'Deficit')
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

