# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:08:04 2023

@author: oaklin keefe

NOTE: this file needs to be run on the remote desktop.

This file is used to calculate the rate of TKE dissipation using the inertial subrange method and the PSD of the vertical velocity.

Additionally, a Mean Absolute Deviation (MAD) critera from Bluteau et al. (2016) was used to quality control extreme spectra.

Input file location:
    /code_pipeline/Level1_align-interp/
INPUT files:
    sonic files (ports 1-4) input folder
    z_airSide_allSpring.csv
    z_airSide_allFall.csv
    
    
We also set:
    alpha = 0.65 (when working with W and not U)
    alpha = c1 = c1_prime

Output file locations:
    /code_pipeline/Level2/
OUTPUT files:
    Puu_exampleSpectra.png
    epsW_terms_sonic1_MAD_k_UoverZbar.csv
    epsW_terms_sonic2_MAD_k_UoverZbar.csv
    epsW_terms_sonic3_MAD_k_UoverZbar.csv
    epsW_terms_sonic4_MAD_k_UoverZbar.csv
    epsW_terms_combinedAnalysis_MAD_k_UoverZbar
"""


#%%
import os
import natsort
import numpy as np
import pandas as pd
import math
import scipy.signal as signal
import matplotlib.pyplot as plt
import datetime

filepath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
plot_savePath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'

print('done with imports')

#%%

#inputs:
    # n = number of observations (for freq 32 Hz, n = 38400)
    # phi_m = fit of spectrum as a function of k (wavenumber) from polyfit
    # phi_m_eqn5 = fit from eqn 5 of Bluteau (2016)
    # phi_w = the measured spectrum, as a function of k (wavenumber)
### function start
#######################################################################################
def MAD_epsilon(n, phi_m, phi_w):
    mad_arr = []
    # mad_eq5_arr = []
    for i in range(0,n):
        MAD_i = np.abs((phi_w[i]/phi_m[i]) - np.nanmean(phi_w/phi_m))
        mad_arr.append(MAD_i)
        
        # MAD_i_eq5 = (phi_w[i]/phi_m_eq5[i]) - np.mean(phi_w/phi_m_eq5)
        # mad_eq5_arr.append(MAD_i_eq5)
    
    MAD = 1/n*(np.sum(mad_arr))
    # MAD_eq5 = np.mean(mad_eq5_arr)
    
    return MAD
#######################################################################################
### function end
# returns: output_df
print('done with MAD_epsilon function')

### function start
#######################################################################################
def despikeThis(input_df,n_std):
    n = input_df.shape[1]
    output_df = pd.DataFrame()
    for i in range(0,n):
        elements_input = input_df.iloc[:,i]
        elements = elements_input
        mean = np.nanmean(elements)
        sd = np.nanstd(elements)
        extremes = np.abs(elements-mean)>(n_std*sd)
        elements[extremes]=np.NaN
        despiked = np.array(elements)
        colname = input_df.columns[i]
        output_df[str(colname)]=despiked

    return output_df
#######################################################################################
### function end
# returns: output_df
print('done with despike_this function')

### function start
#######################################################################################
# Function for interpolating the RMY sensor (freq = 32 Hz)
def interp_sonics123(df_sonics123):
    sonics123_xnew = np.arange(0, (32*60*20))   # this will be the number of points per file based
    df_align_interp= df_sonics123.reindex(sonics123_xnew).interpolate(limit_direction='both')
    return df_align_interp
#######################################################################################
### function end
# returns: df_align_interp
print('done with interp_sonics123 simple function')

### function start
#######################################################################################
# Function for interpolating the Gill sensor (freq = 20 Hz)
def interp_sonics4(df_sonics4):
    sonics4_xnew = np.arange(0, (20*60*20))   # this will be the number of points per file based
    df_align_interp_s4= df_sonics4.reindex(sonics4_xnew).interpolate(limit_direction='both')
    return df_align_interp_s4
#######################################################################################
### function end
# returns: df_align_interp_s4
print('done with interp_sonics4 function')
#%% testing on one file

filepath_PC = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-interp/'

my_file_level1_PC = filepath_PC+ 'mNode_Port1_20221012_192000_1.csv'

alpha = 0.65 #this is for W; alpha = 18/55*C*4/3 for C = Kolmogorov's constant 1.5 +/- 0.1
s1_df = pd.read_csv(my_file_level1_PC, index_col=0, header=0)
s1_df_despiked = despikeThis(s1_df,2)
s1_df_despiked = interp_sonics123(s1_df_despiked)
w_prime = np.array(s1_df_despiked['Wr']-s1_df_despiked['Wr'].mean())
U = np.abs(np.array(s1_df_despiked['Ur']))
U_mean = np.nanmean(U)
fs = 32 #since using a test file from sonic 1
N = 2048
N_s = fs*60*20
freq, Pww = signal.welch(w_prime,fs,nperseg=N,detrend=False) #pwelch function   16384
k = freq*(2*math.pi)/U_mean #converting to wavenumber spectrum
dfreq = np.max(np.diff(freq,axis=0))
dk = np.max(np.diff(k,axis=0))
Sww = Pww*dfreq/dk
k_fit = np.polyfit(k,np.log(Sww),3)
trendpoly = np.poly1d(k_fit) 
phi_m = np.exp(trendpoly(k))

isr = np.nonzero(k >= (2*np.pi/2.288))[0]   #for 2.288 agv sonic 1 height # multiply by <u>/z, then convert to wavenumber by mult. 2pi/<u> 
                                            # so <u> cancels out and we are left with 2pi/z

b = slice(isr.item(0),isr.item(-1))
spec_isr = Sww[b]
k_isr = k[b]
phi_m_isr = phi_m[b]

# eps = <spectra_isr * (k_isr^(5/3)) / alpha > ^(3/2)
eps_wnoise = np.mean((spec_isr*(k_isr**(5/3)))/alpha)**(3/2)

#least-squares minimization to exclude the noisefloor
X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
X=X_t.T
B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
noise = B.item(0)
eps = (B.item(1)/alpha)**(3/2) #slope of regression has alpha and eps^2/3
real_imag = isinstance(eps, complex)
if real_imag == True:
    print('eps is complex')

model_absNoise = alpha*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
model_realNoise = alpha*eps**(2/3)*k_isr**(-5/3)+(noise)
model_raw = alpha*eps_wnoise**(2/3)*k_isr**(-5/3)

d = np.abs(1.89*(2*N/N_s-1))
MAD_limit = 2*(2/d)**(1/2)

MAD_absNoise = MAD_epsilon(len(model_absNoise), model_absNoise, spec_isr)
MAD = MAD_epsilon(len(model_realNoise), model_realNoise, spec_isr)

epsilon_string = np.round(eps,6)

#PLOT spectra
fig = plt.figure()
# plt.figure(1,figsize=(8,6))
plt.loglog(k,Sww, color = 'k', label='spectra')
plt.loglog(k_isr,model_absNoise,color='r',label='ISR accounting for noise')
# plt.loglog(k_isr,model_realNoise,color='g',label='ISR accounting for true noise')
# plt.loglog(k_isr, model_raw, color = 'silver',label='ISR with no noise')
# plt.loglog(k,phi_m, color = 'g', label = 'polyfit')
plt.xlabel('Wavenumber ($k$)')
plt.ylabel('$P_{ww}$')
plt.legend(loc = 'lower left')
ax = plt.gca() 
plt.text(.03, .22, "$\epsilon$ = {:.4f}".format(epsilon_string)+" [$m^2s^{-3}$]", transform=ax.transAxes)
plt.title('$P_{ww}$ with Modeled Inertial Subrange (ISR)')
plt.savefig(plot_savePath + "Pww_exampleSpectra.png",dpi=300)

#%%
## Compare to a fixed frequency range (not necessary to run this cell; just for fun)
c_primeW = 0.65 # W
c_prime = c_primeW
freq, Pww = signal.welch(w_prime,fs,nperseg=N,detrend=False) #pwelch function   
k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
dfreq = np.max(np.diff(freq,axis=0))
dk = np.max(np.diff(k,axis=0))
Sww = Pww*dfreq/dk


k_lim_lower_freq = 0.5
k_lim_upper_freq = 10
k_lim_lower_waveNum = k_lim_lower_freq*(2*math.pi)/U_mean
k_lim_upper_waveNum = k_lim_upper_freq*(2*math.pi)/U_mean

isr = np.nonzero((k >= k_lim_lower_waveNum)&(k <= k_lim_upper_waveNum))[0]
if len(isr)>2:                        
    b = slice(isr.item(0),isr.item(-1))
    spec_isr = Sww[b]
    k_isr = k[b]
    eps_simplefreq = (np.nanmean(spec_isr*(k_isr**(5/3))/c_prime))**(3/2)
model_realNoise = alpha*eps_simplefreq**(2/3)*k_isr**(-5/3)

#PLOT spectra
fig = plt.figure()
# plt.figure(1,figsize=(8,6))
plt.loglog(k,Sww, color = 'k', label='spectra')
plt.loglog(k_isr,model_realNoise,color='g',label='ISR from Eps Simple Freq')
# plt.loglog(k_isr, model_raw, color = 'silver',label='ISR with no noise')
# plt.loglog(k,phi_m, color = 'g', label = 'polyfit')
plt.xlabel('Wavenumber ($k$)')
plt.ylabel('Pww')
plt.legend()
plt.title('Wavenumber Spectra (Pww) with Fixed Inertial Subrange (Fixed ISR)')

#%% For running on multiple files, we need the heights of the sonics relative to the sea-surface:

z_filepath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'
z_avg_df_spring = pd.read_csv(z_filepath+"z_airSide_allSpring.csv")
z_avg_df_spring = z_avg_df_spring.drop('Unnamed: 0', axis=1)
z_avg_df_fall = pd.read_csv(z_filepath+"z_airSide_allFall.csv")
z_avg_df_fall = z_avg_df_fall.drop('Unnamed: 0', axis=1)
print(z_avg_df_fall.columns)
print(z_avg_df_spring.columns)

plt.figure()
z_avg_df_spring.plot()
plt.title('Spring z-heights')
print('done with spring plot')

plt.figure()
z_avg_df_fall.plot()
plt.title('Fall z-heights')
print('done with fall plot')


#%% Here, we run on all the files in the /Level1_align-despike-interp/ folder
c_primeW = 0.65
c_prime = c_primeW #for simplicity so I don't have to change the code below that has c_prime not c_prime1

#set these as placeholders, but we will remove later
B_all_1 = [0,0]
noise_all_1 =[0]
eps_all_1 =[0]
eps_wnoise_all_1 = [0]
MAD_criteria_fit_1 = [0]
MAD_all_1 = [0]

file_save_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level2/'

'''
NOTE: you need to run this cell 4 different times, changing teh mNode_arr to each sonic number, before proceeding onto the next cell!

You may comment out the sonic's you aren't running, as done below
'''

mNode_arr = ['1',]
# mNode_arr = ['2',]
# mNode_arr = ['3',]
# mNode_arr = ['4',]


start=datetime.datetime.now()

for root, dirnames, filenames in os.walk(filepath): 
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)

        for mNode in mNode_arr:
            #spring deployment = APRIL 04, MAY 05, JUNE 06 (2022)
            if filename.startswith("mNode_Port"+mNode+"_202204"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.747
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.537
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 1.842
                    w_prime = np.array(df_interp['Wr']-df_interp['Wr'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Pww = signal.welch(w_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Sww = Pww*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Sww[b]
                        k_isr = k[b]                        
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            elif filename.startswith("mNode_Port"+mNode+"_202205"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.747
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.537
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 1.842
                    w_prime = np.array(df_interp['Wr']-df_interp['Wr'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Pww = signal.welch(w_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Sww = Pww*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Sww[b]
                        k_isr = k[b]                        
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            elif filename.startswith("mNode_Port"+mNode+"_202206"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.747
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.537
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 1.842
                    w_prime = np.array(df_interp['Wr']-df_interp['Wr'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Pww = signal.welch(w_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Sww = Pww*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Sww[b]
                        k_isr = k[b]                        
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            #fall deployment = SPETEMBER 09, OCTOBER 10, NOVEMBER 11 (2022)
            elif filename.startswith("mNode_Port"+mNode+"_202209"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.800
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.116
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 2.287
                    w_prime = np.array(df_interp['Wr']-df_interp['Wr'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Pww = signal.welch(w_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Sww = Pww*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Sww[b]
                        k_isr = k[b]                        
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            elif filename.startswith("mNode_Port"+mNode+"_202210"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.800
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.116
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 2.287
                    w_prime = np.array(df_interp['Wr']-df_interp['Wr'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Pww = signal.welch(w_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Sww = Pww*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Sww[b]
                        k_isr = k[b]                        
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            elif filename.startswith("mNode_Port"+mNode+"_202211"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.800
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.116
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 2.287
                    w_prime = np.array(df_interp['Wr']-df_interp['Wr'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Pww = signal.welch(w_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Sww = Pww*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Sww[b]
                        k_isr = k[b]                        
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            else:
                # error_files.append(filename[:-4])
                continue
end=datetime.datetime.now()

print('done with this part') 
print(start)
print(end) 
# printing start and end just tells you how long it takes to run     

# this gets rid of the placeholders from the previous step
noise_all_1 = np.delete(noise_all_1, 0,0)
eps_all_1 = np.delete(eps_all_1, 0,0)
eps_wnoise_all_1 = np.delete(eps_wnoise_all_1, 0,0)
MAD_all_1 = np.delete(MAD_all_1, 0,0)
MAD_criteria_fit_1 = np.delete(MAD_criteria_fit_1, 0,0)

# this tags the fils where the MAD criteria was met
MAD_goodFiles = np.where(MAD_criteria_fit_1 == 'False', np.nan,MAD_criteria_fit_1)

# Create a new df, and add the epsilon values to it, and the MAD values, and the MAD criteria, for if the MAD is acceptable or not
eps_df = pd.DataFrame()
eps_df['index']=np.arange(len(eps_all_1))
eps_df['epsW_sonic'+mNode] = eps_all_1
eps_df['MAD_value_'+mNode] = MAD_all_1
eps_df['MAD_criteria_met_'+mNode] = MAD_goodFiles

# keep the lines where the MAD criteria is met, and make the lines where it is not met equal NaN
eps_df['epsW_sonic'+mNode] = np.where(eps_df['MAD_criteria_met_'+mNode].isnull(),np.nan,eps_df['epsW_sonic'+mNode])
# save to a csv file
eps_df.to_csv(file_save_path+"epsW_terms_sonic"+mNode+"_MAD_k_UoverZbar.csv")

# #plot the MAD values versus the epsilon values, if you want
# plt.figure()
# plt.scatter(MAD_all_1,eps_all_1, color = 'b')
# plt.axvline(x=MAD_limit, color = 'gray')
# plt.yscale('log')
# plt.xlabel('MAD value')
# plt.ylabel('Dissipation Rate (with noise)')
# plt.title('MAD vs. Dissipation Rate sonic '+mNode)

print('done with mNode'+ str(mNode_arr))
print('MAKE SURE YOU RUN THIS FOR ALL 4 SONICS BEFORE GOING ONTO THE NEXT CELL') 

# import winsound
# duration = 3000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration) #can do winsound to alert you it has finished this step if you have windows
#%% Here, we will combine all the data frames we made from running the last cell 4 times 
# we will remove unnecessary columns first
eps_1 = pd.read_csv(file_save_path+"epsW_terms_sonic1_MAD_k_UoverZbar.csv")
eps_1 = eps_1.loc[:, ~eps_1.columns.str.contains('^Unnamed')]
eps_1 = eps_1.loc[:, ~eps_1.columns.str.contains('^index')]

eps_2 = pd.read_csv(file_save_path+"epsW_terms_sonic2_MAD_k_UoverZbar.csv")
eps_2 = eps_2.loc[:, ~eps_2.columns.str.contains('^Unnamed')]
eps_2 = eps_2.loc[:, ~eps_2.columns.str.contains('^index')]

eps_3 = pd.read_csv(file_save_path+"epsW_terms_sonic3_MAD_k_UoverZbar.csv")
eps_3 = eps_3.loc[:, ~eps_3.columns.str.contains('^Unnamed')]
eps_3 = eps_3.loc[:, ~eps_3.columns.str.contains('^index')]

eps_4 = pd.read_csv(file_save_path+"epsW_terms_sonic4_MAD_k_UoverZbar.csv")
eps_4 = eps_4.loc[:, ~eps_4.columns.str.contains('^Unnamed')]
eps_4 = eps_4.loc[:, ~eps_4.columns.str.contains('^index')]

# concat the dfs, then save
eps_combined = pd.concat([eps_1, eps_2, eps_3, eps_4], axis=1)
eps_combined.to_csv(file_save_path+'epsW_terms_combinedAnalysis_MAD_k_UoverZbar.csv')

print('done saving all sonic epsilon values to one file (for both spring and fall deployments)')
#%%
# create a column of epsilon values for where the MAD criteria is met, and make the lines where it is not met equal NaN
eps_combined= pd.read_csv(file_save_path+'epsW_terms_combinedAnalysis_MAD_k_UoverZbar.csv')
eps_combined['epsW_sonic1_MAD'] = np.where(eps_combined['MAD_criteria_met_1']!=True,np.nan,eps_combined['epsW_sonic1'])
eps_combined['epsW_sonic2_MAD'] = np.where(eps_combined['MAD_criteria_met_2']!=True,np.nan,eps_combined['epsW_sonic2'])
eps_combined['epsW_sonic3_MAD'] = np.where(eps_combined['MAD_criteria_met_3']!=True,np.nan,eps_combined['epsW_sonic3'])
eps_combined['epsW_sonic4_MAD'] = np.where(eps_combined['MAD_criteria_met_4']!=True,np.nan,eps_combined['epsW_sonic4'])

#save df with new columns to csv
eps_combined.to_csv(file_save_path+'epsW_terms_combinedAnalysis_MAD_k_UoverZbar.csv')
print('done saving epsilon df combined analysis to csv')
