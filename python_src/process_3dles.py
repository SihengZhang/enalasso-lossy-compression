#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:54:27 2026

@author: vghate
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import netCDF4 as nc

files = sorted(glob.glob('/archives/disk1/hi_res_sim/data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1/*.v0'))

time = np.full(len(files),np.nan)
wpup = np.full((len(files),260),np.nan)
wpvp = np.full((len(files),260),np.nan)
wppp = np.full((len(files),260),np.nan)
wvar = np.full((len(files),260),np.nan)
uvar = np.full((len(files),260),np.nan)
vvar = np.full((len(files),260),np.nan)
wskew = np.full((len(files),260),np.nan)
wptp = np.full((len(files),260),np.nan)
wpqvp = np.full((len(files),260),np.nan)

for i in range(0,len(files)):
    dataset = nc.Dataset(files[i])
    time[i] = dataset.variables['time'][:].astype(float).data 
    x = dataset.variables['x'][:].astype(float).data 
    y = dataset.variables['y'][:].astype(float).data 
    z = dataset.variables['z'][:].astype(float).data  
    u = dataset.variables['U'][:].astype(float).data 
    v = dataset.variables['V'][:].astype(float).data 
    w = dataset.variables['W'][:].astype(float).data 
    pp = dataset.variables['PP'][:].astype(float).data 
    t = dataset.variables['TABS'][:].astype(float).data 
    qv = dataset.variables['QV'][:].astype(float).data 
    dataset.close()
    
    ### remove means 
    up = u - u.mean(axis=(0, 2, 3), keepdims=True) 
    vp = v - v.mean(axis=(0, 2, 3), keepdims=True) 
    wp = w - w.mean(axis=(0, 2, 3), keepdims=True) 
    tp = t - t.mean(axis=(0, 2, 3), keepdims=True) 
    qvp = qv - qv.mean(axis=(0, 2, 3), keepdims=True) 
    
    ### compute statistics 
    
    ## momentum fluxes 
    wpup[i,:] = np.mean(wp * up, axis=(0, 2, 3))
    wpvp[i,:] = np.mean(wp * vp, axis=(0, 2, 3))
    wppp[i,:] = np.mean(wp * pp, axis=(0, 2, 3))
    
    ## turb stats
    wvar[i,:] = np.mean(wp * wp, axis=(0, 2, 3))
    uvar[i,:] = np.mean(up * up, axis=(0, 2, 3))
    vvar[i,:] = np.mean(vp * vp, axis=(0, 2, 3))
    
    mu = wp.mean(axis=(0, 2, 3), keepdims=True)
    sg = wp.std(axis=(0, 2, 3), ddof=0, keepdims=True)
    wskew[i,:] = np.mean(((wp - mu) / sg)**3, axis=(0, 2, 3))  
    
    ## entropy fluxes
    wptp[i,:] = np.mean(wp * tp, axis=(0, 2, 3))
    wpqvp[i,:] = np.mean(wp * qvp, axis=(0, 2, 3))
    
### Plot things 

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)

m0 = axes[0].pcolormesh((time-time[0])*24, z/1000, wptp.T, shading="auto", cmap="RdBu_r")
axes[0].set_title("Heat flux: $\\overline{w' T'}$")
axes[0].set_ylabel("Height (km)")
axes[0].set_xlim(0,24)
axes[0].set_ylim(0,8)
axes[0].set_yticks(np.arange(0,9,1))
axes[0].set_xticks(np.arange(0,25,4))
c0 = fig.colorbar(m0, ax=axes[0], pad=0.01)
c0.set_label("wptp")

m1 = axes[1].pcolormesh((time-time[0])*24, z/1000, wpqvp.T, shading="auto", cmap="RdBu_r")
axes[1].set_title("Moisture flux: $\\overline{w' q_v'}$")
axes[1].set_xlabel("Time (Hour)")
axes[1].set_ylabel("Height (km)")
axes[1].set_xlim(0,24)
axes[1].set_ylim(0,8)
axes[1].set_yticks(np.arange(0,9,1))
axes[1].set_xticks(np.arange(0,25,4))
c1 = fig.colorbar(m1, ax=axes[1], pad=0.01)
c1.set_label("wpqvp")

plt.show()    
plt.savefig('plots/entropy_flux.png')
    

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)

m0 = axes[0].pcolormesh((time-time[0])*24, z/1000, wpup.T, shading="auto", cmap="RdBu_r")
axes[0].set_title("$\\overline{w' u'}$")
axes[0].set_ylabel("Height (km)")
axes[0].set_xlim(0,24)
axes[0].set_ylim(0,8)
axes[0].set_yticks(np.arange(0,9,1))
axes[0].set_xticks(np.arange(0,25,4))
c0 = fig.colorbar(m0, ax=axes[0], pad=0.01)
c0.set_label("wpup")

m1 = axes[1].pcolormesh((time-time[0])*24, z/1000, wpvp.T, shading="auto", cmap="RdBu_r")
axes[1].set_title("$\\overline{w' v'}$")
axes[1].set_ylabel("Height (km)")
axes[1].set_xlim(0,24)
axes[1].set_ylim(0,8)
axes[1].set_yticks(np.arange(0,9,1))
axes[1].set_xticks(np.arange(0,25,4))
c1 = fig.colorbar(m1, ax=axes[1], pad=0.01)
c1.set_label("wpvp")

m2 = axes[2].pcolormesh((time-time[0])*24, z/1000, wppp.T, shading="auto", cmap="RdBu_r")
axes[2].set_title("$\\overline{w' p'}$")
axes[2].set_xlabel("Time (Hour)")
axes[2].set_ylabel("Height (km)")
axes[2].set_xlim(0,24)
axes[2].set_ylim(0,8)
axes[2].set_yticks(np.arange(0,9,1))
axes[2].set_xticks(np.arange(0,25,4))
c2 = fig.colorbar(m2, ax=axes[2], pad=0.01)
c2.set_label("wppp")

plt.show()    
plt.savefig('plots/momentum_flux.png')
    

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)

m0 = axes[0].pcolormesh((time-time[0])*24, z/1000, uvar.T, shading="auto", cmap="RdBu_r")
axes[0].set_title("u-var")
axes[0].set_ylabel("Height (km)")
axes[0].set_xlim(0,24)
axes[0].set_ylim(0,8)
axes[0].set_yticks(np.arange(0,9,1))
axes[0].set_xticks(np.arange(0,25,4))
c0 = fig.colorbar(m0, ax=axes[0], pad=0.01)
c0.set_label("u-var")

m1 = axes[1].pcolormesh((time-time[0])*24, z/1000, vvar.T, shading="auto", cmap="RdBu_r")
axes[1].set_title("v-var")
axes[1].set_ylabel("Height (km)")
axes[1].set_xlim(0,24)
axes[1].set_ylim(0,8)
axes[1].set_yticks(np.arange(0,9,1))
axes[1].set_xticks(np.arange(0,25,4))
c1 = fig.colorbar(m1, ax=axes[1], pad=0.01)
c1.set_label("v-var")

m2 = axes[2].pcolormesh((time-time[0])*24, z/1000, wvar.T, shading="auto", cmap="RdBu_r")
axes[2].set_title("w-var")
axes[2].set_ylabel("Height (km)")
axes[2].set_xlim(0,24)
axes[2].set_ylim(0,8)
axes[2].set_yticks(np.arange(0,9,1))
axes[2].set_xticks(np.arange(0,25,4))
c2 = fig.colorbar(m2, ax=axes[2], pad=0.01)
c2.set_label("w-var")

m3 = axes[3].pcolormesh((time-time[0])*24, z/1000, wskew.T, shading="auto", cmap="RdBu_r")
axes[3].set_title("w-skew")
axes[3].set_xlabel("Time (Hour)")
axes[3].set_ylabel("Height (km)")
axes[3].set_xlim(0,24)
axes[3].set_ylim(0,8)
axes[3].set_yticks(np.arange(0,9,1))
axes[3].set_xticks(np.arange(0,25,4))
c3 = fig.colorbar(m3, ax=axes[3], pad=0.01)
c3.set_label("w-skew")


plt.show()    
plt.savefig('plots/turbstats.png')
    
   
       
    
   
       
       
    
   
    
    
    
    
    


