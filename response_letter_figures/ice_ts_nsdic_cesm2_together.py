import xarray as xr
import numpy as np
import datetime as dt
import ipdb
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import pandas as pd
import scipy
import xesmf as xe

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools

if __name__ == "__main__":

    vars=['ACCESS-ESM1-5_sic_en','obs4_sic_en']; ratios=[1,100]; ensembles=[range(31,41), ['']]
    vars=['CESM2_sic_en','obs4_sic_en']; ratios=[100,100]; ensembles=[range(41,51), ['']]
    years=[(1861,2101),(1941,2022)]
    ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Seas - adjusted)
    mons=[9,10,11,12] # SOND

    growths={var:{} for var in vars}
    sep_tss={var:{} for var in vars}
    for i, var in enumerate(vars):
        for en in ensembles[i]:
            print(var,en)
            st_yr,end_yr=years[i]
            data = tools.read_data(var+str(en), months='all', slicing=False, limit_lat=False) * ratios[i]
            data = data.sel(time=slice('%s-01-01'%st_yr, '%s-12-01'%end_yr))
            # Compute the weighted-area average of the data
            lons = data.longitude.values; lats = data.latitude.values
            data_ts = ct.weighted_area_average(data.values,ilat1,ilat2,ilon1,ilon2,lons,lats,lon_reverse=False, return_extract3d=False)
            data_ts = xr.DataArray(data_ts, dims=['time'], coords={'time':data.time}) 
            # Extract Decemebr and October
            mon_mask = data_ts.time.dt.month.isin(mons[0])
            sep_ts = data_ts.sel(time=mon_mask)
            mon_mask = data_ts.time.dt.month.isin(mons[-1])
            dec_ts = data_ts.sel(time=mon_mask)
            growth = dec_ts -sep_ts.assign_coords(time=dec_ts.time)
            growths[var][en]=growth
            sep_tss[var][en]=sep_ts 

    ### For models - all ensembles (Figure 1C)
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(5,1.5))
    x_model = list(growths[vars[0]][ensembles[0][0]].time.dt.year.values)
    ax1.plot(x_model, np.mean([sep_tss[vars[0]][en] for en in ensembles[0]],axis=0), color='orange', linewidth=1, label='$ice_{sep}$', zorder=2)
    ax1.plot(x_model,np.mean([growths[vars[0]][en] for en in ensembles[0]],axis=0),color='blue',linewidth=1,label='$ice_{growth}$',zorder=2)
    for en in ensembles[0]: 
        # Intitial sea ice state, and Dec ice
        ax1.plot(x_model,sep_tss[vars[0]][en], color='bisque', linewidth=0.5, zorder=0.5)
        ax1.plot(x_model, growths[vars[0]][en], color='royalblue', linewidth=0.2,zorder=1)
    if True: # Set the legend 
        legend_scatter=ax1.legend(bbox_to_anchor=(-0.03, 0.92), ncol=2, loc='lower left',
                frameon=False, columnspacing=1.5, handletextpad=0.4, prop={'size':10})
    ### For observations (Figure 1B)
    # For obs Sep-ice
    x_obs=sep_tss[vars[1]][''].time.dt.year.values
    ax1.plot(x_obs,sep_tss[vars[1]][''], color='orange', linewidth=1, linestyle='-', zorder=10)
    ax1.plot(x_obs,sep_tss[vars[1]][''], color='black', linewidth=3, linestyle='-', zorder=5)
    ## For obs ice grwoth 
    x_obs= growths[vars[1]][''].time.dt.year.values
    ax1.plot(x_obs, growths[vars[1]][''], color='blue', linewidth=1, linestyle='-',zorder=11)
    ax1.plot(x_obs, growths[vars[1]][''], color='black', linewidth=3, linestyle='-',zorder=10.5)
    ###
    if True: # Set the legend 
        legend_scatter=ax1.legend(bbox_to_anchor=(-0.03, 0.92), ncol=2, loc='lower left',
                frameon=False, columnspacing=1.5, handletextpad=0.4, prop={'size':9})
    ### Set up the xticks
    ax1.set_xticks(x_model[::10])
    ax1.set_xticklabels(x_model[::10], rotation=45)
    ax1.set_xlim(x_model[0], x_model[-1])
    ax1.set_ylim(-5,90)
    ax1.set_ylabel('Sea ice\nconcentration (%)')
    # Set the frame
    for i in ['right', 'top']:
        for ax in [ax1]:
            ax.spines[i].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2); ax1.tick_params(axis='y', which='both',length=2)
    ### Save the figure
    fig_name = 'ice_index_together_NSIDC_CESM2'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

