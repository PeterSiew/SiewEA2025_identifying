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

    vars=['CESM2_sic_en','obs4_sic_en']; ratios=[100,100]
    ensembles=[range(41,51), ['']]
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

    if True: # For models - all ensembles (Figure 1C)
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(5,1.5))
        x_model = list(growths[vars[0]][ensembles[0][0]].time.dt.year.values)
        ax1.plot(x_model, np.mean([sep_tss[vars[0]][en] for en in ensembles[0]],axis=0), color='orange', linewidth=1, label='$ice_{sep}$', zorder=2)
        ax1.plot(x_model,np.mean([growths[vars[0]][en] for en in ensembles[0]],axis=0),color='blue',linewidth=1,label='$ice_{growth}$',zorder=2)
        for en in ensembles[0]: 
            # Intitial sea ice state, and Dec ice
            #ax1.plot(x, dec_ts, linewidth=0.5, color='royalblue', label='Dec')
            ax1.plot(x_model,sep_tss[vars[0]][en], color='bisque', linewidth=0.5, zorder=0.5)
            ax1.plot(x_model, growths[vars[0]][en], color='royalblue', linewidth=0.2,zorder=1)
        ### Setup the ax1
        #x_model=x_model+[2100]
        ax1.set_xticks(x_model[::10])
        ax1.set_xticklabels(x_model[::10], rotation=45)
        ax1.set_xlim(x_model[0], x_model[-1])
        ax1.set_ylim(-5,90)
        ax1.set_ylabel('Sea ice\nconcentration (%)')
        if True: # Plot the correlation for CESM2
            corr_y=85
            fs=8
            #
            st_date='1941-01-01'
            end_date='1960-12-31'
            ax1.fill_between([1941,1960], -100, 100, alpha=0.3, fc='grey')
            growth_ts=np.array([growths['CESM2_sic_en'][en].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
            sep_ts=np.array([sep_tss['CESM2_sic_en'][en].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
            corr= tools.correlation_nan(sep_ts,growth_ts)
            ax1.annotate(r"$\rho$=%s"%(str(round(corr,2))),xy=(1941,corr_y), xycoords='data', fontsize=fs)
            #
            st_date='1981-01-01'
            end_date='2000-12-31'
            ax1.fill_between([1981,2000], -100, 100, alpha=0.3, fc='grey')
            growth_ts=np.array([growths['CESM2_sic_en'][en].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
            sep_ts=np.array([sep_tss['CESM2_sic_en'][en].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
            corr= tools.correlation_nan(sep_ts,growth_ts)
            ax1.annotate(r"$\rho$=%s"%(str(round(corr,2))),xy=(1981,corr_y), xycoords='data', fontsize=fs)
            #
            st_date='2021-01-01'
            end_date='2040-12-31'
            ax1.fill_between([2021,2040], -100, 100, alpha=0.3, fc='grey')
            growth_ts=np.array([growths['CESM2_sic_en'][en].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
            sep_ts=np.array([sep_tss['CESM2_sic_en'][en].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
            corr= tools.correlation_nan(sep_ts,growth_ts)
            ax1.annotate(r"$\rho$=%s"%(str(round(corr,2))),xy=(2021,corr_y), xycoords='data', fontsize=fs)
            #
            st_date='2061-01-01'
            end_date='2080-12-31'
            ax1.fill_between([2061,2080], -100, 100, alpha=0.3, fc='grey')
            growth_ts=np.array([growths['CESM2_sic_en'][en].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
            sep_ts=np.array([sep_tss['CESM2_sic_en'][en].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
            corr= tools.correlation_nan(sep_ts,growth_ts)
            ax1.annotate(r"$\rho$=%s"%(str(round(corr,2))),xy=(2061,corr_y), xycoords='data', fontsize=fs)
        ax1.annotate("CESM2",xy=(0.01,0.05),xycoords='axes fraction', fontsize=9)
        if True: # Set the legend 
            legend_scatter=ax1.legend(bbox_to_anchor=(-0.03, 0.92), ncol=2, loc='lower left',
                    frameon=False, columnspacing=1.5, handletextpad=0.4, prop={'size':10})
        # Set the frame
        for i in ['right', 'top']:
            #for ax in [ax1,ax2]:
            for ax in [ax1]:
                ax.spines[i].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2); ax1.tick_params(axis='y', which='both',length=2)
        fig_name = 'fig1C_SOND_ice_growth_ts'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.001)
    if True: # For observations (Figure 1B)
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(5,1.5))
        ####
        # For obs Sep-ice
        x_obs=sep_tss[vars[1]][''].time.dt.year.values
        ax1.plot(x_obs,sep_tss[vars[1]][''], color='orange', linewidth=1, linestyle='-', label='$ice_{sep}$', zorder=2)
        ## For obs ice grwoth 
        x_obs= growths[vars[1]][''].time.dt.year.values
        ax1.plot(x_obs, growths[vars[1]][''], color='blue', linewidth=1, linestyle='-',label='$ice_{growth}$',zorder=2)
        ## Setup the ax2
        nn=5
        xticks=x_obs
        ax1.set_xticks(xticks[::nn])
        ax1.set_xticklabels(xticks[::nn], rotation=50)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylim(-5,80)
        ax1.set_ylabel('Sea ice\nconcentration (%)')
        ## Add correlations for obs
        corr_y=70
        fs=9.5
        #
        st_date='1941-01-01'
        end_date='1960-12-31'
        ax1.fill_between([1941,1960], -100, 100, alpha=0.3, fc='grey')
        growth_ts=np.array([growths['obs4_sic_en'][''].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
        sep_ts=np.array([sep_tss['obs4_sic_en'][''].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
        corr= tools.correlation_nan(sep_ts,growth_ts)
        ax1.annotate(r"$\rho$=%s"%(str(round(corr,2))),xy=(1941,corr_y), xycoords='data', fontsize=fs)
        #
        st_date='1981-01-01'
        end_date='2000-12-31'
        ax1.fill_between([1981,2000], -100, 100, alpha=0.3, fc='grey')
        growth_ts=np.array([growths['obs4_sic_en'][''].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
        sep_ts=np.array([sep_tss['obs4_sic_en'][''].sel(time=slice(st_date,end_date)).values for en in range(41,51)])
        corr= tools.correlation_nan(sep_ts,growth_ts)
        ax1.annotate("Observations",xy=(0.01,0.05),xycoords='axes fraction', fontsize=9)
        ###
        ax1.annotate(r"$\rho$=%s"%(str(round(corr,2))),xy=(1981,corr_y), xycoords='data', fontsize=fs)
        if True: # Set the legend 
            legend_scatter=ax1.legend(bbox_to_anchor=(-0.03, 0.92), ncol=2, loc='lower left',
                    frameon=False, columnspacing=1.5, handletextpad=0.4, prop={'size':9})
        # Set the frame
        for i in ['right', 'top']:
            #for ax in [ax1,ax2]:
            for ax in [ax1]:
                ax.spines[i].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2); ax1.tick_params(axis='y', which='both',length=2)
        fig_name = 'fig1B_SOND_ice_growth_ts'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

