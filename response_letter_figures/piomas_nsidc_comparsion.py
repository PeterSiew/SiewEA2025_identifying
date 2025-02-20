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
import tools; reload(tools)

if __name__ == "__main__":

    # obs2 is PIOMAS
    vars=['obs1_sic_en','obs2_sic_en']; ratios=[100,100]
    labels=['NSIDC','PIOMAS-20C']
    ensembles=[range(41,51), ['']]
    ensembles=[[''], ['']]
    st_yr=1901; end_yr=2025
    ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Seas - adjusted)
    mons=[9,10,11,12] # SOND

    growths={var:None for var in vars}
    sep_tss={var:None for var in vars}
    dec_tss={var:None for var in vars}
    en=''
    for i, var in enumerate(vars):
        data = tools.read_data(var+str(en), months='all', slicing=False, limit_lat=False) * ratios[i]
        data = data.sel(time=slice('%s-01-01'%st_yr, '%s-12-01'%end_yr))
        # Compute the weighted-area average of the data
        lons = data.longitude.values; lats = data.latitude.values
        data_ts = ct.weighted_area_average(data.values,ilat1,ilat2,ilon1,ilon2,lons,lats,lon_reverse=False, return_extract3d=False)
        data_ts = xr.DataArray(data_ts, dims=['time'], coords={'time':data.time}) 
        # Extract Sep and Dec
        mon_mask = data_ts.time.dt.month.isin(mons[0])
        sep_ts = data_ts.sel(time=mon_mask)
        mon_mask = data_ts.time.dt.month.isin(mons[-1])
        dec_ts = data_ts.sel(time=mon_mask)
        growth = dec_ts-sep_ts.assign_coords(time=dec_ts.time)
        growths[var]=growth
        sep_tss[var]=sep_ts.assign_coords(time=sep_ts.time.dt.year) 
        dec_tss[var]=dec_ts.assign_coords(time=dec_ts.time.dt.year)

    ## Mark 1987 data
    dec_tss['obs1_sic_en'][9]=np.nan

    ### Start plotting
    plt.close()
    colors=['red','green']
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,3))
    for i, var in enumerate(vars):
        years=sep_tss[var].time.values
        ax1.plot(years,sep_tss[var],color=colors[i],label=labels[i])
        years=dec_tss[var].time.values
        ax2.plot(years,dec_tss[var],color=colors[i],label=labels[i])

    legend_scatter=ax2.legend(bbox_to_anchor=(0.01, 0.01), ncol=1, loc='lower left',
            frameon=False, columnspacing=1.5, handletextpad=0.4, prop={'size':10})
    ## For ax1
    ts_sel1=sep_tss['obs1_sic_en'].sel(time=slice('1979-01-01','2010-12-31'))
    ts_sel2=sep_tss['obs2_sic_en'].sel(time=slice('1979-01-01','2010-12-31'))
    corr=tools.correlation_nan(ts_sel1,ts_sel2)
    ax1.annotate(r"$\rho$=%s"%(str(round(corr,2))),xy=(0.7,0.15), xycoords='axes fraction', fontsize=10)
    ax1.annotate(str('September SIC (%)'),xy=(-0.05,1.05),xycoords='axes fraction', fontsize=10)
    ## For ax2
    ts_sel1=dec_tss['obs1_sic_en'].sel(time=slice('1979-01-01','2010-12-31'))
    ts_sel2=dec_tss['obs2_sic_en'].sel(time=slice('1979-01-01','2010-12-31'))
    corr=tools.correlation_nan(ts_sel1,ts_sel2)
    ax2.annotate(r"$\rho$=%s"%(str(round(corr,2))),xy=(0.7,0.3), xycoords='axes fraction', fontsize=10)
    ax2.annotate(str('December SIC (%)'),xy=(-0.05,1.05),xycoords='axes fraction', fontsize=10)
    for i in ['right', 'top']:
        for ax in [ax1,ax2]:
            ax.spines[i].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2); ax1.tick_params(axis='y', which='both',length=2)
            #ax.set_xlim(years[0],years[-1])
            ax.set_xlim(1901,2023)
    fig_name = 'piomas_versus_nsdic'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.5) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.001)
    
