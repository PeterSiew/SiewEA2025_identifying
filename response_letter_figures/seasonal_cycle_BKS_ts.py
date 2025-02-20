import xarray as xr
import numpy as np
import ipdb
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime as dt
import xesmf as xe

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools

if __name__ == "__main__":


    sic_var='obs_sic_en'; sic_ratio=100
    ensembles=['']
    en=''

    sic_var='CESM2_sic_en'; sic_ratio=100
    en='1'
    ensembles=range(1,51)
    years=range(1920,1940)
    years=range(1940,1960)
    years=range(1960,1980)
    years=range(1980,2000)
    years=range(2000,2020)
    years=range(2040,2060)
    years=range(2060,2080)
    years=range(2080,2100)
    years=range(2020,2040)
    years_fit = [range(1861,1881),range(1881,1901),range(1901,1921),range(1921,1941),range(1941,1961),range(1961,1981),
                range(1981,2001),range(2001,2021),range(2021,2041),range(2041,2061),range(2061,2081),range(2081,2101)]
    years_fit=[range(1921,1941),range(1941,1961),range(1961,1981),range(1981,2001),range(2001,2021),range(2021,2041),range(2041,2061),range(2061,2081),range(2081,2101)]

    tss_climatology={years:{} for years in years_fit}
    for years in years_fit:
        print(years)
        for en in ensembles:
            data = tools.read_data(sic_var+str(en), months='all', slicing=False, limit_lat=False) * sic_ratio
            # Constrain the data range at the beginning (useful for obs sic)
            data=data.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
            # Regrid (Can be earlier)
            #lats = np.arange(54,90,1); lons=np.arange(-179.5,180,1)
            #ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
            #regridder = xe.Regridder(data.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
            #data = regridder(data)
            data = data.compute()
            #data = data.sel(latitude=slice(65,85)).sel(longitude=slice(5,120))
            # Extract Barents-Kara and Leptev sea 
            lons=data.longitude.values; lats=data.latitude.values
            ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Seas - adjusted)
            ice_ts=ct.weighted_area_average(data.values, ilat1, ilat2, ilon1, ilon2, lons, lats)
            ice_ts=xr.DataArray(ice_ts,dims=['time'],coords={'time':data.time}) # has a size of 20
            tss = []
            for year in years:
                yr_mask=ice_ts.time.dt.year==year
                ts = ice_ts.sel(time=yr_mask)
                ts = ts.assign_coords(time=range(1,12+1))
                tss.append(ts)
            # The climatology of a 20-year period for each ensemble
            tss_climatology[years][en]=xr.concat(tss,dim='years').mean(dim='years') # mean over 20-year in each ensemble

    # Plot the seasonal cycle
    plt.close()
    fig, axs = plt.subplots(int(len(years_fit)/3),3,figsize=(12,len(years_fit)*0.8))
    axs=axs.flatten()
    for i, years in enumerate(years_fit):
        for en in ensembles:
            x = range(tss_climatology[years][en].size)
            axs[i].plot(x,tss_climatology[years][en],color='k',lw=0.1)
    for i, ax in enumerate(axs):
        ax.set_xlim(x[0],x[-1])
        ax.set_xticks(x)
        ax.set_xticklabels([i+1 for i in x])
        ax.set_ylim(0,85)
        ax.set_yticks((0,25,50,75,100))
        ax.annotate(str(years_fit[i][0])+' to '+str(years_fit[i][-1]),xy=(0.01,0.05),xycoords='axes fraction', fontsize=12)
        ax.fill_between([8,11], -100, 100, alpha=0.3, fc='orange')
        ax.annotate(str('SOND'),xy=(0.8,0.9),xycoords='axes fraction', fontsize=10)
        ax.annotate(str('JFM'),xy=(0.02,0.9),xycoords='axes fraction', fontsize=10)
        ax.fill_between([0,2], -100, 100, alpha=0.3, fc='yellow')
    fig_name = 'seasonal_cycle_BKL_sea'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.12, hspace=0.2) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.001)
