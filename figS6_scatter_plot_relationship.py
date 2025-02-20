import xarray as xr
import datetime as dt
import numpy as np
import ipdb
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import cartopy.crs as ccrs
from datetime import date
import matplotlib.pyplot as plt
from importlib import reload

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools

if __name__ == "__main__":

    #sic_var='miroc6_sic_en'; psl_var='miroc6_psl_en'; sic_ratio=1
    ensembles=range(1,51)

    # Whole periods
    years_fit = [range(1861,1881),range(1881,1901),range(1901,1921),range(1921,1941),range(1941,1961),range(1961,1981),
                range(1981,2001),range(2001,2021),range(2021,2041),range(2041,2061),range(2061,2081),range(2081,2101)]

    #ilat1, ilat2, ilon1, ilon2 = 80,88,20,150 # Upper region
    #ilat1, ilat2, ilon1, ilon2 = 70,75,20,80 # Lower region
    ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Sea - latest)
    if False: # for psl
        Ulat1, Ulat2, Ulon1, Ulon2 = 55,75,40,120 # urals
        Ilat1, Ilat2, Ilon1, Ilon2 = 65,85,-50,20 # Iceland
    if False: # For heat transport via BSO
        bso_lat1, bso_lat2, bso_lon1, bso_lon2 = 71,78,19,21 # Line of BSO
        bso_lat1, bso_lat2, bso_lon1, bso_lon2 = 71,78,89,91 # Line of Region above BKS
        bso_lat1, bso_lat2, bso_lon1, bso_lon2 = 70,73,10,80 # Region above BKS
    if False: # For lower and upper region
        #dlat1, dlat2, dlon1, dlon2 = 80,88,20,150 # Upper region
        #dlat1, dlat2, dlon1, dlon2 = 70,75,20,80 # Lower region
        dlat1=70; dlat2=88; dlon1=5; dlon2=120 # whole region

    ### Set x-var
    xvar ='cesm2_sicdynam_en'; xvar_ratio=1; xlabel='SOND dynam sea ice growth (%)'; xseason='SOND'
    xvar ='intcesm2_sic_en'; xvar_ratio=100; xlabel='Oct sea ice (%)'; xseason='OND'
    xvar ='cesm2_otemp_en_sep'; xvar_ratio=1; xlabel='Sep 50-m ocean temperature (K)'; xseason='SOND'
    xvar ='cesm2_otemp_en'; xvar_ratio=1; xlabel='SOND 50-m ocean temperature (K)'; xseason='SOND'
    xvar ='cesm2_psl_en'; xvar_ratio=1; xlabel='SOND circulation (%)'; xseason='SOND'
    xvar ='intCESM2_sic_en'; xvar_ratio=100; xlabel='Sep sea ice (%)'; xseason='SOND'
    ### Set y-var
    yvar ='cesm2_T2M_en'; yvar_ratio=1; ylabel='SOND T2M (K)'; yseason='SOND'
    yvar ='cesm2_iceoceanflux_en'; yvar_ratio=1; ylabel='SOND ice-to-ocean flux (Wm-2)'; yseason='SOND'
    yvar ='cesm2_sst_en'; yvar_ratio=1; ylabel='SST (K)'; yseason='SOND'
    yvar ='cesm2_otemp_en_sep'; yvar_ratio=1; ylabel='Sep 50-m ocean temperature (K)'; yseason='SOND'
    yvar ='cesm2_out_en'; yvar_ratio=1; ylabel='SOND BSO heat transport (K cm/s)'; yseason='SOND'
    yvar ='cesm2_otemp_en'; yvar_ratio=1; ylabel='SOND 50-m ocean temperature (K)'; yseason='SOND'
    yvar ='cesm2_sicthermo_en'; yvar_ratio=1; ylabel='SOND thermo sea ice growth (%)'; yseason='SOND'
    yvar ='cesm2_sicdynam_en'; yvar_ratio=1; ylabel='SOND dynam sea ice growth (%)'; yseason='SOND'
    yvar='intcesm2_sic_en'; yvar_ratio=100; ylabel='Sep sea ice (%)'; yseason='SOND'
    yvar ='CESM2_sic_en'; yvar_ratio=100; ylabel='SOND sea ice growth (%)'; yseason='SOND'

    xs = {years:[] for years in years_fit}
    ys = {years:[] for years in years_fit}
    for years in years_fit:
        print(years)
        for en in ensembles:
            for year in years:
                # X-var
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(xseason,xvar,en,year)
                data=xr.open_dataset(path)['training']*xvar_ratio
                data=data.squeeze() # Remove the single dimension of the data
                if xvar in ['cesm2_psl_en']:
                    ural_data=data.sel(latitude=slice(Ulat1,Ulat2)).sel(longitude=slice(Ulon1,Ulon2))
                    lats=ural_data.latitude.values; lons=ural_data.longitude.values
                    ural_ts=ct.weighted_area_average(ural_data.values[np.newaxis,:,:],Ulat1,Ulat2,Ulon1,Ulon2,lons,lats)
                    #
                    iceland_data=data.sel(latitude=slice(Ilat1,Ilat2)).sel(longitude=slice(Ilon1,Ilon2))
                    lats=iceland_data.latitude.values; lons=iceland_data.longitude.values
                    iceland_ts=ct.weighted_area_average(iceland_data.values[np.newaxis,:,:],Ilat1,Ilat2,Ilon1,Ilon2,lons,lats)
                    xs[years].append(iceland_ts.item()-ural_ts.item())
                else:
                    data=data.sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2))
                    lats=data.latitude.values; lons=data.longitude.values
                    data_ts=ct.weighted_area_average(data.values[np.newaxis,:,:],ilat1,ilat2,ilon1,ilon2,lons,lats)
                    xs[years].append(data_ts.item())
                # Y-var
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(yseason,yvar,en,year)
                data=xr.open_dataset(path)['training']*yvar_ratio
                data=data.squeeze() # Remove the single dimension of the data
                if yvar in ['cesm2_out_en']:
                    data=data.sel(latitude=slice(bso_lat1,bso_lat2)).sel(longitude=slice(bso_lon1,bso_lon2))
                elif yvar in ['cesm2_sicdynam_en', 'cesm2_sicthermo_en', 'cesm2_sic_en']:
                    data=data.sel(latitude=slice(dlat1,dlat2)).sel(longitude=slice(dlon1,dlon2))
                else:
                    data=data.sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2))
                lats=data.latitude.values; lons=data.longitude.values
                data_ts=ct.weighted_area_average(data.values[np.newaxis,:,:],ilat1,ilat2,ilon1,ilon2,lons,lats)
                ys[years].append(data_ts.item())

    if True: # Plot the scatter relationship
        plt.close(); fig,ax1=plt.subplots(1,1,figsize=(9,3))
        colors=['royalblue','r','g','orange','gray','lime','cyan','gold','pink','violet',
                'brown','darkviolet','peru','orchid','crimson']*2
        colors=['#fbefec','#f7e0d9','#f3d1c8','#f0c3b8',
                '#ebb0a3','#e79e8e','#e28875','#dd735d',
                '#d75a40','#d14124','#c3381c','#b63014']
        colors=['#8dd3c7','#ffffb3','#bebada','#fb8072',
                '#80b1d3','#fdb462','#b3de69','#fccde5',
                '#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
        colors=['#a6cee3','#1f78b4','#b2df8a','#33a02c',
                '#fb9a99','#e31a1c','#fdbf6f','#ff7f00',
                '#cab2d6','#6a3d9a','yellow','#b15928']
        colors=['#a6cee3','#1f78b4','#b2df8a','#33a02c',
                '#fb9a99','goldenrod','#fdbf6f','#ff7f00',
                '#cab2d6','#6a3d9a','yellow','darkred']
        corrs=[]
        for i, years in enumerate(years_fit):
            corr=round(tools.correlation_nan(xs[years],ys[years]),2)
            corrs.append(corr)
            regress=round(tools.linregress_nan(xs[years],ys[years])[0],5)
            #label=str(years[0])+ ' to ' + str(years[-1]+1) + ' (' + str(corr) + ')'
            #label=str(years[0])+ '-' + str(years[-1]+1) + ' (' + 'B='+ str(regress) + ')' + ' (' + 'rho='+ str(corr) + ')' 
            label=str(years[0])+ '-' + str(years[-1]) + ' (' + r'$\rho$='+ str(corr) + ')' 
            if False: # Stanadraize the index
                ax1.scatter(xs[years]-np.mean(xs[years]), ys[years], s=1, color=colors[i], label=label)
            else: # Don't do standardization
                xs[years]=np.array(xs[years]); ys[years]=np.array(ys[years])
                ax1.scatter(xs[years], ys[years], s=0.3, color=colors[i])
                ax1.scatter([-10,-10], [-10,-10], s=15, color=colors[i], label=label)
                if True: # Plot the high 10% and low 10%
                    pass
                    #high_growth=np.percentile(xs[years],90); high_mask=(xs[years]>high_growth).nonzero()[0]
                    #low_growth=np.percentile(xs[years],10); low_mask=(xs[years]<low_growth).nonzero()[0]
                    #ax1.scatter(xs[years][high_mask], ys[years][high_mask], s=0.3, color='none',edgecolors='k',zorder=100)
                    #ax1.scatter(xs[years][low_mask], ys[years][low_mask], s=1, marker='d',color='none',edgecolors='none',zorder=100)
        if True: # Setup the legend
            ax1.legend(bbox_to_anchor=(-0.05,1.01), ncol=3, loc='lower left', frameon=False, columnspacing=2, 
                        handletextpad=0.1, labelspacing=0.1, fontsize=10)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.axhline(y=0, color='lightgray', linestyle='--', linewidth=1)
        ax1.axvline(x=0, color='lightgray', linestyle='--', linewidth=1)
        ax1.set_xlim(-1,65)
        ax1.set_ylim(-2,83)
        for ax in [ax1]:
            for i in ['right', 'top']:
                ax.spines[i].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
            ax.tick_params(axis='x', direction="out", length=3, colors='black')
            ax.tick_params(axis='y', direction="out", length=3, colors='black')
        fig_name = 'scatter_between_octice_and_icegrowth'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

