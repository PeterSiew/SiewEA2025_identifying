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

    ensembles=range(1,11)
    ensembles=range(1,51)
    years_fit=[range(1941,1961),range(1961,1981),range(1981,2001),range(2001,2021)] # For testing
    years_fit = [range(1861,1881),range(1881,1901),range(1901,1921),range(1921,1941),range(1941,1961),range(1961,1981),
                range(1981,2001),range(2001,2021),range(2021,2041),range(2041,2061),range(2061,2081),range(2081,2101)]

    # Sea sea ice
    ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Seas - adjusted)

    # Set x-var
    xvar ='CESM2_psl_en'; xvar_ratio=1; xvar_season='SOND'
    # Set y-var
    yvar ='CESM2_sic_en'; yvar_ratio=100; yvar_season='SOND' # sea ice growth

    all_years=[yr for years in years_fit for yr in years]
    X = {yr:{en:None for en in ensembles} for yr in all_years}
    Y = {yr:{en:None for en in ensembles} for yr in all_years}
    for years in years_fit:
        print(years)
        for en in ensembles:
            for year in years:
                # X-var (the PSL)
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(xvar_season,xvar,en,year)
                data=xr.open_dataset(path)['training']*xvar_ratio
                data=data.squeeze() 
                X[year][en]=data
                # Y-var (the timeseries)
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(yvar_season,yvar,en,year)
                data=xr.open_dataset(path)['training']*yvar_ratio
                data=data.squeeze() # Remove the single dimension of the data
                data=data.sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2))
                lats=data.latitude.values; lons=data.longitude.values
                data_ts=ct.weighted_area_average(data.values[np.newaxis,:,:],ilat1,ilat2,ilon1,ilon2,lons,lats)
                Y[year][en]=data_ts.item()

    # Regress SLP on to the sea ice growth index
    corrs,slopes,pvalues={},{},{}
    for years in years_fit:
        tss=[]
        X_regresses=[]
        for year in years:
            for en in ensembles:
                ts = Y[year][en]
                X_regress = X[year][en]
                tss.append(ts)
                X_regresses.append(X_regress)
                #ts_std = (ts - ts.mean()) / ts.std()
                #ts_std = xr.DataArray(ts_std, dims=['time'])
                #results = tools.linregress_xarray(X_regress, ts_std, null_hypo=0)
                #slope = results['slope']
                #corr = results['correlation']
                #pvalues = results['pvalues']
                #regresses.append(slope)
                #corrs.append(corr)
                #regress_mean[years]=xr.concat(regresses,dim='en').mean(dim='en')
                #corr_mean[years]=xr.concat(corrs,dim='en').mean(dim='en')
        X_temp=xr.concat(X_regresses,dim='time')
        Y_temp=xr.DataArray(tss,dims=['time'])
        results = tools.linregress_xarray(X_temp, Y_temp, null_hypo=0)
        slope = results['slope']
        corr = results['correlation']
        pvalue = results['pvalues']
        slopes[years]=slope; corrs[years]=corr; pvalues[years]=pvalue

    reload(tools)
    ### Start the regression plotting
    shading_grids=[corrs[years] for years in years_fit]
    hatching_grids=[pvalues[years] for years in years_fit]
    row=2; col=int(len(shading_grids)/2.0)
    grid = row*col
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF', '#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    shading_level_grid = [np.linspace(-100,100,11)] * grid
    shading_level_grid = [np.linspace(-0.8,0.8,11)] * grid
    shading_level_grid = [np.linspace(-0.4,0.4,11)] * grid
    contour_grids = shading_grids; contour_clevels = shading_level_grid
    clabels_row = ['']*grid
    top_title = [''] * col
    left_title = [''] * row
    leftcorner_text= [str(i[0])+'-'+str(i[-1]) for i in years_fit]
    projection=ccrs.NorthPolarStereo(); xsize=1.5; ysize=1.5
    xlims = [-180,180]
    ylims = (55,90)
    pval_hatches = [[[0, 0.01, 1000], ['XX', None]]] * grid 
    pval_hatches = [[[0, 0.01, 1000], [None, '..']]] * grid 
    matplotlib.rcParams['hatch.linewidth']=1; matplotlib.rcParams['hatch.color']='lightgray'
    fill_continent=False
    # Get the region boxes
    Ulat1, Ulat2, Ulon1, Ulon2 = 55,75,40,120 # urals
    region_boxes = [tools.create_region_box(Ulat1, Ulat2, Ulon1, Ulon2)] * grid
    Ilat1, Ilat2, Ilon1, Ilon2 = 65,85,-50,20 # Iceland
    region_boxes_extra = [tools.create_region_box(Ilat1, Ilat2, Ilon1, Ilon2)] * grid
    ###
    plt.close()
    fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
    tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row, top_titles=top_title, 
                left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=True,
                region_boxes=region_boxes, region_boxes_extra=region_boxes_extra, region_box_extra_color='red',leftcorner_text=leftcorner_text,
                ylim=ylims, xlim=xlims, quiver_grids=None, colorbar=False,
                pval_map=hatching_grids, pval_hatches=pval_hatches, fill_continent=fill_continent, coastlines=False,
                contour_map_grids=contour_grids, contour_clevels=contour_clevels, pltf=fig, ax_all=ax_all.flatten())
    import cartopy.feature as cf
    for ax in ax_all.flatten():
        coast = cf.GSHHSFeature(scale='coarse',edgecolor='gray',linewidth=1)
        ax.add_feature(coast)
    ### Setup the coloar bar
    #cba = fig.add_axes([0.91, 0.3, 0.02, 0.4])
    cba = fig.add_axes([0.2, 0.03, 0.6, 0.03])
    cNorm  = matplotlib.colors.Normalize(vmin=shading_level_grid[0][0], vmax=shading_level_grid[0][-1])
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
    #cticks = yticklabels = [round(i,1) for i in shading_level_grid[0] if i!=0]
    cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, norm=cNorm, orientation='horizontal',
                extend='both')
    #cb1.ax.set_yticklabels(cticks, fontsize=10)
    cb1.set_label('Correlation', fontsize=10, rotation=0, y=-0.05, labelpad=1)
    fig_name = 'SLP_regression_onto_icegrowth'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.15, hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)
    
