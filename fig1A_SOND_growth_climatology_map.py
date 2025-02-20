import xarray as xr
import numpy as np
import ipdb
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime as dt
import xesmf as xe

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools

if __name__ == "__main__":

    sic_vars=['cdr_seaice_conc_monthly_0.25x0.25','CESM2_sicraw_en']
    sic_ratios=[100,100]
    ensembles=[[''],range(41,43)]
    ensembles=[[''],range(41,51)]
    #years_sel=[range(2001,2021),range(2001,2021)]
    years_sel=[range(1981,2021),range(1981,2021)]

    sep_ices = {var:[] for var in sic_vars}
    dec_ices= {var:[] for var in sic_vars}
    mar_ices= {var:[] for var in sic_vars}
    ice_growths= {var:[] for var in sic_vars}
    for i, sic_var in enumerate(sic_vars):
        for en in ensembles[i]:
            print(sic_var,en)
            ## Read the data
            data = tools.read_data(sic_var+str(en), months='all', slicing=False, limit_lat=False, reverse_lat=False) * sic_ratios[i]
            if 'CESM2' in sic_var: ## Regrid that to higher resolution for CESM data
                data = data.rename({"TLON": "lon", "TLAT": "lat"})
                data = data.drop('ULAT').drop('ULON')
                lats = np.arange(35,90,0.25); lons=np.arange(-179.5,180,0.25)
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(data.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                data = regridder(data)
            ## Compuate the data
            data_regrid = data.compute()
            for year in years_sel[i]:
                # Constrain the data range at the beginning (useful for obs sic)
                data=data_regrid.sel(time=slice('%s-01-01'%year,'%s-12-31'%year))
                ## Extract March
                mar_mask= data.time.dt.month.isin([3])
                mar_ice=data.sel(time=mar_mask).isel(time=0)
                mar_ices[sic_var].append(mar_ice)
                ## Extract December
                dec_mask = data.time.dt.month.isin([12])
                dec_ice=data.sel(time=dec_mask).isel(time=0)
                dec_ices[sic_var].append(dec_ice)
                ## Extract September
                sep_mask= data.time.dt.month.isin([9]) # Use September
                sep_ice=data.sel(time=sep_mask).isel(time=0)
                sep_ices[sic_var].append(sep_ice)
                ## Do sea ice grwoth
                #ice_growths[sic_var].append(dec_ice-sep_ice)
                ice_growths[sic_var].append(dec_ice-sep_ice)

    if True: # Climatology
        shading_grids=[xr.concat(ice_growths[var],dim='en_year').mean(dim='en_year') for var in sic_vars]
        shading_grids=[tools.map_fill_white_gap(i) for i in shading_grids]
        #shading_grids_std=[xr.concat(ice_growths[var],dim='en_year').std(dim='en_year') for var in sic_vars]
        contour_map_grids=[xr.concat(sep_ices[var],dim='en_year').mean(dim='en_year') for var in sic_vars]
        contour_map_grids=[tools.map_fill_white_gap(i) for i in contour_map_grids]
        contour1_map_grids=[xr.concat(dec_ices[var],dim='en_year').mean(dim='en_year') for var in sic_vars]
        contour1_map_grids=[tools.map_fill_white_gap(i) for i in contour1_map_grids]
        contour2_map_grids=[xr.concat(mar_ices[var],dim='en_year').mean(dim='en_year') for var in sic_vars]
        contour2_map_grids=[tools.map_fill_white_gap(i) for i in contour2_map_grids]

    ### Plot the maps
    # Set the grid
    row, col = 1, len(shading_grids)
    grid = row*col
    mapcolors = ['#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b'] 
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    # Set shading grids
    shading_level_grid = [np.linspace(-50,50,11)]*grid
    shading_level_grid = [np.linspace(-80,80,11)]*grid
    shading_level_grid = [np.linspace(-10,10,11)]*grid
    shading_level_grid = [np.linspace(0,90,7)]*grid
    # Set contour grids
    contour_clevels_grids = [[15]] * grid
    clabels_row = ['']*grid
    top_title = ['']*col
    left_title = ['']*row
    ind_titles = None
    leftcorner_text=[str(years[0])+'-'+str(years[-1]) for years in years_sel]
    leftcorner_text=['Observations: '+str(years_sel[0][0])+'-'+str(years_sel[0][-1]),'CESM2: '+str(years_sel[0][0])+'-'+str(years_sel[0][-1])]
    projection=ccrs.Orthographic(central_longitude=0, central_latitude=90); xsize=2.5; ysize=2.5
    xlims = [-180,180]
    ylims = (67,90)
    ylims = (52,90)
    ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Sea)
    region_boxes = None
    region_boxes = [tools.create_region_box(ilat1, ilat2, ilon1, ilon2)]*2 + [None]*10
    xylims = None
    #########
    plt.close()
    fig, ax_all = plt.subplots(1,len(shading_grids),figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
    tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row,
            top_titles=top_title, left_titles=left_title,ind_titles=ind_titles, projection=projection,
            xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,ylim=ylims,xlim=xlims,set_xylim=xylims,
            fill_continent=False, colorbar=True,contour_map_grids=contour_map_grids,contour_clevels=contour_clevels_grids,
            leftcorner_text=leftcorner_text,leftcorner_text_pos=(0.02,0.08),box_outline_gray=True, pltf=fig, ax_all=ax_all, contour_lw=0.5,coastlines=False)
    import cartopy
    import cartopy.feature as cfeature
    if True: # Add the contour as Dec ice
        for i, ax in enumerate(ax_all):
            lons=contour1_map_grids[i].longitude.values; lats=contour1_map_grids[i].latitude.values
            csf=ax.contour(lons,lats,contour1_map_grids[i],[15],colors='black', linewidths=1, transform=ccrs.PlateCarree())
    if True: # Add the contour as Mar ice
        for i, ax in enumerate(ax_all):
            lons=contour2_map_grids[i].longitude.values; lats=contour2_map_grids[i].latitude.values
            csf=ax.contour(lons,lats,contour2_map_grids[i],[15],colors='black', linewidths=2, transform=ccrs.PlateCarree())
    for i, ax in enumerate(ax_all):
        #ax.add_feature(cfeature.OCEAN,facecolor=(0.5,0.5,0.5), zorder=100)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', 
                                facecolor='lightgrey'), zorder=0) # Can be 110m for filling
    ######
    fig_name = 'fig1A_ice_growth_map'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

    if False: # Plot the standard deviation of sea ice growth
        ### Plot the maps
        # Set the grid
        row, col = 1, len(shading_grids_std)
        grid = row*col
        mapcolors = ['#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b'] 
        cmap= matplotlib.colors.ListedColormap(mapcolors)
        mapcolor_grid = [cmap] * grid
        # Set shading grids
        shading_level_grid = [np.linspace(0,90,7)]*grid
        shading_level_grid = [np.linspace(0,40,7)]*grid
        #
        clabels_row = ['']*grid
        top_title = ['']*col
        left_title = ['']*row
        ind_titles = None
        leftcorner_text=[str(years[0])+'-'+str(years[-1]) for years in years_sel]
        leftcorner_text=['Obs: '+str(years_sel[0][0])+'-'+str(years_sel[0][-1]),'CESM2: '+str(years_sel[0][0])+'-'+str(years_sel[0][-1])]
        projection=ccrs.Orthographic(central_longitude=0, central_latitude=90); xsize=2; ysize=2
        xlims = [-180,180]
        ylims = (67,90)
        ylims = (52,90)
        ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Sea)
        region_boxes = None
        region_boxes = [tools.create_region_box(ilat1, ilat2, ilon1, ilon2)]*2 + [None]*10
        xylims = None
        #########
        plt.close()
        fig, ax_all = plt.subplots(1,len(shading_grids_std),figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
        tools.map_grid_plotting(shading_grids_std, row, col, mapcolor_grid, shading_level_grid, clabels_row,
                top_titles=top_title, left_titles=left_title,ind_titles=ind_titles, projection=projection,
                xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,ylim=ylims,xlim=xlims,set_xylim=xylims,
                fill_continent=True, colorbar=True,contour_map_grids=None,contour_clevels=None,
                leftcorner_text=leftcorner_text,box_outline_gray=True, pltf=fig, ax_all=ax_all, contour_lw=0.5,coastlines=False)
        ######
        fig_name = 'figS1_ice_growth_standard_deviation'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)
