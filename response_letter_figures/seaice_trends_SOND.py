import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
from scipy import stats

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools; reload(tools)

if __name__ == "__main__":

    vars = ['obs1_sic_en','CESM2_sic_en']
    ensembles=[[''],range(1,51)]

    st_yr=1980; end_yr=2022
    var_ratio=100
    trends={var:{} for var in vars}
    climatologies={var:{} for var in vars}
    trends_mean={}
    climatology_mean={}
    for i, var in enumerate(vars):
        for en in ensembles[i]:
            print(var,en)
            data = tools.read_data(var+str(en),months='all',slicing=False).sel(latitude=slice(25,90))*var_ratio
            mons=[9,10,11,12]; data=data.sel(time=slice('%s-09-01'%str(st_yr).zfill(4), '%s-12-31'%str(end_yr).zfill(4)))
            #mons=[9]; data=data.sel(time=slice('%s-09-01'%str(st_yr).zfill(4), '%s-09-30'%str(end_yr).zfill(4)))
            mon_mask = data.time.dt.month.isin(mons)
            data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
            # Compute the climatology
            climatology = data.mean(dim='time')
            # Compute the linear trend
            x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
            y=data; ymean=y.mean(dim='time')
            trend = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 # per deacde
            # Save the trend
            trends[var][en]=trend.compute()
            climatologies[var][en]=climatology.compute()
        trends_mean[var]=xr.concat([trends[var][en] for en in ensembles[i]],dim='en').mean(dim='en')
        climatology_mean[var]=xr.concat([climatologies[var][en] for en in ensembles[i]],dim='en').mean(dim='en')

    ### Starting the plotting
    plt.close()
    projection=ccrs.NorthPolarStereo(); xsize=2; ysize=2
    # ax1 (left) is the sea ice lossing map
    shading_grids = [trends_mean[var] for var in vars]
    contour_map_grids = [climatology_mean[var] for var in vars]
    contour_clevels = [[15]]*len(shading_grids)
    row=1; col=2; grid=row*col
    #mapcolors = ['#fddbc7','#FFFFFF','#FFFFFF','#d1e5f0','#4292c6','#2171b5','#08519c'][::-1]
    mapcolors = ['#d1e5f0','#6baed6','#2171b5','#08519c'][::-1]
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    shading_level_grid = [-25,-20,-15,-10,-5,0,5,10]
    shading_level_grid = [[-25,-20,-15,-10,-5]]*len(shading_grids)
    shading_level_grid = [[-18,-14,-10,-6,-2]]*len(shading_grids)
    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    region_boxes = None
    ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Seas - adjusted)
    region_boxes = [tools.create_region_box(ilat1, ilat2, ilon1, ilon2)]*2
    leftcorner_text=None
    freetext=['NSIDIC','CESM2']
    freetext_pos=[(-0.01,1.03)]*2
    xlim=[-180,180]; ylim=(60,90)
    tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row, top_titles=top_title, 
                    left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=False,
                    region_boxes=region_boxes, shading_extend='neither', freetext=freetext, freetext_pos=freetext_pos,
                    leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,
                    pval_map=None, pval_hatches=None, fill_continent=True, coastlines=False,
                    contour_map_grids=contour_map_grids, contour_clevels=contour_clevels, set_xylim=None,
                    set_extent=True, transpose=False,contour_lw=0.6)
    fig_name = 'seaice_trends_SOND'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.3) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.001)
