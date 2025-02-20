import xarray as xr
import datetime as dt
import numpy as np
import ipdb
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import cartopy.crs as ccrs
from datetime import date
import matplotlib.pyplot as plt
from importlib import reload
import multiprocessing
import scipy

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools
import fig3_fig4_low_high_ice_samples as f3f4

if __name__ == "__main__":

    reload(f3f4)

    if False:
        years_fit = [range(1940,1960),range(1960,1980)] # testing
        grid=len(years_fit)

    if False: ## Figure S7A
        sic_var='CESM2_psl_en'; sic_ratio=1 
        shading_var='CESM2_sic_en'; shading_var_ratio=100 
        contour_var1='CESM2_psl_en'; contour_var1_ratio=1
        contour_var2=None; contour_var2_ratio=None
        contour_var3=None; contour_var3_ratio=None 
        spatial_plotting=True; boxplot_plotting=False
        set_top_title=True
        set_region_box=True
        contour1_level_grids = [np.linspace(-1000,1000,13)]*grid
        shading_level_grids= [np.linspace(-90,90,13)]*grid
        fig_add='S7A'
        ylims = (60,90)

    if False: ## Figure 3C
        sic_var='intCESM2_sic_en'; sic_ratio=100 # Default (new - using initial sea ice for picking)
        shading_var='CESM2_sicdynam_en'; shading_var_ratio=1
        contour_var1=None; contour_var1_ratio=None
        contour_var2=None; contour_var2_ratio=None
        contour_var3=None; contour_var3_ratio=None
        spatial_plotting=True; boxplot_plotting=False
        ylims = (67,90)

    if False: ## Figure 3B
        sic_var='intCESM2_sic_en'; sic_ratio=100 # Default (new - using initial sea ice for picking)
        shading_var='CESM2_sicthermo_en'; shading_var_ratio=1
        contour_var1=None; contour_var1_ratio=None
        contour_var2=None; contour_var2_ratio=None
        contour_var3=None; contour_var3_ratio=None
        spatial_plotting=True; boxplot_plotting=False
        ylims = (67,90)

    if False: ## Default (Figure 3A)
        sic_var='intCESM2_sic_en'; sic_ratio=100 # Default (new - using initial sea ice for picking)
        shading_var='CESM2_sic_en'; shading_var_ratio=100 # This is the sea ice growth
        contour_var1=None; contour_var1_ratio=None
        contour_var2='intCESM2_sic_en'; contour_var2_ratio=100 # contour 2 is particular for Sep sea ice state
        contour_var3='endCESM2_sic_en'; contour_var3_ratio=100  # contour 3 is particular for Dec sea ice state
        fig_add='3A'
        set_top_title=True
        set_region_box=False
        shading_level_grids= [np.linspace(-90,90,13)]*grid # For SIC growth
        spatial_plotting=True; boxplot_plotting=False
        ylims = (67,90)
        contour1_level_grids=[None]*grid

    plot_fig4=False
    plot_fig4=True
    if plot_fig4: 
        years_fit = [range(1861,1881), range(1881,1901),range(1901,1921),range(1921,1941),range(1941,1961),range(1961,1981),range(1981,2001),
                    range(2001,2021),range(2021,2041),range(2041,2061),range(2061,2081),range(2081,2101)]  
        sic_var='intCESM2_sic_en'; sic_ratio=100 # Default (new - using initial sea ice for picking)
        if False: # 5-m ocean temp.
            shading_var='CESM2_otemp_en_sep'; shading_var_ratio=1
            contour_var1='CESM2_otemp_en_oct'; contour_var1_ratio=1
            contour_var2='CESM2_otemp_en_nov'; contour_var2_ratio=1
            contour_var3='CESM2_otemp_en_dec'; contour_var3_ratio=1
        elif True: # SST
            shading_var='CESM2_sst_en_sep'; shading_var_ratio=1
            contour_var1='CESM2_sst_en_oct'; contour_var1_ratio=1
            contour_var2='CESM2_sst_en_nov'; contour_var2_ratio=1
            contour_var3='CESM2_sst_en_dec'; contour_var3_ratio=1
        else: # TAS
            shading_var='CESM2_tas_en_sep'; shading_var_ratio=1
            contour_var1='CESM2_tas_en_oct'; contour_var1_ratio=1
            contour_var2='CESM2_tas_en_nov'; contour_var2_ratio=1
            contour_var3='CESM2_tas_en_dec'; contour_var3_ratio=1
        fig_add='4'
        spatial_plotting=False; boxplot_plotting=True

    pool = multiprocessing.Pool(len(years_fit))
    argus=[]
    for years in years_fit:
        argu=(years, sic_var, sic_ratio, shading_var, shading_var_ratio, contour_var1, contour_var1_ratio,
                                        contour_var2, contour_var2_ratio, contour_var3, contour_var3_ratio, fig_add)
        argus.append(argu)
    if True: # Not debug model
        results = pool.map(f3f4.read_samples, argus)
        print("%s:Finishing rading all years: %s"%(fig_add,years_fit))
    else:# debug mode
        results=[]
        for argu in argus:
            #result=f3f4.read_samples((argu[0],argu[1],argu[2],argue[3],argue[4],argu[5],argu[6],argu[7]))
            result=f3f4.read_samples(argu)
            results.append(result)

    # Sampels before taking average (used for figure 5)
    shading_lows={}; shading_highs={} 
    contour1_lows={}; contour1_highs={} 
    contour2_lows={}; contour2_highs={} 
    contour3_lows={}; contour3_highs={} 
    # Samples after taking average (but in low and high seperately)
    shading_low_grids,shading_high_grids=[],[]
    contour1_low_grids,contour1_high_grids=[],[]
    contour2_low_grids,contour2_high_grids=[],[]
    contour3_low_grids,contour3_high_grids=[],[]
    # Taking the difference between low and high samples
    shading_grids = []
    shading_grids_pvalues = []
    contour1_map_grids=[]
    contour2_map_grids=[]
    contour3_map_grids=[]
    no=0
    for years in years_fit:
        sic_idx,shading_map,contour1_map,contour2_map,contour3_map=results[no]
        no=no+1
        ## 
        sic_idx = np.array(sic_idx)
        ## Pick sampels with smalli and large SIC growth
        high_growth=np.percentile(sic_idx,90); high_mask=(sic_idx>high_growth).nonzero()[0]
        low_growth=np.percentile(sic_idx,10); low_mask=(sic_idx<low_growth).nonzero()[0]
        if False: # Set the low samples as the climatology in that 20-year period (==> high growth minus climatology)
            #low_growth=np.percentile(sic_idx,100); low_mask=(sic_idx<=low_growth).nonzero()[0]
            #high_growth=np.percentile(sic_idx,100); high_mask=(sic_idx<=high_growth).nonzero()[0]
            pass
        ## Start with the shading maps
        shading_high_temp = xr.concat([shading_map[i] for i in high_mask],dim='samples')
        shading_low_temp = xr.concat([shading_map[i] for i in low_mask],dim='samples')
        # Save the shading grids before taking the mean
        shading_lows[years]=shading_low_temp
        shading_highs[years]=shading_high_temp
        # Take the average
        shading_high = shading_high_temp.mean(dim='samples')
        shading_low = shading_low_temp.mean(dim='samples')
        shading_high_grids.append(shading_high)
        shading_low_grids.append(shading_low)
        ## Set pvalues
        pvalues=scipy.stats.ttest_ind(shading_high_temp,shading_low_temp,axis=0).pvalue
        pvalues=shading_high.copy(data=pvalues,deep=True)
        shading_grids_pvalues.append(pvalues)
        if True: # Do the difference between high and low
            #shading_grids.append(shading_high-shading_low)
            shading_grids.append(shading_low-shading_high)
            #ipdb.set_trace()
        else: # Don't do the difference for shading (showing the climatology)
            shading_grids.append(shading_high)
            #shading_grids.append(shading_low)
        ## And then contour maps
        if not contour_var1 is None: # For SLP (we need diff)
            contour1_high_temp = xr.concat([contour1_map[i] for i in high_mask],dim='samples')
            contour1_low_temp = xr.concat([contour1_map[i] for i in low_mask],dim='samples')
            # Save before taking the mean
            contour1_highs[years]=contour1_high_temp
            contour1_lows[years]=contour1_low_temp
            # Save after taking the mean
            contour1_high = contour1_high_temp.mean(dim='samples')
            contour1_low = contour1_low_temp.mean(dim='samples')
            contour1_high_grids.append(contour1_high)
            contour1_low_grids.append(contour1_low)
             # Do the difference 
            contour1_map_grids.append(tools.map_fill_white_gap(contour1_low-contour1_high))
        else:
            contour1_map_grids=None
        if not contour_var2 is None: # For Oct sea ice 
            contour2_high_temp = xr.concat([contour2_map[i] for i in high_mask],dim='samples')
            contour2_low_temp = xr.concat([contour2_map[i] for i in low_mask],dim='samples')
            # Save before taking the mean
            contour2_highs[years]=contour2_high_temp
            contour2_lows[years]=contour2_low_temp
            # Average and save
            contour2_high = contour2_high_temp.mean(dim='samples')
            contour2_low = contour2_low_temp.mean(dim='samples')
            contour2_high_grids.append(contour2_high)
            contour2_low_grids.append(contour2_low)
            # Difference
            #contour2_map_grids.append(tools.map_fill_white_gap(contour2_low-contour2_high))
            contour2_map_grids.append(contour2_low-contour2_high)
        if not contour_var3 is None: # For Dec sea ice
            contour3_high_temp = xr.concat([contour3_map[i] for i in high_mask],dim='samples')
            contour3_low_temp = xr.concat([contour3_map[i] for i in low_mask],dim='samples')
            # Save before taking the mean
            contour3_highs[years]=contour3_high_temp
            contour3_lows[years]=contour3_low_temp
            # Average and save
            contour3_high = contour3_high_temp.mean(dim='samples')
            contour3_low = contour3_low_temp.mean(dim='samples')
            contour3_high_grids.append(contour3_high)
            contour3_low_grids.append(contour3_low)
            # Difference
            contour3_map_grids.append(tools.map_fill_white_gap(contour3_low-contour3_high))
    ###
    ### Start Plotting the spatial maps
    if spatial_plotting: 
        row, col = 1, len(years_fit)
        grid = row*col
        #####
        shading_grids=[i+0.00001 for i in shading_grids]
        mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF',
                     '#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b'] # 8 colors
        mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff',
                    '#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b'] # 10 colors
        cmap= matplotlib.colors.ListedColormap(mapcolors)
        #cmap= 'coolwarm'
        mapcolor_grid = [cmap] * grid
        clabels_row = ['']*grid
        left_title = ['']*row
        ind_titles = None
        leftcorner_text=None
        if set_top_title:
            top_title=[str(yr[0])+ ' to ' + str(yr[-1]) for yr in years_fit]
        else:
            top_title=['']*grid
        projection=ccrs.Orthographic(central_longitude=0, central_latitude=90); xsize=2; ysize=2
        xlims = [-180,180]
        #ylims = (67,90)
        if set_region_box:
            ilat1=84; ilat2=85; ilon1=88; ilon2=100 # According to smaples picked in Figure 4
            region_boxes = [tools.create_region_box(ilat1, ilat2, ilon1, ilon2)] * grid
            region_boxes_extra = None
        else:
            region_boxes = None
            region_boxes_extra = None
        matplotlib.rcParams['hatch.linewidth']=0.2; matplotlib.rcParams['hatch.color'] = 'lightgray'
        pval_hatches = [[[0, 0.01, 1000], [None, '..']]] * grid # Mask the insignificant regions
        #pval_hatches = [[[0, 0.01, 1000], ['..', None]]] * grid # Mask the insignificant regions
        xylims = None
        coastlines=True
        coastlines=False
        plt.close()
        fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
        tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grids, clabels_row,
                top_titles=top_title, left_titles=left_title,ind_titles=ind_titles, projection=projection,
                xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,region_boxes_extra=region_boxes_extra,
                ylim=ylims,xlim=xlims, region_box_color='black',
                set_xylim=xylims,fill_continent=True,coastlines=coastlines,colorbar=True,contour_map_grids=contour1_map_grids,
                contour_clevels=contour1_level_grids, contour_lw=1,
                leftcorner_text=leftcorner_text,box_outline_gray=True,indiv_colorbar=[False]*grid,pval_map=shading_grids_pvalues,pval_hatches=pval_hatches,
                pltf=fig,ax_all=ax_all)
        #ax.scatter(95,85,marker='X',s=20,color='black',transform=ccrs.PlateCarree(central_longitude=0))
        if not contour_var2 is None: # Usually the Sep sea ice
            for i, ax in enumerate(ax_all):
                lons=contour2_low_grids[i].longitude.values; lats=contour2_low_grids[i].latitude.values
                csf=ax.contour(lons,lats,contour2_low_grids[i],[15],colors='cyan',linewidths=1,transform=ccrs.PlateCarree())
                csf=ax.contour(lons,lats,contour2_high_grids[i],[15],colors='darkblue', linewidths=1,transform=ccrs.PlateCarree())
        if not contour_var3 is None: # Usually the Dec sea ice
            for i, ax in enumerate(ax_all):
                lons=contour3_low_grids[i].longitude.values; lats=contour3_low_grids[i].latitude.values
                csf=ax.contour(lons,lats,contour3_low_grids[i],[15],colors='cyan',linewidths=2,transform=ccrs.PlateCarree())
                csf=ax.contour(lons,lats,contour3_high_grids[i],[15],colors='darkblue', linewidths=2,transform=ccrs.PlateCarree())
        # Save figures
        fig_name = 'fig3_physical_processes_composite_%s'%fig_add
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.05, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

    ### Plotting the boxplot (raw temperature change over time) - Figure 4
    if boxplot_plotting: 
        plt.close()
        fig,ax1=plt.subplots(1,1,figsize=(8,1.5))
        # select a point for averaging
        #ilat1=84; ilat2=85; ilon1=88; ilon2=100
        ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Seas - adjusted)
        if False: # Min all months together
            shading_samples_low = []
            shading_samples_high = []
            for yf in years_fit:
                aa=shading_lows[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                bb=contour1_lows[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                cc=contour2_lows[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                dd=contour3_lows[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                shading_samples_low.append(np.concatenate((aa,bb,cc,dd)))
                aa=shading_highs[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                bb=contour1_highs[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                cc=contour2_highs[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                dd=contour3_highs[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                shading_samples_high.append(np.concatenate((aa,bb,cc,dd)))
        else: # Individual months (Sep, Oct, Nov, Dec)
            sep_lows, oct_lows, nov_lows, dec_lows = [],[],[],[]
            sep_highs, oct_highs, nov_highs, dec_highs = [],[],[],[]
            percentage_lows, percentage_highs = [],[]
            for yf in years_fit:
                # Low samples
                sep_low=shading_lows[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                sep_lows.append(sep_low)
                oct_low=contour1_lows[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                oct_lows.append(oct_low)
                nov_low=contour2_lows[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                nov_lows.append(nov_low)
                dec_low=contour3_lows[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                dec_lows.append(dec_low)
                #all_low=np.concatenate([sep_low,oct_low,nov_low,dec_low])
                all_low=np.concatenate([oct_low,nov_low,dec_low]) # Only OND
                #percentage_low=((all_low<-1.7) & (all_low>-1.85)).sum() / all_low.size*100
                percentage_low=(all_low<-1.8).sum() / all_low.size*100
                percentage_lows.append(percentage_low)
                # High samples
                sep_high=shading_highs[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                sep_highs.append(sep_high)
                oct_high=contour1_highs[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                oct_highs.append(oct_high)
                nov_high=contour2_highs[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                nov_highs.append(nov_high)
                dec_high=contour3_highs[yf].sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2)).mean(dim='latitude').mean('longitude')
                dec_highs.append(dec_high)
                #all_high=np.concatenate([sep_high,oct_high,nov_high,dec_high])
                all_high=np.concatenate([oct_high,nov_high,dec_high])
                #percentage_high=((all_high<-1.7) & (all_high>-1.8)).sum() / all_high.size*100
                percentage_high=(all_high<-1.8).sum() / all_high.size*100
                percentage_highs.append(percentage_high)
        x=np.arange(0,len(years_fit))
        xadjust=0.02
        minor_x=0.07
        ms=2
        width=1
        ## For low Sep sea ice samples (from Sep to Dec)
        xs=[x-xadjust-minor_x*4,x-xadjust-minor_x*3,x-xadjust-minor_x*2,x-xadjust-minor_x*1]
        bpcolor='orange'
        lows=[sep_lows,oct_lows,nov_lows,dec_lows]
        months=['S','O','N','D']
        mon_text_size=6
        for i, low in enumerate(lows):
            medians=[j.median().item() for j in low]
            ymins=[j.min().item() for j in low]
            ymaxs=[j.max().item() for j in low]
            yerr_min = np.array(medians)-np.array(ymins)
            yerr_max = np.array(ymaxs)-np.array(medians)
            ax1.errorbar(xs[i],medians,yerr=[yerr_min,yerr_max],color=bpcolor,fmt='o',elinewidth=width,ms=ms)
            for k, xxx in enumerate(xs[i]): # Plot the "S", "O", "N", "D" markers
                ax1.annotate(months[i],xy=(xs[i][k]-0.015,ymaxs[k]+0.5), xycoords='data', fontsize=mon_text_size, color=bpcolor)
        ypos=-2.15
        #ypos=-38
        percentage_fs=6
        for i, xx in enumerate(xs[0]): # Plot the low percentages
            ax1.annotate(str(int(percentage_lows[i]))+'%',xy=(xx-0.02,ypos), xycoords='data', fontsize=percentage_fs, color=bpcolor)
        ## For high Sep sea ice samples (from Sep to Dec)
        xadjust=0.3
        xs=[x+xadjust-minor_x*4,x+xadjust-minor_x*3,x+xadjust-minor_x*2,x+xadjust-minor_x*1]
        bpcolor='royalblue'
        highs=[sep_highs,oct_highs,nov_highs,dec_highs]
        for i, high in enumerate(highs):
            medians=[j.median().item() for j in high]
            ymins=[j.min().item() for j in high]
            ymaxs=[j.max().item() for j in high]
            yerr_min = np.array(medians)-np.array(ymins)
            yerr_max = np.array(ymaxs)-np.array(medians)
            ax1.errorbar(xs[i],medians,yerr=[yerr_min,yerr_max],color=bpcolor,fmt='o',elinewidth=width,ms=ms)
            for k, xxx in enumerate(xs[i]): 
                ax1.annotate(months[i],xy=(xs[i][k]-0.015,ymaxs[k]+0.5), xycoords='data', fontsize=mon_text_size, color=bpcolor)
        for i, xx in enumerate(xs[0]): # Plot the high percetges
            ax1.annotate(str(int(percentage_highs[i]))+'%',xy=(xx+0.02,ypos), xycoords='data', fontsize=percentage_fs, color=bpcolor)
        # Set ylim
        if 'tas' in shading_var:
            ax1.set_ylim(-40,8)
        else:
            ax1.set_ylim(-3,5)
        # Set xticks
        ax1.set_xticks(x)
        xticklabels=[str(years[0])+ '\n' + 'to' +'\n' + str(years[-1]) for years in years_fit]
        ax1.set_xticklabels(xticklabels,size=9)
        # Add a horizonal line
        ax1.axhline(y=-1.8, color='lightgray', linestyle='-', linewidth=2, label='Freezing temperature (-1.8 °C)', zorder=-5)
        #ax1.fill_between([xs[0][0]-0.4,xs[0][-1]+0.3], -1.8, -1.7, alpha=0.3, fc='grey', label='-1.7 to -1.8 degree celsius')
        ax1.set_xlim([xs[0][0]-0.4,xs[0][-1]+0.3])
        if True: # Create fake legend for Figure 4
            ax1.errorbar([-100],[-100],yerr=[[1],[2]],color='orange',fmt='o',elinewidth=width,ms=ms,label='Low $ice_{sep}$ samples')
            ax1.errorbar([-100],[-100],yerr=[[1],[2]],color='royalblue',fmt='o',elinewidth=width,ms=ms,label='High $ice_{sep}$ samples')
            ax1.legend(bbox_to_anchor=(0.02,0.9),ncol=3,loc='lower left', frameon=False, columnspacing=1,
                        handletextpad=0.1, labelspacing=0.05, fontsize=9)
        ax1.set_ylabel("Surface air\ntemperature (°C)")
        # Set the frame
        for i in ['right', 'top']:
            ax1.spines[i].set_visible(False)
            ax1.tick_params(axis='x', which='both',length=2); ax1.tick_params(axis='y', which='both',length=2)
        ## Save
        fig_name = 'fig%s_otemp_boxplots_two_groups'%fig_add
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.05, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)


def read_samples(argus):

    ensembles=range(1,51)
    mons = [1,2,3]; season='JFM'
    mons = [10,11,12]; season='OND'
    mons = [9,10,11,12]; season='SOND'
    ### Setting what to pick for high and low samples
    ilat1=70; ilat2=88; ilon1=10; ilon2=140 # For SIC only (BKS+Laptev Seas - adjusted)
    Ulat1, Ulat2, Ulon1, Ulon2 = 55,70,30,90 # urals
    Ilat1, Ilat2, Ilon1, Ilon2 = 55,70,-45,-1 # Iceland

    years=argus[0]
    sic_var=argus[1]
    sic_ratio=argus[2]
    shading_var=argus[3]
    shading_var_ratio=argus[4]
    contour_var1=argus[5]
    contour_var1_ratio=argus[6]
    contour_var2=argus[7]
    contour_var2_ratio=argus[8]
    contour_var3=argus[9]
    contour_var3_ratio=argus[10]
    fig_add=argus[11]
    print(fig_add,years)

    sic_idx = []
    shading_map= []
    contour1_map = []
    contour2_map = []
    contour3_map = []
    vector1_map = []
    vector2_map = []
    for en in ensembles:
        for year in years:
            path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,sic_var,en,year)
            data=xr.open_dataset(path)['training']*sic_ratio
            if sic_var=='CESM2_psl_en':
                ural_data=data.sel(latitude=slice(Ulat1,Ulat2)).sel(longitude=slice(Ulon1,Ulon2))
                lats=ural_data.latitude.values; lons=ural_data.longitude.values
                ural_ts=ct.weighted_area_average(ural_data.values[np.newaxis,:,:],Ulat1,Ulat2,Ulon1,Ulon2,lons,lats)
                iceland_data=data.sel(latitude=slice(Ilat1,Ilat2)).sel(longitude=slice(Ilon1,Ilon2))
                lats=iceland_data.latitude.values; lons=iceland_data.longitude.values
                iceland_ts=ct.weighted_area_average(iceland_data.values[np.newaxis,:,:],Ilat1,Ilat2,Ilon1,Ilon2,lons,lats)
                #data_ts=iceland_ts.item()-ural_ts.item()
                data_ts=ural_ts.item()-iceland_ts.item()
            else: # Get the timeseries
                data=data.sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2))
                lats=data.latitude.values; lons=data.longitude.values
                data_ts=ct.weighted_area_average(data.values[np.newaxis,:,:],ilat1,ilat2,ilon1,ilon2,lons,lats)
            sic_idx.append(data_ts)
            ### Start doing shading, contour vars
            if not shading_var is None: 
                if shading_var==sic_var: # The sea ice grwoth (OND) as the shaading
                    shading_map.append(data)
                else:
                    path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,shading_var,en,year)
                    data=xr.open_dataset(path)['training'].squeeze().compute() * shading_var_ratio
                    shading_map.append(data)
            if not contour_var1 is None:
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,contour_var1,en,year)
                data=xr.open_dataset(path)['training'].squeeze().compute() * contour_var1_ratio
                contour1_map.append(data)
            if not contour_var2 is None:
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,contour_var2,en,year)
                data=xr.open_dataset(path)['training'].squeeze().compute() * contour_var2_ratio
                contour2_map.append(data)
            if not contour_var3 is None:
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,contour_var3,en,year)
                data=xr.open_dataset(path)['training'].squeeze().compute() * contour_var3_ratio
                contour3_map.append(data)

    return sic_idx, shading_map, contour1_map, contour2_map, contour3_map
