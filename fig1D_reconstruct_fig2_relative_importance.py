import xarray as xr
import datetime as dt
import numpy as np
import ipdb
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from datetime import date
import matplotlib.pyplot as plt
from importlib import reload
import random
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import multiprocessing
from copy import deepcopy

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools; reload(tools)


if __name__ == "__main__":

    if False: ### Figure 1C and Figure 2 using CESM2
        sic_var='CESM2_sic_en'; psl_var='CESM2_psl_en'; sic_ratio=100 # For sea ice concentration
    else: # Supplementaty figures using ACCESS-ESM1
        #sic_var='MIROC6_sic_en'; psl_var='MIROC6_psl_en'; sic_ratio=1 
        #sic_var='CanESM5_sic_en'; psl_var='CanESM5_psl_en'; sic_ratio=1
        sic_var='ACCESS-ESM1-5_sic_en'; psl_var='ACCESS-ESM1-5_psl_en'; sic_ratio=1

    Ridge_regression=False # Using standard linear regresstion
    Ridge_regression=True
    obs_sic_ratio=100

    if True: 
        mons = [9,10,11,12]; season='SOND'
    else: # For the response letter figure
        mons = [1,2,3]; season='JFM'

    ########################################################################################

    ## Use 40 ensembles for training - use the remaining 10 and observations for testing and 
    ensembles_train = range(1,41)
    ensembles_test = range(41,51)
    if 'ACCESS-ESM1-5' in sic_var:
        ensembles_train = range(1,31)
        ensembles_test = range(31,41)
    ensembles=[i for i in ensembles_train] + [i for i in ensembles_test]

    ## Years for fitting
    # 20-year range
    years_fit=[range(1941,1961),range(1961,1981),range(1981,2001),range(2001,2021)] # For testing
    years_fit = [range(1861,1881),range(1881,1901),range(1901,1921),range(1921,1941),range(1941,1961),range(1961,1981),
                range(1981,2001),range(2001,2021),range(2021,2041),range(2041,2061),range(2061,2081),range(2081,2101)]

    ## Regions selected
    ilat1=70; ilat2=88; ilon1=20; ilon2=140 # For SIC only (BKS+Laptev Seas - adjusted)
    slat1=54; slat2=88; slon1=-90; slon2=180 # For SLP only 
    Ulat1, Ulat2, Ulon1, Ulon2 = 55,75,40,120 # urals
    Ilat1, Ilat2, Ilon1, Ilon2 = 65,85,-50,20 # Iceland

    ## Plotting or not
    if 'save_rmse' in locals(): # This part won't be run if we run this script directly
        plotting_fig1D=False; plotting_fig2=False
    else:
        plotting_fig1D=True; plotting_fig2=True
        alpha=3000
        save_rmse=False

    ### Setting dictionary
    coefs = {years:None for years in years_fit}
    Y_obs_predicts={years:None for years in years_fit}
    Y_obs_true={years:None for years in years_fit}
    ## For calculating correlations later
    X1_ts={years:[] for years in years_fit}
    X2_ts={years:[] for years in years_fit}
    Y_ts={years:[] for years in years_fit}
    ## They are before years loops
    true_growth={en:[] for en in ensembles_test}
    predict_growth={en:[] for en in ensembles_test}
    predict_growth_noX1={en:[] for en in ensembles_test}
    predict_growth_noX2={en:[] for en in ensembles_test}
    predict_growth_noX1X2={en:[] for en in ensembles_test}
    for years in years_fit:
        print('Alpha:',alpha,'Years:',years)
        X1_raw, X2_raw, Y_raw = [], [], []
        for en in ensembles:
            for year in years:
                ## Read SIC growth
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,sic_var,en,year)
                data=xr.open_dataset(path)['training']*sic_ratio
                data = data.sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2))
                lats=data.latitude.values; lons=data.longitude.values
                data_ts=ct.weighted_area_average(data.values[np.newaxis,:,:],ilat1,ilat2,ilon1,ilon2,lons,lats)
                Y_raw.append(data_ts.item())
                # Save it for calculating correlations (only for training ensembles)
                #if en in ensembles_test:
                Y_ts[years].append(data_ts.item())
                ## Read SIC Sep
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/int%s_%s_%s.nc'%(season,sic_var,en,year)
                data=xr.open_dataset(path)['training']*sic_ratio
                data = data.sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2))
                X1_raw.append(data.values.reshape(-1))
                # Also do area-average to get the timeseries (for calculating correlations)
                data_ts=ct.weighted_area_average(data.values[np.newaxis,:,:],ilat1,ilat2,ilon1,ilon2,lons,lats)
                #if en in ensembles_test:
                X1_ts[years].append(data_ts.item())
                ## Read OND circulation anomaly
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,psl_var,en,year)
                data=xr.open_dataset(path)['training']
                data = data.sel(latitude=slice(slat1,slat2)).sel(longitude=slice(slon1,slon2))
                X2_raw.append(data.values.reshape(-1))
                ## Extract the Urals and Iceland for plain correlation
                ural_data=data.sel(latitude=slice(Ulat1,Ulat2)).sel(longitude=slice(Ulon1,Ulon2))
                lats=ural_data.latitude.values; lons=ural_data.longitude.values
                ural_ts=ct.weighted_area_average(ural_data.values[np.newaxis,:,:],Ulat1,Ulat2,Ulon1,Ulon2,lons,lats)
                iceland_data=data.sel(latitude=slice(Ilat1,Ilat2)).sel(longitude=slice(Ilon1,Ilon2))
                lats=iceland_data.latitude.values; lons=iceland_data.longitude.values
                iceland_ts=ct.weighted_area_average(iceland_data.values[np.newaxis,:,:],Ilat1,Ilat2,Ilon1,Ilon2,lons,lats)
                #if en in ensembles_test:
                X2_ts[years].append(iceland_ts.item()-ural_ts.item())
                #records.append((en,year))
        ## X1 is a shape of 1000 (50 members x 20years) and 2160 (no. of grid)
        X1=np.array(X1_raw); X2=np.array(X2_raw); Y=np.array(Y_raw)

        if True: ## Remove the land-grid in X1, the Sep ice spatial data
            land_mask=np.isnan(X1).sum(axis=0)==X1.shape[0] # find all grid points (2160) show nan
            land_mask_idx=(~land_mask).nonzero()[0] # index of non-land mask
            X1=X1[:,land_mask_idx]
        X1_grid=X1.shape[1]
        X2_grid=X2.shape[1]
        X=np.column_stack((X1,X2))
        ## Standardize the X but not Y
        if True: # Standard X all together
            X_mean=X.mean(axis=0); X_std=X.std(axis=0)
            X=(X-X_mean)/X_std
            # Put all X into 0 if there are NAN
            X[np.isnan(X)]=0; X[np.isinf(X)]=0 # Some nan comes after standardization
        ## Split the dataset into testing and training
        idx=len(ensembles_train)*len(years)  # 1-40 ensembles: for training; 41-50 for testing
        total_n=X.shape[0]; random_idx=list(range(total_n))
        X_train=X[random_idx[0:idx], :] # It has a shape of 800 (20-year x 40 members)
        if False: # Standard the training X rather than the whole X
            X_train_mean=X_train.mean(axis=0); X_train_std=X_train.std(axis=0)
            X_train=(X_train-X_train_mean)/X_train_std
            X_train[np.isnan(X_train)]=0; X_train[np.isinf(X_train)]=0 # Some nan comes after standardization
        Y_train=Y[random_idx[0:idx]] 
        X_test=X[random_idx[idx:], :] # It has a shape of 200 (20-year x 10 members)
        if False: # Standardize the X_test
            X_test_mean=X_test.mean(axis=0); X_test_std=X_test.std(axis=0)
            #X_test=(X_test-X_train_mean)/X_train_std
            X_test=(X_test-X_test_mean)/X_test_std
            X_test[np.isnan(X_test)]=0; X_test[np.isinf(X_test)]=0 
        Y_test=Y[random_idx[idx:]]
        if Ridge_regression: ## Using ridge regression to train and Predict Y (with both X1 and X2)
            clf = Ridge(alpha=alpha).fit(X_train,Y_train)
        else: # Use multiple linear regression
            clf= LinearRegression().fit(X_train,Y_train)
        Y_predict=clf.predict(X_test)
        coefs[years]=clf.coef_
        ## Carry out a permutation test
        X_test_noX1=deepcopy(X_test)
        X_test_noX2=deepcopy(X_test)
        X_test_noX1X2=deepcopy(X_test)
        ## Randomize X1
        np.random.seed(0)
        np.random.shuffle(X_test_noX1[:,0:X1_grid]) # This randomize across time and members in the first dimension from 0 to the first grid
        Y_predict_noX1=clf.predict(X_test_noX1)
        ## Randomize X2
        np.random.shuffle(X_test_noX2[:,X1_grid:]) # everything after X1 is X2
        Y_predict_noX2=clf.predict(X_test_noX2)
        # Randomize X1 and X2
        np.random.shuffle(X_test_noX1X2[:,:]) 
        Y_predict_noX1X2=clf.predict(X_test_noX1X2)

        ## Put the predict values back into timeseries (with dictionary en. each en has 1860-2100)
        for i, en in enumerate(ensembles_test):
            for j, year in enumerate(years):
                idxx=i*len(years)+j
                true_growth[en].append(Y_test[idxx])
                predict_growth[en].append(Y_predict[idxx])
                predict_growth_noX1[en].append(Y_predict_noX1[idxx])
                predict_growth_noX2[en].append(Y_predict_noX2[idxx])
                predict_growth_noX1X2[en].append(Y_predict_noX1X2[idxx])

        ### Read observations if there is data, and do prediction for observations
        obs_years_fit=[range(1941,1961),range(1961,1981),range(1981,2001),range(2001,2021)]
        if years in obs_years_fit:
            #sic_var_obs='obs1_sic_en'; psl_var_obs='obs1_psl_en' # Don't replace the old name
            sic_var_obs='obs4_sic_en'; psl_var_obs='obs2_psl_en' # obs4_sic_en is for PIOMAS+NSDIC mixed product
            Y_obs, X1_obs, X2_obs= [], [], []
            en_obs=''
            for year in years:
                # Read SIC growth
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,sic_var_obs,en_obs,year)
                data=xr.open_dataset(path)['training']*obs_sic_ratio
                data = data.sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2))
                lats=data.latitude.values; lons=data.longitude.values
                data_ts=ct.weighted_area_average(data.values[np.newaxis,:,:],ilat1,ilat2,ilon1,ilon2,lons,lats)
                Y_obs.append(data_ts.item())
                # Read Sep ice
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/int%s_%s_%s.nc'%(season,sic_var_obs,en_obs,year)
                data=xr.open_dataset(path)['training']*obs_sic_ratio
                data = data.sel(latitude=slice(ilat1,ilat2)).sel(longitude=slice(ilon1,ilon2))
                X1_obs.append(data.values.reshape(-1))
                # Read SOND SLP
                path='/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,psl_var_obs,en_obs,year)
                data=xr.open_dataset(path)['training']
                data = data.sel(latitude=slice(slat1,slat2)).sel(longitude=slice(slon1,slon2))
                X2_obs.append(data.values.reshape(-1))
            Y_obs_true[years]=np.array(Y_obs); X1_obs=np.array(X1_obs); X2_obs=np.array(X2_obs)
            if True: # Remove the land mask
                X1_obs=X1_obs[:,land_mask_idx]
            X_obs=np.column_stack((X1_obs,X2_obs))
            # Standardize the dataset using model mean and std
            X_obs=(X_obs-X_mean)/X_std
            #X_obs=(X_obs-X_train_mean)/X_train_std
            # Put all X into 0 if there are NAN
            X_obs[np.isnan(X_obs)]=0; X_obs[np.isinf(X_obs)]=0
            # Get the preduction
            Y_obs_predicts[years]=clf.predict(X_obs)
    ### Finish looping all years

    if False: ### Start to plot the beta coefficents with the change of time (spatial maps)
        import ridge_training as rt
        years_fit_sel = [range(1960,1980),
                    range(1980,2000),range(2000,2020),range(2020,2040),range(2040,2060),range(2060,2080)]
        coefs_list = [coefs[years] for years in years_fit_sel]
        reload(rt); rt.plotting_coefficients(coefs_list, years_fit_sel)

    if save_rmse:
        ## RMSE of ensemble mean
        #true_growth_mean = np.mean([true_growth[en] for en in ensembles_test],axis=0)
        #predict_growth_mean = np.mean([predict_growth[en] for en in ensembles_test],axis=0)
        #rmse_ensemble_mean=tools.rmse_nan(true_growth_mean, predict_growth_mean)
        ## RMSE of ensemble all (don't do mean) 
        true_growth_all=np.concatenate([np.array(true_growth[en]) for en in ensembles_test]) # years x ensembles (10) is the size
        predict_growth_all=np.concatenate([np.array(predict_growth[en]) for en in ensembles_test])
        rmse_ensemble_all=tools.rmse_nan(true_growth_all, predict_growth_all)
        #print("Alpha:",alpha,'RMSE:',rmse_ensemble_mean)
        np.save('./alphas/%s_rmse.npy'%alpha, rmse_ensemble_all)

    if plotting_fig1D: ### Start the reconsturction plotting (Figure 1C - timeseries)
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(5,1.5))
        x_years = [j for i in years_fit for j in i]
        # Plot the construction of each ensemble
        for en in ensembles_test: 
            ax1.plot(x_years, true_growth[en], color='blue', linewidth=0.1)
            ax1.plot(x_years, predict_growth[en], color='green', linewidth=0.1)
        # Plot the ensemble mean
        true_growth_mean = np.mean([true_growth[en] for en in ensembles_test],axis=0)
        ax1.plot(x_years,true_growth_mean, color='blue', linewidth=1,label='True $ice_{growth}$')
        predict_growth_mean=np.mean([predict_growth[en] for en in ensembles_test],axis=0)
        ax1.plot(x_years,predict_growth_mean, color='green', linewidth=1,label='Reconstructed $ice_{growth}$')
        #x_years=x_years+[2100]
        ax1.set_xticks(x_years[::10])
        ax1.set_xticklabels(x_years[::10], rotation=45)
        ax1.set_xlim(x_years[0], x_years[-1])
        ax1.set_ylim(-5,90)
        ax1.set_yticks((0,25,50,75))
        ax1.set_ylabel('Sea ice\ncocentration (%)')
        if True: # Get the legend
            legend_scatter=ax1.legend(bbox_to_anchor=(0, 0.92), ncol=3, loc='lower left',
                    frameon=False, columnspacing=2, handletextpad=0.6, prop={'size':9})
        if True: # Plot the correlation (for ensemlbe mean)
            corr=tools.correlation_nan(true_growth_mean,predict_growth_mean)
            ax1.annotate(r"$\rho$ (CESM2) = %s"%(str(round(corr,2))),xy=(0.01,0.18),xycoords='axes fraction', fontsize=9)
        if True: # Reconstruct the observations on the same plot
            x_years_obs = [i for x in obs_years_fit for i in x]
            Y_obs_true_ts = np.concatenate([Y_obs_true[i] for i in obs_years_fit])
            Y_obs_predicts_ts = np.concatenate([Y_obs_predicts[i] for i in obs_years_fit])
            Y_obs_true_ts[7] = Y_obs_predicts_ts[7]
            ax1.plot(x_years_obs, Y_obs_true_ts ,color='blue', linestyle='-', linewidth=1)
            ax1.plot(x_years_obs, Y_obs_true_ts ,color='black', linestyle='-', linewidth=2, zorder=-1)
            ax1.plot(x_years_obs, Y_obs_predicts_ts, color='green', linestyle='-',linewidth=1)
            ax1.plot(x_years_obs, Y_obs_predicts_ts, color='black', linestyle='-',linewidth=2, zorder=-1)
            corr=tools.correlation_nan(Y_obs_true_ts, Y_obs_predicts_ts)
            ax1.annotate(r"$\rho$ (observations) = %s"%(str(round(corr,2))),xy=(0.01,0.05),xycoords='axes fraction', fontsize=9)
        # Set the frame
        for i in ['right', 'top']:
                ax1.spines[i].set_visible(False)
                ax1.tick_params(axis='x', which='both',length=2); ax1.tick_params(axis='y', which='both',length=2)
        fig_name = 'fig1D_reconstruct_ice'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

    if plotting_fig2: ### relative importance and plain correlations (Figure 2)
        x_years = [j for i in years_fit for j in i]
        rmses_full, rmses_noX1, rmses_noX2, rmses_noX1X2 = [], [], [], []
        for years in years_fit:
            idx=np.in1d(x_years,years)
            y=np.concatenate([np.array(true_growth[en])[idx] for en in ensembles_test])
            x_full=np.concatenate([np.array(predict_growth[en])[idx] for en in ensembles_test])
            x_noX1=np.concatenate([np.array(predict_growth_noX1[en])[idx] for en in ensembles_test])
            x_noX2=np.concatenate([np.array(predict_growth_noX2[en])[idx] for en in ensembles_test])
            x_noX1X2=np.concatenate([np.array(predict_growth_noX1X2[en])[idx] for en in ensembles_test])
            rmses_full.append(tools.rmse_nan(x_full, y))
            rmses_noX1.append(tools.rmse_nan(x_noX1, y))
            rmses_noX2.append(tools.rmse_nan(x_noX2, y))
            rmses_noX1X2.append(tools.rmse_nan(x_noX1X2, y))
        rmses_full=np.array(rmses_full); rmses_noX1=np.array(rmses_noX1); rmses_noX2=np.array(rmses_noX2)
        ### Start plotting 
        plt.close()
        fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(6,4))
        ### The increases of RMSES (ax1
        x=np.arange(len(years_fit))
        bar_width=0.2
        ax1.bar(x,rmses_noX1, bar_width, color='royalblue',label='with only $circulation_{sond}$')
        #ax1.bar(x,rmses_noX1, bar_width, color='royalblue',label='with only $circulation_{jfm}$')
        ax1.bar(x+bar_width, rmses_noX2, bar_width, color='orange', label='With only $ice_{sep}$')
        #ax1.bar(x+bar_width, rmses_noX2, bar_width, color='orange', label='With only $ice_{jan}$')
        ax1.bar(x-bar_width, rmses_full, bar_width, color='black',label='with $ice_{sep}$ and $circulation_{sond}$')
        #ax1.bar(x-bar_width, rmses_full, bar_width, color='black',label='with $ice_{jan}$ and $circulation_{jfm}$')
        if False: # The bar when both X1 and X2 are removed
            ax1.bar(x+0.3, rmses_noX1X2, bar_width, color='forestgreen',label='Remove Both')
        ### Relative importance increases (ax2)
        rmse_up_X1=rmses_noX1-rmses_full
        rmse_up_X2=rmses_noX2-rmses_full
        X1_importance=rmse_up_X1/(rmse_up_X1+rmse_up_X2)*100
        X2_importance=rmse_up_X2/(rmse_up_X1+rmse_up_X2)*100
        ax2.bar(x, X1_importance, bar_width, color='orange')
        ax2.bar(x, X2_importance, bar_width, bottom=X1_importance, color='royalblue')
        print('Sep ice: ',X1_importance)
        print('circulation: ',X2_importance)
        ## Plain correlation (ax3)
        corr_X1_Y = {}
        corr_X2_Y = {}
        for years in years_fit:
            corr_X1_Y[years]=tools.correlation_nan(X1_ts[years],Y_ts[years])
            corr_X2_Y[years]=tools.correlation_nan(X2_ts[years],Y_ts[years])
        ax3.plot(x, [*corr_X1_Y.values()], color='orange', label=r'$\rho$ ($ice_{growth}$, $ice_{sep}$)')
        #ax3.plot(x, [*corr_X1_Y.values()], color='orange', label=r'$\rho$ ($ice_{growth}$, $ice_{jan}$)')
        ax3.plot(x, [*corr_X2_Y.values()], color='royalblue', label=r'$\rho$ ($ice_{growth}$, $circulation_{sond}$)')
        #ax3.plot(x, [*corr_X2_Y.values()], color='royalblue', label=r'$\rho$ ($ice_{growth}$, $circulation_{jfm}$)')
        ax3.axhline(y=0, color='lightgray', linestyle='--', linewidth=1)
        # Add the significant line
        import scipy
        nn=200 # Each year has 1000x4/5 samples (only the testing samples) for correlation calculation
        tt=scipy.stats.t.isf(0.025, nn, loc=0, scale=1); critical_corr=tt/(nn+tt**2)**0.5
        ax3.axhline(y=critical_corr, color='black', linestyle='--', linewidth=0.5, zorder=-1)
        ax3.axhline(y=-critical_corr, color='black', linestyle='--', linewidth=0.5, zorder=-1)
        ## Set up the xticks
        ax1.set_xticks(x)
        ax2.set_xticks(x)
        ax3.set_xticks(x)
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        xticklabels=[str(years[0])+'\nto\n'+str(years[-1]) for years in years_fit]
        ax3.set_xticklabels(xticklabels)
        ax3.set_yticks([-0.9,-0.6,-0.3,0,0.3,0.6,0.9])
        ax1.set_ylabel("RMSE (%)")
        ax2.set_ylabel("Relative\nimportance (%)")
        ax3.set_ylabel("Correlation")
        ax1.set_title(r'$\bf({A})$',loc='left',x=-0.17,y=0.85,size=9)
        ax2.set_title(r'$\bf({B})$',loc='left',x=-0.17,y=0.85,size=9)
        ax3.set_title(r'$\bf({C})$',loc='left',x=-0.17,y=0.85,size=9)
        ## Setup the axis:
        for ax in [ax1,ax2,ax3]: 
            for i in ['right', 'top']:
                ax.spines[i].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2); ax1.tick_params(axis='y', which='both',length=2)
            ax.set_xlim(x[0]-0.5,x[-1]+0.5)
        ## Setup the legends for ax1 and ax2
        for ax in [ax1]: # Get the legend for both
            legend_scatter=ax.legend(bbox_to_anchor=(0.02, 0.6), ncol=2, loc='lower left',
                    frameon=False, columnspacing=1, handletextpad=0.6, prop={'size':9.5})
        ## Setup lends for ax3
        legend_scatter=ax3.legend(bbox_to_anchor=(0.02, 0.7), ncol=2, loc='lower left',
                frameon=False, columnspacing=1, handletextpad=0.6, prop={'size':9.5})
        fig_name = 'fig2_SONDice_growth_relative_importance'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.15) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)
