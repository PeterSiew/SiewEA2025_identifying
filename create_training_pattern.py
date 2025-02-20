import xarray as xr
import numpy as np
import ipdb
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import pandas as pd
import scipy
import xesmf as xe

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools; reload(tools)

if __name__ == "__main__":

    #vars=['CanESM5_psl_en', 'CanESM5_sic_en']
    vars=['CESM2_sitempbot_en']
    vars=['CESM2_fswabs_en']
    vars=['CESM2_hs_en'] # Snow thickness
    vars=['CESM2_iceu_en'] # This also includes icev_en
    vars=['CESM2_out_en']
    vars=['CESM2_ow_en'] # Vertical velocity in ocean
    vars=['CESM2_isenbot_en'] # sensible heat flux at the bottom
    vars=['CESM2_iconductbot_en'] 
    vars=['CESM2_isentop_en'] # sensible heat flux at the top
    # Ocean variable
    vars=['CESM2_iceoceanflux_en']
    vars=['CESM2_ou_en'] # Ocean drift here. Also include ov_en (25-m ocean)
    vars=['CESM2_out_en'] # Ocean heat transport
    vars=['CESM2_surlatent_en'] # This will also include the sensible
    # Standard one
    vars=['CESM2_hi_en']
    vars=['MIROC6_psl_en', 'MIROC6_sic_en']
    vars=['CanESM5_psl_en', 'CanESM5_sic_en']
    vars=['CESM2_sicdynam_en', 'CESM2_sicthermo_en'] # Only for SOND
    vars=['CESM2_otemp_en'] # Ocean temeprature from surface at 5-m deep

    if True:
        vars=['ACCESS-ESM1-5_sic_en', 'ACCESS-ESM1-5_psl_en']; ensembles=range(1,41) # Access-ESM1
        vars=['CESM2_sic_en', 'CESM2_psl_en']; ensembles=range(1,51) # CESM2, CanESM5 and MIROC6
        vars=['CESM2_sst_en']; ensembles=range(1,51)
        vars=['CESM2_tas_en']; ensembles=range(1,51)
        mons = [1,2,3]; season='JFM'; years=range(1850,2101) # For JFM
        mons = [9,10,11,12]; season='SOND'; years=range(1850,2101) # For SON/SOND

    if False: # For observations 
        vars=['obs2_psl_en', 'obs4_sic_en'] # obs2_psl is ERA5; obs4_sic is to combine PIOMAS and NSIDC (updated to 2023 Sep)
        ensembles=[''] # For observations
        years=range(1940,2023) # From 1940 to 2022

    for var in vars:
        for en in ensembles:
            print(var, en)
            data = tools.read_data(var+str(en), months='all', slicing=False, limit_lat=False,reverse_lat=False)
            # Constrain the data range at the beginning (useful for obs sic)
            data=data.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
            ### Regrid (Can be earlier)
            if 'sic_en' in var: 
                #lats = np.arange(45.5,90,1); lons=np.arange(-179.5,180,1)
                pass # (1x1 grid is good enough)
            ### SST data (atmospheric grid)
            ### Atmospheric grid variables (just do normal regrid)
            elif ('psl_en' in var) or ('tas_en' in var) or ('sst_en' in var):
                if 'sst_en' in var: # (sst atmospheric data has 0 - this has to be done before regrid)
                    zero_mask=data==0
                    data=xr.where(zero_mask,np.nan,data)
                    data=data-273.15 # Change K to C for CESM2
                lats=np.arange(54,88,3)
                lons = np.arange(-177,180,3).tolist(); lons.remove(0)
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(data.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                data = regridder(data)
                if 'tas_en' in var:
                    data=data-273.15 # Change K to C for CESM2
            # Surface sensible and latent (atmosphere) together
            elif 'surlatent_en' in var:
                latent= data
                new_var=var.replace('surlatent', 'sursensible')
                sensible=tools.read_data(new_var+str(en), months='all', slicing=False, limit_lat=False,reverse_lat=False)
                sensible=sensible.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
                lats=np.arange(54,88,3)
                lons = np.arange(-177,180,3).tolist(); lons.remove(0)
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(latent.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                latent= regridder(latent)
                regridder = xe.Regridder(sensible.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                sensible= regridder(sensible)
                data=latent+sensible
            ### ICE grid varable (sic is not here becuase they are already regrided)
            elif ('hi_en' in var) or ('sicdynam_en' in var) or ('sicthermo_en' in var) or ('hs_en' in var) \
                    or ('fswabs_en' in var) or ('iceoceanflux_en' in var) or ('sitempbot_en' in var) \
                    or ('isenbot_en' in var) or ('iconductbot_en' in var):
                # Regrid the ocean grid data
                data = data.rename({"TLON": "lon", "TLAT": "lat"})
                data = data.drop('ULAT').drop('ULON')
                lats = np.arange(45.5,90,1); lons=np.arange(0.5,360,1)
                #ds_out = xe.util.grid_2d(0, 360, 1, 45, 89.5, 1) 
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(data.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                data = regridder(data)
            ## Ice drift (u and v together)
            elif ('iceu_en' in var):
                u_data = data
                new_var=var.replace('iceu_en', 'icev_en')
                v_data=tools.read_data(new_var+str(en), months='all', slicing=False, limit_lat=False,reverse_lat=False)
                v_data=v_data.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
                # Multiple the angle before regriding
                # Read the angle file
                path = '/dx13/pyfsiew/LENS/CESM2/ice_u/b.e21.*.f09_g17.LE2-%s.cice.h.uvel.*.nc'%'1001.001'
                angle_xr = xr.open_mfdataset(path, chunks={'time':500}, decode_times=True)['ANGLET']
                angle_xr=angle_xr.assign_coords({'time':u_data.time})
                if False:
                    cos_angle = np.cos(-angle_xr)
                    sin_angle = np.sin(-angle_xr)
                    udrift = u_data*cos_angle + v_data*sin_angle 
                    vdrift= v_data*cos_angle - u_data*sin_angle
                else: # new
                    cos_angle = np.cos(angle_xr)
                    sin_angle = np.sin(angle_xr)
                    udrift = u_data*cos_angle 
                    #vdrift= v_data*sin_angle
                    vdrift= v_data*cos_angle
                # After correcting the angle - do the regrid
                # udrift
                udrift=udrift.rename({"TLON": "lon", "TLAT": "lat"}) 
                udrift=udrift.drop('ULAT').drop('ULON')
                lats=np.arange(45.5,90,1); lons=np.arange(0.5,360,1)
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(udrift.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                udrift= regridder(udrift)
                # vdrift
                vdrift=vdrift.rename({"TLON": "lon", "TLAT": "lat"})
                vdrift=vdrift.drop('ULAT').drop('ULON')
                regridder = xe.Regridder(vdrift.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                vdrift= regridder(vdrift)
            ### Ocean grid varable
            elif ('otemp_en' in var):
                if 'z_t' in data.dims:
                    #data=data.mean('z_t') # Average from 5-to-55 meter
                    data=data.sel(z_t=500) # 5m ocean depth
                # Regrid the ocean grid data
                data = data.rename({"TLONG": "lon", "TLAT": "lat"})
                lats = np.arange(45.5,90,1); lons=np.arange(0.5,360,1)
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(data.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                data=regridder(data)
            elif ('ow_en' in var): # Not sure why the coordinates are wrong - needed to add back manually
                data=data.mean('z_w_top')  # This is z_w_top
                # Add back TLAT and TLONG as coordinates. Missing somehow
                filename='b.e21.BSSP370smbb.f09_g17.LE2-1301.020.pop.h.WVEL.209501-210012.nc'
                data_supp=xr.open_dataset('/dx02/pyfsiew/CESM2_LENS_ocean/WVEL/%s'%filename)
                data=data.assign_coords(TLAT=data_supp.TLAT).assign_coords(TLONG=data_supp.TLONG)
                # Regrid the ocean grid data
                data = data.rename({"TLONG": "lon", "TLAT": "lat"})
                lats = np.arange(45.5,90,1); lons=np.arange(0.5,360,1)
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(data.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                data=regridder(data)
            elif ('ou_en' in var): # ov is handled here together
                u_data = data
                #u_data=u_data.mean('z_t') # average across the z
                u_data=u_data.sel(z_t=2500).drop('z_t')
                new_var=var.replace('ou_en', 'ov_en')
                v_data=tools.read_data(new_var+str(en), months='all', slicing=False, limit_lat=False,reverse_lat=False)
                v_data=v_data.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
                #v_data=v_data.mean('z_t') # average across the z
                v_data=v_data.sel(z_t=2500).drop('z_t')
                # Multiple the angle before regriding
                # Read the angle file
                path = '/dx02/pyfsiew/CESM2_LENS_ocean/ocean_grid_angle.nc' # Read the ANGLE on U-grid 
                angle_xr = xr.open_dataset(path, chunks={'time':500}, decode_times=True)['ANGLE'].compute()
                angle_xr=angle_xr.drop('TLAT').drop('TLONG')
                if True:
                    cos_angle = np.cos(-angle_xr)
                    sin_angle = np.sin(-angle_xr)
                    udrift = u_data*cos_angle + v_data*sin_angle 
                    vdrift= v_data*cos_angle - u_data*sin_angle
                else:
                    cos_angle = np.cos(angle_xr) # The single angle method
                    sin_angle = np.sin(angle_xr)
                    udrift = u_data*cos_angle 
                    vdrift= v_data*cos_angle 
                # After multiplying angle - do the regrid
                # udrift
                udrift=udrift.rename({"ULONG": "lon", "ULAT": "lat"}) 
                lats=np.arange(45.5,90,1); lons=np.arange(0.5,360,1)
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(udrift.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                udrift= regridder(udrift)
                # vdrift
                vdrift=vdrift.rename({"ULONG": "lon", "ULAT": "lat"})
                regridder = xe.Regridder(vdrift.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                vdrift= regridder(vdrift)
            elif ('out_en' in var): # ovt is handled here together
                # Do the u and v
                uvar=var.replace('out_en', 'ou_en')
                u_data=tools.read_data(uvar+str(en), months='all', slicing=False, limit_lat=False,reverse_lat=False)
                u_data=u_data.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
                u_data=u_data.sel(z_t=2500).drop('z_t')
                vvar=var.replace('out_en', 'ov_en')
                v_data=tools.read_data(vvar+str(en), months='all', slicing=False, limit_lat=False,reverse_lat=False)
                v_data=v_data.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
                v_data=v_data.sel(z_t=2500).drop('z_t')
                # Multiple the angle before regriding
                path = '/dx02/pyfsiew/CESM2_LENS_ocean/ocean_grid_angle.nc' # Read the ANGLE on U-grid 
                angle_xr = xr.open_dataset(path, chunks={'time':500}, decode_times=True)['ANGLE'].compute()
                angle_xr=angle_xr.drop('TLAT').drop('TLONG')
                cos_angle = np.cos(-angle_xr); sin_angle = np.sin(-angle_xr)
                udrift = u_data*cos_angle + v_data*sin_angle 
                vdrift= v_data*cos_angle - u_data*sin_angle
                # Start the regrid
                # udrift
                udrift=udrift.rename({"ULONG": "lon", "ULAT": "lat"}) 
                lats=np.arange(45.5,90,1); lons=np.arange(0.5,360,1)
                ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                regridder = xe.Regridder(udrift.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                udrift= regridder(udrift)
                # vdrift
                vdrift=vdrift.rename({"ULONG": "lon", "ULAT": "lat"})
                regridder = xe.Regridder(vdrift.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                vdrift= regridder(vdrift)
                # do the temperature
                tvar=var.replace('out_en', 'otemp_en')
                t_data=tools.read_data(tvar+str(en), months='all', slicing=False, limit_lat=False,reverse_lat=False)
                t_data=t_data.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
                t_data=t_data.sel(z_t=2500).drop('z_t')
                # Regrid the ocean grid data
                t_data = t_data.rename({"TLONG": "lon", "TLAT": "lat"})
                regridder = xe.Regridder(t_data.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                t_data=regridder(t_data)
                # Multiple temp
                udrift=udrift*t_data
                vdrift=vdrift*t_data
                new_var=var.replace('out_en', 'ovt_en') # For saving
                if False: # use the direct variable from CESM2 (UET and VNT)
                    u_data = data
                    u_data=u_data.mean('z_t') # average across the z
                    new_var=var.replace('out_en', 'ovt_en')
                    v_data=tools.read_data(new_var+str(en), months='all', slicing=False, limit_lat=False,reverse_lat=False)
                    v_data=v_data.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%years[-1]))
                    v_data=v_data.mean('z_t') # average across the z
                    # Multiple the angle before regriding
                    # Read the angle file
                    path = '/dx02/pyfsiew/CESM2_LENS_ocean/ocean_grid_angle.nc' # Read the ANGLE 
                    if False: # use angle U-grid
                        angle_xr = xr.open_dataset(path, chunks={'time':500}, decode_times=True)['ANGLE'].compute()
                        angle_xr=angle_xr.drop('TLAT').drop('TLONG')
                    else: # Use angle for T-grid
                        angle_u = xr.open_dataset(path, chunks={'time':500}, decode_times=True)['ANGLE'].compute()
                        angle_u=angle_u.drop('ULAT').drop('TLONG')
                        angle_v = xr.open_dataset(path, chunks={'time':500}, decode_times=True)['ANGLET'].compute()
                        angle_v=angle_v.drop('TLAT').drop('ULONG')
                    udrift = u_data*np.cos(-angle_u) + v_data*np.sin(-angle_v) 
                    vdrift= v_data*np.cos(-angle_v) - u_data*np.sin(-angle_u)
                    # After multiplying angle - do the regrid
                    # udrift
                    udrift=udrift.rename({"ULONG": "lon", "TLAT": "lat"}) 
                    lats=np.arange(45.5,90,1); lons=np.arange(0.5,360,1)
                    ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
                    regridder = xe.Regridder(udrift.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                    udrift= regridder(udrift)
                    # vdrift
                    vdrift=vdrift.rename({"TLONG": "lon", "ULAT": "lat"})
                    regridder = xe.Regridder(vdrift.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
                    vdrift= regridder(vdrift)
            ######
            ### Further process and save the data
            if ('sic_en' in var) or ('hi_en' in var): # Get Sep, Dec and sea ice growth
                data = data.compute() # Load the data in RAM
                # Extract Dec data
                mon_mask = data.time.dt.month.isin([mons[0]])
                data_oct = data.isel(time=mon_mask) 
                mon_mask = data.time.dt.month.isin([mons[-1]])
                data_dec = data.isel(time=mon_mask)
                diff = data_dec - data_oct.assign_coords(time=data_dec.time)
                # Save the difference, Oct and Dec ice states
                for year in years:
                    # For Sep (Jan) sea ice state
                    data_sel = data_oct.sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-28'%(year,mons[-1])))
                    data_sel = data_sel.squeeze().rename('training') # Remove the single time dimension
                    data_sel.to_netcdf('/mnt/data/data_b/changing_circulation_ice_training/%s/int%s_%s_%s.nc'%(season,var,en,year))
                    # For Dec (Mar) sea ice state
                    data_sel = data_dec.sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-31'%(year,mons[-1])))
                    data_sel = data_sel.squeeze().rename('training') 
                    data_sel.to_netcdf('/mnt/data/data_b/changing_circulation_ice_training/%s/end%s_%s_%s.nc'%(season,var,en,year))
                    # For Dec minus Sep ice
                    data_sel = diff.sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-28'%(year,mons[-1])))
                    data_sel = data_sel.squeeze().rename('training') 
                    data_sel.to_netcdf('/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,var,en,year))
            ### Sea ice dynamic and thermodynamic growth
            if ('sicdynam_en' in var) or ('sicthermo_en' in var): # Only work for SOND but not JFM
                data = data.compute() # Load the data in RAM
                if False: # Sea ice grwoth for OND
                    mon_mask = data.time.dt.month.isin([mons[0]])
                    data_oct = data.isel(time=mon_mask) 
                    mon_mask = data.time.dt.month.isin([mons[1]])
                    data_nov = data.isel(time=mon_mask).assign_coords({'time':data_oct.time})
                    mon_mask = data.time.dt.month.isin([mons[2]])
                    data_dec = data.isel(time=mon_mask).assign_coords({'time':data_oct.time}) 
                    # Mulitple to the number of days
                    data_sum = data_oct*31 + data_nov*30 + data_dec*31
                if True: # sea ice grwoth for SOND
                    mon_mask = data.time.dt.month.isin([mons[0]])
                    data_sep = data.isel(time=mon_mask) 
                    mon_mask = data.time.dt.month.isin([mons[1]])
                    data_oct = data.isel(time=mon_mask).assign_coords({'time':data_sep.time})
                    mon_mask = data.time.dt.month.isin([mons[2]])
                    data_nov = data.isel(time=mon_mask).assign_coords({'time':data_sep.time}) 
                    mon_mask = data.time.dt.month.isin([mons[3]])
                    data_dec = data.isel(time=mon_mask).assign_coords({'time':data_sep.time}) 
                    data_sum = data_sep*30 + data_oct*31 + data_nov*30 + data_dec*31 # They are monthly mean so multiple the days
                for year in years:
                    data_sel=data_sum.sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-28'%(year,mons[-1])))
                    data_sel= xr.DataArray(data_sel).rename('training')
                    data_sel.to_netcdf('/dx02/pyfsiew/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,var,en,year))
            ### Simply SOND average (all are atmospheric grids)
            elif ('psl_en' in var) or ('tas_en' in var) or ('sst_en' in var) or ('fswabs_en' in var) or ('hs_en' in var) \
                or ('surlatent_en' in var) or ('iceoceanflux_en' in var) or ('sitempbot_en' in var) or ('otemp_en' in var) \
                or ('ow_en' in var) or ('isenbot_en' in var) or ('iconductbot_en' in var):
                data = data.compute() # Load the data in RAM
                if False: # Real SOND average (Normal)
                    mon_mask = data.time.dt.month.isin([mons])
                    data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
                    for year in years:
                        data_sel = data.sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-28'%(year,mons[-1])))
                        data_sel = data_sel.squeeze().rename('training') # Remove the single time dimension
                        data_sel.to_netcdf('/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s.nc' %(season,var,en,year)) # Save
                else: # Extract Sep, Oct, Nov and Dec individually
                    mon_strs=['09','10','11','12']; mon_saves=['sep','oct','nov','dec']
                    for mon_str,mon_save in zip(mon_strs,mon_saves):
                        for year in years:
                            data_sel = data.sel(time=slice('%s-%s-01'%(year,mon_str),'%s-%s-30'%(year,mon_str)))
                            data_sel = data_sel.squeeze().rename('training') 
                            data_sel.to_netcdf('/mnt/data/data_b/changing_circulation_ice_training/%s/%s_%s_%s_%s.nc'%(season,var,mon_save,en,year))
            elif ('iceu_en' in var) or ('ou_en' in var) or ('out_en' in var): # Two variables - for udrift and icedrift
                udrift=udrift.compute() 
                vdrift=vdrift.compute() 
                mon_mask=udrift.time.dt.month.isin([mons])
                udrift=udrift.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
                vdrift=vdrift.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
                for year in years:
                    udrift_sel=udrift.sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-28'%(year,mons[-1])))
                    vdrift_sel=vdrift.sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-28'%(year,mons[-1])))
                    udrift_sel=udrift_sel.squeeze().rename('training') # Remove the single time dimension
                    vdrift_sel=vdrift_sel.squeeze().rename('training') 
                    # Save the file
                    udrift_sel.to_netcdf('/dx02/pyfsiew/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,var,en,year))
                    vdrift_sel.to_netcdf('/dx02/pyfsiew/changing_circulation_ice_training/%s/%s_%s_%s.nc'%(season,new_var,en,year))



def backup():
    #data = data.differentiate('time', datetime_unit='M') 
    data_temp = np.diff(data,axis=0)
    data_temp = np.concatenate([data_temp[0][None,:,:],data_temp])
    data = data.copy(data=data_temp)
    # Remove seasonal cycle using the long-term mean
    #data = tools.remove_seasonal_cycle_simple(data, detrend=True)
