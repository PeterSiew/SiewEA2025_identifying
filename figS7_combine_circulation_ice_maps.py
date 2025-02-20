import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime as dt
import subprocess
from multiprocessing import Process
import ipdb

years_fit = [range(1941,1961),range(1961,1981),range(1981,2001)] # testing
years_fit = [range(1961,1981),range(1981,2001), range(2001,2021),range(2021,2041),range(2041,2061),range(2061,2081),range(2081,2101)]  # Figure S7

grid=len(years_fit)
fig_adds=[]

def figure3A():
    fig_add='S7A'
    print("Starting Figure S7A")
    ### Create Figure 4A
    sic_var='CESM2_psl_en'; sic_ratio=1 # Using circulation for picking
    shading_var='CESM2_sic_en'; shading_var_ratio=100 # This is the sea ice growth
    contour_var1='CESM2_psl_en'; contour_var1_ratio=1; contour1_level_grids = [np.linspace(-900,900,13)]*grid
    contour_var2=None; contour_var2_ratio=None
    contour_var3=None; contour_var3_ratio=None 
    spatial_plotting=True; boxplot_plotting=False
    set_top_title=True
    set_region_box=False
    shading_level_grids= [np.linspace(-60,60,13)]*grid 
    ylims = (60,90)
    exec(open("./fig3_fig4_low_high_ice_samples.py").read())

def figure3B():
    fig_add='S7B'
    print("Starting Figure S7B")
    sic_var='CESM2_psl_en'; sic_ratio=1 
    shading_var='CESM2_sicthermo_en'; shading_var_ratio=1
    contour_var1=None; contour_var1_ratio=None; contour1_level_grids=[None]*grid
    contour_var2=None; contour_var2_ratio=None
    contour_var3=None; contour_var3_ratio=None
    spatial_plotting=True; boxplot_plotting=False
    set_top_title=True
    set_region_box=False
    shading_level_grids= [np.linspace(-60,60,13)]*grid 
    ylims = (60,90)
    exec(open("./fig3_fig4_low_high_ice_samples.py").read()) 

def figure3C():
    fig_add='S7C'
    print("Starting Figure S7C")
    ### Create Figure 4C
    sic_var='CESM2_psl_en'; sic_ratio=1 
    shading_var='CESM2_sicdynam_en'; shading_var_ratio=1
    contour_var1=None; contour_var1_ratio=None; contour1_level_grids=[None]*grid
    contour_var2=None; contour_var2_ratio=None
    contour_var3=None; contour_var3_ratio=None
    spatial_plotting=True; boxplot_plotting=False
    set_top_title=True
    set_region_box=False
    shading_level_grids= [np.linspace(-60,60,13)]*grid 
    ylims = (60,90)
    exec(open("./fig3_fig4_low_high_ice_samples.py").read()) 

p1 = Process(target=figure3A)
p1.start()
p2 = Process(target=figure3B)
p2.start()
p3 = Process(target=figure3C)
p3.start()

### To wait for all results to be here 
p1.join()
print("P1 finish")
p2.join()
print("P2 finish")
p3.join()
print("P3 finish")

fig_adds=['S7A','S7B','S7C']
zorders=[3,2,1]

plt.close()
fig, axs = plt.subplots(len(fig_adds),1, figsize=(25,6.8*len(fig_adds)))
axs=axs.flatten()
today_date=dt.date.today()
filenames=["/Users/home/siewpe/codes/graphs/%s_fig3_physical_processes_composite_%s.png"%(today_date,fig_add) for fig_add in fig_adds]
ABC=[r'$\bf({A})$',r'$\bf({B})$',r'$\bf({C})$']
ylabels_texts=['\nTotal\nsea ice\ngrowth','\nThermo-\ndynamic\nsea ice\ngrowth','\nDynamic\nsea ice\ngrowth']
import cartopy.feature as cfeature
for i in range(len(filenames)):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        axs[i].imshow(image, interpolation='none')
        axs[i].axis('off')
        axs[i].set_zorder(zorders[i])
        axs[i].annotate(ABC[i]+ylabels_texts[i], xy=(-0.08,0.85), xycoords='axes fraction',
            fontsize=25, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='white',alpha=1, pad=0.001), zorder=100)
        #axs[i].outline_patch.set_edgecolor('black')
        #axs[i].add_feature(cfeature.BORDERS, linewidth=1)
fig_name = 'figS6_combine'
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.1, hspace=-0.75)
plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.05)

