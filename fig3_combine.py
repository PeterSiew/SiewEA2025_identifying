import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime as dt
import subprocess
from multiprocessing import Process
import ipdb
import scipy

years_fit = [range(1941,1961),range(1961,1981),range(1981,2001)] # testing
years_fit = [range(1981,2001), range(2001,2021),range(2021,2041),range(2041,2061),range(2061,2081),range(2081,2101)]  # Figure 3
years_fit = [range(1861,1881), range(1881,1901),range(1901,1921),range(1921,1941),range(1941,1961),range(1961,1981)]  # Supplementarty figure (early periods for CESM2)

grid=len(years_fit)
fig_adds=[]

def figure3A():
    fig_add='3A'
    print("Startin Figure 3A")
    ### Create Figure 4A
    sic_var='intCESM2_sic_en'; sic_ratio=100 # Default (new - using initial sea ice for picking)
    shading_var='CESM2_sic_en'; shading_var_ratio=100 # This is the sea ice growth
    #contour_var1='CESM2_psl_en'; contour_var1_ratio=1
    contour_var1=None; contour_var1_ratio=None; contour1_level_grids=[None]
    contour_var2='intCESM2_sic_en'; contour_var2_ratio=100 # contour 2 is particular for Sep sea ice state
    contour_var3='endCESM2_sic_en'; contour_var3_ratio=100  # contour 3 is particular for Dec sea ice state
    spatial_plotting=True; boxplot_plotting=False
    #fig_adds.append(fig_add)
    set_top_title=True
    set_region_box=False
    shading_level_grids= [np.linspace(-90,90,13)]*grid # For SIC growth
    ylims = (67,90)
    exec(open("./fig3_fig4_low_high_ice_samples.py").read())
    #process1 = subprocess.Popen(["python","fig3_fig4_low_high_ice_samples.py","&"])


def figure3B():
    fig_add='3B'
    print("Startin Figure 3B")
    ### Create Figure 4B
    sic_var='intCESM2_sic_en'; sic_ratio=100 # Default (new - using initial sea ice for picking)
    shading_var='CESM2_sicthermo_en'; shading_var_ratio=1
    contour_var1=None; contour_var1_ratio=None; contour1_level_grids=[None]
    contour_var2=None; contour_var2_ratio=None
    contour_var3=None; contour_var3_ratio=None
    spatial_plotting=True; boxplot_plotting=False
    #fig_adds.append(fig_add)
    set_top_title=True
    set_region_box=False
    shading_level_grids= [np.linspace(-90,90,13)]*grid # For SIC growth
    ylims = (67,90)
    exec(open("./fig3_fig4_low_high_ice_samples.py").read()) 

def figure3C():
    fig_add='3C'
    print("Startin Figure 3C")
    ### Create Figure 4C
    sic_var='intCESM2_sic_en'; sic_ratio=100 # Default (new - using initial sea ice for picking)
    shading_var='CESM2_sicdynam_en'; shading_var_ratio=1
    contour_var1=None; contour_var1_ratio=None; contour1_level_grids=[None]
    contour_var2=None; contour_var2_ratio=None
    contour_var3=None; contour_var3_ratio=None
    spatial_plotting=True; boxplot_plotting=False
    #fig_adds.append(fig_add)
    set_top_title=True
    set_region_box=False
    shading_level_grids= [np.linspace(-90,90,13)]*grid # For SIC growth
    ylims = (67,90)
    exec(open("./fig3_fig4_low_high_ice_samples.py").read()) 

def figure3D():
    fig_add='3D'
    print("Startin Figure 3D")
    ### Create Figure 4D
    sic_var='intCESM2_sic_en'; sic_ratio=100 # Default (new - using initial sea ice for picking)
    #shading_var='CESM2_otemp_en'; shading_var_ratio=1
    shading_var='CESM2_tas_en'; shading_var_ratio=1
    contour_var1=None; contour_var1_ratio=None
    contour_var2=None; contour_var2_ratio=None
    contour_var3=None; contour_var3_ratio=None
    spatial_plotting=True; boxplot_plotting=False
    #fig_adds.append(fig_add)
    set_top_title=True
    set_region_box=False
    #shading_level_grids= [np.linspace(-1.5,1.5,13)]*grid # Ocean temp
    shading_level_grids= [np.linspace(-6,6,13)]*grid # Ocean temp
    ylims = (67,90)
    exec(open("./fig3_fig4_low_high_ice_samples.py").read()) 

p1 = Process(target=figure3A)
p1.start()
p2 = Process(target=figure3B)
p2.start()
p3 = Process(target=figure3C)
p3.start()
#p4 = Process(target=figure3D)
#p4.start()

### To wait for all results to be here 
p1.join()
print("P1 finish")
p2.join()
print("P2 finish")
p3.join()
print("P3 finish")
#p4.join()
#print("P4 finish")

fig_adds=['3A','3B','3C']
zorders=[3,2,1]
ABC=[r'$\bf({A})$',r'$\bf({B})$',r'$\bf({C})$']
ylabels_texts=['\nTotal\nsea ice\ngrowth','\nThermo-\ndynamic\nsea ice\ngrowth','\nDynamic\nsea ice\ngrowth']
#ylabels=['(A)\nTotal\nice\ngrowth','(B)\nThermo\nice\ngrowth','(C)\nDynamic\nice\ngrowth','(D)\nSAT']
#ylabels=[r'$\bf{(A)}$\nTotal\nice\ngrowth','(B)\nThermo-\ndynamic\nice\ngrowth','(C)\nDynamic\nice\ngrowth']
plt.close()
fig, axs = plt.subplots(len(fig_adds),1, figsize=(25,6.8*len(fig_adds)))
axs=axs.flatten()
today_date=dt.date.today()
filenames=["/Users/home/siewpe/codes/graphs/%s_fig3_physical_processes_composite_%s.png"%(today_date,fig_add) for fig_add in fig_adds]
for i in range(len(filenames)):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        axs[i].imshow(image, interpolation='none')
        axs[i].axis('off')
        axs[i].set_zorder(zorders[i])
        axs[i].annotate(ABC[i]+ylabels_texts[i], xy=(-0.08,0.85), xycoords='axes fraction',
            fontsize=25, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='white',alpha=1, pad=0.001), zorder=100)
fig_name = 'fig3_combine'
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-1, hspace=-0.7)
plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.05)

