import matplotlib.pyplot as plt
from PIL import Image
import datetime as dt


plt.close()
date=dt.date.today()
fileA=["/Users/home/siewpe/codes/graphs/%s_fig1A_ice_growth_map.png"%date]
fileB=["/Users/home/siewpe/codes/graphs/%s_fig1B_SOND_ice_growth_ts.png"%date]
fileC=["/Users/home/siewpe/codes/graphs/%s_fig1C_SOND_ice_growth_ts.png"%date]
fileD=["/Users/home/siewpe/codes/graphs/%s_fig1D_reconstruct_ice.png"%date]
filenames=fileA+fileB+fileC+fileD
titles=[r'$\bf({A})$',r'$\bf({B})$',r'$\bf({C})$',r'$\bf({D})$']
fig, axs = plt.subplots(len(filenames),1, figsize=(6,8.5))
axs=axs.flatten()
today_date=dt.date.today()
xpos=[-0.05,0,0,0]
xpos=[-0.08,-0.02,-0.038,-0.03]
ypos=[1.02,0.98,0.98,0.98]
ypos=[0.92]*4
for i in range(len(filenames)):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        #image=plt.imread(f)
        #axs[i].imshow(image, interpolation='none')
        axs[i].imshow(image, interpolation='gaussian')
        axs[i].axis('off')
        #axs[i].set_title(titles[i],loc='left',pad=pads[i])
        axs[i].annotate(titles[i],xy=(xpos[i],ypos[i]),xycoords='axes fraction', fontsize=10)
fig_name = 'fig1_combine_icegrowth'
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=50, hspace=0.03) # hsapce control the vertical seperation
plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

