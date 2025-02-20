import numpy as np
from multiprocessing import Process
import multiprocessing
import ipdb

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools
import matplotlib.pyplot as plt
import datetime as dt


def find_rmse(alpha, return_dict):
    save_rmse=True
    plotting_fig1C=False; plotting_fig2=False
    exec(open("./fig1D_reconstruct_fig2_relative_importance.py").read())
    # Read the saved rmse files
    rmse=np.load('./alphas/%s_rmse.npy'%alpha)
    return_dict[alpha]=(alpha,rmse)

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    ps=[]
    alphas=[10]+[50]+[100]+[200]+[300]+[400]+[500]+[i for i in range(1000,15000,500)]
    #alphas=[20,200,2000,20000]
    for alpha in alphas:
        p = Process(target=find_rmse, args=(alpha,return_dict))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()
    #print(return_dict.values())

    ### Start plotting
    rmses=[]
    for alpha in alphas:
        rmse=np.load('./alphas/%s_rmse.npy'%alpha)
        rmses.append(rmse)
    ## Plotting
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(5,1))
    x=range(len(rmses))
    ax1.plot(x,rmses)
    ax1.set_xticks(x)
    xticklabels=alphas
    ax1.set_xticklabels(xticklabels,size=7,rotation=45)
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("Alpha used in Ridge regression")
    ax1.set_xlim(x[0],x[-1])
    #ax1.set_yticks([3,5,7,9])
    fig_name = 'RMSE_over_training_periods'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.001)
