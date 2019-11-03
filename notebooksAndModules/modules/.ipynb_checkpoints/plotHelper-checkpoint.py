from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter1d
from matplotlib.patches import Rectangle


def plot(re,gt,vmax,vmin,detail):
    
    Plot_data=pd.DataFrame(re,columns=['DP','BD','E','FS','A','RC'])
    #max filter data
    filtered_data=Plot_data.copy()
    for index, row in filtered_data.iteritems():
        filtered_data[index]=maximum_filter1d(filtered_data[index],mode='nearest',size= 3)
        
    def plot_predict_result(df):
        y=[x for x in range(1,7)]
        ax.plot(y, gt, label='ground truth',color='green')
#        ax.plot(y, np.mean(df,axis=0), label='mean predict',color='blue')
#         ax.plot(y, np.median(df,axis=0), label='median predict',color='red')
        ax.plot(y, np.round(np.mean(df,axis=0)), label='mean round predict',color='black')
        ax.plot(y, np.round(np.median(df,axis=0)), label='median round predict',color='blue')
        plt.xticks(y)
        plt.yticks(y_range)
        labels=['DP','BD','E','FS','A','RC']
        ax.set_xticklabels(labels)
        ax.legend()    
        it=1
        for types in ['DP','BD','E','FS','A','RC']:
            Range=vmax[types]-vmin[types]
            rect = Rectangle((it, vmin[types]), 0.2, Range, color='green')
            ax.add_patch(rect)   
            it=it+1
    #------------------------------------------------------------------------------------------
    x_range = range(0,len(Plot_data),1)
    y_range = np.linspace(1,5,num=9)    
        
    # raw data    
    ax=plt.figure(figsize=(20,20))
    ax = plt.subplot(421)
    Plot_data.plot(ax=ax)
    ax.legend(['DP','BD','E','FS','A','RC'],bbox_to_anchor=(1,1))
    plt.ylabel('Gear Score')
    plt.xlabel('Video clips length')
    

    plt.xticks(x_range)
    plt.yticks(y_range)

    # filtered data   
    ax = plt.subplot(422)
    filtered_data.plot(ax=ax)
    ax.legend(['DP','BD','E','FS','A','RC'],bbox_to_anchor=(1,1))
    plt.ylabel('Gear Score')
    plt.xlabel('Video clips length')
    plt.xticks(x_range)
    plt.yticks(y_range)
    
    
    ax = plt.subplot(423)
    plot_predict_result(Plot_data)

    ax = plt.subplot(424)
    plot_predict_result(filtered_data)
    
    
    if detail == True:
        print("Ground Truth",gt)
        print("mean prediction MAE:", sum(abs(np.mean(Plot_data,axis=0)-gt)))
        print("median prediction MAE:", sum(abs(np.median(Plot_data,axis=0)-gt)))
        print("round prediction MAE:", sum(abs(np.round(np.mean(Plot_data,axis=0))-gt)))
        print('------------------------------------------------------------------------------')
        print("Filtered Ground Truth",gt)
        print("Filtered mean prediction MAE:", sum(abs(np.mean(filtered_data,axis=0)-gt)))
        print("Filteredmedian prediction MAE:", sum(abs(np.median(filtered_data,axis=0)-gt)))
        print("Filtered round prediction MAE:", sum(abs(np.round(np.mean(filtered_data,axis=0))-gt)))
