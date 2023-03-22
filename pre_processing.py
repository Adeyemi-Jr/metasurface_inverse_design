import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


plot_figure = False

#-----------------------------------------------#
#----------- DownSample datasets ---------------#
#-----------------------------------------------#

data_lists = ['1','2']
s_par_types = ['SZmax_Zmin', 'SZmin_Zmin']



for data_list in data_lists:

    for s_par_type in s_par_types:

        df = pd.read_csv('../data/processed/'+s_par_type+'_dataset_'+ data_list +'.csv')
        ##########################################
        #           DownSample  S- parameter Signals
        ##########################################

        down_sample_factor = 10
        x = np.array(df.columns)

        length_df = [*range(df.shape[1])]

        Selected_index = length_df[::down_sample_factor]
        #drop_idx = list(range(0,df.shape[1],down_sample_factor))
        new_df_cols = [j for i,j in enumerate(df.columns) if i in Selected_index]

        new_df = df[new_df_cols]
        new_x = np.array(new_df.columns)

        '''
        df_col = new_df.columns
        with open('your_file.txt', 'w') as f:
            for line in df_col:
                f.write("'"+line+"',")
        '''

        new_df.columns = [  s_par_type+'_'+ col for col in new_df.columns]
        new_df.to_csv('../data/processed/'+ s_par_type +'_dataset_'+ data_list +'_downsampled.csv',index= False)




##########################################
#     Import Design Parameters
##########################################


if plot_figure==True:
    n_rows = 4
    n_col = 2
    fig, axes = plt.subplots(nrows = n_rows, ncols = n_col, figsize=(15,12))
    fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axes.ravel()
    #tickers = np.random.randint(0,len(df),n_rows)
    tickers = np.asarray([9834,10118,9162,10054 ])
    for i in range(n_rows):
        #axs[i].plot(np.asarray(x,float), df.iloc[tickers[i],:],'go' ,np.asarray(new_x,float), new_df.iloc[tickers[i],:],linewidth=0.5, markersize= 0.5)
        axes[i,0].plot(np.asarray(x,float), df.iloc[tickers[i],:])
        axes[i,0].set_title('Iter '+ str(tickers[i]) + '  Original Sample points: ' + str(len(x))    )
        axes[i,1].plot(np.asarray(new_x, float), new_df.iloc[tickers[i], :],'go-', markersize = 1)
        axes[i,1].set_title('      '+ '  Downsampled Sample points: ' + str(len(new_x)))
        #ax.df.iloc[:,ticker].plot()

        #axes[i].set_title(i)
    plt.show()





##########################################
#     Join Design Par and response
##########################################

df_tmp = []
for data_list in data_lists:
    directory = '../data/raw/Data_'+data_list+'/Results'

    sub_dir_list = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    sub_dir_list = [x.replace('iteration_', '') for x in sub_dir_list]
    sub_dir_values = [int(x)-1 for x in sub_dir_list]
    sub_dir_values.sort()

    par_dir = '../data/raw/Data_'+data_list+'/parameter_space_'+data_list+'.csv'
    par_df = pd.read_csv(par_dir,index_col=[0])
    par_df.reset_index(inplace = True, drop=True)


    df_SZmax_Zmin= pd.read_csv('../data/processed/'+s_par_types[0]+'_dataset_'+ data_list +'_downsampled.csv')
    df_SZmin_Zmin= pd.read_csv('../data/processed/'+s_par_types[1]+'_dataset_'+ data_list +'_downsampled.csv')


    #select only the parameter setting that worked
    para_working = par_df[par_df.index.isin(sub_dir_values)]
    para_working.reset_index(inplace=True,drop = True)
    #join df along column
    df_concated = pd.concat([para_working, df_SZmax_Zmin,df_SZmin_Zmin],axis = 1)


    #select only the parameter setting that worked
    #df_concated = par_df_s_par_df[par_df_s_par_df.index.isin(sub_dir_values)]
    df_concated.reset_index(inplace=True,drop= True)
    df_tmp.append(df_concated)


final_df = pd.concat(df_tmp,axis=0)
final_df.to_csv('../data/processed/data.csv',index = False)





'''
test_df = df.iloc[9834,:]
gradient = np.gradient(test_df)
gradient_df = pd.DataFrame(gradient)
ax = test_df.T.plot()
gradient_df.plot(ax=ax)

plt.show()
'''
