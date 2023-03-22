import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fwd_test_history_path = 'loss-history/fwdTestHistory.csv'
fwd_train_history_path = 'loss-history/fwdTrainHistory.csv'

inv_test_history_path = 'loss-history/invTestHistory.csv'
inv_train_history_path = 'loss-history/invTrainHistory.csv'


fwd_test_history = pd.read_csv(fwd_test_history_path,header = None)
fwd_train_history = pd.read_csv(fwd_train_history_path,header = None)


inv_test_history = pd.read_csv(inv_test_history_path,header = None)
inv_train_history = pd.read_csv(inv_train_history_path,header = None)



fwd_test_history_df = pd.DataFrame(fwd_test_history.T)
fwd_train_history_df = pd.DataFrame(fwd_train_history.T)


inv_test_history_df = pd.DataFrame(inv_test_history.T)
inv_train_history_df = pd.DataFrame(inv_train_history.T)


fig, axes = plt.subplots(1,2)

axes[0].plot(fwd_test_history_df, label = 'test')
axes[0].plot(fwd_train_history_df, label = 'train')
axes[0].set_title('forward')

axes[1].plot(inv_test_history_df, label = 'test')
axes[1].plot(inv_train_history_df, label = 'train')
axes[1].set_title('inverse')

plt.legend()
#plt.plot(test_history.iloc[1:])

plt.show()
