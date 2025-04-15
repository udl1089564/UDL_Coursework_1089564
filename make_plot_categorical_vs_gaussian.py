import os
import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'size': 15})

# Make Categorical vs Gaussian plots
data_dir = 'saved_outputs'
data = [['permuted_categorical_noDINO_0.npy', 'permuted_categorical_noDINO_200.npy',
         'permuted_gaussian_noDINO_0.npy', 'permuted_gaussian_noDINO_200.npy'],
         ['split_categorical_noDINO_0.npy', 'split_categorical_noDINO_200.npy',
          'split_gaussian_noDINO_0.npy', 'split_gaussian_noDINO_200.npy']]

for idx in range(2):
    fig = plt.figure(figsize=(7, 4), dpi=300)

    ax = plt.gca()
    ax.set_xlabel('Number of Tasks')
    ax.set_xticks(list(range(1, 11 if idx == 0 else 6)))

    # Plot categorical
    ax.set_ylabel(r'Average Accuracy ($\uparrow$)', color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.set_ylim(0.7, 1)

    all_accs = np.load(os.path.join(data_dir, data[idx][0]))
    accs = np.nanmean(all_accs, axis=1)
    vcl_categorical = ax.plot(np.arange(len(accs))+1, accs, label='Categorical', marker='o', color='tab:red', markersize=4)

    all_accs = np.load(os.path.join(data_dir, data[idx][1]))
    accs = np.nanmean(all_accs, axis=1)
    vcl_coreset_categorical = ax.plot(np.arange(len(accs))+1, accs, label='Categorical + Coreset', linestyle='dashed', marker='o', color='tab:red', markersize=4)

    # Plot gaussian
    ax2 = ax.twinx()
    ax2.set_ylabel(r'RMSE ($\downarrow$)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    all_mse = np.load(os.path.join(data_dir, data[idx][2]))
    rmse = np.sqrt(np.nanmean(all_mse, axis=1))
    vcl_gaussian = ax2.plot(np.arange(len(rmse))+1, rmse, label='Gaussian', marker='o', color='tab:blue', markersize=4)

    all_mse = np.load(os.path.join(data_dir, data[idx][3]))
    rmse = np.sqrt(np.nanmean(all_mse, axis=1))
    vcl_coreset_gaussian = ax2.plot(np.arange(len(rmse))+1, rmse, label='Gaussian + Coreset', linestyle='dashed', marker='o', color='tab:blue', markersize=4)
    ax2.set_ylim(0.05, 0.5)
    
    lns = vcl_categorical + vcl_coreset_categorical + vcl_gaussian + vcl_coreset_gaussian
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower right' if idx == 0 else 'lower right', fontsize=10, handlelength=2.6)

    fig.savefig(f"plots/{'PermutedMNIST' if idx == 0 else 'SplitMNIST'}_Categorical_Vs_Gaussian.pdf", bbox_inches='tight')
    plt.close()