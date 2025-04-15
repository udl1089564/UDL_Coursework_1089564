import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt

from vcl import VCL

import data
from models import PermutedModel
from utils import attach_random_coreset_permuted

num_tasks = 10
single_head = True
coreset_method = attach_random_coreset_permuted
device = 'cuda'

def permuted_gaussian_noDINO(num_epochs=100, batch_size=256, coreset_size=None, beta=1):
    filename = 'permuted_gaussian_noDINO'
    assert coreset_size.__class__ is list
    dataloaders = data.get_permuted_dataloaders(num_tasks, batch_size)
    model = PermutedModel()
    model.to(device)

    vcl = VCL(model, 'gaussian', beta=beta, device=device)

    for i in range(len(coreset_size)):
        csz = coreset_size[i]
        model = PermutedModel()
        model.to(device)
        if "{}-{}.npy".format(filename, csz) in os.listdir("logs/"):
            print("Loading existing checkpoint..")
            all_mse = np.load("saved_outputs/{}-{}.npy".format(filename, csz))
        else:
            all_mse = vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
                                   model, coreset_method, csz, beta, checkpoint=f'{filename}.pth')
            np.save('saved_outputs/{}-{}.npy'.format(filename, csz), all_mse)
        rmse = np.sqrt(np.nanmean(all_mse, axis=1))
        print("RMSE after each task:", rmse)

if __name__=='__main__':
    num_epochs = 100
    batch_size = 256
    coreset_size = [0, 200]
    permuted_gaussian_noDINO(num_epochs=num_epochs, batch_size=batch_size, coreset_size=coreset_size, beta=1)