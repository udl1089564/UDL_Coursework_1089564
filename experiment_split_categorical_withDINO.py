import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vcl import VCL
import data
from models import SplitModel
from utils import attach_random_coreset_split

num_tasks = 5
single_head = False
coreset_method = attach_random_coreset_split
device = 'cuda'

class_distribution = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]

def split_categorical_withDINO(num_epochs=100, batch_size=15000, coreset_size=None, beta=1):
    filename = 'split_categorical_withDINO'
    assert coreset_size.__class__ is list
    dataloaders = data.get_split_dataloaders(
        class_distribution, dino_transform=True, batch_size=batch_size)
    model = SplitModel(starting_dim=384)   # 384 or 768 depending on DINO small or base

    model.to(device)

    vcl = VCL(model, 'categorical', beta=beta, device=device)

    for i in range(len(coreset_size)):
        model = SplitModel(starting_dim=384)
        model.to(device)
        csz = coreset_size[i]

        if "{}-{}.npy".format(filename, csz) in os.listdir("logs/"):
            print("Loading existing checkpoint..")
            all_accs = np.load("saved_outputs/{}-{}.npy".format(filename, csz))
        else:
            all_accs = vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
                               model, coreset_method, csz, beta, checkpoint=f'{filename}.pth')
        
            np.save('saved_outputs/{}-{}.npy'.format(filename, csz), all_accs)

        accs = np.nanmean(all_accs, axis=1)
        print("Average accuracy after each task:", accs)

if __name__=='__main__':
    num_epochs = 15
    batch_size = 256
    coreset_size = [0, 200]
    split_categorical_withDINO(num_epochs=num_epochs, batch_size=batch_size, coreset_size=coreset_size, beta=1)