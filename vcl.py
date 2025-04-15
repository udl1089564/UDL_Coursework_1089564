# Largely based on: https://github.com/Piyush-555/VCL-in-PyTorch/blob/main/vcl.py
import os
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

from models import DINO
from utils import ELBO, calculate_accuracy, gaussian_log_likelihood_loss

class VCL():
    def __init__(self, model, mode, beta=1, device='cuda'):
        self.model = model
        
        if mode == 'categorical':
            log_likelihood_loss = F.nll_loss
        elif mode == 'gaussian':
            log_likelihood_loss = gaussian_log_likelihood_loss
        else:
            raise ValueError("Invalid mode. Choose 'categorical' or 'gaussian'.")
        
        self.elbo = ELBO(model, log_likelihood_loss, beta)

        self.mode = mode
        self.device = device

    def train(self, model, num_epochs, dataloader, single_head, task_id, beta, replay=False):
        beta = 0 if replay else beta
        lr_start = 1e-3

        if single_head:
            offset = 0
            output_nodes = 10
        else:
            output_nodes = model.classifiers[0].out_features
            offset = task_id * output_nodes

        # train_size = len(dataloader.dataset) if single_head else dataloader.sampler.indices.shape[0]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

        model.train()
        for epoch in tqdm(range(num_epochs)):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                targets -= offset

                net_out = model(inputs, task_id)

                if self.mode == 'categorical':
                    output = F.log_softmax(net_out, dim=-1)
                else:
                    output = net_out

                kl = model.get_kl(task_id)
                
                loss = self.elbo(output, targets, kl)

                loss.backward()
                optimizer.step()

    def predict(self, model, dataloader, single_head, task_id):
        if single_head:
            offset = 0
            output_nodes = 10
        else:
            output_nodes = model.classifiers[0].out_features
            offset = task_id * output_nodes

        model.train()
        output_metric = []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            targets -= offset

            with torch.no_grad():
                net_out = model(inputs, task_id)

                if self.mode == 'categorical':
                    log_output = F.log_softmax(net_out, dim=-1)

            if self.mode == 'categorical':
                output_metric.append(calculate_accuracy(log_output, targets))
            else:
                # Convert target to one-hot encoding
                target_one_hot = torch.nn.functional.one_hot(targets, num_classes=net_out.shape[1]).float()
                output_metric.append(np.mean((net_out.cpu().numpy() - target_one_hot.cpu().numpy()) ** 2))
        
        return np.mean(output_metric)


    def run_vcl(self, num_tasks, single_head, num_epochs, dataloaders, model,
                coreset_method, coreset_size=0, beta=1, update_prior=True, checkpoint=None):

        if not single_head:
            assert 10 // num_tasks == 10 / num_tasks
        
        coreset_list = []
        all_metrics = np.empty(shape=(num_tasks, num_tasks))
        all_metrics.fill(np.nan)
        for task_id in range(num_tasks):

            # Initialize first model using means from MLE fit and small variance
            if task_id == 0:
                # Hacky way to save time, load saved checkpoint that was saved from other script
                if os.path.exists(checkpoint):
                    loaded_checkpoint = torch.load(checkpoint)
                    self.model.load_state_dict(loaded_checkpoint['model_state_dict'])

            # Train on non-coreset data
            trainloader, testloader = dataloaders[task_id]
            self.train(model, num_epochs, trainloader, single_head, task_id, beta)

            # Attach a new coreset
            if coreset_size > 0:
                coreset_method(coreset_list, trainloader, num_samples=coreset_size)

                # Replay old tasks using coresets
                for task in range(task_id + 1):
                    self.train(model, num_epochs, coreset_list[task], single_head, task, beta, replay=True)

            # Evaluate on old tasks
            for task in range(task_id + 1):
                _, testloader_i = dataloaders[task]
                metric = self.predict(model, testloader_i, single_head, task)
                all_metrics[task_id][task] = metric

            if update_prior:
                model.update_prior()
        return all_metrics