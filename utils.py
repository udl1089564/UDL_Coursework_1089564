import numpy as np
import torch
import torch.nn.functional as F

class ELBO(torch.nn.Module):
    def __init__(self, model, log_likelihood_loss, beta):
        super().__init__()
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.log_likelihood_loss = log_likelihood_loss
        self.beta = beta

    def forward(self, y, y_hat, kl):
        return self.log_likelihood_loss(y, y_hat) + self.beta * kl / self.num_params
    
def gaussian_log_likelihood_loss(y_hat, y):
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=y_hat.shape[1]).float()
    return 0.5 * F.mse_loss(y_one_hot, y_hat, reduction='sum')


def calculate_accuracy(outputs, targets):
    return np.mean(outputs.argmax(dim=-1).cpu().numpy() == targets.cpu().numpy())

def KL_DIV(mu_p, sig_p, mu_q, sig_q):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def attach_random_coreset_split(coresets, sub_train_loader, num_samples=200):
    """
    coresets: list of collection of coreset dataloaders
    sub_train_loader: loader from which a random coreset is to be drawn
    num_samples: number of samples in each coreset
    """
    task_indices = sub_train_loader.sampler.indices
    shuffled_task_indices = task_indices[torch.randperm(len(task_indices))]
    coreset_indices = shuffled_task_indices[:num_samples]
    sub_train_loader.sampler.indices = shuffled_task_indices[num_samples:]  # Delete coreset from orginal data
    coreset_sampler = torch.utils.data.SubsetRandomSampler(coreset_indices)
    coreset_loader = torch.utils.data.DataLoader(
        sub_train_loader.dataset, batch_size=sub_train_loader.batch_size, sampler=coreset_sampler)
    coresets.append(coreset_loader)


def attach_random_coreset_permuted(coresets, sub_train_loader, num_samples=200):
    """
    coresets: list of collection of coreset dataloaders
    sub_train_loader: loader from which a random coreset is to be drawn
    num_samples: number of samples in each coreset
    """
    shuffled_task_indices = torch.randperm(len(sub_train_loader.dataset))
    coreset_indices = shuffled_task_indices[:num_samples]
    coreset_sampler = torch.utils.data.SubsetRandomSampler(coreset_indices)
    coreset_loader = torch.utils.data.DataLoader(
        sub_train_loader.dataset, batch_size=sub_train_loader.batch_size, sampler=coreset_sampler)
    coresets.append(coreset_loader)