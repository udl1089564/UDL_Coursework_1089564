import torch
import torch.nn.functional as F
from transformers import Dinov2Backbone
from torchvision import transforms
import torchvision

from utils import KL_DIV

class DINO(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.dino_backbone = Dinov2Backbone.from_pretrained('facebook/dinov2-small')
        # self.dino_dim = self.dino_backbone.encoder.config.hidden_size

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        for param in self.dinov2.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Upsample the image to 224 x 224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Grayscale to RGB
        x = x.repeat(1, 3, 1, 1)

        # Normalize
        x = self.normalize(x)

        dino_outputs = self.dinov2.forward_features(x)
        cls_token = dino_outputs['x_norm_clstoken']

        return cls_token
    
class BBBLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.01,
            }
        self.prior_W_mu = torch.tensor(priors['prior_mu'])
        self.prior_W_sigma = torch.tensor(priors['prior_sigma'])
        self.prior_bias_mu = torch.tensor(priors['prior_mu'])
        self.prior_bias_sigma = torch.tensor(priors['prior_sigma'])

        self.W_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(0, 0.1)
        self.W_rho.data.fill_(-3)

        if self.use_bias:
            self.bias_mu.data.normal_(0, 0.1)
            self.bias_rho.data.fill_(-3)

    def forward(self, x, sample=True):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(act_mu.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = KL_DIV(self.prior_W_mu, self.prior_W_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_bias_mu, self.prior_bias_sigma, self.bias_mu, self.bias_sigma)
        return kl
    
class PermutedModel(torch.nn.Module):
    def __init__(self, starting_dim=784):
        """
        input: 1 x 28 x 28
        output: 1 classifiers with 10 nodes
        hidden: [100, 100]
        """
        super().__init__()
        self.fc1 = BBBLinear(starting_dim, 100)
        self.fc2 = BBBLinear(100, 100)
        self.classifier = BBBLinear(100, 10)
        self.starting_dim = starting_dim
        # self.classifiers = torch.nn.ModuleList([BBBLinear(100, 10) for i in range(5)])

    def forward(self, x, task_id):
        out = x.view(-1, self.starting_dim)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.classifier(out)

    def get_kl(self, task_id):
        kl = 0.0
        kl += self.fc1.kl_loss()
        kl += self.fc2.kl_loss()
        kl += self.classifier.kl_loss()
        return kl

    def update_prior(self):
        self.fc1.prior_W_mu = self.fc1.W_mu.data
        self.fc1.prior_W_sigma = self.fc1.W_sigma.data
        if self.fc1.use_bias:
            self.fc1.prior_bias_mu = self.fc1.bias_mu.data
            self.fc1.prior_bias_sigma = self.fc1.bias_sigma.data

        self.fc2.prior_W_mu = self.fc2.W_mu.data
        self.fc2.prior_W_sigma = self.fc2.W_sigma.data
        if self.fc2.use_bias:
            self.fc2.prior_bias_mu = self.fc2.bias_mu.data
            self.fc2.prior_bias_sigma = self.fc2.bias_sigma.data

        self.classifier.prior_W_mu = self.classifier.W_mu.data
        self.classifier.prior_W_sigma = self.classifier.W_sigma.data
        if self.classifier.use_bias:
            self.classifier.prior_bias_mu = self.classifier.bias_mu.data
            self.classifier.prior_bias_sigma = self.classifier.bias_sigma.data


class SplitModel(torch.nn.Module):
    def __init__(self, starting_dim=784):
        """
        input: 1 x 28 x 28
        output: 5 classifiers 2 nodes each
        hidden: [256, 256]
        """
        super().__init__()
        self.fc1 = BBBLinear(starting_dim, 256)
        self.fc2 = BBBLinear(256, 256)
        self.classifiers = torch.nn.ModuleList([BBBLinear(256, 2) for i in range(5)])
        self.starting_dim = starting_dim

    def forward(self, x, task_id):
        out = x.view(-1, self.starting_dim)
        for layer in self.children():
            if layer.__class__ is not torch.nn.ModuleList:
                out = F.relu(layer(out))
        return self.classifiers[task_id](out)

    def get_kl(self, task_id):
        kl = 0.0
        for layer in self.children():
            if layer.__class__ is not torch.nn.ModuleList:
                kl += layer.kl_loss()
        kl += self.classifiers[task_id].kl_loss()
        return kl

    def update_prior(self):
        for layer in self.children():
            if layer.__class__ is not torch.nn.ModuleList:
                layer.prior_W_mu = layer.W_mu.data
                layer.prior_W_sigma = layer.W_sigma.data
                if layer.use_bias:
                    layer.prior_bias_mu = layer.bias_mu.data
                    layer.prior_bias_sigma = layer.bias_sigma.data