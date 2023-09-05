import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torchvision.models import resnet18
from math import sqrt

class ReMPL(nn.Module):
    def __init__(self, conv, reduction=1, size_thresh=128, save_low_param=False):
        super().__init__()
        self.n_1, self.n_2 = conv.weight.shape[0], conv.weight.shape[1]
        self.kernel_shape = conv.weight.shape[2:]
        
        total_conv_param = self.n_1 * self.n_2 * self.kernel_shape[0] * self.kernel_shape[1]
        
        # if save_low_param and total_conv_param < 40000:
        #     reduction = 0.5
            
        total_conv_param = reduction * total_conv_param
        
        #determ mlp hidden_size and and n_layers
        n_hiddens = 1
        while True:
            ks = self.kernel_shape[0] * self.kernel_shape[1]
            pos_sol =  1 / (2 * n_hiddens)  * (-(2 + ks) + sqrt((2 + ks)**2 + 4 * n_hiddens * total_conv_param))
            hidden_size = int(pos_sol)
            if hidden_size > size_thresh:
                n_hiddens = n_hiddens + 1
            else:
                break
            
        print(hidden_size, n_hiddens)
            
        modulelist = nn.ModuleList()
        modulelist.append(nn.Linear(2, hidden_size))
        modulelist.append(nn.ReLU())

        for _ in range(n_hiddens):
            modulelist.append(nn.Linear(hidden_size, hidden_size))
            modulelist.append(nn.ReLU())
    
        modulelist.append(nn.Linear(hidden_size, self.kernel_shape[0] * self.kernel_shape[1]))
        self.mlp = nn.Sequential(*modulelist)
        
    def forward(self, input):
        x = torch.linspace(-1, 1, self.n_1).to(input.device)
        y = torch.linspace(-1, 1, self.n_2).to(input.device)
        xx, yy = torch.meshgrid(x, y)
        xy = torch.stack((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1).squeeze(2)

        output = self.mlp(xy)
        output = output.view(self.n_1, self.n_2, *self.kernel_shape)
        return output


class SameReMPL(nn.Module):
    def __init__(self, conv, hidden_sizes):
        super().__init__()

        self.n_1, self.n_2 = conv.weight.shape[0], conv.weight.shape[1]
        self.kernel_shape = conv.weight.shape[2:]
     
        model_list = nn.ModuleList()

        for index in range(len(hidden_sizes)):
            if index == 0:
                model_list.append(nn.Linear(2, hidden_sizes[index]))
                model_list.append(nn.ReLU())
            elif index == len(hidden_sizes) - 1:
                model_list.append(nn.Linear(hidden_sizes[index-1], hidden_sizes[index]))
                model_list.append(nn.ReLU())
                model_list.append(nn.Linear(hidden_sizes[index], self.kernel_shape[0] * self.kernel_shape[1] ))
            else:
                model_list.append(nn.Linear(hidden_sizes[index-1], hidden_sizes[index]))
                model_list.append(nn.ReLU())
                
        self.mlp = nn.Sequential(*model_list)
        
    def forward(self, input):
        x = torch.linspace(-1, 1, self.n_1).to(input.device)
        y = torch.linspace(-1, 1, self.n_2).to(input.device)
        xx, yy = torch.meshgrid(x, y)
        xy = torch.stack((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1).squeeze(2)

        output = self.mlp(xy)
        output = output.view(self.n_1, self.n_2, *self.kernel_shape)
        return output


def get_basic_resnet(device="cuda"):
    model = resnet18(pretrained=False)
    model.fc == nn.Linear(512, 10)
    model.to(device)
    return model

def get_reparam_resnet(reduction=1, size_thresh=128, save_low_param=False, device="cuda"):
    model = resnet18(pretrained=False)
    model.fc == nn.Linear(512, 10)
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parametrize.register_parametrization(module, "weight", ReMPL(module, reduction, size_thresh, save_low_param))
        
    model.to(device)
    return model

def get_same_reparam_resnet(hidden_sizes, device="cuda"):
    model = resnet18(pretrained=False)
    model.fc == nn.Linear(512, 10)
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parametrize.register_parametrization(module, "weight", SameReMPL(module, hidden_sizes))
        
    model.to(device)
    return model

