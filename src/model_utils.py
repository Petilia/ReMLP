import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torchvision.models import resnet18


class MPLReparametrization(nn.Module):
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
        return output.to(input.device)


def get_basic_resnet(device="cuda"):
    model = resnet18(pretrained=False)
    model.fc == nn.Linear(512, 10)
    model.to(device)
    return model

def get_reparam_resnet(hidden_sizes, device="cuda"):
    model = resnet18(pretrained=False)
    model.fc == nn.Linear(512, 10)
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parametrize.register_parametrization(module, "weight", MPLReparametrization(module, hidden_sizes))
        
    model.to(device)
    return model