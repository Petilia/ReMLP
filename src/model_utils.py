import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torchvision.models import resnet18
from math import sqrt

class OneLayerReMPL(nn.Module):
    def __init__(self, conv, reduction=1):
        super().__init__()

        self.n_1, self.n_2 = conv.weight.shape[0], conv.weight.shape[1]
        self.kernel_shape = conv.weight.shape[2:]
        
        total_conv_param = self.n_1 * self.n_2 * self.kernel_shape[0] * self.kernel_shape[1]
        total_conv_param = reduction * total_conv_param
        
        # eq for 2-layer perceptron:
        # hidden_size**2 + hidden_size * (2 + kernel_shape[0] * kernel_shape[1]) - total_conv_param = 0
        # h*2 + h * (2 + ks) - total = 0
        # pos_sol = 0.5(-(2+ks) + sqrt( (2+ks)**2 + 4*total))
        # ks = self.kernel_shape[0] * self.kernel_shape[1]
        # pos_sol =  0.5 * (-(2 + ks) + sqrt((2 + ks)**2 + 4 * total_conv_param))
        # hidden_size = int(pos_sol)
        # self.mlp = nn.Sequential(nn.Linear(2, hidden_size), 
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, hidden_size), 
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, self.kernel_shape[0] * self.kernel_shape[1] ))  
        
        # eq for 3-layer perceptron:
        # 2*hidden_size**2 + hidden_size * (2 + kernel_shape[0] * kernel_shape[1]) - total_conv_param = 0
        # 2*h*2 + h * (2 + ks) - total = 0
        # pos_sol = 0.25(-(2+ks) + sqrt( (2+ks)**2 + 8*total))
        # ks = self.kernel_shape[0] * self.kernel_shape[1]
        # pos_sol =  0.25 * (-(2 + ks) + sqrt((2 + ks)**2 + 8 * total_conv_param))
        # hidden_size = int(pos_sol)
        
        # self.mlp = nn.Sequential(nn.Linear(2, hidden_size), 
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, hidden_size), 
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, hidden_size), 
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, self.kernel_shape[0] * self.kernel_shape[1] ))  
        
        ks = self.kernel_shape[0] * self.kernel_shape[1]
        pos_sol =  0.125 * (-(2 + ks) + sqrt((2 + ks)**2 + 16 * total_conv_param))
        hidden_size = int(pos_sol)
        
        self.mlp = nn.Sequential(nn.Linear(2, hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, self.kernel_shape[0] * self.kernel_shape[1] )) 
        
        print(hidden_size)
        
  
    
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

def get_same_reparam_resnet(hidden_sizes, device="cuda"):
    model = resnet18(pretrained=False)
    model.fc == nn.Linear(512, 10)
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parametrize.register_parametrization(module, "weight", SameReMPL(module, hidden_sizes))
        
    model.to(device)
    return model

def get_one_layer_reparam_resnet(reduction=1, device="cuda"):
    model = resnet18(pretrained=False)
    model.fc == nn.Linear(512, 10)
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parametrize.register_parametrization(module, "weight", OneLayerReMPL(module, reduction))
        
    model.to(device)
    return model