import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from train_utils import train_epoch, val_epoch              
from metric_utils import determ_best_stats, log_wandb
from data_utils import get_cifar_loaders
from model_utils import get_reparam_resnet       

wandb.login()

reductions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
size_thresh = 512
save_low_param = False

batch_size = 384
n_epoch = 250
lr = 1e-2

for reduction in reductions:
    print(reduction)
    model = get_reparam_resnet(reduction, size_thresh, save_low_param)
    trainloader, testloader, _ = get_cifar_loaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    name_run=f"Reparam resnet, red={reduction}, size_thresh={size_thresh}, bs={batch_size}, lr={lr}, save_low_param={save_low_param} "

    wandb.init(
        project="ReMLP",
        name=name_run
    )

    all_train_stats = []
    all_val_stats = []

    for epoch in range(n_epoch):  
        train_stats = train_epoch(model, optimizer, criterion, trainloader, epoch)
        val_stats = val_epoch(model, criterion, testloader, epoch)
        all_train_stats.append(train_stats)
        all_val_stats.append(val_stats)
        
    best_train_stats = determ_best_stats(all_train_stats)
    best_val_stats = determ_best_stats(all_val_stats)
    log_wandb(best_train_stats, "best_train")
    log_wandb(best_val_stats, "best_val")

    wandb.finish()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           