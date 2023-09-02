import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from train_utils import train_epoch, val_epoch
from metric_utils import determ_best_stats, log_wandb
from data_utils import get_cifar_loaders
from model_utils import get_reparam_resnet

wandb.login()

n_epoch = 15

all_hidden_sizes = [ 
                    [64, 128, 128, 128], 
                    [64, 128, 128, 128, 128], 
                    [128, 256, 256, 256], 
                    [128, 256, 256, 256, 256], 
                    [256, 256, 256, 256, 256],
                    [256, 256, 256, 256, 256, 256]
                    ]

for hidden_sizes in all_hidden_sizes:

    model = get_reparam_resnet(hidden_sizes)
    trainloader, testloader, _ = get_cifar_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    subname = "_".join(str(i) for i in hidden_sizes)
    name_run=f"Reparam resnet mlp_{subname}"

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