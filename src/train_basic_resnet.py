import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from train_utils import train_epoch, val_epoch
from metric_utils import determ_best_stats, log_wandb
from data_utils import get_cifar_loaders
from model_utils import get_basic_resnet

wandb.login()

batch_size = 32
n_epoch = 40
lrs = [1e-5, 1e-4, 1e-3, 1e-2]

for lr in lrs:

    name_run = f"Basic resnet lr={lr}, batch_size={batch_size}"

    wandb.init(
        project="ReMLP",
        name=name_run
    )

    model = get_basic_resnet()
    trainloader, testloader, _ = get_cifar_loaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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