import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from metric_utils import calc_metrics, log_wandb


def train_epoch(model, optimizer, criterion, trainloader, epoch, device="cuda"):
    preds = []
    gt = []
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        _, predictions = torch.max(outputs, 1)
        preds += predictions.cpu().tolist()
        gt += labels.cpu().tolist()
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    running_loss /= i
    metrics = calc_metrics(gt, preds)
    metrics["loss"] = running_loss
    log_wandb(metrics, "train", epoch)
    return metrics
        
            
def val_epoch(model, criterion, testloader, epoch, device="cuda"):
    preds = []
    gt = []
    
    with torch.no_grad():
        running_loss = 0.0
        for i, data in tqdm(enumerate(testloader, 0)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predictions = torch.max(outputs, 1)
            preds += predictions.cpu().tolist()
            gt += labels.cpu().tolist()
    
            running_loss += loss.item()
            
        running_loss /= i
        metrics = calc_metrics(gt, preds)
        metrics["loss"] = running_loss
        log_wandb(metrics, "val", epoch)
        return metrics


