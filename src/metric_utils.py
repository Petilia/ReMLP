import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def log_wandb(metrics, mode, epoch=None):
    log_dict = {
                f"{mode}_accuracy": metrics["accuracy"],  
                f"{mode}_recall": metrics["recall"], 
                f"{mode}_precision": metrics["precision"], 
                f"{mode}_f1": metrics["f1"], 
                f"{mode}_loss": metrics["loss"]
                }
    if epoch is not None:
        log_dict["epoch"] = epoch
    
    wandb.log(log_dict)
    
def determ_best_stats(all_stats):
    best_acc = max([i["accuracy"] for i in all_stats])
    best_recall = max([i["recall"] for i in all_stats])
    best_precision = max([i["precision"] for i in all_stats])
    best_f1 = max([i["f1"] for i in all_stats])
    best_loss = min([i["loss"] for i in all_stats])
    return {"accuracy" : best_acc,
            "recall" : best_recall,
            "precision" : best_precision,
            "f1" : best_f1,
            "loss" : best_loss
            }
    
def calc_metrics(gt, preds):
    accuracy = accuracy_score(gt, preds)
    recall = recall_score(gt, preds, average='micro')
    precision = precision_score(gt, preds, average='micro')
    f1 = f1_score(gt, preds, average='micro')
    
    return {"accuracy" : accuracy,
            "recall" : recall,
            "precision" : precision,
            "f1" : f1
            }