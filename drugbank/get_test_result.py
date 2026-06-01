import os 
import pandas as pd 
import numpy as np
import torch
from model import SA_DDI_kge as SA_DDI
from data_preprocessing import CustomData
from dataset import load_ddi_dataset
from metrics import *

def val(model,dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        # head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]
        head_pairs, tail_pairs, rel, label,head_embedding,tail_embedding = [d.to(device) for d in data]
        with torch.no_grad():
            pred = model((head_pairs, tail_pairs, rel),head_embedding,tail_embedding)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)
    return acc, auroc, f1_score, precision, recall, ap

if __name__ == "__main__":
    device = torch.device('cuda:0')
    train_loader, val_loader, test_loader = load_ddi_dataset(root='data/preprocessed/drugbank', batch_size=256, fold=0)
    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim = data[0].edge_attr.size(-1)

    model_dir = r'E:\DDI\SA-DDI-test\drugbank\save\20230502_182427_drugbank (CNN+TGAP+GAT+INTRA+LN FOLD1)\model'
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir,model_name)
        model = SA_DDI(node_dim, edge_dim, n_iter=10).to(device)
        model.load_state_dict(torch.load(model_path))
        acc, auroc, f1_score, precision, recall, ap = val(model,test_loader, device)
        print(model_name,"acc:",acc, "auroc:",auroc, "f1-score:",f1_score, "precision:",precision,"recall:", recall,"ap:",ap)

