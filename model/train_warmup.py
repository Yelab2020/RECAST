import os
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import grad

import torch_geometric
from torch_geometric.nn import TransformerConv, GATConv, GATv2Conv, GCNConv, LayerNorm
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from model.model import main_classifier

from esda.join_counts import Join_Counts
from libpysal.weights import DistanceBand

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
  
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class TsallisEntropy(nn.Module):
    
    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape
        
        pred = F.softmax(logits / self.temperature, dim=1) 
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  
        
        sum_dim = torch.sum(pred * entropy_weight, dim = 0).unsqueeze(dim=0)
      
        return 1 / (self.alpha - 1) * torch.sum((1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim = -1)))

def model_define(num_classes=2, in_dim=1000, 
                 hiddens_graph=[512,256,128], hiddens_linear=[128,64,32],
                 layer_drug=3,
                 graphFunc="TransformerConv", 
                 drugFunc = "GINE",
                 dropout_rate=0.5, use_layer_norm=True, bulk_drop = 0):
    if graphFunc == "GCNConv":
        print('Using GCNConv')
        graphFunc = GCNConv
    elif graphFunc == "GATConv":
        print('Using GATConv')
        graphFunc = GATConv
    elif graphFunc == "GATv2Conv":
        print('Using GATv2Conv')
        graphFunc = GATv2Conv
    elif graphFunc == "TransformerConv":
        print('Using TransformerConv')
        graphFunc = TransformerConv
    else:
        raise NotImplementedError

    model = main_classifier(num_classes=num_classes, in_dim=in_dim, 
                            hiddens_graph=hiddens_graph, 
                            hiddens_linear=hiddens_linear,
                            layer_drug=layer_drug,
                            graphFunc=graphFunc,
                            drugFunc = drugFunc,
                            dropout_rate=dropout_rate, 
                            use_layer_norm=use_layer_norm, 
                            bulk_drop = bulk_drop)
    return model

def model_init(model):
    """Layer-wise parameter initialization strategy (three-level priority)"""
    for name, module in model.named_modules():
        # Priority 1: Use module's built-in initialization
        if hasattr(module, 'reset_parameters'):
            # print(f"Using built-in initialization: {name} ({module.__class__.__name__})")
            module.reset_parameters()

# Early stopping settings (lower is better)
class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True

# Calculate gradient norm for each loss
def compute_grad_norm(loss, params):
    grads = grad(loss, params, retain_graph=True, allow_unused=True)
    grad_norm = 0.0
    for g in grads:
        if g is not None:
            grad_norm += g.norm(2).item() ** 2
    return grad_norm ** 0.5

def train(model, src, tar, tar_twin, drug, num_classes, optimizer, scheduler, epoch, 
          grad_clip=0.5, warmup_epochs=25, temperature=4, alpha=1.8, threshold=0.8):
    
    model.train()

    src_pred, src_label_tarpred, tar_pred = model(src.x, src.edge_index, tar.x, tar.edge_index, drug)
    _, _, tar_pred_twin = model(src.x, src.edge_index, tar_twin.x, tar_twin.edge_index, drug)

    train_mask = src.train_idx

    tar_max_prob, pred_tar = torch.max(F.softmax(tar_pred), dim=-1)

    Lu = (F.cross_entropy(tar_pred_twin, pred_tar, 
                    reduction='none') * tar_max_prob.ge(threshold).float().detach()).mean()
    
    # Loss between pseudo-labeled source labels and true source labels
    reverse_loss = nn.MSELoss()(src_label_tarpred[train_mask], torch.nn.functional.one_hot(src.y.long()[train_mask], num_classes) - 1 / float(num_classes))

    # Loss of source prediction results
    src_loss = F.cross_entropy(src_pred[train_mask], src.y.long()[train_mask]) 

    # TsallisEntropy
    ts_loss = TsallisEntropy(temperature=temperature, alpha = alpha)
    transfer_loss = ts_loss(tar_pred)

    shared_params = list(model.graphNet.parameters()) + list(model.drugNet.parameters()) + list(model.attentionFusion.parameters()) + list(model.classifier.parameters())

    # Weight scheduling logic
    if epoch < warmup_epochs:
        # Warm-up phase: src_weight fixed at 0.5, remaining loss adaptively distributed 0.5
        src_weight = 0.5
        
        if Lu != 0:
            # Calculate gradient norms for transfer_loss, reverse_loss, Lu
            transfer_grad_norm = compute_grad_norm(transfer_loss, shared_params)
            reverse_grad_norm = compute_grad_norm(reverse_loss, shared_params)
            Lu_grad_norm = compute_grad_norm(Lu, shared_params)
            
            # Adaptive weights (inverse of gradient norms)
            transfer_weight = 1 / (transfer_grad_norm + 1e-8)
            reverse_weight = 1 / (reverse_grad_norm + 1e-8)
            Lu_weight = 1 / (Lu_grad_norm + 1e-8)
            
            # Normalize so that transfer_weight + reverse_weight + Lu_weight = 0.5
            total_other_weight = transfer_weight + reverse_weight + Lu_weight
            transfer_weight = (transfer_weight / total_other_weight) * 0.5
            reverse_weight = (reverse_weight / total_other_weight) * 0.5
            Lu_weight = (Lu_weight / total_other_weight) * 0.5

            loss = src_weight * src_loss + transfer_weight * transfer_loss + reverse_weight * reverse_loss + Lu_weight * Lu
        
        else:
            # If Lu == 0, distribute 0.5 only between transfer_loss and reverse_loss
            transfer_grad_norm = compute_grad_norm(transfer_loss, shared_params)
            reverse_grad_norm = compute_grad_norm(reverse_loss, shared_params)
            
            transfer_weight = 1 / (transfer_grad_norm + 1e-8)
            reverse_weight = 1 / (reverse_grad_norm + 1e-8)
            
            total_other_weight = transfer_weight + reverse_weight
            transfer_weight = (transfer_weight / total_other_weight) * 0.5
            reverse_weight = (reverse_weight / total_other_weight) * 0.5
            Lu_weight = 0.0

            loss = src_weight * src_loss + transfer_weight * transfer_loss + reverse_weight * reverse_loss
    
    else:
        # Adaptive phase: All loss weights based on gradient norms, summing to 1.0
        if Lu != 0:
            src_grad_norm = compute_grad_norm(src_loss, shared_params)
            transfer_grad_norm = compute_grad_norm(transfer_loss, shared_params)
            reverse_grad_norm = compute_grad_norm(reverse_loss, shared_params)
            Lu_grad_norm = compute_grad_norm(Lu, shared_params)
            
            src_weight = 1 / (src_grad_norm + 1e-8)
            transfer_weight = 1 / (transfer_grad_norm + 1e-8)
            reverse_weight = 1 / (reverse_grad_norm + 1e-8)
            Lu_weight = 1 / (Lu_grad_norm + 1e-8)
            
            total_weight = src_weight + transfer_weight + reverse_weight + Lu_weight
            src_weight = src_weight / total_weight
            transfer_weight = transfer_weight / total_weight
            reverse_weight = reverse_weight / total_weight
            Lu_weight = Lu_weight / total_weight

            loss = src_weight * src_loss + transfer_weight * transfer_loss + reverse_weight * reverse_loss + Lu_weight * Lu
        
        else:
            src_grad_norm = compute_grad_norm(src_loss, shared_params)
            transfer_grad_norm = compute_grad_norm(transfer_loss, shared_params)
            reverse_grad_norm = compute_grad_norm(reverse_loss, shared_params)
            
            src_weight = 1 / (src_grad_norm + 1e-8)
            transfer_weight = 1 / (transfer_grad_norm + 1e-8)
            reverse_weight = 1 / (reverse_grad_norm + 1e-8)
            
            total_weight = src_weight + transfer_weight + reverse_weight
            src_weight = src_weight / total_weight
            transfer_weight = transfer_weight / total_weight
            reverse_weight = reverse_weight / total_weight
            Lu_weight = 0.0

            loss = src_weight * src_loss + transfer_weight * transfer_loss + reverse_weight * reverse_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip)
    optimizer.step()
    # scheduler.step(loss)
    
    return loss, src_weight, transfer_weight, reverse_weight, Lu_weight, src_pred, tar_pred, tar_pred_twin

def validate(model, src, tar, drug, loc_data):
    model.eval()

    valid_loc = src.test_idx
    src_pred, _, tar_pred = model(src.x, src.edge_index, tar.x, tar.edge_index, drug)

    # Get valid set prediction results and true labels
    v_preds = torch.argmax(src_pred[valid_loc], dim=1).cpu().numpy()
    v_labels = src.y.long()[valid_loc].cpu().numpy()

    # Get valid set prediction probabilities (positive class probabilities) - use detach() to separate gradients
    v_probs = torch.softmax(src_pred[valid_loc], dim=1)[:, 1].detach().cpu().numpy()
    
    # Accuracy
    valid_acc = (v_preds == v_labels).mean()

    # F1 Score
    valid_f1 = f1_score(v_labels, 
                        v_preds, 
                        average='binary', 
                        pos_label=1) # pos_label specifies positive class as 1

    # AUC
    valid_auc = roc_auc_score(v_labels, v_probs)
    
    # AP
    valid_ap = average_precision_score(v_labels, v_probs)
    
    # Sensitive proportion
    choose = tar_pred.argmax(dim=1).cpu().numpy()
    sen_prop = sum(choose)/len(choose)

    # Join Count
    cancer_loc = loc_data.query("cell_type == 'Cancer'")
    
    # Construct binary adjacency weight matrix
    coordinates = cancer_loc[['x', 'y']].values
    w = DistanceBand(coordinates, threshold=5000, binary=True)
    
    # Check if 'choose' contains only 0s or only 1s
    if len(np.unique(choose)) < 2:
        # Return 0 if all 0s or all 1s
        bb, mean_bb, p_sim_bb = 0, 0, 0
    else:
        # Normal Join Counts calculation
        jc = Join_Counts(choose, w)
        bb, mean_bb, p_sim_bb = jc.bb, jc.mean_bb, jc.p_sim_bb
    
    return valid_acc, valid_f1, valid_auc, valid_ap, sen_prop, bb, mean_bb, p_sim_bb, choose
    
def train_epoch(bulkgraph, stgtaph, twingraph, drug_graph,
                loc_data, num_classes=2, in_dim=1000,
                hiddens_graph=[512,256,192], hiddens_linear=[128,64,32], layer_drug=3,
                graphFunc="GATv2Conv", drugFunc = "GINE",
                epoches=1000, lr=0.001, patience=25,
                weight_decay=0.001, grad_clip=0.5,warm_up=10,
                dropout_rate=0.5, use_layer_norm=True, bulk_drop = 0,
                temperature = 4, alpha = 1.8, threshold = 0.8,
                save_path='./', drug_name = None, kneigh=6, gene_num=500,
                test_size=0.2, sampling='SMOTE'):

    """
    dropout_rate: Network dropout
    dp: How much bulk expression to drop to simulate spatial transcriptomics
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model definition
    model = model_define(num_classes=num_classes, 
                         in_dim=in_dim, 
                         hiddens_graph=hiddens_graph, 
                         hiddens_linear=hiddens_linear, 
                         layer_drug=layer_drug,
                         graphFunc=graphFunc, 
                         drugFunc = drugFunc,
                         dropout_rate=dropout_rate, 
                         use_layer_norm=use_layer_norm, 
                         bulk_drop = bulk_drop).to(device)
    model_init(model)

    # Data definition
    src = bulkgraph.to(device)
    tar = stgtaph.to(device)
    tar_twin = twingraph.to(device)
    drug = drug_graph.to(device)

    # Optimization scheme definition
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=float(lr), 
                                weight_decay=float(weight_decay))
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, 
                                               patience=patience, verbose=False, 
                                               threshold=0.001, threshold_mode='rel', 
                                               cooldown=0, min_lr=0, eps=1e-09)
    
    
    early_stopping = EarlyStopping(patience=100)
    b_valid_f1 = float('-inf')

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for epoch in range(epoches):
        loss, src_weight, transfer_weight, reverse_weight, Lu_weight, src_pred, tar_pred, tar_pred_twin = train(model, src, tar, tar_twin, drug, num_classes,
                                                                                                                optimizer, scheduler, epoch, grad_clip, warm_up,
                                                                                                                temperature, alpha, threshold)
        valid_acc, valid_f1, valid_auc, valid_ap, sen_prop, bb, e_bb, sig_bb, choose = validate(model, src, tar, drug, loc_data)
        
        scheduler.step(valid_f1)
        
        if epoch >= 60:
            
            if b_valid_f1 < valid_f1:
                best_loss = loss
                b_src_weight = src_weight
                b_transfer_weight = transfer_weight
                b_reverse_weight = reverse_weight
                b_Lu_weight = Lu_weight
                
                best_epoch = epoch
                
                b_valid_acc = valid_acc  # Record accuracy of the best model
                b_valid_f1 = valid_f1
                b_valid_auc = valid_auc  # Record accuracy of the best model
                b_valid_ap = valid_ap
                
                b_sen_prop = sen_prop # Proportion of sensitive predictions
                
                b_bb = bb  
                b_e_bb = e_bb
                b_sig_bb = sig_bb

                b_state = {
                    'model': copy.deepcopy(model).cpu(),
                    'epoch': epoch,
                
                    'model_args': {
                        'num_classes': num_classes,
                        'in_dim': in_dim,
                        'hiddens_graph': hiddens_graph,
                        'hiddens_linear': hiddens_linear,
                        'layer_drug': layer_drug,
                        'graphFunc': graphFunc,
                        'drugFunc': drugFunc,
                        'dropout_rate': dropout_rate,
                        'use_layer_norm': use_layer_norm,
                        'bulk_drop': bulk_drop
                    },
                
                    'train_args': {
                        'lr': lr,
                        'weight_decay': weight_decay,
                        'grad_clip': grad_clip,
                        'warm_up': warm_up,
                        'alpha': alpha,
                        'temperature': temperature,
                        'threshold': threshold,
                        'sampling': sampling,
                        'gene_num': gene_num,
                        'k_neigh': kneigh,
                        'gene_num': gene_num,
                        'patience': patience
                    },
                
                    'results': {
                        'best_loss': best_loss,
                        'b_valid_auc': b_valid_auc,
                        'b_valid_ap': b_valid_ap,
                        'b_valid_acc': b_valid_acc,
                        'b_valid_f1': b_valid_f1,
                        'b_sen_prop': b_sen_prop,
                        'b_bb': b_bb,
                        'b_e_bb': b_e_bb,
                        'b_sig_bb': b_sig_bb,
                        'best_epoch': best_epoch
                    }
                }

                b_choose = choose.copy()

                

            early_stopping(1 - valid_f1)
            
            last_loss = loss
            last_epoch = epoch
            
            l_valid_acc = valid_acc  # Record accuracy of the best model
            l_valid_f1 = valid_f1
            l_sen_prop  = sen_prop
            
            l_bb = bb  
            l_e_bb = e_bb
            l_sig_bb = sig_bb
            
            if early_stopping.early_stop:
                break      
                
    torch.save(b_state, os.path.join(save_path, drug_name + '_best.pth'))
    np.save(os.path.join(save_path, drug_name + '_best.npy'), b_choose)
    
    return {
        'best_loss': best_loss,
        'b_valid_acc': b_valid_acc,
        'b_valid_f1': b_valid_f1,
        'b_valid_auc': b_valid_auc,
        'b_valid_ap': b_valid_ap,
        'b_sen_prop': b_sen_prop,
        'b_bb' : b_bb,
        'b_e_bb' : b_e_bb,
        'b_sig_bb' : b_sig_bb,
        'best_epoch': best_epoch,
        'last_loss': last_loss,
        'l_valid_acc': l_valid_acc,
        'l_valid_f1': l_valid_f1,
        'l_sen_prop': l_sen_prop,
        'l_bb' : l_bb,
        'l_e_bb' : l_e_bb,
        'l_sig_bb' : l_sig_bb,
        'last_epoch': last_epoch
}