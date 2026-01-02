import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), 'code'))

import numpy as np 
import torch 
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, classification_report

from utils.utils import _logger

# from model.model import NTXentLoss
# from model.model import TripletCenterLoss, compute_center_vectors

def move_to_device(data, device):
    """ 将包含张量的字典移动到指定设备 """
    for key, value in data.items():
        data[key] = value.to(device)
    return data




def train_or_eval_model(model, dataset_name, loss_function, consistency_loss, contrastive_loss, dataloader, epoch, optimizer=None, train=False, device='cpu', gammas=[], classify_flag='bin', train_id=[], test_id=[]):
    
    losses, preds, labels, masks = [], [], [], []

    #  loss
    loss_1_list, loss_2_list, loss_3_list = [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
        model.to(device)
    else:
        model.eval()
    
    model.to(device)

    id_list = []

    if device != 'cpu':
        # scaler = GradScaler(device) # 混合精度计算，加速
        scaler = GradScaler()
    
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # 准备模型输入的数据
        stimuls_data, response_data, y = data
        stimuls_data = move_to_device(stimuls_data, device) 
        response_data = move_to_device(response_data, device)
        
        y = y.to(device).long()
        
        fact_bs = y.size(0)
        # print('fact_bs:', fact_bs)
        # with autocast(device_type=device, enabled=True):   ##版本问题
        with autocast(enabled=True):
            predictions, features, attention_weights, positive_emo, neutral_emo, negative_emo, pos_interacted, neu_interacted, neg_interacted = model(stimuls_data, response_data)
            pred_labels = torch.argmax(predictions,1)
            
            # 计算损失
            loss_1 = loss_function(predictions, y)
            loss_2 = consistency_loss(positive_emo, neutral_emo, negative_emo, pos_interacted, neu_interacted, neg_interacted, y)
            loss_3 = contrastive_loss(positive_emo, neutral_emo, negative_emo)

            # loss_all
            loss_all = gammas[0]*loss_1 + gammas[1]*loss_2 + gammas[2]*loss_3  # 计算所有loss和
                
        preds.append(pred_labels.data.cpu().numpy())
        labels.append(y.data.cpu().numpy())


        losses.append(loss_all.item())
        loss_1_list.append(loss_1.item())
        loss_2_list.append(loss_2.item())
        loss_3_list.append(loss_3.item())


        if train:
            scaler.scale(loss_all).backward()

            scaler.step(optimizer)
            scaler.update()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        # masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_loss_1 = round(np.sum(loss_1_list)/len(loss_1_list), 4)
    avg_loss_2 = round(np.sum(loss_2_list)/len(loss_2_list), 4)
    avg_loss_3 = round(np.sum(loss_3_list)/len(loss_3_list), 4)

    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels, preds, average='binary')*100, 2)  

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, [avg_loss_1, avg_loss_2, avg_loss_3]



def train_or_eval_model_baseline(model, dataset_name, loss_function, dataloader, epoch, optimizer=None, train=False, device='cpu', gammas=[], classify_flag='bin', train_id=[], test_id=[]):
    
    losses, preds, labels, masks = [], [], [], []

    #  loss
    loss_1_list, loss_2_list, loss_3_list, loss_4_list, loss_5_list = [], [], [], [], []

    features_out = []

    label_type = 'y_bin' if classify_flag == 'bin' else 'y_penta'

    assert not train or optimizer!=None
    if train:
        model.train()
        model.to(device)
    else:
        model.eval()
    
    model.to(device)

    id_list = []
    if device != 'cpu':
        scaler = GradScaler(device) # 混合精度计算，加速
    
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # 准备模型输入的数据
        stimuls_data, response_data, y = data
        stimuls_data = move_to_device(stimuls_data, device) 
        response_data = move_to_device(response_data, device)
        
        y = y.to(device).long()
        
        fact_bs = y.size(0)
        # print('fact_bs:', fact_bs)
        
        with autocast(device_type=device, enabled=True):
            predictions, predictions_log, attention_weights = model(stimuls_data, response_data)
            pred_labels = torch.argmax(predictions,1)
        
            # 计算损失
            loss_1 = loss_function(predictions, y)
            
            # loss_all
            loss_all = loss_1 + 0  # 计算所有loss和
                
        preds.append(pred_labels.data.cpu().numpy())
        labels.append(y.data.cpu().numpy())


        losses.append(loss_all.item())
        loss_1_list.append(loss_1.item())

        if train:
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
            # update centers embedding 

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        # masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_loss_1 = round(np.sum(loss_1_list)/len(loss_1_list), 4)

    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted')*100, 2)  

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, [avg_loss_1]




def performance_testing(model, dataset_name, dataloader, device='cpu', average='binary'):
    preds, labels= [], []
    
    model.eval()
    
    model.to(device)

    id_list = []

    if device != 'cpu':
        # scaler = GradScaler(device) # 混合精度计算，加速
        scaler = GradScaler()
    
    for data in dataloader:
        # 准备模型输入的数据
        stimuls_data, response_data, y = data
        stimuls_data = move_to_device(stimuls_data, device) 
        response_data = move_to_device(response_data, device)
        
        y = y.to(device).long()
        
        fact_bs = y.size(0)
        # print('fact_bs:', fact_bs)
        # with autocast(device_type=device, enabled=True):   ##版本问题
        with autocast(enabled=True):
            predictions, predictions_log, attention_weights, positive_emo, neutral_emo, negative_emo, pos_interacted, neu_interacted, neg_interacted = model(stimuls_data, response_data)
            pred_labels = torch.argmax(predictions,1)
            
        preds.append(pred_labels.data.cpu().numpy())
        labels.append(y.data.cpu().numpy())

        # save id 
        # for id in x['id']:
        #     id_list.append(id.item())

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        # masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_accuracy = accuracy_score(labels, preds)
    avg_fscore = f1_score(labels, preds, average=average)
    avg_recall = recall_score(labels, preds, average=average)
    avg_precision = precision_score(labels, preds, average=average)

    return avg_accuracy, avg_recall, avg_precision, avg_fscore, labels, preds
    
