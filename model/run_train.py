import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), 'code'))

import numpy as np 

import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.optim as optim

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import time
import copy

from utils.utils import _logger
from model.model import create_model, create_bert_basline_model
from model.model import ContrastiveLearning, EmotionalConsistencyLoss
from model.train import train_or_eval_model, train_or_eval_model_baseline, performance_testing
import model.config as config

def run(dataset, dataset_name, logdir_root, classify_flag, device, gammas, SEED, BS, LR, Weight_Decay, Dropout=0.5, EPOCHS=50, K=5, alpha=1, beta=1, return_id=False):

    # 设置 K 折交叉验证的 K 值
    K = 5

    # 创建 KFold 对象
    kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
    

    fold_index = 0

    test_acc_perfold = []

    test_best_fscore_perfold = []
    test_best_acc_perfold = []


    #save label and pred 
    test_label_list_perfold = []
    test_pred_list_perfold = []
    test_id_list_perfold = []

    # save id
    train_id_perfold = []
    test_id_perfold = []

    best_model_list = []

    for train_index, test_index in kf.split(dataset):
        # 创建训练集和测试集子集
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        
        # 创建训练集和测试集的 DataLoader
        train_loader = DataLoader(train_subset, batch_size=BS, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BS, shuffle=False)
        
        logdir = logdir_root + '_runs/' + datetime.now().strftime('%m-%d_%H-%M') + f'_{dataset_name}_{classify_flag}_lr{LR}_dropout{Dropout}_bs{BS}_epochs{EPOCHS}_fold{fold_index+1}'

        writer = SummaryWriter(logdir)
        logger = _logger(os.path.join(logdir, f'logger_fold{fold_index+1}.txt'))
        
        # 打印参数
        logger.debug('bert_model_path: {}'.format(config.ptm))

        best_model = None
        model = create_model(dataset_name=dataset_name, device=device, hidden_size=768, dropout=Dropout)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=Weight_Decay)

        loss_function = torch.nn.CrossEntropyLoss()
        consist_loss = EmotionalConsistencyLoss(alpha=alpha, beta=beta)
        contra_loss = ContrastiveLearning()

        best_fscore, best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None
        all_fscore, all_acc, all_loss = [], [], []

        for e in range(EPOCHS):
            start_time = time.time()

            train_loss, train_acc, train_labels, train_preds, train_fscore, train_loss_list = train_or_eval_model(
                model=model, dataset_name=dataset_name, loss_function=loss_function, consistency_loss=consist_loss, contrastive_loss=contra_loss, 
                dataloader=train_loader, epoch=EPOCHS, optimizer=optimizer, train=True, device=device, gammas=gammas, classify_flag=classify_flag, train_id=[])

            test_loss, test_acc, test_label, test_pred, test_fscore, test_loss_list = train_or_eval_model(
                model=model, dataset_name=dataset_name, loss_function=loss_function, consistency_loss=consist_loss, contrastive_loss=contra_loss, 
                dataloader=test_loader, epoch=EPOCHS, optimizer=optimizer, train=False, device=device, gammas=gammas, classify_flag=classify_flag, test_id=[])
            
            
            # 记录到 TensorBoard
            writer.add_scalar('Loss/train', train_loss, e)
            writer.add_scalar('Accuracy/train', train_acc, e)
            writer.add_scalar('F-Score/train', train_fscore, e)
            writer.add_scalar('Loss/test', test_loss, e)
            writer.add_scalar('Accuracy/test', test_acc, e)
            writer.add_scalar('F-Score/test', test_fscore, e)

            for idx in range(len(train_loss_list)):
                writer.add_scalar(f'Loss_{idx+1}/train', train_loss_list[idx], e)
            for idx in range(len(test_loss_list)):
                writer.add_scalar(f'Loss_{idx+1}/test', test_loss_list[idx], e)

            all_fscore.append(test_fscore)
            all_acc.append(test_acc)


            if best_acc == None or best_acc < test_acc:
                best_acc = test_acc
                best_label, best_pred = test_label, test_pred, 

                # best_mask, ids = test_mask, id_test
                # for x, y in test_loader:
                #     print('id: ', x['id'])

                best_model = copy.deepcopy(model)

                
            logger.debug('fold: {}, epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                    format(fold_index+1, e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
          
            # model saved 
            if (e+1) == EPOCHS:
                logger.debug(classification_report(best_label, best_pred, digits=4))
                logger.debug(confusion_matrix(best_label, best_pred))

                # save model
                file_name = f'{dataset_name}_model_fold{fold_index + 1}_acc.pt'
                save_path = os.path.join(logdir, 'saved_model/')
                os.makedirs(save_path, exist_ok=True)
                print('best model saved in: ', os.path.join(save_path, file_name))
                torch.save(best_model, os.path.join(save_path, file_name))
                best_model_list.append(best_model)

        
                #save label and pred 
                test_label_list_perfold.append(best_label)
                test_pred_list_perfold.append(best_pred)
                # test_id_list_perfold.append(ids)
                
        logger.debug('Test performance..')
        logger.debug('fold: {}, F-Score: {}'.format(fold_index+1, max(all_fscore)))
        logger.debug('fold: {}, F-Score-index: {}'.format(fold_index+1, all_fscore.index(max(all_fscore)) + 1))
        logger.debug('fold: {}, Acc: {}'.format(fold_index+1, max(all_acc)))
        logger.debug('fold: {}, Acc-index: {}'.format(fold_index+1, all_acc.index(max(all_acc)) + 1))
        test_best_fscore_perfold.append(max(all_fscore))
        test_best_acc_perfold.append(max(all_acc))
        fold_index += 1

        writer.close()


    print('5-fold ave fscore:', np.mean(np.array(test_best_fscore_perfold)))
    print('5-fold ave acc:', np.mean(np.array(test_best_acc_perfold)))

    if return_id:
        return test_label_list_perfold, test_pred_list_perfold, test_id_list_perfold, test_best_acc_perfold, test_best_fscore_perfold, train_id_perfold, test_id_perfold
    else:
        return test_label_list_perfold, test_pred_list_perfold, test_id_list_perfold, test_best_acc_perfold, test_best_fscore_perfold

def run_bert_baseline(dataset, dataset_name, logdir_root, classify_flag, device, gammas, SEED, BS, LR, Weight_Decay, Dropout=0.5, EPOCHS=50, K=5, return_id=False):

    # 设置 K 折交叉验证的 K 值
    K = 5

    # 创建 KFold 对象
    kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

    fold_index = 0

    test_acc_perfold = []

    test_best_fscore_perfold = []
    test_best_acc_perfold = []


    #save label and pred 
    test_label_list_perfold = []
    test_pred_list_perfold = []
    test_id_list_perfold = []

    # save id
    train_id_perfold = []
    test_id_perfold = []

    best_model_list = []

    for train_index, test_index in kf.split(dataset):
        # 创建训练集和测试集子集
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)

        # 创建训练集和测试集的 DataLoader
        train_loader = DataLoader(train_subset, batch_size=BS, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BS, shuffle=False)
        
        logdir = logdir_root + '_runs/' + datetime.now().strftime('%m-%d_%H-%M') + f'_{dataset_name}_{classify_flag}_lr{LR}_dropout{Dropout}_bs{BS}_epochs{EPOCHS}_fold{fold_index+1}'

        writer = SummaryWriter(logdir)
        logger = _logger(os.path.join(logdir, f'logger_fold{fold_index+1}.txt'))

        # 打印参数
        logger.debug('bert_model_path: {}'.format(config.ptm))
        
        best_model = None
        model = create_bert_basline_model(dataset_name=dataset_name, device=device, hidden_size=768, dropout=Dropout)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=Weight_Decay)

        loss_function = torch.nn.CrossEntropyLoss()

        best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
        all_fscore, all_acc, all_loss = [], [], []
        
        for e in range(EPOCHS):
            start_time = time.time()

            train_loss, train_acc, train_labels, train_preds, train_fscore, train_loss_list = train_or_eval_model_baseline(
                model=model, dataset_name=dataset_name, loss_function=loss_function, dataloader=train_loader, epoch=EPOCHS, 
                optimizer=optimizer, train=True, device=device, gammas=[], classify_flag=classify_flag, train_id=[])

            test_loss, test_acc, test_label, test_pred, test_fscore, test_loss_list = train_or_eval_model_baseline(
                model=model, dataset_name=dataset_name, loss_function=loss_function, dataloader=test_loader, epoch=EPOCHS,
                optimizer=optimizer, train=False, device=device, gammas=[], classify_flag=classify_flag, test_id=[])
            
            
            # 记录到 TensorBoard
            writer.add_scalar('Loss/train', train_loss, e)
            writer.add_scalar('Accuracy/train', train_acc, e)
            writer.add_scalar('F-Score/train', train_fscore, e)
            writer.add_scalar('Loss/test', test_loss, e)
            writer.add_scalar('Accuracy/test', test_acc, e)
            writer.add_scalar('F-Score/test', test_fscore, e)

            for idx in range(len(train_loss_list)):
                writer.add_scalar(f'Loss_{idx+1}/train', train_loss_list[idx], e)
            for idx in range(len(test_loss_list)):
                writer.add_scalar(f'Loss_{idx+1}/test', test_loss_list[idx], e)

            all_fscore.append(test_fscore)
            all_acc.append(test_acc)

            if best_fscore == None or best_fscore < test_fscore:
                best_fscore = test_fscore
                best_label, best_pred = test_label, test_pred, 


                best_model = copy.deepcopy(model)

                
            logger.debug('fold: {}, epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                    format(fold_index+1, e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
            
        
            # model saved 
            if (e+1) == EPOCHS:
                logger.debug(classification_report(best_label, best_pred, digits=4))
                logger.debug(confusion_matrix(best_label, best_pred))

                # save model
                file_name = f'{dataset_name}_model_fold{fold_index + 1}_fs{best_fscore:.4f}.pt'
                save_path = os.path.join(logdir, 'saved_model/')
                os.makedirs(save_path, exist_ok=True)
                print('best model saved in: ', os.path.join(save_path, file_name))
                torch.save(best_model, os.path.join(save_path, file_name))
                best_model_list.append(best_model)

            
                #save label and pred 
                test_label_list_perfold.append(best_label)
                test_pred_list_perfold.append(best_pred)
                # test_id_list_perfold.append(ids)
                
        logger.debug('Test performance..')
        logger.debug('fold: {}, F-Score: {}'.format(fold_index+1, max(all_fscore)))
        logger.debug('fold: {}, F-Score-index: {}'.format(fold_index+1, all_fscore.index(max(all_fscore)) + 1))
        logger.debug('fold: {}, Acc: {}'.format(fold_index+1, max(all_acc)))
        logger.debug('fold: {}, Acc-index: {}'.format(fold_index+1, all_acc.index(max(all_acc)) + 1))
        test_best_fscore_perfold.append(max(all_fscore))
        test_best_acc_perfold.append(max(all_acc))
        fold_index += 1

        writer.close()


    print('5-fold ave fscore:', np.mean(np.array(test_best_fscore_perfold)))
    print('5-fold ave acc:', np.mean(np.array(test_best_acc_perfold)))

    if return_id:
        return test_label_list_perfold, test_pred_list_perfold, test_id_list_perfold, test_best_acc_perfold, test_best_fscore_perfold, train_id_perfold, test_id_perfold
    else:
        return test_label_list_perfold, test_pred_list_perfold, test_id_list_perfold, test_best_acc_perfold, test_best_fscore_perfold

    
def run_performance_testing(dataset, dataset_name, path, device, SEED, BS, average='binary', Dropout=0.5):

    # 设置 K 折交叉验证的 K 值
    K = 5

    # 创建 KFold 对象
    kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
    

    fold_index = 0

    test_acc_perfold = []
    test_recall_perfold = []
    test_precision_perfold = []
    test_fs_perfold = []

    for train_index, test_index in kf.split(dataset):
        
        # 创建训练集和测试集子集
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)

        
        # 创建训练集和测试集的 DataLoader
        # train_loader = DataLoader(train_subset, batch_size=BS, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BS, shuffle=False)
        
        # model = create_model(dataset_name=dataset_name, device=device, hidden_size=768, dropout=Dropout)
        # model.load_state_dict(torch.load(path[fold_index]))
        model = torch.load(path[fold_index])
        avg_accuracy, avg_recall, avg_precision, avg_fscore, labels, preds = performance_testing(model=model, 
                                                                                                 dataset_name=dataset_name, 
                                                                                                 dataloader=test_loader, 
                                                                                                 device=device, 
                                                                                                 average=average)
            
        test_acc_perfold.append(avg_accuracy)
        test_recall_perfold.append(avg_recall)
        test_precision_perfold.append(avg_precision)
        test_fs_perfold.append(avg_fscore)

        fold_index += 1


    print(f'{K}-fold ave acc:', np.mean(np.array(test_acc_perfold)))
    print(f'{K}-fold ave recall:', np.mean(np.array(test_recall_perfold)))
    print(f'{K}-fold ave precision:', np.mean(np.array(test_precision_perfold)))
    print(f'{K}-fold ave fscore:', np.mean(np.array(test_fs_perfold)))

    
    return test_acc_perfold, test_recall_perfold, test_precision_perfold, test_fs_perfold



    