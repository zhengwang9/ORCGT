import torch
import sys
torch.set_num_threads(6)
import setproctitle


import argparse
from torch import nn
import pickle
import torch.optim as optim
from dataloader_graph_wz import StasDataset_cv_curva
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
from PatchGCN import PatchgcnCurvaPool
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from datetime import datetime
import logging
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_func

# torch.manual_seed(3407)
hiddim=256
numlayers=3
seed=9888
epochs=120



parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save_epoch', type=int, default=20)
parser.add_argument('--version', type=str, default='test')

parser.add_argument('--seed', type=int, default=9888)


parser.add_argument('--cv_fold_id_path', type=str, default='/data10/wz/wz/Total_New/5_fold_864.pkl')
parser.add_argument('--root', type=str,
                    default='/data10/wz/wz/Total_New/Feats/huandai_Graph_40x_512')
parser.add_argument('--data_label_path', type=str, default='/data10/wz/wz/Total_New/label_STAS_all.csv')
parser.add_argument('--model_save_path', type=str, default=None)
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
args = parser.parse_args()
# torch.manual_seed(3407)



# lr=0.0002
# save_epoch=20
# version = f'test'
version=args.version
i=args.fold
device = [torch.device(f'cuda:{args.device}')]
save_id='curvapool_newM'
log_root=os.path.join('/data10/wz/wz/Total_New/train_log',save_id)
model_root=os.path.join('/data10/wz/wz/Total_New/model',save_id)
if not os.path.exists(log_root):
    os.makedirs(log_root)
if not os.path.exists(model_root):
    os.makedirs(model_root)

setproctitle.setproctitle(f"curvaPooling")



def seed_torch(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

seed_torch()
date = str(datetime.now().date())
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
file_handler = logging.FileHandler(f'{log_root}/fold{i}_{date}-v{version}.log')
file_handler.setLevel(logging.INFO)
logger = logging.getLogger()
logger.addHandler(file_handler)


def lsim_loss_cal(representations, label, T=0.7):  # Contrastive Learning Similarity Loss
    n = label.shape[0]

    similarity_matrix = F.cosine_similarity(representations.unsqueeze(0), representations.unsqueeze(1), dim=2)
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

    mask_no_sim = torch.ones_like(mask) - mask

    mask_eig_0 = torch.ones(n, n) - torch.eye(n, n)
    mask_eig_0 = mask_eig_0.to(device[0])

    similarity_matrix = torch.exp(similarity_matrix / T)

    similarity_matrix = similarity_matrix * mask_eig_0

    sim = mask * similarity_matrix

    no_sim = similarity_matrix - sim

    no_sim_sum = torch.sum(no_sim, dim=1)

    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)
    loss = mask_no_sim + loss + torch.eye(n, n).to(device[0])

    loss = - torch.log(loss)
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

    return loss


kwargs = {'num_workers': 4, 'pin_memory': True} if device[0].type == "cuda" else {}
with open(args.cv_fold_id_path, 'rb') as file:
    cv_fold_id = pickle.load(file)




def train_zzf(train_loader):
    model.train()
    train_loss = 0.
    train_error = 0.
    batch_num = 0
    labels = []
    label_batch = []
    out_batch = None
    logits_batch = None
    preds = []

    batch_size = args.batch_size
    for batch_idx, (data,_) in enumerate(train_loader):
        m = batch_idx + 1
        label = data.y
        if device[0].type == "cuda":
            data = data.to(device[0])
            label = label.to(device[0])

        labels.append(label.item())
        # calculate loss and metrics
        logits, Y_hat,fea = model.forward(data)
        preds.append(Y_hat.item())

        #if batch_idx == 0:
        print('Y_prob:', F.softmax(logits, dim=-1))

        # loss=loss_fn(logits,label)
        # train_loss += loss.item()

        error = Y_hat != label
        train_error += error.item()
        # print('m',m)
        # print('out_batch',out_batch)
        # print('logits_batch',logits_batch)

        if m % batch_size == 1:
            # print('OK')
            out_batch = fea
            logits_batch = logits
            label_batch.append(label.item())
        else:
            out_batch = torch.cat([out_batch, fea], dim=0)
            logits_batch = torch.cat([logits_batch, logits], dim=0)
            label_batch.append(label.item())

        if m % batch_size == 0 and len(np.unique(label_batch)) == 2:
            # print(out_batch.shape,torch.LongTensor(label_batch).to(device[0]))
            sim_loss = lsim_loss_cal(out_batch, torch.LongTensor(label_batch).to(device[0]))
            ce_loss = loss_fn(logits_batch, torch.LongTensor(label_batch).to(device[0]))
            loss = ce_loss + sim_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_num += 1
            out_batch = None
            logits_batch = None
            label_batch = []
        elif m % batch_size == 0 and len(np.unique(label_batch)) < 2:
            out_batch = None
            logits_batch = None
            label_batch = []

    # calculate loss and error for epoch
    train_loss /= batch_num
    train_error /= len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print("Learning Rate:", current_lr)
    print('Loss: {:.6f},Train error:{:.6f}'.format(train_loss, train_error))
    return train_loss




def val_zzf(val_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    labels = []
    preds = []
    scores = []

    for batch_idx, (data,_) in enumerate(val_loader):
        label = data.y
        labels.append(label.item())
        if device[0].type == "cuda":
            data, label = data.to(device[0]), label.to(device[0])

        logits, Y_hat,_= model.forward(data)
        preds.append(Y_hat.item())
        loss = loss_fn(logits, label)
        test_loss += loss.item()

        # error, predicted_label = model.calculate_classification_error(data, bag_label)
        # test_error += error
        # predicted_label=Y_prob.argmax(dim=1)
        Y_prob = logits.cpu().data.numpy()
        scores.append(Y_prob[0][1])
        error = Y_hat != label
        test_error += error.item()

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    # print(labels)
    # print(preds)
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    roc = roc_auc_score(labels, scores)

    # 打印混淆矩阵
    # print("Confusion Matrix:")
    # print(cm)
    logging.info('Val Confusion Matrix:\n{}'.format(cm))

    report = classification_report(labels, preds)
    precision, recall, f1_score_class_1, _ = precision_recall_fscore_support(labels, preds, labels=[1],
                                                                             average='binary')
    precision, recall, f1_score_class_0, _ = precision_recall_fscore_support(labels, preds, labels=[0],
                                                                             average='binary')
    # print("Classification Report:")
    # print(report)
    logging.info('auc:%f', roc)
    pc,recall,_=precision_recall_curve(labels,scores)
    auprc=auc_func(recall,pc)

    logging.info('auprc:%f', auprc)

    logging.info('Val Classification Report:\n{}'.format(report))

    print('Loss: {:.6f}, Val error: {:.6f}\n\n'.format(test_loss, test_error))
    return test_loss, f1_score_class_1, acc, roc


def test_zzf(test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    labels = []
    preds = []
    scores = []

    for batch_idx, (data,_) in enumerate(test_loader):
        label = data.y
        labels.append(label.item())
        if device[0].type == "cuda":
            data, label = data.to(device[0]), label.to(device[0])

        logits, Y_hat,_ = model.forward(data)
        preds.append(Y_hat.item())
        loss = loss_fn(logits, label)
        test_loss += loss.item()

        # error, predicted_label = model.calculate_classification_error(data, bag_label)
        # test_error += error
        # predicted_label=Y_prob.argmax(dim=1)
        Y_prob = logits.cpu().data.numpy()
        scores.append(Y_prob[0][1])
        error = Y_hat != label
        test_error += error.item()

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    # print(labels)
    # print(preds)
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    roc = roc_auc_score(labels, scores)

    # 打印混淆矩阵
    # print("Confusion Matrix:")
    # print(cm)
    logging.info('Test Confusion Matrix:\n{}'.format(cm))

    report = classification_report(labels, preds)
    precision, recall, f1_score_class_1, _ = precision_recall_fscore_support(labels, preds, labels=[1],
                                                                             average='binary')
    precision, recall, f1_score_class_0, _ = precision_recall_fscore_support(labels, preds, labels=[0],
                                                                             average='binary')
    # print("Classification Report:")
    # print
    logging.info('auc:%f', roc)
    pc, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc_func(recall, pc)
    logging.info('auprc:%f', auprc)
    logging.info('Test Classification Report:\n{}'.format(report))

    print('Loss: {:.6f}, Test error: {:.6f}\n\n'.format(test_loss, test_error))
    return test_loss, f1_score_class_1, acc, roc




if __name__ == "__main__":
    # for i in range(5):
    # i=0
    fold = str(i)
    logging.info('Fold {}'.format(fold))
    print('Fold {}'.format(fold))
    train_loader = DataLoader(StasDataset_cv_curva(args.root, cv_fold_id, args.data_label_path, f'fold_{fold}', 'train'),
                              batch_size=1,
                              shuffle=True,
                              **kwargs)
    # print(len(train_loader))
    val_loader = DataLoader(StasDataset_cv_curva(args.root, cv_fold_id, args.data_label_path, f'fold_{fold}', 'val'),
                            batch_size=1,
                            shuffle=False,
                            **kwargs)
    # print(len(val_loader))
    test_loader = DataLoader(StasDataset_cv_curva(args.root, cv_fold_id, args.data_label_path, f'fold_{fold}', 'test'),
                             batch_size=1,
                             shuffle=False,
                             **kwargs)
    # print(len(test_loader))
    model = PatchgcnCurvaPool(num_layers=numlayers,hidden_dim=hiddim).to(device[0])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.35, 0.65]).to(device[0]))  # 285
    # zzf try:

    val_eeror = 0
    test_eeror = 0
    f1_1 = 0
    auc = 0
    test_eeror_f1 = 0
    test_eeror_auc = 0
    test_eeror_add = 0
    best_epoch_add = 0
    best_epoch_auc = 0
    best_epoch_f1 = 0
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        logging.info('\nEpoch {}'.format(epoch))

        print('Start Training : Epoch {}'.format(epoch))
        train_loss = train_zzf(train_loader)

        logging.info('Train Loss : {}'.format(train_loss))

        print('\nStart Validing:')
        val_loss, val_1_f1, acc_val, val_auc = val_zzf(val_loader)
        # print(acc_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('\nStart Testing:')
        test_loss, test_1_f1, acc_test, _ = test_zzf(test_loader)
        if epoch >= 20 and (val_1_f1 + acc_val) >= val_eeror:
            best_epoch_add = epoch
            torch.save(model, f'{model_root}/best_f1_add_acc_{date}_v{version}_fold{i}.pth')
            val_eeror = val_1_f1 + acc_val
            test_eeror_add = test_1_f1
        if epoch >= 20 and val_1_f1 >= f1_1:
            best_epoch_f1 = epoch
            torch.save(model, f'{model_root}/best_f1_{date}_v{version}_fold{i}.pth')
            f1_1 = val_1_f1
            test_eeror_f1 = test_1_f1
        if epoch >= 20 and val_auc >= auc:
            best_epoch_auc = epoch
            torch.save(model, f'{model_root}/best_auc_{date}_v{version}_fold{i}.pth')
            auc = val_auc
            test_eeror_auc = test_1_f1
        logging.info('best_epoch_add:{}'.format(best_epoch_add))
        logging.info('best_epoch_f1:{}'.format(best_epoch_f1))
        logging.info('best_epoch_auc:{}'.format(best_epoch_auc))

        print('add', best_epoch_add, ': ', val_eeror, ' / ', test_eeror_add)
        print('f1', best_epoch_f1, ': ', f1_1, ' / ', test_eeror_f1)
        print('auc', best_epoch_auc, ': ', auc, ' / ', test_eeror_auc)

    fig_path = f'{log_root}/loss_{date}_fold{i}_v{version}.png'
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validating Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fig_path)
    plt.clf()
