import torch
import torch.nn as nn
import numpy as np
import config
from dataset import fre_stack, spa_stack, create_contrastive, create_graph, create_jigsaw
from utils import calc_con_loss
from model import JointlyTrainModel
from AutoWeight import AutomaticWeightedLoss
import os
from itertools import cycle
import time
import pdb

path = '/GNN_NPY_DATASETS/SEED/data_dependent'
# path = '/GNN_NPY_DATASETS/MPED/data_dependent'
# path = '/GNN_NPY_DATASETS/SEED/data_independent'
batch_size = config.batch_size
epochs = config.epochs
lr = config.lr
weight_decay = config.weight_decay
device = config.device
num_jigsaw = config.num_jigsaw
DATASETS = ['SEED', 'SEED_IV', 'MPED']
DATASET = path.strip().split('/')[-2]
assert DATASET in DATASETS
DEPENDENT = path.strip().split('/')[-1]
if DEPENDENT == 'data_independent':
    DATASET = DATASET+'_'+DEPENDENT


def writeEachEpoch(people, epoch, batchsize, lr, temperature, acc):
    import model
    log = []
    log.append(f'{DATASET}\t{people}\t{temperature}\t'
               f'{batchsize}\t{epoch}\t{lr}\t{model.drop_rate}\t{acc:.4f}\n')
    with open(
            f'/xxx/{DATASET}_All_log.txt',
            'a') as f:
        f.writelines(log)


def updatelog(people, epoch, acc):
    log = []
    log.append(f'{DATASET}\t{people}\t{epoch}\t{lr}\t{batch_size}\t{acc:.4f}\n')
    with open('/xxx/{DATASET}_UPDATE_LOG.txt', 'a') as f:
        f.writelines(log)


def test(net, test_data, test_label, people, highest_acc, epoch):
    criterion = nn.CrossEntropyLoss().to(device)

    gloader = create_graph(test_data, test_label, shuffle=True, batch_size=batch_size, drop_last=True)
    net.testmode = True
    net.eval()
    epoch_loss = 0.0
    correct_pred = 0
    for ind, data in enumerate(gloader):
        data = data.to(device)
        out = net(data)
        y = data.y
        _, pre = torch.max(out, dim=1)

        correct_pred += sum([1 for a, b in zip(pre, y) if a == b])
        loss = criterion(out, y)

        epoch_loss += float(loss.item())

    ACC = correct_pred / ((ind + 1) * batch_size)
    if ACC > highest_acc:
        updatelog(people, epoch, ACC)
        highest_acc = ACC
        ck = {}
        ck['epoch'] = epoch
        ck['model'] = net.state_dict()
        ck['ACC'] = ACC
        
        torch.save(ck, f'{DATASET}_jointly_checkpoint/checkpoint_{people}.pkl')

    net.train()
    net.testmode=False
    return highest_acc, ACC


def train(train_data, train_label, test_data, test_label, people):
    highest_acc = 0.0
    
    if not os.path.exists(f'{DATASET}_jointly_checkpoint'):
        os.makedirs(f'{DATASET}_jointly_checkpoint')
    
    if os.path.exists(f'{DATASET}_jointly_checkpoint/checkpoint_{people}.pkl'):
        check = torch.load(f'{DATASET}_jointly_checkpoint/checkpoint_{people}.pkl')
        highest_acc = check['ACC']
    
    HC = None
    if 'SEED_IV' in DATASET:
        HC = 4
    elif 'MPED' in DATASET:
        HC = 7
    else:
        HC = 3
    assert HC is not None

    awl = AutomaticWeightedLoss(4)
    net = JointlyTrainModel(5, 32, batch_size, testmode=False, HF=120, HS=128, HC=HC)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8)

    floader = create_jigsaw(fre_stack, train_data, shuffle=True, batch_size=batch_size, num_jigsaw=num_jigsaw)
    sloader = create_jigsaw(spa_stack, train_data, shuffle=True, batch_size=batch_size, num_jigsaw=num_jigsaw)
    gloader = create_graph(train_data, train_label, shuffle=True, batch_size=batch_size)
    timeseed = time.time()
    train_loader1 = create_contrastive(fre_stack, spa_stack, train_data.copy(), timeseed, shuffle=True, batch_size=batch_size, num_jigsaw=num_jigsaw)
    train_loader2 = create_contrastive(fre_stack, spa_stack, train_data.copy(), timeseed, shuffle=True, batch_size=batch_size,  num_jigsaw=num_jigsaw)

    for epoch in range(epochs):

        loader = zip(floader, sloader, cycle(gloader), train_loader1, train_loader2)

        epoch_loss = 0.0
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        epoch_loss3 = 0.0
        epoch_loss4 = 0.0
        correct_pred1 = 0
        correct_pred2 = 0
        correct_pred3 = 0

        for ind, datas in enumerate(loader):
            fdata, sdata, gdata, cdata1, cdata2 = datas
            fdata, sdata, gdata, cdata1, cdata2 = fdata.to(device), sdata.to(device), gdata.to(device), cdata1.to(device), cdata2.to(device)
            x1, x2, x3, x4, x5 = net(fdata, sdata, gdata, cdata1, cdata2)
            y1, y2, y3 = fdata.y, sdata.y, gdata.y
            _, pred1 = torch.max(x1, dim=1)
            _, pred2 = torch.max(x2, dim=1)
            _, pred3 = torch.max(x3, dim=1)
            correct_pred1 += sum([1 for a,b in zip(pred1, y1) if a==b])
            correct_pred2 += sum([1 for a,b in zip(pred2, y2) if a==b])
            correct_pred3 += sum([1 for a,b in zip(pred3, y3) if a==b])
            loss1 = criterion(x1, y1)
            loss2 = criterion(x2, y2)
            loss3 = criterion(x3, y3)
            loss4 = calc_con_loss(x4, x5)
            loss = awl(loss1, loss2, loss3, loss4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_loss1 += float(loss1.item())
            epoch_loss2 += float(loss2.item())
            epoch_loss3 += float(loss3.item())
            epoch_loss4 += float(loss4.item())

        highest_acc, current_acc = test(net, test_data, test_label, people, highest_acc, epoch)
        writeEachEpoch(people, epoch, batch_size, lr, 0.25, current_acc)

        scheduler.step(epoch_loss)

        denominator = (ind+1)*batch_size
        if epoch % 5 == 0:
            print()
            print(f'-----highest_acc {highest_acc:.4f} current_acc {current_acc:.4f}-----')
            print('Dataset: ', DATASET)
            print(f'batch {batch_size}, lr {lr}')
            print()
            
        print(f'Epoch [{epoch}/{epochs}] \n'
              f'Loss eLoss[{epoch_loss/(ind+1):.4f}] fLoss[{epoch_loss1/(ind+1):.4f}] sLoss[{epoch_loss2/(ind+1):.4f}] '
              f'gLoss[{epoch_loss3/(ind+1):.4f}] cLoss[{epoch_loss4/(ind+1):.4f}] \n'
              f'ACC@1 fACC[{correct_pred1/denominator:.4f}] sACC[{correct_pred2/denominator:.4f}] gACC[{correct_pred3/denominator:.4f}] \n')

def runs(people):
    print(f'load object {people}\'s data.....')
    train_data = np.load(path + '/' + 'train_dataset_{}.npy'.format(people))
    train_label = np.load(path + '/' + 'train_labelset_{}.npy'.format(people))
    test_data = np.load(path + '/' + 'test_dataset_{}.npy'.format(people))
    test_label = np.load(path + '/' + 'test_labelset_{}.npy'.format(people))
    print('loaded!')

    train(train_data, train_label, test_data, test_label, people)


if __name__ == '__main__':
    for i in range(45):
        runs(i+1)
