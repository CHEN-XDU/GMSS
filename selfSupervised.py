import torch
import torch.nn as nn
import numpy as np
import config
from utils import calc_con_loss
from dataset import fre_stack, spa_stack, create_contrastive, create_graph, create_jigsaw
from tqdm import tqdm
from model import SelfSupervisedTrain, SelfSupervisedTest
from AutoWeight import AutomaticWeightedLoss
import time
import os
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


def validateTest(validateNet, test_data, test_label, people, highest_acc):

    criterion = nn.CrossEntropyLoss().to(device)
    validateNet.eval()
    epoch_loss = 0.0
    correct_pred = 0

    test_loader = create_graph(test_data, test_label, batch_size=batch_size)
    for ind, data in enumerate(test_loader):
        data = data.to(device)
        out = validateNet(data)
        y = data.y
        _, pre = torch.max(out, dim=1)

        correct_pred += sum([1 for a, b in zip(pre, y) if a == b])
        loss = criterion(out, y)

        epoch_loss += float(loss.item())

    ACC = correct_pred / ((ind + 1) * batch_size)
    with open(f'./{DATASET}_unsupervised.txt', 'a') as f:
        f.write(f'{DATASET}\t{people}\t{ACC:.4f}\n')

    # print(f'Test loss {epoch_loss/(ind+1):.4f} Test ACC@1 {ACC:.4f}')

    return highest_acc, ACC

def validateTrain(train_data, train_label, test_data, test_label, people, HC):

    validate_lr = 0.001
    validate_epochs = 70

    highest_acc = 0.0
    
    if os.path.exists(f'unsupervisedDirectory/{DATASET}_checkpoint/checkpoint_{people}.pkl'):
        check = torch.load(f'unsupervisedDirectory/{DATASET}_checkpoint/checkpoint_{people}.pkl')
        highest_acc = check['ACC']

    validateNet = SelfSupervisedTest(5, 32, batch=batch_size, classes=HC)
    validateNet = validateNet.to(device)

    pretrain_dict = None
    
    if os.path.exists(f'unsupervisedDirectory/pretrain/{DATASET}_checkpoint/checkpoint_{people}.pkl'):
        pretrain_dict = torch.load(f'unsupervisedDirectory/pretrain/{DATASET}_checkpoint/checkpoint_{people}.pkl')

    frozen_list = ['conv1.weight', 'conv1.bias']

    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in frozen_list}

    net_dict = validateNet.state_dict()
    net_dict.update(pretrain_dict)
    validateNet.load_state_dict(net_dict)

    for k, v in validateNet.named_parameters():
        if k in frozen_list:
            v.requires_grad = False

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(validateNet.parameters(), lr=validate_lr, weight_decay=weight_decay)

    for epoch in range(validate_epochs):
        epoch_loss = 0.0
        correct_pred = 0

        train_loader = create_graph(train_data, train_label, shuffle=True, batch_size=batch_size)
        train_loader = tqdm(train_loader)
        validateNet.train()

        for ind, data in enumerate(train_loader):
            data = data.to(device)
            out = validateNet(data)
            y = data.y
            _, pre = torch.max(out, dim=1)

            correct_pred += sum([1 for a,b in zip(pre, y) if a==b])
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

            train_loader.set_description(f'Epoch [{epoch}/{validate_epochs}] validation Loss {epoch_loss/(ind+1):.4f} '
                                         f'ACC@1 {correct_pred/((ind+1)*batch_size)}')
           
        highest_acc, current_acc = validateTest(validateNet, test_data, test_label, people, highest_acc)

        if epoch % 5 == 0:
            print()
            print('-'*100)
            print(f'epoch: [{epoch}/{validate_epochs}]')
            print('Dataset: ', DATASET)
            print(f'highest_acc {highest_acc:.4f} current_acc {current_acc:.4f}')
            print('-' * 100)
            print()

def train(train_data, train_label, test_data, test_label, people):

    HC = None

    if 'SEED_IV' in path:
        HC = 4
    elif 'MPED' in path:
        HC = 7
    else:
        HC = 3
    assert HC is not None

    awl = AutomaticWeightedLoss(3)
    net = SelfSupervisedTrain(5, 32, batch_size, HF=120, HS=128)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    for epoch in range(epochs):

        floader = create_jigsaw(fre_stack, train_data, shuffle=True, batch_size=batch_size)
        sloader = create_jigsaw(spa_stack, train_data, shuffle=True, batch_size=batch_size)
        # gloader = create_graph(train_data, train_label, SEED=SEED, shuffle=True, batch_size=batch_size)
        timeseed = time.time()
        train_loader1 = create_contrastive(fre_stack, spa_stack, train_data.copy(), timeseed, shuffle=True, batch_size=batch_size)
        train_loader2 = create_contrastive(fre_stack, spa_stack, train_data.copy(), timeseed, shuffle=True, batch_size=batch_size)
        loader = zip(floader, sloader, train_loader1, train_loader2)
        epoch_loss = 0.0
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        epoch_loss3 = 0.0 # contrastive loss

        correct_pred1 = 0
        correct_pred2 = 0

        for ind, datas in enumerate(loader):
            fdata, sdata, cdata1, cdata2 = datas
            fdata, sdata, cdata1, cdata2 = fdata.to(device), sdata.to(device), cdata1.to(device), cdata2.to(device)
            x1, x2, x3, x4 = net(fdata, sdata, cdata1, cdata2)
            y1, y2 = fdata.y, sdata.y
            _, pred1 = torch.max(x1, dim=1)
            _, pred2 = torch.max(x2, dim=1)

            correct_pred1 += sum([1 for a,b in zip(pred1, y1) if a==b])
            correct_pred2 += sum([1 for a,b in zip(pred2, y2) if a==b])

            loss1 = criterion(x1, y1)
            loss2 = criterion(x2, y2)

            loss3 = calc_con_loss(x3, x4)
            loss = awl(loss1, loss2, loss3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_loss1 += float(loss1.item())
            epoch_loss2 += float(loss2.item())
            epoch_loss3 += float(loss3.item())

        scheduler.step(epoch_loss)

        denominator = (ind+1)*batch_size
        
        if epoch%5==0:   
            torch.save(net.state_dict(), f'unsupervisedDirectory/pretrain/{DATASET}_checkpoint/checkpoint_{people}.pkl')
            validateTrain(train_data, train_label, test_data, test_label, people, HC)

        print(f'Epoch [{epoch}/{epochs}] \n'
              f'Loss eLoss[{epoch_loss/(ind+1):.4f}] fLoss[{epoch_loss1/(ind+1):.4f}] sLoss[{epoch_loss2/(ind+1):.4f}] '
              f'cLoss[{epoch_loss3/(ind+1):.4f}] \n'
              f'ACC@1 fACC[{correct_pred1/denominator:.4f}] sACC[{correct_pred2/denominator:.4f}] \n')

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
