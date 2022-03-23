import torch

epochs = 100
batch_size = 256 # 200 400 worse around 33-35 （ what is funny that the loss of them lower than 300, around 1.65)
lr = 0.01
weight_decay = 8e-5
drop_rate = 0.25 
num_workers = 8
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
num_jigsaw = 4
K = 2