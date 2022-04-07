import torch
import numpy as np
import config
import numpy as np
import itertools
from scipy.spatial.distance import cdist
import h5py


def hamming_set(num_crops, num_permutations, selection, output_file_name):
    """
    generate and save the hamming set
    :param num_crops: number of tiles from each image
    :param num_permutations: Number of permutations to select (i.e. number of classes for the pretext task)
    :param selection: Sample selected per iteration based on hamming distance: [max] highest; [mean] average
    :param output_file_name: name of the output HDF5 file
    """
    P_hat = np.array(list(itertools.permutations(list(range(num_crops)), num_crops)))
    n = P_hat.shape[0]

    for i in range(num_permutations):
        if i == 0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1, -1])
        else:
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

        if selection == 'max':
            j = D.argmax()
        elif selection == 'mean':
            m = int(D.shape[0] / 2)
            S = D.argsort()
            j = S[np.random.randint(m - 10, m + 10)]


    np.save(f'max_hamming_set_{num_crops}_{num_permutations}.npy', P)
    print('file created --> ' + output_file_name + str(num_permutations) + '.npy')

def move(_out, shift, num_jigsaw=config.num_jigsaw):
    i = 0
    while (i+1)*num_jigsaw <= _out.shape[0]:
        _out[i*num_jigsaw : (i+1)*num_jigsaw] = torch.roll(_out[i*num_jigsaw : (i+1)*num_jigsaw], shift, 0)
        i += 1
    return _out

def calc_con_loss(out_1, out_2, temperature=0.5):

    out = torch.cat([out_1, out_2], dim=0)

    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * config.batch_size, device=sim_matrix.device)).bool()

    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * config.batch_size, -1)

    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]

    if config.num_jigsaw > 1:
        for i in range(1, config.num_jigsaw):
            _out_1 = move(out_1.clone(), i)
            pos_sim = pos_sim + torch.exp(torch.sum(out_1 * _out_1, dim=-1) / temperature)
            pos_sim = pos_sim + torch.exp(torch.sum(out_2 * _out_1, dim=-1) / temperature)
            _out_2 = move(out_2.clone(), i)
            pos_sim = pos_sim + torch.exp(torch.sum(out_2 * _out_2, dim=-1) / temperature)

    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = 2*(- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss


def cacl_std(arr):
    return np.array(arr).std()


def cacl_acc():
    denominator = 45
    part2of3 = 0.0
    total = 0.0
    tmp = []
    arr = []
    f = open('ACC.txt', 'w')
    for i in range(denominator):
        ck = torch.load(f'SEED_jointly_checkpoint/checkpoint_{i+1}.pkl', map_location='cuda:0')
        print(ck['ACC'])
        f.write('{}\t{:.4f}\n'.format(i+1, ck['ACC']))
        total += ck['ACC']
        tmp.append(ck['ACC'])
        # arr.append(ck['ACC'])
        if len(tmp) == 3:
            tmp.sort(reverse=False)
            arr.append(tmp[1])
            arr.append(tmp[2])
            part2of3 = part2of3 + tmp[1] + tmp[2]
            tmp = []
    print('total: {:.4f}'.format(total/denominator))
    print('2/3: {:.4f}'.format(part2of3/(denominator*2/3))) # for SEED subject-dependent
    print(cacl_std(arr))


if __name__ == '__main__':
    cacl_acc()
   
