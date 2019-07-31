import torch
from torch.utils.data import Dataset
import numpy as np

def get_snapshot(path, node_num, max_thres):
    file = open(path, 'r', encoding='utf-8')
    snapshot = np.zeros(shape=(node_num, node_num), dtype=np.float32)
    for line in file.readlines():
        line = line.strip().split(' ')
        node1 = int(line[0])
        node2 = int(line[1])
        edge = float(line[2])
        edge = min(edge, max_thres)
        snapshot[node1, node2] = edge
        snapshot[node2, node1] = edge
    snapshot /= max_thres
    return snapshot

class LPDataset(Dataset):

    def __init__(self, path, window_size):
        super(LPDataset, self).__init__()
        self.data = torch.from_numpy(np.load(path))
        self.window_size = window_size
        self.num = self.data.size(0) - window_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size], self.data[item + self.window_size]

def MSE(input, target):
    num = 1
    for s in input.size():
        num = num * s
    return (input - target).pow(2).sum().item() / num

def EdgeWiseKL(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask = (input > 0) & (target > 0)
    input = input.masked_select(mask)
    target = target.masked_select(mask)
    kl = (target * torch.log(target / input)).sum().item() / num
    return kl

def MissRate(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask1 = (input > 0) & (target == 0)
    mask2 = (input == 0) & (target > 0)
    mask = mask1 | mask2
    return mask.sum().item() / num