import yaml
import os
import numpy as np
from utils import get_snapshot

# load config

config = yaml.load(open('config.yml'))

# build path

base_path = os.path.join('./data/', config['dataset'])
raw_base_path = os.path.join(base_path, 'raw')
train_save_path = os.path.join(base_path, 'train.npy')
test_save_path = os.path.join(base_path, 'test.npy')

# load data

num = len(os.listdir(raw_base_path))
data = np.zeros(shape=(num, config['node_num'], config['node_num']), dtype=np.float32)
for i in range(num):
    path = os.path.join(raw_base_path, 'edge_list_' + str(i) + '.txt')
    data[i] = get_snapshot(path, config['node_num'], config['max_thres'])

total_num = num - config['window_size']
test_num = int(config['test_rate'] * total_num)
train_num = total_num - test_num

train_data = data[0: train_num + config['window_size']]
test_data = data[train_num: num]

# save data

np.save(train_save_path, train_data)
np.save(test_save_path, test_data)