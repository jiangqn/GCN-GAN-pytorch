import yaml
import os
import torch
from torch.utils.data.dataloader import DataLoader
from utils import LPDataset
from utils import MSE, EdgeWiseKL, MissRate

config = yaml.load(open('config.yml'))

node_num = config['node_num']
window_size = config['window_size']

base_path = os.path.join('./data/', config['dataset'])
generator = torch.load(os.path.join(base_path, 'generator.pkl'))

test_save_path = os.path.join(base_path, 'test.npy')
test_data = LPDataset(test_save_path, window_size)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True
)

total_samples = 0
total_mse = 0
total_kl = 0
total_missrate = 0

for i, data in enumerate(test_loader):
    in_shots, out_shot = data
    predicted_shot = generator(in_shots)
    predicted_shot = predicted_shot.view(-1, config['node_num'], config['node_num'])
    predicted_shot = (predicted_shot + predicted_shot.transpose(1, 2)) / 2
    for j in range(config['node_num']):
        predicted_shot[:, j, j] = 0
    mask = predicted_shot >= config['epsilon']
    predicted_shot = predicted_shot * mask.float()
    batch_size = in_shots.size(0)
    total_samples += batch_size
    total_mse += batch_size * MSE(predicted_shot, out_shot)
    total_kl += batch_size * EdgeWiseKL(predicted_shot, out_shot)
    total_missrate += batch_size * MissRate(predicted_shot, out_shot)

print('MSE: %.4f' % (total_mse / total_samples))
print('edge wise KL: %.4f' % (total_kl / total_samples))
print('miss rate: %.4f' % (total_missrate / total_samples))