import yaml
import os
import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from model import Generator, Discriminator
from utils import LPDataset

config = yaml.load(open('config.yml'))

node_num = config['node_num']
window_size = config['window_size']

base_path = os.path.join('./data/', config['dataset'])
train_save_path = os.path.join(base_path, 'train.npy')

train_data = LPDataset(train_save_path, window_size)
sample_data = LPDataset(train_save_path, window_size)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True
)
sample_loader = DataLoader(
    dataset=sample_data,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True
)

generator = Generator(
    window_size=window_size,
    node_num=node_num,
    in_features=config['in_features'],
    out_features=config['out_features'],
    lstm_features=config['lstm_features']
)

discriminator = Discriminator(
    input_size=node_num * node_num,
    hidden_size=config['disc_hidden']
)

generator = generator.cuda()
discriminator = discriminator.cuda()

mse = nn.MSELoss(reduction='sum')

pretrain_optimizer = optim.RMSprop(generator.parameters(), lr=config['pretrain_learning_rate'])
generator_optimizer = optim.RMSprop(generator.parameters(), lr=config['g_learning_rate'])
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=config['d_learning_rate'])
#
print('pretrain generator')

for epoch in range(config['pretrain_epoches']):
    for i, data in enumerate(train_loader):
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()
        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
        predicted_shot = generator(in_shots)
        out_shot = out_shot.view(config['batch_size'], -1)
        loss = mse(predicted_shot, out_shot)
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), config['gradient_clip'])
        pretrain_optimizer.step()
        print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, loss.item()))

print('train GAN')

for epoch in range(config['gan_epoches']):
    for i, (data, sample) in enumerate(zip(train_loader, sample_loader)):
        # update discriminator
        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
        predicted_shot = generator(in_shots)
        _, sample = sample
        sample = sample.cuda()
        sample = sample.view(config['batch_size'], -1)
        real_logit = discriminator(sample).mean()
        fake_logit = discriminator(predicted_shot).mean()
        discriminator_loss = -real_logit + fake_logit
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()
        for p in discriminator.parameters():
            p.data.clamp_(-config['weight_clip'], config['weight_clip'])
        # update generator
        generator_loss = -fake_logit
        generator_loss.backward()
        generator_optimizer.step()
        out_shot = out_shot.view(config['batch_size'], -1)
        mse_loss = mse(predicted_shot, out_shot)
        print('[epoch %d] [step %d] [d_loss %.4f] [g_loss %.4f] [mse_loss %.4f]' % (epoch, i,
                discriminator_loss.item(), generator_loss.item(), mse_loss.item()))

torch.save(generator, os.path.join(base_path, 'generator.pkl'))
