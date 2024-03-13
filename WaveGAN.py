#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from pickle import load, dump
from glob import glob
import os
import librosa
import soundfile
import random
import argparse
import time
import math


# In[ ]:


class MuLaw:
    def mulaw(self, x, mu=255):
        return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    
    def quantize(self, y, mu=255, offset=1):
        return ((y + offset) / 2 * mu).astype(np.int64)
    
    def encode(self, x, mu=255):
        return self.quantize(self.mulaw(x, mu), mu)
    
    def inv_mulaw(self, y, mu=255):
        return np.sign(y) * (1.0 / mu) * ((1.0 + mu)**np.abs(y) - 1.0)
    
    def inv_quantize(self, y, mu=255):
        return 2 * y.astype(np.float32) / mu - 1
    
    def decode(self, y, mu=255):
        return self.inv_mulaw(self.inv_quantize(y, mu), mu)


# In[ ]:


class Mish(nn.Module):
    @staticmethod
    def mish(x):
        return x * torch.tanh(F.softplus(x))
    
    def forward(self, x):
        return Mish.mish(x)


# In[ ]:


class GatedDilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size=kernel_size, padding=self.padding, dilation=dilation, **kwargs)
        
    def forward(self, x):
        y = self.conv(x)
        if self.padding > 0:
            y = y[:, :, :-self.padding]
        
        a, b = y.split(y.size(1) // 2, dim=1)
        y = torch.tanh(a) + torch.sigmoid(b)
        
        return y


# In[ ]:


class FReLU(nn.Module):
    def __init__(self, n_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self.funnel_condition = nn.Conv1d(n_channel, n_channel, kernel_size=kernel, stride=stride, padding=padding, groups=n_channel)
        self.bn = nn.BatchNorm1d(n_channel)
        
    def forward(self, x):
        tx = self.bn(self.funnel_condition(x))
        out = torch.max(x, tx)
        return out


# In[ ]:


# For Checkerboard-Artifacts. (change from ConvTranspose1d)
class UpsampleConv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv.weight.data.fill_(1.0 / np.prod(kernel_size))
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


# In[ ]:


# For Checkerboard-Artifacts. (for Discriminator)
class PhaseShuffle(nn.Module):
    def __init__(self, n_shift=2):
        super().__init__()
        self.n_shift = n_shift

    def forward(self, x):
        if self.n_shift == 0:
            return x
        
        shift = random.randint(-self.n_shift, self.n_shift)
        x = x[:, :, shift:] if shift > 0 else x[:, :, :shift] if shift < 0 else x
        x_shift = x[:, :, :shift] if shift > 0 else x[:, :, shift:] if shift < 0 else None
        if x_shift != None:
            x_shift = x_shift.fliplr()
            x = torch.cat([x, x_shift], dim=2)
        
        return x


# In[ ]:


class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, conv_dim=16, n_repeat=8):
        super().__init__()
        
        out_features = conv_dim * 2 ** n_repeat
        model = [
            nn.Conv1d(input_nc, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_features),
            FReLU(out_features)
        ]
        in_features = out_features
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(n_repeat):
            model += [
                UpsampleConv(in_features, out_features, scale_factor=2),
                #nn.ConvTranspose1d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_features),
                FReLU(out_features)
            ]
            in_features = out_features
            out_features = in_features // 2
            
        # Output layer
        model += [
            nn.Conv1d(in_features, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, conv_dim=16, n_repeat=8):
        super().__init__()

        out_features = conv_dim
        model = [
            GatedDilatedCausalConv1d(input_nc, out_features, kernel_size=3),
            nn.BatchNorm1d(out_features)#,
            #PhaseShuffle()
        ]
        in_features = out_features
        
        # downsampling
        out_features = in_features * 2
        for i in range(n_repeat):
            model += [
                nn.Conv1d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                FReLU(out_features),
                nn.BatchNorm1d(out_features)#,
                #PhaseShuffle()
            ]
            in_features = out_features
            out_features = in_features * 2
        
        self.model = nn.Sequential(*model)

        # PatchGAN
        self.conv_patch = nn.Conv1d(in_features, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x =  self.model(x)
        out = self.conv_patch(x)
        return out


# In[ ]:


class WaveDataset(Dataset):
    def __init__(self, dir_path, ext='*', sound_length=65536, sampling_rate=16000, keep_alive=1000):
        self.files = glob(os.path.join(dir_path, ext))
        self.sound_length = sound_length
        self.sampling_rate = sampling_rate
        self.keep_alive = keep_alive
        if(len(self.files) <= keep_alive):
            self.contents = []
            for file in self.files:
                sound, _ = librosa.load(file, sr=self.sampling_rate)
                self.contents += [sound]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if len(self.files) <= self.keep_alive:
            sound = self.contents[index % len(self.files)]
        else:
            file = self.files[index % len(self.files)]
            sound, _ = librosa.load(file, sr=self.sampling_rate)
        
        # Normalize.
        max_amplitude = np.max(np.abs(sound))
        if max_amplitude > 1:
            sound /= max_amplitude

        # Padding.
        if sound.shape[0] < self.sound_length:
            padding_length = self.sound_length - sound.shape[0]
            left = np.zeros(padding_length//2)
            right = np.zeros(padding_length - padding_length//2)
            sound = np.concatenate([left, sound, right], axis=0)

        # Floor.
        if sound.shape[0] > self.sound_length:
            start_index = random.randrange(sound.shape[0] - self.sound_length)
            end_index = start_index + self.sound_length
            sound = sound[start_index : end_index]
        
        sound = torch.from_numpy(sound.astype(np.float32))
        
        if sound.dim() < 2:
            sound = sound.unsqueeze(0)
            
        return sound
    
    def save_wavfile(self, sounddata, filename):
        soundfile.write(filename, sounddata.T, self.sampling_rate, format="WAV", subtype="FLOAT")


# In[ ]:


class Util:
    @staticmethod
    def loadWaveData(batch_size, dir_path, sound_length):
        dataset = WaveDataset(dir_path, sound_length=sound_length)
        return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


class Solver:
    def __init__(self, args):
        use_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f'Use Device: {self.device}')
        
        self.args = args
        
        n_repeat = int(math.log(self.args.sound_length /  self.args.feed_dim, 2))
        self.netG = Generator(n_repeat=n_repeat).to(self.device)
        self.netD = Discriminator(n_repeat=n_repeat).to(self.device)

        #self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(0.5, 0.9))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.args.lr * self.args.mul_lr_dis, betas=(0.5, 0.9))
        
        self.load_dataset()
        
        self.pseudo_aug = 0.0
        self.epoch = 0
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv1d or type(module) == nn.ConvTranspose1d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias != None:
                module.bias.data.fill_(0)
            
    def load_dataset(self):
        self.dataset, self.dataloader = Util.loadWaveData(self.args.batch_size,
                                                          self.args.sound_dir, self.args.sound_length)
        self.max_iters = len(iter(self.dataloader))
            
    def save_state(self, epoch):
        self.netG.cpu(), self.netD.cpu()
        torch.save(self.netG.state_dict(), os.path.join(self.args.weight_dir, f'weight_G.{epoch}.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.args.weight_dir, f'weight_D.{epoch}.pth'))
        self.netG.to(self.device), self.netD.to(self.device)
        
    def load_state(self):
        if (os.path.exists('weight_G.pth') and os.path.exists('weight_D.pth')):
            self.netG.load_state_dict(torch.load('weight_G.pth', map_location=self.device))
            self.netD.load_state_dict(torch.load('weight_D.pth', map_location=self.device))
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    def load_resume(self):
        if os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                return load(f)
        else:
            return self
        
    def trainGAN(self, epoch, iters, max_iters, real_wav, a=0, b=1, c=1):
        ### Train with LSGAN.
        ### for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
        
        feeds = torch.randn(real_wav.size(0), 1, self.args.feed_dim).to(self.device)
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        # Compute loss with real images.
        real_src_score = self.netD(real_wav)
        real_src_loss = torch.sum((real_src_score - b) ** 2)
        
        # Compute loss with fake images.
        fake_wav = self.netG(feeds)
        fake_src_score = self.netD(fake_wav)
        
        # Compute loss with fake images.
        p = random.uniform(0, 1)
        if 1 - self.pseudo_aug < p:
            fake_src_loss = torch.sum((fake_src_score - b) ** 2) # Pseudo: fake is real.
        else:
            fake_src_loss = torch.sum((fake_src_score - a) ** 2)
        
        # Update Pseudo Augmentation.
        lz = (torch.sign(torch.logit(real_src_score)).mean()
              - torch.sign(torch.logit(fake_src_score)).mean()) / 2
        if lz > 0.6:
            self.pseudo_aug += 0.01
        else:
            self.pseudo_aug -= 0.01
        self.pseudo_aug = min(1, max(0, self.pseudo_aug))
        
        # Backward and optimize.
        d_loss = 0.5 * (real_src_loss + fake_src_loss) / self.args.batch_size
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        
        # Logging.
        loss = {}
        loss['D/loss'] = d_loss.item()
        loss['D/pseudo_aug'] = self.pseudo_aug
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        # Compute loss with reconstruction loss
        fake_wav = self.netG(feeds)
        fake_src_score = self.netD(fake_wav)
        fake_src_loss = torch.sum((fake_src_score - c) ** 2)
        
        # Backward and optimize.
        g_loss = 0.5 * fake_src_loss / self.args.batch_size
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()
        
        # Logging.
        loss['G/loss'] = g_loss.item()
        
        # Save
        if iters == max_iters:
            self.save_state(epoch)
            wav_name = str(epoch) + '_' + str(iters) + '.wav'
            wav_path = os.path.join(self.args.result_dir, wav_name)
            self.dataset.save_wavfile(fake_wav[0].detach().cpu(), wav_path)
        
        return loss
    
    def train(self, resume=True):
        self.netG.train()
        self.netD.train()
        
        while self.args.num_train > self.epoch:
            self.epoch += 1
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            
            self.mean_path_length = 0
            for iters, data in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                data = data.to(self.device)
                
                loss = self.trainGAN(self.epoch, iters, self.max_iters, data)
                
                epoch_loss_D += loss['D/loss']
                epoch_loss_G += loss['G/loss']
                #experiment.log_metrics(loss)
            
            epoch_loss = epoch_loss_G + epoch_loss_D
            
            print(f'Epoch[{self.epoch}]'
                  + f' Loss[G({epoch_loss_G}) + D({epoch_loss_D}) = {epoch_loss}]')
                    
            if resume:
                self.save_resume()
    
    def generate(self, num=100):
        self.netG.eval()
        
        for _ in range(num):
            feeds = torch.randn(1, 1, self.args.feed_dim).to(self.device)
            fake_wav = self.netG(feeds)
            wav_path = os.path.join(self.args.result_dir, f'generated_{time.time()}.wav')
            self.dataset.save_wavfile(fake_wav[0].detach().cpu(), wav_path)
        print('New wave-file was generated.')


# In[ ]:


def main(args):
    hyper_params = {}
    hyper_params['Sound Dir'] = args.sound_dir
    hyper_params['Result Dir'] = args.result_dir
    hyper_params['Weight Dir'] = args.weight_dir
    hyper_params['Sound Length'] = args.sound_length
    hyper_params["Feed's dim"] = args.feed_dim
    hyper_params['Learning Rate'] = args.lr
    hyper_params["Mul Discriminator's LR"] = args.mul_lr_dis
    hyper_params['Batch Size'] = args.batch_size
    hyper_params['Num Train'] = args.num_train
    
    solver = Solver(args)
    solver.load_state()
    
    if not args.noresume:
        solver = solver.load_resume()
    
    if args.generate > 0:
        solver.generate(args.generate)
        return
        
    for key in hyper_params.keys():
        print(f'{key}: {hyper_params[key]}')
    #experiment.log_parameters(hyper_params)
    
    solver.train(not args.noresume)
    
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sound_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--sound_length', type=int, default=16384)
    parser.add_argument('--feed_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mul_lr_dis', type=float, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--noresume', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

