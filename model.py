import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import torch.optim as optim

lat_dim= 128

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear_3 = nn.Linear(129, 256) ##161
        self.linear_4 = nn.Linear(256, 1344)
        self.conv_4 = nn.ConvTranspose2d(64,32, kernel_size=(11,3), stride=(2,2), padding=0,output_padding=(0,0))
        self.conv_5 = nn.ConvTranspose2d(32,16, kernel_size=(11,3), stride=(2,2), padding=0,output_padding=(0,1))
        self.conv_6 = nn.ConvTranspose2d(16,1, kernel_size=(11,3), stride=(2,2),padding=0,output_padding=(1,0))
        self.relu = nn.ReLU()
    
    def forward(self, z, y):
        z_cond = torch.cat((z,y.unsqueeze(1)), dim=1)
        z_cond = F.selu(self.linear_3(z_cond))
        z_cond = F.selu(self.linear_4(z_cond))
        z_cond = z_cond.view(z_cond.size(0), 64, 7, 3)# (N,C,H)\n",
        z_cond = self.relu(self.conv_4(z_cond))
        z_cond = self.relu(self.conv_5(z_cond))
        z_cond = self.relu(self.conv_6(z_cond))
        y0 = z_cond.contiguous().view(z_cond.size(0), -1) # (N,C,H)\n",
        y1 = F.softmax(y0, dim=1)

        y = y1.contiguous().view(z_cond.size(0), z_cond.size(2), z_cond.size(3))
        
        return y


class MolecularICVAE(nn.Module):
    def __init__(self):
        super(MolecularICVAE, self).__init__()

        self.conv_1 = nn.Conv2d(1, 16, (11,3), stride=(2,2))
        self.conv_2 =nn.Conv2d(16, 32, (11,3), stride=(2,2))
        self.conv_3 = nn.Conv2d(32, 64, (11,3), stride=(2,2))
        self.linear_0 = nn.Linear(1344, 256)
        self.linear_1 = nn.Linear(256, lat_dim)
        self.linear_2 = nn.Linear(256, lat_dim)
        self.relu = nn.ReLU()
        self.decode = Decoder()
        
    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)
    
    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def forward(self, x, y, y_arg):
        x_cond = torch.cat((x,y.view(y.size(0), 1, 1, -1)), dim=2)
        z_mean, z_logvar = self.encode(x_cond)
        z = self.sampling(z_mean, z_logvar)
        decoder = self.decode(z, y_arg)
        
        return decoder, z_mean, z_logvar,z

