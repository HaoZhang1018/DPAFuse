import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def group_norm(channels):
    return nn.GroupNorm(32, channels)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CABM(nn.Module):
    def __init__(self, channel):
        super(CABM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class FusionNet(nn.Module):
    def __init__(self,
                 in_channels=16,
                 model_channels=64,
                 out_channels=8,        
                 ):
        super(FusionNet,self).__init__()

   
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
       

        self.inBlock = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.middleBlock1 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )
        self.CABM1=CABM(channel=model_channels)
        
        self.middleBlock2 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )
        self.middleBlock3 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )
        # self.middleBlock4 = nn.Sequential(
        #     group_norm(model_channels),
        #     nn.SiLU(),
        #     nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        # )   
        
        self.CABM2=CABM(channel=model_channels)
        self.CABM3=CABM(channel=model_channels)        
        # self.CABM4=CABM(channel=model_channels)  
        
        self.outBlock1= nn.Sequential(
            group_norm(model_channels),                  
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
      
        self.outBlock2 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.alpha = Parameter(torch.tensor(1.5))
        # self.outBlock3 = nn.Sequential(
        #     group_norm(model_channels),
        #     nn.SiLU(),
        #     nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1)
        # )

    def forward(self, h1,h2):
        # print("h1 shape: ", h1.shape, "h2 shape: ", h2.shape)
        h=torch.cat([h1,h2],  dim=1)   
        middle1=self.inBlock(h) 
        
        middle2=self.middleBlock1(middle1)
        m22=self.CABM1(middle2)
        
        middle3=self.middleBlock2(m22)
        m1=self.CABM2(middle3)        
        
        middle4=self.middleBlock3(m22)
        m2=self.CABM3(middle4)
        
        # middle5=self.middleBlock4(m22)
        # m3=self.CABM4(middle5)   
                        
        w_v=self.outBlock1(m1)
        w_i = self.outBlock2(m2)
        # w_i_scaled = w_i * self.alpha
        # bias=self.outBlock3(m3)
        # print("w_v mean: ", torch.mean(w_v).item(), "w_i mean: ", torch.mean(w_i).item())
        alpha = w_v / (w_v + w_i )
        beta  = w_i / (w_v + w_i )

        out = alpha * h1 + beta * h2
        # out= w_v/(w_i_scaled+w_v)*h1 + w_i_scaled/(w_i_scaled+w_v)*h2
        # out=(1-w_i)*h1  +  w_i*h2
        return out

       


class Discriminator_Fea(nn.Module):
    def __init__(self, 
                 input_channel=8, num_filters_last=64, n_layers=3):
        super(Discriminator_Fea, self).__init__()

        layers = [nn.Conv2d(input_channel, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 2, 4, 2, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), 2)
    
        return  out



class Discriminator_Img(nn.Module):
    def __init__(self, 
                 input_channel=3, num_filters_last=64, n_layers=3):
        super(Discriminator_Img, self).__init__()

        layers = [nn.Conv2d(input_channel, num_filters_last, 7, 4, 3), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 7,
                          4, 3, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 2, 7, 4, 3))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), 2)
    
        return  out



