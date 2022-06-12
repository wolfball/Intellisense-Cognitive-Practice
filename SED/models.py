import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)

class Crnn(nn.Module):
    def __init__(self, num_freq, class_num):
        super(Crnn, self).__init__()
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        self.num_freq = num_freq
        self.class_num = class_num
        self.fc = nn.Linear(num_freq, class_num)
        self.batchnorm = nn.BatchNorm1d(num_freq)
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2))
        self.conv_block4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2))
        # self.conv_block5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
        #                                  nn.BatchNorm2d(256),
        #                                  nn.ReLU(),
        #                                  nn.MaxPool2d(2))
        self.biGRU = nn.GRU(64*8, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(64*2, class_num)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        bs, ts, nf = x.shape
        x_bn = self.batchnorm(x.permute(0,2,1)).view(bs,1,nf,ts)  # (bs, 1, 64, 501)
        h1 = self.conv_block1(x_bn)  # (bs, 16, 32, 250)
        h2 = self.conv_block2(h1)  # (bs, 32, 16, 125)
        h3 = self.conv_block3(h2)  # (bs, 64, 8, 62)
        h4 = self.conv_block4(h3)
        timestep = h4.shape[-1]
        f, _ = self.biGRU(h4.view(bs,-1,timestep).permute(0,2,1))  # (bs, 62, 128)
        y = self.fc(f)  # (bs, 62, 10)
        out = torch.sigmoid(y)  # (bs, 62, 10)
        out_ = F.interpolate(out.permute(0,2,1), ts).permute(0,2,1)
        return out_
        
    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''

        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }
