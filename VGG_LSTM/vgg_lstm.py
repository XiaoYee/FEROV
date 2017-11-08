import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

class LSTMNet(nn.Module):

    def __init__(self):
        super(LSTMNet, self).__init__()
        # input 3 channel
        self.conv1_1 = nn.Conv2d(3,  64, 3,padding=(1,1))
        self.conv1_2 = nn.Conv2d(64, 64, 3,padding=(1,1))
        self.conv2_1 = nn.Conv2d(64, 128,3,padding=(1,1))
        self.conv2_2 = nn.Conv2d(128,128,3,padding=(1,1))
        self.conv3_1 = nn.Conv2d(128,256,3,padding=(1,1))
        self.conv3_2 = nn.Conv2d(256,256,3,padding=(1,1))
        self.conv3_3 = nn.Conv2d(256,256,3,padding=(1,1))
        self.conv4_1 = nn.Conv2d(256,512,3,padding=(1,1))
        self.conv4_2 = nn.Conv2d(512,512,3,padding=(1,1))
        self.conv4_3 = nn.Conv2d(512,512,3,padding=(1,1))
        self.conv5_1 = nn.Conv2d(512,512,3,padding=(1,1))
        self.conv5_2 = nn.Conv2d(512,512,3,padding=(1,1))
        self.conv5_3 = nn.Conv2d(512,512,3,padding=(1,1))

        # self.features = self.make_layers(cfg['D'])

        self.fc6 = nn.Linear(512*7*7, 4096)
        # lstm input_dim=4096,output_dim=128
        self.lstm = nn.LSTM(4096, 128, batch_first=True)
        # h0,c0  first 1 mean net architecture,batchsize=1,output_dim=128
        self.h0 = Variable(torch.randn(1, 1, 128).cuda())
        self.c0 = Variable(torch.randn(1, 1 ,128).cuda())

        self.fc8_final = nn.Linear(128, 7)
        self.seen = 0

    def forward(self, x):

    	x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(F.relu(self.conv1_2(x)), 2, stride=2)
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(F.relu(self.conv2_2(x)), 2, stride=2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.max_pool2d(F.relu(self.conv3_3(x)), 2, stride=2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.max_pool2d(F.relu(self.conv4_3(x)), 2, stride=2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.max_pool2d(F.relu(self.conv5_3(x)), 2, stride=2)
        # x = self.features(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc6(x))
        x = F.dropout(x,0.6)        
        # print x.size()
        x = x.view(1, -1, self.num_flat_features(x))   #1,16,4096 batchsize,sequence_length,data_dim
        x, (h_out, c_out) = self.lstm(x, (self.h0, self.c0))
        x = F.dropout(x,0.6) 
        x = self.fc8_final(x)
        return x


    def make_layers(self,cfg,batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)



    def num_flat_features(self, x):
        size = x.size()[1:]   
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
