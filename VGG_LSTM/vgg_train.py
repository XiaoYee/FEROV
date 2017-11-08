from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable
import numpy

import dataset
import random
import math
from utils import *
# from cfg import parse_cfg
from vgg_lstm import LSTMNet

cfgfile = "network.cfg"

net_options   = parse_cfg(cfgfile)[0]


trainlist     = "TrainTestlist/trainval.txt"
testlist      = "TrainTestlist/test.txt"
backupdir     = "backup"
num_workers   = 4

nsamples      = file_lines(trainlist)

batch_size    = int(net_options['batchsize'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]
init_width    = int(net_options['width'])
init_height   = int(net_options['height'])
crop_length   = int(net_options['length'])

#Train parameters
max_epochs    = max_batches*batch_size/nsamples+1
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
save_interval = 100      #epoches

torch.manual_seed(seed)

if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    torch.cuda.manual_seed(seed)

model         = LSTMNet()

# pretrained model 
# vgg16 = torch.load('vgg16.pth')
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in vgg16.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

# vgg16 = models.vgg16(pretrained=True)
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in vgg16.state_dict().items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

#load model
model = torch.load('backup/model_0800.pth')

processed_batches = model.seen/batch_size

init_epoch        = model.seen/nsamples 

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

model = model.cuda()

params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]

optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)


def adjust_learning_rate(optimizer, batch):
    #Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr


loss_function = nn.CrossEntropyLoss()


def train(epoch):

    global processed_batches

    transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                       ])

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),length = crop_length,
                       shuffle=True,
                       transform=transform, 
                       train=True, 
                       seen=model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))

    running_loss = 0.0

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # sequence input,squeeze to fit input of conv2d
        data = data.squeeze(0)

        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        if use_cuda:
            data = data.cuda()
            target= target.cuda()
        data, target = Variable(data), Variable(target.long())
        # print(data)  torch.cuda.FloatTensor of size 16x3x224x224 (GPU 0)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(0)
        loss = loss_function(output, target[0])
        running_loss += loss.data[0] * target.size(0)

        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 20 == 0:
            logging('Loss:{:.6f}'.format(running_loss / (batch_size * (batch_idx+1))))


    if (epoch+1) % save_interval == 0:
        logging('save weights to %s/model_%04d.pth' % (backupdir, epoch+1))
        model.seen = (epoch + 1) * len(train_loader.dataset)
        torch.save(model,'%s/model_%04d.pth' % (backupdir, epoch+1))


def test(epoch):

    transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])

    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(testlist, shape=(init_width, init_height),length = crop_length,
                    shuffle=False,
                    transform=transform,
                    train=False,
                    num_workers=num_workers),
                    batch_size=batch_size, shuffle=False, **kwargs)

    model.eval()
    right_number = 0
    num = 0

    for batch_idx, (data, target) in enumerate(test_loader):

        data = data.squeeze(0)
        if use_cuda:
            data = data.cuda()
            # target= target.cuda()

        # data, target = Variable(data), Variable(target.long())
        data = Variable(data,volatile=True)

        output = model(data)
        output = output.squeeze(0)

        output_predict = output.data.cpu().numpy() 
        label = target[0][0]

        avg_pred_batch = np.mean(output_predict, axis=0)
        avg_pred = np.array(softmax(avg_pred_batch))
        prediction_label = np.argmax(avg_pred)

        if prediction_label == label:
            right_number += 1
        num += 1
    accuracy = float(right_number)/float(num)
    logging("test accuracy: %f" % (accuracy))


for epoch in range(init_epoch, max_epochs): 
    train(epoch)
    test(epoch)