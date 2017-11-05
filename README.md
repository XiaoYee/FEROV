This project is a video facial expression recognization project

Network structure is based on VGG16+LSTM,implemented by Pytorch0.2

This is also my first Pytorch project,now can only process one video per batchsize

本项目是一个基于视频的面部表情分析项目

1.将训练和测试视频目录分别提取到TrainTestlist下的trainval和test中，自行提取

2.运行get_information.py，提取所有视频的长度到info.pkl中

3.network.cfg可调整网络配置

4.运行vgg_train.py进行网络的训练和测试




