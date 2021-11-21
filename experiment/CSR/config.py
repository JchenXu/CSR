import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
	def __init__(self):
		self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
		self.EXP_NAME = 'CSR'


		# default setting Unified / Person
		self.DATA_NAME = 'Unified'
		self.DATA_AUG = False
		self.DATA_WORKERS = 4
		self.DATA_RESCALE = 512
		self.DATA_RANDOMCROP = 512
		self.DATA_RANDOMROTATION = 0
		self.DATA_RANDOMSCALE = 2
		self.DATA_RANDOM_H = 10
		self.DATA_RANDOM_S = 10
		self.DATA_RANDOM_V = 10
		self.DATA_RANDOMFLIP = 0
		
		self.MODEL_NAME = 'BSANet'
		self.MODEL_BACKBONE = 'res101_atrous'


		## output stride=16 for memory and time saving. 
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_ASPP_OUTDIM = 256


		# this could be modified to 3 for better results.
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = 58
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model',self.EXP_NAME)


		# Modify the parameters if fine-tuning

		self.TRAIN_LR = 0.007
		self.TRAIN_LR_GAMMA = 0.1
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_WEIGHT_DECAY = 0.00004
		self.TRAIN_BN_MOM = 0.0003
		self.TRAIN_POWER = 0.9

		# Better results can be achieved by large batches. We use at least 2 GPUs to get the given results.
		# If using single GPU, modify the cuda, set parameters .to(device) and set net = nn.DataParallel(net, device_ids=[0])
		self.TRAIN_GPUS = 2
		self.TRAIN_BATCHES = 4
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 0
		self.TRAIN_EPOCHS = 100
		self.TRAIN_LOSS_LAMBDA = 0
		self.TRAIN_TBLOG = True


		### modify this if using pretrained model. If None, download the imagenet model automatically.
		self.TRAIN_CKPT = 'resnet101-5d3b4d8f.pth'

		### Modify the log dir

		self.LOG_DIR = os.path.join(self.ROOT_DIR,'log',self.EXP_NAME)

		self.TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
		# self.TEST_MULTISCALE = [1.0]
		self.TEST_FLIP = True

		### Model Path for test 
		self.TEST_CKPT = os.path.join(self.ROOT_DIR,'your_download_ckpt.pth')

		### If using single GPU, modify the cuda, set parameters .to(device) and set net = nn.DataParallel(net, device_ids=[0])
		self.TEST_GPUS = 2
		self.TEST_BATCHES = 2

		self.__check()
		self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
		
	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.TRAIN_GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)

	def __add_path(self, path):
		if path not in sys.path:
			sys.path.insert(0, path)



cfg = Configuration() 	
