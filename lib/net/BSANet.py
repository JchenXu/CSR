import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from net.backbone import build_backbone
from net.ASPP import ASPP

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class OurModule(nn.Module):
    def __init__(self, in_features, cfg):
        super(OurModule, self).__init__()

        self.branch1_1 = nn.Sequential(
                       conv3x3(in_features, in_features), 
                       SynchronizedBatchNorm2d(in_features, momentum=cfg.TRAIN_BN_MOM),
                       nn.ReLU(inplace=False))
        self.branch1_2 = nn.Sequential(
                       conv3x3(in_features, in_features), 
                       SynchronizedBatchNorm2d(in_features, momentum=cfg.TRAIN_BN_MOM),
                       nn.ReLU(inplace=False))

        self.branch2 = nn.Sequential(
                       conv3x3(in_features, in_features), 
                       SynchronizedBatchNorm2d(in_features, momentum=cfg.TRAIN_BN_MOM),
                       nn.ReLU(inplace=False),
                       ) # nn.AvgPool2d(size)

        self.branch3_1 = nn.Sequential(
                       conv3x3(in_features, in_features), 
                       SynchronizedBatchNorm2d(in_features, momentum=cfg.TRAIN_BN_MOM),
                       nn.ReLU(inplace=False))
        self.branch3_2 = nn.Sequential(
                       conv3x3(in_features, in_features), 
                       SynchronizedBatchNorm2d(in_features, momentum=cfg.TRAIN_BN_MOM),
                       nn.ReLU(inplace=False))

        self.fuse_minus_plus = nn.Sequential(
                       conv3x3(2*in_features, in_features),
                       SynchronizedBatchNorm2d(in_features, momentum=cfg.TRAIN_BN_MOM),
                       nn.ReLU(inplace=False))

        self.sigmoid = nn.Sigmoid() 
          
    def forward(self, x, sal_img):
        batch_size, im_size = sal_img.shape[:2]

        local1 = self.branch1_1(x)
        local1 = self.branch1_2(local1)

        local2 = self.branch3_1(x)
        local2 = self.branch3_2(local2)

        sal_img_flatten = sal_img.reshape((batch_size, im_size * im_size))

        x = self.branch2(x) #.repeat([1, 1, self.size[0], self.size[1]])
        x_flatten_trans = torch.reshape(x, (batch_size, -1, im_size * im_size)).transpose(1, 2)

        not_bg_index = torch.nonzero(sal_img_flatten > 35)	
        not_fg_index = torch.nonzero(sal_img_flatten < 36)

        bg_feature = x_flatten_trans.clone()
        fg_feature = x_flatten_trans.clone()

        bg_flag = torch.ones((batch_size, im_size*im_size, 1)).cuda()

        if not_bg_index.shape[0] != 0:
            bg_feature[not_bg_index[:, 0], not_bg_index[:, 1]] = 0

            # 1 if bg pixel, else 0;  so that 1-bg_flag is fg_flag
            bg_flag[not_bg_index[:, 0], not_bg_index[:, 1]] = 0 

        if not_fg_index.shape[0] != 0:
            fg_feature[not_fg_index[:, 0], not_fg_index[:, 1]] = 0

        mean_bg = torch.sum(bg_feature, 1) / (torch.sum(bg_flag, 1) + 1e-4)
        mean_fg = torch.sum(fg_feature, 1) / (torch.sum(1-bg_flag, 1) + 1e-4)

        mean_bg = bg_flag * mean_bg.unsqueeze(1).repeat([1, im_size*im_size, 1])
        mean_fg = (1-bg_flag) * mean_fg.unsqueeze(1).repeat([1, im_size*im_size, 1])

        global_feature = (mean_bg + mean_fg).transpose(1, 2).reshape(batch_size, -1, im_size, im_size)

        minus_feature = -(global_feature - local1)        
        plus_feature = global_feature + local2

        minus_plus_feature = torch.cat([minus_feature, plus_feature], dim=1)
        
        return self.sigmoid(self.fuse_minus_plus(minus_plus_feature))


class BSANet(nn.Module):
	def __init__(self, cfg, SEMANTIC_NUM=21):
		super(BSANet, self).__init__()
		print('semantic num : {}'.format(SEMANTIC_NUM))

		self.SEMANTIC_NUM = SEMANTIC_NUM

		self.backbone = None
		self.backbone_layers = None
		input_channel = 2048
		self.aspp = ASPP(dim_in=input_channel,
				dim_out=cfg.MODEL_ASPP_OUTDIM,
				rate=16//cfg.MODEL_OUTPUT_STRIDE,
				bn_mom = cfg.TRAIN_BN_MOM)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)
		self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)


		# set the dim of ASPP output, default set as 256 for memory limitation
		indim = 256
		asppdim=256

		print('ASPP')


		self.edge_conv = nn.Sequential(
				nn.Conv2d(indim, indim, 1, 1,
						  padding=0, bias=True),
				SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),

				nn.Conv2d(indim, indim // 2, 3, 1,
						  padding=1, bias=True),
				SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),

				nn.Conv2d(indim // 2, indim // 2, 1, 1,
						  padding=0, bias=True),
				SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
		)

		self.edge_outconv=nn.Conv2d(indim //2,2, kernel_size=1, padding=0, dilation=1, bias=True)

		self.edge_up=nn.UpsamplingBilinear2d(scale_factor=4)

		self.sigmoid = nn.Sigmoid()

		print('EDGE')

		##mid level encoder
		self.midedge_conv = nn.Sequential(
			nn.Conv2d(indim*2, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim // 2, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim // 2, indim // 2, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
		)

		self.shortcut_conv_mid = nn.Sequential(
			nn.Conv2d(indim*2, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim // 2, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim // 2, indim // 2, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
		)

		self.shortcut_conv_high = nn.Sequential(
			nn.Conv2d(indim*4, indim, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL // 2,
					  bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)

		self.highedge_conv = nn.Sequential(
			nn.Conv2d(indim*4, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
		)

		self.highedge_outconv = nn.Conv2d(indim, 2, kernel_size=1, padding=0, dilation=1, bias=True)


		self.midedge_outconv = nn.Conv2d(indim // 2, 2, kernel_size=1, padding=0, dilation=1, bias=True)

		self.midedge_up = nn.UpsamplingBilinear2d(scale_factor=8)

		## low-level feature transformation 
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim  , indim //2, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
				SynchronizedBatchNorm2d(indim //2, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
		)


		self.query = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv2d(indim, indim//2, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
			SynchronizedBatchNorm2d(indim//2, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)


		#
		self.cat_conv1 = nn.Sequential(
			nn.Conv2d(cfg.MODEL_ASPP_OUTDIM + indim // 2, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0, bias=True),
			SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)
		self.relu=nn.ReLU(inplace=True)


		## semantic encoder with 21 classes
		self.semantic_encoding = nn.Sequential(
			nn.Conv2d(asppdim, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
		)

		self.avg_pool=nn.AdaptiveAvgPool2d(1)
		self.semantic_fc = nn.Sequential(
			nn.Conv2d(indim, asppdim, 3, 2,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(asppdim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)


		self.fc = nn.Sequential(
			nn.Linear(asppdim, asppdim // 4,bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(asppdim //4, asppdim, bias=False),
			nn.Sigmoid(),
		)


		self.semantic_output = nn.Conv2d(indim, 21, 1, 1, padding=0)

		self.upsample_conv = nn.Sequential(
			nn.Conv2d(indim, indim, 1, 1, padding=1 // 2,
					  bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)
		##cfg.MODEL_SHORTCUT_DIM
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+indim, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)

		self.fuse = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM*2, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
		)

		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM*2, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone_layers = self.backbone.get_layers()

		# self.before_aspp = OurModule(2048, cfg)
		# self.after_aspp = OurModule(256, cfg)

		print("SALIENCY NOT USED - oneused")

	def forward(self, x,mode, saliency):

		# mode: option "train" "test"
		# This code is simplified to get readability from the original implementation

		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)

		feature_0 = self.shortcut_conv(layers[1])
		feature_1 = self.shortcut_conv_mid(layers[2])
		feature_2 = self.shortcut_conv_high(layers[3])

		edge2 = self.highedge_conv(layers[3])

		edge2_r = self.highedge_outconv(edge2)

		feature_2 = torch.mul(feature_2, edge2)

		feature_2 = self.query(feature_2)

		feature_cat = torch.cat([feature_aspp,feature_2],1)

		feature_cat = self.cat_conv1(feature_cat)

		feature_edge = self.edge_conv(layers[1])

		edge = self.edge_outconv(feature_edge)
		edge_r = self.edge_up(edge)

		attention_edg= self.sigmoid(feature_edge)

		feature_low=torch.mul(feature_0, feature_edge)

		##mid
		feature_edge1 = self.midedge_conv(layers[2])
		edge1 = self.midedge_outconv(feature_edge1)

		edge1_r = self.midedge_up(edge1)

		b, c, h, w = edge1.size()

		attention_edg1 = self.sigmoid(feature_edge1)
		feature_mid = torch.mul(feature_1, attention_edg1)

		feature_mid = self.upsample2(feature_mid)

		feature_cat = self.upsample_sub(feature_cat)
		feature_cat = torch.cat([feature_cat,feature_mid,feature_low],1)


		feature_cat = self.cat_conv(feature_cat)

		b, c, _, _ = feature_cat.size()

		#### semantic encoding


		feature_semantic = self.semantic_encoding(feature_cat)

		fc_att = self.semantic_fc(feature_semantic)
		fc_att = self.avg_pool(fc_att).view(b,c)
		fc_att = self.fc(fc_att)
		# use dense attention to get a little higher performance boost +0.30% miou

		fc_att= fc_att.view(b, c, 1, 1)
		feature_final = F.relu_(feature_cat + torch.mul(feature_cat,self.sigmoid(feature_semantic)))


		ins_r = self.semantic_output(feature_semantic)

		# ==========================
		semantic_map = torch.argmax(ins_r, 1, keepdim=False) # (B, H, W)
		batch_size, channel_num, H, W = feature_final.shape

		global_map = torch.zeros_like(feature_final)
		for i in range(self.SEMANTIC_NUM):
			bool_map = (semantic_map == i).float().unsqueeze(1) # (B, 1, H, W)
			i_feature_map = feature_final * bool_map # (B, channel, H, W)
			i_feature_map_mean = torch.sum(i_feature_map, [2, 3]) / (torch.sum(bool_map, [2, 3]) + 1e-4) # (B, channel)
			i_global_map = torch.bmm(i_feature_map_mean.unsqueeze(-1), bool_map.reshape(batch_size, 1, -1))
			i_global_map = i_global_map.reshape(batch_size, channel_num, H, W)

			global_map = global_map + i_global_map
		# # ==========================

		feature_final = torch.cat([feature_final, global_map], dim=1)


		result = self.cls_conv(feature_final)
		result = self.upsample4(result)

		if mode=='train':
			return result,edge_r,edge1_r,edge2_r,ins_r
		else:
			return result
