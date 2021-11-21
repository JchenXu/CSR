import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np

import torch.nn.functional as F

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.loss import MaskCrossEntropyLoss, MaskBCELoss, MaskBCEWithLogitsLoss
from net.sync_batchnorm.replicate import patch_replication_callback

map_lst = {
            1: [1, 2, 3, 4, 5],
            2: [6, 7],
            3: [8, 9, 10, 11],
            4: [12],
            5: [13, 14],
            6: [15, 16, 17],
            7: [18, 19, 20, 21, 22],
            8: [23, 24, 25, 26],
            9: [27],
            10: [28, 29, 30, 31],
            11: [32],
            12: [33, 34, 35, 36],
            13: [37, 38, 39, 40],
            14: [41, 42],
            15: [43, 44, 45, 46, 47, 48],
            16: [49, 50],
            17: [51, 52, 53],
            18: [54],
            19: [55],
            20: [56, 57]
           }

import rmi_utils

_euler_num = 2.718281828                # euler number
_pi = 3.14159265                        #     pi
_ln_2_pi = 1.837877                        #     ln(2 * pi)
_CLIP_MIN = 1e-6                        #     min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0                            #     max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4                        #     add this factor to ensure the AA^T is positive definite
_IS_SUM = 1                                #     sum the loss per channel


class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """
    def __init__(self,
                    num_classes=21,
                    rmi_radius=3,
                    rmi_pool_way=0,
                    rmi_pool_size=3,
                    rmi_pool_stride=3,
                    loss_weight_lambda=0.5,
                    lambda_way=1):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = 255

    def forward(self, logits_4D, labels_4D):
        loss = self.forward_sigmoid(logits_4D, labels_4D)
        return loss

    def onehot(self, labels, class_num):
        N, H, W = labels.shape[0], labels.shape[1], labels.shape[2]
        labels_flat = labels.view(-1, 1)
        bs = labels_flat.shape[0]
        one_hot = torch.zeros(bs, class_num).to(1).scatter_(1, labels_flat, 1)

        return one_hot.view(N, H, W, -1)


    def forward_sigmoid(self, logits_4D, labels_4D):
        """
        Using the sigmiod operation both.
        Args:
            logits_4D     :    [N, C, H, W], dtype=float32
            labels_4D     :    [N, H, W], dtype=long
        """
        # label mask -- [N, H, W, 1]
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4D = self.onehot(labels_4D.long() * label_mask_3D.long(), class_num=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4D.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
                                                            target=valid_onehot_label_flat,
                                                            weight=label_mask_flat.unsqueeze(dim=1),
                                                            reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                else bce_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
            labels_4D     :    [N, C, H, W], dtype=float32
            probs_4D     :    [N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = rmi_utils.map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).to(1).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        #pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        #appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        #appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * rmi_utils.log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        #rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        #is_half = False
        #if is_half:
        #    rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        #else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss


def train_net():
    dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
    dataloader = DataLoader(dataset, 
                batch_size=cfg.TRAIN_BATCHES, 
                shuffle=cfg.TRAIN_SHUFFLE, 
                num_workers=cfg.DATA_WORKERS,
                drop_last=True)

    print('dataset finishhhh')
    
    net = generate_net(cfg)
    if cfg.TRAIN_TBLOG:
        from tensorboardX import SummaryWriter
        # Set the Tensorboard logger
        tblogger = SummaryWriter(cfg.LOG_DIR)

    

    print('Use %d GPU'%cfg.TRAIN_GPUS)
    device = torch.device(0)
    if cfg.TRAIN_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.to(device)        

    if cfg.TRAIN_CKPT:
        pretrained_dict = torch.load(cfg.TRAIN_CKPT)
        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        # net.load_state_dict(torch.load(cfg.TRAIN_CKPT),False)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    criterion_edge = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1,0.9]).cuda(1))
    optimizer = optim.SGD(
        params = [
            {'params': get_params_init(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
            {'params': get_params_init(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
        ],
        momentum=cfg.TRAIN_MOMENTUM
    )
    itr = cfg.TRAIN_MINEPOCH * len(dataloader)
    max_itr = cfg.TRAIN_EPOCHS*len(dataloader)
    running_loss = 0.0
    tblogger = SummaryWriter(cfg.LOG_DIR)

    f_out = open('loss.txt', 'w')

    upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
    for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):

        for i_batch, sample_batched in enumerate(dataloader):


            now_lr = adjust_lr(optimizer, itr, max_itr)

            inputs_batched, labels_batched,ins_batched,edge_batched = sample_batched['image'], sample_batched['segmentation'],sample_batched['instance'],sample_batched['edge']
            optimizer.zero_grad()

            saliency_batched_32x32 = F.interpolate(sample_batched['saliency'].unsqueeze(1), size=[32,32], mode='nearest')
            predicts_batched,predicts_edge,predicts_edge1,predicts_edge2,predicts_ins = net(inputs_batched,'train',saliency_batched_32x32.squeeze(1))

            labels_batched = labels_batched.long().to(1)



            predicts_ins=upsample4(predicts_ins)
            predicts_ins =predicts_ins.to(1)


            ##aux loss
            '''
            predicts_cat = predicts_cat.to(1)
            predicts_semantic = predicts_semantic.to(1)
            #
            '''


            edge_batched = edge_batched.long().to(1)
            ins_batched = ins_batched.long().to(1)



            loss_ins = criterion(predicts_ins, ins_batched)


            predicts_edge = predicts_edge.to(1)
            predicts_edge1 = predicts_edge1.to(1)
            predicts_edge2 =predicts_edge2.to(1)
            predicts_edge2=F.interpolate(predicts_edge2,scale_factor=16,mode='bilinear')

            loss_edge3 = criterion_edge(predicts_edge2, edge_batched)
            loss_edge1 = criterion_edge(predicts_edge, edge_batched)

            predicts_batched = predicts_batched.to(1)
            # loss_p = criterion(predicts_batched, labels_batched)
            loss_edge_2 = criterion_edge(predicts_edge1, edge_batched)

            # ===================================

            # feature_map: [B, C, H, W]
            
            rmi = RMILoss(58)
            logits = predicts_batched
            labels = labels_batched
            rmi_l = rmi(logits, labels)

            feature_map = predicts_batched
            C = feature_map.shape[1]

            preds_possi = F.softmax(predicts_batched, 1)

            # print(feature_map.shape, preds_possi.shape)
            feature_map = feature_map.permute(0, 2, 3, 1).contiguous()
            preds_possi = preds_possi.permute(0, 2, 3, 1).contiguous()
            # print(feature_map.shape, preds_possi.shape, target.shape)

            feature_map_flatten = feature_map.view(-1, C)
            target_flatten = labels_batched.view(-1)
            possi_flatten = preds_possi.view(-1, 58)

            L1_list = []
            L2_list = []
            for i in range(1, len(map_lst)+1, 1):
                feature_same_semantic = []
                feature_distin = []
                for part_id in map_lst[i]:
                    idx = torch.nonzero(target_flatten == part_id).view(-1)
                    numel = idx.numel()

                    if numel <= 10:
                        continue

                    feature_select = feature_map_flatten[idx]
                    # sort_idx = torch.argsort(possi_flatten[idx][:, part_id])

                    top_idx = torch.topk(possi_flatten[idx][:, part_id], int(numel*0.3))[1]
                    low_idx = torch.topk(possi_flatten[idx][:, part_id], int(numel*0.3), largest=False)[1]

                    feature_same_semantic.append(feature_select[low_idx])
                    feature_distin.append(torch.mean(feature_select[top_idx], 0))

                for j in range(len(feature_distin)):
                    target_feature = feature_distin[j].unsqueeze(0)  # [1, C]

                    L1_feature = feature_same_semantic[j] # [num, C]
                    L1 = torch.mean(1 - F.cosine_similarity(target_feature.detach(), L1_feature, 1))
                    L1_list.append(L1.view(1))

                    L2_feature_list = feature_same_semantic[:j] + feature_same_semantic[j+1:]
                    if len(L2_feature_list) > 0:
                        L2_feature = torch.cat(L2_feature_list, 0) # [num2, C]
                        L2 = torch.mean(1 - F.cosine_similarity(target_feature.detach(), L2_feature, 1))
                        # L2_list.append(torch.where(L2 > L1, L2.view(1), L1.view(1)))

                        max_value = torch.log(torch.exp(L2.view(1)) + torch.exp(L1.view(1)))
                        L2_list.append(max_value)

            if L1_list:
                L1 = torch.mean(torch.cat(L1_list))
            else:
                L1_list = [torch.tensor([[2.0]]).to(0)]
                L1 = torch.mean(torch.cat(L1_list))

            if L2_list:
                L2 = torch.mean(torch.cat(L2_list))
            else:
                L2_list = [torch.tensor([[2.0]]).to(0)]
                L2 = torch.mean(torch.cat(L2_list))

            # ===================================

            loss = 0.1 * loss_edge1.to(1) + 0.1*loss_edge_2.to(1)+loss_edge3.to(1)*0.03+loss_ins.to(1)*0.2 + rmi_l.to(1) + 0.01 * L1.to(1) + 0.01*L2.to(1)



            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            print('epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g ' % 
                (epoch, cfg.TRAIN_EPOCHS, i_batch, dataset.__len__()//cfg.TRAIN_BATCHES,
                itr+1, now_lr, running_loss))

            f_out.write('epoch:{}/batch:{}/itr:{}/total_loss:{}/rmi_l:{}/L1:{}/L2:{}\n'.format(epoch, i_batch, itr+1, running_loss, rmi_l.item(), L1.item(), L2.item()))
            f_out.flush()

            if cfg.TRAIN_TBLOG and itr%200 == 0:

                inputs = inputs_batched.numpy()[0]/2.0 + 0.5
                labels = labels_batched[0].cpu().numpy()
                labels_color = dataset.label2colormap(labels).transpose((2,0,1))
                predicts = torch.argmax(predicts_batched[0],dim=0).cpu().numpy()
                predicts_color = dataset.label2colormap(predicts).transpose((2,0,1))
                pix_acc = np.sum(labels==predicts)/(cfg.DATA_RESCALE**2)

                tblogger.add_scalar('loss', running_loss, itr)
                # tblogger.add_scalar('loss_part', loss_p, itr)
                tblogger.add_scalar('loss_ins', loss_ins, itr)


                tblogger.add_scalar('lr', now_lr, itr)
                tblogger.add_scalar('pixel acc', pix_acc, itr)
                tblogger.add_image('Input', inputs, itr)
                tblogger.add_image('Label', labels_color, itr)
                tblogger.add_image('Output', predicts_color, itr)

                edges = edge_batched[0].cpu().numpy()
                edges_color = dataset.label2colormap(edges).transpose((2, 0, 1))

                edge_pre = torch.argmax(predicts_edge[0], dim=0).cpu().numpy()
                edge_pre_color = dataset.label2colormap(edge_pre).transpose((2, 0, 1))
                tblogger.add_image('edge_label', edges_color, itr)
                tblogger.add_image('edge_pred', edge_pre_color, itr)

                instances = ins_batched[0].cpu().numpy()
                instances_color = dataset.label2colormap(instances).transpose((2, 0, 1))

                ins_pre = torch.argmax(predicts_ins[0], dim=0).cpu().numpy()
                ins_pre_color = dataset.label2colormap(ins_pre).transpose((2, 0, 1))
                tblogger.add_image('instances_color', instances_color, itr)
                tblogger.add_image('ins_prediction_color', ins_pre_color, itr)



            running_loss = 0.0
            
            if itr % 5000 == 0:
                save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,itr))
                torch.save(net.state_dict(), save_path)
                print('%s has been saved'%save_path)

            itr += 1
        
    save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_EPOCHS))        
    torch.save(net.state_dict(),save_path)
    if cfg.TRAIN_TBLOG:
        tblogger.close()
    print('%s has been saved'%save_path)


def myloss(predicts_batched, labels_batched):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    loss_part = criterion(predicts_batched, labels_batched)
    weight_criterion=nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1,0.9]))
    loss_edge = weight_criterion(predicts_batched, labels_batched)
    loss=loss_part+loss_edge
    return loss

def adjust_lr(optimizer, itr, max_itr):
    now_lr = cfg.TRAIN_LR * (1 - itr/(max_itr+1)) ** cfg.TRAIN_POWER
    optimizer.param_groups[0]['lr'] = now_lr*0.1
    optimizer.param_groups[1]['lr'] = 10*now_lr
    return now_lr

def adjust_lr_2(optimizer, itr, max_itr):
    now_lr = cfg.TRAIN_LR * (1 - itr/(max_itr+1)) ** cfg.TRAIN_POWER
    optimizer.param_groups[0]['lr'] = now_lr
    optimizer.param_groups[1]['lr'] = now_lr
    return now_lr

def get_params(model, key):
    for m in model.named_modules():
        if key == '1x':
            if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p

def get_params_init(model, key):
    for m in model.named_modules():
        if key == '1x':
            if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if 'backbone' not in m[0] and 'stage2' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p

if __name__ == '__main__':
    train_net()


