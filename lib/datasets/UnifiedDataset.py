from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *

class UnifiedDataset(Dataset):
    def __init__(self, dataset_name, cfg, period, aug):
        self.dataset_name = dataset_name

        ### these roots must be modified

        #root directory for part segmentation
        # self.root_dir = os.path.join('/disk0/tanxin/SemanticSegmentation/SegData/pascal-part/')
        self.root_dir = os.path.join('/home/tanxin/SemanticSegmentation/SegData/pascal-part/')
        self.dataset_dir = os.path.join(self.root_dir)

        # save results directory
        self.rst_dir = os.path.join(self.root_dir,'results',dataset_name,'segmentation_rmi&ours_smoothmax_001_001_new')
        self.eval_dir = os.path.join(self.root_dir,'eval_result',dataset_name,'Segmentation')
        self.period = period

        # images directory
        self.img_dir = os.path.join(self.dataset_dir, 'img')
        self.ann_dir = os.path.join(self.dataset_dir, 'GT_part')
        self.seg_dir = os.path.join(self.dataset_dir, 'GT_part')

        ##edge and object semantic segmentation directory
        self.edge_dir = os.path.join(self.dataset_dir, 'edge')
        self.ins_dir = os.path.join(self.dataset_dir, 'Part_obj')

        # saliency director
        self.sal_dir = os.path.join(self.dataset_dir, 'pascal-part-sal')

        self.set_dir = os.path.join(self.dataset_dir )
        file_name = None



        #training / testing list txt follow the VOC rules
        if aug:
            file_name = self.set_dir+'/'+period+'aug.txt'
        else:
            file_name = self.set_dir+'/'+period+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values
        self.rescale = None
        self.centerlize = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.multiscale = None
        self.totensor = ToTensor()
        self.cfg = cfg
	
        if dataset_name == 'Unified':
            self.categories = [
                        'class1',    #1
                        'class2',      #2
                        'class3',         #3
                        'class4',         #4
                        'class5',       #5
                        'class6',          #6
                        'class7',          #6
                        'class8',  # 6
                        'class9',  # 6
                        'class10',  # 6
                        'class11',  # 1
                        'class12',  # 2
                        'class13',  # 3
                        'class14',  # 4
                        'class15',  # 5
                        'class16',  # 6
                        'class17',  # 6
                        'class18',  # 6
                        'class19',  # 6
                        'class20',  # 6
                        'class21',  # 1
                        'class22',  # 2
                        'class23',  # 3
                        'class24',  # 4
                        'class25',  # 5
                        'class26',  # 6
                        'class27',  # 6
                        'class28',  # 6
                        'class29',  # 6
                        'class30',  # 6
                        'class31',  # 1
                        'class32',  # 2
                        'class33',  # 3
                        'class34',  # 4
                        'class35',  # 5
                        'class36',  # 6
                        'class37',  # 6
                        'class38',  # 6
                        'class39',  # 6
                        'class40',  # 6
                        'class41',  # 1
                        'class42',  # 2
                        'class43',  # 3
                        'class44',  # 4
                        'class45',  # 5
                        'class46',  # 6
                        'class47',  # 6
                        'class48',  # 6
                        'class49',  # 6
                        'class50',  # 6
                        'class51',  # 1
                        'class52',  # 2
                        'class53',  # 3
                        'class54',  # 4
                        'class55',  # 5
                        'class56',  # 6
                        'class57',  # 6
                        ]   

            self.num_categories = len(self.categories)
            assert(self.num_categories+1 == self.cfg.MODEL_NUM_CLASSES)
            self.cmap = self.__colormap(len(self.categories)+1)


        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE,fix=False)
            #self.centerlize = Centerlize(cfg.DATA_RESCALE)
        if 'train' in self.period:        
        # if False:
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        else:
            self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)

        self.map_lst = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
		    		   6: 2, 7: 2,
		               8: 3, 9: 3, 10: 3, 11: 3,
		               12: 4,
		               13: 5, 14: 5,
		               15: 6, 16: 6, 17: 6,
		               18: 7, 19: 7, 20: 7, 21: 7, 22: 7,
		               23: 8, 24: 8, 25: 8, 26: 8,
		               27: 9,

		               28: 10, 29: 10, 30: 10, 31: 10,
		               32: 11,
		               33: 12, 34: 12, 35: 12, 36: 12,
		               37: 13, 38: 13, 39: 13, 40: 13,
		               41: 14, 42: 14,
		               43: 15, 44: 15, 45: 15, 46: 15, 47: 15, 48: 15,
		               49: 16, 50: 16,
		               51: 17, 52: 17, 53: 17,
		               54: 18,
		               55: 19,
		               56: 20, 57: 20,
           }

    def __len__(self):
        return len(self.name_list)

    def trainId2abstractId(self, label, reverse=False):
        label_copy = label.copy()
        for k, v in self.map_lst.items():
            label_copy[label == k] = v

        return label_copy

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file = self.img_dir + '/' + name + '.jpg'
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = np.array(io.imread(img_file),dtype=np.uint8)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        
        if 'train' in self.period:
        # if False:
            seg_file = self.seg_dir + '/' + name + '.png'
            segmentation = np.array(Image.open(seg_file))
            sample['segmentation'] = segmentation

            # instance = self.trainId2abstractId(segmentation)

            # blur = cv2.GaussianBlur(segmentation, (3, 3), 0)  
            # edge = cv2.Canny(blur, 0, 0)  

            ins_file = self.ins_dir + '/' + name + '.png'
            instance = np.array(Image.open(ins_file))
            sample['instance'] = instance

            edge_file = self.edge_dir + '/' + name + '.png'
            edge = np.array(Image.open(edge_file))
            sample['edge'] = edge

            sal_file = self.sal_dir + '/' + name + '_sal.png'
            saliency = np.array(Image.open(sal_file))
            sample['saliency'] = saliency

            if self.cfg.DATA_RANDOM_H>0 or self.cfg.DATA_RANDOM_S>0 or self.cfg.DATA_RANDOM_V>0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)
            if self.cfg.DATA_RESCALE > 0:
                #sample = self.centerlize(sample)
                sample = self.rescale(sample)
        else:
            seg_file = self.seg_dir + '/' + name + '.png'
            segmentation = np.array(Image.open(seg_file))
            sample['segmentation'] = segmentation
            
            sal_file = self.sal_dir + '/' + name + '_sal.png'
            saliency = np.array(Image.open(sal_file))
            sample['saliency'] = saliency
            
            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
            sample = self.multiscale(sample)

        if 'segmentation' in sample.keys():
            sample['mask'] = sample['segmentation'] < self.cfg.MODEL_NUM_CLASSES
            t = sample['segmentation']
            t[t >= self.cfg.MODEL_NUM_CLASSES] = 0
            sample['segmentation_onehot']=onehot(t,self.cfg.MODEL_NUM_CLASSES)

            ##
        if 'edge' in sample.keys():
            t = sample['edge']
            sample['edge_onehot']=onehot(t,2)



        ### for Pascal part classes= 21
        if 'instance' in sample.keys():
            pascal_class=21
            t = sample['instance']
            t[t >= pascal_class] = 0
            sample['instance_onehot'] = onehot(t, pascal_class)

        sample = self.totensor(sample)

        return sample



    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype = np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap
    
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
        cmap[:,:,2] = (m&4)<<5
        return cmap

    def colorify(self, label):
        import pickle
        f = open('/home/tanxin/SemanticSegmentation/SegData/pascal-part/colormap.pkl', 'rb')
        color_map = pickle.load(f)

        r, c = label.shape
        cmap = np.zeros((r, c, 3))
        for k, v in color_map.items():
            cmap[label == k] = v

        return cmap
    
    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            color_file_path = os.path.join(folder_path, '%s_color.png'%sample['name'])
            predict_color = self.colorify(sample['predict'])
            # p = self.__coco2voc(sample['predict'])
            cv2.imwrite(color_file_path, predict_color)

            file_path = os.path.join(folder_path, '%s.png'%sample['name'])
            cv2.imwrite(file_path, sample['predict'])

            print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1

    def do_matlab_eval(self, model_id):
        import subprocess
        path = os.path.join(self.root_dir, 'VOCcode')
        eval_filename = os.path.join(self.eval_dir,'%s_result.mat'%model_id)
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; VOCinit; '
        cmd += 'VOCevalseg(VOCopts,\'{:s}\');'.format(model_id)
        cmd += 'accuracies,avacc,conf,rawcounts = VOCevalseg(VOCopts,\'{:s}\'); '.format(model_id)
        cmd += 'save(\'{:s}\',\'accuracies\',\'avacc\',\'conf\',\'rawcounts\'); '.format(eval_filename)
        cmd += 'quit;"'

        print('start subprocess for matlab evaluation...')
        print(cmd)
        subprocess.call(cmd, shell=True)
    
    def do_python_eval(self, model_id):
        predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        gt_folder = self.seg_dir
        TP = []
        P = []
        T = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            TP.append(multiprocessing.Value('i', 0, lock=True))
            P.append(multiprocessing.Value('i', 0, lock=True))
            T.append(multiprocessing.Value('i', 0, lock=True))
        
        def compare(start,step,TP,P,T):
            for idx in range(start,len(self.name_list),step):
                print('%d/%d'%(idx,len(self.name_list)))
                name = self.name_list[idx]
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                gt_file = os.path.join(gt_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                gt = np.array(Image.open(gt_file))
                cal = gt<255
                mask = (predict==gt) * cal
          
                for i in range(self.cfg.MODEL_NUM_CLASSES):
                    P[i].acquire()
                    P[i].value += np.sum((predict==i)*cal)
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt==i)*cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt==i)*mask)
                    TP[i].release()
        p_list = []
        for i in range(8):
            p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        IoU = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                print('%11s:%7.3f%%'%('backbound',IoU[i]*100),end='\t')
            else:
                if i%2 != 1:
                    print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100),end='\t')
                else:
                    print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100))
                    
        miou = np.mean(np.array(IoU))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))    

    #def do_python_eval(self, model_id):
    #    predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
    #    gt_folder = self.seg_dir
    #    TP = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
    #    P = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
    #    T = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
    #    for idx in range(len(self.name_list)):
    #        print('%d/%d'%(idx,len(self.name_list)))
    #        name = self.name_list[idx]
    #        predict_file = os.path.join(predict_folder,'%s.png'%name)
    #        gt_file = os.path.join(gt_folder,'%s.png'%name)
    #        predict = cv2.imread(predict_file)
    #        gt = cv2.imread(gt_file)
    #        cal = gt<255
    #        mask = (predict==gt) & cal
    #        for i in range(self.cfg.MODEL_NUM_CLASSES):
    #            P[i] += np.sum((predict==i)*cal)
    #            T[i] += np.sum((gt==i)*cal)
    #            TP[i] += np.sum((gt==i)*mask)
    #    TP = TP.astype(np.float64)
    #    T = T.astype(np.float64)
    #    P = P.astype(np.float64)
    #    IoU = TP/(T+P-TP)
    #    for i in range(self.cfg.MODEL_NUM_CLASSES):
    #        if i == 0:
    #            print('%15s:%7.3f%%'%('backbound',IoU[i]*100))
    #        else:
    #            print('%15s:%7.3f%%'%(self.categories[i-1],IoU[i]*100))
    #    miou = np.mean(IoU)
    #    print('==================================')
    #    print('%15s:%7.3f%%'%('mIoU',miou*100))

   
