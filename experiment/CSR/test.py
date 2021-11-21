import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader

def test_net():
    dataset = generate_dataset(cfg.DATA_NAME, cfg, 'val')
    dataloader = DataLoader(dataset, 
                batch_size=cfg.TEST_BATCHES, 
                shuffle=False, 
                num_workers=cfg.DATA_WORKERS)
    
    net = generate_net(cfg)
    print('net initialize')
    if cfg.TEST_CKPT is None:
        raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
    

    print('Use %d GPU'%cfg.TEST_GPUS)
    device = torch.device('cuda')
    if cfg.TEST_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.to(device)

    print('start loading model %s'%cfg.TEST_CKPT)
    model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
    net.load_state_dict(model_dict)
    
    net.eval()    
    result_list = []
    count = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            name_batched = sample_batched['name']
            row_batched = sample_batched['row']
            col_batched = sample_batched['col']

            [batch, channel, height, width] = sample_batched['image'].size()
            multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(1)
            for rate in cfg.TEST_MULTISCALE:
                inputs_batched = sample_batched['image_%f'%rate]
                saliency_batched = sample_batched['saliency_%f'%rate]
                saliency_batched_resize = F.interpolate(saliency_batched.unsqueeze(1), scale_factor=(32/512), mode='nearest')
                predicts = net(inputs_batched, 'test', saliency_batched_resize.squeeze(1)).to(1)
                predicts_batched = predicts.clone()
                del predicts
                if cfg.TEST_FLIP:
                    inputs_batched_flip = torch.flip(inputs_batched,[3]) 
                    saliency_batched_resize_flip = torch.flip(saliency_batched_resize, [3])
                    predicts_flip = torch.flip(net(inputs_batched_flip,'test',saliency_batched_resize_flip.squeeze(1)),[3]).to(1)
                    predicts_batched_flip = predicts_flip.clone()
                    del predicts_flip
                    predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0
            
                predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
                multi_avg = multi_avg + predicts_batched
                del predicts_batched
            
            multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
            result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(batch):
                row = row_batched[i]
                col = col_batched[i]
            #    max_edge = max(row,col)
            #    rate = cfg.DATA_RESCALE / max_edge
            #    new_row = row*rate
            #    new_col = col*rate
            #    s_row = (cfg.DATA_RESCALE-new_row)//2
            #    s_col = (cfg.DATA_RESCALE-new_col)//2
     
            #    p = predicts_batched[i, s_row:s_row+new_row, s_col:s_col+new_col]
                p = result[i,:,:]
                p = cv2.resize(p, dsize=(col.item(),row.item()), interpolation=cv2.INTER_NEAREST)
                result_list.append({'predict':p, 'name':name_batched[i]})

            print('%d/%d'%(i_batch,len(dataloader)))
    dataset.save_result(result_list, cfg.MODEL_NAME)
    dataset.do_python_eval(cfg.MODEL_NAME)
    print('Test finished')

if __name__ == '__main__':
    test_net()


