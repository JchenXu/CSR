## How to run:

#### Dataset

1. Download PASCAL-Part dataset [https://cs.stanford.edu/~roozbeh/pascal-parts/pascal-parts.html]

2. **[Update]** Additional PASCAL-Part-Saliency dataset is available at [https://dmcv.sjtu.edu.cn/data/project/pascal-part-sal.tar.gz] and Part Object Mask is avalable at [https://dmcv.sjtu.edu.cn/data/project/Part_obj.tar.gz]

2. Download the multi-class annotations from [http://cvteam.net/projects/2019/multiclass-part.html]

3. Modify the configurations in /experiments/CSR/config.py. (The initial performance is about 59.45, then the reported performance can be achieved by fine-tuning.)

4. Modify the dataset path in /lib/datasets

   (There might be different versions of this dataset, we follow the annotations of CVPR17 to make fair comparisons.)

   ##### PASCAL-Part-multi-class Dataset: http://cvteam.net/projects/2019/figs/Affined.zip


### For Test

1. Download the [pretrained model](https://drive.google.com/file/d/1MMWoexMgtXH1aqzOR734DE6p3JQ3pkQy/view?usp=sharing) and modify the path in /experiments/config.py

2. RUN /experiments/CSR/test.py

3. **[Update]** The color map for visualization is available [here](https://dmcv.sjtu.edu.cn/data/project/colormap.pkl)

4. (Additionally) If customize data, you need to generate a filelist following the VOC format and modify the dataset path.

### For Training 

If training from scratch, simply run. If not, customize the dir in /experiments/CSR config.py.

 (A training demo code is provided in train.py)

1. (Additionally) download the ImageNet pretrained model:

   model_urls = {

     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',

     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

   }

2. Prerequisites: generate semantic part boundaries and semantic object labels. (will be provided soon)

3. RUN /experiments/CSR/train.py for 100 epochs. (Achieve 59.45 mIoU)

4. Fine-tune the model using learning rate=0.003 for another 40 epochs.  (Achieve 60.70 mIoU)

   
### Acknowledgement

The code is based on the below project:

##### Yifan Zhao, Jia Li, Yu Zhang, and Yonghong Tian. Multi-class Part Parsing with Joint Boundary-Semantic Awareness in ICCV 2019.



### Citation
```
@inproceedings{tan2021confident,
  title={Confident Semantic Ranking Loss for Part Parsing},
  author={Tan, Xin and Xu, Jiachen and Ye, Zhou and Hao, Jinkun and Ma, Lizhuang},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
