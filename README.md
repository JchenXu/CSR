## How to run:

#### Dataset

1. Download PASCAL-Part dataset [https://cs.stanford.edu/~roozbeh/pascal-parts/pascal-parts.html]

2. Download the multi-class annotations from [http://cvteam.net/projects/2019/multiclass-part.html]

3. Modify the configurations in /experiments/BSANet/config.py. (The parameters are not carefully tuned, simply change them may get higher performance.)

4. Modify the dataset path in /lib/datasets

   ##### PASCAL-Person-Part Dataset:  http://cvteam.net/projects/2019/figs/pascal_person_part.zip

   (There might be different versions of this dataset, we follow the annotations of CVPR17 to make fair comparisons.)

   ##### PASCAL-Part-multi-class Dataset: http://cvteam.net/projects/2019/figs/Affined.zip


### For Test

1. Download the pretrained model and modify the path in /experiments/config.py

2. RUN /experiments/BSANet/test.py

3. (Additionally) If customize data, you need to generate a filelist following the VOC format and modify the dataset path.

### For Training 

If training from scratch, simply run. If not, customize the dir in /experiments/BSANet config.py.

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

3. RUN /experiments/BSANet/train.py for 100 epochs. (Achieve 58.09 mIoU)

4. Fine-tune the model using learning rate=0.003 for another 60 epochs.  (Achieve 58.43 mIoU)

   

   


### Models & Results
ResNet-101 Model for 58 classes：https://drive.google.com/open?id=1SIHjcCq70ToLfve7T7tEfnpGtEKtQGTK

The code is an modified version of original code, which follows the same settings and achieves slightly better performance. But you can still compare your results following the paper.

See **Results.md** for details. (58.43 mIoU using ResNet-101 backbone on 58 part classes, and 68.477 mIoU using ResNet-152 backbone on PASCAL-Person).

This code is tested on python 3.6 and PyTorch=0.4.0 and modified from deeplabv3 re-implementation [http://github.com/YudeWang/deeplabv3plus-pytorch].
The performance will slightly drop down under different environment (bugs requires to be fixed).

For problems and bugs, we will pursue better code in the next few months.



### To do List

1. More backbones, including Xception.

2. Light-weight models, including ResNet-18.




### Citation


The code is the beta version to provide basic implementation of the original paper:

##### Yifan Zhao, Jia Li, Yu Zhang, and Yonghong Tian. Multi-class Part Parsing with Joint Boundary-Semantic Awareness in ICCV 2019.