import numpy as np
import torch
from torch.utils.data import Dataset
#import scipy.misc as m
import random
from PIL import Image
import cv2
import imageio

from sklearn.model_selection import train_test_split

total = []
    
import os

def print_file_paths(directory_path, label):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            total.append([file_path, label ])

# Example usage:
directory_path = '/drive0-storage/VPE/aug_db/train/00000'
print_file_paths(directory_path, 0 )

directory_path = '/drive0-storage/VPE/aug_db/train/00001'
print_file_paths(directory_path, 1)
    
directory_path = '/drive0-storage/VPE/aug_db/train/00002'
print_file_paths(directory_path, 2)

directory_path = '/drive0-storage/VPE/db/GTSRB/00000'
print_file_paths(directory_path, 0 )

directory_path = '/drive0-storage/VPE/db/GTSRB/00002'
print_file_paths(directory_path, 1)
    
directory_path = '/drive0-storage/VPE/db/GTSRB/00002'
print_file_paths(directory_path, 2)


# Splitting the data into training and testing sets
train_data, test_data = train_test_split(total, test_size=0.1, random_state=42)
print(type(train_data))
print(type(train_data[0]))
print(train_data[0])
# Printing the results
print("Training data:", len(train_data))

print("Testing data:", len(test_data))
    
    

class gtsrbLoader(Dataset):

  def __init__(self, root, exp, split='train', is_transform=False, img_size=None, augmentations=None, prototype_sampling_rate=0.005):
    super().__init__()

    if split == 'train':
        self.proto_rate = prototype_sampling_rate
    else:
        self.proto_rate = 0.0
        
    self.inputs = []
    self.targets = []
    self.class_names = []
    self.split = split
    self.img_size = img_size
    self.is_transform = is_transform
    self.augmentations = augmentations
    self.mean = np.array([125.00, 125.00, 125.00]) # average intensity

    self.root = root + 'GTSRB/'
    exp  = exp + '/exp_gtsrb/'
    self.dataPath = root + exp + self.split + '_impaths.txt'
    self.labelPath = root + exp + self.split + '_imclasses.txt'

    # self.tr_class = torch.LongTensor([1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,25,26,31,33,35,38])
    # self.te_class = torch.LongTensor([0,6,16,19,20,21,22,23,24,27,28,29,30,32,34,36,37,39,40,41,42])
    self.tr_class = torch.LongTensor([0, 1, 2, 3])
    self.te_class = torch.LongTensor([0, 1, 2, 3])

    # f_data = open(self.dataPath,'r')
    # f_label = open(self.labelPath,'r')
    # data_lines = f_data.readlines()
    # label_lines = f_label.readlines()
    

    data_lines = []
    label_lines = []
    
    
    if split == 'train':
        for pair in train_data:
            data_lines.append(pair[0])
            label_lines.append(pair[1])

    
    if split == 'test':
        for pair in test_data:
            data_lines.append(pair[0])
            label_lines.append(pair[1])
    print(data_lines[0])
    print(label_lines[0])
          
     
    

    # for i in range(len(data_lines)):
    #   self.inputs.append(root+data_lines[i][0:-1])
      #self.targets.append(int(label_lines[i].split()[0])) # label: [road class, wet/dry, video index]
    
    self.targets = label_lines
    print((self.targets[0]))
    self.inputs = data_lines
    print(self.inputs[0])
    
    
    
    # classnamesPath = root + exp + '/classnames_.txt'
    # f_classnames = open(classnamesPath, 'r')
    # data_lines = f_classnames.readlines()
    # for i in range(len(data_lines)):
    #     self.class_names.append(data_lines[i][0:-1])
    # if split == 'train':
    self.class_names = ['prohibit_20', 'prohibit_30', 'prohibit_50', 'prohibit_60'] #prohibit_20 prohibit_30 prohibit_50 prohibit_60
#     if split == 'val':
#         self.class_names = ['prohibit_30']
#     if split == 'test':
#         self.class_names = ['prohibit_50']
        

    self.n_classes = 4
    # print(len(self.class_names))
    assert(self.n_classes == len(self.class_names))

    print('GTSRB %d classes'%(len(self.class_names)))
    print('Load GTSRB %s: %d samples'%(split, len(self.targets)))


  def __len__(self):
    return len(self.inputs)


  def __getitem__(self, index):
    img_path = self.inputs[index]
    gt = self.targets[index]
    gt = torch.ones(1).type(torch.LongTensor)*gt

    # Load images and templates. perform augmentations
    img = imageio.imread(img_path)
    img = np.array(img, dtype=np.uint8)
    template = imageio.imread('/drive0-storage/VPE/db/' + 'template_ordered/%02d.jpg'%(gt+1))
    template = np.array(template, dtype=np.uint8)

    if random.random() < self.proto_rate:
        img = np.copy(template)

    if self.augmentations is not None:
        img, template = self.augmentations(img, template)

    if self.is_transform:
        img = self.transform(img)
        template = self.transform(template)

    return img, gt, template
    
  def transform(self, img):
    # print(type(img))
    img = img.astype(np.float64)
    #print(img.shape)
    img -= self.mean
    #print(img.shape)
    if self.img_size is not None:
      #img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    
    return img


  def load_template(self, target, augmentations=None):

    # if augmentation is not specified, use self.augmentations. Unless use input augmentation option.
    if augmentations is None:
        augmentations = self.augmentations
    img_paths = []
    
    for id in target:
        img_paths.append('/drive0-storage/VPE/db/' + '/template_ordered/%02d.jpg'%(id+1))

    target_img = []
    for img_path in img_paths:
        img = imageio.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if augmentations is not None:
            img, _ = augmentations(img, img)
        if self.transform:
            img = self.transform(img)

        target_img.append(img)

    return torch.stack(target_img, dim=0)

from argparse import ArgumentParser
import os
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
import math
import os
from pathlib import Path

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

import scipy
print(scipy.__version__)

import torch
print(torch.__version__)

import matplotlib
print(matplotlib.__version__)


import numpy
print(numpy.__version__)

# import Pillow
# print(Pillow.__version__)

import torchvision
print(torchvision.__version__)

'''
torchvision==0.2.1 -
torch==1.0.0 - 
matplotlib==2.2.2 - 
scipy==1.1.0 - 
numpy==1.15.4 - 
Pillow==6.0.0

'''
# from loader import get_loader, get_data_path
from models import get_model
from augmentations import *



# Setup
parser = ArgumentParser(description='Variational Prototyping Encoder (VPE)')
parser.add_argument('--seed',       type=int,   default=42,             help='Random seed')
parser.add_argument('--arch',       type=str,   default='vaeIdsiaStn',  help='network type: vaeIdsia, vaeIdsiaStn')
parser.add_argument('--dataset',    type=str,   default='gtsrb', help='dataset to use [gtsrb, belga2flickr, belga2toplogo]') # for gtsrb2TT100K scenario, use main_train_test.py
parser.add_argument('--exp',        type=str,   default='exp_list',     help='training scenario')
parser.add_argument('--resume',     type=str,   default=None,           help='Resume training from previously saved model')

parser.add_argument('--epochs',     type=int,   default=1,           help='Training epochs')
parser.add_argument('--lr',         type=float, default=1e-4,           help='Learning rate')
parser.add_argument('--batch_size', type=int,   default=2,            help='Batch size')

parser.add_argument('--img_cols',   type=int,   default=64,             help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,             help='resized image height')
parser.add_argument('--workers',    type=int,   default=0,              help='Data loader workers')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
plt.switch_backend('agg')  # Allow plotting when running remotely

save_epoch = 10 # save log images per save_epoch

# 02 rotation + flip augmentation option
# Setup Augmentations
data_aug_tr= Compose([Scale(args.img_cols), # resize longer side of an image to the defined size
                      CenterPadding([args.img_rows, args.img_cols]), # zero pad remaining regions
                      RandomHorizontallyFlip(), # random horizontal flip
                      RandomRotate(180)])  # ramdom rotation

data_aug_te= Compose([Scale(args.img_cols), 
                     CenterPadding([args.img_rows, args.img_cols])])

result_path = 'results_trial_' + args.dataset
if not os.path.exists(result_path):
  os.makedirs(result_path)
outimg_path =  "./img_log+trial" + args.dataset
if not os.path.exists(outimg_path):
  os.makedirs(outimg_path)

f_loss = open(os.path.join(result_path, "log_loss.txt"),'w')
f_loss.write('Network type: %s\n'%args.arch)
f_loss.write('Learning rate: %05f\n'%args.lr)
f_loss.write('batch-size: %s\n'%args.batch_size)
f_loss.write('img_cols: %s\n'%args.img_cols)
f_loss.write('Augmentation type: flip, centercrop\n\n')
f_loss.close()

f_iou = open(os.path.join(result_path, "log_acc.txt"),'w')
f_iou.close()

f_iou = open(os.path.join(result_path, "log_val_acc.txt"),'w')
f_iou.close()

# set up GPU

# we could do os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Data 
# data_loader = get_loader(args.dataset)
# data_path = get_data_path(args.dataset)



tr_loader = gtsrbLoader('', args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)
te_loader = gtsrbLoader('', args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)
# val_loader = gtsrbLoader('', args.exp, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)

trainloader = DataLoader(tr_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
testloader = DataLoader(te_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
# valloader = DataLoader(val_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)


# define model or load model
net = get_model(args.arch, n_classes=None)
net.cuda()

if args.resume is not None:
  pre_params = torch.load(args.resume)
  net.init_params(pre_params)


reconstruction_function = nn.BCELoss()
reconstruction_function.reduction = 'sum'
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

# Construct optimiser
optimizer = optim.Adam(net.parameters(), lr=args.lr) # 1e-4

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
# num_val = len(val_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)
# batch_iter_val = math.ceil(num_val/args.batch_size)

def train(e):
  print('in training')
  n_classes = tr_loader.n_classes
  n_classes_te = te_loader.n_classes
  # n_classes_val = val_loader.n_classes
  print('start train epoch: %d'%e)
  net.train()
  
  for i, (input, target, template) in enumerate(trainloader):
    print(i)

    optimizer.zero_grad()
    target = torch.squeeze(target)
    input, template = input.cuda(), template.cuda()
    print(input.size())
    print(template.size())
    #input, template = input, template

    recon, mu, logvar, input_stn = net(input)
    loss = loss_function(recon, template, mu, logvar) # reconstruction loss
    print('Epoch:%d  Batch:%d/%d  loss:%08f'%(e, i, batch_iter, loss/input.numel()))
    
    f_loss = open(os.path.join(result_path, "log_loss.txt"),'a')
    f_loss.write('Epoch:%d  Batch:%d/%d  loss:%08f\n'%(e, i, batch_iter, loss/input.numel()))
    f_loss.close()
    
    loss.backward()
    optimizer.step()
    break

    if i < 1 and (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

  if e%save_epoch == 0:
    class_target = torch.LongTensor(list(range(n_classes)))
    class_template = tr_loader.load_template(class_target)
    class_template = class_template.cuda()
    #class_template = class_template
    with torch.no_grad():
      class_recon, class_mu, class_logvar, _ = net(class_template)
    
    torchvision.utils.save_image(class_template, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)
    torchvision.utils.save_image(class_recon, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)
  
def score_NN(pred, class_feature, label, n_classes):
  #label = torch.unsqueeze(label, dim = 1)
  # print('label',label)
  
    #mu, class_mu, target, n_classes

  sample_correct = torch.zeros(n_classes)
  # sample_wrong = torch.zeros(n_classes)
    
  sample_all = torch.zeros(n_classes)
  sample_rank = torch.zeros(n_classes, n_classes) # rank per class
  sample_distance = torch.ones(pred.shape[0], n_classes)*math.inf

  pred = pred.cpu() # batch x latent size
  class_feature = class_feature.cpu() # n_classes x latent size
  
  for i in range(n_classes):
    cls_feat = class_feature[i,:]
    cls_mat = cls_feat.repeat(pred.shape[0],1)
    # euclidean distance
    sample_distance[:,i] = torch.norm(pred - cls_mat,p=2, dim=1)
  
  sample_distance = sample_distance.cpu().numpy()
  indices = np.argsort(sample_distance, axis=1) # sort ascending order\
 
  result = indices[:,0].tolist()
    
  preds = result
  true = label.tolist()
  print(preds, true)
    
  label = label.numpy()
  for i in range(indices.shape[0]):
    #print(np.where(indices[i,:] == label[i]))
    rank = np.where(indices[i,:] == label[i])[0][0] # find rank
    print('rank', rank)
    sample_rank[label[i]][rank:] += 1 # update rank 
    print('sample_rank', sample_rank)
    sample_all[label[i]] += 1 # count samples per class
    print('sample_all', sample_all)
    if rank == 0:
      sample_correct[label[i]] += 1 # count rank 1 (correct classification)
      print('sample_correct', sample_correct)
    # else:
    #     sample_wrong[label[i]] += 1
    #     print('wrong')
    #     #print('\n')
    
  
  

  return sample_correct, sample_all, sample_rank, preds, true


mean_scores = []
mean_rank = []
def test(e, best_acc, val_trigger):
  predictions = []
  true_labels = []
  n_classes = te_loader.n_classes
  print('start test epoch: %d'%e)
  net.eval()
  accum_all = torch.zeros(n_classes)
  rank_all = torch.zeros(n_classes, n_classes) # rank per class
  accum_class = torch.zeros(n_classes)
  accum_class_wrong = torch.zeros(n_classes)

  # get template latent z
  class_target = torch.LongTensor(list(range(n_classes)))
  class_template = te_loader.load_template(class_target)
  class_template = class_template.cuda()
  #class_template = class_template
  print(class_template.size())
  
  with torch.no_grad():
    class_recon, class_mu, class_logvar, _ = net(class_template)
  c = 0
  for i, (input, target, template) in enumerate(testloader):
    c = c + 1

    target = torch.squeeze(target)
    input, template = input.cuda(), template.cuda()
    #input, template = input, template
    # print(input.size())
    # print(template.size())
    with torch.no_grad():
      recon, mu, logvar, input_stn  = net(input)
    preds = []
    true = []
    sample_correct, sample_all, sample_rank, preds, true = score_NN(mu, class_mu, target, n_classes)
    accum_class += sample_correct
    # accum_class_wrong += sample_wrong
    accum_all += sample_all
    rank_all = rank_all + sample_rank # [class_id, topN]
    for p in preds:
        predictions.append(p)
    for t in true:
        true_labels.append(t)
    
    print('accum_class', accum_class)
    # print('accum_class _wrong', accum_class_wrong)
    print('accum_all', accum_all)
    print('rank_all', rank_all)
    print(predictions)
    print(true_labels)
    
    from sklearn.metrics import classification_report
    print(classification_report (true_labels, predictions, labels = [0,1,2]))
    
    
    
    if c == 50:
        break
    #break
 
    
    print('Epoch:%d  Batch:%d/%d  processing...'%(e, i, batch_iter_test))

    if i < 1 and (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

  if e%save_epoch == 0:
    torchvision.utils.save_image(class_template, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
    torchvision.utils.save_image(class_recon, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)  

  acc_all = accum_class.sum() / accum_all.sum() 
  acc_cls = torch.div(accum_class, accum_all)
  
  rank_sample_avg = rank_all.sum(0) / accum_all.sum() # [class_id, topN]
  rank_cls = torch.div(rank_all, torch.transpose(accum_all.expand_as(rank_all),0,1))
  rank_cls_avg = torch.mean(rank_cls,dim=0)
    
  acc_per_class = torch.gather(acc_cls, 0, te_loader.tr_class)
    
 

  # write result part
  acc_trcls = torch.gather(acc_cls, 0, te_loader.tr_class)
  acc_tecls =torch.gather(acc_cls, 0, te_loader.te_class)


  print('========epoch(%d)========='%e)
  print('Seen Classes')
  for i, class_acc in enumerate(acc_trcls):
    print('cls:%d  acc:%02f'%(te_loader.tr_class[i], class_acc))
  print('Unseen Classes')
  for i, class_acc in enumerate(acc_tecls):
    print('cls:%d  acc:%02f'%(te_loader.te_class[i], class_acc))
  print('====================================')
  print('acc_avg:%02f'%acc_all)
  print('acc_cls:%02f'%acc_cls.mean())
  print('acc_trcls:%02f'%acc_trcls.mean())
  print('acc_tecls:%02f'%acc_tecls.mean())
  print('rank sample avg: %02f'%rank_sample_avg.mean())
  print('rank cls avg: %02f'%rank_cls_avg.mean())
  print('====================================')

  f_iou = open(os.path.join(result_path, "log_acc.txt"),'a')
  try:
      f_iou.write('epoch(%d), acc_cls: %04f  acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f  top5: %04f\n'%(e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2], rank_sample_avg[4]))
  except:
     f_iou.write('epoch(%d), acc_cls: %04f  acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f'%(e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2]))
    
  f_iou.close()

#   if val_trigger: # when validation performance higher than prev val_best performance.
#     # in the paper, we report best accuracy triggered by validation performance.
#     f_iou_class = open(os.path.join(result_path, "best_iou_triggeredByVal.txt"),'w')
#     f_rank = open(os.path.join(result_path, "best_rank_triggeredByVal.txt"),'w')

#     f_iou_class.write('Best score epoch:  %d\n'%e)
#     f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank mean: %.4f \n'%(acc_cls.mean(), acc_all, rank_all.mean()))
#     f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n'%(acc_trcls.mean(), acc_tecls.mean()))
#     try:
#         f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2], rank_sample_avg[4]))
#     except:
#         f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2]))

#     f_iou_class.write('\nSeen classes\n')
#     for i, class_acc in enumerate(acc_trcls):
#       f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.tr_class[i], class_acc))
#     f_iou_class.write('\nUnseen classes\n')
#     for i, class_acc in enumerate(acc_tecls):
#       f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.te_class[i], class_acc))
#     f_iou_class.close()
    
#     for i, rank_acc in enumerate(rank_sample_avg):
#       f_rank.write('rank sample %d: %.4f\n'%(i+1, rank_acc))
#     f_rank.write('\n')
#     for i, rank_acc in enumerate(rank_cls_avg):
#       f_rank.write('rank cls %d: %.4f\n'%(i+1, rank_acc))
#     f_rank.close()

#   if best_acc < acc_tecls.mean(): # update best score
#     # best accuracy during the training stage. Just for reference.
#     f_iou_class = open(os.path.join(result_path, "best_iou.txt"),'w')
#     f_rank = open(os.path.join(result_path, "best_rank.txt"),'w')
#     torch.save(net.state_dict(), os.path.join('%s_testBest_net_aug.pth'%args.dataset)) # if best_acc == 0, then valBest model is saved

#     best_acc = acc_tecls.mean()
#     f_iou_class.write('Best score epoch:  %d\n'%e)
#     f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank mean: %.4f \n'%(acc_cls.mean(), acc_all, rank_all.mean()))
#     f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n'%(acc_trcls.mean(), acc_tecls.mean()))
#     try:
#         f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2], rank_sample_avg[4]))
#     except:
#         f_iou_class.write('top3: %.4f'%(rank_sample_avg[2]))

#     f_iou_class.write('\nSeen classes\n')
#     for i, class_acc in enumerate(acc_trcls):
#       f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.tr_class[i], class_acc))
#     f_iou_class.write('\nUnseen classes\n')
#     for i, class_acc in enumerate(acc_tecls):
#       f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.te_class[i], class_acc))
#     f_iou_class.close()
    
#     for i, rank_acc in enumerate(rank_sample_avg):
#       f_rank.write('rank sample %d: %.4f\n'%(i+1, rank_acc))
#     f_rank.write('\n')
#     for i, rank_acc in enumerate(rank_cls_avg):
#       f_rank.write('rank cls %d: %.4f\n'%(i+1, rank_acc))
    # f_rank.close()

  # Save weights and scores
  # if e % 100 == 0:
    # torch.save(net.state_dict(), os.path.join('flickr2belga_latest_net.pth'))

  ###### Plot scores
  # mean_scores.append(acc_tecls.mean())
  # es = list(range(len(mean_scores)))
  # plt.plot(es, mean_scores, 'b-')
  # plt.xlabel('Epoch')
  # plt.ylabel('Unseen mean IoU')
  # plt.savefig(os.path.join(result_path, 'unseen_ious.png'))
  # plt.close()

  ####### plot rank
  # mean_rank.append(rank_all.mean())
  # rank_es = list(range(len(mean_rank)))
  # plt.plot(rank_es, mean_rank, 'b-')
  # plt.xlabel('Epoch')
  # plt.ylabel('Mean rank')
  # plt.savefig(os.path.join(result_path, 'rank.png'))
  # plt.close()

  return best_acc

# mean_scores_val = []
# mean_rank_val = []

# def validation(e, best_acc):
#   n_classes = val_loader.n_classes 
  
#   print('start validation epoch: %d'%e)
#   net.eval()

#   accum_all = torch.zeros(n_classes)
#   rank_all = torch.zeros(n_classes, n_classes)
#   accum_class = torch.zeros(n_classes)

#   # get template latent z
#   class_target = torch.LongTensor(list(range(n_classes)))
#   class_template = val_loader.load_template(class_target)
#   class_template = class_template.cuda()
#   #class_template = class_template
#   with torch.no_grad():
#     class_recon, class_mu, class_logvar, _ = net(class_template)
  

#   for i, (input, target, template) in enumerate(valloader):

#     target = torch.squeeze(target)
#     input, template = input.cuda(), template.cuda()
#     #input, template = input, template
#     with torch.no_grad():
#       recon, mu, logvar, input_stn = net(input)
   
# #     print(mu.size())
# #     print(class_mu.size())
# #     print(target.size())
# #     print(n_classes)
    
#     sample_correct, sample_all, sample_rank = score_NN(mu, class_mu, target, n_classes)
#     accum_class += sample_correct
#     accum_all += sample_all
#     rank_all = rank_all + sample_rank
    
#     print('Epoch:%d  Batch:%d/%d  processing...'%(e, i, batch_iter_val))
    
#     if i < 1 and (e%save_epoch == 0) :
#       out_folder =  "%s/Epoch_%d_val"%(outimg_path, e)
#       out_root = Path(out_folder)
#       if not out_root.is_dir():
#         os.mkdir(out_root)
#       torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
#       torchvision.utils.save_image(input_stn, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
#       torchvision.utils.save_image(recon, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
#       torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

#   if e%save_epoch == 0:
#     torchvision.utils.save_image(class_template, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
#     torchvision.utils.save_image(class_recon, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)  

#   acc_all = accum_class.sum() / accum_all.sum() 
#   acc_cls = torch.div(accum_class, accum_all)
  
#   rank_sample_avg = rank_all.sum(0) / accum_all.sum() # [class_id, topN]
#   rank_cls = torch.div(rank_all, torch.transpose(accum_all.expand_as(rank_all),0,1))
#   rank_cls_avg = torch.mean(rank_cls,dim=0)
  
#   # write result part
#   acc_trcls = torch.gather(acc_cls, 0, val_loader.tr_class)
#   acc_tecls =torch.gather(acc_cls, 0, val_loader.te_class)

#   print('========epoch(%d)========='%e)
#   print('Seen Classes')
#   for i, class_acc in enumerate(acc_trcls):
#     print('cls:%d  acc:%02f'%(val_loader.tr_class[i], class_acc))
#   print('Unseen Classes')
#   for i, class_acc in enumerate(acc_tecls):
#     print('cls:%d  acc:%02f'%(val_loader.te_class[i], class_acc))
#   print('====================================')
#   print('acc_avg:%02f'%acc_all)
#   print('acc_cls:%02f'%acc_cls.mean())
#   print('acc_trcls:%02f'%acc_trcls.mean())
#   print('acc_tecls:%02f'%acc_tecls.mean())
#   print('rank sample avg:%02f'%rank_sample_avg.mean())
#   print('rank cls avg:%02f'%rank_cls_avg.mean())
#   print('====================================')

#   f_iou = open(os.path.join(result_path, "log_val_acc.txt"),'a')
#   # print(rank_sample_avg)
#   try:
#       f_iou.write('epoch(%d), acc_cls: %04f  acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f  top5: %04f\n'%(e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2], rank_sample_avg[4]))
#   except:
#        f_iou.write('epoch(%d), acc_cls: %04f  acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f'%(e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2]))
#   f_iou.close()

#   if best_acc < acc_tecls.mean(): # update best score
#     best_acc = acc_tecls.mean()
#     f_iou_class = open(os.path.join(result_path, "best_iou_val.txt"),'w')
#     f_iou_class.write('Best score epoch:  %d\n'%e)
#     f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank sample mean: %.4f  rank cls mean: %.4f\n'%(acc_cls.mean(), acc_all, rank_sample_avg.mean(), rank_cls_avg.mean()))
#     f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n'%(acc_trcls.mean(), acc_tecls.mean()))
#     f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2], rank_sample_avg[4]))

#     f_iou_class.write('\nSeen classes\n')
#     for i, class_acc in enumerate(acc_trcls):
#       f_iou_class.write('cls:%d  acc:%02f\n'%(val_loader.tr_class[i], class_acc))
#     f_iou_class.write('\nUnseen classes\n')
#     for i, class_acc in enumerate(acc_tecls):
#       f_iou_class.write('cls:%d  acc:%02f\n'%(val_loader.te_class[i], class_acc))
#     f_iou_class.close()
#     torch.save(net.state_dict(), os.path.join('%s_valBest_net_aug.pth'%args.dataset))

#     f_rank = open(os.path.join(result_path, "best_rank_val.txt"),'w')
#     for i, rank_acc in enumerate(rank_sample_avg):
#       f_rank.write('rank sample %d: %.4f\n'%(i+1, rank_acc))
#     f_rank.write('\n')
#     for i, rank_acc in enumerate(rank_cls_avg):
#       f_rank.write('rank cls %d: %.4f\n'%(i+1, rank_acc))
#     f_rank.close()

#   # Plot scores
#   mean_scores_val.append(acc_tecls.mean())
#   es = list(range(len(mean_scores_val)))
#   plt.plot(es, mean_scores_val, 'b-')
#   plt.xlabel('Epoch')
#   plt.ylabel('Mean IoU')
#   plt.savefig(os.path.join(result_path, 'unseen_ious_val.png'))
#   plt.close()

#   # # plot rank
#   # mean_rank_val.append(rank_all.mean())
#   # rank_es = list(range(len(mean_rank_val)))
#   # plt.plot(rank_es, mean_rank_val, 'b-')
#   # plt.xlabel('Epoch')
#   # plt.ylabel('Mean rank')
#   # plt.savefig(os.path.join(result_path, 'rank_val.png'))
#   # plt.close()

#   return best_acc

if __name__ == "__main__":
  out_root = Path(outimg_path)
  if not out_root.is_dir():
    os.mkdir(out_root)

  best_acc = 0
  best_acc_val = 0

  for e in range(1, args.epochs + 1):
    val_trigger = False
    #train(e)
    # temp_acc_val = validation(e, best_acc_val)
    # if temp_acc_val > best_acc_val:
    #   best_acc_val = temp_acc_val
    #   val_trigger = True # force test function to save log when validation performance is updated
    best_acc = test(e, best_acc, val_trigger)
    
