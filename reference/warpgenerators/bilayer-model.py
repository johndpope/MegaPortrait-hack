import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys
import pathlib
import numpy as np
import cv2
import importlib
import ssl

from datasets import utils as ds_utils
from runners import utils as rn_utils
from external.Graphonomy import wrapper
import face_alignment



class InferenceWrapper(nn.Module):
    @staticmethod
    def get_args(args_dict):
        # Read and parse args of the module being loaded
        args_path = pathlib.Path(args_dict['project_dir']) / 'runs' / args_dict['experiment_name'] / 'args.txt'

        parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add = parser.add_argument

        with open(args_path, 'rt') as args_file:
            lines = args_file.readlines()
            for line in lines:
                k, v, v_type = rn_utils.parse_args_line(line)
                parser.add('--%s' % k, type=v_type, default=v)

        args, _ = parser.parse_known_args()

        # Add args from args_dict that overwrite the default ones
        for k, v in args_dict.items():
            setattr(args, k, v)

        args.world_size = args.num_gpus

        return args

    def __init__(self, args_dict):
        super(InferenceWrapper, self).__init__()
        # Get a config for the network
        self.args = self.get_args(args_dict)
        self.to_tensor = transforms.ToTensor()

        # Load the model
        self.runner = importlib.import_module(f'runners.{self.args.runner_name}').RunnerWrapper(self.args, training=False)
        self.runner.eval()

        # Load pretrained weights
        checkpoints_dir = pathlib.Path(self.args.project_dir) / 'runs' / self.args.experiment_name / 'checkpoints'

        # Load pre-trained weights
        init_networks = rn_utils.parse_str_to_list(self.args.init_networks) if self.args.init_networks else {}
        networks_to_train = self.runner.nets_names_to_train

        if self.args.init_which_epoch != 'none' and self.args.init_experiment_dir:
            for net_name in init_networks:
                self.runner.nets[net_name].load_state_dict(torch.load(
                    pathlib.Path(self.args.init_experiment_dir) 
                        / 'checkpoints' 
                        / f'{self.args.init_which_epoch}_{net_name}.pth', 
                    map_location='cpu'))

        for net_name in networks_to_train:
            if net_name not in init_networks and net_name in self.runner.nets.keys():
                self.runner.nets[net_name].load_state_dict(torch.load(
                    checkpoints_dir 
                        / f'{self.args.which_epoch}_{net_name}.pth', 
                    map_location='cpu'))
        
        # Remove spectral norm to improve the performance
        self.runner.apply(rn_utils.remove_spectral_norm)

        # Stickman/facemasks drawer
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

        self.net_seg = wrapper.SegmentationWrapper(self.args)

        if self.args.num_gpus > 0:
            self.cuda()

    def change_args(self, args_dict):
        self.args = self.get_args(args_dict)

    def preprocess_data(self, input_imgs, crop_data=True):
        imgs = []
        poses = []
        stickmen = []

        if len(input_imgs.shape) == 3:
            input_imgs = input_imgs[None]
            N = 1

        else:
            N = input_imgs.shape[0]

        for i in range(N):
            pose = self.fa.get_landmarks(input_imgs[i])[0]

            center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)
            size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
            center[1] -= size // 6

            if input_imgs is None:
                # Crop poses
                if crop_data:
                    s = size * 2
                    pose -= center - size

            else:
                # Crop images and poses
                img = Image.fromarray(input_imgs[i])

                if crop_data:
                    img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                    s = img.size[0]
                    pose -= center - size

                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)

                imgs.append((self.to_tensor(img) - 0.5) * 2)

            if crop_data:
                pose = pose / float(s)

            poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))

        poses = torch.stack(poses, 0)[None]

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, poses[0])
            stickmen = stickmen[None]

        if input_imgs is not None:
            imgs = torch.stack(imgs, 0)[None]

        if self.args.num_gpus > 0:
            poses = poses.cuda()
            
            if input_imgs is not None:
                imgs = imgs.cuda()

                if self.args.output_stickmen:
                    stickmen = stickmen.cuda()

        segs = None
        if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            segs = self.net_seg(imgs)[None]

        return poses, imgs, segs, stickmen

    def forward(self, data_dict, crop_data=True, no_grad=True):
        if 'target_imgs' not in data_dict.keys():
            data_dict['target_imgs'] = None

        # Inference without finetuning
        (source_poses, 
         source_imgs, 
         source_segs, 
         source_stickmen) = self.preprocess_data(data_dict['source_imgs'], crop_data)

        (target_poses,
         target_imgs, 
         target_segs, 
         target_stickmen) = self.preprocess_data(data_dict['target_imgs'], crop_data)

        data_dict = {
            'source_imgs': source_imgs,
            'source_poses': source_poses,
            'target_poses': target_poses}

        if len(target_imgs):
           data_dict['target_imgs'] = target_imgs

        if source_segs is not None:
            data_dict['source_segs'] = source_segs

        if target_segs is not None:
            data_dict['target_segs'] = target_segs

        if source_stickmen is not None:
            data_dict['source_stickmen'] = source_stickmen

        if target_stickmen is not None:
            data_dict['target_stickmen'] = target_stickmen

        if no_grad:
            with torch.no_grad():
                self.runner(data_dict)

        else:
            self.runner(data_dict)

        return self.runner.data_dict

import sys
sys.path.append('./')
# PyTorch includes
import torch
import numpy as np

from utils import test_human
from PIL import Image

#
import argparse

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker',default=12,type=int)
    parser.add_argument('--freezeBN', choices=dict(true=True, false=False), default=True, action=LookupChoices)
    parser.add_argument('--step', default=30, type=int)
    parser.add_argument('--txt_file',default=None,type=str)
    parser.add_argument('--pred_path',default=None,type=str)
    parser.add_argument('--gt_path',default=None,type=str)
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--testepoch', default=10, type=int)
    opts = parser.parse_args()
    return opts

def eval_(pred_path, gt_path, classes, txt_file):
    pred_path = pred_path
    gt_path = gt_path

    with open(txt_file,) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    output_list = []
    label_list = []
    for i,file in enumerate(lines):
        print(i)
        file_name = file + '.png'
        try:
            predict_pic = np.array(Image.open(pred_path+file_name))
            gt_pic = np.array(Image.open(gt_path+file_name))
            output_list.append(torch.from_numpy(predict_pic))
            label_list.append(torch.from_numpy(gt_pic))
        except:
            print(file_name,flush=True)
            raise RuntimeError('no predict/gt image.')
            # gt_pic = np.array(Image.open(gt_path + file_name))
            # output_list.append(torch.from_numpy(gt_pic))
            # label_list.append(torch.from_numpy(gt_pic))


    miou = test_human.get_iou_from_list(output_list, label_list, n_cls=classes)

    print('Validation:')
    print('MIoU: %f\n' % miou)

if __name__ == '__main__':
    opts = get_parser()
    eval_(pred_path=opts.pred_path, gt_path=opts.gt_path, classes=opts.classes, txt_file=opts.txt_file)
from .test_from_disk import eval_

__all__ = ['eval_']
import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
import glob
from collections import OrderedDict
sys.path.append('../../')
# PyTorch includes
import torch
import pdb
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import cv2

# Tensorboard include
# from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import  cihp
from utils import util
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse
import copy
import torch.nn.functional as F
from test_from_disk import eval_


gpu_id = 1

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker', default=12, type=int)
    parser.add_argument('--step', default=30, type=int)
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--testepoch', default=10, type=int)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--txt_file', default='', type=str)
    parser.add_argument('--hidden_layers', default=128, type=int)
    parser.add_argument('--gpus', default=4, type=int)
    parser.add_argument('--output_path', default='./results/', type=str)
    parser.add_argument('--gt_path', default='./results/', type=str)
    opts = parser.parse_args()
    return opts


def main(opts):
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    p = OrderedDict()  # Parameters to include in report
    p['trainBatch'] = opts.batch  # Training batch size
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = opts.lr  # Learning rate
    p['lrFtr'] = 1e-5
    p['lraspp'] = 1e-5
    p['lrpro'] = 1e-5
    p['lrdecoder'] = 1e-5
    p['lrother']  = 1e-5
    p['wd'] = 5e-4  # Weight decay
    p['momentum'] = 0.9  # Momentum
    p['epoch_size'] = 10  # How many epochs to change learning rate
    p['num_workers'] = opts.numworker
    backbone = 'xception' # Use xception or resnet as feature extractor,

    with open(opts.txt_file, 'r') as f:
        img_list = f.readlines()

    max_id = 0
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    runs = glob.glob(os.path.join(save_dir_root, 'run', 'run_*'))
    for r in runs:
        run_id = int(r.split('_')[-1])
        if run_id >= max_id:
            max_id = run_id + 1
    # run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    # Network definition
    if backbone == 'xception':
        net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=opts.classes, os=16,
                                                                                     hidden_layers=opts.hidden_layers, source_classes=7,
                                                                                     )
    elif backbone == 'resnet':
        # net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True)
        raise NotImplementedError
    else:
        raise NotImplementedError

    if gpu_id >= 0:
        net.cuda()

    # net load weights
    if not opts.loadmodel =='':
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print('load model:' ,opts.loadmodel)
    else:
        print('no model load !!!!!!!!')

    ## multi scale
    scale_list=[1,0.5,0.75,1.25,1.5,1.75]
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_(pv),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_(pv),
            tr.HorizontalFlip(),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

        voc_val = cihp.VOCSegmentation(split='test', transform=composed_transforms_ts)
        voc_val_f = cihp.VOCSegmentation(split='test', transform=composed_transforms_ts_flip)

        testloader = DataLoader(voc_val, batch_size=1, shuffle=False, num_workers=p['num_workers'])
        testloader_flip = DataLoader(voc_val_f, batch_size=1, shuffle=False, num_workers=p['num_workers'])

        testloader_list.append(copy.deepcopy(testloader))
        testloader_flip_list.append(copy.deepcopy(testloader_flip))

    print("Eval Network")

    if not os.path.exists(opts.output_path + 'cihp_output_vis/'):
        os.makedirs(opts.output_path + 'cihp_output_vis/')
    if not os.path.exists(opts.output_path + 'cihp_output/'):
        os.makedirs(opts.output_path + 'cihp_output/')

    start_time = timeit.default_timer()
    # One testing epoch
    total_iou = 0.0
    net.eval()
    for ii, large_sample_batched in enumerate(zip(*testloader_list, *testloader_flip_list)):
        print(ii)
        #1 0.5 0.75 1.25 1.5 1.75 ; flip:
        sample1 = large_sample_batched[:6]
        sample2 = large_sample_batched[6:]
        for iii,sample_batched in enumerate(zip(sample1,sample2)):
            inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
            inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
            inputs = torch.cat((inputs,inputs_f),dim=0)
            if iii == 0:
                _,_,h,w = inputs.size()
            # assert inputs.size() == inputs_f.size()

            # Forward pass of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)

            with torch.no_grad():
                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # outputs = net.forward(inputs)
                # pdb.set_trace()
                outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
                outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
                outputs = outputs.unsqueeze(0)

                if iii>0:
                    outputs = F.upsample(outputs,size=(h,w),mode='bilinear',align_corners=True)
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs.clone()
        ################ plot pic
        predictions = torch.max(outputs_final, 1)[1]
        prob_predictions = torch.max(outputs_final,1)[0]
        results = predictions.cpu().numpy()
        prob_results = prob_predictions.cpu().numpy()
        vis_res = decode_labels(results)

        parsing_im = Image.fromarray(vis_res[0])
        parsing_im.save(opts.output_path + 'cihp_output_vis/{}.png'.format(img_list[ii][:-1]))
        cv2.imwrite(opts.output_path + 'cihp_output/{}.png'.format(img_list[ii][:-1]), results[0,:,:])
        # np.save('../../cihp_prob_output/{}.npy'.format(img_list[ii][:-1]), prob_results[0, :, :])
        # pred_list.append(predictions.cpu())
        # label_list.append(labels.squeeze(1).cpu())
        # loss = criterion(outputs, labels, batch_average=True)
        # running_loss_ts += loss.item()

        # total_iou += utils.get_iou(predictions, labels)
    end_time = timeit.default_timer()
    print('time use for '+str(ii) + ' is :' + str(end_time - start_time))

    # Eval
    pred_path = opts.output_path + 'cihp_output/'
    eval_(pred_path=pred_path, gt_path=opts.gt_path,classes=opts.classes, txt_file=opts.txt_file)

if __name__ == '__main__':
    opts = get_parser()
    main(opts)
import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
import glob
from collections import OrderedDict
sys.path.append('../../')
# PyTorch includes
import torch
import pdb
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import cv2

# Tensorboard include
# from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal
from utils import util
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse
import copy
import torch.nn.functional as F
from test_from_disk import eval_


gpu_id = 1

label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0), (0,128,0), (128,128,0), (0,0,128), (128,0,128), (0,128,128)]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

# def flip_cihp(tail_list):
#     '''
#
#     :param tail_list: tail_list size is 1 x n_class x h x w
#     :return:
#     '''
#     # tail_list = tail_list[0]
#     tail_list_rev = [None] * 20
#     for xx in range(14):
#         tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
#     tail_list_rev[14] = tail_list[15].unsqueeze(0)
#     tail_list_rev[15] = tail_list[14].unsqueeze(0)
#     tail_list_rev[16] = tail_list[17].unsqueeze(0)
#     tail_list_rev[17] = tail_list[16].unsqueeze(0)
#     tail_list_rev[18] = tail_list[19].unsqueeze(0)
#     tail_list_rev[19] = tail_list[18].unsqueeze(0)
#     return torch.cat(tail_list_rev,dim=0)

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker', default=12, type=int)
    parser.add_argument('--step', default=30, type=int)
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--testepoch', default=10, type=int)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--txt_file', default='', type=str)
    parser.add_argument('--hidden_layers', default=128, type=int)
    parser.add_argument('--gpus', default=4, type=int)
    parser.add_argument('--output_path', default='./results/', type=str)
    parser.add_argument('--gt_path', default='./results/', type=str)
    opts = parser.parse_args()
    return opts


def main(opts):
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda()

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj1_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj3_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    p = OrderedDict()  # Parameters to include in report
    p['trainBatch'] = opts.batch  # Training batch size
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = opts.lr  # Learning rate
    p['lrFtr'] = 1e-5
    p['lraspp'] = 1e-5
    p['lrpro'] = 1e-5
    p['lrdecoder'] = 1e-5
    p['lrother']  = 1e-5
    p['wd'] = 5e-4  # Weight decay
    p['momentum'] = 0.9  # Momentum
    p['epoch_size'] = 10  # How many epochs to change learning rate
    p['num_workers'] = opts.numworker
    backbone = 'xception' # Use xception or resnet as feature extractor,

    with open(opts.txt_file, 'r') as f:
        img_list = f.readlines()

    max_id = 0
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    runs = glob.glob(os.path.join(save_dir_root, 'run', 'run_*'))
    for r in runs:
        run_id = int(r.split('_')[-1])
        if run_id >= max_id:
            max_id = run_id + 1
    # run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    # Network definition
    if backbone == 'xception':
        net = deeplab_xception_transfer.deeplab_xception_transfer_projection(n_classes=opts.classes, os=16,
                                                                                     hidden_layers=opts.hidden_layers, source_classes=20,
                                                                                     )
    elif backbone == 'resnet':
        # net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True)
        raise NotImplementedError
    else:
        raise NotImplementedError

    if gpu_id >= 0:
        net.cuda()

    # net load weights
    if not opts.loadmodel =='':
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print('load model:' ,opts.loadmodel)
    else:
        print('no model load !!!!!!!!')

    ## multi scale
    scale_list=[1,0.5,0.75,1.25,1.5,1.75]
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_(pv),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_(pv),
            tr.HorizontalFlip(),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

        voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
        voc_val_f = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

        testloader = DataLoader(voc_val, batch_size=1, shuffle=False, num_workers=p['num_workers'])
        testloader_flip = DataLoader(voc_val_f, batch_size=1, shuffle=False, num_workers=p['num_workers'])

        testloader_list.append(copy.deepcopy(testloader))
        testloader_flip_list.append(copy.deepcopy(testloader_flip))

    print("Eval Network")

    if not os.path.exists(opts.output_path + 'pascal_output_vis/'):
        os.makedirs(opts.output_path + 'pascal_output_vis/')
    if not os.path.exists(opts.output_path + 'pascal_output/'):
        os.makedirs(opts.output_path + 'pascal_output/')

    start_time = timeit.default_timer()
    # One testing epoch
    total_iou = 0.0
    net.eval()
    for ii, large_sample_batched in enumerate(zip(*testloader_list, *testloader_flip_list)):
        print(ii)
        #1 0.5 0.75 1.25 1.5 1.75 ; flip:
        sample1 = large_sample_batched[:6]
        sample2 = large_sample_batched[6:]
        for iii,sample_batched in enumerate(zip(sample1,sample2)):
            inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
            inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
            inputs = torch.cat((inputs,inputs_f),dim=0)
            if iii == 0:
                _,_,h,w = inputs.size()
            # assert inputs.size() == inputs_f.size()

            # Forward pass of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)

            with torch.no_grad():
                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # outputs = net.forward(inputs)
                # pdb.set_trace()
                outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
                outputs = (outputs[0] + flip(outputs[1], dim=-1)) / 2
                outputs = outputs.unsqueeze(0)

                if iii>0:
                    outputs = F.upsample(outputs,size=(h,w),mode='bilinear',align_corners=True)
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs.clone()
        ################ plot pic
        predictions = torch.max(outputs_final, 1)[1]
        prob_predictions = torch.max(outputs_final,1)[0]
        results = predictions.cpu().numpy()
        prob_results = prob_predictions.cpu().numpy()
        vis_res = decode_labels(results)

        parsing_im = Image.fromarray(vis_res[0])
        parsing_im.save(opts.output_path + 'pascal_output_vis/{}.png'.format(img_list[ii][:-1]))
        cv2.imwrite(opts.output_path + 'pascal_output/{}.png'.format(img_list[ii][:-1]), results[0,:,:])
        # np.save('../../cihp_prob_output/{}.npy'.format(img_list[ii][:-1]), prob_results[0, :, :])
        # pred_list.append(predictions.cpu())
        # label_list.append(labels.squeeze(1).cpu())
        # loss = criterion(outputs, labels, batch_average=True)
        # running_loss_ts += loss.item()

        # total_iou += utils.get_iou(predictions, labels)
    end_time = timeit.default_timer()
    print('time use for '+str(ii) + ' is :' + str(end_time - start_time))

    # Eval
    pred_path = opts.output_path + 'pascal_output/'
    eval_(pred_path=pred_path, gt_path=opts.gt_path,classes=opts.classes, txt_file=opts.txt_file)

if __name__ == '__main__':
    opts = get_parser()
    main(opts)
import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict
sys.path.append('./')
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2


# Custom includes
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse
import torch.nn.functional as F

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample

def inference(net, img_path='', output_path='./', output_name='f', use_gpu=True):
    '''

    :param net:
    :param img_path:
    :param output_path:
    :return:
    '''
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    img = read_img(img_path)
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.HorizontalFlip_only_img(),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        testloader_list.append(img_transform(img, composed_transforms_ts))
        # print(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
    # print(testloader_list)
    start_time = timeit.default_timer()
    # One testing epoch
    net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = inputs.unsqueeze(0)
        inputs_f = inputs_f.unsqueeze(0)
        inputs = torch.cat((inputs, inputs_f), dim=0)
        if iii == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            if use_gpu >= 0:
                inputs = inputs.cuda()
            # outputs = net.forward(inputs)
            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if iii > 0:
                outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs.clone()
    ################ plot pic
    predictions = torch.max(outputs_final, 1)[1]
    results = predictions.cpu().numpy()
    vis_res = decode_labels(results)

    parsing_im = Image.fromarray(vis_res[0])
    parsing_im.save(output_path+'/{}.png'.format(output_name))
    cv2.imwrite(output_path+'/{}_gray.png'.format(output_name), results[0, :, :])

    end_time = timeit.default_timer()
    print('time used for the multi-scale image inference' + ' is :' + str(end_time - start_time))

if __name__ == '__main__':
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--img_path', default='', type=str)
    parser.add_argument('--output_path', default='', type=str)
    parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--use_gpu', default=1, type=int)
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )
    if not opts.loadmodel == '':
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print('load model:', opts.loadmodel)
    else:
        print('no model load !!!!!!!!')
        raise RuntimeError('No model!!!!')

    if opts.use_gpu >0 :
        net.cuda()
        use_gpu = True
    else:
        use_gpu = False
        raise RuntimeError('must use the gpu!!!!')

    inference(net=net, img_path=opts.img_path,output_path=opts.output_path , output_name=opts.output_name, use_gpu=use_gpu)


import socket
import timeit
from datetime import datetime
import os
import sys
import glob
import numpy as np
from collections import OrderedDict
sys.path.append('./')
sys.path.append('./networks/')
# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import random

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal, cihp_pascal_atr
from utils import get_iou_from_list
from utils import util as ut
from networks import deeplab_xception_universal, graph
from dataloaders import custom_transforms as tr
from utils import sampler as sam
#
import argparse

'''
source is cihp
target is pascal
'''

gpu_id = 1
# print('Using GPU: {} '.format(gpu_id))

# nEpochs = 100  # Number of epochs for training
resume_epoch = 0   # Default is 0, change if want to resume

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker',default=12,type=int)
    # parser.add_argument('--freezeBN', choices=dict(true=True, false=False), default=True, action=LookupChoices)
    parser.add_argument('--step', default=10, type=int)
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--testepoch', default=10, type=int)
    parser.add_argument('--loadmodel',default='',type=str)
    parser.add_argument('--pretrainedModel', default='', type=str)
    parser.add_argument('--hidden_layers',default=128,type=int)
    parser.add_argument('--gpus',default=4, type=int)
    parser.add_argument('--testInterval', default=5, type=int)
    opts = parser.parse_args()
    return opts

def get_graphs(opts):
    '''source is pascal; target is cihp; middle is atr'''
    # target 1
    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj1_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1 = adj1_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 20, 20).cuda()
    adj1_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20)
    #source 2
    adj2_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj2 = adj2_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 7).cuda()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7)
    # s to target 3
    adj3_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj3 = adj3_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20).transpose(2,3).cuda()
    adj3_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).transpose(2,3)
    # middle 4
    atr_adj = graph.preprocess_adj(graph.atr_graph)
    adj4_ = Variable(torch.from_numpy(atr_adj).float())
    adj4 = adj4_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 18, 18).cuda()
    adj4_test = adj4_.unsqueeze(0).unsqueeze(0).expand(1, 1, 18, 18)
    # source to middle 5
    adj5_ = torch.from_numpy(graph.pascal2atr_nlp_adj).float()
    adj5 = adj5_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 18).cuda()
    adj5_test = adj5_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 18)
    # target to middle 6
    adj6_ = torch.from_numpy(graph.cihp2atr_nlp_adj).float()
    adj6 = adj6_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 20, 18).cuda()
    adj6_test = adj6_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 18)
    train_graph = [adj1, adj2, adj3, adj4, adj5, adj6]
    test_graph = [adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test]
    return train_graph, test_graph


def main(opts):
    # Set parameters
    p = OrderedDict()  # Parameters to include in report
    p['trainBatch'] = opts.batch  # Training batch size
    testBatch = 1  # Testing batch size
    useTest = True  # See evolution of the test set when training
    nTestInterval = opts.testInterval # Run on test set every nTestInterval epochs
    snapshot = 1  # Store a model every snapshot epochs
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = opts.lr  # Learning rate
    p['wd'] = 5e-4  # Weight decay
    p['momentum'] = 0.9  # Momentum
    p['epoch_size'] = opts.step  # How many epochs to change learning rate
    p['num_workers'] = opts.numworker
    model_path = opts.pretrainedModel
    backbone = 'xception' # Use xception or resnet as feature extractor
    nEpochs = opts.epochs

    max_id = 0
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    runs = glob.glob(os.path.join(save_dir_root, 'run', 'run_*'))
    for r in runs:
        run_id = int(r.split('_')[-1])
        if run_id >= max_id:
            max_id = run_id + 1
    # run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(max_id))

    # Network definition
    if backbone == 'xception':
        net_ = deeplab_xception_universal.deeplab_xception_end2end_3d(n_classes=20, os=16,
                                                                      hidden_layers=opts.hidden_layers,
                                                                      source_classes=7,
                                                                      middle_classes=18, )
    elif backbone == 'resnet':
        # net_ = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True)
        raise NotImplementedError
    else:
        raise NotImplementedError

    modelName = 'deeplabv3plus-' + backbone + '-voc'+datetime.now().strftime('%b%d_%H-%M-%S')
    criterion = ut.cross_entropy2d

    if gpu_id >= 0:
        # torch.cuda.set_device(device=gpu_id)
        net_.cuda()

    # net load weights
    if not model_path == '':
        x = torch.load(model_path)
        net_.load_state_dict_new(x)
        print('load pretrainedModel.')
    else:
        print('no pretrainedModel.')

    if not opts.loadmodel =='':
        x = torch.load(opts.loadmodel)
        net_.load_source_model(x)
        print('load model:' ,opts.loadmodel)
    else:
        print('no trained model load !!!!!!!!')

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('load model',opts.loadmodel,1)
    writer.add_text('setting',sys.argv[0],1)

    # Use the following optimizer
    optimizer = optim.SGD(net_.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])

    composed_transforms_tr = transforms.Compose([
            tr.RandomSized_new(512),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

    composed_transforms_ts = transforms.Compose([
        tr.Normalize_xception_tf(),
        tr.ToTensor_()])

    composed_transforms_ts_flip = transforms.Compose([
        tr.HorizontalFlip(),
        tr.Normalize_xception_tf(),
        tr.ToTensor_()])

    all_train = cihp_pascal_atr.VOCSegmentation(split='train', transform=composed_transforms_tr, flip=True)
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
    voc_val_flip = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

    num_cihp,num_pascal,num_atr = all_train.get_class_num()
    ss = sam.Sampler_uni(num_cihp,num_pascal,num_atr,opts.batch)
    # balance datasets based pascal
    ss_balanced = sam.Sampler_uni(num_cihp,num_pascal,num_atr,opts.batch, balance_id=1)

    trainloader = DataLoader(all_train, batch_size=p['trainBatch'], shuffle=False, num_workers=p['num_workers'],
                             sampler=ss, drop_last=True)
    trainloader_balanced = DataLoader(all_train, batch_size=p['trainBatch'], shuffle=False, num_workers=p['num_workers'],
                             sampler=ss_balanced, drop_last=True)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=p['num_workers'])
    testloader_flip = DataLoader(voc_val_flip, batch_size=testBatch, shuffle=False, num_workers=p['num_workers'])

    num_img_tr = len(trainloader)
    num_img_balanced = len(trainloader_balanced)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_tr_atr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    print("Training Network")
    net = torch.nn.DataParallel(net_)

    id_list = torch.LongTensor(range(opts.batch))
    pascal_iter = int(num_img_tr//opts.batch)

    # Get graphs
    train_graph, test_graph = get_graphs(opts)
    adj1, adj2, adj3, adj4, adj5, adj6 = train_graph
    adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test = test_graph

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, int(1.5*nEpochs)):
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1 and epoch<nEpochs:
            lr_ = ut.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            optimizer = optim.SGD(net_.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
            print('(poly lr policy) learning rate: ', lr_)
            writer.add_scalar('data/lr_',lr_,epoch)
        elif epoch % p['epoch_size'] == p['epoch_size'] - 1 and epoch > nEpochs:
            lr_ = ut.lr_poly(p['lr'], epoch-nEpochs, int(0.5*nEpochs), 0.9)
            optimizer = optim.SGD(net_.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
            print('(poly lr policy) learning rate: ', lr_)
            writer.add_scalar('data/lr_', lr_, epoch)

        net_.train()
        if epoch < nEpochs:
            for ii, sample_batched in enumerate(trainloader):
                inputs, labels = sample_batched['image'], sample_batched['label']
                dataset_lbl = sample_batched['pascal'][0].item()
                # Forward-Backward of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                global_step += 1

                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()

                if dataset_lbl == 0:
                    # 0 is cihp -- target
                    _, outputs,_ = net.forward(None, input_target=inputs, input_middle=None, adj1_target=adj1, adj2_source=adj2,
                        adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2,3), adj4_middle=adj4,adj5_transfer_s2m=adj5.transpose(2, 3),
                        adj6_transfer_t2m=adj6.transpose(2, 3),adj5_transfer_m2s=adj5,adj6_transfer_m2t=adj6,)
                elif dataset_lbl == 1:
                    # pascal is source
                    outputs, _, _ = net.forward(inputs, input_target=None, input_middle=None, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                else:
                    # atr
                    _, _, outputs = net.forward(None, input_target=None, input_middle=inputs, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                # print(sample_batched['pascal'])
                # print(outputs.size(),)
                # print(labels)
                loss = criterion(outputs, labels,  batch_average=True)
                running_loss_tr += loss.item()

                # Print stuff
                if ii % num_img_tr == (num_img_tr - 1):
                    running_loss_tr = running_loss_tr / num_img_tr
                    writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                    print('[Epoch: %d, numImages: %5d]' % (epoch, epoch))
                    print('Loss: %f' % running_loss_tr)
                    running_loss_tr = 0
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                # Backward the averaged gradient
                loss /= p['nAveGrad']
                loss.backward()
                aveGrad += 1

                # Update the weights once in p['nAveGrad'] forward passes
                if aveGrad % p['nAveGrad'] == 0:
                    writer.add_scalar('data/total_loss_iter', loss.item(), global_step)
                    if dataset_lbl == 0:
                        writer.add_scalar('data/total_loss_iter_cihp', loss.item(), global_step)
                    if dataset_lbl == 1:
                        writer.add_scalar('data/total_loss_iter_pascal', loss.item(), global_step)
                    if dataset_lbl == 2:
                        writer.add_scalar('data/total_loss_iter_atr', loss.item(), global_step)
                    optimizer.step()
                    optimizer.zero_grad()
                    # optimizer_gcn.step()
                    # optimizer_gcn.zero_grad()
                    aveGrad = 0

                # Show 10 * 3 images results each epoch
                if ii % (num_img_tr // 10) == 0:
                    grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                    writer.add_image('Image', grid_image, global_step)
                    grid_image = make_grid(ut.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
                                           range=(0, 255))
                    writer.add_image('Predicted label', grid_image, global_step)
                    grid_image = make_grid(ut.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                    writer.add_image('Groundtruth label', grid_image, global_step)

                print('loss is ',loss.cpu().item(),flush=True)
        else:
            # Balanced the number of datasets
            for ii, sample_batched in enumerate(trainloader_balanced):
                inputs, labels = sample_batched['image'], sample_batched['label']
                dataset_lbl = sample_batched['pascal'][0].item()
                # Forward-Backward of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                global_step += 1

                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()

                if dataset_lbl == 0:
                    # 0 is cihp -- target
                    _, outputs, _ = net.forward(None, input_target=inputs, input_middle=None, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                elif dataset_lbl == 1:
                    # pascal is source
                    outputs, _, _ = net.forward(inputs, input_target=None, input_middle=None, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                else:
                    # atr
                    _, _, outputs = net.forward(None, input_target=None, input_middle=inputs, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                # print(sample_batched['pascal'])
                # print(outputs.size(),)
                # print(labels)
                loss = criterion(outputs, labels, batch_average=True)
                running_loss_tr += loss.item()

                # Print stuff
                if ii % num_img_balanced == (num_img_balanced - 1):
                    running_loss_tr = running_loss_tr / num_img_balanced
                    writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                    print('[Epoch: %d, numImages: %5d]' % (epoch, epoch))
                    print('Loss: %f' % running_loss_tr)
                    running_loss_tr = 0
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                # Backward the averaged gradient
                loss /= p['nAveGrad']
                loss.backward()
                aveGrad += 1

                # Update the weights once in p['nAveGrad'] forward passes
                if aveGrad % p['nAveGrad'] == 0:
                    writer.add_scalar('data/total_loss_iter', loss.item(), global_step)
                    if dataset_lbl == 0:
                        writer.add_scalar('data/total_loss_iter_cihp', loss.item(), global_step)
                    if dataset_lbl == 1:
                        writer.add_scalar('data/total_loss_iter_pascal', loss.item(), global_step)
                    if dataset_lbl == 2:
                        writer.add_scalar('data/total_loss_iter_atr', loss.item(), global_step)
                    optimizer.step()
                    optimizer.zero_grad()

                    aveGrad = 0

                # Show 10 * 3 images results each epoch
                if ii % (num_img_balanced // 10) == 0:
                    grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                    writer.add_image('Image', grid_image, global_step)
                    grid_image = make_grid(
                        ut.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3,
                        normalize=False,
                        range=(0, 255))
                    writer.add_image('Predicted label', grid_image, global_step)
                    grid_image = make_grid(
                        ut.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3,
                        normalize=False, range=(0, 255))
                    writer.add_image('Groundtruth label', grid_image, global_step)

                print('loss is ', loss.cpu().item(), flush=True)

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            val_pascal(net_=net_, testloader=testloader, testloader_flip=testloader_flip, test_graph=test_graph,
                       criterion=criterion, epoch=epoch, writer=writer)


def val_pascal(net_, testloader, testloader_flip, test_graph, criterion, epoch, writer, classes=7):
    running_loss_ts = 0.0
    miou = 0
    adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test = test_graph
    num_img_ts = len(testloader)
    net_.eval()
    pred_list = []
    label_list = []
    for ii, sample_batched in enumerate(zip(testloader, testloader_flip)):
        # print(ii)
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = torch.cat((inputs, inputs_f), dim=0)
        # Forward pass of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)

        with torch.no_grad():
            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs, _, _ = net_.forward(inputs, input_target=None, input_middle=None,
                                         adj1_target=adj1_test.cuda(),
                                         adj2_source=adj2_test.cuda(),
                                         adj3_transfer_s2t=adj3_test.cuda(),
                                         adj3_transfer_t2s=adj3_test.transpose(2, 3).cuda(),
                                         adj4_middle=adj4_test.cuda(),
                                         adj5_transfer_s2m=adj5_test.transpose(2, 3).cuda(),
                                         adj6_transfer_t2m=adj6_test.transpose(2, 3).cuda(),
                                         adj5_transfer_m2s=adj5_test.cuda(),
                                         adj6_transfer_m2t=adj6_test.cuda(), )
        # pdb.set_trace()
        outputs = (outputs[0] + flip(outputs[1], dim=-1)) / 2
        outputs = outputs.unsqueeze(0)
        predictions = torch.max(outputs, 1)[1]
        pred_list.append(predictions.cpu())
        label_list.append(labels.squeeze(1).cpu())
        loss = criterion(outputs, labels, batch_average=True)
        running_loss_ts += loss.item()

        # total_iou += utils.get_iou(predictions, labels)

        # Print stuff
        if ii % num_img_ts == num_img_ts - 1:
            # if ii == 10:
            miou = get_iou_from_list(pred_list, label_list, n_cls=classes)
            running_loss_ts = running_loss_ts / num_img_ts

            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii * 1 + inputs.data.shape[0]))
            writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
            writer.add_scalar('data/test_miour', miou, epoch)
            print('Loss: %f' % running_loss_ts)
            print('MIoU: %f\n' % miou)
    # return miou


if __name__ == '__main__':
    opts = get_parser()
    main(opts)
import socket
import timeit
from datetime import datetime
import os
import sys
import glob
import numpy as np
from collections import OrderedDict
sys.path.append('../../')
sys.path.append('../../networks/')
# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import cihp
from utils import util,get_iou_from_list
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse

gpu_id = 0

nEpochs = 100  # Number of epochs for training
resume_epoch = 0   # Default is 0, change if want to resume

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker',default=12,type=int)
    parser.add_argument('--freezeBN', choices=dict(true=True, false=False), default=True, action=LookupChoices)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--classes', default=20, type=int)
    parser.add_argument('--testInterval', default=10, type=int)
    parser.add_argument('--loadmodel',default='',type=str)
    parser.add_argument('--pretrainedModel', default='', type=str)
    parser.add_argument('--hidden_layers',default=128,type=int)
    parser.add_argument('--gpus',default=4, type=int)

    opts = parser.parse_args()
    return opts

def get_graphs(opts):
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2 = adj2_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20).transpose(2, 3).cuda()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3 = adj1_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 7).cuda()
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7)

    # adj2 = torch.from_numpy(graph.cihp2pascal_adj).float()
    # adj2 = adj2.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20)
    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1 = adj3_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 20, 20).cuda()
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20)
    train_graph = [adj1, adj2, adj3]
    test_graph = [adj1_test, adj2_test, adj3_test]
    return train_graph, test_graph


def val_cihp(net_, testloader, testloader_flip, test_graph, epoch, writer, criterion, classes=20):
    adj1_test, adj2_test, adj3_test = test_graph
    num_img_ts = len(testloader)
    net_.eval()
    pred_list = []
    label_list = []
    running_loss_ts = 0.0
    miou = 0
    for ii, sample_batched in enumerate(zip(testloader, testloader_flip)):

        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = torch.cat((inputs, inputs_f), dim=0)
        # Forward pass of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)
        if gpu_id >= 0:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = net_.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
        # pdb.set_trace()
        outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
        outputs = outputs.unsqueeze(0)
        predictions = torch.max(outputs, 1)[1]
        pred_list.append(predictions.cpu())
        label_list.append(labels.squeeze(1).cpu())
        loss = criterion(outputs, labels, batch_average=True)
        running_loss_ts += loss.item()
        # total_iou += utils.get_iou(predictions, labels)
        # Print stuff
        if ii % num_img_ts == num_img_ts - 1:
            # if ii == 10:
            miou = get_iou_from_list(pred_list, label_list, n_cls=classes)
            running_loss_ts = running_loss_ts / num_img_ts

            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii * 1 + inputs.data.shape[0]))
            writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
            writer.add_scalar('data/test_miour', miou, epoch)
            print('Loss: %f' % running_loss_ts)
            print('MIoU: %f\n' % miou)


def main(opts):
    p = OrderedDict()  # Parameters to include in report
    p['trainBatch'] = opts.batch  # Training batch size
    testBatch = 1  # Testing batch size
    useTest = True  # See evolution of the test set when training
    nTestInterval = opts.testInterval # Run on test set every nTestInterval epochs
    snapshot = 1  # Store a model every snapshot epochs
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = opts.lr  # Learning rate
    p['lrFtr'] = 1e-5
    p['lraspp'] = 1e-5
    p['lrpro'] = 1e-5
    p['lrdecoder'] = 1e-5
    p['lrother']  = 1e-5
    p['wd'] = 5e-4  # Weight decay
    p['momentum'] = 0.9  # Momentum
    p['epoch_size'] = opts.step  # How many epochs to change learning rate
    p['num_workers'] = opts.numworker
    model_path = opts.pretrainedModel
    backbone = 'xception' # Use xception or resnet as feature extractor,
    nEpochs = opts.epochs

    max_id = 0
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    runs = glob.glob(os.path.join(save_dir_root, 'run_cihp', 'run_*'))
    for r in runs:
        run_id = int(r.split('_')[-1])
        if run_id >= max_id:
            max_id = run_id + 1
    save_dir = os.path.join(save_dir_root, 'run_cihp', 'run_' + str(max_id))

    # Network definition
    if backbone == 'xception':
        net_ = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=opts.classes, os=16,
                                                                                      hidden_layers=opts.hidden_layers, source_classes=7, )
    elif backbone == 'resnet':
        # net_ = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True)
        raise NotImplementedError
    else:
        raise NotImplementedError

    modelName = 'deeplabv3plus-' + backbone + '-voc'+datetime.now().strftime('%b%d_%H-%M-%S')
    criterion = util.cross_entropy2d

    if gpu_id >= 0:
        # torch.cuda.set_device(device=gpu_id)
        net_.cuda()

    # net load weights
    if not model_path == '':
        x = torch.load(model_path)
        net_.load_state_dict_new(x)
        print('load pretrainedModel:', model_path)
    else:
        print('no pretrainedModel.')
    if not opts.loadmodel =='':
        x = torch.load(opts.loadmodel)
        net_.load_source_model(x)
        print('load model:' ,opts.loadmodel)
    else:
        print('no model load !!!!!!!!')

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('load model',opts.loadmodel,1)
    writer.add_text('setting',sys.argv[0],1)

    if opts.freezeBN:
        net_.freeze_bn()

    # Use the following optimizer
    optimizer = optim.SGD(net_.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])

    composed_transforms_tr = transforms.Compose([
            tr.RandomSized_new(512),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

    composed_transforms_ts = transforms.Compose([
        tr.Normalize_xception_tf(),
        tr.ToTensor_()])

    composed_transforms_ts_flip = transforms.Compose([
        tr.HorizontalFlip(),
        tr.Normalize_xception_tf(),
        tr.ToTensor_()])

    voc_train = cihp.VOCSegmentation(split='train', transform=composed_transforms_tr, flip=True)
    voc_val = cihp.VOCSegmentation(split='val', transform=composed_transforms_ts)
    voc_val_flip = cihp.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

    trainloader = DataLoader(voc_train, batch_size=p['trainBatch'], shuffle=True, num_workers=p['num_workers'],drop_last=True)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=p['num_workers'])
    testloader_flip = DataLoader(voc_val_flip, batch_size=testBatch, shuffle=False, num_workers=p['num_workers'])

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    print("Training Network")

    net = torch.nn.DataParallel(net_)
    train_graph, test_graph = get_graphs(opts)
    adj1, adj2, adj3 = train_graph


    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            lr_ = util.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            optimizer = optim.SGD(net_.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
            writer.add_scalar('data/lr_', lr_, epoch)
            print('(poly lr policy) learning rate: ', lr_)

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, labels = sample_batched['image'], sample_batched['label']
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs, adj1, adj3, adj2)

            loss = criterion(outputs, labels, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            # Show 10 * 3 images results each epoch
            if ii % (num_img_tr // 10) == 0:
                grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)
                grid_image = make_grid(util.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(util.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)
            print('loss is ', loss.cpu().item(), flush=True)

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

        torch.cuda.empty_cache()

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            val_cihp(net_,testloader=testloader, testloader_flip=testloader_flip, test_graph=test_graph,
                     epoch=epoch,writer=writer,criterion=criterion, classes=opts.classes)
        torch.cuda.empty_cache()




if __name__ == '__main__':
    opts = get_parser()
    main(opts)
import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps
from torchvision import transforms

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            return {'image': img,
                    'label': mask}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}

class RandomCrop_new(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}

        new_img = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_mask = Image.new('L',(tw,th),'white')  # same above

        # if w > tw or h > th
        x1 = y1 = 0
        if w > tw:
            x1 = random.randint(0,w - tw)
        if h > th:
            y1 = random.randint(0,h - th)
        # crop
        img = img.crop((x1,y1, x1 + tw, y1 + th))
        mask = mask.crop((x1,y1, x1 + tw, y1 + th))
        new_img.paste(img,(0,0))
        new_mask.paste(mask,(0,0))

        # x1 = random.randint(0, w - tw)
        # y1 = random.randint(0, h - th)
        # img = img.crop((x1, y1, x1 + tw, y1 + th))
        # mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': new_img,
                'label': new_mask}

class Paste(object):
    def __init__(self, size,):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        assert (w <=tw) and (h <= th)
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}

        new_img = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_mask = Image.new('L',(tw,th),'white')  # same above

        new_img.paste(img,(0,0))
        new_mask.paste(mask,(0,0))

        return {'image': new_img,
                'label': new_mask}

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class HorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class HorizontalFlip_only_img(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip_cihp(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # mask = Image.open()

        return {'image': img,
                'label': mask}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class Normalize_255(object):
    """Normalize a tensor image with mean and standard deviation. tf use 255.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(123.15, 115.90, 103.06), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        # img = 255.0
        img -= self.mean
        img /= self.std
        img = img
        img = img[[0,3,2,1],...]
        return {'image': img,
                'label': mask}

class Normalize_xception_tf(object):
    # def __init__(self):
    #     self.rgb2bgr =

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img = (img*2.0)/255.0 - 1
        # print(img.shape)
        # img = img[[0,3,2,1],...]
        return {'image': img,
                'label': mask}

class Normalize_xception_tf_only_img(object):
    # def __init__(self):
    #     self.rgb2bgr =

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        # mask = np.array(sample['label']).astype(np.float32)
        img = (img*2.0)/255.0 - 1
        # print(img.shape)
        # img = img[[0,3,2,1],...]
        return {'image': img,
                'label': sample['label']}

class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'label': mask}

class ToTensor_(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        # mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        img = self.rgb2bgr(img)
        mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': mask}

class ToTensor_only_img(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        # mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        # mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        img = self.rgb2bgr(img)
        # mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': sample['label']}

class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class Keep_origin_size_Resize(object):
    def __init__(self, max_size, scale=1.0):
        self.size = tuple(reversed(max_size))  # size: (h, w)
        self.scale = scale
        self.paste = Paste(int(max_size[0]*scale))

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size
        h, w = self.size
        h = int(h*self.scale)
        w = int(w*self.scale)
        img = img.resize((h, w), Image.BILINEAR)
        mask = mask.resize((h, w), Image.NEAREST)

        return self.paste({'image': img,
                'label': mask})

class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class Scale_(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        ow = int(w*self.scale)
        oh = int(h*self.scale)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class Scale_only_img(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # assert img.size == mask.size
        w, h = img.size
        ow = int(w*self.scale)
        oh = int(h*self.scale)
        img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomSized_new(object):
    '''what we use is this class to aug'''
    def __init__(self, size,scale1=0.5,scale2=2):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop_new(self.size)
        self.small_scale = scale1
        self.big_scale = scale2

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        w = int(random.uniform(self.small_scale, self.big_scale) * img.size[0])
        h = int(random.uniform(self.small_scale, self.big_scale) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {'image': img, 'label': mask}
        # finish resize
        return self.crop(sample)
# class Random

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': mask}
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'atr':
            return './data/datasets/ATR/'  # folder that contains atr/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return './data/datasets/pascal/'  # folder that contains pascal/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from .mypath_pascal import Path

class VOCSegmentation(Dataset):
    """
    Pascal dataset
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 transform=None
                 ):
        """
        :param base_dir: path to PASCAL dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationPart')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'list')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line+'.jpg' )
                _cat = os.path.join(self._cat_dir, line +'.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB') # return is RGB pic
        _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return 'PASCAL(split=' + str(self.split) + ')'

class test_segmentation(VOCSegmentation):
    def __init__(self,base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 transform=None,
                 flip=True):
        super(test_segmentation, self).__init__(base_dir=base_dir,split=split,transform=transform)
        self._flip_flag = flip

    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample





from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from .mypath_cihp import Path
import random

class VOCSegmentation(Dataset):
    """
    CIHP dataset
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('cihp'),
                 split='train',
                 transform=None,
                 flip=False,
                 ):
        """
        :param base_dir: path to CIHP dataset directory
        :param split: train/val/test
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()
        self._flip_flag = flip

        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'Images')
        self._cat_dir = os.path.join(self._base_dir, 'Category_ids')
        self._flip_dir = os.path.join(self._base_dir,'Category_rev_ids')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'lists')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.flip_categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line+'.jpg' )
                _cat = os.path.join(self._cat_dir, line +'.png')
                _flip = os.path.join(self._flip_dir,line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)


        assert (len(self.images) == len(self.categories))
        assert len(self.flip_categories) == len(self.categories)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')  # return is RGB pic
        if self._flip_flag:
            if random.random() < 0.5:
                _target = Image.open(self.flip_categories[index])
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                _target = Image.open(self.categories[index])
        else:
            _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return 'CIHP(split=' + str(self.split) + ')'




from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from .mypath_atr import Path
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VOCSegmentation(Dataset):
    """
    ATR dataset
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('atr'),
                 split='train',
                 transform=None,
                 flip=False,
                 ):
        """
        :param base_dir: path to ATR dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()
        self._flip_flag = flip

        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        self._flip_dir = os.path.join(self._base_dir,'SegmentationClassAug_rev')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'list')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.flip_categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line+'.jpg' )
                _cat = os.path.join(self._cat_dir, line +'.png')
                _flip = os.path.join(self._flip_dir,line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)


        assert (len(self.images) == len(self.categories))
        assert len(self.flip_categories) == len(self.categories)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')  # return is RGB pic
        if self._flip_flag:
            if random.random() < 0.5:
                _target = Image.open(self.flip_categories[index])
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                _target = Image.open(self.categories[index])
        else:
            _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return 'ATR(split=' + str(self.split) + ')'




class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'cihp':
            return './data/datasets/CIHP_4w/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .mypath_cihp import Path
from .mypath_pascal import Path as PP
from .mypath_atr import Path as PA
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VOCSegmentation(Dataset):
    """
    Pascal dataset
    """

    def __init__(self,
                 cihp_dir=Path.db_root_dir('cihp'),
                 split='train',
                 transform=None,
                 flip=False,
                 pascal_dir = PP.db_root_dir('pascal'),
                 atr_dir = PA.db_root_dir('atr'),
                 ):
        """
        :param cihp_dir: path to CIHP dataset directory
        :param pascal_dir: path to PASCAL dataset directory
        :param atr_dir: path to ATR dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()
        ## for cihp
        self._flip_flag = flip
        self._base_dir = cihp_dir
        self._image_dir = os.path.join(self._base_dir, 'Images')
        self._cat_dir = os.path.join(self._base_dir, 'Category_ids')
        self._flip_dir = os.path.join(self._base_dir,'Category_rev_ids')
        ## for Pascal
        self._base_dir_pascal = pascal_dir
        self._image_dir_pascal = os.path.join(self._base_dir_pascal, 'JPEGImages')
        self._cat_dir_pascal = os.path.join(self._base_dir_pascal, 'SegmentationPart')
        # self._flip_dir_pascal = os.path.join(self._base_dir_pascal, 'Category_rev_ids')
        ## for atr
        self._base_dir_atr = atr_dir
        self._image_dir_atr = os.path.join(self._base_dir_atr, 'JPEGImages')
        self._cat_dir_atr = os.path.join(self._base_dir_atr, 'SegmentationClassAug')
        self._flip_dir_atr = os.path.join(self._base_dir_atr, 'SegmentationClassAug_rev')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'lists')
        _splits_dir_pascal = os.path.join(self._base_dir_pascal, 'list')
        _splits_dir_atr = os.path.join(self._base_dir_atr, 'list')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.flip_categories = []
        self.datasets_lbl = []

        # num
        self.num_cihp = 0
        self.num_pascal = 0
        self.num_atr = 0
        # for cihp is 0
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()
            self.num_cihp += len(lines)
            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line+'.jpg' )
                _cat = os.path.join(self._cat_dir, line +'.png')
                _flip = os.path.join(self._flip_dir,line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)
                self.datasets_lbl.append(0)

        # for pascal is 1
        for splt in self.split:
            if splt == 'test':
                splt='val'
            with open(os.path.join(os.path.join(_splits_dir_pascal, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()
            self.num_pascal += len(lines)
            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir_pascal, line+'.jpg' )
                _cat = os.path.join(self._cat_dir_pascal, line +'.png')
                # _flip = os.path.join(self._flip_dir,line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                # assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append([])
                self.datasets_lbl.append(1)

        # for atr is 2
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir_atr, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()
            self.num_atr += len(lines)
            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir_atr, line + '.jpg')
                _cat = os.path.join(self._cat_dir_atr, line + '.png')
                _flip = os.path.join(self._flip_dir_atr, line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)
                self.datasets_lbl.append(2)

        assert (len(self.images) == len(self.categories))
        # assert len(self.flip_categories) == len(self.categories)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def get_class_num(self):
        return self.num_cihp,self.num_pascal,self.num_atr



    def __getitem__(self, index):
        _img, _target,_lbl= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target,}

        if self.transform is not None:
            sample = self.transform(sample)
        sample['pascal'] = _lbl
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')  # return is RGB pic
        type_lbl = self.datasets_lbl[index]
        if self._flip_flag:
            if random.random() < 0.5 :
                # _target = Image.open(self.flip_categories[index])
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                if type_lbl == 0 or type_lbl == 2:
                    _target = Image.open(self.flip_categories[index])
                else:
                    _target = Image.open(self.categories[index])
                    _target = _target.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                _target = Image.open(self.categories[index])
        else:
            _target = Image.open(self.categories[index])

        return _img, _target,type_lbl

    def __str__(self):
        return 'datasets(split=' + str(self.split) + ')'












if __name__ == '__main__':
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        # tr.RandomHorizontalFlip(),
        tr.RandomSized_new(512),
        tr.RandomRotate(15),
        tr.ToTensor_()])



    voc_train = VOCSegmentation(split='train',
                                transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=1)

    for ii, sample in enumerate(dataloader):
        if ii >10:
            break
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn import Parameter
from . import deeplab_xception, gcn, deeplab_xception_synBN



class deeplab_xception_transfer_basemodel_savememory(deeplab_xception.DeepLabv3_plus):
    def __init__(self, nInputChannels=3, n_classes=7, os=16, input_channels=256, hidden_layers=128, out_channels=256,
                 source_classes=20, transfer_graph=None):
        super(deeplab_xception_transfer_basemodel_savememory, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                  os=os,)

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_graph' not in name \
                    and 'transpose_graph' not in name and 'middle' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if 'target' in name or 'semantic' in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'source' in name:
                l.append(k)
        return l

    def top_forward(self, input, adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### source graph
        source_graph = self.source_featuremap_2_graph(x)

        source_graph1 = self.source_graph_conv1.forward(source_graph, adj=adj2_source, relu=True)
        source_graph2 = self.source_graph_conv2.forward(source_graph1, adj=adj2_source, relu=True)
        source_graph3 = self.source_graph_conv2.forward(source_graph2, adj=adj2_source, relu=True)

        ### target source
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)


    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph


        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


class deeplab_xception_transfer_basemodel_savememory_synbn(deeplab_xception_synBN.DeepLabv3_plus):
    def __init__(self, nInputChannels=3, n_classes=7, os=16, input_channels=256, hidden_layers=128, out_channels=256,
                 source_classes=20, transfer_graph=None):
        super(deeplab_xception_transfer_basemodel_savememory_synbn, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                  os=os,)


    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_graph' not in name \
                    and 'transpose_graph' not in name and 'middle' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if 'target' in name or 'semantic' in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'source' in name:
                l.append(k)
        return l

    def top_forward(self, input, adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### source graph
        source_graph = self.source_featuremap_2_graph(x)

        source_graph1 = self.source_graph_conv1.forward(source_graph, adj=adj2_source, relu=True)
        source_graph2 = self.source_graph_conv2.forward(source_graph1, adj=adj2_source, relu=True)
        source_graph3 = self.source_graph_conv2.forward(source_graph2, adj=adj2_source, relu=True)

        ### target source
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)


    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph


        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


class deeplab_xception_end2end_3d(deeplab_xception_transfer_basemodel_savememory):
    def __init__(self, nInputChannels=3, n_classes=20, os=16, input_channels=256, hidden_layers=128, out_channels=256,
                 source_classes=7, middle_classes=18, transfer_graph=None):
        super(deeplab_xception_end2end_3d, self).__init__(nInputChannels=nInputChannels,
                                                          n_classes=n_classes,
                                                          os=os, )
        ### source graph
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=source_classes)
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.source_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=source_classes
                                                                   )
        self.source_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])
        self.source_semantic = nn.Conv2d(out_channels,source_classes,1)
        self.middle_semantic = nn.Conv2d(out_channels, middle_classes, 1)

        ### target graph 1
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=n_classes)
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.target_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])

        ### middle
        self.middle_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=middle_classes)
        self.middle_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.middle_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.middle_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.middle_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.middle_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])

        ### multi transpose
        self.transpose_graph_source2target = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=source_classes, end_nodes=n_classes)
        self.transpose_graph_target2source = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=n_classes, end_nodes=source_classes)

        self.transpose_graph_middle2source = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=middle_classes, end_nodes=source_classes)
        self.transpose_graph_middle2target = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=middle_classes, end_nodes=source_classes)

        self.transpose_graph_source2middle = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=source_classes, end_nodes=middle_classes)
        self.transpose_graph_target2middle = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=n_classes, end_nodes=middle_classes)


        self.fc_graph_source = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)
        self.fc_graph_target = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)
        self.fc_graph_middle = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)

    def freeze_totally_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def freeze_backbone_bn(self):
        for m in self.xception_features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def top_forward(self, input, adj1_target=None, adj2_source=None, adj3_transfer_s2t=None, adj3_transfer_t2s=None,
            adj4_middle=None,adj5_transfer_s2m=None,adj6_transfer_t2m=None,adj5_transfer_m2s=None,adj6_transfer_m2t=None,):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### source graph
        source_graph = self.source_featuremap_2_graph(x)
        ### target source
        target_graph = self.target_featuremap_2_graph(x)
        ### middle source
        middle_graph = self.middle_featuremap_2_graph(x)

        ##### end2end multi task

        ### first task
        # print(source_graph.size(),target_graph.size())
        source_graph1 = self.source_graph_conv1.forward(source_graph, adj=adj2_source, relu=True)
        target_graph1 = self.target_graph_conv1.forward(target_graph, adj=adj1_target, relu=True)
        middle_graph1 = self.target_graph_conv1.forward(middle_graph, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph1_v5 = self.transpose_graph_source2target.forward(source_graph1, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph1_v5 = self.transpose_graph_source2middle.forward(source_graph1,adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph1_v5 = self.transpose_graph_target2source.forward(target_graph1, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph1_v5 = self.transpose_graph_target2middle.forward(target_graph1, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph1_v5 = self.transpose_graph_middle2source.forward(middle_graph1, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph1_v5 = self.transpose_graph_middle2target.forward(middle_graph1, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph1 = self.similarity_trans(source_graph1, target_graph1)
        source_2_middle_graph1 = self.similarity_trans(source_graph1, middle_graph1)
        # target 2 source middle
        target_2_source_graph1 = self.similarity_trans(target_graph1, source_graph1)
        target_2_middle_graph1 = self.similarity_trans(target_graph1, middle_graph1)
        # middle 2 source target
        middle_2_source_graph1 = self.similarity_trans(middle_graph1, source_graph1)
        middle_2_target_graph1 = self.similarity_trans(middle_graph1, target_graph1)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph1 = torch.cat(
            (source_graph1, target_2_source_graph1, target_2_source_graph1_v5,
             middle_2_source_graph1, middle_2_source_graph1_v5), dim=-1)
        source_graph1 = self.fc_graph_source.forward(source_graph1, relu=True)
        # target
        target_graph1 = torch.cat(
            (target_graph1, source_2_target_graph1, source_2_target_graph1_v5,
             middle_2_target_graph1, middle_2_target_graph1_v5), dim=-1)
        target_graph1 = self.fc_graph_target.forward(target_graph1, relu=True)
        # middle
        middle_graph1 = torch.cat((middle_graph1, source_2_middle_graph1, source_2_middle_graph1_v5,
                                   target_2_middle_graph1, target_2_middle_graph1_v5), dim=-1)
        middle_graph1 = self.fc_graph_middle.forward(middle_graph1, relu=True)


        ### seconde task
        source_graph2 = self.source_graph_conv1.forward(source_graph1, adj=adj2_source, relu=True)
        target_graph2 = self.target_graph_conv1.forward(target_graph1, adj=adj1_target, relu=True)
        middle_graph2 = self.target_graph_conv1.forward(middle_graph1, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph2_v5 = self.transpose_graph_source2target.forward(source_graph2, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph2_v5 = self.transpose_graph_source2middle.forward(source_graph2, adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph2_v5 = self.transpose_graph_target2source.forward(target_graph2, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph2_v5 = self.transpose_graph_target2middle.forward(target_graph2, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph2_v5 = self.transpose_graph_middle2source.forward(middle_graph2, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph2_v5 = self.transpose_graph_middle2target.forward(middle_graph2, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph2 = self.similarity_trans(source_graph2, target_graph2)
        source_2_middle_graph2 = self.similarity_trans(source_graph2, middle_graph2)
        # target 2 source middle
        target_2_source_graph2 = self.similarity_trans(target_graph2, source_graph2)
        target_2_middle_graph2 = self.similarity_trans(target_graph2, middle_graph2)
        # middle 2 source target
        middle_2_source_graph2 = self.similarity_trans(middle_graph2, source_graph2)
        middle_2_target_graph2 = self.similarity_trans(middle_graph2, target_graph2)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph2 = torch.cat(
            (source_graph2, target_2_source_graph2, target_2_source_graph2_v5,
             middle_2_source_graph2, middle_2_source_graph2_v5), dim=-1)
        source_graph2 = self.fc_graph_source.forward(source_graph2, relu=True)
        # target
        target_graph2 = torch.cat(
            (target_graph2, source_2_target_graph2, source_2_target_graph2_v5,
             middle_2_target_graph2, middle_2_target_graph2_v5), dim=-1)
        target_graph2 = self.fc_graph_target.forward(target_graph2, relu=True)
        # middle
        middle_graph2 = torch.cat((middle_graph2, source_2_middle_graph2, source_2_middle_graph2_v5,
                                   target_2_middle_graph2, target_2_middle_graph2_v5), dim=-1)
        middle_graph2 = self.fc_graph_middle.forward(middle_graph2, relu=True)


        ### third task
        source_graph3 = self.source_graph_conv1.forward(source_graph2, adj=adj2_source, relu=True)
        target_graph3 = self.target_graph_conv1.forward(target_graph2, adj=adj1_target, relu=True)
        middle_graph3 = self.target_graph_conv1.forward(middle_graph2, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph3_v5 = self.transpose_graph_source2target.forward(source_graph3, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph3_v5 = self.transpose_graph_source2middle.forward(source_graph3, adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph3_v5 = self.transpose_graph_target2source.forward(target_graph3, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph3_v5 = self.transpose_graph_target2middle.forward(target_graph3, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph3_v5 = self.transpose_graph_middle2source.forward(middle_graph3, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph3_v5 = self.transpose_graph_middle2target.forward(middle_graph3, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph3 = self.similarity_trans(source_graph3, target_graph3)
        source_2_middle_graph3 = self.similarity_trans(source_graph3, middle_graph3)
        # target 2 source middle
        target_2_source_graph3 = self.similarity_trans(target_graph3, source_graph3)
        target_2_middle_graph3 = self.similarity_trans(target_graph3, middle_graph3)
        # middle 2 source target
        middle_2_source_graph3 = self.similarity_trans(middle_graph3, source_graph3)
        middle_2_target_graph3 = self.similarity_trans(middle_graph3, target_graph3)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph3 = torch.cat(
            (source_graph3, target_2_source_graph3, target_2_source_graph3_v5,
             middle_2_source_graph3, middle_2_source_graph3_v5), dim=-1)
        source_graph3 = self.fc_graph_source.forward(source_graph3, relu=True)
        # target
        target_graph3 = torch.cat(
            (target_graph3, source_2_target_graph3, source_2_target_graph3_v5,
             middle_2_target_graph3, middle_2_target_graph3_v5), dim=-1)
        target_graph3 = self.fc_graph_target.forward(target_graph3, relu=True)
        # middle
        middle_graph3 = torch.cat((middle_graph3, source_2_middle_graph3, source_2_middle_graph3_v5,
                                   target_2_middle_graph3, target_2_middle_graph3_v5), dim=-1)
        middle_graph3 = self.fc_graph_middle.forward(middle_graph3, relu=True)

        return source_graph3, target_graph3, middle_graph3, x

    def similarity_trans(self,source,target):
        sim = torch.matmul(F.normalize(target, p=2, dim=-1), F.normalize(source, p=2, dim=-1).transpose(-1, -2))
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def bottom_forward_source(self, input, source_graph):
        # print('input size')
        # print(input.size())
        # print(source_graph.size())
        graph = self.source_graph_2_fea.forward(source_graph, input)
        x = self.source_skip_conv(input)
        x = x + graph
        x = self.source_semantic(x)
        return x

    def bottom_forward_target(self, input, target_graph):
        graph = self.target_graph_2_fea.forward(target_graph, input)
        x = self.target_skip_conv(input)
        x = x + graph
        x = self.semantic(x)
        return x

    def bottom_forward_middle(self, input, target_graph):
        graph = self.middle_graph_2_fea.forward(target_graph, input)
        x = self.middle_skip_conv(input)
        x = x + graph
        x = self.middle_semantic(x)
        return x

    def forward(self, input_source, input_target=None, input_middle=None, adj1_target=None, adj2_source=None,
                adj3_transfer_s2t=None, adj3_transfer_t2s=None, adj4_middle=None,adj5_transfer_s2m=None,
                adj6_transfer_t2m=None,adj5_transfer_m2s=None,adj6_transfer_m2t=None,):
        if input_source is None and input_target is not None and input_middle is None:
            # target
            target_batch = input_target.size(0)
            input = input_target

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target, adj2_source=adj2_source,
                                                             adj3_transfer_s2t=adj3_transfer_s2t,
                                                             adj3_transfer_t2s=adj3_transfer_t2s,
                                                           adj4_middle=adj4_middle,
                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            # source_x = self.bottom_forward_source(source_x, source_graph)
            target_x = self.bottom_forward_target(x, target_graph)

            target_x = F.upsample(target_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return None, target_x, None

        if input_source is not None and input_target is None and input_middle is None:
            # source
            source_batch = input_source.size(0)
            source_list = range(source_batch)
            input = input_source

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target,
                                                                           adj2_source=adj2_source,
                                                                           adj3_transfer_s2t=adj3_transfer_s2t,
                                                                           adj3_transfer_t2s=adj3_transfer_t2s,
                                                                           adj4_middle=adj4_middle,
                                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            source_x = self.bottom_forward_source(x, source_graph)
            source_x = F.upsample(source_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return source_x, None, None

        if input_middle is not None and input_source is None and input_target is None:
            # middle
            input = input_middle

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target,
                                                                           adj2_source=adj2_source,
                                                                           adj3_transfer_s2t=adj3_transfer_s2t,
                                                                           adj3_transfer_t2s=adj3_transfer_t2s,
                                                                           adj4_middle=adj4_middle,
                                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            middle_x = self.bottom_forward_middle(x, source_graph)
            middle_x = F.upsample(middle_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return None, None, middle_x


class deeplab_xception_end2end_3d_synbn(deeplab_xception_transfer_basemodel_savememory_synbn):
    def __init__(self, nInputChannels=3, n_classes=20, os=16, input_channels=256, hidden_layers=128, out_channels=256,
                 source_classes=7, middle_classes=18, transfer_graph=None):
        super(deeplab_xception_end2end_3d_synbn, self).__init__(nInputChannels=nInputChannels,
                                                                n_classes=n_classes,
                                                                os=os, )
        ### source graph
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=source_classes)
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.source_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=source_classes
                                                                   )
        self.source_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])
        self.source_semantic = nn.Conv2d(out_channels,source_classes,1)
        self.middle_semantic = nn.Conv2d(out_channels, middle_classes, 1)

        ### target graph 1
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=n_classes)
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.target_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])

        ### middle
        self.middle_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels,
                                                                  hidden_layers=hidden_layers,
                                                                  nodes=middle_classes)
        self.middle_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.middle_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.middle_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.middle_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels,
                                                                   output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.middle_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.ReLU(True)])

        ### multi transpose
        self.transpose_graph_source2target = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=source_classes, end_nodes=n_classes)
        self.transpose_graph_target2source = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=n_classes, end_nodes=source_classes)

        self.transpose_graph_middle2source = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=middle_classes, end_nodes=source_classes)
        self.transpose_graph_middle2target = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=middle_classes, end_nodes=source_classes)

        self.transpose_graph_source2middle = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=source_classes, end_nodes=middle_classes)
        self.transpose_graph_target2middle = gcn.Graph_trans(in_features=hidden_layers, out_features=hidden_layers,
                                                             adj=transfer_graph,
                                                             begin_nodes=n_classes, end_nodes=middle_classes)


        self.fc_graph_source = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)
        self.fc_graph_target = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)
        self.fc_graph_middle = gcn.GraphConvolution(hidden_layers * 5, hidden_layers)


    def top_forward(self, input, adj1_target=None, adj2_source=None, adj3_transfer_s2t=None, adj3_transfer_t2s=None,
            adj4_middle=None,adj5_transfer_s2m=None,adj6_transfer_t2m=None,adj5_transfer_m2s=None,adj6_transfer_m2t=None,):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### source graph
        source_graph = self.source_featuremap_2_graph(x)
        ### target source
        target_graph = self.target_featuremap_2_graph(x)
        ### middle source
        middle_graph = self.middle_featuremap_2_graph(x)

        ##### end2end multi task

        ### first task
        # print(source_graph.size(),target_graph.size())
        source_graph1 = self.source_graph_conv1.forward(source_graph, adj=adj2_source, relu=True)
        target_graph1 = self.target_graph_conv1.forward(target_graph, adj=adj1_target, relu=True)
        middle_graph1 = self.target_graph_conv1.forward(middle_graph, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph1_v5 = self.transpose_graph_source2target.forward(source_graph1, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph1_v5 = self.transpose_graph_source2middle.forward(source_graph1,adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph1_v5 = self.transpose_graph_target2source.forward(target_graph1, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph1_v5 = self.transpose_graph_target2middle.forward(target_graph1, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph1_v5 = self.transpose_graph_middle2source.forward(middle_graph1, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph1_v5 = self.transpose_graph_middle2target.forward(middle_graph1, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph1 = self.similarity_trans(source_graph1, target_graph1)
        source_2_middle_graph1 = self.similarity_trans(source_graph1, middle_graph1)
        # target 2 source middle
        target_2_source_graph1 = self.similarity_trans(target_graph1, source_graph1)
        target_2_middle_graph1 = self.similarity_trans(target_graph1, middle_graph1)
        # middle 2 source target
        middle_2_source_graph1 = self.similarity_trans(middle_graph1, source_graph1)
        middle_2_target_graph1 = self.similarity_trans(middle_graph1, target_graph1)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph1 = torch.cat(
            (source_graph1, target_2_source_graph1, target_2_source_graph1_v5,
             middle_2_source_graph1, middle_2_source_graph1_v5), dim=-1)
        source_graph1 = self.fc_graph_source.forward(source_graph1, relu=True)
        # target
        target_graph1 = torch.cat(
            (target_graph1, source_2_target_graph1, source_2_target_graph1_v5,
             middle_2_target_graph1, middle_2_target_graph1_v5), dim=-1)
        target_graph1 = self.fc_graph_target.forward(target_graph1, relu=True)
        # middle
        middle_graph1 = torch.cat((middle_graph1, source_2_middle_graph1, source_2_middle_graph1_v5,
                                   target_2_middle_graph1, target_2_middle_graph1_v5), dim=-1)
        middle_graph1 = self.fc_graph_middle.forward(middle_graph1, relu=True)


        ### seconde task
        source_graph2 = self.source_graph_conv1.forward(source_graph1, adj=adj2_source, relu=True)
        target_graph2 = self.target_graph_conv1.forward(target_graph1, adj=adj1_target, relu=True)
        middle_graph2 = self.target_graph_conv1.forward(middle_graph1, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph2_v5 = self.transpose_graph_source2target.forward(source_graph2, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph2_v5 = self.transpose_graph_source2middle.forward(source_graph2, adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph2_v5 = self.transpose_graph_target2source.forward(target_graph2, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph2_v5 = self.transpose_graph_target2middle.forward(target_graph2, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph2_v5 = self.transpose_graph_middle2source.forward(middle_graph2, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph2_v5 = self.transpose_graph_middle2target.forward(middle_graph2, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph2 = self.similarity_trans(source_graph2, target_graph2)
        source_2_middle_graph2 = self.similarity_trans(source_graph2, middle_graph2)
        # target 2 source middle
        target_2_source_graph2 = self.similarity_trans(target_graph2, source_graph2)
        target_2_middle_graph2 = self.similarity_trans(target_graph2, middle_graph2)
        # middle 2 source target
        middle_2_source_graph2 = self.similarity_trans(middle_graph2, source_graph2)
        middle_2_target_graph2 = self.similarity_trans(middle_graph2, target_graph2)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph2 = torch.cat(
            (source_graph2, target_2_source_graph2, target_2_source_graph2_v5,
             middle_2_source_graph2, middle_2_source_graph2_v5), dim=-1)
        source_graph2 = self.fc_graph_source.forward(source_graph2, relu=True)
        # target
        target_graph2 = torch.cat(
            (target_graph2, source_2_target_graph2, source_2_target_graph2_v5,
             middle_2_target_graph2, middle_2_target_graph2_v5), dim=-1)
        target_graph2 = self.fc_graph_target.forward(target_graph2, relu=True)
        # middle
        middle_graph2 = torch.cat((middle_graph2, source_2_middle_graph2, source_2_middle_graph2_v5,
                                   target_2_middle_graph2, target_2_middle_graph2_v5), dim=-1)
        middle_graph2 = self.fc_graph_middle.forward(middle_graph2, relu=True)


        ### third task
        source_graph3 = self.source_graph_conv1.forward(source_graph2, adj=adj2_source, relu=True)
        target_graph3 = self.target_graph_conv1.forward(target_graph2, adj=adj1_target, relu=True)
        middle_graph3 = self.target_graph_conv1.forward(middle_graph2, adj=adj4_middle, relu=True)

        # source 2 target & middle
        source_2_target_graph3_v5 = self.transpose_graph_source2target.forward(source_graph3, adj=adj3_transfer_s2t,
                                                                               relu=True)
        source_2_middle_graph3_v5 = self.transpose_graph_source2middle.forward(source_graph3, adj=adj5_transfer_s2m,
                                                                               relu=True)
        # target 2 source & middle
        target_2_source_graph3_v5 = self.transpose_graph_target2source.forward(target_graph3, adj=adj3_transfer_t2s,
                                                                               relu=True)
        target_2_middle_graph3_v5 = self.transpose_graph_target2middle.forward(target_graph3, adj=adj6_transfer_t2m,
                                                                               relu=True)
        # middle 2 source & target
        middle_2_source_graph3_v5 = self.transpose_graph_middle2source.forward(middle_graph3, adj=adj5_transfer_m2s,
                                                                               relu=True)
        middle_2_target_graph3_v5 = self.transpose_graph_middle2target.forward(middle_graph3, adj=adj6_transfer_m2t,
                                                                               relu=True)
        # source 2 middle target
        source_2_target_graph3 = self.similarity_trans(source_graph3, target_graph3)
        source_2_middle_graph3 = self.similarity_trans(source_graph3, middle_graph3)
        # target 2 source middle
        target_2_source_graph3 = self.similarity_trans(target_graph3, source_graph3)
        target_2_middle_graph3 = self.similarity_trans(target_graph3, middle_graph3)
        # middle 2 source target
        middle_2_source_graph3 = self.similarity_trans(middle_graph3, source_graph3)
        middle_2_target_graph3 = self.similarity_trans(middle_graph3, target_graph3)

        ## concat
        # print(source_graph1.size(), target_2_source_graph1.size(), )
        source_graph3 = torch.cat(
            (source_graph3, target_2_source_graph3, target_2_source_graph3_v5,
             middle_2_source_graph3, middle_2_source_graph3_v5), dim=-1)
        source_graph3 = self.fc_graph_source.forward(source_graph3, relu=True)
        # target
        target_graph3 = torch.cat(
            (target_graph3, source_2_target_graph3, source_2_target_graph3_v5,
             middle_2_target_graph3, middle_2_target_graph3_v5), dim=-1)
        target_graph3 = self.fc_graph_target.forward(target_graph3, relu=True)
        # middle
        middle_graph3 = torch.cat((middle_graph3, source_2_middle_graph3, source_2_middle_graph3_v5,
                                   target_2_middle_graph3, target_2_middle_graph3_v5), dim=-1)
        middle_graph3 = self.fc_graph_middle.forward(middle_graph3, relu=True)

        return source_graph3, target_graph3, middle_graph3, x

    def similarity_trans(self,source,target):
        sim = torch.matmul(F.normalize(target, p=2, dim=-1), F.normalize(source, p=2, dim=-1).transpose(-1, -2))
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def bottom_forward_source(self, input, source_graph):
        # print('input size')
        # print(input.size())
        # print(source_graph.size())
        graph = self.source_graph_2_fea.forward(source_graph, input)
        x = self.source_skip_conv(input)
        x = x + graph
        x = self.source_semantic(x)
        return x

    def bottom_forward_target(self, input, target_graph):
        graph = self.target_graph_2_fea.forward(target_graph, input)
        x = self.target_skip_conv(input)
        x = x + graph
        x = self.semantic(x)
        return x

    def bottom_forward_middle(self, input, target_graph):
        graph = self.middle_graph_2_fea.forward(target_graph, input)
        x = self.middle_skip_conv(input)
        x = x + graph
        x = self.middle_semantic(x)
        return x

    def forward(self, input_source, input_target=None, input_middle=None, adj1_target=None, adj2_source=None,
                adj3_transfer_s2t=None, adj3_transfer_t2s=None, adj4_middle=None,adj5_transfer_s2m=None,
                adj6_transfer_t2m=None,adj5_transfer_m2s=None,adj6_transfer_m2t=None,):

        if input_source is None and input_target is not None and input_middle is None:
            # target
            target_batch = input_target.size(0)
            input = input_target

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target, adj2_source=adj2_source,
                                                             adj3_transfer_s2t=adj3_transfer_s2t,
                                                             adj3_transfer_t2s=adj3_transfer_t2s,
                                                           adj4_middle=adj4_middle,
                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            # source_x = self.bottom_forward_source(source_x, source_graph)
            target_x = self.bottom_forward_target(x, target_graph)

            target_x = F.upsample(target_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return None, target_x, None

        if input_source is not None and input_target is None and input_middle is None:
            # source
            source_batch = input_source.size(0)
            source_list = range(source_batch)
            input = input_source

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target,
                                                                           adj2_source=adj2_source,
                                                                           adj3_transfer_s2t=adj3_transfer_s2t,
                                                                           adj3_transfer_t2s=adj3_transfer_t2s,
                                                                           adj4_middle=adj4_middle,
                                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            source_x = self.bottom_forward_source(x, source_graph)
            source_x = F.upsample(source_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return source_x, None, None

        if input_middle is not None and input_source is None and input_target is None:
            # middle
            input = input_middle

            source_graph, target_graph, middle_graph, x = self.top_forward(input, adj1_target=adj1_target,
                                                                           adj2_source=adj2_source,
                                                                           adj3_transfer_s2t=adj3_transfer_s2t,
                                                                           adj3_transfer_t2s=adj3_transfer_t2s,
                                                                           adj4_middle=adj4_middle,
                                                                           adj5_transfer_s2m=adj5_transfer_s2m,
                                                                           adj6_transfer_t2m=adj6_transfer_t2m,
                                                                           adj5_transfer_m2s=adj5_transfer_m2s,
                                                                           adj6_transfer_m2t=adj6_transfer_m2t)

            middle_x = self.bottom_forward_middle(x, source_graph)
            middle_x = F.upsample(middle_x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return None, None, middle_x


if __name__ == '__main__':
    net = deeplab_xception_end2end_3d()
    net.freeze_totally_bn()
    img1 = torch.rand((1,3,128,128))
    img2 = torch.rand((1, 3, 128, 128))
    a1 = torch.ones((1,1,7,20))
    a2 = torch.ones((1,1,20,7))
    net.eval()
    net.forward(img1,img2,adj3_transfer_t2s=a2,adj3_transfer_s2t=a1)
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from . import graph
# import pdb

class GraphConvolution(nn.Module):

    def __init__(self,in_features,out_features,bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1./math.sqrt(self.weight(1))
        # self.weight.data.uniform_(-stdv,stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv,stdv)

    def forward(self, input,adj=None,relu=False):
        support = torch.matmul(input, self.weight)
        # print(support.size(),adj.size())
        if adj is not None:
            output = torch.matmul(adj, support)
        else:
            output = support
        # print(output.size())
        if self.bias is not None:
            return output + self.bias
        else:
            if relu:
                return F.relu(output)
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Featuremaps_to_Graph(nn.Module):

    def __init__(self,input_channels,hidden_layers,nodes=7):
        super(Featuremaps_to_Graph, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels,nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels,hidden_layers))
        self.reset_parameters()

    def forward(self, input):
        n,c,h,w = input.size()
        # print('fea input',input.size())
        input1 = input.view(n,c,h*w)
        input1 = input1.transpose(1,2) # n x hw x c
        # print('fea input1', input1.size())
        ############## Feature maps to node ################
        fea_node = torch.matmul(input1,self.pre_fea) # n x hw x n_classes
        weight_node = torch.matmul(input1,self.weight) # n x hw x hidden_layer
        # softmax fea_node
        fea_node = F.softmax(fea_node,dim=-1)
        # print(fea_node.size(),weight_node.size())
        graph_node = F.relu(torch.matmul(fea_node.transpose(1,2),weight_node))
        return graph_node # n x n_class x hidden_layer

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv,stdv)

class Featuremaps_to_Graph_transfer(nn.Module):

    def __init__(self,input_channels,hidden_layers,nodes=7, source_nodes=20):
        super(Featuremaps_to_Graph_transfer, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels,nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels,hidden_layers))
        self.pre_fea_transfer = nn.Sequential(*[nn.Linear(source_nodes, source_nodes),nn.LeakyReLU(True),
                                                nn.Linear(source_nodes, nodes), nn.LeakyReLU(True)])
        self.reset_parameters()

    def forward(self, input, source_pre_fea):
        self.pre_fea.data = self.pre_fea_learn(source_pre_fea)
        n,c,h,w = input.size()
        # print('fea input',input.size())
        input1 = input.view(n,c,h*w)
        input1 = input1.transpose(1,2) # n x hw x c
        # print('fea input1', input1.size())
        ############## Feature maps to node ################
        fea_node = torch.matmul(input1,self.pre_fea) # n x hw x n_classes
        weight_node = torch.matmul(input1,self.weight) # n x hw x hidden_layer
        # softmax fea_node
        fea_node = F.softmax(fea_node,dim=1)
        # print(fea_node.size(),weight_node.size())
        graph_node = F.relu(torch.matmul(fea_node.transpose(1,2),weight_node))
        return graph_node # n x n_class x hidden_layer

    def pre_fea_learn(self, input):
        pre_fea = self.pre_fea_transfer.forward(input.unsqueeze(0)).squeeze(0)
        return self.pre_fea.data + pre_fea

class Graph_to_Featuremaps(nn.Module):
    # this is a special version
    def __init__(self,input_channels,output_channels,hidden_layers,nodes=7):
        super(Graph_to_Featuremaps, self).__init__()
        self.node_fea = Parameter(torch.FloatTensor(input_channels+hidden_layers,1))
        self.weight = Parameter(torch.FloatTensor(hidden_layers,output_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input, res_feature):
        '''

        :param input: 1 x batch x nodes x hidden_layer
        :param res_feature: batch x channels x h x w
        :return:
        '''
        batchi,channeli,hi,wi = res_feature.size()
        # print(res_feature.size())
        # print(input.size())
        try:
            _,batch,nodes,hidden = input.size()
        except:
            # print(input.size())
            input = input.unsqueeze(0)
            _,batch, nodes, hidden = input.size()

        assert batch == batchi
        input1 = input.transpose(0,1).expand(batch,hi*wi,nodes,hidden)
        res_feature_after_view = res_feature.view(batch,channeli,hi*wi).transpose(1,2)
        res_feature_after_view1 = res_feature_after_view.unsqueeze(2).expand(batch,hi*wi,nodes,channeli)
        new_fea = torch.cat((res_feature_after_view1,input1),dim=3)

        # print(self.node_fea.size(),new_fea.size())
        new_node = torch.matmul(new_fea, self.node_fea) # batch x hw x nodes x 1
        new_weight = torch.matmul(input, self.weight)  # batch x node x channel
        new_node = new_node.view(batch, hi*wi, nodes)
        # 0721
        new_node = F.softmax(new_node, dim=-1)
        #
        feature_out = torch.matmul(new_node,new_weight)
        # print(feature_out.size())
        feature_out = feature_out.transpose(2,3).contiguous().view(res_feature.size())
        return F.relu(feature_out)

class Graph_to_Featuremaps_savemem(nn.Module):
    # this is a special version for saving gpu memory. The process is same as Graph_to_Featuremaps.
    def __init__(self, input_channels, output_channels, hidden_layers, nodes=7):
        super(Graph_to_Featuremaps_savemem, self).__init__()
        self.node_fea_for_res = Parameter(torch.FloatTensor(input_channels, 1))
        self.node_fea_for_hidden = Parameter(torch.FloatTensor(hidden_layers, 1))
        self.weight = Parameter(torch.FloatTensor(hidden_layers,output_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input, res_feature):
        '''

        :param input: 1 x batch x nodes x hidden_layer
        :param res_feature: batch x channels x h x w
        :return:
        '''
        batchi,channeli,hi,wi = res_feature.size()
        # print(res_feature.size())
        # print(input.size())
        try:
            _,batch,nodes,hidden = input.size()
        except:
            # print(input.size())
            input = input.unsqueeze(0)
            _,batch, nodes, hidden = input.size()

        assert batch == batchi
        input1 = input.transpose(0,1).expand(batch,hi*wi,nodes,hidden)
        res_feature_after_view = res_feature.view(batch,channeli,hi*wi).transpose(1,2)
        res_feature_after_view1 = res_feature_after_view.unsqueeze(2).expand(batch,hi*wi,nodes,channeli)
        # new_fea = torch.cat((res_feature_after_view1,input1),dim=3)
        ## sim
        new_node1 = torch.matmul(res_feature_after_view1, self.node_fea_for_res)
        new_node2 = torch.matmul(input1, self.node_fea_for_hidden)
        new_node = new_node1 + new_node2
        ## sim end
        # print(self.node_fea.size(),new_fea.size())
        # new_node = torch.matmul(new_fea, self.node_fea) # batch x hw x nodes x 1
        new_weight = torch.matmul(input, self.weight) # batch x node x channel
        new_node = new_node.view(batch, hi*wi, nodes)
        # 0721
        new_node = F.softmax(new_node, dim=-1)
        #
        feature_out = torch.matmul(new_node,new_weight)
        # print(feature_out.size())
        feature_out = feature_out.transpose(2,3).contiguous().view(res_feature.size())
        return F.relu(feature_out)


class Graph_trans(nn.Module):

    def __init__(self,in_features,out_features,begin_nodes=7,end_nodes=2,bias=False,adj=None):
        super(Graph_trans, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if adj is not None:
            h,w = adj.size()
            assert (h == end_nodes) and (w == begin_nodes)
            self.adj = torch.autograd.Variable(adj,requires_grad=False)
        else:
            self.adj = Parameter(torch.FloatTensor(end_nodes,begin_nodes))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        # self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1./math.sqrt(self.weight(1))
        # self.weight.data.uniform_(-stdv,stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv,stdv)

    def forward(self, input, relu=False, adj_return=False, adj=None):
        support = torch.matmul(input,self.weight)
        # print(support.size(),self.adj.size())
        if adj is None:
            adj = self.adj
        adj1 = self.norm_trans_adj(adj)
        output = torch.matmul(adj1,support)
        if adj_return:
            output1 = F.normalize(output,p=2,dim=-1)
            self.adj_mat = torch.matmul(output1,output1.transpose(-2,-1))
        if self.bias is not None:
            return output + self.bias
        else:
            if relu:
                return F.relu(output)
            else:
                return output

    def get_adj_mat(self):
        adj = graph.normalize_adj_torch(F.relu(self.adj_mat))
        return adj

    def get_encode_adj(self):
        return self.adj

    def norm_trans_adj(self,adj):  # maybe can use softmax
        adj = F.relu(adj)
        r = F.softmax(adj,dim=-1)
        # print(adj.size())
        # row_sum = adj.sum(-1).unsqueeze(-1)
        # d_mat = row_sum.expand(adj.size())
        # r = torch.div(row_sum,d_mat)
        # r[torch.isnan(r)] = 0

        return r


if __name__ == '__main__':

    graph = torch.randn((7,128))
    en = GraphConvolution(128,128)
    a = en.forward(graph)
    print(a)
    # a = en.forward(graph,pred)
    # print(a.size())
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch

pascal_graph = {0:[0],
                1:[1, 2],
                2:[1, 2, 3, 5],
                3:[2, 3, 4],
                4:[3, 4],
                5:[2, 5, 6],
                6:[5, 6]}

cihp_graph = {0: [],
              1: [2, 13],
              2: [1, 13],
              3: [14, 15],
              4: [13],
              5: [6, 7, 9, 10, 11, 12, 14, 15],
              6: [5, 7, 10, 11, 14, 15, 16, 17],
              7: [5, 6, 9, 10, 11, 12, 14, 15],
              8: [16, 17, 18, 19],
              9: [5, 7, 10, 16, 17, 18, 19],
              10:[5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17],
              11:[5, 6, 7, 10, 13],
              12:[5, 7, 10, 16, 17],
              13:[1, 2, 4, 10, 11],
              14:[3, 5, 6, 7, 10],
              15:[3, 5, 6, 7, 10],
              16:[6, 8, 9, 10, 12, 18],
              17:[6, 8, 9, 10, 12, 19],
              18:[8, 9, 16],
              19:[8, 9, 17]}

atr_graph = {0: [],
              1: [2, 11],
              2: [1, 11],
              3: [11],
              4: [5, 6, 7, 11, 14, 15, 17],
              5: [4, 6, 7, 8, 12, 13],
              6: [4,5,7,8,9,10,12,13],
              7: [4,11,12,13,14,15],
              8: [5,6],
              9: [6, 12],
              10:[6, 13],
              11:[1,2,3,4,7,14,15,17],
              12:[5,6,7,9],
              13:[5,6,7,10],
              14:[4,7,11,16],
              15:[4,7,11,16],
              16:[14,15],
              17:[4,11],
              }

cihp2pascal_adj = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

cihp2pascal_nlp_adj = \
    np.array([[ 1.,  0.35333052,  0.32727194,  0.17418084,  0.18757584,
         0.40608522,  0.37503981,  0.35448462,  0.22598555,  0.23893579,
         0.33064262,  0.28923404,  0.27986573,  0.4211553 ,  0.36915778,
         0.41377746,  0.32485771,  0.37248222,  0.36865639,  0.41500332],
       [ 0.39615879,  0.46201529,  0.52321467,  0.30826114,  0.25669527,
         0.54747773,  0.3670523 ,  0.3901983 ,  0.27519473,  0.3433325 ,
         0.52728509,  0.32771333,  0.34819325,  0.63882953,  0.68042925,
         0.69368576,  0.63395791,  0.65344337,  0.59538781,  0.6071375 ],
       [ 0.16373166,  0.21663339,  0.3053872 ,  0.28377612,  0.1372435 ,
         0.4448808 ,  0.29479995,  0.31092595,  0.22703953,  0.33983576,
         0.75778818,  0.2619818 ,  0.37069392,  0.35184867,  0.49877512,
         0.49979437,  0.51853277,  0.52517541,  0.32517741,  0.32377309],
       [ 0.32687232,  0.38482461,  0.37693463,  0.41610834,  0.20415749,
         0.76749079,  0.35139853,  0.3787411 ,  0.28411737,  0.35155421,
         0.58792618,  0.31141718,  0.40585111,  0.51189218,  0.82042737,
         0.8342413 ,  0.70732188,  0.72752501,  0.60327325,  0.61431337],
       [ 0.34069369,  0.34817292,  0.37525998,  0.36497069,  0.17841617,
         0.69746208,  0.31731463,  0.34628951,  0.25167277,  0.32072379,
         0.56711286,  0.24894776,  0.37000453,  0.52600859,  0.82483993,
         0.84966274,  0.7033991 ,  0.73449378,  0.56649608,  0.58888791],
       [ 0.28477487,  0.35139564,  0.42742352,  0.41664321,  0.20004676,
         0.78566833,  0.42237487,  0.41048549,  0.37933812,  0.46542516,
         0.62444759,  0.3274493 ,  0.49466009,  0.49314658,  0.71244233,
         0.71497003,  0.8234787 ,  0.83566589,  0.62597135,  0.62626812],
       [ 0.3011378 ,  0.31775977,  0.42922647,  0.36896257,  0.17597556,
         0.72214655,  0.39162804,  0.38137872,  0.34980296,  0.43818419,
         0.60879174,  0.26762545,  0.46271161,  0.51150476,  0.72318109,
         0.73678399,  0.82620388,  0.84942166,  0.5943811 ,  0.60607602]])

pascal2atr_nlp_adj = \
    np.array([[ 1.,  0.35333052,  0.32727194,  0.18757584,  0.40608522,
         0.27986573,  0.23893579,  0.27600672,  0.30964391,  0.36865639,
         0.41500332,  0.4211553 ,  0.32485771,  0.37248222,  0.36915778,
         0.41377746,  0.32006291,  0.28923404],
       [ 0.39615879,  0.46201529,  0.52321467,  0.25669527,  0.54747773,
         0.34819325,  0.3433325 ,  0.26603942,  0.45162929,  0.59538781,
         0.6071375 ,  0.63882953,  0.63395791,  0.65344337,  0.68042925,
         0.69368576,  0.44354613,  0.32771333],
       [ 0.16373166,  0.21663339,  0.3053872 ,  0.1372435 ,  0.4448808 ,
         0.37069392,  0.33983576,  0.26563416,  0.35443504,  0.32517741,
         0.32377309,  0.35184867,  0.51853277,  0.52517541,  0.49877512,
         0.49979437,  0.21750868,  0.2619818 ],
       [ 0.32687232,  0.38482461,  0.37693463,  0.20415749,  0.76749079,
         0.40585111,  0.35155421,  0.28271333,  0.52684576,  0.60327325,
         0.61431337,  0.51189218,  0.70732188,  0.72752501,  0.82042737,
         0.8342413 ,  0.40137029,  0.31141718],
       [ 0.34069369,  0.34817292,  0.37525998,  0.17841617,  0.69746208,
         0.37000453,  0.32072379,  0.27268885,  0.47426719,  0.56649608,
         0.58888791,  0.52600859,  0.7033991 ,  0.73449378,  0.82483993,
         0.84966274,  0.37830796,  0.24894776],
       [ 0.28477487,  0.35139564,  0.42742352,  0.20004676,  0.78566833,
         0.49466009,  0.46542516,  0.32662614,  0.55780359,  0.62597135,
         0.62626812,  0.49314658,  0.8234787 ,  0.83566589,  0.71244233,
         0.71497003,  0.41223219,  0.3274493 ],
       [ 0.3011378 ,  0.31775977,  0.42922647,  0.17597556,  0.72214655,
         0.46271161,  0.43818419,  0.3192333 ,  0.50979216,  0.5943811 ,
         0.60607602,  0.51150476,  0.82620388,  0.84942166,  0.72318109,
         0.73678399,  0.39259827,  0.26762545]])

cihp2atr_nlp_adj = np.array([[ 1.,  0.35333052,  0.32727194,  0.18757584,  0.40608522,
         0.27986573,  0.23893579,  0.27600672,  0.30964391,  0.36865639,
         0.41500332,  0.4211553 ,  0.32485771,  0.37248222,  0.36915778,
         0.41377746,  0.32006291,  0.28923404],
       [ 0.35333052,  1.        ,  0.39206695,  0.42143438,  0.4736689 ,
         0.47139544,  0.51999208,  0.38354847,  0.45628529,  0.46514124,
         0.50083501,  0.4310595 ,  0.39371443,  0.4319752 ,  0.42938598,
         0.46384034,  0.44833757,  0.6153155 ],
       [ 0.32727194,  0.39206695,  1.        ,  0.32836702,  0.52603065,
         0.39543695,  0.3622627 ,  0.43575346,  0.33866223,  0.45202552,
         0.48421   ,  0.53669903,  0.47266611,  0.50925436,  0.42286557,
         0.45403656,  0.37221304,  0.40999322],
       [ 0.17418084,  0.46892601,  0.25774838,  0.31816231,  0.39330317,
         0.34218382,  0.48253904,  0.22084125,  0.41335728,  0.52437572,
         0.5191713 ,  0.33576117,  0.44230914,  0.44250678,  0.44330833,
         0.43887264,  0.50693611,  0.39278795],
       [ 0.18757584,  0.42143438,  0.32836702,  1.        ,  0.35030067,
         0.30110947,  0.41055555,  0.34338879,  0.34336307,  0.37704433,
         0.38810141,  0.34702081,  0.24171562,  0.25433078,  0.24696241,
         0.2570884 ,  0.4465962 ,  0.45263213],
       [ 0.40608522,  0.4736689 ,  0.52603065,  0.35030067,  1.        ,
         0.54372584,  0.58300258,  0.56674191,  0.555266  ,  0.66599594,
         0.68567555,  0.55716359,  0.62997328,  0.65638548,  0.61219615,
         0.63183318,  0.54464151,  0.44293752],
       [ 0.37503981,  0.50675565,  0.4761106 ,  0.37561813,  0.60419403,
         0.77912403,  0.64595517,  0.85939662,  0.46037144,  0.52348817,
         0.55875094,  0.37741886,  0.455671  ,  0.49434392,  0.38479954,
         0.41804074,  0.47285709,  0.57236283],
       [ 0.35448462,  0.50576632,  0.51030446,  0.35841033,  0.55106903,
         0.50257274,  0.52591451,  0.4283053 ,  0.39991808,  0.42327211,
         0.42853819,  0.42071825,  0.41240559,  0.42259136,  0.38125352,
         0.3868255 ,  0.47604934,  0.51811717],
       [ 0.22598555,  0.5053299 ,  0.36301185,  0.38002282,  0.49700941,
         0.45625243,  0.62876479,  0.4112051 ,  0.33944371,  0.48322639,
         0.50318714,  0.29207815,  0.38801966,  0.41119094,  0.29199072,
         0.31021029,  0.41594871,  0.54961962],
       [ 0.23893579,  0.51999208,  0.3622627 ,  0.41055555,  0.58300258,
         0.68874251,  1.        ,  0.56977937,  0.49918447,  0.48484363,
         0.51615925,  0.41222306,  0.49535971,  0.53134951,  0.3807616 ,
         0.41050298,  0.48675801,  0.51112664],
       [ 0.33064262,  0.306412  ,  0.60679935,  0.25592294,  0.58738706,
         0.40379627,  0.39679161,  0.33618385,  0.39235148,  0.45474013,
         0.4648476 ,  0.59306762,  0.58976007,  0.60778661,  0.55400397,
         0.56551297,  0.3698029 ,  0.33860535],
       [ 0.28923404,  0.6153155 ,  0.40999322,  0.45263213,  0.44293752,
         0.60359359,  0.51112664,  0.46578181,  0.45656936,  0.38142307,
         0.38525582,  0.33327223,  0.35360175,  0.36156453,  0.3384992 ,
         0.34261229,  0.49297863,  1.        ],
       [ 0.27986573,  0.47139544,  0.39543695,  0.30110947,  0.54372584,
         1.        ,  0.68874251,  0.67765588,  0.48690078,  0.44010641,
         0.44921156,  0.32321099,  0.48311542,  0.4982002 ,  0.39378102,
         0.40297733,  0.45309735,  0.60359359],
       [ 0.4211553 ,  0.4310595 ,  0.53669903,  0.34702081,  0.55716359,
         0.32321099,  0.41222306,  0.25721705,  0.36633509,  0.5397475 ,
         0.56429928,  1.        ,  0.55796926,  0.58842844,  0.57930828,
         0.60410597,  0.41615326,  0.33327223],
       [ 0.36915778,  0.42938598,  0.42286557,  0.24696241,  0.61219615,
         0.39378102,  0.3807616 ,  0.28089866,  0.48450394,  0.77400821,
         0.68813814,  0.57930828,  0.8856886 ,  0.81673412,  1.        ,
         0.92279623,  0.46969152,  0.3384992 ],
       [ 0.41377746,  0.46384034,  0.45403656,  0.2570884 ,  0.63183318,
         0.40297733,  0.41050298,  0.332879  ,  0.48799542,  0.69231828,
         0.77015091,  0.60410597,  0.79788484,  0.88232104,  0.92279623,
         1.        ,  0.45685017,  0.34261229],
       [ 0.32485771,  0.39371443,  0.47266611,  0.24171562,  0.62997328,
         0.48311542,  0.49535971,  0.32477932,  0.51486622,  0.79353556,
         0.69768738,  0.55796926,  1.        ,  0.92373745,  0.8856886 ,
         0.79788484,  0.47883134,  0.35360175],
       [ 0.37248222,  0.4319752 ,  0.50925436,  0.25433078,  0.65638548,
         0.4982002 ,  0.53134951,  0.38057074,  0.52403969,  0.72035243,
         0.78711147,  0.58842844,  0.92373745,  1.        ,  0.81673412,
         0.88232104,  0.47109935,  0.36156453],
       [ 0.36865639,  0.46514124,  0.45202552,  0.37704433,  0.66599594,
         0.44010641,  0.48484363,  0.39636574,  0.50175258,  1.        ,
         0.91320249,  0.5397475 ,  0.79353556,  0.72035243,  0.77400821,
         0.69231828,  0.59087008,  0.38142307],
       [ 0.41500332,  0.50083501,  0.48421,  0.38810141,  0.68567555,
         0.44921156,  0.51615925,  0.45156472,  0.50438158,  0.91320249,
         1.,  0.56429928,  0.69768738,  0.78711147,  0.68813814,
         0.77015091,  0.57698754,  0.38525582]])



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj)) # return a adjacency matrix of adj ( type is numpy)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) #
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.todense()

def row_norm(inputs):
    outputs = []
    for x in inputs:
        xsum = x.sum()
        x = x / xsum
        outputs.append(x)
    return outputs


def normalize_adj_torch(adj):
    # print(adj.size())
    if len(adj.size()) == 4:
        new_r = torch.zeros(adj.size()).type_as(adj)
        for i in range(adj.size(1)):
            adj_item = adj[0,i]
            rowsum = adj_item.sum(1)
            d_inv_sqrt = rowsum.pow_(-0.5)
            d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            r = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_item), d_mat_inv_sqrt)
            new_r[0,i,...] = r
        return new_r
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum.pow_(-0.5)
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    r = torch.matmul(torch.matmul(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)
    return r

# def row_norm(adj):




if __name__ == '__main__':
    a= row_norm(cihp2pascal_adj)
    print(a)
    print(cihp2pascal_adj)
    # print(a.shape)

from .deeplab_xception import *
from .deeplab_xception_transfer import *
from .deeplab_xception_universal import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from collections import OrderedDict
from ..sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback, SynchronizedBatchNorm2d


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConv2d_aspp(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0):
        super(SeparableConv2d_aspp, self).__init__()

        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        self.depthwise_bn = SynchronizedBatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.pointwise_bn = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        #         x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x

class Decoder_module(nn.Module):
    def __init__(self, inplanes, planes, rate=1):
        super(Decoder_module, self).__init__()
        self.atrous_convolution = SeparableConv2d_aspp(inplanes, planes, 3, stride=1, dilation=rate,padding=1)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            raise RuntimeError()
        else:
            kernel_size = 3
            padding = rate
            self.atrous_convolution = SeparableConv2d_aspp(inplanes, planes, 3, stride=1, dilation=rate,
                                                           padding=padding)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x


class ASPP_module_rate0(nn.Module):
    def __init__(self, inplanes, planes, rate=1):
        super(ASPP_module_rate0, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
            self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                                stride=1, padding=padding, dilation=rate, bias=False)
            self.bn = SynchronizedBatchNorm2d(planes, eps=1e-5, affine=True)
            self.relu = nn.ReLU()
        else:
            raise RuntimeError()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0):
        super(SeparableConv2d_same, self).__init__()

        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        self.depthwise_bn = SynchronizedBatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.pointwise_bn = SynchronizedBatchNorm2d(planes)

    def forward(self, x):
        x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=2, bias=False)
            if is_last:
                self.skip = nn.Conv2d(inplanes, planes, 1, stride=1, bias=False)
            self.skipbn = SynchronizedBatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2,dilation=dilation))

        if is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1,dilation=dilation))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        # print(x.size(),skip.size())
        x += skip

        return x

class Block2(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block2, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = SynchronizedBatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            self.block2_lastconv = nn.Sequential(*[self.relu,SeparableConv2d_same(planes, planes, 3, stride=2,dilation=dilation)])

        if is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        low_middle = x.clone()
        x1 = x
        x1 = self.block2_lastconv(x1)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x1 += skip

        return x1,low_middle

class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block2(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d_aspp(1024, 1536, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d_aspp(1536, 1536, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_aspp(1536, 2048, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn5 = nn.BatchNorm2d(2048)

        # Init weights
        # self.__init_weight()

        # Load pretrained model
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('conv1 ',x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # print('block1',x.size())
        # low_level_feat = x
        x,low_level_feat = self.block2(x)
        # print('block2',x.size())
        x = self.block3(x)
        # print('xception block3 ',x.size())

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        # x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.xception_features = Xception(nInputChannels, os, pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module_rate0(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             SynchronizedBatchNorm2d(256),
                                             nn.ReLU()
                                             )

        self.concat_projection_conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.concat_projection_bn1 = SynchronizedBatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.feature_projection_conv1 = nn.Conv2d(256, 48, 1, bias=False)
        self.feature_projection_bn1 = SynchronizedBatchNorm2d(48)

        self.decoder = nn.Sequential(Decoder_module(304, 256),
                                     Decoder_module(256, 256)
                                     )
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)

    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.xception_features.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m,SynchronizedBatchNorm2d):
                m.eval()

    def freeze_aspp_bn(self):
        for m in self.aspp1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp4.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def learnable_parameters(self):
        layer_features_BN = []
        layer_features = []
        layer_aspp = []
        layer_projection  =[]
        layer_decoder = []
        layer_other = []
        model_para = list(self.named_parameters())
        for name,para in model_para:
            if 'xception' in name:
                if 'bn' in name or 'downsample.1.weight' in name or 'downsample.1.bias' in name:
                    layer_features_BN.append(para)
                else:
                    layer_features.append(para)
                    # print (name)
            elif 'aspp' in name:
                layer_aspp.append(para)
            elif 'projection' in name:
                layer_projection.append(para)
            elif 'decode' in name:
                layer_decoder.append(para)
            else:
                layer_other.append(para)
        return layer_features_BN,layer_features,layer_aspp,layer_projection,layer_decoder,layer_other


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_state_dict_new(self, state_dict):
        own_state = self.state_dict()
        #for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.','')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print ('unexpected key "{}" in state_dict'
                       .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                        name, own_state[name].size(), param.size()))
                continue # i add inshop_cos 2018/02/01
                # raise
                    # print 'copying %s' %name
                # if isinstance(param, own_state):
                # backwards compatibility for serialized parameters
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))




def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.xception_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True)
    model.eval()
    # ckt = torch.load('C:\\Users\gaoyi\code_python\deeplab_v3plus.pth')
    # model.load_state_dict_new(ckt)


    image = torch.randn(1, 3, 512, 512)*255
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())
    # print(output)







import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from collections import OrderedDict
from torch.nn import Parameter
from . import deeplab_xception,gcn, deeplab_xception_synBN
import pdb

#######################
# base model
#######################

class deeplab_xception_transfer_basemodel(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256):
        super(deeplab_xception_transfer_basemodel, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                  os=os,)
        ### source graph
        # self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
        #                                                    nodes=n_classes)
        # self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        #
        # self.source_graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels, output_channels=out_channels,
        #                                             hidden_layers=hidden_layers, nodes=n_classes
        #                                             )
        # self.source_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
        #                                  nn.ReLU(True)])

        ### target graph
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels, output_channels=out_channels,
                                                    hidden_layers=hidden_layers, nodes=n_classes
                                                    )
        self.target_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                         nn.ReLU(True)])

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_graph' not in name and 'transpose_graph' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if 'target' in name or 'semantic' in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'source' in name:
                l.append(k)
        return l

    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph


        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

class deeplab_xception_transfer_basemodel_savememory(deeplab_xception.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256):
        super(deeplab_xception_transfer_basemodel_savememory, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                  os=os,)
        ### source graph

        ### target graph
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels, output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.target_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                         nn.ReLU(True)])

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_graph' not in name and 'transpose_graph' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if 'target' in name or 'semantic' in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'source' in name:
                l.append(k)
        return l

    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph


        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

class deeplab_xception_transfer_basemodel_synBN(deeplab_xception_synBN.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256):
        super(deeplab_xception_transfer_basemodel_synBN, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                  os=os,)
        ### source graph
        # self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
        #                                                    nodes=n_classes)
        # self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        #
        # self.source_graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels, output_channels=out_channels,
        #                                             hidden_layers=hidden_layers, nodes=n_classes
        #                                             )
        # self.source_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
        #                                  nn.ReLU(True)])

        ### target graph
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels, output_channels=out_channels,
                                                    hidden_layers=hidden_layers, nodes=n_classes
                                                    )
        self.target_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                         nn.ReLU(True)])

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')

            if 'graph' in name and 'source' not in name and 'target' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if 'target' in name or 'semantic' in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'source' in name:
                l.append(k)
        return l

    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph


        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

class deeplab_xception_transfer_basemodel_synBN_savememory(deeplab_xception_synBN.DeepLabv3_plus):
    def __init__(self,nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256):
        super(deeplab_xception_transfer_basemodel_synBN_savememory, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                                   os=os, )
        ### source graph
        # self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
        #                                                    nodes=n_classes)
        # self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        #
        # self.source_graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels, output_channels=out_channels,
        #                                             hidden_layers=hidden_layers, nodes=n_classes
        #                                             )
        # self.source_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
        #                                  nn.ReLU(True)])

        ### target graph
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=input_channels, output_channels=out_channels,
                                                                   hidden_layers=hidden_layers, nodes=n_classes
                                                                   )
        self.target_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
                                                nn.BatchNorm2d(input_channels),
                                                nn.ReLU(True)])

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')

            if 'graph' in name and 'source' not in name and 'target' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if 'target' in name or 'semantic' in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'semantic' in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if 'source' in name:
                l.append(k)
        return l

    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph


        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

#######################
# transfer model
#######################

class deeplab_xception_transfer_projection(deeplab_xception_transfer_basemodel):
    def __init__(self, nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,
                 transfer_graph=None, source_classes=20):
        super(deeplab_xception_transfer_projection, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                   os=os, input_channels=input_channels,
                                                                   hidden_layers=hidden_layers, out_channels=out_channels, )
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=source_classes)
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.transpose_graph = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,adj=transfer_graph,
                                               begin_nodes=source_classes,end_nodes=n_classes)
        self.fc_graph = gcn.GraphConvolution(hidden_layers*3, hidden_layers)

    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph
        # source graph
        source_graph = self.source_featuremap_2_graph(x)
        source_graph1 = self.source_graph_conv1.forward(source_graph,adj=adj2_source, relu=True)
        source_graph2 = self.source_graph_conv2.forward(source_graph1, adj=adj2_source, relu=True)
        source_graph3 = self.source_graph_conv2.forward(source_graph2, adj=adj2_source, relu=True)

        source_2_target_graph1_v5 = self.transpose_graph.forward(source_graph1, adj=adj3_transfer, relu=True)
        source_2_target_graph2_v5 = self.transpose_graph.forward(source_graph2, adj=adj3_transfer, relu=True)
        source_2_target_graph3_v5 = self.transpose_graph.forward(source_graph3, adj=adj3_transfer, relu=True)

        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        source_2_target_graph1 = self.similarity_trans(source_graph1, graph)
        # graph combine 1
        # print(graph.size())
        # print(source_2_target_graph1.size())
        # print(source_2_target_graph1_v5.size())
        graph = torch.cat((graph,source_2_target_graph1.squeeze(0), source_2_target_graph1_v5.squeeze(0)),dim=-1)
        graph = self.fc_graph.forward(graph,relu=True)

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph2 = self.similarity_trans(source_graph2, graph)
        # graph combine 2
        graph = torch.cat((graph, source_2_target_graph2, source_2_target_graph2_v5), dim=-1)
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph3 = self.similarity_trans(source_graph3, graph)
        # graph combine 3
        graph = torch.cat((graph, source_2_target_graph3, source_2_target_graph3_v5), dim=-1)
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)

        # print(graph.size(),x.size())

        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def similarity_trans(self,source,target):
        sim = torch.matmul(F.normalize(target, p=2, dim=-1), F.normalize(source, p=2, dim=-1).transpose(-1, -2))
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')

            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_' not in name and 'transpose_graph' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

class deeplab_xception_transfer_projection_savemem(deeplab_xception_transfer_basemodel_savememory):
    def __init__(self, nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,
                 transfer_graph=None, source_classes=20):
        super(deeplab_xception_transfer_projection_savemem, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                           os=os, input_channels=input_channels,
                                                                           hidden_layers=hidden_layers, out_channels=out_channels, )
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=source_classes)
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.transpose_graph = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,adj=transfer_graph,
                                               begin_nodes=source_classes,end_nodes=n_classes)
        self.fc_graph = gcn.GraphConvolution(hidden_layers*3, hidden_layers)

    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph
        # source graph
        source_graph = self.source_featuremap_2_graph(x)
        source_graph1 = self.source_graph_conv1.forward(source_graph,adj=adj2_source, relu=True)
        source_graph2 = self.source_graph_conv2.forward(source_graph1, adj=adj2_source, relu=True)
        source_graph3 = self.source_graph_conv2.forward(source_graph2, adj=adj2_source, relu=True)

        source_2_target_graph1_v5 = self.transpose_graph.forward(source_graph1, adj=adj3_transfer, relu=True)
        source_2_target_graph2_v5 = self.transpose_graph.forward(source_graph2, adj=adj3_transfer, relu=True)
        source_2_target_graph3_v5 = self.transpose_graph.forward(source_graph3, adj=adj3_transfer, relu=True)

        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        source_2_target_graph1 = self.similarity_trans(source_graph1, graph)
        # graph combine 1
        graph = torch.cat((graph,source_2_target_graph1.squeeze(0), source_2_target_graph1_v5.squeeze(0)),dim=-1)
        graph = self.fc_graph.forward(graph,relu=True)

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph2 = self.similarity_trans(source_graph2, graph)
        # graph combine 2
        graph = torch.cat((graph, source_2_target_graph2, source_2_target_graph2_v5), dim=-1)
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph3 = self.similarity_trans(source_graph3, graph)
        # graph combine 3
        graph = torch.cat((graph, source_2_target_graph3, source_2_target_graph3_v5), dim=-1)
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)

        # print(graph.size(),x.size())

        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def similarity_trans(self,source,target):
        sim = torch.matmul(F.normalize(target, p=2, dim=-1), F.normalize(source, p=2, dim=-1).transpose(-1, -2))
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')

            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_' not in name and 'transpose_graph' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


class deeplab_xception_transfer_projection_synBN_savemem(deeplab_xception_transfer_basemodel_synBN_savememory):
    def __init__(self, nInputChannels=3, n_classes=7, os=16,input_channels=256,hidden_layers=128,out_channels=256,
                 transfer_graph=None, source_classes=20):
        super(deeplab_xception_transfer_projection_synBN_savemem, self).__init__(nInputChannels=nInputChannels, n_classes=n_classes,
                                                                                 os=os, input_channels=input_channels,
                                                                                 hidden_layers=hidden_layers, out_channels=out_channels, )
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
                                                           nodes=source_classes)
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.transpose_graph = gcn.Graph_trans(in_features=hidden_layers,out_features=hidden_layers,adj=transfer_graph,
                                               begin_nodes=source_classes,end_nodes=n_classes)
        self.fc_graph = gcn.GraphConvolution(hidden_layers*3 ,hidden_layers)

    def forward(self, input,adj1_target=None, adj2_source=None,adj3_transfer=None ):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph
        # source graph
        source_graph = self.source_featuremap_2_graph(x)
        source_graph1 = self.source_graph_conv1.forward(source_graph,adj=adj2_source, relu=True)
        source_graph2 = self.source_graph_conv2.forward(source_graph1, adj=adj2_source, relu=True)
        source_graph3 = self.source_graph_conv2.forward(source_graph2, adj=adj2_source, relu=True)

        source_2_target_graph1_v5 = self.transpose_graph.forward(source_graph1, adj=adj3_transfer, relu=True)
        source_2_target_graph2_v5 = self.transpose_graph.forward(source_graph2, adj=adj3_transfer, relu=True)
        source_2_target_graph3_v5 = self.transpose_graph.forward(source_graph3, adj=adj3_transfer, relu=True)

        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        source_2_target_graph1 = self.similarity_trans(source_graph1, graph)
        # graph combine 1
        graph = torch.cat((graph,source_2_target_graph1.squeeze(0), source_2_target_graph1_v5.squeeze(0)),dim=-1)
        graph = self.fc_graph.forward(graph,relu=True)

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph2 = self.similarity_trans(source_graph2, graph)
        # graph combine 2
        graph = torch.cat((graph, source_2_target_graph2, source_2_target_graph2_v5), dim=-1)
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph3 = self.similarity_trans(source_graph3, graph)
        # graph combine 3
        graph = torch.cat((graph, source_2_target_graph3, source_2_target_graph3_v5), dim=-1)
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)

        # print(graph.size(),x.size())

        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def similarity_trans(self,source,target):
        sim = torch.matmul(F.normalize(target, p=2, dim=-1), F.normalize(source, p=2, dim=-1).transpose(-1, -2))
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def load_source_model(self,state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')

            if 'graph' in name and 'source' not in name and 'target' not in name and 'fc_' not in name and 'transpose_graph' not in name:
                if 'featuremap_2_graph' in name:
                    name = name.replace('featuremap_2_graph','source_featuremap_2_graph')
                else:
                    name = name.replace('graph','source_graph')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print('unexpected key "{}" in state_dict'
                      .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


# if __name__ == '__main__':
    # net = deeplab_xception_transfer_projection_v3v5_more_savemem()
    # img = torch.rand((2,3,128,128))
    # net.eval()
    # a = torch.rand((1,1,7,7))
    # net.forward(img, adj1_target=a)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from collections import OrderedDict

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_aspp(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0):
        super(SeparableConv2d_aspp, self).__init__()

        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        self.depthwise_bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.pointwise_bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        #         x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x

class Decoder_module(nn.Module):
    def __init__(self, inplanes, planes, rate=1):
        super(Decoder_module, self).__init__()
        self.atrous_convolution = SeparableConv2d_aspp(inplanes, planes, 3, stride=1, dilation=rate,padding=1)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            raise RuntimeError()
        else:
            kernel_size = 3
            padding = rate
            self.atrous_convolution = SeparableConv2d_aspp(inplanes, planes, 3, stride=1, dilation=rate,
                                                           padding=padding)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module_rate0(nn.Module):
    def __init__(self, inplanes, planes, rate=1):
        super(ASPP_module_rate0, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
            self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                                stride=1, padding=padding, dilation=rate, bias=False)
            self.bn = nn.BatchNorm2d(planes, eps=1e-5, affine=True)
            self.relu = nn.ReLU()
        else:
            raise RuntimeError()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0):
        super(SeparableConv2d_same, self).__init__()

        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        self.depthwise_bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.pointwise_bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        return x

class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=2, bias=False)
            if is_last:
                self.skip = nn.Conv2d(inplanes, planes, 1, stride=1, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2,dilation=dilation))

        if is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1,dilation=dilation))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        # print(x.size(),skip.size())
        x += skip

        return x

class Block2(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block2, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            self.block2_lastconv = nn.Sequential(*[self.relu,SeparableConv2d_same(planes, planes, 3, stride=2,dilation=dilation)])

        if is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        low_middle = x.clone()
        x1 = x
        x1 = self.block2_lastconv(x1)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x1 += skip

        return x1,low_middle

class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block2(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d_aspp(1024, 1536, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d_aspp(1536, 1536, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_aspp(1536, 2048, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn5 = nn.BatchNorm2d(2048)

        # Init weights
        # self.__init_weight()

        # Load pretrained model
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('conv1 ',x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # print('block1',x.size())
        # low_level_feat = x
        x,low_level_feat = self.block2(x)
        # print('block2',x.size())
        x = self.block3(x)
        # print('xception block3 ',x.size())

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        # x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.xception_features = Xception(nInputChannels, os, pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module_rate0(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU()
                                             )

        self.concat_projection_conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.concat_projection_bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.feature_projection_conv1 = nn.Conv2d(256, 48, 1, bias=False)
        self.feature_projection_bn1 = nn.BatchNorm2d(48)

        self.decoder = nn.Sequential(Decoder_module(304, 256),
                                     Decoder_module(256, 256)
                                     )
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)

    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = F.upsample(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.xception_features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_totally_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_aspp_bn(self):
        for m in self.aspp1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp4.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def learnable_parameters(self):
        layer_features_BN = []
        layer_features = []
        layer_aspp = []
        layer_projection  =[]
        layer_decoder = []
        layer_other = []
        model_para = list(self.named_parameters())
        for name,para in model_para:
            if 'xception' in name:
                if 'bn' in name or 'downsample.1.weight' in name or 'downsample.1.bias' in name:
                    layer_features_BN.append(para)
                else:
                    layer_features.append(para)
                    # print (name)
            elif 'aspp' in name:
                layer_aspp.append(para)
            elif 'projection' in name:
                layer_projection.append(para)
            elif 'decode' in name:
                layer_decoder.append(para)
            elif 'global' not in name:
                layer_other.append(para)
        return layer_features_BN,layer_features,layer_aspp,layer_projection,layer_decoder,layer_other

    def get_backbone_para(self):
        layer_features = []
        other_features = []
        model_para = list(self.named_parameters())
        for name, para in model_para:
            if 'xception' in name:
                layer_features.append(para)
            else:
                other_features.append(para)

        return layer_features, other_features

    def train_fixbn(self, mode=True, freeze_bn=True, freeze_bn_affine=False):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super(DeepLabv3_plus, self).train(mode)
        if freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if freeze_bn:
            for m in self.xception_features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            # for m in self.aspp1.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.aspp2.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.aspp3.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.aspp4.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.global_avg_pool.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.concat_projection_bn1.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.feature_projection_bn1.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_state_dict_new(self, state_dict):
        own_state = self.state_dict()
        #for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.','')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print ('unexpected key "{}" in state_dict'
                       .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                        name, own_state[name].size(), param.size()))
                continue # i add inshop_cos 2018/02/01
                # raise
                    # print 'copying %s' %name
                # if isinstance(param, own_state):
                # backwards compatibility for serialized parameters
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.xception_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True)
    model.eval()
    image = torch.randn(1, 3, 512, 512)*255
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())
    # print(output)







import os
import numpy as np
from PIL import Image


def main():
    image_paths, label_paths = init_path()
    hist = compute_hist(image_paths, label_paths)
    show_result(hist)


def init_path():
    list_file = './human/list/val_id.txt'
    file_names = []
    with open(list_file, 'rb') as f:
        for fn in f:
            file_names.append(fn.strip())

    image_dir = './human/features/attention/val/results/'
    label_dir = './human/data/labels/'

    image_paths = []
    label_paths = []
    for file_name in file_names:
        image_paths.append(os.path.join(image_dir, file_name + '.png'))
        label_paths.append(os.path.join(label_dir, file_name + '.png'))
    return image_paths, label_paths


def fast_hist(lbl, pred, n_cls):
    '''
    compute the miou
    :param lbl: label
    :param pred: output
    :param n_cls: num of class
    :return:
    '''
    # print(n_cls)
    k = (lbl >= 0) & (lbl < n_cls)
    return np.bincount(n_cls * lbl[k].astype(int) + pred[k], minlength=n_cls ** 2).reshape(n_cls, n_cls)


def compute_hist(images, labels,n_cls=20):
    hist = np.zeros((n_cls, n_cls))
    for img_path, label_path in zip(images, labels):
        label = Image.open(label_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape
        imgsz = image_array.shape
        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)

        hist += fast_hist(label_array, image_array, n_cls)

    return hist


def show_result(hist):
    classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
               'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
               'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
               'rightShoe']
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print('=' * 50)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    print('-' * 50)

    # @evaluation 2: mean accuracy & per-class accuracy
    print('Accuracy for each class (pixel accuracy):')
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    print('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IU', np.nanmean(iu))
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print('>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)

def get_iou(pred,lbl,n_cls):
    '''
    need tensor cpu
    :param pred:
    :param lbl:
    :param n_cls:
    :return:
    '''
    hist = np.zeros((n_cls,n_cls))
    for i,j in zip(range(pred.size(0)),range(lbl.size(0))):
        pred_item = pred[i].data.numpy()
        lbl_item = lbl[j].data.numpy()
        hist += fast_hist(lbl_item, pred_item, n_cls)
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    # for i in range(20):
    #     print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IU', np.nanmean(iu))
    miou = np.nanmean(iu)
    print('-' * 50)
    return miou

def get_iou_from_list(pred,lbl,n_cls):
    '''
    need tensor cpu
    :param pred: list
    :param lbl: list
    :param n_cls:
    :return:
    '''
    hist = np.zeros((n_cls,n_cls))
    for i,j in zip(range(len(pred)),range(len(lbl))):
        pred_item = pred[i].data.numpy()
        lbl_item = lbl[j].data.numpy()
        # print(pred_item.shape,lbl_item.shape)
        hist += fast_hist(lbl_item, pred_item, n_cls)

    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    # for i in range(20):
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    print('-' * 50)
    #     print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IU', np.nanmean(iu))
    miou = np.nanmean(iu)
    print('-' * 50)

    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    print('-' * 50)

    return miou


if __name__ == '__main__':
    import torch
    pred = torch.autograd.Variable(torch.ones((2,1,32,32)).int())*20
    pred2 = torch.autograd.Variable(torch.zeros((2,1, 32, 32)).int())
    # lbl = [torch.zeros((32,32)).int() for _ in range(len(pred))]
    get_iou(pred,pred2,7)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
from torchvision import transforms
from PIL import Image

__all__ = ['cusSampler','Sampler_uni']

'''common N-pairs sampler'''
def index_dataset(dataset):
    '''
    get the index according to the dataset type(e.g. pascal or atr or cihp)
    :param dataset:
    :return:
    '''
    return_dict = {}
    for i in range(len(dataset)):
        tmp_lbl = dataset.datasets_lbl[i]
        if tmp_lbl in return_dict:
            return_dict[tmp_lbl].append(i)
        else :
            return_dict[tmp_lbl] = [i]
    return return_dict

def sample_from_class(dataset,class_id):
    return dataset[class_id][random.randrange(len(dataset[class_id]))]

def sampler_npair_K(batch_size,dataset,K=2,label_random_list = [0,0,1,1,2,2,2]):
    images_by_class = index_dataset(dataset)
    for batch_idx in range(int(math.ceil(len(dataset) * 1.0 / batch_size))):
        example_indices = [sample_from_class(images_by_class, class_label_ind) for _ in range(batch_size)
                           for class_label_ind in [label_random_list[random.randrange(len(label_random_list))]]
                           ]
        yield example_indices[:batch_size]

def sampler_(images_by_class,batch_size,dataset,K=2,label_random_list = [0,0,1,1,]):
    # images_by_class = index_dataset(dataset)
    a = label_random_list[random.randrange(len(label_random_list))]
    # print(a)
    example_indices = [sample_from_class(images_by_class, a) for _ in range(batch_size)
                           for class_label_ind in [a]
                           ]
    return example_indices[:batch_size]

class cusSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, dataset, batchsize, label_random_list=[0,1,1,1,2,2,2]):
        self.images_by_class = index_dataset(dataset)
        self.batch_size = batchsize
        self.dataset = dataset
        self.label_random_list = label_random_list
        self.len = int(math.ceil(len(dataset) * 1.0 / batchsize))

    def __iter__(self):
        # return [sample_from_class(self.images_by_class, class_label_ind) for _ in range(self.batchsize)
        #                    for class_label_ind in [self.label_random_list[random.randrange(len(self.label_random_list))]]
        #                    ]
        # print(sampler_(self.images_by_class,self.batch_size,self.dataset))
        return iter(sampler_(self.images_by_class,self.batch_size,self.dataset,self.label_random_list))

    def __len__(self):
        return self.len

def shuffle_cus(d1=20,d2=10,d3=5,batch=2):
    return_list = []
    total_num = d1 + d2 + d3
    list1 = list(range(d1))
    batch1 = d1//batch
    list2 = list(range(d1,d1+d2))
    batch2 = d2//batch
    list3 = list(range(d1+d2,d1+d2+d3))
    batch3 = d3// batch
    random.shuffle(list1)
    random.shuffle(list2)
    random.shuffle(list3)
    random_list = list(range(batch1+batch2+batch3))
    random.shuffle(random_list)
    for random_batch_index in random_list:
        if random_batch_index < batch1:
            random_batch_index1 = random_batch_index
            return_list += list1[random_batch_index1*batch : (random_batch_index1+1)*batch]
        elif random_batch_index < batch1 + batch2:
            random_batch_index1 = random_batch_index - batch1
            return_list += list2[random_batch_index1*batch : (random_batch_index1+1)*batch]
        else:
            random_batch_index1 = random_batch_index - batch1 - batch2
            return_list += list3[random_batch_index1*batch : (random_batch_index1+1)*batch]
    return return_list

def shuffle_cus_balance(d1=20,d2=10,d3=5,batch=2,balance_index=1):
    return_list = []
    total_num = d1 + d2 + d3
    list1 = list(range(d1))
    # batch1 = d1//batch
    list2 = list(range(d1,d1+d2))
    # batch2 = d2//batch
    list3 = list(range(d1+d2,d1+d2+d3))
    # batch3 = d3// batch
    random.shuffle(list1)
    random.shuffle(list2)
    random.shuffle(list3)
    total_list = [list1,list2,list3]
    target_list = total_list[balance_index]
    for index,list_item in enumerate(total_list):
        if index == balance_index:
            continue
        if len(list_item) > len(target_list):
            list_item = list_item[:len(target_list)]
            total_list[index] = list_item
    list1 = total_list[0]
    list2 = total_list[1]
    list3 = total_list[2]
    # list1 = list(range(d1))
    d1 = len(list1)
    batch1 = d1 // batch
    # list2 = list(range(d1, d1 + d2))
    d2 = len(list2)
    batch2 = d2 // batch
    # list3 = list(range(d1 + d2, d1 + d2 + d3))
    d3 = len(list3)
    batch3 = d3 // batch

    random_list = list(range(batch1+batch2+batch3))
    random.shuffle(random_list)
    for random_batch_index in random_list:
        if random_batch_index < batch1:
            random_batch_index1 = random_batch_index
            return_list += list1[random_batch_index1*batch : (random_batch_index1+1)*batch]
        elif random_batch_index < batch1 + batch2:
            random_batch_index1 = random_batch_index - batch1
            return_list += list2[random_batch_index1*batch : (random_batch_index1+1)*batch]
        else:
            random_batch_index1 = random_batch_index - batch1 - batch2
            return_list += list3[random_batch_index1*batch : (random_batch_index1+1)*batch]
    return return_list

class Sampler_uni(torch.utils.data.sampler.Sampler):
    def __init__(self, num1, num2, num3, batchsize,balance_id=None):
        self.num1 = num1
        self.num2 = num2
        self.num3 = num3
        self.batchsize = batchsize
        self.balance_id = balance_id

    def __iter__(self):
        if self.balance_id is not None:
            rlist = shuffle_cus_balance(self.num1, self.num2, self.num3, self.batchsize, balance_index=self.balance_id)
        else:
            rlist = shuffle_cus(self.num1, self.num2, self.num3, self.batchsize)
        return iter(rlist)


    def __len__(self):
        if self.balance_id is not None:
            return self.num1*3
        return self.num1+self.num2+self.num3

from .test_human import get_iou_from_list
import utils


__all__ = ['get_iou_from_list','utils']
import os

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def get_mhp_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128], # 21
                       [96, 0, 0], [0, 96, 0], [96, 96, 0],
                       [0, 0, 96], [96, 0, 96], [0, 96, 96], [96, 96, 96],
                       [32, 0, 0], [160, 0, 0], [32, 96, 0], [160, 96, 0],
                       [32, 0, 96], [160, 0, 96], [32, 96, 96], [160, 96, 96],
                       [0, 32, 0], [96, 32, 0], [0, 160, 0], [96, 160, 0],
                       [0, 32, 96], # 41
                       [48, 0, 0], [0, 48, 0], [48, 48, 0],
                       [0, 0, 96], [48, 0, 48], [0, 48, 48], [48, 48, 48],
                       [16, 0, 0], [80, 0, 0], [16, 48, 0], [80, 48, 0],
                       [16, 0, 48], [80, 0, 48], [16, 48, 48], [80, 48, 48],
                       [0, 16, 0], [48, 16, 0], [0, 80, 0],   # 59

                       ])

def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'mhp':
        n_classes = 59
        label_colours = get_mhp_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,size_average=size_average)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=size_average)
    loss = criterion(logit, target.long())

    return loss

def cross_entropy2d_dataparallel(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.DataParallel(nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,size_average=size_average))
    else:
        criterion = nn.DataParallel(nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=size_average))
    loss = criterion(logit, target.long())

    return loss.sum()

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou

def scale_tensor(input,size=512,mode='bilinear'):
    print(input.size())
    # b,h,w = input.size()
    _, _, h, w = input.size()
    if mode == 'nearest':
        if h == 512 and w == 512:
            return input
        return F.upsample_nearest(input,size=(size,size))
    if h>512 and w > 512:
        return F.upsample(input, size=(size,size), mode=mode, align_corners=True)
    return F.upsample(input, size=(size,size), mode=mode, align_corners=True)

def scale_tensor_list(input,):

    output = []
    for i in range(len(input)-1):
        output_item = []
        for j in range(len(input[i])):
            _, _, h, w = input[-1][j].size()
            output_item.append(F.upsample(input[i][j], size=(h,w), mode='bilinear', align_corners=True))
        output.append(output_item)
    output.append(input[-1])
    return output

def scale_tensor_list_0(input,base_input):

    output = []
    assert  len(input) == len(base_input)
    for j in range(len(input)):
        _, _, h, w = base_input[j].size()
        after_size = F.upsample(input[j], size=(h,w), mode='bilinear', align_corners=True)
        base_input[j] = base_input[j] + after_size
    # output.append(output_item)
    # output.append(input[-1])
    return base_input

if __name__ == '__main__':
    print(lr_poly(0.007,iter_=99,max_iter=150))
# -*- coding: utf-8 -*-
# File   : unittest.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import unittest

import numpy as np
from torch.autograd import Variable


def as_numpy(v):
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, a, b, atol=1e-3, rtol=1e-3):
        npa, npb = as_numpy(a), as_numpy(b)
        self.assertTrue(
                np.allclose(npa, npb, atol=atol),
                'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())
        )

# -*- coding: utf-8 -*-
# File   : batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import collections

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

from .comm import SyncMaster

__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)

# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .replicate import DataParallelWithCallback, patch_replication_callback

# -*- coding: utf-8 -*-
# File   : comm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import queue
import collections
import threading

__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)

# -*- coding: utf-8 -*-
# File   : replicate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import functools

from torch.nn.parallel.data_parallel import DataParallel

__all__ = [
    'CallbackContext',
    'execute_replication_callbacks',
    'DataParallelWithCallback',
    'patch_replication_callback'
]


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate

import sys
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2

from .networks import deeplab_xception_transfer, graph



class SegmentationWrapper(nn.Module):
    def __init__(self, args):
        super(SegmentationWrapper, self).__init__()
        self.use_gpus = args.num_gpus > 0

        self.net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
            n_classes=20, hidden_layers=128, source_classes=7)
        
        x = torch.load(f'{args.project_dir}/pretrained_weights/graphonomy/pretrained_model.pth')
        self.net.load_source_model(x)

        if self.use_gpus:
            self.net.cuda()

        self.net.eval()

        # transforms
        self.rgb2bgr = transforms.Lambda(lambda x:x[:, [2,1,0],...])

        # adj
        adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
        self.adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

        adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
        self.adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

        cihp_adj = graph.preprocess_adj(graph.cihp_graph)
        adj3_ = Variable(torch.from_numpy(cihp_adj).float())
        self.adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

        # Erosion kernel
        SIZE = 5

        grid = np.meshgrid(np.arange(-SIZE//2+1, SIZE//2+1), np.arange(-SIZE//2+1, SIZE//2+1))[:2]
        self.kernel = (grid[0]**2 + grid[1]**2 < (SIZE / 2.)**2).astype('uint8')

    def forward(self, imgs):
        b, t = imgs.shape[:2]
        imgs = imgs.view(b*t, *imgs.shape[2:])

        inputs = self.rgb2bgr(imgs)
        inputs = Variable(inputs, requires_grad=False)

        if self.use_gpus:
            inputs = inputs.cuda()

        outputs = []
        with torch.no_grad():
            for input in inputs.split(1, 0):
                outputs.append(self.net.forward(input, self.adj1_test, self.adj3_test, self.adj2_test))
        outputs = torch.cat(outputs, 0)

        outputs = F.softmax(outputs, 1)

        segs = 1 - outputs[:, [0]] # probabilities for FG

        # Erosion
        segs_eroded = []
        for seg in segs.split(1, 0):
            seg = cv2.erode(seg[0, 0].cpu().numpy(), self.kernel, iterations=1)
            segs_eroded.append(torch.from_numpy(seg))
        segs = torch.stack(segs_eroded)[:, None].to(imgs.device)

        return segs
import torch
from torch import nn
import torch.nn.functional as F

from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--adv_pred_type', type=str, default='ragan', choices=['gan', 'rgan', 'ragan'])
        parser.add('--adv_loss_weight', type=float, default=0.5)

    def __init__(self, args):
        super(LossWrapper, self).__init__()
        # Supported prediction functions
        get_preds = {
            'gan'  : lambda real_scores, fake_scores: 
                (real_scores, fake_scores),
            'rgan' : lambda real_scores, fake_scores: 
                (real_scores - fake_scores, 
                 fake_scores - real_scores),
            'ragan': lambda real_scores, fake_scores: 
                (real_scores - fake_scores.mean(),
                 fake_scores - real_scores.mean())}

        self.get_preds = get_preds[args.adv_pred_type]

        # The only (currently) supported loss type is hinge loss
        self.loss_dis  = lambda real_preds, fake_preds: torch.relu(1 - real_preds).mean() + torch.relu(1 + fake_preds).mean()
        if 'r' in args.adv_pred_type:
            self.loss_gen = lambda real_preds, fake_preds: torch.relu(1 - fake_preds).mean() + torch.relu(1 + real_preds).mean()
        else:
            self.loss_gen = lambda real_preds, fake_preds: -fake_preds.mean()

        self.weight = args.adv_loss_weight

    def forward(self, data_dict, losses_dict):    
        # Calculate loss for dis
        real_scores = data_dict['real_scores']
        fake_scores = data_dict['fake_scores_dis']
        real_preds, fake_preds = self.get_preds(real_scores, fake_scores)
        losses_dict['D_ADV'] = self.loss_dis(real_preds, fake_preds) * self.weight
            
        # Calculate loss for gen
        real_scores = real_scores.detach()
        fake_scores = data_dict['fake_scores_gen']
        real_preds, fake_preds = self.get_preds(real_scores, fake_scores)
        losses_dict['G_ADV'] = self.loss_gen(real_preds, fake_preds) * self.weight

        return losses_dict
# Third party
import torch
from torch import nn
import torch.nn.functional as F

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--wpr_loss_type', type=str, default='l1')
        parser.add('--wpr_loss_weight', type=float, default=10.0)
        parser.add('--wpr_loss_weight_decay', type=float, default=0.9, help='multiplicative decay of loss weight')
        parser.add('--wpr_loss_decay_schedule', type=int, default=50, help='num iters after which decay happends')
        parser.add('--wpr_loss_apply_to', type=str, default='pred_target_delta_uvs', help='tensors this loss is applied to')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.apply_to = rn_utils.parse_str_to_list(args.wpr_loss_apply_to)
        self.eps = args.eps

        self.reg_type = args.wpr_loss_type
        self.weight = args.wpr_loss_weight

        self.weight_decay = args.wpr_loss_weight_decay
        self.decay_schedule = args.wpr_loss_decay_schedule
        self.num_iters = 0

    def forward(self, data_dict, losses_dict):
        if self.num_iters == self.decay_schedule:
            self.weight = max(self.weight * self.weight_decay, self.eps)
            self.num_iters = 1

        if self.weight == self.eps:
            return losses_dict

        loss = 0

        for tensor_name in self.apply_to:
            if self.reg_type == 'l1':
                loss += data_dict[tensor_name].abs().mean()
            else:
                raise # Unknown reg_type

        loss /= len(self.apply_to)
 
        losses_dict['G_WPR'] = loss * self.weight

        if self.weight_decay != 1.0:
            self.num_iters += 1

        return losses_dict
import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import math
import pathlib
from skimage import transform



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--csm_model', type=str, default='insight_face')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.use_gpu = args.num_gpus > 0

        model_path = pathlib.Path(args.project_dir) / 'pretrained_weights' / 'csim' / f'{args.csm_model}.pth'

        self.model = InsightFaceWrapper(model_path)
        if self.use_gpu:
            self.model.cuda()

    def forward(self, data_dict, losses_dict):
        real_imgs = data_dict['target_imgs']
        fake_imgs = data_dict['pred_target_imgs']
        real_poses = data_dict['target_poses']
        
        b, t, c, h, w = real_imgs.shape
        real_imgs = real_imgs.view(-1, c, h, w)
        fake_imgs = fake_imgs.view(-1, c, h, w)
        real_poses = real_poses.view(b*t, -1)

        affine_matrices = find_affine_transformation(real_poses, h, w)

        with torch.no_grad():
            real_embeds = self.model(affine_matrices, real_imgs)
            fake_embeds = self.model(affine_matrices, fake_imgs)

        real_embeds = real_embeds.detach().cpu().numpy()
        fake_embeds = fake_embeds.detach().cpu().numpy()

        # Calc cosine similarity
        csim = (real_embeds * fake_embeds).sum(1) / (real_embeds**2).sum(1)**0.5 / (fake_embeds**2).sum(1)**0.5 

        losses_dict['G_CSIM'] = torch.FloatTensor([csim.mean()])

        if self.use_gpu:
            losses_dict['G_CSIM'] = losses_dict['G_CSIM'].cuda()

        return losses_dict

    def __repr__(self):
        return '(cdist): LossWrapper()'

def find_affine_transformation(poses, h_img=256, w_img=256):
    """
    Function return matrix of affine transformation to use in torch.nn.functional.affine_grid

    Input:
    facial_keypoints: np.array of size (5, 2) - coordinates of key facial points in pixel coords
                      right eye, left eye, nose, right mouse, left mouse
    h_img: int, height of input image
    w_img: int, width of input image
    returns: np.array of size (2, 3) - affine matrix
    """
    poses = (poses.detach() + 1) / 2

    right_eye = list(range(36, 42))
    left_eye = list(range(42, 48))
    nose = [30]
    right_mouth = [48]
    left_mouth = [54]

    keypoints = poses.cpu().numpy().reshape(-1, 68, 2)

    facial_keypoints = np.concatenate([
        keypoints[:, right_eye].astype('float32').mean(1, keepdims=True), # right eye
        keypoints[:, left_eye].astype('float32').mean(1, keepdims=True), # left eye
        keypoints[:, nose].astype('float32'), # nose
        keypoints[:, right_mouth].astype('float32'), # right mouth
        keypoints[:, left_mouth].astype('float32'), # left mouth
    ], 1)

    facial_keypoints[:, :, 0] *= h_img
    facial_keypoints[:, :, 1] *= w_img

    #affine_matrix = torch.from_numpy(find_affine_transformation(facial_keypoints, 
    #    h_img=h_img, w_img=w_img)).float()

    h_grid = 112
    w_grid = 112

    src = np.array([
        [35.343697, 51.6963],
        [76.453766, 51.5014],
        [56.029396, 71.7366],
        [39.14085 , 92.3655],
        [73.18488 , 92.2041]], dtype=np.float32)

    affine_matrices = []

    for facial_keypoints_i in facial_keypoints:
        tform = transform.estimate_transform('similarity', src, facial_keypoints_i)
        affine_matrix = tform.params[:2, :]

        affine_matrices.append(affine_matrix)

    affine_matrices = np.stack(affine_matrices, axis=0)

    # do transformation for grid in [-1, 1]
    affine_matrices[:, 0, 0] = affine_matrices[:, 0, 0]*w_grid/w_img
    affine_matrices[:, 0, 1] = affine_matrices[:, 0, 1]*h_grid/w_img
    affine_matrices[:, 0, 2] = affine_matrices[:, 0, 2]*2/w_img + affine_matrices[:, 0, 1] + affine_matrices[:, 0, 0] - 1
    affine_matrices[:, 1, 0] = affine_matrices[:, 1, 0]*w_grid/h_img
    affine_matrices[:, 1, 1] = affine_matrices[:, 1, 1]*h_grid/h_img
    affine_matrices[:, 1, 2] = affine_matrices[:, 1, 2]*2/h_img + affine_matrices[:, 1, 0] + affine_matrices[:, 1, 1] - 1

    affine_matrices = torch.from_numpy(affine_matrices)

    if affine_matrices.type() != poses.type():
        affine_matrices = affine_matrices.type(poses.type())

    return affine_matrices


class InsightFaceWrapper(nn.Module):
    """
    Wrapper of InsightFaceModel
    """
    def __init__(self, path_weights, num_layers=50, drop_ratio=0.6, mode='ir_se'):
        super(InsightFaceWrapper, self).__init__()
        self.model = Backbone(num_layers, drop_ratio, mode)
        self.model.load_state_dict(torch.load(path_weights))
        self.model.train(False)

    def forward(self, affine_matrix, image):
        batch_size = image.shape[0]
        grid = nn.functional.affine_grid(affine_matrix, torch.Size((batch_size, 3, 112, 112)))
        warped_image = nn.functional.grid_sample(image, grid)
        return self.model(warped_image)

##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      BatchNorm2d(64), 
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), 
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

##################################  MobileFaceNet #############################################################
    
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
    
    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        
        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        return l2_norm(out)

##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

##################################  Cosface head #############################################################    
    
class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self,embedding_size=512,classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = 0.35 # additive margin recommended by the paper
        self.s = 30. # see normface https://arxiv.org/abs/1704.06369
    def forward(self,embbedings,label):
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1,1) #size=(B,1)
        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,label.data.view(-1,1),1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output
import torch
from torch import nn
import torch.nn.functional as F

from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--fem_loss_type', type=str, default='l1', help='l1|mse')
        parser.add('--fem_loss_weight', type=float, default=10.)

    def __init__(self, args):
        super(LossWrapper, self).__init__()
        # Supported loss functions
        losses = {
            'mse': F.mse_loss,
            'l1': F.l1_loss}

        self.loss = losses[args.fem_loss_type]
        self.weight = args.fem_loss_weight

    def forward(self, data_dict, losses_dict):
        real_feats_gen = data_dict['real_feats_gen']
        fake_feats_gen = data_dict['fake_feats_gen']

        # Calculate the loss
        loss = 0
        for real_feats, fake_feats in zip(real_feats_gen, fake_feats_gen):
            loss += self.loss(fake_feats, real_feats.detach())
        loss /= len(real_feats_gen)
        loss *= self.weight

        losses_dict['G_FM'] = loss

        return losses_dict

import pathlib
import torch
from torch import nn
import torch.nn.functional as F



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--lps_model', type=str, default='net-lin')
        parser.add('--lps_net', type=str, default='vgg')
        parser.add('--lps_calc_grad', action='store_true', help='if True, the loss is differentiable')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.calc_grad = args.lps_calc_grad

        model_path = pathlib.Path(args.project_dir) / 'pretrained_weights' / 'lpips' / f'{args.lps_net}.pth'

        self.loss = PerceptualLoss(
            model=args.lps_model, net=args.lps_net, model_path=model_path, 
            use_gpu=args.num_gpus > 0, gpu_ids=[args.local_rank]).eval()

    def forward(self, data_dict, losses_dict):
        real_imgs = data_dict['target_imgs']
        fake_imgs = data_dict['pred_target_imgs']
        
        b, t, c, h, w = real_imgs.shape
        real_imgs = real_imgs.view(-1, c, h, w)
        fake_imgs = fake_imgs.view(-1, c, h, w)

        # Calculate the loss
        if self.calc_grad:
            loss = self.loss(fake_imgs, real_imgs)
        else:
            with torch.no_grad():
                loss = self.loss(fake_imgs.detach(), real_imgs)

        losses_dict['G_LPIPS'] = loss.mean()

        return losses_dict

############################################################
# The contents below have been combined using files in the #
# following repository:                                    #
# https://github.com/richzhang/PerceptualSimilarity        #
############################################################

# Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

############################################################
#                       __init__.py                        #
############################################################

import numpy as np
from skimage.measure import compare_ssim
import torch
from torch.autograd import Variable

class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='alex', colorspace='rgb', model_path=None, spatial=False, use_gpu=True, gpu_ids=[0]): # VGG using our perceptually-learned weights (LPIPS metric)
        super(PerceptualLoss, self).__init__()
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace,
                              model_path=model_path, spatial=self.spatial, gpu_ids=gpu_ids)

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]
        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target  - 1
            pred = 2 * pred  - 1

        return self.model.forward(target, pred)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def l2(p0, p1, range=255.):
    return .5*np.mean((p0 / range - p1 / range)**2)

def psnr(p0, p1, peak=255.):
    return 10*np.log10(peak**2/np.mean((1.*p0-1.*p1)**2))

def dssim(p0, p1, range=255.):
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.

def rgb2lab(in_img,mean_cent=False):
    from skimage import color
    img_lab = color.rgb2lab(in_img)
    if(mean_cent):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    return img_lab

def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1,2,0))

def np2tensor(np_obj):
     # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2tensorlab(image_tensor,to_norm=True,mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if(mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    if(to_norm and not mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
        img_lab = img_lab/100.

    return np2tensor(img_lab)

def tensorlab2tensor(lab_tensor,return_inbnd=False):
    from skimage import color
    import warnings
    warnings.filterwarnings("ignore")

    lab = tensor2np(lab_tensor)*100.
    lab[:,:,0] = lab[:,:,0]+50

    rgb_back = 255.*np.clip(color.lab2rgb(lab.astype('float')),0,1)
    if(return_inbnd):
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        mask = 1.*np.isclose(lab_back,lab,atol=2.)
        mask = np2tensor(np.prod(mask,axis=2)[:,:,np.newaxis])
        return (im2tensor(rgb_back),mask)
    else:
        return im2tensor(rgb_back)

def rgb2lab(input):
    from skimage import color
    return color.rgb2lab(input / 255.)

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2vec(vector_tensor):
    return vector_tensor.data.cpu().numpy()[:, :, 0, 0]

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

############################################################
#                      base_model.py                       #
############################################################

import os
import torch
from torch.autograd import Variable
from pdb import set_trace as st
from IPython import embed

class BaseModel():
    def __init__(self):
        pass;
        
    def name(self):
        return 'BaseModel'

    def initialize(self, use_gpu=True, gpu_ids=[0]):
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path, map_location='cpu'))

    def update_learning_rate():
        pass

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')

############################################################
#                      dist_model.py                       #
############################################################

import sys
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from scipy.ndimage import zoom
import fractions
import functools
import skimage.transform
from tqdm import tqdm

from IPython import embed

class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False, 
            is_train=False, lr=.0001, beta1=0.5, version='0.1', gpu_ids=[0]):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        BaseModel.initialize(self, use_gpu=use_gpu, gpu_ids=gpu_ids)

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model_name = '%s [%s]'%(model,net)

        if(self.model == 'net-lin'): # pretrained net + linear layer
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,
                use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize), '..', 'weights/v%s/%s.pth'%(version,net)))

            if(not is_train):
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)

        elif(self.model=='net'): # pretrained network
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()

        if(use_gpu):
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0]) # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if(self.use_gpu):
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)
        self.var_p1 = Variable(self.input_p1,requires_grad=True)

    def forward_train(self): # run forward pass
        # print(self.net.module.scaling_layer.shift)
        # print(torch.norm(self.net.module.net.slice1[0].weight).item(), torch.norm(self.net.module.lin0.model[1].weight).item())

        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0,self.d1,self.input_judge)

        self.var_judge = Variable(1.*self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(self.d0, self.d1, self.var_judge*2.-1.)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self,d0,d1,judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                            ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.data.size()[2]

        ref_img = tensor2im(self.var_ref.data)
        p0_img = tensor2im(self.var_p0.data)
        p1_img = tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)
        p1_img_vis = zoom(p1_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr

def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s+=func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        d1s+=func(data['ref'],data['p1']).data.cpu().numpy().flatten().tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader, func, name=''):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds+=func(data['p0'],data['p1']).data.cpu().numpy().tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))

############################################################
#                    networks_basic.py                     #
############################################################

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as st
from skimage import color
from IPython import embed

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_H=64): # assumes scale factor is same for H and W
    in_H = in_tens.shape[2]
    scale_factor = 1.*out_H/in_H

    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)

# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = squeezenet
            self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace=colorspace

class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = l2(tensor2np(tensor2tensorlab(in0.data,to_norm=False)), 
                tensor2np(tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = dssim(1.*tensor2im(in0.data), 1.*tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = dssim(tensor2np(tensor2tensorlab(in0.data,to_norm=False)), 
                tensor2np(tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)

############################################################
#                 pretrained_networks.py                   #
############################################################

from collections import namedtuple
import torch
from torchvision import models as tv
from IPython import embed

class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        out = vgg_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out



class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if(num==18):
            self.net = tv.resnet18(pretrained=pretrained)
        elif(num==34):
            self.net = tv.resnet34(pretrained=pretrained)
        elif(num==50):
            self.net = tv.resnet50(pretrained=pretrained)
        elif(num==101):
            self.net = tv.resnet101(pretrained=pretrained)
        elif(num==152):
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ['relu1','conv2','conv3','conv4','conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--ssm_use_masks', action='store_true', help='use masks before application of the loss')
        parser.add('--ssm_calc_grad', action='store_true', help='if True, the loss is differentiable')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.calc_grad = args.ssm_calc_grad
        self.use_masks = args.ssm_use_masks

        self.loss = SSIM()

    def forward(self, data_dict, losses_dict):
        real_imgs = data_dict['target_imgs']
        fake_imgs = data_dict['pred_target_imgs']
        
        b, t, c, h, w = real_imgs.shape
        real_imgs = real_imgs.view(-1, c, h, w)
        fake_imgs = fake_imgs.view(-1, c, h, w)

        if self.use_masks:
            real_segs = data_dict['real_segs'].view(b*t, -1, h, w)

            real_imgs = real_imgs * real_segs
            fake_imgs = fake_imgs * real_segs

        # Calculate the loss
        if self.calc_grad:
            loss = self.loss(fake_imgs, real_imgs)
        else:
            with torch.no_grad():
                loss = self.loss(fake_imgs.detach(), real_imgs)

        losses_dict['G_SSIM'] = loss.mean()

        return losses_dict


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import pathlib

# This project
from runners import utils as rn_utils
from networks import utils as nt_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        # Extractor parameters
        parser.add('--per_full_net_names', type=str, default='vgg19_imagenet_pytorch, vgg16_face_caffe')
        parser.add('--per_net_layers', type=str, default='1,6,11,20,29; 1,6,11,18,25', help='a list of layers indices')
        parser.add('--per_pooling', type=str, default='avgpool', choices=['maxpool', 'avgpool'])
        parser.add('--per_loss_apply_to', type=str, default='pred_target_imgs_lf_detached, target_imgs')

        # Loss parameters
        parser.add('--per_loss_type', type=str, default='l1')
        parser.add('--per_loss_weights', type=str, default='10.0, 0.01')
        parser.add('--per_layer_weights', type=str, default='0.03125, 0.0625, 0.125, 0.25, 1.0')
        parser.add('--per_loss_names', type=str, default='VGG19, VGGFace')

    def __init__(self, args):
        super(LossWrapper, self).__init__()
        ### Define losses ###
        losses = {
            'mse': F.mse_loss,
            'l1': F.l1_loss}

        self.loss = losses[args.per_loss_type]

        # Weights for each feature extractor
        self.weights = rn_utils.parse_str_to_list(args.per_loss_weights, value_type=float, sep=',')
        self.layer_weights = rn_utils.parse_str_to_list(args.per_layer_weights, value_type=float, sep=',')
        self.names = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.per_loss_names, sep=';')]

        ### Define extractors ###
        self.apply_to = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.per_loss_apply_to, sep=';')]
        weights_dir = pathlib.Path(args.project_dir) / 'pretrained_weights' / 'perceptual'

        # Architectures for the supported networks 
        networks = {
            'vgg16': models.vgg16,
            'vgg19': models.vgg19}

        # Build a list of used networks
        self.nets = nn.ModuleList()
        self.full_net_names = rn_utils.parse_str_to_list(args.per_full_net_names, sep=',')

        for full_net_name in self.full_net_names:
            net_name, dataset_name, framework_name = full_net_name.split('_')

            if dataset_name == 'imagenet' and framework_name == 'pytorch':
                self.nets.append(networks[net_name](pretrained=True))
                mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None] * 2 - 1
                std  = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None] * 2
            
            elif framework_name == 'caffe':
                self.nets.append(networks[net_name]())
                self.nets[-1].load_state_dict(torch.load(weights_dir / f'{full_net_name}.pth'))
                self.nets[-1] = self.nets[-1]
                mean = torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 127.5 - 1
                std  = torch.FloatTensor([     1.,      1.,      1.])[None, :, None, None] / 127.5
            
            # Register means and stds as buffers
            self.register_buffer(f'{full_net_name}_mean', mean)
            self.register_buffer(f'{full_net_name}_std', std)

        # Perform the slicing according to the required layers
        for n, (net, block_idx) in enumerate(zip(self.nets, rn_utils.parse_str_to_list(args.per_net_layers, sep=';'))):
            net_blocks = nn.ModuleList()

            # Parse indices of slices
            block_idx = rn_utils.parse_str_to_list(block_idx, value_type=int, sep=',')
            for i, idx in enumerate(block_idx):
                block_idx[i] = idx

            # Slice conv blocks
            layers = []
            for i, layer in enumerate(net.features):
                if layer.__class__.__name__ == 'MaxPool2d' and args.per_pooling == 'avgpool':
                    layer = nn.AvgPool2d(2)
                layers.append(layer)
                if i in block_idx:
                    net_blocks.append(nn.Sequential(*layers))
                    layers = []

            # Add layers for prediction of the scores (if needed)
            if block_idx[-1] == 'fc':
                layers.extend([
                    nn.AdaptiveAvgPool2d(7),
                    utils.Flatten(1)])
                for layer in net.classifier:
                    layers.append(layer)
                net_blocks.append(nn.Sequential(*layers))

            # Store sliced net
            self.nets[n] = net_blocks

    def forward(self, data_dict, losses_dict):
        for i, (tensor_name, target_tensor_name) in enumerate(self.apply_to):
            # Extract inputs
            real_imgs = data_dict[target_tensor_name]
            fake_imgs = data_dict[tensor_name]

            # Prepare inputs
            b, t, c, h, w = real_imgs.shape
            real_imgs = real_imgs.view(-1, c, h, w)
            fake_imgs = fake_imgs.view(-1, c, h, w)

            with torch.no_grad():
                real_feats_ext = self.forward_extractor(real_imgs)

            fake_feats_ext = self.forward_extractor(fake_imgs)

            # Calculate the loss
            for n in range(len(self.names[i])):
                loss = 0
                for real_feats, fake_feats, weight in zip(real_feats_ext[n], fake_feats_ext[n], self.layer_weights):
                    loss += self.loss(fake_feats, real_feats.detach()) * weight
                loss *= self.weights[n]

                losses_dict[f'G_{self.names[i][n]}'] = loss

        return losses_dict

    def forward_extractor(self, imgs):
        # Calculate features
        feats = []
        for net, full_net_name in zip(self.nets, self.full_net_names):
            # Preprocess input image
            mean = getattr(self, f'{full_net_name}_mean')
            std = getattr(self, f'{full_net_name}_std')
            feats.append([(imgs - mean) / std])

            # Forward pass through blocks
            for block in net:
                feats[-1].append(block(feats[-1][-1]))

            # Remove input image
            feats[-1].pop(0)

        return feats

# Third party
import torch
from torch import nn
import torch.nn.functional as F

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--seg_loss_type', type=str, default='bce')
        parser.add('--seg_loss_weights', type=float, default=10.)
        parser.add('--seg_loss_apply_to', type=str, default='pred_target_inf_segs_logits, target_segs', help='can specify multiple tensor names from data_dict')
        parser.add('--seg_loss_names', type=str, default='BCE', help='name for each loss')

    def __init__(self, args):
        super(LossWrapper, self).__init__()   
        self.apply_to = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.seg_loss_apply_to, sep=';')]
        self.names = rn_utils.parse_str_to_list(args.seg_loss_names, sep=',')

        # Supported loss functions
        losses = {
            'bce': F.binary_cross_entropy_with_logits,
            'dice': lambda fake_seg, real_seg: torch.log((fake_seg**2).sum() + (real_seg**2).sum()) - torch.log((2 * fake_seg * real_seg).sum())}

        self.loss = losses[args.seg_loss_type]

        self.weights = args.seg_loss_weights

        self.eps = args.eps

    def forward(self, data_dict, losses_dict):
        for i, (tensor_name, target_tensor_name) in enumerate(self.apply_to):
            real_segs = data_dict[target_tensor_name]
            fake_segs = data_dict[tensor_name]

            b, t = fake_segs.shape[:2]
            fake_segs = fake_segs.view(b*t, *fake_segs.shape[2:])

            if 'HalfTensor' in fake_segs.type():  
                real_segs = real_segs.type(fake_segs.type())

            real_segs = real_segs.view(b*t, *real_segs.shape[2:])

            losses_dict['G_' + self.names[i]] = self.loss(fake_segs, real_segs) * self.weights

        return losses_dict
# Third party
import torch
from torch import nn
import torch.nn.functional as F
from joblib import Parallel, delayed
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import sys
import face_alignment

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--nme_num_threads', type=int, default=8)
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.num_threads = args.nme_num_threads

        # Supported loss functions
        losses = {
            'mse': F.mse_loss,
            'l1': F.l1_loss}

        self.fa = []

        for i in range(self.num_threads):
            self.fa.append(face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda'))

        # Used to calculate a normalization factor
        self.right_eye = list(range(36, 42))
        self.left_eye = list(range(42, 48))

    def forward(self, data_dict, losses_dict):
        fake_imgs = data_dict['pred_target_imgs']
        real_imgs = data_dict['target_imgs']

        b, t = real_imgs.shape[:2]

        fake_imgs = fake_imgs.view(b*t, *fake_imgs.shape[2:])
        real_imgs = real_imgs.view(b*t, *real_imgs.shape[2:])

        losses = [self.calc_metric(fake_img, real_img, i % self.num_threads) for i, (fake_img, real_img) in enumerate(zip(fake_imgs, real_imgs))]

        losses_dict['G_PME'] = sum(losses) / len(losses)

        return losses_dict

    @torch.no_grad()
    def calc_metric(self, fake_img, real_img, worker_id):
        fake_img = (((fake_img.detach() + 1.0) / 2.0) * 255.0).cpu().numpy().astype('uint8').transpose(1, 2, 0)
        fake_keypoints = torch.from_numpy(self.fa[worker_id].get_landmarks(fake_img)[0])[:, :2]

        real_img = (((real_img.detach() + 1.0) / 2.0) * 255.0).cpu().numpy().astype('uint8').transpose(1, 2, 0)
        real_keypoints = torch.from_numpy(self.fa[worker_id].get_landmarks(real_img)[0])[:, :2]

        # Calcualte normalization factor
        d = ((real_keypoints[self.left_eye].mean(0) - real_keypoints[self.right_eye].mean(0))**2).sum()**0.5

        # Calculate the mean error
        error = torch.mean(((fake_keypoints - real_keypoints)**2).sum(1)**0.5)

        loss = error / d

        return loss

# Third party
import torch
from torch import nn
import torch.nn.functional as F

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--pix_loss_type', type=str, default='l1')
        parser.add('--pix_loss_weights', type=str, default='10.0', help='comma separated floats')
        parser.add('--pix_loss_apply_to', type=str, default='pred_target_delta_lf_rgbs, target_imgs', help='can specify multiple tensor names from data_dict')
        parser.add('--pix_loss_names', type=str, default='L1', help='name for each loss')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.apply_to = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.pix_loss_apply_to, sep=';')]
        
        # Supported loss functions
        losses = {
            'mse': F.mse_loss,
            'l1': F.l1_loss,
            'ce': F.cross_entropy}

        self.loss = losses[args.pix_loss_type]

        # Weights for each feature extractor
        self.weights = rn_utils.parse_str_to_list(args.pix_loss_weights, value_type=float)
        self.names = rn_utils.parse_str_to_list(args.pix_loss_names)

    def forward(self, data_dict, losses_dict):
        for i, (tensor_name, target_tensor_name) in enumerate(self.apply_to):
            real_imgs = data_dict[target_tensor_name]
            fake_imgs = data_dict[tensor_name]

            b, t = fake_imgs.shape[:2]
            fake_imgs = fake_imgs.view(b*t, *fake_imgs.shape[2:])

            if 'HalfTensor' in fake_imgs.type():  
                real_imgs = real_imgs.type(fake_imgs.type())

            real_imgs = real_imgs.view(b*t, *real_imgs.shape[2:])

            loss = self.loss(fake_imgs, real_imgs.detach())

            losses_dict['G_' + self.names[i]] = loss * self.weights[i]

        return losses_dict
import os
import numpy as np
import pickle
import tensorboardX
import pathlib
from torchvision import transforms



class Logger(object):
    def __init__(self, args, experiment_dir):
        super(Logger, self).__init__()
        self.num_iter = {'train': 0, 'test': 0}
        
        self.no_disk_write_ops = args.no_disk_write_ops
        self.rank = args.rank

        if not self.no_disk_write_ops:
            self.experiment_dir = experiment_dir

            for phase in ['train', 'test']:
                os.makedirs(experiment_dir / 'images' / phase, exist_ok=True)

            self.to_image = transforms.ToPILImage()

            if args.rank == 0:
                if args.which_epoch != 'none' and args.init_experiment_dir == '':
                    self.losses = pickle.load(open(self.experiment_dir / 'losses.pkl', 'rb'))
                else:
                    self.losses = {}
                
                self.writer = tensorboardX.SummaryWriter('/tensorboard')

    def output_logs(self, phase, visuals, losses, time):
        if not self.no_disk_write_ops:
            # Increment iter counter
            self.num_iter[phase] += 1

            # Save visuals
            self.to_image(visuals).save(self.experiment_dir / 'images' / phase / ('%04d_%02d.jpg' % (self.num_iter[phase], self.rank)))

            if self.rank != 0:
                return

            self.writer.add_image(f'results_{phase}', visuals, self.num_iter[phase])

            # Save losses
            for key, value in losses.items():
                if key in self.losses:
                    self.losses[key].append(value)
                else:
                    self.losses[key] = [value]

                self.writer.add_scalar(f'{key}_{phase}', value, self.num_iter[phase])

            # Save losses
            pickle.dump(self.losses, open(self.experiment_dir / 'losses.pkl', 'wb'))

        elif self.rank != 0:
            return

        # Print losses
        print(', '.join('%s: %.3f' % (key, value) for key, value in losses.items()) + ', time: %.3f' % time)

    def set_num_iter(self, train_iter, test_iter):
        self.num_iter = {
            'train': train_iter,
            'test': test_iter}
import torch
from torch import nn

import argparse
import os
import pathlib
import importlib
import ssl
import time
import copy
import sys

from datasets import utils as ds_utils
from networks import utils as nt_utils
from runners import utils as rn_utils
from logger import Logger



class TrainingWrapper(object):
    @staticmethod
    def get_args(parser):
        # General options
        parser.add('--project_dir',              default='.', type=str,
                                                 help='root directory of the code')

        parser.add('--torch_home',               default='', type=str,
                                                 help='directory used for storage of the checkpoints')

        parser.add('--experiment_name',          default='test', type=str,
                                                 help='name of the experiment used for logging')

        parser.add('--dataloader_name',          default='voxceleb2', type=str,
                                                 help='name of the file in dataset directory which is used for data loading')

        parser.add('--dataset_name',             default='voxceleb2_512px', type=str,
                                                 help='name of the dataset in the data root folder')

        parser.add('--data_root',                default=".", type=str,
                                                 help='root directory of the data')

        parser.add('--debug',                    action='store_true',
                                                 help='turn on the debug mode: fast epoch, useful for testing')

        parser.add('--runner_name',              default='default', type=str,
                                                 help='class that wraps the models and performs training and inference steps')

        parser.add('--no_disk_write_ops',        action='store_true',
                                                 help='avoid doing write operations to disk')

        parser.add('--redirect_print_to_file',   action='store_true',
                                                 help='redirect stdout and stderr to file')

        parser.add('--random_seed',              default=0, type=int,
                                                 help='used for initialization of pytorch and numpy seeds')

        # Initialization options
        parser.add('--init_experiment_dir',      default='', type=str,
                                                 help='directory of the experiment used for the initialization of the networks')

        parser.add('--init_networks',            default='', type=str,
                                                 help='list of networks to intialize')

        parser.add('--init_which_epoch',         default='none', type=str,
                                                 help='epoch to initialize from')

        parser.add('--which_epoch',              default='none', type=str,
                                                 help='epoch to continue training from')

        # Distributed options
        parser.add('--num_gpus',                 default=1, type=int,
                                                 help='>1 enables DDP')

        # Training options
        parser.add('--num_epochs',               default=1000, type=int,
                                                 help='number of epochs for training')

        parser.add('--checkpoint_freq',          default=25, type=int,
                                                 help='frequency of checkpoints creation in epochs')

        parser.add('--test_freq',                default=5, type=int, 
                                                 help='frequency of testing in epochs')
        
        parser.add('--batch_size',               default=1, type=int,
                                                 help='batch size across all GPUs')
        
        parser.add('--num_workers_per_process',  default=20, type=int,
                                                 help='number of workers used for data loading in each process')
        
        parser.add('--skip_test',                action='store_true',
                                                 help='do not perform testing')
        
        parser.add('--calc_stats',               action='store_true',
                                                 help='calculate batch norm standing stats')
        
        parser.add('--visual_freq',              default=-1, type=int, 
                                                 help='in iterations, -1 -- output logs every epoch')

        # Mixed precision options
        parser.add('--use_half',                 action='store_true',
                                                 help='enable half precision calculation')
        
        parser.add('--use_closure',              action='store_true',
                                                 help='use closure function during optimization (required by LBFGS)')
        
        parser.add('--use_apex',                 action='store_true',
                                                 help='enable apex')
        
        parser.add('--amp_opt_level',            default='O0', type=str,
                                                 help='full/mixed/half precision, refer to apex.amp docs')
        
        parser.add('--amp_loss_scale',           default='dynamic', type=str,
                                                 help='fixed or dynamic loss scale')

        # Technical options that are set automatically
        parser.add('--local_rank', default=0, type=int)
        parser.add('--rank',       default=0, type=int)
        parser.add('--world_size', default=1, type=int)
        parser.add('--train_size', default=1, type=int)

        # Dataset options
        args, _ = parser.parse_known_args()

        os.environ['TORCH_HOME'] = args.torch_home

        importlib.import_module(f'datasets.{args.dataloader_name}').DatasetWrapper.get_args(parser)

        # runner options
        importlib.import_module(f'runners.{args.runner_name}').RunnerWrapper.get_args(parser)

        return parser

    def __init__(self, args, runner=None):
        super(TrainingWrapper, self).__init__()
        # Initialize and apply general options
        ssl._create_default_https_context = ssl._create_unverified_context
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        # Set distributed training options
        if args.num_gpus > 1 and args.num_gpus <= 8:
            args.rank = args.local_rank
            args.world_size = args.num_gpus
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        elif args.num_gpus > 8:
            raise # Not supported

        # Prepare experiment directories and save options
        project_dir = pathlib.Path(args.project_dir)
        self.checkpoints_dir = project_dir / 'runs' / args.experiment_name / 'checkpoints'

        # Store options
        if not args.no_disk_write_ops:
            os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.experiment_dir = project_dir / 'runs' / args.experiment_name

        if not args.no_disk_write_ops:
            # Redirect stdout
            if args.redirect_print_to_file:
                logs_dir = self.experiment_dir / 'logs'
                os.makedirs(logs_dir, exist_ok=True)
                sys.stdout = open(os.path.join(logs_dir, f'stdout_{args.rank}.txt'), 'w')
                sys.stderr = open(os.path.join(logs_dir, f'stderr_{args.rank}.txt'), 'w')

            if args.rank == 0:
                print(args)
                with open(self.experiment_dir / 'args.txt', 'wt') as args_file:
                    for k, v in sorted(vars(args).items()):
                        args_file.write('%s: %s\n' % (str(k), str(v)))

        # Initialize model
        self.runner = runner

        if self.runner is None:
            self.runner = importlib.import_module(f'runners.{args.runner_name}').RunnerWrapper(args)

        # Load pre-trained weights (if needed)
        init_networks = rn_utils.parse_str_to_list(args.init_networks) if args.init_networks else {}
        networks_to_train = self.runner.nets_names_to_train

        if args.init_which_epoch != 'none' and args.init_experiment_dir:
            for net_name in init_networks:
                self.runner.nets[net_name].load_state_dict(torch.load(pathlib.Path(args.init_experiment_dir) / 'checkpoints' / f'{args.init_which_epoch}_{net_name}.pth', map_location='cpu'))

        if args.which_epoch != 'none':
            for net_name in networks_to_train:
                if net_name not in init_networks:
                    self.runner.nets[net_name].load_state_dict(torch.load(self.checkpoints_dir / f'{args.which_epoch}_{net_name}.pth', map_location='cpu'))

        if args.num_gpus > 0:
            self.runner.cuda()

        if args.rank == 0:
            print(self.runner)

    def train(self, args):
        # Reset amp
        if args.use_apex:
            from apex import amp
            
            amp.init(False)

        # Get dataloaders
        train_dataloader = ds_utils.get_dataloader(args, 'train')
        if not args.skip_test:
            test_dataloader = ds_utils.get_dataloader(args, 'test')

        model = runner = self.runner

        if args.use_half:
            runner.half()

        # Initialize optimizers, schedulers and apex
        opts = runner.get_optimizers(args)

        # Load pre-trained params for optimizers and schedulers (if needed)
        if args.which_epoch != 'none' and not args.init_experiment_dir:
            for net_name, opt in opts.items():
                opt.load_state_dict(torch.load(self.checkpoints_dir / f'{args.which_epoch}_opt_{net_name}.pth', map_location='cpu'))

        if args.use_apex and args.num_gpus > 0 and args.num_gpus <= 8:
            # Enfornce apex mixed precision settings
            nets_list, opts_list = [], []
            for net_name in sorted(opts.keys()):
                nets_list.append(runner.nets[net_name])
                opts_list.append(opts[net_name])

            loss_scale = float(args.amp_loss_scale) if args.amp_loss_scale != 'dynamic' else args.amp_loss_scale

            nets_list, opts_list = amp.initialize(nets_list, opts_list, opt_level=args.amp_opt_level, num_losses=1, loss_scale=loss_scale)

            # Unpack opts_list into optimizers
            for net_name, net, opt in zip(sorted(opts.keys()), nets_list, opts_list):
                runner.nets[net_name] = net
                opts[net_name] = opt

            if args.which_epoch != 'none' and not args.init_experiment_dir and os.path.exists(self.checkpoints_dir / f'{args.which_epoch}_amp.pth'):
                amp.load_state_dict(torch.load(self.checkpoints_dir / f'{args.which_epoch}_amp.pth', map_location='cpu'))

        # Initialize apex distributed data parallel wrapper
        if args.num_gpus > 1 and args.num_gpus <= 8:
            from apex import parallel

            model = parallel.DistributedDataParallel(runner, delay_allreduce=True)

        epoch_start = 1 if args.which_epoch == 'none' else int(args.which_epoch) + 1

        # Initialize logging
        train_iter = epoch_start - 1

        if args.visual_freq != -1:
            train_iter /= args.visual_freq

        logger = Logger(args, self.experiment_dir)
        logger.set_num_iter(
            train_iter=train_iter, 
            test_iter=(epoch_start - 1) // args.test_freq)

        if args.debug and not args.use_apex:
            torch.autograd.set_detect_anomaly(True)

        total_iters = 1

        for epoch in range(epoch_start, args.num_epochs + 1):
            if args.rank == 0: 
                print('epoch %d' % epoch)

            # Train for one epoch
            model.train()
            time_start = time.time()

            # Shuffle the dataset before the epoch
            train_dataloader.dataset.shuffle()

            for i, data_dict in enumerate(train_dataloader, 1):               
                # Prepare input data
                if args.num_gpus > 0 and args.num_gpus > 0:
                    for key, value in data_dict.items():
                        data_dict[key] = value.cuda()

                # Convert inputs to FP16
                if args.use_half:
                    for key, value in data_dict.items():
                        data_dict[key] = value.half()

                output_logs = i == len(train_dataloader)

                if args.visual_freq != -1:
                    output_logs = not (total_iters % args.visual_freq)

                output_visuals = output_logs and not args.no_disk_write_ops

                # Accumulate list of optimizers that will perform opt step
                for opt in opts.values():
                    opt.zero_grad()

                # Perform a forward pass
                if not args.use_closure:
                    loss = model(data_dict)
                    closure = None

                if args.use_apex and args.num_gpus > 0 and args.num_gpus <= 8:
                    # Mixed precision requires a special wrapper for the loss
                    with amp.scale_loss(loss, opts.values()) as scaled_loss:
                        scaled_loss.backward()

                elif not args.use_closure:
                    loss.backward()

                else:
                    def closure():
                        loss = model(data_dict)
                        loss.backward()
                        return loss

                # Perform steps for all optimizers
                for opt in opts.values():
                    opt.step(closure)

                if output_logs:
                    logger.output_logs('train', runner.output_visuals(), runner.output_losses(), time.time() - time_start)

                    if args.debug:
                        break

                if args.visual_freq != -1:
                    total_iters += 1
                    total_iters %= args.visual_freq
            
            # Increment the epoch counter in the training dataset
            train_dataloader.dataset.epoch += 1

            # If testing is not required -- continue
            if epoch % args.test_freq:
                continue

            # If skip test flag is set -- only check if a checkpoint if required
            if not args.skip_test:
                # Calculate "standing" stats for the batch normalization
                if args.calc_stats:
                    runner.calculate_batchnorm_stats(train_dataloader, args.debug)

                # Test
                time_start = time.time()
                model.eval()

                for data_dict in test_dataloader:
                    # Prepare input data
                    if args.num_gpus > 0:
                        for key, value in data_dict.items():
                            data_dict[key] = value.cuda()

                    # Forward pass
                    with torch.no_grad():
                        model(data_dict)
                    
                    if args.debug:
                        break

            # Output logs
            logger.output_logs('test', runner.output_visuals(), runner.output_losses(), time.time() - time_start)
            
            # If creation of checkpoint is not required -- continue
            if epoch % args.checkpoint_freq and not args.debug:
                continue

            # Create or load a checkpoint
            if args.rank == 0  and not args.no_disk_write_ops:
                with torch.no_grad():
                    for net_name in runner.nets_names_to_train:
                        # Save a network
                        torch.save(runner.nets[net_name].state_dict(), self.checkpoints_dir / f'{epoch}_{net_name}.pth')

                        # Save an optimizer
                        torch.save(opts[net_name].state_dict(), self.checkpoints_dir / f'{epoch}_opt_{net_name}.pth')

                    # Save amp
                    if args.use_apex:
                        torch.save(amp.state_dict(), self.checkpoints_dir / f'{epoch}_amp.pth')

        return runner

if __name__ == "__main__":
    ## Parse options ##
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    TrainingWrapper.get_args(parser)

    args, _ = parser.parse_known_args()

    ## Initialize the model ##
    m = TrainingWrapper(args)

    ## Perform training ##
    nets = m.train(args)

# Third party
import torch
from torch import nn
import torch.nn.functional as F
import math
import time

from runners import utils as rn_utils
from networks import utils as nt_utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--enh_num_channels',            default=64, type=int, 
                                                    help='minimum number of channels')

        parser.add('--enh_max_channels',            default=128, type=int, 
                                                    help='maximum number of channels')

        parser.add('--enh_bottleneck_tensor_size',  default=128, type=int, 
                                                    help='spatial size of the tensor in the bottleneck')

        parser.add('--enh_num_blocks',              default=8, type=int, 
                                                    help='number of convolutional blocks at the bottleneck resolution')

        parser.add('--enh_unrolling_depth',         default=4, type=int, 
                                                    help='number of consequtive unrolling iterations')

        parser.add('--enh_guiding_rgb_loss_type',   default='sse', type=str, choices=['sse', 'l1'],
                                                    help='lightweight loss that guides the enhates of the rgb texture')

        parser.add('--enh_detach_inputs',           default='True', type=rn_utils.str2bool, choices=[True, False],
                                                    help='detach input tensors (for efficient training)')

        parser.add('--enh_norm_layer_type',         default='none', type=str,
                                                    help='norm layer inside the enhancer')

        parser.add('--enh_activation_type',         default='leakyrelu', type=str,
                                                    help='activation layer inside the enhancer')

        parser.add('--enh_downsampling_type',       default='avgpool', type=str,
                                                    help='downsampling layer inside the enhancer')

        parser.add('--enh_upsampling_type',         default='nearest', type=str,
                                                    help='upsampling layer inside the enhancer')

        parser.add('--enh_apply_masks',             default='True', type=rn_utils.str2bool, choices=[True, False],
                                                    help='apply segmentation masks to predicted and ground-truth images')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args

        self.net = Generator(args)

        rgb_losses = {
            'sse': lambda fake, real: ((real - fake)**2).sum() / 2,
            'l1': lambda fake, real: (real - fake).abs().sum()}

        self.rgb_loss = rgb_losses[args.enh_guiding_rgb_loss_type]

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:

        # Do not store activations if this network is not being trained
        if 'texture_enhancer' not in networks_to_train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        source_uvs = data_dict['pred_source_uvs']
        source_delta_lf_rgbs = data_dict['pred_source_delta_lf_rgbs']
        source_imgs = data_dict['source_imgs']

        enh_tex_hf_rgbs = data_dict['pred_tex_hf_rgbs'].clone()

        target_uvs = data_dict['pred_target_uvs']
        pred_target_delta_lf_rgbs = data_dict['pred_target_delta_lf_rgbs']
        
        if self.args.enh_apply_masks and self.args.inf_pred_segmentation:
            target_imgs = data_dict['target_imgs']
            pred_target_imgs = data_dict['pred_target_imgs']

        if self.args.inf_pred_segmentation:
            pred_source_segs = data_dict['pred_source_segs']
            pred_target_segs = data_dict['pred_target_segs']

        # Reshape inputs
        b, t, c, h, w = pred_target_delta_lf_rgbs.shape
        n = source_uvs.shape[1]

        source_uvs = source_uvs.view(b*n, h, w, 2)
        source_delta_lf_rgbs = source_delta_lf_rgbs.view(b*n, c, h, w)
        source_imgs = source_imgs.view(b*n, c, h, w)

        enh_tex_hf_rgbs = enh_tex_hf_rgbs[:, 0]

        if self.args.enh_detach_inputs:
            source_uvs = source_uvs.detach()
            source_delta_lf_rgbs = source_delta_lf_rgbs.detach()
            enh_tex_hf_rgbs = enh_tex_hf_rgbs.detach()

        if self.args.enh_apply_masks and self.args.inf_pred_segmentation:
            target_imgs = target_imgs.view(b*t, c, h, w)
            pred_target_imgs = pred_target_imgs.view(b*t, c, h, w)
        
        target_uvs = target_uvs.view(b*t, h, w, 2)
        pred_target_delta_lf_rgbs = pred_target_delta_lf_rgbs.view(b*t, c, h, w)

        if self.args.enh_detach_inputs:
            target_uvs = target_uvs.detach()
            pred_target_delta_lf_rgbs = pred_target_delta_lf_rgbs.detach()

        if self.args.inf_pred_segmentation:
            pred_source_segs = pred_source_segs.view(b*n, 1, h, w)
            pred_target_segs = pred_target_segs.view(b*t, 1, h, w)

            source_imgs = source_imgs * pred_source_segs + (-1) * (1 - pred_source_segs)

            if self.args.enh_detach_inputs:
                pred_source_segs = pred_source_segs.detach()
                pred_target_segs = pred_target_segs.detach()

        for i in range(self.args.enh_unrolling_depth):
            # Calculation of gradients is required for enhancer losses
            prev_enh = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

            # Repeat the texture n times for n train frames
            enh_tex_hf_rgbs_i = torch.cat([enh_tex_hf_rgbs[:, None]]*n, dim=1).view(b*n, *enh_tex_hf_rgbs.shape[1:])
            enh_tex_hf_rgbs_i_grad = enh_tex_hf_rgbs_i.clone().detach()
            enh_tex_hf_rgbs_i_grad.requires_grad = True

            # Current approximation of the source image
            pred_source_imgs = source_delta_lf_rgbs.detach() + F.grid_sample(enh_tex_hf_rgbs_i_grad, source_uvs.detach())

            if self.args.inf_pred_segmentation:
                pred_source_imgs = pred_source_imgs * pred_source_segs + (-1) * (1 - pred_source_segs)

            # Calculate the gradients with respect to the enhancer losses
            loss_enh = self.rgb_loss(pred_source_imgs, source_imgs.detach())

            loss_enh.backward()

            # Forward pass through the enhancer network
            inputs = torch.cat([
                enh_tex_hf_rgbs_i,
                enh_tex_hf_rgbs_i_grad.grad.detach()],
                dim=1)

            torch.set_grad_enabled(prev_enh)

            outputs = self.net(inputs)

            # Update the texture
            delta_enh_tex_hf_rgbs_i = torch.tanh(outputs[:, :3])

            # Aggregate data (if needed)
            if delta_enh_tex_hf_rgbs_i.shape[0] == b*n:
                delta_enh_tex_hf_rgbs_i = delta_enh_tex_hf_rgbs_i.view(b, n, *delta_enh_tex_hf_rgbs_i.shape[1:]).mean(dim=1)

            enh_tex_hf_rgbs = enh_tex_hf_rgbs + delta_enh_tex_hf_rgbs_i

        # Evaluate on real frames
        enh_tex_hf_rgbs = torch.cat([enh_tex_hf_rgbs[:, None]]*t, 1)
        enh_tex_hf_rgbs = enh_tex_hf_rgbs.view(b*t, *enh_tex_hf_rgbs.shape[2:])

        pred_enh_target_delta_hf_rgbs = F.grid_sample(enh_tex_hf_rgbs, target_uvs) # high-freq. component
        pred_enh_target_imgs = pred_target_delta_lf_rgbs + pred_enh_target_delta_hf_rgbs # final image

        if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
            # Get an image with a low-frequency component detached
            pred_enh_target_imgs_lf_detached = pred_target_delta_lf_rgbs.detach() + pred_enh_target_delta_hf_rgbs

        if self.args.inf_pred_segmentation and self.args.enh_apply_masks:
            pred_target_masks = pred_target_segs.detach()

            # Apply segmentation predicted by the main model
            pred_target_imgs = pred_target_imgs * pred_target_segs + (-1) * (1 - pred_target_segs)

            # Apply possbily enhanced segmentation
            target_imgs = target_imgs * pred_target_masks + (-1) * (1- pred_target_masks)
            pred_enh_target_imgs = pred_enh_target_imgs * pred_target_masks + (-1) * (1 - pred_target_masks)

            if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
                pred_enh_target_imgs_lf_detached = pred_enh_target_imgs_lf_detached * pred_target_masks + (-1) * (1 - pred_target_masks)

        if 'texture_enhancer' not in networks_to_train:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])

        data_dict['pred_enh_tex_hf_rgbs'] = reshape_target_data(enh_tex_hf_rgbs)

        data_dict['pred_enh_target_imgs'] = reshape_target_data(pred_enh_target_imgs)

        # Output debugging results
        data_dict['pred_enh_target_delta_hf_rgbs'] = reshape_target_data(pred_enh_target_delta_hf_rgbs)

        if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
            data_dict['pred_enh_target_imgs_lf_detached'] = reshape_target_data(pred_enh_target_imgs_lf_detached)

        if self.args.enh_apply_masks and self.args.inf_pred_segmentation:
            data_dict['target_imgs'] = reshape_target_data(target_imgs)
            data_dict['pred_target_imgs'] = reshape_target_data(pred_target_imgs)

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = []

        return visuals

    def __repr__(self):
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        output = self.net.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        num_down_blocks = int(math.log(args.image_size // args.enh_bottleneck_tensor_size, 2))

        # Initialize the residual blocks
        layers = []

        in_channels = 6
        out_channels = args.enh_num_channels

        layers = [nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1)]

        for i in range(1, num_down_blocks + 1):
            in_channels = out_channels
            out_channels = min(int(args.enh_num_channels * 2**i), args.enh_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.enh_activation_type, 
                norm_layer_type=args.enh_norm_layer_type,
                resize_layer_type=args.enh_downsampling_type)]

        for i in range(args.enh_num_blocks):
            layers += [nt_utils.ResBlock(
                in_channels=out_channels, 
                out_channels=out_channels,
                eps=args.eps,
                activation_type=args.enh_activation_type, 
                norm_layer_type=args.enh_norm_layer_type,
                resize_layer_type='none',
                frames_per_person=args.num_source_frames,
                output_aggregated=i == args.enh_num_blocks - 1)]

        for i in range(num_down_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(args.enh_num_channels * 2**i), args.enh_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.enh_activation_type, 
                norm_layer_type=args.enh_norm_layer_type,
                resize_layer_type=args.enh_upsampling_type)]

        in_channels = out_channels
        out_channels = 3

        norm_layer = nt_utils.norm_layers[args.enh_norm_layer_type]
        activation = nt_utils.activations[args.enh_activation_type]

        layers += [
            norm_layer(out_channels),
            activation(inplace=True),
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1)]

        self.blocks = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.blocks(inputs)

        return outputs

import torch
from torch import nn
import torch.nn.functional as F
import math
import functools



############################################################
# PixelUnShuffle layer from https://github.com/cszn/FFDNet #
# Should be removed after it is implemented in PyTorch     #
############################################################

def pixel_unshuffle(inputs, upscale_factor):
    batch_size, channels, in_height, in_width = inputs.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = inputs.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        return pixel_unshuffle(inputs, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

############################################################
#                      Adaptive layers                     #
############################################################

class AdaptiveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(AdaptiveConv2d, self).__init__()
        # Set options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.finetuning = False # set to True by prep_adanorm_for_finetuning method
        
    def forward(self, inputs):
        # Cast parameters into inputs.dtype
        if inputs.type() != self.weight.type():
            self.weight = self.weight.type(inputs.type())
            self.bias = self.bias.type(inputs.type())

        # Reshape parameters into inputs shape
        if self.weight.shape[0] != inputs.shape[0]:
            b = self.weight.shape[0]
            t = inputs.shape[0] // b
            weight = self.weight[:, None].repeat(1, t, 1, 1, 1, 1).view(b*t, *self.weight.shape[1:])
            bias = self.bias[:, None].repeat(1, t, 1).view(b*t, self.bias.shape[1])

        else:
            weight = self.weight
            bias = self.bias

        # Apply convolution
        if self.kernel_size > 1:
            outputs = []
            for i in range(inputs.shape[0]):
                outputs.append(F.conv2d(inputs[i:i+1], weight[i], bias[i], 
                                        self.stride, self.padding, self.dilation, self.groups))
            outputs = torch.cat(outputs, 0)

        else:
            b, c, h, w = inputs.shape
            weight = weight[:, :, :, 0, 0].transpose(1, 2)
            outputs = torch.bmm(inputs.view(b, c, -1).transpose(1, 2), weight).transpose(1, 2).view(b, -1, h, w)
            outputs = outputs + bias[..., None, None]

        return outputs

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        
        if self.padding != 0:
            s += ', padding={padding}'

        if self.dilation != 1:
            s += ', dilation={dilation}'
        
        if self.groups != 1:
            s += ', groups={groups}'
        
        return s.format(**self.__dict__)

class AdaptiveBias(nn.Module):
    def __init__(self, num_features, spatial_size, weight):
        super(AdaptiveBias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, num_features, spatial_size, spatial_size))

        self.conv = AdaptiveConv2d(num_features, num_features, 1, 1, 0)

        # Init biases
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        b = self.conv.weight.shape[0]
        bias = torch.cat([self.bias]*b)
        bias = self.conv(bias)

        if b != inputs.shape[0]:
            n = inputs.shape[0] // b
            inputs = inputs.view(b, n, *inputs.shape[1:])
            outputs = inputs + self.bias[:, None, :]
            outputs = outputs.view(b*n, *outputs.shape[2:])

        else:
            outputs = inputs + self.bias

        return outputs


class AdaptiveNorm2d(nn.Module):
    def __init__(self, num_features, spatial_size, norm_layer_type, eps=1e-4):
        super(AdaptiveNorm2d, self).__init__()
        # Set options
        self.num_features = num_features
        self.spatial_size = spatial_size
        self.norm_layer_type = norm_layer_type
        self.finetuning = False # set to True by prep_adanorm_for_finetuning method

        if 'spade' in self.norm_layer_type:
            self.pixel_feats = nn.Parameter(torch.empty(1, num_features, spatial_size, spatial_size))
            nn.init.kaiming_uniform_(self.pixel_feats, a=math.sqrt(5))

            self.conv_weight = AdaptiveConv2d(num_features, num_features, 1, 1, 0)
            self.conv_bias = AdaptiveConv2d(num_features, num_features, 1, 1, 0)

        # Supported normalization layers
        norm_layers = {
            'bn': lambda num_features: nn.BatchNorm2d(num_features, eps=eps, affine=False),
            'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=eps, affine=False),
            'none': lambda num_features: nn.Identity()}

        self.norm_layer = norm_layers[self.norm_layer_type.replace('spade_', '')](num_features)
        
    def forward(self, inputs):
        outputs = self.norm_layer(inputs)

        if 'spade' in self.norm_layer_type:
            b = self.conv_weight.weight.shape[0]
            pixel_feats = torch.cat([self.pixel_feats]*b)
            self.weight = self.conv_weight(pixel_feats) + 1.0
            self.bias = self.conv_bias(pixel_feats)

        if len(self.weight.shape) == 2:
            self.weight = self.weight[..., None, None]
            self.bias = self.bias[..., None, None]

        if outputs.type() != self.weight.type():
            # Cast parameters into outputs.dtype
            self.weight = self.weight.type(outputs.type())
            self.bias = self.bias.type(outputs.type())

        if self.weight.shape[0] != outputs.shape[0]:
            b = self.weight.shape[0]
            n = outputs.shape[0] // b
            outputs = outputs.view(b, n, *outputs.shape[1:])
            outputs = outputs * self.weight[:, None] + self.bias[:, None]
            outputs = outputs.view(b*n, *outputs.shape[2:])

        else:
            outputs = outputs * self.weight + self.bias

        return outputs

############################################################
#                      Utility layers                      #
############################################################

class SPADE(nn.Module):
    def __init__(self, num_features, spatial_size, norm_layer_type, eps):
        super(SPADE, self).__init__()
        self.norm_layer_type = norm_layer_type
        self.pixel_feats = nn.Parameter(torch.empty(1, num_features, spatial_size, spatial_size))
        nn.init.kaiming_uniform_(self.pixel_feats, a=math.sqrt(5))

        # Init biases
        self.conv_weight = nn.Conv2d(num_features, num_features, 1, 1, 0)
        self.conv_bias = nn.Conv2d(num_features, num_features, 1, 1, 0)

        # Supported normalization layers
        norm_layers = {
            'bn': lambda num_features: nn.BatchNorm2d(num_features, eps=eps, affine=False),
            'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=eps, affine=False),
            'none': lambda num_features: nn.Identity()}

        self.norm_layer = norm_layers[self.norm_layer_type](num_features)

    def forward(self, inputs):
        outputs = self.norm_layer(inputs)

        weight = self.conv_weight(self.pixel_feats) + 1.0
        bias = self.conv_bias(self.pixel_feats)

        return outputs * weight + bias


class Flatten(nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super(Flatten, self).__init__()
        self.flatten = lambda input: torch.flatten(input, start_dim, end_dim)

    def forward(self, inputs):
        return self.flatten(inputs)


class PixelwiseBias(nn.Module):
    def __init__(self, num_features, spatial_size, weight):
        super(PixelwiseBias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, num_features, spatial_size, spatial_size))

        # Init biases
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        return inputs + self.bias


class StochasticBias(nn.Module):
    def __init__(self, num_features, spatial_size, weight):
        super(StochasticBias, self).__init__()
        self.spatial_size = spatial_size
        self.scales = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, inputs):
        noise = torch.randn(inputs.shape[0], 1, self.spatial_size, self.spatial_size)

        if noise.type() != inputs.type():
            noise = noise.type(inputs.type())
        
        return inputs + self.scales * noise


def init_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()

############################################################
#                Definitions for the layers                #
############################################################

# Supported activations
activations = {
    'relu': nn.ReLU,
    'leakyrelu': functools.partial(nn.LeakyReLU, negative_slope=0.2)}

# Supported upsampling layers
upsampling_layers = {
    'nearest': lambda stride: nn.Upsample(scale_factor=stride, mode='nearest'),
    'bilinear': lambda stride: nn.Upsample(scale_factor=stride, mode='bilinear'),
    'pixelshuffle': nn.PixelShuffle}

# Supported downsampling layers
downsampling_layers = {
    'avgpool': nn.AvgPool2d,
    'maxpool': nn.MaxPool2d,
    'pixelunshuffle': PixelUnShuffle}

# Supported normalization layers
norm_layers = {
    'none': nn.Identity,
    'bn': lambda num_features, spatial_size, eps: nn.BatchNorm2d(num_features, eps, affine=True),
    'bn_1d': lambda num_features, spatial_size, eps: nn.BatchNorm1d(num_features, eps, affine=True),
    'in': lambda num_features, spatial_size, eps: nn.InstanceNorm2d(num_features, eps, affine=True),
    'ada_bn': functools.partial(AdaptiveNorm2d, norm_layer_type='bn'),
    'ada_in': functools.partial(AdaptiveNorm2d, norm_layer_type='in'),
    'ada_none': functools.partial(AdaptiveNorm2d, norm_layer_type='none'),
    'spade_bn': functools.partial(SPADE, norm_layer_type='bn'),
    'spade_in': functools.partial(SPADE, norm_layer_type='in'),
    'ada_spade_in': functools.partial(AdaptiveNorm2d, norm_layer_type='spade_in'),
    'ada_spade_bn': functools.partial(AdaptiveNorm2d, norm_layer_type='spade_bn')}

# Supported layers for skip connections
skip_layers = {
    'conv': nn.Conv2d,
    'ada_conv': AdaptiveConv2d}

# Supported layers for pixelwise biases
pixelwise_bias_layers = {
    'stochastic': StochasticBias,
    'fixed': PixelwiseBias,
    'adaptive': AdaptiveBias}

############################################################
# Residual block is the base class used for upsampling and #
# downsampling operations inside the networks              #
############################################################

class ResBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, # Parameters for the convolutions
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1,
        dilation: int = 1, 
        groups: int = 1, 
        eps: float = 1e-4, # Used in normalization layers
        spatial_size: int = 1, # Spatial size of the first tensor
        activation_type: str = 'relu',
        norm_layer_type: str = 'none',
        resize_layer_type: str = 'none',
        pixelwise_bias_type: str = 'none', # If not 'none', pixelwise bias is used after each conv
        skip_layer_type: str = 'conv', # Type of convolution in skip connections
        separable_conv: bool = False, # Use separable convolutions
        efficient_upsampling: bool = False, # Place upsampling layer after the first convolution
        first_norm_is_not_adaptive: bool = False, # Force standard normalization in the first norm layer
        return_feats: bool = False, # Output features taken after the first convolution
        return_first_feats: bool = False, # Output additional features taken after the first activation
        few_shot_aggregation: bool = False, # Aggregate few-shot training data in skip connection via mean
        frames_per_person: int = 1, # Number of frames per one person in a batch
        output_aggregated: bool = False, # Output aggregated features
    ) -> nn.Sequential:
        """This is a base module for preactivation residual blocks"""
        super(ResBlock, self).__init__()
        ### Set options for the block ###
        self.return_feats = return_feats
        self.return_first_feats = return_first_feats
        self.few_shot_aggregation = few_shot_aggregation
        self.num_frames = frames_per_person
        self.output_aggregated = output_aggregated

        channel_bias = pixelwise_bias_type == 'none'
        pixelwise_bias = pixelwise_bias_type != 'none'

        normalize = norm_layer_type != 'none'

        upsample = resize_layer_type in upsampling_layers
        downsample = resize_layer_type in downsampling_layers

        ### Set used layers ###
        if pixelwise_bias:
            pixelwise_bias_layer = pixelwise_bias_layers[pixelwise_bias_type]

        activation = activations[activation_type]

        if normalize:
            norm_layer_1 = norm_layers[norm_layer_type if not first_norm_is_not_adaptive else norm_layer_type.replace('ada_', '').replace('spade_', '')]
            norm_layer_2 = norm_layers[norm_layer_type]

        if upsample:
            resize_layer = upsampling_layers[resize_layer_type]

        if downsample:
            resize_layer = downsampling_layers[resize_layer_type]

        skip_layer = skip_layers[skip_layer_type]

        ### Initialize the layers of the first half of the block ###
        layers = []

        if normalize:
            layers += [norm_layer_1(in_channels, spatial_size, eps=eps)]

        layers += [activation(inplace=normalize)] # inplace is set to False if it is the first layer

        if self.return_first_feats:
            self.block_first_feats = nn.Sequential(*layers)

            layers = []

        if upsample and not efficient_upsampling:
            layers += [resize_layer(stride)]

            if spatial_size != 1: spatial_size *= 2

        layers += [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=channel_bias and not separable_conv)]

        if separable_conv:
            layers += [nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=1,
                bias=channel_bias)]

        if pixelwise_bias:
            layers += [pixelwise_bias_layer(out_channels, spatial_size, layers[-1].weight)]

        if normalize:
            layers += [norm_layer_2(out_channels, spatial_size, eps=eps)]
        
        layers += [activation(inplace=True)]

        self.block_feats = nn.Sequential(*layers)

        ### And initialize the second half ###
        layers = []
        
        if upsample and efficient_upsampling:
            layers += [resize_layer(stride)]

            if spatial_size != 1: spatial_size *= 2

        layers += [nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=channel_bias and not separable_conv)]

        if separable_conv:
            layers += [nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=1,
                bias=channel_bias)]

        if pixelwise_bias:
            layers += [pixelwise_bias_layer(out_channels, spatial_size, layers[-1].weight)]

        if downsample:
            layers += [resize_layer(stride)]

        self.block = nn.Sequential(*layers)

        ### Initialize a skip connection block, if needed ###
        if in_channels != out_channels or upsample or downsample:
            layers = []

            if upsample:
                layers += [resize_layer(stride)]

            layers += [skip_layer(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=1)]

            if downsample:
                layers += [resize_layer(stride)]
            
            self.skip = nn.Sequential(*layers)

        else:
            self.skip = nn.Identity()

    def forward(self, inputs):
        feats = []

        if hasattr(self, 'block_first_feats'):
            feats += [self.block_first_feats(inputs)]

            outputs = feats[-1]

        else:
            outputs = inputs

        feats += [self.block_feats(outputs)]

        outputs_main = self.block(feats[-1])

        if self.few_shot_aggregation:
            n = self.num_frames
            b = outputs_main.shape[0] // n

            outputs_main = outputs_main.view(b, n, *outputs_main.shape[1:]).mean(dim=1, keepdims=True) # aggregate
            outputs_main = torch.cat([outputs_main]*n, dim=1).view(b*n, *outputs_main.shape[2:]) # repeat

        outputs_skip = self.skip(inputs)

        outputs = outputs_main + outputs_skip

        if self.output_aggregated:
            n = self.num_frames
            b = outputs.shape[0] // n

            outputs = outputs.view(b, n, *outputs.shape[1:]).mean(dim=1) # aggregate

        if self.return_feats: 
            outputs = [outputs, feats]

        return outputs
# Third party
import torch
from torch import nn
import torch.nn.functional as F
import math

# This project
from runners import utils as rn_utils
from networks import utils as nt_utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--tex_num_channels',         default=64, type=int, 
                                                 help='minimum number of channels')

        parser.add('--tex_max_channels',         default=512, type=int, 
                                                 help='maximum number of channels')

        parser.add('--tex_norm_layer_type',      default='ada_spade_bn', type=str,
                                                 help='norm layer inside the texture generator')

        parser.add('--tex_pixelwise_bias_type',  default='none', type=str,
                                                 help='pixelwise bias type for convolutions')

        parser.add('--tex_input_tensor_size',    default=4, type=int, 
                                                 help='input spatial size of the generators')

        parser.add('--tex_activation_type',      default='leakyrelu', type=str,
                                                 help='activation layer inside the generators')

        parser.add('--tex_upsampling_type',      default='nearest', type=str,
                                                 help='upsampling layer inside the generator')

        parser.add('--tex_skip_layer_type',      default='ada_conv', type=str,
                                                 help='skip connection layer type')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        # Initialize options
        self.args = args

        # Generator
        self.gen_tex_input = nn.Parameter(torch.randn(1, args.tex_max_channels, args.tex_input_tensor_size, args.tex_input_tensor_size))
        self.gen_tex = Generator(args)

        # Projector (prediction of adaptive parameters)
        self.prj_tex = Projector(args)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:

        # Do not store activations if this network is not being trained
        if 'texture_generator' not in networks_to_train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        idt_embeds = data_dict['source_idt_embeds']
        b = idt_embeds[0].shape[0]

        ### Forward through the projectors ###
        tex_weights, tex_biases = self.prj_tex(idt_embeds)
        self.assign_adaptive_params(self.gen_tex, tex_weights, tex_biases)

        ### Forward through the texture generator ###
        tex_inputs = torch.cat([self.gen_tex_input]*b, dim=0)
        outputs = self.gen_tex(tex_inputs)
        pred_tex_hf_rgbs = outputs[0]

        if 'texture_generator' not in networks_to_train:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])
        reshape_source_data = lambda data: data.view(b, n, *data.shape[1:])

        data_dict['pred_tex_hf_rgbs'] = pred_tex_hf_rgbs[:, None]

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        # All visualization is done in the inference generator
        visuals = []

        return visuals

    @staticmethod
    def assign_adaptive_params(net, weights, biases):
        i = 0
        for m in net.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d" and 'spade' not in m.norm_layer_type:
                m.weight = weights[i] + 1.0
                m.bias = biases[i]
                i += 1

            elif m.__class__.__name__ == 'AdaptiveConv2d':
                m.weight = weights[i]
                m.bias = biases[i]
                i += 1

    @staticmethod
    def adaptive_params_mixing(net, indices):
        for m in net.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d" and 'spade' not in m.norm_layer_type:
                m.weight = m.weight[indices]
                m.bias = m.bias[indices]

            elif m.__class__.__name__ == 'AdaptiveConv2d':
                m.weight = m.weight[indices]
                m.bias = m.bias[indices]

    def __repr__(self):
        output = ''

        num_params = 0
        for p in self.prj_tex.parameters():
            num_params += p.numel()
        output += self.prj_tex.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params
        output += '\n'

        num_params = 0
        for p in self.gen_tex.parameters():
            num_params += p.numel()
        output += self.gen_tex.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        # Set options for the blocks
        num_blocks = int(math.log(args.image_size // args.tex_input_tensor_size, 2))
        
        out_channels = min(int(args.tex_num_channels * 2**num_blocks), args.tex_max_channels)
        spatial_size = 1

        if 'spade' in args.tex_norm_layer_type or args.tex_pixelwise_bias_type != 'none':
            spatial_size = args.tex_input_tensor_size

        layers = []

        # Construct the upsampling blocks
        for i in range(num_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(args.tex_num_channels * 2**i), args.tex_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                spatial_size=spatial_size,
                activation_type=args.tex_activation_type, 
                norm_layer_type=args.tex_norm_layer_type,
                resize_layer_type=args.tex_upsampling_type,
                pixelwise_bias_type=args.tex_pixelwise_bias_type,
                skip_layer_type=args.tex_skip_layer_type,
                first_norm_is_not_adaptive=i == num_blocks - 1)]

            if 'spade' in args.tex_norm_layer_type or args.tex_pixelwise_bias_type != 'none':
                spatial_size *= 2

        norm_layer = nt_utils.norm_layers[args.tex_norm_layer_type]
        activation = nt_utils.activations[args.tex_activation_type]

        layers += [
            norm_layer(out_channels, spatial_size, eps=args.eps),
            activation(inplace=True)]

        self.blocks = nn.Sequential(*layers)

        # Get the list of required heads
        heads = [(3, nn.Tanh)]

        # Initialize the heads
        self.heads = nn.ModuleList()

        for num_outputs, final_activation in heads:
            layers = [nn.Conv2d(
                in_channels=out_channels, 
                out_channels=num_outputs, 
                kernel_size=3, 
                stride=1, 
                padding=1)]

            if final_activation is not None:
                layers += [final_activation()]

            self.heads += [nn.Sequential(*layers)]

    def forward(self, inputs):
        outputs = self.blocks(inputs).contiguous()

        results = []

        for head in self.heads:
            results += [head(outputs)]

        return results


class Projector(nn.Module):
    def __init__(self, args, bottleneck_size=1024):
        super(Projector, self).__init__()
        # Calculate parameters of the blocks
        num_blocks = int(math.log(args.image_size // args.tex_input_tensor_size, 2))
        
        # FC channels perform a lowrank matrix decomposition
        self.channel_mults = []
        self.avgpools = nn.ModuleList()
        self.fc_blocks = nn.ModuleList()

        for i in range(num_blocks, 0, -1):
            in_channels = min(int(args.emb_num_channels * 2**i), args.emb_max_channels)

            out_in_channels = min(int(args.tex_num_channels * 2**i), args.tex_max_channels)
            out_channels = min(int(args.tex_num_channels * 2**(i-1)), args.tex_max_channels)

            channel_mult = out_in_channels / float(in_channels)
            self.channel_mults += [channel_mult]

            # Average pooling is applied to embeddings before FC
            s = int(bottleneck_size**0.5 * channel_mult)
            self.avgpools += [nn.AdaptiveAvgPool2d((s, s))]

            # Define decompositions for the i-th block
            self.fc_blocks += [nn.ModuleList()]

            # First AdaptiveBias
            if args.tex_pixelwise_bias_type == 'adaptive':
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1))]

            # First AdaNorm or SPADE weights and biases
            if 'ada_spade' in args.tex_norm_layer_type:
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1)),
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1))]

            elif 'ada' in args.tex_norm_layer_type:
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, 2))]

            # Second AdaptiveBias
            if args.tex_pixelwise_bias_type == 'adaptive':
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1))]

            # Skip conv weights and biases
            if args.tex_skip_layer_type == 'ada_conv':
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_in_channels + 1))]

            # Second AdaNorm or SPADE weights and biases
            if 'ada_spade' in args.tex_norm_layer_type:
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1)),
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1))]

            elif 'ada' in args.tex_norm_layer_type:
                self.fc_blocks[-1] += [nn.Sequential(
                    nn.Linear(int(s**2 / channel_mult), in_channels),
                    nn.Linear(in_channels, in_channels),
                    nn.Linear(in_channels, 2))]

    def forward(self, embeds):
        weights = []
        biases = []

        for embed, fc_block, channel_mult, avgpool in zip(embeds, self.fc_blocks, self.channel_mults, self.avgpools):
            b, c, h, w = embed.shape
            embed = avgpool(embed)

            c_out = int(c * channel_mult)
            embed = embed.view(b * c_out, -1)

            for fc in fc_block:
                params = fc(embed)

                params = params.view(b, c_out, -1)
                
                weight = params[:, :, :-1].squeeze()

                if weight.shape[0] != b and len(weight.shape) > 1 or len(weight.shape) > 2:
                    weight = weight[..., None, None] # 1x1 conv weight

                bias = params[:, :, -1].squeeze()

                if b == 1:
                    weight = weight[None]
                    bias = bias[None]
                    
                weights += [weight]
                biases += [bias]

        return weights, biases
# Third party
import torch
from torch import nn
import torch.nn.functional as F
import math

# This project
from runners import utils as rn_utils
from networks import utils as nt_utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--emb_num_channels',          default=64, type=int, 
                                                  help='minimum number of channels')

        parser.add('--emb_max_channels',          default=512, type=int, 
                                                  help='maximum number of channels')

        parser.add('--emb_no_stickman',           action='store_true', 
                                                  help='do not input stickman into the embedder')

        parser.add('--emb_output_tensor_size',    default=8, type=int,
                                                  help='spatial size of the last tensor')

        parser.add('--emb_norm_layer_type',       default='none', type=str,
                                                  help='norm layer inside the embedder')

        parser.add('--emb_activation_type',       default='leakyrelu', type=str,
                                                  help='activation layer inside the embedder')

        parser.add('--emb_downsampling_type',     default='avgpool', type=str,
                                                  help='downsampling layer inside the embedder')

        parser.add('--emb_apply_masks',           default='True', type=rn_utils.str2bool, choices=[True, False],
                                                  help='apply segmentation masks to source ground-truth images')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args
        
        self.net = Embedder(args)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        """The function modifies the input data_dict to contain the embeddings for the source images"""

        # Do not store activations if this network is not being trained
        if 'identity_embedder' not in networks_to_train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        inputs = data_dict['source_imgs']
        b, n = inputs.shape[:2]

        if self.args.emb_apply_masks:
            inputs = inputs * data_dict['source_segs'] + (-1) * (1 - data_dict['source_segs'])

        if not self.args.emb_no_stickman:
            inputs = torch.cat([inputs, data_dict['source_stickmen']], 2)

        ### Main forward pass ###
        source_embeds = self.net(inputs)

        if 'identity_embedder' not in networks_to_train:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        data_dict['source_idt_embeds'] = source_embeds

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = [data_dict['source_imgs'].detach()]

        if 'source_stickmen' in data_dict.keys():
            visuals += [data_dict['source_stickmen']]
        
        return visuals

    def __repr__(self):
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        output = self.net.__repr__()

        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()
        # Number of encoding blocks
        num_enc_blocks = int(math.log(args.image_size // args.emb_output_tensor_size, 2))

        # Number of decoding blocks (which is equal to the number of blocks in the generator)
        num_dec_blocks = int(math.log(args.image_size // args.tex_input_tensor_size, 2))

        ### Source images embedding ###
        out_channels = args.emb_num_channels

        # Construct the encoding blocks
        layers = [
            nn.Conv2d(
                in_channels=3 + 3 * (not args.emb_no_stickman), 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1)]

        for i in range(1, num_enc_blocks + 1):
            in_channels = out_channels
            out_channels = min(int(args.emb_num_channels * 2**i), args.emb_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.emb_activation_type, 
                norm_layer_type=args.emb_norm_layer_type,
                resize_layer_type=args.emb_downsampling_type,
                frames_per_person=args.num_source_frames,
                output_aggregated=i == num_enc_blocks)]

        self.enc = nn.Sequential(*layers)

        # And the decoding blocks
        layers = []

        for i in range(num_dec_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(args.tex_num_channels * 2**i), args.tex_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels,
                eps=args.eps,
                activation_type=args.emb_activation_type, 
                norm_layer_type=args.emb_norm_layer_type,
                resize_layer_type='none',
                return_feats=True)]

        self.dec_blocks = nn.ModuleList(layers)

    def forward(self, inputs):
        b, n, c, h, w = inputs.shape
        outputs = self.enc(inputs.view(-1, c, h, w))
        
        # Obtain embeddings at the final resolution
        embeds = []

        # Produce a stack of embeddings with different channels num
        for block in self.dec_blocks:
            outputs, embeds_block = block(outputs)
            embeds += embeds_block

        # Average over all source images (if needed)
        if embeds[0].shape[0] == b*n:
            embeds = [embeds_block.view(b, n, *embeds_block.shape[1:]).mean(dim=1) for embeds_block in embeds]

        return embeds
# Third party
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# This project
from networks import utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--pse_num_channels',      default=256, type=int, 
                                              help='number of intermediate channels')

        parser.add('--pse_num_blocks',        default=4, type=int, 
                                              help='number of encoding blocks')

        parser.add('--pse_in_channels',       default=394, type=int,
                                              help='number of channels in either latent pose (if present) of keypoints')

        parser.add('--pse_emb_source_pose',   action='store_true', 
                                              help='predict embeddings for the source pose')

        parser.add('--pse_norm_layer_type',   default='none', type=str,
                                              help='norm layer inside the pose embedder')

        parser.add('--pse_activation_type',   default='leakyrelu', type=str,
                                              help='activation layer inside the pose embedder')

        parser.add('--pse_use_harmonic_enc',  action='store_true', 
                                              help='encode keypoints with harmonics')

        parser.add('--pse_num_harmonics',     default=4, type=int, 
                                              help='number of frequencies used')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args

        self.net = PoseEmbedder(args)

        if self.args.pse_use_harmonic_enc:
            frequencies = torch.ones(args.pse_num_harmonics) * np.pi * 2**torch.arange(args.pse_num_harmonics)
            frequencies = frequencies[None, None]

            self.register_buffer('frequencies', frequencies)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        """The function modifies the input data_dict to contain the pose 
        embeddings for the target and (optinally) source images"""

        # Do not store activations if this network is not being trained
        if 'keypoints_embedder' not in networks_to_train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        target_poses = data_dict['target_poses']

        b, t = target_poses.shape[:2]
        target_poses = target_poses.view(b*t, -1)

        # Encode with harmonics (if needed)
        if self.args.pse_use_harmonic_enc:
            target_poses = (target_poses[..., None] * self.frequencies).view(b*t, -1)
            target_poses = torch.cat([torch.sin(target_poses), torch.cos(target_poses)], dim=1)

        if self.args.pse_emb_source_pose:
            source_poses = data_dict['source_poses']

            n = source_poses.shape[1]
            source_poses = source_poses.view(b*n, -1)

            # Encode with harmonics (if needed)
            if self.args.pse_use_harmonic_enc:
                source_poses = (source_poses[..., None] * self.frequencies).view(b*t, -1)
                source_poses = torch.cat([torch.sin(source_poses), torch.cos(source_poses)], dim=1)

        ### Main forward pass ###
        target_embeds = self.net(target_poses)

        if self.args.pse_emb_source_pose:
            source_embeds = self.net(source_poses)

        if 'keypoints_embedder' not in networks_to_train:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        data_dict['target_pose_embeds'] = target_embeds.view(b, t, *target_embeds.shape[1:])

        if self.args.pse_emb_source_pose:
            data_dict['source_pose_embeds'] = source_embeds.view(b, n, *source_embeds.shape[1:])

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = []

        return visuals

    def __repr__(self):
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        output = self.net.__repr__()

        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class PoseEmbedder(nn.Module):
    def __init__(self, args):
        super(PoseEmbedder, self).__init__()
        # Calculate output size of the embedding
        self.num_channels = args.inf_max_channels
        self.spatial_size = args.inf_input_tensor_size

        # Initialize keypoints-encoding MLP
        norm_layer = utils.norm_layers[args.pse_norm_layer_type]
        activation = utils.activations[args.pse_activation_type]

        # Set input number of channels
        if args.pse_use_harmonic_enc:
            in_channels = args.pse_in_channels * args.pse_num_harmonics * 2
        else:
            in_channels = args.pse_in_channels

        # Set latent number of channels
        if args.pse_num_blocks == 1:
            num_channels = self.num_channels * self.spatial_size**2
        else:
            num_channels = args.pse_num_channels

        # Define encoding blocks
        layers = [nn.Linear(in_channels, num_channels)]
        
        for i in range(1, args.pse_num_blocks - 1):
            if args.pse_norm_layer_type != 'none':
                layers += [norm_layer(num_channels, None, eps=args.eps)]

            layers += [
                activation(inplace=True),
                nn.Linear(num_channels, num_channels)]

        if args.pse_num_blocks != 1:
            if args.pse_norm_layer_type != 'none':
                layers += [norm_layer(num_channels, None, eps=args.eps)]

            layers += [
                activation(inplace=True),
                nn.Linear(num_channels, self.num_channels * self.spatial_size**2)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        pose_embeds = self.mlp(inputs)

        pose_embeds = pose_embeds.view(-1, self.num_channels, self.spatial_size, self.spatial_size)

        return pose_embeds

# Third party
import torch
from torch import nn
import torch.nn.functional as F
import math

# This project
from runners import utils as rn_utils
from networks import utils as nt_utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--inf_num_channels',         default=32, type=int, 
                                                 help='minimum number of channels')

        parser.add('--inf_max_channels',         default=256, type=int, 
                                                 help='maximum number of channels')

        parser.add('--inf_pred_segmentation',    default='True', type=rn_utils.str2bool, choices=[True, False],
                                                 help='set inference generator to output a segmentation mask')

        parser.add('--inf_norm_layer_type',      default='ada_bn', type=str,
                                                 help='norm layer inside the inference generator')

        parser.add('--inf_input_tensor_size',    default=4, type=int, 
                                                 help='input spatial size of the convolutional part')

        parser.add('--inf_activation_type',      default='leakyrelu', type=str,
                                                 help='activation layer inside the generators')

        parser.add('--inf_upsampling_type',      default='nearest', type=str,
                                                 help='upsampling layer inside the generator')

        parser.add('--inf_skip_layer_type',      default='ada_conv', type=str,
                                                 help='skip connection layer type')

        parser.add('--inf_pred_source_data',     default='False', type=rn_utils.str2bool, choices=[True, False], 
                                                 help='predict inference generator outputs for the source data')

        parser.add('--inf_calc_grad',            default='False', type=rn_utils.str2bool, choices=[True, False], 
                                                 help='force gradients calculation in the generator')

        parser.add('--inf_apply_masks',          default='True', type=rn_utils.str2bool, choices=[True, False], 
                                                 help='apply segmentation masks to predicted and ground-truth images')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        # Initialize options
        self.args = args

        # Generator
        self.gen_inf = Generator(args)

        # Projector (prediction of adaptive parameters)
        self.prj_inf = Projector(args)

        # Greate a meshgrid, which is used for UVs calculation from deltas
        grid = torch.linspace(-1, 1, args.image_size + 1)
        grid = (grid[1:] + grid[:-1]) / 2
        v, u = torch.meshgrid(grid, grid)
        identity_grid = torch.stack([u, v], 2)[None] # 1 x h x w x 2
        self.register_buffer('identity_grid', identity_grid)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:

        # Do not store activations if this network is not being trained
        if 'inference_generator' not in networks_to_train and not self.args.inf_calc_grad:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        idt_embeds = data_dict['source_idt_embeds']
        target_pose_embeds = data_dict['target_pose_embeds']

        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            # Predicted segmentation masks are applied to target images
            target_imgs = data_dict['target_imgs']

        pred_tex_hf_rgbs = data_dict['pred_tex_hf_rgbs'][:, 0]

        b, t = target_pose_embeds.shape[:2]
        target_pose_embeds = target_pose_embeds.view(b*t, *target_pose_embeds.shape[2:])

        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            target_imgs = target_imgs.view(b*t, *target_imgs.shape[2:])

        if self.args.inf_pred_source_data:
            source_pose_embeds = data_dict['source_pose_embeds']

            n = source_pose_embeds.shape[1]
            source_pose_embeds = source_pose_embeds.view(b*n, *source_pose_embeds.shape[2:])

        ### Forward through the projectors ###
        inf_weights, inf_biases = self.prj_inf(idt_embeds)
        self.assign_adaptive_params(self.gen_inf, inf_weights, inf_biases)

        ### Forward target poses through the inference generator ###
        outputs = self.gen_inf(target_pose_embeds)

        # Parse the outputs
        pred_target_delta_uvs = outputs[0]
        pred_target_uvs = self.identity_grid + pred_target_delta_uvs.permute(0, 2, 3, 1)

        pred_target_delta_lf_rgbs = outputs[1]

        if self.args.inf_pred_segmentation:
            pred_target_segs_logits = outputs[2]
            pred_target_segs = torch.sigmoid(pred_target_segs_logits)

        ### Forward source poses through the inference generator (if needed) ###
        if self.args.inf_pred_source_data:
            outputs = self.gen_inf(source_pose_embeds)

            # Parse the outputs
            source_delta_uvs = outputs[0]
            pred_source_uvs = self.identity_grid + source_delta_uvs.permute(0, 2, 3, 1)

            pred_source_delta_lf_rgbs = outputs[1]

            if self.args.inf_pred_segmentation:
                pred_source_segs_logits = outputs[2]
                pred_source_segs = torch.sigmoid(pred_source_segs_logits)

        ### Combine components into an output target image
        pred_tex_hf_rgbs_repeated = torch.cat([pred_tex_hf_rgbs[:, None]]*t, dim=1)
        pred_tex_hf_rgbs_repeated = pred_tex_hf_rgbs_repeated.view(b*t, *pred_tex_hf_rgbs.shape[1:])

        pred_target_delta_hf_rgbs = F.grid_sample(pred_tex_hf_rgbs_repeated, pred_target_uvs)

        # Final image
        pred_target_imgs = pred_target_delta_lf_rgbs + pred_target_delta_hf_rgbs

        if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
            # Get an image with a low-frequency component detached
            pred_target_imgs_lf_detached = pred_target_delta_lf_rgbs.detach() + pred_target_delta_hf_rgbs

        # Mask output images (if needed)
        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            pred_target_masks = pred_target_segs.detach()

            target_imgs = target_imgs * pred_target_masks + (-1) * (1 - pred_target_masks)

            pred_target_imgs = pred_target_imgs * pred_target_masks + (-1) * (1 - pred_target_masks)

            pred_target_delta_lf_rgbs = pred_target_delta_lf_rgbs * pred_target_masks + (-1) * (1 - pred_target_masks)

            if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
                pred_target_imgs_lf_detached = pred_target_imgs_lf_detached * pred_target_masks + (-1) * (1 - pred_target_masks)

        if 'inference_generator' not in networks_to_train and not self.args.inf_calc_grad:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])
        reshape_source_data = lambda data: data.view(b, n, *data.shape[1:])

        data_dict['pred_target_imgs'] = reshape_target_data(pred_target_imgs)
        if self.args.inf_pred_segmentation:
            data_dict['pred_target_segs'] = reshape_target_data(pred_target_segs)

        # Output debugging results
        data_dict['pred_target_uvs'] = reshape_target_data(pred_target_uvs)
        data_dict['pred_target_delta_lf_rgbs'] = reshape_target_data(pred_target_delta_lf_rgbs)
        data_dict['pred_target_delta_hf_rgbs'] = reshape_target_data(pred_target_delta_hf_rgbs)

        # Output results needed for training
        if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
            data_dict['pred_target_delta_uvs'] = reshape_target_data(pred_target_delta_uvs)
            data_dict['pred_target_imgs_lf_detached'] = reshape_target_data(pred_target_imgs_lf_detached)

            if self.args.inf_pred_segmentation:
                data_dict['pred_target_segs_logits'] = reshape_target_data(pred_target_segs_logits)

        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            data_dict['target_imgs'] = reshape_target_data(target_imgs)

        # Output results related to source imgs (if needed)
        if self.args.inf_pred_source_data:
            data_dict['pred_source_uvs'] = reshape_source_data(pred_source_uvs)
            data_dict['pred_source_delta_lf_rgbs'] = reshape_source_data(pred_source_delta_lf_rgbs)

            if self.args.inf_pred_segmentation:
                data_dict['pred_source_segs'] = reshape_source_data(pred_source_segs)

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = []

        if self.args.inf_pred_source_data:
            # Predicted source LF rgbs
            visuals += [data_dict['pred_source_delta_lf_rgbs']]

            # Predicted source HF rgbs
            if 'pred_source_delta_hf_rgbs' in data_dict.keys():
                visuals += [data_dict['pred_source_delta_hf_rgbs']]

            # Predicted source UVs
            pred_source_uvs = data_dict['pred_source_uvs'].permute(0, 3, 1, 2)

            b, _, h, w = pred_source_uvs.shape
            pred_source_uvs = torch.cat([
                    pred_source_uvs, 
                    torch.empty(b, 1, h, w, dtype=pred_source_uvs.dtype, device=pred_source_uvs.device).fill_(-1)
                ], 
                dim=1)

            visuals += [torch.cat([pred_source_uvs])]

            # Predicted source segs
            if self.args.inf_pred_segmentation:
                pred_source_segs = data_dict['pred_source_segs']

                visuals += [torch.cat([(pred_source_segs - 0.5) * 2] * 3, 1)]

        # Predicted textures
        visuals += [data_dict['pred_tex_hf_rgbs']]

        if 'pred_enh_tex_hf_rgbs' in data_dict.keys():
            # Predicted enhated textures
            visuals += [data_dict['pred_enh_tex_hf_rgbs']]

        # Target images
        visuals += [data_dict['target_imgs']]

        # Predicted images
        visuals += [data_dict['pred_target_imgs']]

        # Predicted enhated images
        if 'pred_enh_target_imgs' in data_dict.keys():
            visuals += [data_dict['pred_enh_target_imgs']]

        # Predicted target LF rgbs
        visuals += [data_dict['pred_target_delta_lf_rgbs']]

        # Predicted target HF rgbs
        visuals += [data_dict['pred_target_delta_hf_rgbs']]

        if 'pred_enh_target_delta_hf_rgbs' in data_dict.keys():
            # Predicted enhated target HF rgbs
            visuals += [data_dict['pred_enh_target_delta_hf_rgbs']]

        # Predicted target UVs
        pred_target_uvs = data_dict['pred_target_uvs'].permute(0, 3, 1, 2)

        b, _, h, w = pred_target_uvs.shape
        pred_target_uvs = torch.cat([
                pred_target_uvs, 
                torch.empty(b, 1, h, w, dtype=pred_target_uvs.dtype, device=pred_target_uvs.device).fill_(-1)
            ], 
            dim=1)

        visuals += [torch.cat([pred_target_uvs])]

        if self.args.inf_pred_segmentation:
            # Target segmentation
            target_segs = data_dict['target_segs']
            visuals += [torch.cat([(target_segs - 0.5) * 2] * 3, 1)]

            # Predicted target segmentation
            pred_target_segs = data_dict['pred_target_segs']
            visuals += [torch.cat([(pred_target_segs - 0.5) * 2] * 3, 1)]

        return visuals

    @staticmethod
    def assign_adaptive_params(net, weights, biases):
        i = 0
        for m in net.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                m.weight = weights[i] + 1.0
                m.bias = biases[i]
                i += 1

            elif m.__class__.__name__ == 'AdaptiveConv2d':
                m.weight = weights[i]
                m.bias = biases[i]
                i += 1

    @staticmethod
    def adaptive_params_mixing(net, indices):
        for m in net.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                m.weight = m.weight[indices]
                m.bias = m.bias[indices]

            elif m.__class__.__name__ == 'AdaptiveConv2d':
                m.weight = m.weight[indices]
                m.bias = m.bias[indices]

    def __repr__(self):
        output = ''

        num_params = 0
        for p in self.prj_inf.parameters():
            num_params += p.numel()
        output += self.prj_inf.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params
        output += '\n'

        num_params = 0
        for p in self.gen_inf.parameters():
            num_params += p.numel()
        output += self.gen_inf.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        # Set options for the blocks
        num_blocks = int(math.log(args.image_size // args.inf_input_tensor_size, 2))
        out_channels = min(int(args.inf_num_channels * 2**num_blocks), args.inf_max_channels)

        # Construct the upsampling blocks
        layers = []

        for i in range(num_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(args.inf_num_channels * 2**i), args.inf_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.inf_activation_type, 
                norm_layer_type=args.inf_norm_layer_type,
                resize_layer_type=args.inf_upsampling_type,
                skip_layer_type=args.inf_skip_layer_type,
                efficient_upsampling=True,
                first_norm_is_not_adaptive=i == num_blocks - 1)]

        norm_layer = nt_utils.norm_layers[args.inf_norm_layer_type]
        activation = nt_utils.activations[args.inf_activation_type]

        layers += [
            norm_layer(out_channels, spatial_size=1, eps=args.eps),
            activation(inplace=True)]

        self.blocks = nn.Sequential(*layers)

        # Get the list of required heads
        heads = [(2, nn.Tanh), (3, nn.Tanh)]

        if args.inf_pred_segmentation:
            heads += [(1, None)]

        # Initialize the heads
        self.heads = nn.ModuleList()

        for num_outputs, final_activation in heads:
            layers = [nn.Conv2d(
                in_channels=out_channels, 
                out_channels=num_outputs, 
                kernel_size=3, 
                stride=1, 
                padding=1)]

            if final_activation is not None:
                layers += [final_activation()]

            self.heads += [nn.Sequential(*layers)]

    def forward(self, inputs):
        outputs = self.blocks(inputs).contiguous()

        results = []

        for head in self.heads:
            results += [head(outputs)]

        return results


class Projector(nn.Module):
    def __init__(self, args, bottleneck_size=1024):
        super(Projector, self).__init__()
        # Calculate parameters of the blocks
        num_blocks = int(math.log(args.image_size // args.inf_input_tensor_size, 2))
        
        # FC channels perform a lowrank matrix decomposition
        self.channel_mults = []
        self.avgpools = nn.ModuleList()
        self.fc_blocks = nn.ModuleList()

        for i in range(num_blocks, 0, -1):
            in_channels = min(int(args.emb_num_channels * 2**i), args.emb_max_channels)

            out_in_channels = min(int(args.inf_num_channels * 2**i), args.inf_max_channels)
            out_channels = min(int(args.inf_num_channels * 2**(i-1)), args.inf_max_channels)

            channel_mult = out_in_channels / float(in_channels)
            self.channel_mults += [channel_mult]

            # Average pooling is applied to embeddings before FC
            s = int(bottleneck_size**0.5 * channel_mult)
            self.avgpools += [nn.AdaptiveAvgPool2d((s, s))]

            # Define decompositions for the i-th block
            self.fc_blocks += [nn.ModuleList()]

            # First AdaNorm weights and biases
            self.fc_blocks[-1] += [
                nn.Sequential(
                    nn.Linear(int(s**2 / channel_mult), in_channels),
                    nn.Linear(in_channels, in_channels),
                    nn.Linear(in_channels, 2))]

            # Skip conv weights and biases
            if args.inf_skip_layer_type == 'ada_conv':
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_in_channels + 1))]

            # Second AdaNorm weights and biases
            self.fc_blocks[-1] += [nn.Sequential(
                nn.Linear(int(s**2 / channel_mult), in_channels),
                nn.Linear(in_channels, in_channels),
                nn.Linear(in_channels, 2))]

    def forward(self, embeds):
        weights = []
        biases = []

        for embed, fc_block, channel_mult, avgpool in zip(embeds, self.fc_blocks, self.channel_mults, self.avgpools):
            b, c, h, w = embed.shape
            embed = avgpool(embed)

            c_out = int(c * channel_mult)
            embed = embed.view(b * c_out, -1)

            for fc in fc_block:
                params = fc(embed)

                params = params.view(b, c_out, -1)
                
                weight = params[:, :, :-1].squeeze()

                if weight.shape[0] != b and len(weight.shape) > 1 or len(weight.shape) > 2:
                    weight = weight[..., None, None] # 1x1 conv weight

                bias = params[:, :, -1].squeeze()

                if b == 1:
                    weight = weight[None]
                    bias = bias[None]
                    
                weights += [weight]
                biases += [bias]

        return weights, biases
# Third party
import torch
from torch import nn
import math

# This project
from networks import utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--dis_num_channels',        default=64, type=int, 
                                                help='minimum number of channels')

        parser.add('--dis_max_channels',        default=512, type=int, 
                                                help='maximum number of channels')

        parser.add('--dis_no_stickman',         action='store_true', 
                                                help='do not input stickman into the discriminator')

        parser.add('--dis_num_blocks',          default=6, type=int, 
                                                help='number of convolutional blocks')

        parser.add('--dis_output_tensor_size',  default=8, type=int, 
                                                help='spatial size of the last tensor')

        parser.add('--dis_norm_layer_type',     default='bn', type=str,
                                                help='norm layer inside the discriminator')

        parser.add('--dis_activation_type',     default='leakyrelu', type=str,
                                                help='activation layer inside the discriminator')

        parser.add('--dis_downsampling_type',   default='avgpool', type=str,
                                                help='downsampling layer inside the discriminator')

        parser.add('--dis_fake_imgs_name',      default='pred_target_imgs', type=str,
                                                help='name of the tensor with fake images')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args
        
        self.net = Discriminator(args)

    def forward(
            self, 
            data_dict: dict,
            net_names_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        
        # Extract inputs
        real_inputs = data_dict['target_imgs']
        fake_inputs = data_dict[self.args.dis_fake_imgs_name]

        # Input stickman (if needed)
        if not self.args.dis_no_stickman:
            real_inputs = torch.cat([real_inputs, data_dict['target_stickmen']], 2)
            fake_inputs = torch.cat([fake_inputs, data_dict['target_stickmen']], 2)

        # Reshape inputs
        b, t, c, h, w = real_inputs.shape
        real_inputs = real_inputs.view(-1, c, h, w)
        fake_inputs = fake_inputs.view(-1, c, h, w)

        ### Perform a dis forward pass ###
        for p in self.parameters():
            p.requires_grad = True

        # Concatenate batches
        inputs = torch.cat([real_inputs, fake_inputs.detach()])

        # Calculate outputs
        scores_dis, _ = self.net(inputs)

        # Split outputs into real and fake
        real_scores, fake_scores_dis = scores_dis.split(b)

        ### Store outputs ###
        data_dict['real_scores'] = real_scores
        data_dict['fake_scores_dis'] = fake_scores_dis

        ### Perform a gen forward pass ###
        for p in self.parameters():
            p.requires_grad = False

        # Concatenate batches
        inputs = torch.cat([real_inputs, fake_inputs])

        # Calculate outputs
        scores_gen, feats_gen  = self.net(inputs)

        # Split outputs into real and fake
        _, fake_scores_gen = scores_gen.split(b)

        feats = [feats_block.split(b) for feats_block in feats_gen]
        real_feats_gen, fake_feats_gen = map(list, zip(*feats))

        ### Store outputs ###
        data_dict['fake_scores_gen'] = fake_scores_gen

        data_dict['real_feats_gen'] = real_feats_gen
        data_dict['fake_feats_gen'] = fake_feats_gen
            
        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = []
        
        if 'target_stickmen' in data_dict.keys():
            visuals += [data_dict['target_stickmen']]
        
        return visuals

    def __repr__(self):
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        output = self.net.__repr__()

        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        # Set options for the blocks
        num_down_blocks = int(math.log(args.image_size // args.dis_output_tensor_size, 2))

        ### Construct the encoding blocks ###
        out_channels = args.dis_num_channels

        # The first block
        self.first_conv = nn.Conv2d(
            in_channels=3 + 3 * (not args.dis_no_stickman), 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1)

        # Downsampling blocks
        self.blocks = nn.ModuleList()

        for i in range(1, num_down_blocks + 1):
            in_channels = out_channels
            out_channels = min(int(args.dis_num_channels * 2**i), args.dis_max_channels)

            self.blocks += [utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.dis_activation_type, 
                norm_layer_type=args.dis_norm_layer_type,
                resize_layer_type=args.dis_downsampling_type,
                return_feats=True)]

        # And the blocks at the same resolution
        for i in range(num_down_blocks + 1, args.dis_num_blocks + 1):
            self.blocks += [utils.ResBlock(
                in_channels=out_channels, 
                out_channels=out_channels,
                eps=args.eps,
                activation_type=args.dis_activation_type, 
                norm_layer_type=args.dis_norm_layer_type,
                resize_layer_type='none',
                return_feats=True)]

        # Final convolutional block
        norm_layer = utils.norm_layers[args.dis_norm_layer_type]
        activation = utils.activations[args.dis_activation_type]

        self.final_block = nn.Sequential(
            norm_layer(out_channels, None, eps=args.eps),
            activation(inplace=True))

        ### Realism score prediction ###
        self.linear = nn.Conv2d(out_channels, 1, 1)

    def forward(self, inputs):
        # Convolutional part
        conv_outputs = self.first_conv(inputs)

        feats = []
        for block in self.blocks:
            conv_outputs, block_feats = block(conv_outputs)
            feats += block_feats

        conv_outputs = self.final_block(conv_outputs)

        # Linear head
        scores = self.linear(conv_outputs)

        return scores, feats
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from importlib import import_module
import cv2



def get_dataloader(args, phase):
    dataset = import_module(f'datasets.{args.dataloader_name}').DatasetWrapper(args, phase)
    if phase == 'train': 
    	args.train_size = len(dataset)
    return DataLoader(dataset, 
        batch_size=args.batch_size // args.world_size, 
        sampler=DistributedSampler(dataset, args.world_size, args.rank, shuffle=False), # shuffling is done inside the dataset
        num_workers=args.num_workers_per_process,
        drop_last=True)

# Required to draw a stickman for ArcSoft keypoints
def merge_parts(part_even, part_odd):
    output = []
    
    for i in range(len(part_even) + len(part_odd)):
        if i % 2:
            output.append(part_odd[i // 2])
        else:
            output.append(part_even[i // 2])

    return output

# Function for stickman and facemasks drawing
def draw_stickmen(args, poses):
    ### Define drawing options ###
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
        # Arcsoft keypoints
        edges_parts  = [
            merge_parts(range(0, 19), range(103, 121)), # face
            list(range(19, 29)), list(range(29, 39)), # eyebrows
            merge_parts(range(39, 51), range(121, 133)), list(range(165, 181)), [101, 101], # right eye
            merge_parts(range(51, 63), range(133, 145)), list(range(181, 197)), [102, 102], # left eye
            list(range(63, 75)), list(range(97, 101)), # nose
            merge_parts(range(75, 88), range(145, 157)), merge_parts(range(157, 165), range(88, 95))] # lips

        closed_parts = [
            False, 
            True, True, 
            True, True, False, 
            True, True, False, 
            False, False, 
            True, True]

        colors_parts = [
            (  255,  255,  255), 
            (  255,    0,    0), (    0,  255,    0),
            (    0,    0,  255), (    0,    0,  255), (    0,    0,  255),
            (  255,    0,  255), (  255,    0,  255), (  255,    0,  255),
            (    0,  255,  255), (    0,  255,  255),
            (  255,  255,    0), (  255,  255,    0)]

    else:
        edges_parts  = [
            list(range( 0, 17)), # face
            list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
            list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
            list(range(36, 42)), list(range(42, 48)), # right eye, left eye
            list(range(48, 60)), list(range(60, 68))] # lips

        closed_parts = [
            False, False, False, False, False, True, True, True, True]

        colors_parts = [
            (  255,  255,  255), 
            (  255,    0,    0), (    0,  255,    0),
            (    0,    0,  255), (    0,    0,  255), 
            (  255,    0,  255), (    0,  255,  255),
            (  255,  255,    0), (  255,  255,    0)]

    ### Start drawing ###
    stickmen = []

    for pose in poses:
        if isinstance(pose, torch.Tensor):
            # Apply conversion to numpy, asssuming the range to be in [-1, 1]
            xy = (pose.view(-1, 2).cpu().numpy() + 1) / 2 * args.image_size
        
        else:
            # Assuming the range to be [0, 1]
            xy = pose[:, :2] * self.args.image_size

        xy = xy[None, :, None].astype(np.int32)

        stickman = np.ones((args.image_size, args.image_size, 3), np.uint8)

        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
            stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=args.stickmen_thickness)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2. 

    return stickmen

# Flip vector poses via x axis
def flip_poses(args, keypoints, size):
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
        # Arcsoft keypoints
        edges_parts  = [
            merge_parts(range(0, 19), range(103, 121)), # face
            list(range(19, 29)), list(range(29, 39)), # eyebrows
            merge_parts(range(39, 51), range(121, 133)), list(range(165, 181)), [101, 101], # right eye
            merge_parts(range(51, 63), range(133, 145)), list(range(181, 197)), [102, 102], # left eye
            list(range(63, 75)), list(range(97, 101)), # nose
            merge_parts(range(75, 88), range(145, 157)), merge_parts(range(157, 165), range(88, 95))] # lip

    else:
        edges_parts  = [
            list(range( 0, 17)), # face
            list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
            list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
            list(range(36, 42)), list(range(42, 48)), # right eye, left eye
            list(range(48, 60)), list(range(60, 68))] # lips


    keypoints[:, 0] = size - keypoints[:, 0]

    # Swap left and right face parts
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
        l_parts  = edges_parts[1] + edges_parts[3] + edges_parts[4] + edges_parts[5][:1]
        r_parts = edges_parts[2] + edges_parts[6] + edges_parts[7] + edges_parts[8][:1]

    else:
        l_parts = edges_parts[2] + edges_parts[6]
        r_parts = edges_parts[1] + edges_parts[5]

    keypoints[l_parts + r_parts] = keypoints[r_parts + l_parts]

    return keypoints
import torch
from torch.utils import data
from torchvision import transforms
import glob
import pathlib
from PIL import Image
import numpy as np
import pickle as pkl
import cv2
import random
import math

from datasets import utils as ds_utils
from runners import utils as rn_utils



class DatasetWrapper(data.Dataset):
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--num_source_frames',     default=1, type=int,
                                              help='number of frames used for initialization of the model')

        parser.add('--num_target_frames',     default=1, type=int,
                                              help='number of frames per identity used for training')

        parser.add('--image_size',            default=256, type=int,
                                              help='output image size in the model')

        parser.add('--num_keypoints',         default=68, type=int,
                                              help='number of keypoints (depends on keypoints detector)')

        parser.add('--output_segmentation',   default='True', type=rn_utils.str2bool, choices=[True, False],
                                              help='read segmentation mask')

        parser.add('--output_stickmen',       default='True', type=rn_utils.str2bool, choices=[True, False],
                                              help='draw stickmen using keypoints')
        
        parser.add('--stickmen_thickness',    default=2, type=int, help='thickness of lines in the stickman',
                                              help='thickness of lines in the stickman')

        return parser

    def __init__(self, args, phase):
        super(DatasetWrapper, self).__init__()
        # Store options
        self.phase = phase
        self.args = args

        self.to_tensor = transforms.ToTensor()
        self.epoch = 0 if args.which_epoch == 'none' else int(args.which_epoch)

        # Data paths
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        if args.output_segmentation:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase

        # Video sequences list
        sequences = self.imgs_dir.glob('*/*')
        self.sequences = ['/'.join(seq.split('/')[-2:]) for seq in sequences]

        # Parameters of the sampling scheme
        self.delta = math.sqrt(5)
        self.cur_num = torch.rand(1).item()

    def __getitem__(self, index):
        # Sample source and target frames for the current sequence
        while True:
            try:
                filenames_img = list((self.imgs_dir / self.sequences[index]).glob('*/*'))
                filenames_img = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_img]

                filenames_npy = list((self.pose_dir / self.sequences[index]).glob('*/*'))
                filenames_npy = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_npy]

                filenames = list(set(filenames_img).intersection(set(filenames_npy)))

                if self.args.output_segmentation:
                    filenames_seg = list((self.segs_dir / self.sequences[index]).glob('*/*'))
                    filenames_seg = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_seg]

                    filenames = list(set(filenames).intersection(set(filenames_seg)))

                if len(filenames):
                    break
                else:
                    raise

            except:
                # Exception is raised if filenames list is empty or there was an error during read
                print('Encountered an error while reading the dataset')
                index = (index + 1) % len(self)

        filenames = sorted(filenames)

        imgs = []
        poses = []
        stickmen = []
        segs = []

        reserve_index = -1 # take this element of the sequence if loading fails
        sample_from_reserve = False

        if self.phase == 'test':
            # Sample from the beginning of the sequence
            self.cur_num = 0

        while len(imgs) < self.args.num_source_frames + self.args.num_target_frames:
            if reserve_index == len(filenames):
                raise # each element of the filenames list is unavailable for load

            # Sample a frame number
            if sample_from_reserve:
                filename = filenames[reserve_index]

            else:
                frame_num = int(round(self.cur_num * (len(filenames) - 1)))
                self.cur_num = (self.cur_num + self.delta) % 1

                filename = filenames[frame_num]

            # Read images
            img_path = pathlib.Path(self.imgs_dir) / filename.with_suffix('.jpg')
            
            try:
                img = Image.open(img_path)

                # Preprocess an image
                s = img.size[0]
                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)

            except:
                sample_from_reserve = True
                reserve_index += 1
                continue

            imgs += [self.to_tensor(img)]

            # Read keypoints
            keypoints_path = pathlib.Path(self.pose_dir) / filename.with_suffix('.npy')
            try:
                keypoints = np.load(keypoints_path).astype('float32')
            except:
                imgs.pop(-1)

                sample_from_reserve = True
                reserve_index += 1
                continue

            keypoints = keypoints[:self.args.num_keypoints, :]
            keypoints[:, :2] /= s
            keypoints = keypoints[:, :2]

            poses += [torch.from_numpy(keypoints.reshape(-1))]

            if self.args.output_segmentation:
                seg_path = pathlib.Path(self.segs_dir) / filename.with_suffix('.png')

                try:
                    seg = Image.open(seg_path)
                    seg = seg.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
                except:
                    imgs.pop(-1)
                    poses.pop(-1)

                    sample_from_reserve = True
                    reserve_index += 1
                    continue

                segs += [self.to_tensor(seg)]

            sample_from_reserve = False

        imgs = (torch.stack(imgs)- 0.5) * 2.0

        poses = (torch.stack(poses) - 0.5) * 2.0

        if self.args.output_stickmen:
            stickmen = utils.draw_stickmen(self.args, poses)

        if self.args.output_segmentation:
            segs = torch.stack(segs)

        # Split between few-shot source and target sets
        data_dict = {}
        if self.args.num_source_frames:
            data_dict['source_imgs'] = imgs[:self.args.num_source_frames]
        data_dict['target_imgs'] = imgs[self.args.num_source_frames:]
        
        if self.args.num_source_frames:
            data_dict['source_poses'] = poses[:self.args.num_source_frames]
        data_dict['target_poses'] = poses[self.args.num_source_frames:]

        if self.args.output_stickmen:
            if self.args.num_source_frames:
                data_dict['source_stickmen'] = stickmen[:self.args.num_source_frames]
            data_dict['target_stickmen'] = stickmen[self.args.num_source_frames:]
        
        if self.args.output_segmentation:
            if self.args.num_source_frames:
                data_dict['source_segs'] = segs[:self.args.num_source_frames]
            data_dict['target_segs'] = segs[self.args.num_source_frames:]
        
        data_dict['indices'] = torch.LongTensor([index])

        return data_dict

    def __len__(self):
        return len(self.sequences)

    def shuffle(self):
        self.sequences = [self.sequences[i] for i in torch.randperm(len(self.sequences)).tolist()]
import torch
from torch import nn
from torch.nn.utils.spectral_norm import SpectralNorm
from copy import deepcopy

from networks import utils as nt_utils



############################################################
#                 Utils for options parsing                #
############################################################

def str2bool(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise

def parse_str_to_list(string, value_type=str, sep=','):
    if string:
        outputs = string.replace(' ', '').split(sep)
    else:
        outputs = []
    outputs = [value_type(output) for output in outputs]
    return outputs

def parse_str_to_dict(string, value_type=str, sep=','):
    items = [s.split(':') for s in string.replace(' ', '').split(sep)]
    return {k: value_type(v) for k, v in items}

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def set_batchnorm_momentum(module, momentum):
    from apex import parallel

    if isinstance(module, (nn.BatchNorm2d, parallel.SyncBatchNorm)):
        module.momentum = momentum

def parse_args_line(line):
    # Parse a value from string
    parts = line[:-1].split(': ')
    if len(parts) > 2:
        parts = [parts[0], ': '.join(parts[1:])]
    k, v = parts
    v_type = str
    if v.isdigit():
        v = int(v)
        v_type = int
    elif isfloat(v) and k != 'per_loss_weights' and k != 'pix_loss_weights': # per_loss_weights can be a string of floats
        v_type = float
        v = float(v)
    elif v == 'True':
        v = True
    elif v == 'False':
        v = False

    return k, v, v_type

############################################################
# Hook for calculation of "standing" statistics for BN lrs #
# Net. should be run over the validation set in train mode #
############################################################

class StatsCalcHook(object):
    def __init__(self):
        self.num_iter = 0
    
    def update_stats(self, module):
        for stats_name in ['mean', 'var']:
            batch_stats = getattr(module, f'running_{stats_name}')
            accum_stats = getattr(module, f'accumulated_{stats_name}')
            accum_stats = accum_stats + batch_stats
            setattr(module, f'accumulated_{stats_name}', accum_stats)
        
        self.num_iter += 1

    def remove(self, module):
        for stats_name in ['mean', 'var']:
            accum_stats = getattr(module, f'accumulated_{stats_name}') / self.num_iter
            delattr(module, f'accumulated_{stats_name}')
            getattr(module, f'running_{stats_name}').data = accum_stats

    def __call__(self, module, inputs, outputs):
        self.update_stats(module)

    @staticmethod
    def apply(module):
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, StatsCalcHook):
                raise RuntimeError("Cannot register two calc_stats hooks on "
                                   "the same module")
                
        fn = StatsCalcHook()
        
        stats = getattr(module, 'running_mean')
        for stats_name in ['mean', 'var']:
            attr_name = f'accumulated_{stats_name}'
            if hasattr(module, attr_name): 
                delattr(module, attr_name)
            module.register_buffer(attr_name, torch.zeros_like(stats))

        module.register_forward_hook(fn)
        
        return fn


def stats_calculation(module):
    if 'BatchNorm' in module.__class__.__name__:
        StatsCalcHook.apply(module)

    return module

def remove_stats_calculation(module):
    if 'BatchNorm' in module.__class__.__name__:
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, StatsCalcHook):
                hook.remove(module)
                del module._forward_hooks[k]
                return module

    return module

############################################################
# Spectral normalization                                   #
# Can be applied recursively (compared to PyTorch version) #
############################################################

def spectral_norm(module, name='weight', apply_to=['conv2d'], n_power_iterations=1, eps=1e-12):
    # Apply only to modules in apply_to list
    module_name = module.__class__.__name__.lower()
    if module_name not in apply_to or 'adaptive' in module_name:
        return module

    if isinstance(module, nn.ConvTranspose2d):
        dim = 1
    else:
        dim = 0

    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)

    return module

def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break

    return module

############################################################
# Weight averaging (both EMA and plain averaging)          #
# -- works in a similar way to spectral_norm in PyTorch    #
# -- has to be applied and removed AFTER spectral_norm     #
# -- can be applied via .apply method of any nn.Module     #
############################################################

class WeightAveragingHook(object):
    # Mode can be either "running_average" with momentum
    # or "average" for direct averaging
    def __init__(self, name='weight', mode='running_average', momentum=0.9999):
        self.name = name
        self.mode = mode
        self.momentum = momentum # running average parameter
        self.num_iter = 1 # average parameter

    def update_param(self, module):
        # Only update average values
        param = getattr(module, self.name)
        param_avg = getattr(module, self.name + '_avg')
        with torch.no_grad():
            if self.mode == 'running_average':
                param_avg.data = param_avg.data * self.momentum + param.data * (1 - self.momentum)
            elif self.mode == 'average':
                param_avg.data = (param_avg.data * self.num_iter + param.data) / (self.num_iter + 1)
                self.num_iter += 1

    def remove(self, module):
        param_avg = getattr(module, self.name + '_avg')
        delattr(module, self.name)
        delattr(module, self.name + '_avg')
        module.register_parameter(self.name, nn.Parameter(param_avg))

    def __call__(self, module, grad_input, grad_output):
        if module.training: 
            self.update_param(module)

    @staticmethod
    def apply(module, name, mode, momentum):
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, WeightAveragingHook) and hook.name == name:
                raise RuntimeError("Cannot register two weight_averaging hooks on "
                                   "the same parameter {}".format(name))
                
        fn = WeightAveragingHook(name, mode, momentum)
        
        if name in module._parameters:
            param = module._parameters[name].data
        else:
            param = getattr(module, name)

        module.register_buffer(name + '_avg', param.clone())

        module.register_backward_hook(fn)
        
        return fn

class WeightAveragingPreHook(object):
    # Mode can be either "running_average" with momentum
    # or "average" for direct averaging
    def __init__(self, name='weight'):
        self.name = name
        self.spectral_norm = True
        self.enable = False

    def __call__(self, module, inputs):
        if self.enable or not module.training:
            setattr(module, self.name, getattr(module, self.name + '_avg'))

        elif not self.spectral_norm:
            setattr(module, self.name, getattr(module, self.name + '_orig') + 0) # +0 converts a parameter to a tensor with grad fn

    @staticmethod
    def apply(module, name):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightAveragingPreHook) and hook.name == name:
                raise RuntimeError("Cannot register two weight_averaging hooks on "
                                   "the same parameter {}".format(name))
                
        fn = WeightAveragingPreHook(name)

        if not hasattr(module, name + '_orig'):
            param = module._parameters[name]

            delattr(module, name)
            module.register_parameter(name + '_orig', param)
            setattr(module, name, param.data)

            fn.spectral_norm = False

        module.register_forward_pre_hook(fn)
        
        return fn


def weight_averaging(module, names=['weight', 'bias'], mode='running_average', momentum=0.9999):
    for name in names:
        if hasattr(module, name) and getattr(module, name) is not None:
            WeightAveragingHook.apply(module, name, mode, momentum)
            WeightAveragingPreHook.apply(module, name)

    return module


def remove_weight_averaging(module, names=['weight', 'bias']):
    for name in names:               
        for k, hook in module._backward_hooks.items():
            if isinstance(hook, WeightAveragingHook) and hook.name == name:
                hook.remove(module)
                del module._backward_hooks[k]
                break

        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightAveragingPreHook) and hook.name == name:
                hook.remove(module)
                del module._forward_pre_hooks[k]
                break

    return module

#############################################################
#          Postprocessing of modules for inference          #
#############################################################

def prepare_for_mobile_inference(module):
    mod = module

    if isinstance(module, nn.InstanceNorm2d) or isinstance(module, nt_utils.AdaptiveNorm2d) and isinstance(module.norm_layer, nn.InstanceNorm2d):
        # Split affine part of instance norm into separable 1x1 conv
        new_mod_1 = nn.InstanceNorm2d(module.num_features, eps=module.eps, affine=False)
        
        weight_data = module.weight.data.squeeze().detach().clone()
        bias_data = module.bias.data.squeeze().detach().clone()
        
        new_mod_2 = nn.Conv2d(module.num_features, module.num_features, 1, groups=module.num_features)
        
        new_mod_2.weight.data = weight_data.view(module.num_features, 1, 1, 1)
        new_mod_2.bias.data = bias_data

        mod = nn.Sequential(
            new_mod_1,
            new_mod_2)

    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nt_utils.AdaptiveNorm2d) and isinstance(module.norm_layer, nn.BatchNorm2d):
        mod = nn.Conv2d(module.num_features, module.num_features, 1, groups=module.num_features)

        if isinstance(module, nt_utils.AdaptiveNorm2d):
            sigma = (module.norm_layer.running_var + module.norm_layer.eps)**0.5
            mu = module.norm_layer.running_mean
        else:
            sigma = (module.running_var + module.eps)**0.5
            mu = module.running_mean
            
        sigma = sigma.clone()
        mu = mu.clone()

        gamma = module.weight.data.squeeze().detach().clone()
        beta = module.bias.data.squeeze().detach().clone()
        
        mod.weight.data[:, 0, 0, 0] = gamma / sigma
        mod.bias.data = beta - mu / sigma * gamma

    elif isinstance(module, nt_utils.AdaptiveConv2d):
        mod = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, bias=True)
        mod.weight.data = module.weight.data[0].detach().clone()
        mod.bias.data = module.bias.data[0].detach().clone()

    else:
        for name, child in module.named_children():
            mod.add_module(name, prepare_for_mobile_inference(child))

    del module
    return mod
# Third party
import importlib
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

# This project
from runners import utils
from datasets import utils as ds_utils



class RunnerWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        # Networks used in train and test
        parser.add('--networks_train',       default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator', 
                                             help    = 'order of forward passes during the training of gen (or gen and dis for sim sgd)')

        parser.add('--networks_test',        default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator', 
                                             help    = 'order of forward passes during testing')

        parser.add('--networks_calc_stats',  default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator', 
                                             help    = 'order of forward passes during stats calculation')

        parser.add('--networks_to_train',    default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator', 
                                             help    = 'names of networks that are being trained')

        # Losses used in train and test
        parser.add('--losses_train',         default = 'adversarial, feature_matching, perceptual, pixelwise, segmentation, warping_regularizer', 
                                             help    = 'losses evaluated during training')

        parser.add('--losses_test',          default = 'lpips, csim', 
                                             help    = 'losses evaluated during testing')

        # Spectral norm options
        parser.add('--spn_networks',         default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator',
                                             help    = 'networks to apply spectral norm to')

        parser.add('--spn_exceptions',       default = '',
                                             help    = 'a list of exceptional submodules that have spectral norm removed')

        parser.add('--spn_layers',           default = 'conv2d, linear',
                                             help    = 'layers to apply spectral norm to')

        # Weight averaging options
        parser.add('--wgv_mode',             default = 'none', 
                                             help    = 'none|running_average|average -- exponential moving averaging or weight averaging')

        parser.add('--wgv_momentum',         default = 0.999,  type=float, 
                                             help    = 'momentum value in EMA weight averaging')

        # Training options
        parser.add('--eps',                  default = 1e-7,   type=float)

        parser.add('--optims',               default = 'identity_embedder: adam, texture_generator: adam, keypoints_embedder: adam, inference_generator: adam, discriminator: adam',
                                             help    = 'network_name: optimizer')

        parser.add('--lrs',                  default = 'identity_embedder: 2e-4, texture_generator: 2e-4, keypoints_embedder: 2e-4, inference_generator: 2e-4, discriminator: 2e-4',
                                             help    = 'learning rates for each network')

        parser.add('--stats_calc_iters',     default = 100,    type=int, 
                                             help    = 'number of iterations used to calculate standing statistics')

        parser.add('--num_visuals',          default = 32,     type=int, 
                                             help    = 'the total number of output visuals')

        parser.add('--bn_momentum',          default = 1.0,    type=float, 
                                             help    = 'momentum of the batchnorm layers')

        parser.add('--adam_beta1',           default = 0.5,    type=float, 
                                             help    = 'beta1 (momentum of the gradient) parameter for Adam')

        args, _ = parser.parse_known_args()

        # Add args from the required networks
        networks_names = list(set(
            utils.parse_str_to_list(args.networks_train, sep=',')
            + utils.parse_str_to_list(args.networks_test, sep=',')))
        for network_name in networks_names:
            importlib.import_module(f'networks.{network_name}').NetworkWrapper.get_args(parser)
        
        # Add args from the losses
        losses_names = list(set(
            utils.parse_str_to_list(args.losses_train, sep=',')
            + utils.parse_str_to_list(args.losses_test, sep=',')))
        for loss_name in losses_names:
            importlib.import_module(f'losses.{loss_name}').LossWrapper.get_args(parser)

        return parser

    def __init__(self, args, training=True):
        super(RunnerWrapper, self).__init__()
        # Store general options
        self.args = args
        self.training = training

        # Read names lists from the args
        self.load_names(args)

        # Initialize classes for the networks
        nets_names = self.nets_names_test
        if self.training:
            nets_names += self.nets_names_train
        nets_names = list(set(nets_names))

        self.nets = nn.ModuleDict()

        for net_name in sorted(nets_names):
            self.nets[net_name] = importlib.import_module(f'networks.{net_name}').NetworkWrapper(args)

            if args.num_gpus > 1:
                # Apex is only needed for multi-gpu training
                from apex import parallel

                self.nets[net_name] = parallel.convert_syncbn_model(self.nets[net_name])

        # Set nets that are not training into eval mode
        for net_name in self.nets.keys():
            if net_name not in self.nets_names_to_train:
                self.nets[net_name].eval()

        # Initialize classes for the losses
        if self.training:
            losses_names = list(set(self.losses_names_train + self.losses_names_test))
            self.losses = nn.ModuleDict()

            for loss_name in sorted(losses_names):
                self.losses[loss_name] = importlib.import_module(f'losses.{loss_name}').LossWrapper(args)

        # Spectral norm
        if args.spn_layers:
            spn_layers = utils.parse_str_to_list(args.spn_layers, sep=',')
            spn_nets_names = utils.parse_str_to_list(args.spn_networks, sep=',')

            for net_name in spn_nets_names:
                self.nets[net_name].apply(lambda module: utils.spectral_norm(module, apply_to=spn_layers, eps=args.eps))

            # Remove spectral norm in modules in exceptions
            spn_exceptions = utils.parse_str_to_list(args.spn_exceptions, sep=',')

            for full_module_name in spn_exceptions:
                if not full_module_name:
                    continue

                parts = full_module_name.split('.')

                # Get the module that needs to be changed
                module = self.nets[parts[0]]
                for part in parts[1:]:
                    module = getattr(module, part)

                module.apply(utils.remove_spectral_norm)

        # Weight averaging
        if args.wgv_mode != 'none':
            # Apply weight averaging only for networks that are being trained
            for net_name, _ in self.nets_names_to_train:
                self.nets[net_name].apply(lambda module: utils.weight_averaging(module, mode=args.wgv_mode, momentum=args.wgv_momentum))

        # Check which networks are being trained and put the rest into the eval mode
        for net_name in self.nets.keys():
            if net_name not in self.nets_names_to_train:
                self.nets[net_name].eval()

        # Set the same batchnorm momentum accross all modules
        if self.training:
            self.apply(lambda module: utils.set_batchnorm_momentum(module, args.bn_momentum))

        # Store a history of losses and images for visualization
        self.losses_history = {
            True: {}, # self.training = True
            False: {}}

    def forward(self, data_dict):
        ### Set lists of networks' and losses' names ###
        if self.training:
            nets_names = self.nets_names_train
            networks_to_train = self.nets_names_to_train

            losses_names = self.losses_names_train

        else:
            nets_names = self.nets_names_test
            networks_to_train = []

            losses_names = self.losses_names_test

        # Forward pass through all the required networks
        self.data_dict = data_dict
        for net_name in nets_names:
            self.data_dict = self.nets[net_name](self.data_dict, networks_to_train, self.nets)

        # Forward pass through all the losses
        losses_dict = {}
        for loss_name in losses_names:
            if hasattr(self, 'losses') and loss_name in self.losses.keys():
                losses_dict = self.losses[loss_name](self.data_dict, losses_dict)

        # Calculate the total loss and store history
        loss = self.process_losses_dict(losses_dict)

        return loss

    ########################################################
    #                     Utility functions                #
    ########################################################

    def load_names(self, args):
        # Initialize utility lists and dicts for the networks
        self.nets_names_to_train = utils.parse_str_to_list(args.networks_to_train)
        self.nets_names_train = utils.parse_str_to_list(args.networks_train)
        self.nets_names_test = utils.parse_str_to_list(args.networks_test)
        self.nets_names_calc_stats = utils.parse_str_to_list(args.networks_calc_stats)

        # Initialize utility lists and dicts for the networks
        self.losses_names_train = utils.parse_str_to_list(args.losses_train)
        self.losses_names_test = utils.parse_str_to_list(args.losses_test)

    def get_optimizers(self, args):
        # Initialize utility lists and dicts for the optimizers
        nets_optims_names = utils.parse_str_to_dict(args.optims)
        nets_lrs = utils.parse_str_to_dict(args.lrs, value_type=float)

        # Initialize optimizers
        optims = {}

        for net_name, optim_name in nets_optims_names.items():
            # Prepare the options
            lr = nets_lrs[net_name]
            optim_name = optim_name.lower()
            params = self.nets[net_name].parameters()

            # Choose the required optimizer
            if optim_name == 'adam':
                opt = optim.Adam(params, lr=lr, eps=args.eps, betas=(args.adam_beta1, 0.999))

            elif optim_name == 'sgd':
                opt = optim.SGD(params, lr=lr)

            elif optim_name == 'fusedadam':
                from apex import optimizers

                opt = optimizers.FusedAdam(params, lr=lr, eps=args.eps, betas=(args.adam_beta1, 0.999))

            elif optim_name == 'fusedsgd':
                from apex import optimizers

                opt = optimizers.FusedSGD(params, lr=lr)

            elif optim_name == 'lbfgs':
                opt = optim.LBFGS(params, lr=lr)

            else:
                raise 'Unsupported optimizer name'

            optims[net_name] = opt

        return optims

    def process_losses_dict(self, losses_dict):
        # This function appends loss value into losses_dict
        loss = torch.zeros(1)
        if self.args.num_gpus > 0:
            loss = loss.cuda()

        for key, value in losses_dict.items():
            if key not in self.losses_history[self.training]: 
                self.losses_history[self.training][key] = []
            
            self.losses_history[self.training][key] += [value.item()]
            loss += value
            
        return loss

    def output_losses(self):
        losses = {}

        for key, values in self.losses_history[self.training].items():
            value = torch.FloatTensor(values)

            # Average the losses
            losses[key] = value.cpu().mean()

        # Clear losses hist
        self.losses_history[self.training] = {}

        if self.args.rank != 0:
            return None
        else:
            return losses

    def output_visuals(self):
        # This function creates an output grid of visuals
        visuals_data_dict = {}

        # Only first source and target frame is visualized
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                visuals_data_dict[k] = v[:self.args.num_visuals, 0]

        # Collect the visuals from all submodules
        visuals = []
        for net_name in self.nets_names_train:
            visuals += self.nets[net_name].visualize_outputs(visuals_data_dict)

        visuals = torch.cat(visuals, 3) # cat w.r.t. width
        visuals = torch.cat(visuals.split(1, 0), 2)[0] # cat batch dim in lines w.r.t. height
        visuals = (visuals + 1.) * 0.5 # convert back to [0, 1] range
        visuals = visuals.clamp(0, 1)

        return visuals.cpu()

    def train(self, mode=True):
        self.training = mode
        # Only change the mode of modules thst are being trained
        for net_name in self.nets_names_to_train:
            if net_name in self.nets.keys():
                self.nets[net_name].train(mode)

        return self

    def calculate_batchnorm_stats(self, train_dataloader, debug=False):
        for net_name in self.nets_names_calc_stats:
            self.nets[net_name].apply(utils.stats_calculation)

            # Set spectral norm and weight averaging to eval
            def set_modules_to_eval(module):
                if 'BatchNorm' in module.__class__.__name__:
                    return module

                else:
                    module.eval()

                    return module

            self.nets[net_name].apply(set_modules_to_eval)

        for i, self.data_dict in enumerate(train_dataloader, 1):            
            # Prepare input data
            if self.args.num_gpus > 0:
                for key, value in self.data_dict.items():
                    self.data_dict[key] = value.cuda()

            # Forward pass
            with torch.no_grad():
                for net_name in self.nets_names_calc_stats:
                    self.data_dict = self.nets[net_name](self.data_dict, [], self.nets)

            # Break if the required number of iterations is done
            if i == self.args.stats_calc_iters:
                break

            # Do only one iteration in case of debugging
            if debug and i == 10:
                break

        # Merge the buffers into running stats
        for net_name in self.nets_names_calc_stats:
            self.nets[net_name].apply(utils.remove_stats_calculation)

