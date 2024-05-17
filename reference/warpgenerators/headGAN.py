import os
import cv2
import numpy as np
import pandas as pd
import scipy.signal
from PIL import Image
import torch
import argparse
from facenet_pytorch import MTCNN, extract_face
import collections
from tqdm import tqdm
from util.util import *


def save_images(images, name, split, args):
    print('Saving images')
    lim = len(images) - len(images) % args.train_seq_length if split == 'train' else len(images)
    for i in tqdm(range(lim)):
        n_frame = "{:06d}".format(i)
        part = "_{:06d}".format((i) // args.train_seq_length) if split == 'train' else ""
        save_dir = os.path.join(args.dataset_path, split, 'images', name + part)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(images[i], os.path.join(save_dir, n_frame + '.png'))


def get_split_dict(csv_file, args):
    if csv_file and os.path.exists(csv_file):
        csv = pd.read_csv(csv_file)
        names = list(csv['filename'])
        names = [os.path.splitext(name)[0] for name in names]
        splits = list(csv['split'])
        split_dict = dict(zip(names, splits))
        return split_dict
    else:
        print('No metadata file provided. All samples will be saved in the %s split.' % args.split)
        return None


def get_vid_paths_dict(dir):
    # Returns dict: {vid_name: path, ...}
    if os.path.exists(dir) and is_video_file(dir):
        # If path to single .mp4 file was given directly.
        # If '_' in file name remove it, since it causes problems.
        vid_files = {os.path.splitext(os.path.basename(dir))[0].replace('_', '') : dir}
    else:
        vid_files = {}
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_video_file(fname):
                    path = os.path.join(root, fname)
                    name = os.path.splitext(fname)[0]
                    if name not in vid_files:
                        vid_files[name] = path
    return collections.OrderedDict(sorted(vid_files.items()))


def is_vid_path_processed(name, split, args):
    first_part = '_000000' if split == 'train' else ''
    path = os.path.join(args.dataset_path, split, 'images', name + first_part)
    return os.path.isdir(path)


def check_boxes(boxes, img_size, args):
    # Check if there are None boxes.
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = next((item for item in boxes[i+1:] if item is not None), boxes[i-1])
    if boxes[0] is None:
        print('Not enough boxes detected.')
        return False, [None]
    boxes = [box[0] for box in boxes]
    # Check if detected faces are very far from each other. Check distances between all boxes.
    maxim_dst = 0
    for i in range(len(boxes)-1):
        for j in range(len(boxes)-1):
            dst = max(abs(boxes[i] - boxes[j])) / img_size
            if dst > maxim_dst:
                maxim_dst = dst
    if maxim_dst > args.dst_threshold:
         print('L_inf distance between bounding boxes %.4f larger than threshold' % maxim_dst)
         return False, [None]
    # Get average box
    avg_box = np.median(boxes, axis=0)
    # Make boxes square.
    offset_w = avg_box[2] - avg_box[0]
    offset_h = avg_box[3] - avg_box[1]
    offset_dif = (offset_h - offset_w) / 2
    # width
    avg_box[0] = avg_box[2] - offset_w - offset_dif
    avg_box[2] = avg_box[2] + offset_dif
    # height - center a bit lower
    avg_box[3] = avg_box[3] + args.height_recentre * offset_h
    avg_box[1] = avg_box[3] - offset_h
    return True, avg_box


def get_faces(detector, images, args):
    ret_faces = []
    all_boxes = []
    avg_box = None
    all_imgs = []
    # Get bounding boxes
    print('Getting bounding boxes')
    for lb in tqdm(np.arange(0, len(images), args.mtcnn_batch_size)):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+args.mtcnn_batch_size]]
        boxes, _, _ = detector.detect(imgs_pil, landmarks=True)
        all_boxes.extend(boxes)
        all_imgs.extend(imgs_pil)
    # Check if boxes are fine, do temporal smoothing, return average box.
    img_size = (all_imgs[0].size[0] + all_imgs[0].size[1]) / 2
    stat, avg_box = check_boxes(all_boxes, img_size, args)
    # Crop face regions.
    if stat:
        print('Extracting faces')
        for img in tqdm(all_imgs, total=len(all_imgs)):
            face = extract_face(img, avg_box, args.cropped_image_size, args.margin)
            ret_faces.append(face)
    return stat, ret_faces


def detect_and_save_faces(detector, name, file_path, split, args):
    if is_video_file(file_path):
        images, fps = read_mp4(file_path, args.n_replicate_first)
    else:
        images = read_image(file_path, args)
    if not args.no_crop:
        stat, face_images = get_faces(detector, images, args)
    else:
        stat, face_images = True, images
    if stat:
        save_images(tensor2npimage(face_images), name, split, args)
    return stat


def main():
    print('-------------- Face detector -------------- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_files_path', type=str, required=True,
                        help='Path to videos root directory')
    parser.add_argument('--dataset_path', type=str, default='datasets/voxceleb', 
                        help='Path to save dataset.')
    parser.add_argument('--gpu_id', type=str, default='0', 
                        help='Negative value to use CPU, or greater equal than zero for GPU id.')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to metadata (train/test split information).')
    parser.add_argument('--no_crop', action='store_true', help='Save the frames without face detection and cropping.')
    parser.add_argument('--mtcnn_batch_size', default=1, type=int, help='The number of frames for face detection.')
    parser.add_argument('--cropped_image_size', default=256, type=int, help='The size of frames after cropping the face.')
    parser.add_argument('--margin', default=100, type=int, help='.')
    parser.add_argument('--dst_threshold', default=0.45, type=float, help='Max L_inf distance between any bounding boxes in a video. (normalised by image size: (h+w)/2)')
    parser.add_argument('--height_recentre', default=0.0, type=float, help='The amount of re-centring bounding boxes lower on the face.')
    parser.add_argument('--train_seq_length', default=50, type=int, help='The number of frames for each training sub-sequence.')
    parser.add_argument('--split', default='train', choices=['train', 'test'], type=str, help='The split for data [train|test]')
    parser.add_argument('--n_replicate_first', default=0, type=int, help='How many times to replicate and append the first frame to the beginning of the video.')

    args = parser.parse_args()

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Read metadata file to create data split
    split_dict = get_split_dict(args.metadata_path, args)

    # Store file paths in dictionary.
    files_paths_dict = get_vid_paths_dict(args.original_files_path)
    n_files = len(files_paths_dict)
    print('Number of files to process: %d \n' % n_files)

    # Initialize the MTCNN face  detector.
    detector = MTCNN(image_size=args.cropped_image_size, margin=args.margin, post_process=False, device=device)

    # Run detection
    n_completed = 0
    for name, path in files_paths_dict.items():
        n_completed += 1
        split = split_dict[name] if split_dict else args.split
        if not is_vid_path_processed(name, split, args):
            success = detect_and_save_faces(detector, name, path, split, args)
            if success:
                print('(%d/%d) %s (%s file) [SUCCESS]' % (n_completed, n_files, path, split))
            else:
                print('(%d/%d) %s (%s file) [FAILED]' % (n_completed, n_files, path, split))
        else:
            print('(%d/%d) %s (%s file) already processed!' % (n_completed, n_files, path, split))

if __name__ == "__main__":
    main()

import random
import numpy as np
import torch
from torch.autograd import Variable
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

import os
import ntpath
import time
import collections
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=10)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)

                len_ims = len(ims)
                ims_per_batch = int(len_ims / self.opt.batch_size)
                n_prev = self.opt.n_frames_G-1 if not self.opt.no_previousframesencoder else 0
                n_frames_print_second_row = 5
                for i in range(self.opt.batch_size):
                    first_row_ims = ims[ims_per_batch*i:ims_per_batch*i+3+n_prev]
                    first_row_txts = txts[ims_per_batch*i:ims_per_batch*i+3+n_prev]
                    first_row_links = links[ims_per_batch*i:ims_per_batch*i+3 +n_prev]
                    webpage.add_images(first_row_ims, first_row_txts, first_row_links, width=self.win_size)
                    second_row_ims = ims[ims_per_batch*i+3+n_prev:ims_per_batch*i+3+n_prev+n_frames_print_second_row]
                    second_row_txts = txts[ims_per_batch*i+3+n_prev:ims_per_batch*i+3+n_prev+n_frames_print_second_row]
                    second_row_links = links[ims_per_batch*i+3+n_prev:ims_per_batch*i+3+n_prev+n_frames_print_second_row]
                    webpage.add_images(second_row_ims, second_row_txts, second_row_links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        errors_sorted = collections.OrderedDict(sorted(errors.items()))
        if self.tf_log:
            for tag, value in errors_sorted.items():
                summary = self.tf.compat.v1.Summary(value=[self.tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, t_epoch):
        errors_sorted = collections.OrderedDict(sorted(errors.items()))
        message = 'Epoch: %d, sequences seen: %d, total epoch time: %d hrs %d mins (secs per sequence: %.3f) \n' % (epoch, i, t_epoch[0], t_epoch[1], t)
        for k, v in errors_sorted.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, image_dir, visuals, image_path, webpage=None):
        dirname = os.path.basename(os.path.dirname(image_path[0]))
        image_dir = os.path.join(image_dir, dirname)
        util.mkdir(image_dir)
        name = os.path.basename(image_path[0])
        name = os.path.splitext(name)[0]

        if webpage is not None:
            webpage.add_header(name)
            ims, txts, links = [], [], []

        for label, image_numpy in visuals.items():
            util.mkdir(os.path.join(image_dir, label))
            image_name = '%s.%s' % (name, 'png')
            save_path = os.path.join(image_dir, label, image_name)
            util.save_image(image_numpy, save_path)

            if webpage is not None:
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
        if webpage is not None:
            webpage.add_images(ims, txts, links, width=self.win_size)

    def vis_print(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

import datetime
import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir, title, refresh=0):
        if web_dir.endswith('.html'):
            web_dir, html_name = os.path.split(web_dir)
        else:
            web_dir, html_name = web_dir, 'index.html'
        self.title = title
        self.web_dir = web_dir
        self.html_name = html_name
        self.img_dir = os.path.join(self.web_dir, 'images')
        if len(self.web_dir) > 0 and not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if len(self.web_dir) > 0 and not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        with self.doc:
            h1(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=512):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % (width), src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = os.path.join(self.web_dir, self.html_name)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.jpg' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.jpg' % n)
    html.add_images(ims, txts, links)
    html.save()

import importlib
import os
import torch
import numpy as np
from PIL import Image
import torchvision
import cv2
from tqdm import tqdm

VID_EXTENSIONS = ['.mp4']

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM', '.png', '.PNG', 
                  '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.txt', '.json']

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return [image]

def read_mp4(mp4_path, n_replicate_first):
    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    images = []
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Reading %s' % mp4_path)
    for i in tqdm(range(n_frames)):
        _, image = reader.read()
        if image is None:
            break
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    reader.release()
    if n_replicate_first > 0:
        pad = [images[0]] * n_replicate_first
        pad.extend(images)
        images = pad
    return images, fps

def seconds_to_hours_mins(t_sec):
    t_mins = t_sec //  60
    hours = t_mins // 60
    mins = t_mins - 60 * hours
    return hours, mins

def prepare_input(input_A, ref_input_A, ref_input_B):
    N, n_frames_G, channels, height, width = input_A.size()
    input = input_A.view(N, n_frames_G * channels, height, width)
    ref_input = torch.cat([ref_input_A, ref_input_B], dim=1)
    return input, ref_input

# get temporally subsampled frames for real/fake sequences
def get_skipped_frames(B_all, B, t_scales, n_frames_D):
    B_all = torch.cat([B_all.detach(), B], dim=1) if B_all is not None else B
    B_skipped = [None] * t_scales
    for s in range(t_scales):
        n_frames_Ds = n_frames_D ** s
        span = n_frames_Ds * (n_frames_D-1)
        if B_all.size()[1] > span:
            B_skipped[s] = B_all[:, -span-1::n_frames_Ds].contiguous()
    max_prev_frames = n_frames_D ** (t_scales-1) * (n_frames_D-1)
    if B_all.size()[1] > max_prev_frames:
        B_all = B_all[:, -max_prev_frames:]
    return B_all, B_skipped

def tensor2npimage(image_tensor):
    # Input tesnor in range [0, 255]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2npimage(image_tensor[i]))
        return image_numpy
    if torch.is_tensor(image_tensor):
        image_numpy = np.transpose(image_tensor.cpu().float().numpy(), (1, 2, 0))
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(np.uint8)

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    # Input tesnor in range [0, 1] or [-1, 1]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = np.tile(image_numpy, (1,1,3))
    return image_numpy.astype(imtype)

def tensor2flow(output, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float().numpy()
    output = np.transpose(output, (1, 2, 0))
    #mag = np.max(np.sqrt(output[:,:,0]**2 + output[:,:,1]**2))
    #print(mag)
    hsv = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(output[..., 0], output[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_video(frames, save_path, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fit_ROI_in_frame(center, opt):
    center_w, center_h = center[0], center[1]
    center_h = torch.tensor(opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_h < opt.ROI_size // 2 else center_h
    center_w = torch.tensor(opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_w < opt.ROI_size // 2 else center_w
    center_h = torch.tensor(opt.crop_size - opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_h > opt.crop_size - opt.ROI_size // 2 else center_h
    center_w = torch.tensor(opt.crop_size - opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_w > opt.crop_size - opt.ROI_size // 2 else center_w
    return (center_w, center_h)

def crop_ROI(img, center, ROI_size):
    return img[..., center[1] - ROI_size // 2:center[1] + ROI_size // 2,
                    center[0] - ROI_size // 2:center[0] + ROI_size // 2]

def get_ROI(tensors, centers, opt):
    real_B, fake_B = tensors
    # Extract region of interest around the center.
    real_B_ROI = []
    fake_B_ROI = []
    for t in range(centers.shape[0]):
        center = fit_ROI_in_frame(centers[t], opt)
        real_B_ROI.append(crop_ROI(real_B[t], center, opt.ROI_size))
        fake_B_ROI.append(crop_ROI(fake_B[t], center, opt.ROI_size))
    real_B_ROI = torch.stack(real_B_ROI, dim=0)
    fake_B_ROI = torch.stack(fake_B_ROI, dim=0)
    return real_B_ROI, fake_B_ROI

def smoothen_signal(S, window_size=15):
    left_p = window_size // 2
    right_p =  window_size // 2 if window_size % 2 == 1 else window_size // 2 - 1
    window = np.ones(int(window_size))/float(window_size) # kernel-filter
    S = np.array(S)
    # Padding
    left_padding = np.stack([S[0]] * left_p, axis=0)
    right_padding = np.stack([S[-1]] * right_p, axis=0)
    S_padded = np.concatenate([left_padding, S, right_padding])
    if len(S_padded.shape) == 1:
        S = np.convolve(S_padded, window, 'valid')
    else:
        for coord in range(S_padded.shape[1]):
            S[:, coord] = np.convolve(S_padded[:, coord], window, 'valid')
    return S

def adapt_cam_params(s_cam_params, t_cam_params, args):
    cam_params = s_cam_params
    if not args.no_scale_or_translation_adaptation:
        mean_S_target = np.mean([params[0] for params in t_cam_params])
        mean_S_source = np.mean([params[0] for params in s_cam_params])
        S = [params[0] * (mean_S_target / mean_S_source)
             for params in s_cam_params]
        # Smoothen scale
        S = smoothen_signal(S)
        # Normalised Translation for source and target.
        nT_target = [params[2] / params[0] for params in t_cam_params]
        nT_source = [params[2] / params[0] for params in s_cam_params]
        cam_params = [(s, params[1], s * t) \
                      for s, params, t in zip(S, s_cam_params, nT_source)]
        if not args.no_translation_adaptation:
            mean_nT_target = np.mean(nT_target, axis=0)
            mean_nT_source = np.mean(nT_source, axis=0)
            if args.standardize:
                std_nT_target = np.std(nT_target, axis=0)
                std_nT_source = np.std(nT_source, axis=0)
                nT = [(t - mean_nT_source) * std_nT_target / std_nT_source \
                     + mean_nT_target for t in nT_source]
            else:
                nT = [t - mean_nT_source + mean_nT_target
                      for t in nT_source]
            # Smoothen translation
            nT = smoothen_signal(nT)
            cam_params = [(s, params[1], s * t) \
                          for s, params, t in zip(S, s_cam_params, nT)]
    return cam_params

def make_image_square(image):
    h, w = image.shape[:2]
    d = abs(h - w)
    if h > w:
        image = image[d // 2: d // 2 + w, :, :]
    else:
        image = image[:, d // 2: d // 2 + h, :]
    return image
import cv2
import os
import numpy as np
import argparse
import sys
import collections
import torch
from shutil import copyfile, rmtree
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from helpers.reconstruction import NMFCRenderer
from util.util import *


# Find and save the most diverse poses in the video.
def save_reference_samples(SRT, args):
    SRT_data = np.array([srt[1][1:-2] for srt in SRT])
    paths = [srt[0] for srt in SRT]

    # Perform PCA
    SRT_data = StandardScaler().fit_transform(SRT_data)
    pca = PCA(n_components=1)
    data = pca.fit_transform(SRT_data)
    data = [item for sublist in data for item in sublist]

    # Sort component to select most diverse ones.
    data, paths = (list(t) for t in zip(*sorted(zip(data, paths))))
    points = list(np.linspace(0, len(data)-1, args.n_reference_to_save, endpoint=True).astype(np.int32))
    data = [data[p] for p in points]

    # Paths
    nmfcs_paths = [paths[p] for p in points]
    images_paths = [p.replace('/nmfcs/', '/images/') for p in nmfcs_paths]
    dir = os.path.dirname(nmfcs_paths[0])
    parent = os.path.dirname(dir)
    base = os.path.basename(dir)[:-7] if '/train/' in dir else os.path.basename(dir)
    nmfcs_reference_parent = os.path.join(parent, base).replace('/nmfcs/', '/nmfcs_fs/')
    images_reference_parent = nmfcs_reference_parent.replace('/nmfcs_fs/', '/images_fs/')
    mkdir(images_reference_parent)
    mkdir(nmfcs_reference_parent)

    # Copy images
    for i in range(len(nmfcs_paths)):
        copyfile(nmfcs_paths[i], os.path.join(nmfcs_reference_parent, "{:06d}".format(i) + '.png'))
        copyfile(images_paths[i], os.path.join(images_reference_parent, "{:06d}".format(i) + '.png'))


def make_dirs(name, image_pths, args):
    id_coeffs_paths = []
    nmfc_pths = [p.replace('/images/', '/nmfcs/') for p in image_pths]
    out_paths = set(os.path.dirname(nmfc_pth) for nmfc_pth in nmfc_pths)
    for out_path in out_paths:
        mkdir(out_path)
        if args.save_cam_params:
            mkdir(out_path.replace('/nmfcs/', '/misc/'))
        if args.save_landmarks5:
            mkdir(out_path.replace('/nmfcs/', '/landmarks/'))
        if args.save_exp_params:
            mkdir(out_path.replace('/nmfcs/', '/exp_coeffs/'))
    if args.save_id_params:
        splits = set(os.path.dirname(os.path.dirname(os.path.dirname(nmfc_pth))) for nmfc_pth in nmfc_pths)
        for split in splits:
            id_coeffs_path = os.path.join(split, 'id_coeffs')
            mkdir(id_coeffs_path)
            id_coeffs_paths.append(id_coeffs_path)
    return id_coeffs_paths


def remove_images(name, image_pths):
    # Remove images (and landmarks70 if they exist)
    image_dirs_to_remove = set(os.path.dirname(image_pth) for image_pth in image_pths)
    for dir in image_dirs_to_remove:
        if os.path.isdir(dir):
            rmtree(dir)
        landmarks70_dir = dir.replace('/images/', '/landmarks70/')
        if os.path.isdir(landmarks70_dir):
            rmtree(landmarks70_dir)


def save_results(nmfcs, reconstruction_output, name, image_pths, args):
    # Create save directories
    id_coeffs_paths = make_dirs(name, image_pths, args)

    # Save
    SRT_vecs = []
    print('Saving results')
    for nmfc, cam_param, _, exp_param, landmark5, image_pth in tqdm(zip(nmfcs, *reconstruction_output, image_pths), total=len(image_pths)):
        S, R, T = cam_param
        nmfc_pth = image_pth.replace('/images/', '/nmfcs/')
        SRT_vecs.append((nmfc_pth, np.concatenate([np.array([S]), np.array(R).ravel(), np.array(T).ravel()])))
        cv2.imwrite(nmfc_pth, nmfc)
        if args.save_cam_params:
            misc_file = os.path.splitext(image_pth.replace('/images/', '/misc/'))[0] + '.txt'
            misc_file = open(misc_file, "a")
            np.savetxt(misc_file, np.array([S]))
            np.savetxt(misc_file, R)
            np.savetxt(misc_file, T)
            misc_file.close()
        if args.save_landmarks5:
            lands_file = os.path.splitext(image_pth.replace('/images/', '/landmarks/'))[0] + '.txt'
            np.savetxt(lands_file, landmark5)
        if args.save_exp_params:
            exp_params_file = os.path.splitext(image_pth.replace('/images/', '/exp_coeffs/'))[0] + '.txt'
            np.savetxt(exp_params_file, exp_param)
    if args.save_id_params:
        avg_id_params = np.mean(np.array(reconstruction_output[1]), axis=0)
        for id_coeffs_path in id_coeffs_paths:
            id_params_file = os.path.join(id_coeffs_path, name + '.txt')
            np.savetxt(id_params_file, avg_id_params)
    if args.n_reference_to_save > 0:
        save_reference_samples(SRT_vecs, args)


def get_image_paths_dict(dir):
    # Returns dict: {name: [path1, path2, ...], ...}
    image_files = {}
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and '/images/' in root:
                path = os.path.join(root, fname)
                seq_name = os.path.basename(root)[:-7] if '/train/' in root else os.path.basename(root) # Remove part extension for train set
                if seq_name not in image_files:
                    image_files[seq_name] = [path]
                else:
                    image_files[seq_name].append(path)

    # Sort paths for each sequence
    for k, v in image_files.items():
        image_files[k] = sorted(v)

    # Return directory sorted for keys (identity names)
    return collections.OrderedDict(sorted(image_files.items()))


def dirs_exist(image_pths):
    nmfc_pths = [p.replace('/images/', '/nmfcs/') for p in image_pths]
    out_paths = set(os.path.dirname(nmfc_pth) for nmfc_pth in nmfc_pths)
    return all([os.path.exists(out_path) for out_path in out_paths])


def main():
    print('---------- 3D face reconstruction --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='datasets/voxceleb', 
                        help='Path to the dataset directory.')
    parser.add_argument('--gpu_id', type=int, default='0', help='Negative value to use CPU, or greater equal than zero for GPU id.')
    parser.add_argument('--save_cam_params', action='store_true', default=True, help='Save the Scale, Rotation and Translation camera params for each frame.')
    parser.add_argument('--save_id_params', action='store_true', default=True, help='Save the average identity coefficient vector for each video.')
    parser.add_argument('--save_landmarks5', action='store_true', help='Save 5 facial landmarks for each frame.')
    parser.add_argument('--save_exp_params', action='store_true', default=True, help='Save the expression coefficients for each frame.')
    parser.add_argument('--n_reference_to_save', type=int, default='8', help='Number of reference frames to save (32 max). If < 1, dont save reference samples.')
    args = parser.parse_args()

    # Figure out the device
    args.gpu_id = int(args.gpu_id)
    if args.gpu_id < 0:
        args.gpu_id = -1
    elif torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            args.gpu_id = 0
    else:
        print('GPU device not available. Exit.')
        exit(0)

    # Create the directory of image paths.
    images_dict = get_image_paths_dict(args.dataset_path)
    n_image_dirs = len(images_dict)

    print('Number of identities for 3D face reconstruction: %d \n' % n_image_dirs)

    # Initialize the NMFC renderer.
    renderer = NMFCRenderer(args)

    # Iterate through the images_dict
    n_completed = 0
    for name, image_pths in images_dict.items():
        n_completed += 1
        if not dirs_exist(image_pths):
            reconstruction_output = renderer.reconstruct(image_pths)
            if reconstruction_output[0]:
                nmfcs = renderer.computeNMFCs(*reconstruction_output[1:4])
                save_results(nmfcs, reconstruction_output[1:], name, image_pths, args)
                print('(%d/%d) %s [SUCCESS]' % (n_completed, n_image_dirs, name))
            else:
                # If the 3D reconstruction not successful, remove images and video.
                remove_images(name, image_pths)
                print('(%d/%d) %s [FAILED]' % (n_completed, n_image_dirs, name))
        else:
            print('(%d/%d) %s already processed!' % (n_completed, n_image_dirs, name))
            
    # Clean
    renderer.clear()

if __name__=='__main__':
    main()

class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None

        
        

import os
import random
import torch
import numpy as np
from PIL import Image
from dataloader.base_dataset import BaseDataset, get_params, get_transform, get_transform_segmenter, get_video_parameters
from dataloader.image_folder import make_video_dataset, assert_valid_pairs
from dataloader.landmarks_to_sketch import create_landmarks_sketch

class videoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        input_type = 'landmarks70' if opt.use_landmarks_input else 'nmfcs'
        max_seqs_per_identity = self.opt.max_seqs_per_identity if opt.isTrain else None

        # Get dataset directories.
        if not self.opt.no_audio_input:
            self.dir_audio_feats = os.path.join(opt.dataroot, self.opt.phase, 'audio_features')
        self.dir_nmfc = os.path.join(opt.dataroot, self.opt.phase, input_type)
        self.dir_image = os.path.join(opt.dataroot, self.opt.phase, 'images')
        self.dir_nmfc_ref = os.path.join(opt.dataroot, self.opt.phase, input_type + '_fs')
        self.dir_image_ref = os.path.join(opt.dataroot, self.opt.phase, 'images_fs')

        # Collect image paths.
        if not self.opt.no_audio_input:
            self.audio_feats_paths = make_video_dataset(self.dir_audio_feats, opt.target_name, max_seqs_per_identity)
        self.nmfc_paths = make_video_dataset(self.dir_nmfc, opt.target_name, max_seqs_per_identity)
        self.image_paths = make_video_dataset(self.dir_image, opt.target_name, max_seqs_per_identity)
        self.nmfc_ref_paths = make_video_dataset(self.dir_nmfc_ref, opt.target_name)
        self.image_ref_paths = make_video_dataset(self.dir_image_ref, opt.target_name)

        # Make sure paths are okay.
        if not self.opt.no_audio_input:
            assert_valid_pairs(self.audio_feats_paths, self.image_paths)
        assert_valid_pairs(self.nmfc_paths, self.image_paths)
        assert_valid_pairs(self.nmfc_ref_paths, self.image_ref_paths)

        self.dir_landmark = os.path.join(opt.dataroot, self.opt.phase, 'landmarks70')
        self.landmark_paths = make_video_dataset(self.dir_landmark, opt.target_name, max_seqs_per_identity)
        assert_valid_pairs(self.landmark_paths, self.image_paths)

        self.n_of_seqs = len(self.nmfc_paths)
        self.seq_len_max = max([len(A) for A in self.nmfc_paths])
        self.init_frame_index(self.nmfc_paths)
        self.create_identities_dict()

    def __getitem__(self, index):
        # Get sequence paths.
        seq_idx = self.update_frame_index(self.nmfc_paths, index)
        if not self.opt.no_audio_input:
            audio_feats_paths = self.audio_feats_paths[seq_idx]
        nmfc_paths = self.nmfc_paths[seq_idx]
        image_paths = self.image_paths[seq_idx]
        landmark_paths = self.landmark_paths[seq_idx]

        nmfc_len = len(nmfc_paths)

        # Get identity number
        identity_num = self.identities_dict[self.get_identity_name(nmfc_paths[0])]

        # Get reference frames paths.
        if self.opt.isTrain and self.opt.reference_frames_strategy  == 'previous':
            # Do not use the reference image directories.
            # Instead get the paths of previous sequence with the same identity.
            ref_seq_idx = self.get_ref_seq_idx(seq_idx, identity_num)
            ref_nmfc_paths = self.nmfc_paths[ref_seq_idx]
            ref_image_paths = self.image_paths[ref_seq_idx]
        else:
            # Use the reference image directories.
            ref_nmfc_paths = self.nmfc_ref_paths[identity_num]
            ref_image_paths = self.image_ref_paths[identity_num]

        # Get parameters and transforms.
        n_frames_total, start_idx = get_video_parameters(self.opt, self.n_frames_total, nmfc_len, self.frame_idx)
        first_image = Image.open(image_paths[0]).convert('RGB')
        params = get_params(self.opt, first_image.size)
        transform_nmfc = get_transform(self.opt, params, normalize=False) # do not normalize nmfc values
        transform_image = get_transform(self.opt, params)
        transform_image_segmenter = get_transform_segmenter()
        change_seq = False if self.opt.isTrain else self.change_seq

        # Read data
        paths = []
        audio_feats = image = nmfc = 0
        image_segmenter = 0
        mouth_centers = 0
        for i in range(n_frames_total):
            if not self.opt.no_audio_input:
                # Audio features
                audio_feats_path = audio_feats_paths[start_idx + i]
                audio_feats_i = self.get_audio_feats(audio_feats_path)
                audio_feats = audio_feats_i if i == 0 else torch.cat([audio_feats, audio_feats_i], dim=0)
            # NMFCs
            nmfc_path = nmfc_paths[start_idx + i]
            if self.opt.use_landmarks_input:
                nmfc_i = create_landmarks_sketch(nmfc_path, first_image.size, transform_nmfc)
            else:
                nmfc_i = self.get_image(nmfc_path, transform_nmfc)
            nmfc = nmfc_i if i == 0 else torch.cat([nmfc, nmfc_i], dim=0)
            # Images
            image_path = image_paths[start_idx + i]
            image_i = self.get_image(image_path, transform_image)
            image = image_i if i == 0 else torch.cat([image, image_i], dim=0)
            if self.opt.isTrain:
                # Read images using data transform for foreground segmenter network.
                image_segmenter_i = self.get_image(image_path, transform_image_segmenter)
                image_segmenter = image_segmenter_i if i == 0 else torch.cat([image_segmenter, image_segmenter_i], dim=0)
            # Mouth centers
            if self.opt.isTrain and not self.opt.no_mouth_D:
                landmark_path = landmark_paths[start_idx + i]
                mouth_centers_i = self.get_mouth_center(landmark_path)
                mouth_centers = mouth_centers_i if i == 0 else torch.cat([mouth_centers, mouth_centers_i], dim=0)
            # Paths
            paths.append(nmfc_path)

        ref_nmfc = ref_image = 0
        ref_image_segmenter = 0
        if self.opt.isTrain:
            # Sample one frame from the directory with reference frames.
            ref_idx = random.sample(range(0, len(ref_nmfc_paths)), 1)[0]
        else:
            # During test, use the middle frame from the directory with reference frames.
            ref_idx = len(ref_nmfc_paths) // 2

        # Read reference frame and corresponding NMFC.
        ref_nmfc_path = ref_nmfc_paths[ref_idx]
        if self.opt.use_landmarks_input:
            ref_nmfc = create_landmarks_sketch(ref_nmfc_path, first_image_image.size, transform_nmfc)
        else:
            ref_nmfc = self.get_image(ref_nmfc_path, transform_nmfc)
        ref_image_path = ref_image_paths[ref_idx]
        ref_image = self.get_image(ref_image_path, transform_image)
        if self.opt.isTrain:
            # Read reference frame using data transform for foreground segmenter network.
            ref_image_segmenter = self.get_image(ref_image_path, transform_image_segmenter)

        return_list = {'audio_feats': audio_feats, 'nmfc': nmfc, 'image': image, 'image_segmenter': image_segmenter,
                       'ref_image': ref_image, 'ref_nmfc': ref_nmfc, 'mouth_centers': mouth_centers,
                       'paths': paths, 'change_seq': change_seq, 'ref_image_segmenter': ref_image_segmenter}
        return return_list

    def get_ref_seq_idx(self, seq_idx, identity_num):
        # Assuming that each identity has at least 2 sequences in the dataset.
        ref_seq_idx = seq_idx - 1 if seq_idx > 0 and self.sequences_ids[seq_idx-1] == identity_num else seq_idx + 1
        return min(len(self.nmfc_paths)-1, ref_seq_idx)

    def create_identities_dict(self):
        self.identities_dict = {}
        self.sequences_ids = []
        id_cnt = 0
        for path in self.nmfc_paths:
            name = self.get_identity_name(path[0])
            if name not in self.identities_dict:
                self.identities_dict[name] = id_cnt
                id_cnt += 1
            self.sequences_ids.append(self.identities_dict[name])
        self.n_identities = id_cnt

    def get_identity_name(self, A_path):
        identity_name = os.path.basename(os.path.dirname(A_path))
        identity_name = identity_name[:-7] if self.opt.isTrain else identity_name
        return identity_name

    def get_image(self, A_path, transformA, convert_rgb=True):
        A_img = Image.open(A_path)
        if convert_rgb:
            A_img = A_img.convert('RGB')
        A_scaled = transformA(A_img)
        return A_scaled

    def get_audio_feats(self, audio_feats_path):
        audio_feats = np.loadtxt(audio_feats_path)
        assert audio_feats.shape[0] == self.opt.naf, '%s does not have %d audio features' % (audio_feats_path, self.opt.naf)
        return torch.tensor(audio_feats).float()

    def get_mouth_center(self, A_path):
        keypoints = np.loadtxt(A_path, delimiter=' ')
        # Mouth landmarks
        pts = keypoints[48:, :].astype(np.int32) 
        mouth_center = np.median(pts, axis=0)
        mouth_center = mouth_center.astype(np.int32)
        return torch.tensor(mouth_center)

    def __len__(self):
        if self.opt.isTrain:
            return len(self.nmfc_paths)
        else:
            return sum(self.n_frames_in_sequence)

    def name(self):
        return 'video'
import os
import random
from PIL import Image
import torch.utils.data as data
from util.util import *

def make_video_dataset(dir, target_name, max_seqs_per_identity=None):
    images = []
    if dir:
        # Gather sequences
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        fnames = sorted(os.walk(dir))
        for fname in sorted(fnames):
            paths = []
            root = fname[0]
            for f in sorted(fname[2]):
                names = os.path.basename(root).split('_')
                target = names[0]
                if is_image_file(f):
                    if (target_name is None or target_name == target):
                        paths.append(os.path.join(root, f))
            if len(paths) > 0:
                images.append(paths)
        # Find minimun sequence length and reduce all sequences to that.
        # Only for training, in order to be able to form batches.
        if max_seqs_per_identity is not None:
            min_len = float("inf")
            for img_dir in images:
                min_len = min(min_len, len(img_dir))
            for i in range(len(images)):
                images[i] = images[i][:min_len]
            # Keep max_seqs_per_identity for each identity
            trimmed_images = []
            prev_name = None
            temp_seqs = []
            for i in range(len(images)):
                folder = images[i][0].split('/')[-2].split('_')
                name = folder[0]
                if prev_name is None:
                    prev_name = name
                if name == prev_name:
                    temp_seqs.append(i)
                if name != prev_name or i == len(images)-1:
                    if len(temp_seqs) > max_seqs_per_identity:
                        identity_seqs = sorted(temp_seqs)[:max_seqs_per_identity]
                    else:
                        identity_seqs = sorted(temp_seqs)
                    trimmed_images.extend([images[j] for j in identity_seqs])
                    temp_seqs = [i]
                prev_name = name
            images = trimmed_images
    return images

def assert_valid_pairs(A_paths, B_paths):
    assert len(A_paths) > 0 and len(B_paths) > 0, 'No sequences found.'
    assert len(A_paths) == len(B_paths), 'Number of NMFC sequences different than RGB sequences.'
    for i in range(len(A_paths)):
        if abs(len(A_paths[i]) - len(B_paths[i])) <= 3:
            min_len = min(len(A_paths[i]), len(B_paths[i]))
            A_paths[i] = A_paths[i][:min_len]
            B_paths[i] = B_paths[i][:min_len]
        assert len(A_paths[i]) == len(B_paths[i]), 'Number of NMFC frames in sequence different than corresponding RGB frames.'

import torch.utils.data
from dataloader.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    if opt.dataset_mode == 'video':
        from dataloader.video_dataset import videoDataset
        dataset = videoDataset()
    else:
        raise ValueError('Unrecognized dataset')
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, start_idx):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.sampler = MySequentialSampler(self.dataset, start_idx) if opt.serial_batches else None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            sampler=self.sampler,
            num_workers=int(opt.nThreads),
            drop_last=opt.batch_size>1)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

class MySequentialSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, start_idx):
        self.data_source = data_source
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self):
        return len(self.data_source) - self.start_idx


from dataloader.custom_dataset_data_loader import CustomDatasetDataLoader

def CreateDataLoader(opt, start_idx=0):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt, start_idx)
    return data_loader

import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
    else:
        im[yy, xx] = color[0]

def drawCircle(im, x, y, rad, color=(255,0,0)):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-rad, rad):
            for j in range(-rad, rad):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                if np.linalg.norm(np.array([i, j])) < rad:
                    setColor(im, yy, xx, color)

def drawEdge(im, x, y, bw=1, color=(255,255,255)):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

def interpPoints(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:
        if len(x) < 3:
            popt, _ = curve_fit(linear, x, y)
        else:
            popt, _ = curve_fit(func, x, y)
            if abs(popt[0]) > 1:
                return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

def create_landmarks_sketch(A_path, size, transform_scale):
    w, h = size
    landmarks_sketch = np.zeros((h, w, 3), np.int32)

    keypoints = np.loadtxt(A_path, delimiter=' ')
    if keypoints.shape[0] == 70:
        pts = keypoints[:68].astype(np.int32) # Get 68 facial landmarks.
    else:
        raise(RuntimeError('Not enough facial landmarks found in file.'))

    # Draw
    face_list = [
                 [range(0, 17)], # face
                 [range(17, 22)], # left eyebrow
                 [range(22, 27)], # right eyebrow
                 [range(27, 31), range(31, 36)], # nose
                 [[36,37,38,39], [39,40,41,36]], # left eye
                 [[42,43,44,45], [45,46,47,42]], # right eye
                 [range(48, 55), [54,55,56,57,58,59,48]], # mouth exterior
                 [range(60, 65), [64,65,66,67,60]] # mouth interior
                ]
    for edge_list in face_list:
            for edge in edge_list:
                for i in range(0, max(1, len(edge)-1)):
                    sub_edge = edge[i:i+2]
                    x, y = pts[sub_edge, 0], pts[sub_edge, 1]
                    curve_x, curve_y = interpPoints(x, y)
                    drawEdge(landmarks_sketch, curve_x, curve_y)

    landmarks_sketch = transform_scale(Image.fromarray(np.uint8(landmarks_sketch)))
    return landmarks_sketch

import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def init_frame_index(self, A_paths):
        self.seq_idx = 0
        self.frame_idx = -1 if not self.opt.isTrain else 0
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1
        self.n_sequences = len(A_paths)
        self.max_seq_len = max([len(A) for A in A_paths])
        self.n_frames_in_sequence = []
        for path in A_paths:
            self.n_frames_in_sequence.append(len(path) - self.opt.n_frames_G + 1)

    def update_frame_index(self, A_paths, index):
        if self.opt.isTrain:
            seq_idx = index % self.n_sequences
            return seq_idx
        else:
            self.change_seq = self.frame_idx >= self.n_frames_in_sequence[self.seq_idx] - 1
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
            else:
                self.frame_idx += 1
            return self.seq_idx

    def update_sequence_length(self, ratio):
        max_seq_len = self.max_seq_len - self.opt.n_frames_G + 1
        if self.n_frames_total < max_seq_len:
            self.n_frames_total = min(max_seq_len, self.opt.n_frames_total * (2**ratio))
            print('Updated sequence length to %d' % self.n_frames_total)

def get_params(opt, size):
    w, h = size
    if opt.resize:
        new_h = new_w = opt.loadSize
        new_w = int(round(new_w / 4)) * 4
        new_h = int(round(new_h / 4)) * 4
        new_w, new_h = __make_power_2(new_w), __make_power_2(new_h)
    else:
        new_h = h
        new_w = w
    return {'new_size': (new_w, new_h), 'ratio':(new_h / h, new_w / w)}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    ### resize input image
    if opt.resize:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    else:
        transform_list.append(transforms.Lambda(lambda img: __scale(img, params['new_size'], method)))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_segmenter(method=Image.BILINEAR):
    transform_list = []
    ### resize input image
    transform_list.append(transforms.Resize([512, 512], method))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))]
    return transforms.Compose(transform_list)

def __scale(img, size, method=Image.BICUBIC):
    w, h = size
    return img.resize((w, h), method)

def __make_power_2(n, base=32.0):
    return int(round(n / base) * base)

def get_video_parameters(opt, n_frames_total, cur_seq_len, index):
    if opt.isTrain:
        n_frames_total = min(n_frames_total, cur_seq_len - opt.n_frames_G + 1)
        n_frames_total += opt.n_frames_G - 1
        offset_max = max(1, cur_seq_len - n_frames_total + 1)
        start_idx = np.random.randint(offset_max)
    else:
        n_frames_total = opt.n_frames_G
        start_idx = index
    return n_frames_total, start_idx

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from dataloader.data_loader import CreateDataLoader
from models.headGAN import headGANModelG 
from options.test_options import TestOptions
from util.visualizer import Visualizer
import util.util as util

opt = TestOptions().parse(save=False)

visualizer = Visualizer(opt)

modelG = headGANModelG()
modelG.initialize(opt)
modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
modelG.eval()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

print('Generating %d frames' % dataset_size)

save_dir = os.path.join(opt.results_dir, opt.name, opt.which_epoch, opt.phase)

for idx, data in enumerate(dataset):
    _, _, height, width = data['nmfc'].size()
    
    input_A = Variable(data['nmfc']).view(opt.batch_size, -1, opt.input_nc, height, width).cuda(opt.gpu_ids[0])
    image = Variable(data['image']).view(opt.batch_size, -1, opt.output_nc, height, width).cuda(opt.gpu_ids[0])

    if not opt.no_audio_input:
        audio_feats = Variable(data['audio_feats'][:, -opt.naf:]).cuda(opt.gpu_ids[0])
    else:
        audio_feats = None
    ref_input_A = Variable(data['ref_nmfc']).view(opt.batch_size, opt.input_nc, height, width).cuda(opt.gpu_ids[0])
    ref_input_B = Variable(data['ref_image']).view(opt.batch_size, opt.output_nc, height, width).cuda(opt.gpu_ids[0])
    img_path = data['paths']

    print('Processing NMFC image %s' % img_path[-1])
    
    input = input_A.view(opt.batch_size, -1, height, width)
    ref_input = torch.cat([ref_input_A, ref_input_B], dim=1)

    if opt.time_fwd_pass:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    generated_B, warped_B, flow = modelG(input, ref_input, audio_feats)

    if opt.time_fwd_pass:
        end.record()
        # Wait for everything to finish running
        torch.cuda.synchronize()
        print('Forward pass time: %.6f' % start.elapsed_time(end))

    generated = util.tensor2im(generated_B.data[0])
    warped = util.tensor2im(warped_B.data[0])
    flow = util.tensor2flow(flow.data[0])
    nmfc = util.tensor2im(input_A[-1], normalize=False)
    image = util.tensor2im(image[-1])

    visual_list = [('image', image),
                   ('nmfc', nmfc),
                   ('generated', generated),
                   ('warped', warped),
                   ('flow', flow)]

    visuals = OrderedDict(visual_list)
    visualizer.save_images(save_dir, visuals, img_path[-1])

import os
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import dlib
import argparse
import collections
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from util.util import *


def get_image_paths_dict(dir):
    # Returns dict: {name: [path1, path2, ...], ...}
    image_files = {}
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and (('/images/' in root) or ('/images_fs/' in root)):
                path = os.path.join(root, fname)
                seq_name = os.path.basename(root).split('_')[0]
                if seq_name not in image_files:
                    image_files[seq_name] = [path]
                else:
                    image_files[seq_name].append(path)
    # Sort paths for each sequence
    for k, v in image_files.items():
        image_files[k] = sorted(v)
    # Return directory sorted for keys (identity names)
    return collections.OrderedDict(sorted(image_files.items()))


def save_landmarks(image_pths, landmarks):
    # Make dirs
    landmark_pths = [p.replace('/images', '/landmarks70') for p in image_pths]
    out_paths = set(os.path.dirname(landmark_pth) for landmark_pth in landmark_pths)
    for out_path in out_paths:
        mkdir(out_path)
    print('Saving results')
    for landmark, image_pth in tqdm(zip(landmarks, image_pths), total=len(image_pths)):
        landmark_file = os.path.splitext(image_pth.replace('/images', '/landmarks70'))[0] + '.txt'
        np.savetxt(landmark_file, landmark)


def dirs_exist(image_pths):
    nmfc_pths = [p.replace('/images', '/landmarks70') for p in image_pths]
    out_paths = set(os.path.dirname(nmfc_pth) for nmfc_pth in nmfc_pths)
    return all([os.path.exists(out_path) for out_path in out_paths])


def get_mass_center(points, gray):
    im = np.zeros_like(gray)
    cv2.fillPoly(im, [points], 1)
    eyes_image = np.multiply(gray, im)
    inverse_intensity = np.divide(np.ones_like(eyes_image), eyes_image, out=np.zeros_like(eyes_image), where=eyes_image!=0)
    max = np.max(inverse_intensity)
    inverse_intensity = inverse_intensity / max
    coordinates_grid = np.indices((gray.shape[0], gray.shape[1]))
    nom = np.sum(np.multiply(coordinates_grid, np.expand_dims(inverse_intensity, axis=0)), axis=(1,2))
    denom = np.sum(inverse_intensity)
    mass_center = np.flip(nom / denom)
    return mass_center


def add_eye_pupils_landmarks(points, image):
    I = rgb2gray(image)
    left_eye_points = points[36:42,:]
    right_eye_points = points[42:48,:]
    left_pupil = get_mass_center(left_eye_points, I).astype(np.int32)
    right_pupil = get_mass_center(right_eye_points, I).astype(np.int32)
    points[68, :] = left_pupil
    points[69, :] = right_pupil
    return points


def detect_landmarks(img_paths, detector, predictor):
    landmarks = []
    n_fails = 0
    for i in tqdm(range(len(img_paths))):
        img = io.imread(img_paths[i])
        dets = detector(img, 1)
        if len(dets) > 0:
            shape = predictor(img, dets[0])
            points = np.empty([70, 2], dtype=int)
            for b in range(68):
                points[b,0] = shape.part(b).x
                points[b,1] = shape.part(b).y
            points = add_eye_pupils_landmarks(points, img)
            landmarks.append(points)
        else:
            n_fails += 1
            landmarks.append(None)

    # Fix frames where no landmarks were detected with next.
    for i in range(len(landmarks)):
        if landmarks[i] is None:
            landmarks[i] = next((item for item in landmarks[i+1:] if item is not None), landmarks[i-1])
    if landmarks[0] is None:
        print('No landmarks detected.')
    else:
        print('Success in %d/%d of frames,' % (len(img_paths) - n_fails, len(img_paths)))
    return landmarks


def main():
    print('---------- Landmark detector --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='datasets/voxceleb', 
                        help='Path to the dataset directory.')
    args = parser.parse_args()

    predictor_path = 'files/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    images_dict = get_image_paths_dict(args.dataset_path)
    n_image_dirs = len(images_dict)
    print('Number of identities for landmark detection: %d \n' % n_image_dirs)

    # Iterate through the images_dict
    n_completed = 0
    for name, image_pths in images_dict.items():
        n_completed += 1
        if not dirs_exist(image_pths):
            landmarks = detect_landmarks(image_pths, detector, predictor)
            save_landmarks(image_pths, landmarks)
            print('(%d/%d) %s [SUCCESS]' % (n_completed, n_image_dirs, name))
        else:
            print('(%d/%d) %s already processed!' % (n_completed, n_image_dirs, name))

if __name__=='__main__':
    main()

# Copyright (C) 2019 Facesoft Ltd - All Rights Reserved


import cv2
import numpy as np
import mxnet as mx
from skimage import transform as trans

import insightface
from . import img_helper


arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_trans(landmark, input_size, s = 0.8):
  assert landmark.shape==(5,2)
  if s>0.0:
    default_input_size = 224
    S = s*2.0
    D = (2.0-S)/4
    src = arcface_src*S
    src += default_input_size*D
    src[:,1] -= 20
    scale = float(input_size) / default_input_size
    src *= scale

  tform = trans.SimilarityTransform()
  tform.estimate(landmark, src)
  M = tform.params[0:2,:]
  return M

class Handler:
  def __init__(self, prefix, epoch, im_size=128, ctx_id=0):
    if ctx_id>=0:
      ctx = mx.gpu(ctx_id)
    else:
      ctx = mx.cpu()
    if not isinstance(prefix, list):
        prefix = [prefix]
    image_size = (im_size, im_size)
    self.models = []
    for pref in prefix:
        sym, arg_params, aux_params = mx.model.load_checkpoint(pref, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.models.append(model)
    self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
    self.detector.prepare(ctx_id=ctx_id)
    self.aug = 1
    self.aug_value = 5

  def get(self, img):
    out = []
    out_lands = []
    limit = 512
    det_scale = 1.0
    if min(img.shape[0:2])>limit:
      det_scale = float(limit)/min(img.shape[0:2])
    bboxes, landmarks = self.detector.detect(img, scale=det_scale)
    if bboxes.shape[0]==0:
        return out
    for fi in range(bboxes.shape[0]):
      bbox = bboxes[fi]
      landmark = landmarks[fi]
      input_blob = np.zeros( (self.aug, 3)+self.image_size,dtype=np.uint8 )
      M_list = []
      for retry in range(self.aug):
          w, h = (bbox[2]-bbox[0]), (bbox[3]-bbox[1])
          center = (bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2
          rotate = 0
          _scale = 128.0/max(w,h)
          rimg, M = img_helper.transform(img, center, self.image_size[0], _scale, rotate)
          rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
          rimg = np.transpose(rimg, (2,0,1))
          input_blob[retry] = rimg
          M_list.append(M)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      for model in self.models:
          model.forward(db, is_train=False)
      X = None
      for model in self.models:
          x = model.get_outputs()[-1].asnumpy()
          if X is None:
              X = x
          else:
              X += x
      X /= len(self.models)
      if X.shape[1]>=3000:
        X = X.reshape( (X.shape[0], -1, 3) )
      else:
        X = X.reshape( (X.shape[0], -1, 2) )
      X[:,:,0:2] += 1
      X[:,:,0:2] *= (self.image_size[0]//2)
      if X.shape[2]==3:
        X[:,:,2] *= (self.image_size[0]//2)
      for i in range(X.shape[0]):
        M = M_list[i]
        IM = cv2.invertAffineTransform(M)
        x = X[i]
        x = img_helper.trans_points(x, IM)
        X[i] = x
      ret = np.mean(X, axis=0)
      out.append(ret)
      out_lands.append(landmark)
    return out, out_lands

import numpy as np
import cv2
from skimage import transform as trans

def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation)*np.pi/180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0]*scale_ratio
    cy = center[1]*scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1*cx, -1*cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size/2, output_size/2))
    t = t1+t2+t3+t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,M,(output_size, output_size), borderValue = 0.0)
    return cropped, M

def transform_pt(pt, trans):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(trans, new_pt)
    return new_pt[:2]

def gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    assert(sigma>=0)
    if sigma==0:
      img[pt[1], pt[0]] = 1.0
      return True
    assert pt[0]<img.shape[1]
    assert pt[1]<img.shape[0]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return True

def estimate_trans_bbox(face, input_size, s = 2.0):
  w = face[2] - face[0]
  h = face[3] - face[1]
  wc = int( (face[2]+face[0])/2 )
  hc = int( (face[3]+face[1])/2 )
  im_size = max(w, h)
  scale = input_size/(max(w,h)*s)
  M = [ 
        [scale, 0, input_size/2-wc*scale],
        [0, scale, input_size/2-hc*scale],
      ]
  M = np.array(M)
  return M

arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32) 


def estimate_trans(landmark, input_size, s = 0.8):
  assert landmark.shape==(5,2)
  if s>0.0:
    default_input_size = 224
    S = s*2.0
    D = (2.0-S)/4
    src = arcface_src*S
    src += default_input_size*D
    src[:,1] -= 20
    scale = float(input_size) / default_input_size
    src *= scale

  tform = trans.SimilarityTransform()
  tform.estimate(landmark, src)
  M = tform.params[0:2,:]
  return M

def norm_crop(img, landmark, image_size=128, s=0.8):
    M = estimate_trans(landmark, image_size, s)
    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
    return warped, M

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts

def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0]*M[0][0] + M[0][1]*M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2]*scale

    return new_pts

def trans_points(pts, M):
  if pts.shape[1]==2:
    return trans_points2d(pts, M)
  else:
    return trans_points3d(pts, M)


import numpy as np
import math
from math import cos, sin

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])

    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

def angle2matrix_3ddfa(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]

    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  sin(x)],
                 [0, -sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, -sin(y)],
                 [      0, 1,      0],
                 [sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), sin(z), 0],
                 [-sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)


## ------------------------------------------ 1. transform(transform, project, camera).
## ---------- 3d-3d transform. Transform obj in world space
def rotate(vertices, angles):
    ''' rotate vertices.
    X_new = R.dot(X). X: 3 x 1
    Args:
        vertices: [nver, 3].
        rx, ry, rz: degree angles
        rx: pitch. positive for looking down
        ry: yaw. positive for looking left
        rz: roll. positive for tilting head right
    Returns:
        rotated vertices: [nver, 3]
    '''
    R = angle2matrix(angles)
    rotated_vertices = vertices.dot(R.T)

    return rotated_vertices

def similarity_transform(vertices, s, R, t3d):
    ''' similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    Args:(float32)
        vertices: [nver, 3].
        s: [1,]. scale factor.
        R: [3,3]. rotation matrix.
        t3d: [3,]. 3d translation vector.
    Returns:
        transformed vertices: [nver, 3]
    '''
    t3d = np.squeeze(np.array(t3d, dtype = np.float32))
    transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    return transformed_vertices


## -------------- Camera. from world space to camera space
# Ref: https://cs184.eecs.berkeley.edu/lecture/transforms-2
def normalize(x):
    epsilon = 1e-12
    norm = np.sqrt(np.sum(x**2, axis = 0))
    norm = np.maximum(norm, epsilon)
    return x/norm

def lookat_camera(vertices, eye, at = None, up = None):
    """ 'look at' transformation: from world space to camera space
    standard camera space:
        camera located at the origin.
        looking down negative z-axis.
        vertical vector is y-axis.
    Xcam = R(X - C)
    Homo: [[R, -RC], [0, 1]]
    Args:
      vertices: [nver, 3]
      eye: [3,] the XYZ world space position of the camera.
      at: [3,] a position along the center of the camera's gaze.
      up: [3,] up direction
    Returns:
      transformed_vertices: [nver, 3]
    """
    if at is None:
      at = np.array([0, 0, 0], np.float32)
    if up is None:
      up = np.array([0, 1, 0], np.float32)

    eye = np.array(eye).astype(np.float32)
    at = np.array(at).astype(np.float32)
    z_aixs = -normalize(at - eye) # look forward
    x_aixs = normalize(np.cross(up, z_aixs)) # look right
    y_axis = np.cross(z_aixs, x_aixs) # look up

    R = np.stack((x_aixs, y_axis, z_aixs))#, axis = 0) # 3 x 3
    transformed_vertices = vertices - eye # translation
    transformed_vertices = transformed_vertices.dot(R.T) # rotation
    return transformed_vertices

## --------- 3d-2d project. from camera space to image plane
# generally, image plane only keeps x,y channels, here reserve z channel for calculating z-buffer.
def orthographic_project(vertices):
    ''' scaled orthographic projection(just delete z)
        assumes: variations in depth over the object is small relative to the mean distance from camera to object
        x -> x*f/z, y -> x*f/z, z -> f.
        for point i,j. zi~=zj. so just delete z
        ** often used in face
        Homo: P = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    Args:
        vertices: [nver, 3]
    Returns:
        projected_vertices: [nver, 3] if isKeepZ=True. [nver, 2] if isKeepZ=False.
    '''
    return vertices.copy()

def perspective_project(vertices, fovy, aspect_ratio = 1., near = 0.1, far = 1000.):
    ''' perspective projection.
    Args:
        vertices: [nver, 3]
        fovy: vertical angular field of view. degree.
        aspect_ratio : width / height of field of view
        near : depth of near clipping plane
        far : depth of far clipping plane
    Returns:
        projected_vertices: [nver, 3]
    '''
    fovy = np.deg2rad(fovy)
    top = near*np.tan(fovy)
    bottom = -top
    right = top*aspect_ratio
    left = -right

    #-- homo
    P = np.array([[near/right, 0, 0, 0],
                 [0, near/top, 0, 0],
                 [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                 [0, 0, -1, 0]])
    vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1)))) # [nver, 4]
    projected_vertices = vertices_homo.dot(P.T)
    projected_vertices = projected_vertices/projected_vertices[:,3:]
    projected_vertices = projected_vertices[:,:3]
    projected_vertices[:,2] = -projected_vertices[:,2]

    #-- non homo. only fovy
    # projected_vertices = vertices.copy()
    # projected_vertices[:,0] = -(near/right)*vertices[:,0]/vertices[:,2]
    # projected_vertices[:,1] = -(near/top)*vertices[:,1]/vertices[:,2]
    return projected_vertices


def to_image(vertices, h, w, is_perspective = False):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis.
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        projected_vertices: [nver, 3]
    '''
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:,0] = image_vertices[:,0]*w/2
        image_vertices[:,1] = image_vertices[:,1]*h/2
    # move to center of image
    image_vertices[:,0] = image_vertices[:,0] + w/2
    image_vertices[:,1] = image_vertices[:,1] + h/2
    # flip vertices along y-axis.
    image_vertices[:,1] = h - image_vertices[:,1] - 1
    return image_vertices


#### -------------------------------------------2. estimate transform matrix from correspondences.
def estimate_affine_matrix_3d23d(X, Y):
    ''' Using least-squares solution
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    '''
    X_homo = np.hstack((X, np.ones([X.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(X_homo, Y, rcond=None)[0].T # Affine matrix. 3 x 4
    return P

def estimate_affine_matrix_3d22d(X, x):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. in MVGCV
        Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
        x_homo = X_homo.dot(P_Affine)
    Args:
        X: [n, 3]. corresponding 3d points(fixed)
        x: [n, 2]. n>=4. 2d points(moving). x = PX
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    '''
    X = X.T; x = x.T
    assert(x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert(n >= 4)

    #--- 1. normalization
    # 2d points
    mean = np.mean(x, 1) # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x**2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3,3), dtype = np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean*scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1) # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3,:] - X
    average_norm = np.mean(np.sqrt(np.sum(X**2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4,4), dtype = np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean*scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n*2, 8), dtype = np.float32);
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])

    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype = np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))
    return P_Affine

def P2sRt(P):
    ''' decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t

#Ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert(isRotationMatrix)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz
import numpy as np
import os
import sys
import shlex
import subprocess
import wget
import wave
from deepspeech import Model

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)

def binary_search_interval(lst, left, right, key):
    idx = (left + right) // 2
    if key >= lst[idx][1] and key <= lst[idx][2]:
        return lst[idx][0]
    elif key < lst[idx][1]:
        return binary_search_interval(lst, left, idx, key)
    else:
        return binary_search_interval(lst, idx+1, right, key)

def create_window(lst, win):
    # Pad first and last item.
    front = lst[:1] * (win//2)
    back = lst[-1:] * (win//2-1) if (win % 2 == 0) else lst[-1:] * (win//2)
    front.extend(lst)
    front.extend(back)
    window_lst = []
    for i in range(len(lst)):
        w = front[i:i+win]
        window_lst.append(w)
    return window_lst

def token_window_to_onehot(lst):
    data = []
    for token in lst:
        order = ord(token)
        # 27 labels: 'a' -> 0, ..., 'z' --> 25, others --> 26
        if order > 96 and order < 123:
            # if alphabet char
            int_token = ord(token) - 97
        else:
            int_token = 26
        data.append(int_token)
    data = np.array(data)
    shape = (data.size, 27)
    onehot = np.zeros(shape)
    rows = np.arange(data.size)
    onehot[rows, data] = 1

    # Concatenate one-hot embeddings for window.
    onehot_vector = onehot.flatten()
    return onehot_vector


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)

def open_audio(audio_path, desired_sample_rate):
    fin = wave.open(audio_path, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
        fs_new, audio = convert_samplerate(audio_path, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()
    return audio, audio_length

def get_logits(audio_path, fps, window):
    model_path='files/deepspeech-0.8.1-models.pbmm'
    scorer_path='files/deepspeech-0.8.1-models.scorer'

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    if not os.path.isfile(model_path):
        print('DeepSpeech model file not found. Downloading...\n')
        wget.download('https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-models.pbmm',
                      model_path)
    if not os.path.isfile(scorer_path):
        print('DeepSpeech scorer file not found. Downloading...\n')
        wget.download('https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-models.scorer',
                      scorer_path)

    ds = Model(model_path)

    desired_sample_rate = ds.sampleRate()
    ds.enableExternalScorer(scorer_path)
    audio, audio_length = open_audio(audio_path, desired_sample_rate)

    # Get tokens from deepspeech network
    transcripts = ds.sttWithMetadata(audio, 1).transcripts[0]
    print('\nText: ' + metadata_to_string(transcripts))
    tokens = transcripts.tokens
    if len(tokens) == 0:
        # If no speech detected, create a single frame with ' ' token.
        data = np.array([26] * window)
        shape = (data.size, 27)
        onehot = np.zeros(shape)
        rows = np.arange(data.size)
        onehot[rows, data] = 1

        # Concatenate one-hot embeddings for window.
        onehot_vector = onehot.flatten()
        return [onehot_vector]

    # Create results list: [(token, start_time, end_time), ...]
    results = [(' ', 0.0, tokens[0].start_time)]
    for i in range(len(tokens)):
        token = tokens[i]
        next_token = tokens[i+1] if i < len(tokens) - 1 else None
        text = token.text
        start_time = token.start_time
        end_time = next_token.start_time if next_token else audio_length
        results.append((text, start_time, end_time))

    # Get per frame token - align each frame with a token.
    frame_tokens = []
    n_frames = round(fps * audio_length)
    offset_n_frames = 5
    for n in range(n_frames):
        t = (n + offset_n_frames) * audio_length / n_frames
        if t < audio_length:
            token = binary_search_interval(results, 0, len(results), t)
        else:
            token = results[-1][0]
        frame_tokens.append(token)

    # Create windows of tokens around each frame.
    window_tokens = create_window(frame_tokens, window)

    # One-hot encoding of tokens.
    deepspeech_feats = []
    for window_token in window_tokens:
        deepspeech_feats.append(token_window_to_onehot(window_token))
    return deepspeech_feats

# Notice: The following code has been taken from https://github.com/tyiannak/pyAudioAnalysis and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.

import os
import glob
import aifc
import numpy
import shutil
import numpy as np
from pydub import AudioSegment


def read_audio_file(path):
    """
    This function returns a numpy array that stores the audio samples of a
    specified WAV of AIFF file
    """

    sampling_rate = 0
    signal = np.array([])
    extension = os.path.splitext(path)[1].lower()
    if extension in ['.aif', '.aiff']:
        sampling_rate, signal = read_aif(path)
    elif extension in [".mp3", ".wav", ".au", ".ogg"]:
        sampling_rate, signal = read_audio_generic(path)
    else:
        print("Error: unknown file type {extension}")

    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal


def read_aif(path):
    """
    Read audio file with .aif extension
    """
    sampling_rate = -1
    signal = np.array([])
    try:
        with aifc.open(path, 'r') as s:
            nframes = s.getnframes()
            strsig = s.readframes(nframes)
            signal = numpy.fromstring(strsig, numpy.short).byteswap()
            sampling_rate = s.getframerate()
    except:
        print("Error: read aif file. (DECODING FAILED)")
    return sampling_rate, signal


def read_audio_generic(path):
    """
    Function to read audio files with the following extensions
    [".mp3", ".wav", ".au", ".ogg"]
    """
    sampling_rate = -1
    signal = np.array([])
    try:
        audiofile = AudioSegment.from_file(path)
        data = np.array([])
        if audiofile.sample_width == 2:
            data = numpy.fromstring(audiofile._data, numpy.int16)
        elif audiofile.sample_width == 4:
            data = numpy.fromstring(audiofile._data, numpy.int32)

        if data.size > 0:
            sampling_rate = audiofile.frame_rate
            temp_signal = []
            for chn in list(range(audiofile.channels)):
                temp_signal.append(data[chn::audiofile.channels])
            signal = numpy.array(temp_signal).T
    except:
        print("Error: file not found or other I/O error. (DECODING FAILED)")
    return sampling_rate, signal


def stereo_to_mono(signal):
    """
    This function converts the input signal
    (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal

# Notice: The following code has been taken from https://github.com/tyiannak/pyAudioAnalysis and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.

import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from helpers.audio_features.audioBasicIO import read_audio_file, stereo_to_mono
from helpers.audio_features.ShortTermFeatures import feature_extraction

def mid_feature_extraction(signal, sampling_rate, mid_window, mid_step, short_window, short_step):
    short_features, short_feature_names = feature_extraction(signal, sampling_rate, short_window, short_step)
    n_stats = 2
    n_feats = len(short_features)
    mid_window_ratio = int(round(mid_window / short_step))
    mt_step_ratio = int(round(mid_step / short_step))

    mid_features, mid_feature_names = [], []
    for i in range(n_stats * n_feats):
        mid_features.append([])
        mid_feature_names.append("")

    for i in range(n_feats):
        cur_position = 0
        num_short_features = len(short_features[i])
        mid_feature_names[i] = short_feature_names[i] + "_" + "mean"
        mid_feature_names[i + n_feats] = short_feature_names[i] + "_" + "std"

        while cur_position < num_short_features:
            end = cur_position + mid_window_ratio
            if end > num_short_features:
                end = num_short_features
            cur_st_feats = short_features[i][cur_position:end]

            mid_features[i].append(np.mean(cur_st_feats))
            mid_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    return np.array(mid_features), short_features, mid_feature_names

def get_mid_features(file_path, mid_window, mid_step, short_window, short_step):
    sampling_rate, signal = read_audio_file(file_path)
    signal = stereo_to_mono(signal)
    mid_features, _, _ = \
        mid_feature_extraction(signal, sampling_rate,
                               round(sampling_rate * mid_window),
                               round(sampling_rate * mid_step),
                               round(sampling_rate * short_window),
                               round(sampling_rate * short_step))
    return list(mid_features.T)

# Notice: The following code has been taken from https://github.com/tyiannak/pyAudioAnalysis and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.

import math
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.fftpack.realtransforms import dct
from tqdm import tqdm

eps = 0.00000001

def zero_crossing_rate(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)


def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))


def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy


""" Frequency-domain audio features """


def spectral_centroid_spread(fft_magnitude, sampling_rate):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(fft_magnitude) + 1)) * \
          (sampling_rate / (2.0 * len(fft_magnitude)))

    Xt = fft_magnitude.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    centroid = (NUM / DEN)

    # Spread:
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / DEN)

    # Normalize:
    centroid = centroid / (sampling_rate / 2.0)
    spread = spread / (sampling_rate / 2.0)

    return centroid, spread


def spectral_entropy(signal, n_short_blocks=10):
    """Computes the spectral entropy"""
    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))

    return entropy


def spectral_flux(fft_magnitude, previous_fft_magnitude):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        fft_magnitude:            the abs(fft) of the current frame
        previous_fft_magnitude:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum(
        (fft_magnitude / fft_sum - previous_fft_magnitude /
         previous_fft_sum) ** 2)

    return sp_flux


def spectral_rolloff(signal, c):
    """Computes spectral roll-off"""
    energy = np.sum(signal ** 2)
    fft_length = len(signal)
    threshold = c * energy
    # Ffind the spectral rolloff as the frequency position
    # where the respective spectral energy is equal to c*totalEnergy
    cumulative_sum = np.cumsum(signal ** 2) + eps
    a = np.nonzero(cumulative_sum > threshold)[0]
    if len(a) > 0:
        sp_rolloff = np.float64(a[0]) / (float(fft_length))
    else:
        sp_rolloff = 0.0
    return sp_rolloff


def harmonic(frame, sampling_rate):
    """
    Computes harmonic ratio and pitch
    """
    m = np.round(0.016 * sampling_rate) - 1
    r = np.correlate(frame, frame, mode='full')

    g = r[len(frame) - 1]
    r = r[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = np.nonzero(np.diff(np.sign(r)))

    if len(a) == 0:
        m0 = len(r) - 1
    else:
        m0 = a[0]
    if m > len(r):
        m = len(r) - 1

    gamma = np.zeros((m), dtype=np.float64)
    cumulative_sum = np.cumsum(frame ** 2)
    gamma[m0:m] = r[m0:m] / (np.sqrt((g * cumulative_sum[m:m0:-1])) + eps)

    zcr = zero_crossing_rate(gamma)

    if zcr > 0.15:
        hr = 0.0
        f0 = 0.0
    else:
        if len(gamma) == 0:
            hr = 1.0
            blag = 0.0
            gamma = np.zeros((m), dtype=np.float64)
        else:
            hr = np.max(gamma)
            blag = np.argmax(gamma)

        # Get fundamental frequency:
        f0 = sampling_rate / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if hr < 0.1:
            f0 = 0.0

    return hr, f0


def mfcc_filter_banks(sampling_rate, num_fft, lowfreq=133.33, linc=200 / 3,
                      logsc=1.0711703, num_lin_filt=13, num_log_filt=27):
    """
    Computes the triangular filterbank for MFCC computation
    (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    if sampling_rate < 8000:
        nlogfil = 5

    # Total number of filters
    num_filt_total = num_lin_filt + num_log_filt

    # Compute frequency points of the triangle:
    frequencies = np.zeros(num_filt_total + 2)
    frequencies[:num_lin_filt] = lowfreq + np.arange(num_lin_filt) * linc
    frequencies[num_lin_filt:] = frequencies[num_lin_filt - 1] * logsc ** \
                                 np.arange(1, num_log_filt + 3)
    heights = 2. / (frequencies[2:] - frequencies[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((num_filt_total, num_fft))
    nfreqs = np.arange(num_fft) / (1. * num_fft) * sampling_rate

    for i in range(num_filt_total):
        low_freqs = frequencies[i]
        cent_freqs = frequencies[i + 1]
        high_freqs = frequencies[i + 2]

        lid = np.arange(np.floor(low_freqs * num_fft / sampling_rate) + 1,
                        np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        dtype=np.int)
        lslope = heights[i] / (cent_freqs - low_freqs)
        rid = np.arange(np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        np.floor(high_freqs * num_fft / sampling_rate) + 1,
                        dtype=np.int)
        rslope = heights[i] / (high_freqs - cent_freqs)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_freqs)
        fbank[i][rid] = rslope * (high_freqs - nfreqs[rid])

    return fbank, frequencies


def mfcc(fft_magnitude, fbank, num_mfcc_feats):
    """
    Computes the MFCCs of a frame, given the fft mag
    ARGUMENTS:
        fft_magnitude:  fft magnitude abs(FFT)
        fbank:          filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:           MFCCs (13 element vector)
    Note:    MFCC calculation is, in general, taken from the
             scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more
         compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = np.log10(np.dot(fft_magnitude, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:num_mfcc_feats]
    return ceps


def chroma_features_init(num_fft, sampling_rate):
    """
    This function initializes the chroma matrices used in the calculation
    of the chroma features
    """
    freqs = np.array([((f + 1) * sampling_rate) /
                      (2 * num_fft) for f in range(num_fft)])
    cp = 27.50
    num_chroma = np.round(12.0 * np.log2(freqs / cp)).astype(int)

    num_freqs_per_chroma = np.zeros((num_chroma.shape[0],))

    unique_chroma = np.unique(num_chroma)
    for u in unique_chroma:
        idx = np.nonzero(num_chroma == u)
        num_freqs_per_chroma[idx] = idx[0].shape

    return num_chroma, num_freqs_per_chroma


def chroma_features(signal, sampling_rate, num_fft):
    num_chroma, num_freqs_per_chroma = \
        chroma_features_init(num_fft, sampling_rate)
    chroma_names = ['A', 'A#', 'B', 'C', 'C#', 'D',
                    'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = signal ** 2
    if num_chroma.max() < num_chroma.shape[0]:
        C = np.zeros((num_chroma.shape[0],))
        C[num_chroma] = spec
        C /= num_freqs_per_chroma[num_chroma]
    else:
        I = np.nonzero(num_chroma > num_chroma.shape[0])[0][0]
        C = np.zeros((num_chroma.shape[0],))
        C[num_chroma[0:I - 1]] = spec
        C /= num_freqs_per_chroma
    final_matrix = np.zeros((12, 1))
    newD = int(np.ceil(C.shape[0] / 12.0) * 12)
    C2 = np.zeros((newD,))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0] / 12), 12)
    final_matrix = np.matrix(np.sum(C2, axis=0)).T
    final_matrix /= spec.sum()

    return chroma_names, final_matrix


def chromagram(signal, sampling_rate, window, step, plot=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a np array (num_fft x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        sampling_rate:          the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:        the short-term window step (in samples)
        plot:        flag, 1 if results are to be ploted
    RETURNS:
    """
    window = int(window)
    step = int(step)
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    dc_offset = signal.mean()
    maximum = (np.abs(signal)).max()
    signal = (signal - dc_offset) / (maximum - dc_offset)

    num_samples = len(signal)  # total number of signals
    cur_position = 0
    count_fr = 0
    num_fft = int(window / 2)
    chromogram = np.array([], dtype=np.float64)

    while cur_position + window - 1 < num_samples:
        count_fr += 1
        x = signal[cur_position:cur_position + window]
        cur_position = cur_position + step
        X = abs(fft(x))
        X = X[0:num_fft]
        X = X / len(X)
        chroma_names, chroma_feature_matrix = chroma_features(X, sampling_rate,
                                                              num_fft)
        chroma_feature_matrix = chroma_feature_matrix[:, 0]
        if count_fr == 1:
            chromogram = chroma_feature_matrix.T
        else:
            chromogram = np.vstack((chromogram, chroma_feature_matrix.T))
    freq_axis = chroma_names
    time_axis = [(t * step) / sampling_rate
                 for t in range(chromogram.shape[0])]

    if plot:
        fig, ax = plt.subplots()
        chromogram_plot = chromogram.transpose()[::-1, :]
        ratio = int(chromogram_plot.shape[1] / (3 * chromogram_plot.shape[0]))
        if ratio < 1:
            ratio = 1
        chromogram_plot = np.repeat(chromogram_plot, ratio, axis=0)
        imgplot = plt.imshow(chromogram_plot)

        ax.set_yticks(range(int(ratio / 2), len(freq_axis) * ratio, ratio))
        ax.set_yticklabels(freq_axis[::-1])
        t_step = int(count_fr / 3)
        time_ticks = range(0, count_fr, t_step)
        time_ticks_labels = ['%.2f' % (float(t * step) / sampling_rate)
                             for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticks_labels)
        ax.set_xlabel('time (secs)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return chromogram, time_axis, freq_axis


def spectrogram(signal, sampling_rate, window, step, plot=False,
                show_progress=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a np array (num_fft x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        sampling_rate:          the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:        the short-term window step (in samples)
        plot:        flag, 1 if results are to be ploted
        show_progress flag for showing progress using tqdm
    RETURNS:
    """
    window = int(window)
    step = int(step)
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    dc_offset = signal.mean()
    maximum = (np.abs(signal)).max()
    signal = (signal - dc_offset) / (maximum - dc_offset)

    num_samples = len(signal)  # total number of signals
    count_fr = 0
    num_fft = int(window / 2)
    specgram = np.array([], dtype=np.float64)

    for cur_p in tqdm(range(window, num_samples - step, step),
                      disable=not show_progress):
        count_fr += 1
        x = signal[cur_p:cur_p + window]
        X = abs(fft(x))
        X = X[0:num_fft]
        X = X / len(X)

        if count_fr == 1:
            specgram = X ** 2
        else:
            specgram = np.vstack((specgram, X))

    freq_axis = [float((f + 1) * sampling_rate) / (2 * num_fft)
                 for f in range(specgram.shape[1])]
    time_axis = [float(t * step) / sampling_rate
                 for t in range(specgram.shape[0])]

    if plot:
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        fstep = int(num_fft / 5.0)
        frequency_ticks = range(0, int(num_fft) + fstep, fstep)
        frequency_tick_labels = \
            [str(sampling_rate / 2 -
                 int((f * sampling_rate) / (2 * num_fft)))
             for f in frequency_ticks]
        ax.set_yticks(frequency_ticks)
        ax.set_yticklabels(frequency_tick_labels)
        t_step = int(count_fr / 3)
        time_ticks = range(0, count_fr, t_step)
        time_ticks_labels = \
            ['%.2f' % (float(t * step) / sampling_rate) for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticks_labels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return specgram, time_axis, freq_axis


def speed_feature(signal, sampling_rate, window, step):
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    dc_offset = signal.mean()
    maximum = (np.abs(signal)).max()
    signal = (signal - dc_offset) / maximum

    num_samples = len(signal)  # total number of signals
    cur_p = 0
    count_fr = 0

    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    nlinfil = 13
    nlogfil = 27
    n_mfcc_feats = 13
    nfil = nlinfil + nlogfil
    num_fft = window / 2
    if sampling_rate < 8000:
        nlogfil = 5
        nfil = nlinfil + nlogfil
        num_fft = window / 2

    # compute filter banks for mfcc:
    fbank, freqs = mfcc_filter_banks(sampling_rate, num_fft, lowfreq, linsc,
                                       logsc, nlinfil, nlogfil)

    n_time_spectral_feats = 8
    n_harmonic_feats = 1
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats
    st_features = []

    while cur_p + window - 1 < num_samples:
        count_fr += 1
        x = signal[cur_p:cur_p + window]
        cur_p = cur_p + step
        fft_magnitude = abs(fft(x))
        fft_magnitude = fft_magnitude[0:num_fft]
        fft_magnitude = fft_magnitude / len(fft_magnitude)
        Ex = 0.0
        El = 0.0
        fft_magnitude[0:4] = 0
        st_features.append(harmonic(x, sampling_rate))

    return np.array(st_features)


def phormants(x, sampling_rate):
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Get LPC.
    ncoeff = 2 + sampling_rate / 1000
    A, e, k = lpc(x1, ncoeff)
    # A, e, k = lpc(x1, 8)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    frqs = sorted(angz * (sampling_rate / (2 * math.pi)))

    return frqs


""" Windowing and feature extraction """


def feature_extraction(signal, sampling_rate, window, step, deltas=True):
    """
    This function implements the shor-term windowing process.
    For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a np matrix.
    ARGUMENTS
        signal:         the input signal samples
        sampling_rate:  the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:           the short-term window step (in samples)
        deltas:         (opt) True/False if delta features are to be
                        computed
    RETURNS
        features (numpy.ndarray):        contains features
                                         (n_feats x numOfShortTermWindows)
        feature_names (numpy.ndarray):   contains feature names
                                         (n_feats x numOfShortTermWindows)
    """

    window = int(window)
    step = int(step)

    # signal normalization
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    dc_offset = signal.mean()
    signal_max = (np.abs(signal)).max()
    signal = (signal - dc_offset) / (signal_max + 0.0000000001)

    number_of_samples = len(signal)  # total number of samples
    current_position = 0
    count_fr = 0
    num_fft = int(window / 2)
    # compute the triangular filter banks used in the mfcc calculation
    fbank, freqs = mfcc_filter_banks(sampling_rate, num_fft)

    n_time_spectral_feats = 8
    n_harmonic_feats = 0
    n_mfcc_feats = 13
    n_chroma_feats = 13
    #n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats + \
    #                n_chroma_feats
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats

    # define list of feature names
    feature_names = ["zcr", "energy", "energy_entropy"]
    feature_names += ["spectral_centroid", "spectral_spread"]
    feature_names.append("spectral_entropy")
    feature_names.append("spectral_flux")
    feature_names.append("spectral_rolloff")
    feature_names += ["mfcc_{0:d}".format(mfcc_i)
                      for mfcc_i in range(1, n_mfcc_feats + 1)]
    feature_names += ["chroma_{0:d}".format(chroma_i)
                      for chroma_i in range(1, n_chroma_feats)]
    feature_names.append("chroma_std")

    # add names for delta features:
    if deltas:
        feature_names_2 = feature_names + ["delta " + f for f in feature_names]
        feature_names = feature_names_2

    features = []
    # for each short-term window to end of signal
    while current_position + window - 1 < number_of_samples:
        count_fr += 1
        # get current window
        x = signal[current_position:current_position + window]

        # update window position
        current_position = current_position + step

        # get fft magnitude
        fft_magnitude = abs(fft(x))

        # normalize fft
        fft_magnitude = fft_magnitude[0:num_fft]
        fft_magnitude = fft_magnitude / len(fft_magnitude)

        # keep previous fft mag (used in spectral flux)
        if count_fr == 1:
            fft_magnitude_previous = fft_magnitude.copy()
        feature_vector = np.zeros((n_total_feats, 1))

        # zero crossing rate
        feature_vector[0] = zero_crossing_rate(x)

        # short-term energy
        feature_vector[1] = energy(x)

        # short-term entropy of energy
        feature_vector[2] = energy_entropy(x)

        # sp centroid/spread
        [feature_vector[3], feature_vector[4]] = \
            spectral_centroid_spread(fft_magnitude,
                                     sampling_rate)

        # spectral entropy
        feature_vector[5] = \
            spectral_entropy(fft_magnitude)

        # spectral flux
        feature_vector[6] = \
            spectral_flux(fft_magnitude,
                          fft_magnitude_previous)

        # spectral rolloff
        feature_vector[7] = \
            spectral_rolloff(fft_magnitude, 0.90)

        # MFCCs
        mffc_feats_end = n_time_spectral_feats + n_mfcc_feats
        feature_vector[n_time_spectral_feats:mffc_feats_end, 0] = \
            mfcc(fft_magnitude, fbank, n_mfcc_feats).copy()

        if not deltas:
            features.append(feature_vector)
        else:
            # delta features
            if count_fr > 1:
                delta = feature_vector - feature_vector_prev
                feature_vector_2 = np.concatenate((feature_vector, delta))
            else:
                feature_vector_2 = np.concatenate((feature_vector,
                                                   np.zeros(feature_vector.
                                                            shape)))
            feature_vector_prev = feature_vector
            features.append(feature_vector_2)

        fft_magnitude_previous = fft_magnitude.copy()

    features = np.concatenate(features, 1)
    return features, feature_names

# Try to import bindings
try:
    from . import avatars_bindings
except:
    import warnings
    warnings.warn("Failed to import avatars_bindings module", ImportWarning)
    mesh = None
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FlatbuffersSchema

import flatbuffers

class SerializedMorphableModel(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSerializedMorphableModel(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SerializedMorphableModel()
        x.Init(buf, n + offset)
        return x

    # SerializedMorphableModel
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SerializedMorphableModel
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    # SerializedMorphableModel
    def MeanMesh(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .SerializedTriMesh import SerializedTriMesh
            obj = SerializedTriMesh()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SerializedMorphableModel
    def Components(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .SerializedComponent import SerializedComponent
            obj = SerializedComponent()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SerializedMorphableModel
    def ComponentsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def SerializedMorphableModelStart(builder): builder.StartObject(3)
def SerializedMorphableModelAddVersion(builder, Version): builder.PrependUint16Slot(0, Version, 0)
def SerializedMorphableModelAddMeanMesh(builder, MeanMesh): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(MeanMesh), 0)
def SerializedMorphableModelAddComponents(builder, Components): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(Components), 0)
def SerializedMorphableModelStartComponentsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SerializedMorphableModelEnd(builder): return builder.EndObject()


# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FlatbuffersSchema

import flatbuffers

class SerializedKeyframesTimeline(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSerializedKeyframesTimeline(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SerializedKeyframesTimeline()
        x.Init(buf, n + offset)
        return x

    # SerializedKeyframesTimeline
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SerializedKeyframesTimeline
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    # SerializedKeyframesTimeline
    def KeyframesData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SerializedKeyframesTimeline
    def KeyframesDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # SerializedKeyframesTimeline
    def KeyframesDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SerializedKeyframesTimeline
    def SamplesPerFrame(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # SerializedKeyframesTimeline
    def TimeBetweenFramesSecs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

def SerializedKeyframesTimelineStart(builder): builder.StartObject(4)
def SerializedKeyframesTimelineAddVersion(builder, Version): builder.PrependUint16Slot(0, Version, 0)
def SerializedKeyframesTimelineAddKeyframesData(builder, KeyframesData): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(KeyframesData), 0)
def SerializedKeyframesTimelineStartKeyframesDataVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SerializedKeyframesTimelineAddSamplesPerFrame(builder, SamplesPerFrame): builder.PrependUint32Slot(2, SamplesPerFrame, 0)
def SerializedKeyframesTimelineAddTimeBetweenFramesSecs(builder, TimeBetweenFramesSecs): builder.PrependFloat32Slot(3, TimeBetweenFramesSecs, 0.0)
def SerializedKeyframesTimelineEnd(builder): return builder.EndObject()

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FlatbuffersSchema

import flatbuffers

class SerializedTriMesh(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSerializedTriMesh(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SerializedTriMesh()
        x.Init(buf, n + offset)
        return x

    # SerializedTriMesh
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SerializedTriMesh
    def Points(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SerializedTriMesh
    def PointsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # SerializedTriMesh
    def PointsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SerializedTriMesh
    def TriIndices(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SerializedTriMesh
    def TriIndicesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint32Flags, o)
        return 0

    # SerializedTriMesh
    def TriIndicesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SerializedTriMesh
    def UV(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SerializedTriMesh
    def UVAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # SerializedTriMesh
    def UVLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SerializedTriMesh
    def SubmeshIndexOffset(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SerializedTriMesh
    def SubmeshIndexOffsetAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint32Flags, o)
        return 0

    # SerializedTriMesh
    def SubmeshIndexOffsetLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def SerializedTriMeshStart(builder): builder.StartObject(4)
def SerializedTriMeshAddPoints(builder, Points): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(Points), 0)
def SerializedTriMeshStartPointsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SerializedTriMeshAddTriIndices(builder, TriIndices): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(TriIndices), 0)
def SerializedTriMeshStartTriIndicesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SerializedTriMeshAddUV(builder, UV): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(UV), 0)
def SerializedTriMeshStartUVVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SerializedTriMeshAddSubmeshIndexOffset(builder, SubmeshIndexOffset): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(SubmeshIndexOffset), 0)
def SerializedTriMeshStartSubmeshIndexOffsetVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SerializedTriMeshEnd(builder): return builder.EndObject()

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FlatbuffersSchema

import flatbuffers

class SerializedComponent(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSerializedComponent(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SerializedComponent()
        x.Init(buf, n + offset)
        return x

    # SerializedComponent
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SerializedComponent
    def Points(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SerializedComponent
    def PointsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # SerializedComponent
    def PointsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SerializedComponent
    def Scale(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

def SerializedComponentStart(builder): builder.StartObject(2)
def SerializedComponentAddPoints(builder, Points): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(Points), 0)
def SerializedComponentStartPointsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SerializedComponentAddScale(builder, Scale): builder.PrependFloat32Slot(1, Scale, 0.0)
def SerializedComponentEnd(builder): return builder.EndObject()

import pickle
import json
import numpy        
import flatbuffers
from flatbuffers.number_types import UOffsetTFlags

# types generated from schema 
from .FlatbuffersSchema import SerializedTriMesh
from .FlatbuffersSchema import SerializedMorphableModel
from .FlatbuffersSchema import SerializedComponent



def get_version():
    """Returns single integer number with the serialization version"""
    return 2

def deserialize_binary_to_morphable_model(data_blob):
    """De-serialize a binary blob as defined by flatbuffers

    data_blob -- Binary blob created by @serialize_morphable_model_to_binary
    Returns a dict with key-value pairs as described for the input arguments in serialize_morphable_model_to_binary
    """

    buf = bytearray(data_blob)
    mmodel = SerializedMorphableModel.SerializedMorphableModel.GetRootAsSerializedMorphableModel(buf, 0)

    mean_points = mmodel.MeanMesh().PointsAsNumpy()
    mean_indices = mmodel.MeanMesh().TriIndicesAsNumpy()
    weights = []
    components = []
    for i in range(mmodel.ComponentsLength()):
        c = mmodel.Components(i)
        weights.append(c.Scale())
        components.append(c.PointsAsNumpy())

    if len(components)> 0:
        components = numpy.asarray(components, dtype=numpy.float32)
    if len(weights) > 0:
        weights = numpy.asarray(weights, dtype=numpy.float32).ravel()

    submesh_offsets = []
    if mmodel.MeanMesh().SubmeshIndexOffsetLength() > 0:
        submesh_offsets = mmodel.MeanMesh().SubmeshIndexOffsetAsNumpy()

    dict = {'mean_points':mean_points, 'mean_indices':mean_indices, 'weights':weights, 'components':components, 'submesh_offsets':submesh_offsets}
    if mmodel.MeanMesh().UVLength() > 0:
        dict['mean_uvs'] = mmodel.MeanMesh().UVAsNumpy()

    return dict


def serialize_morphable_model_to_binary(mean_points, mean_indices, components, weights, mean_uvs=None, submesh_offsets=None):
    """Serialize the Morphable Model defined by the input vectors to the Flatbuffers Schema used in the package

    mean_points -- 1xN numpy array of floats with the 3D coordinates of the mean mesh points where N = 3 * num_points
    mean_indices -- (optional) flat numpy array of ints with the indices of the mean mesh triangles 
    referencing the mean_points 
    mean_uvs -- (optional) flat numpy array of floats with the UV coords for each mean mesh vertex 
    components -- MxN numpy array of floats with the 3D coordinates of each component of the Morphable Model where M is 
    the total number of components
    weights -- 1xM numpy array of floats with the weight factor for each component  

    Note that the first dimension of components need to be the same as the weights while the 
    second dimension needs to be the same as the mean_points length.
    Returns the serialized binary blob. 
    """

    if mean_points is not None:
        if not isinstance(mean_points, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if mean_indices is not None:
        if not isinstance(mean_indices, numpy.ndarray):
           raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if mean_uvs is not None:
        if not isinstance(mean_uvs, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if submesh_offsets is not None:
        if not isinstance(submesh_offsets, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if components is not None:
        if not isinstance(components, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")
    if weights is not None:
        if not isinstance(weights, numpy.ndarray):
            raise TypeError("non-numpy-ndarray passed to writeMorphableModel")

    if components is not None:
        if weights is not None:
            if len(weights) != components.shape[0]:
                raise ValueError("number of weight values should match total components")
        if mean_points is not None:
            if mean_points.size != components.shape[1]:
                raise ValueError("number of mean points should match component points")

    builder = flatbuffers.Builder(1024)

    # add each component to the builder first
    if components is not None:
        N_components = components.shape[0]
    else:
        N_components = 0
    b_components = []
    if components is not None:
        for i in reversed(range(N_components)):
            if weights is not None:
                weight = weights[i]
            else:
                weight = 1.

            Uvector = components[i]

            x = builder.CreateNumpyVector(Uvector)
            SerializedComponent.SerializedComponentStart(builder)
            SerializedComponent.SerializedComponentAddPoints(builder, x)
            SerializedComponent.SerializedComponentAddScale(builder, weight)
            b_components.append(SerializedComponent.SerializedComponentEnd(builder))

    SerializedMorphableModel.SerializedMorphableModelStartComponentsVector(builder, N_components)
    for c in b_components:
        builder.PrependUOffsetTRelative(c)
    all_components = builder.EndVector(N_components)

    # add the raw data of the mean mesh first
    if mean_points is not None:
        builder_points = builder.CreateNumpyVector(mean_points)
    else:
        builder_points = builder.CreateNumpyVector(numpy.array([], dtype=numpy.float32))
    if mean_indices is not None:
        builder_tri_indices = builder.CreateNumpyVector(mean_indices)
    else:
        builder_tri_indices = builder.CreateNumpyVector(numpy.array([], dtype=numpy.float32))
    if mean_uvs is not None:
        builder_uvs = builder.CreateNumpyVector(mean_uvs)
    if submesh_offsets is not None:
        builder_submeshes = builder.CreateNumpyVector(submesh_offsets)

    # add the mean mesh
    SerializedTriMesh.SerializedTriMeshStart(builder)
    #if mean_points is not None:
    SerializedTriMesh.SerializedTriMeshAddPoints(builder, builder_points)
    #if mean_indices is not None:
    SerializedTriMesh.SerializedTriMeshAddTriIndices(builder, builder_tri_indices)
    if mean_uvs is not None:
        SerializedTriMesh.SerializedTriMeshAddUV(builder, builder_uvs)
    if submesh_offsets is not None:
        SerializedTriMesh.SerializedTriMeshAddSubmeshIndexOffset(builder, builder_submeshes)
    builder_meanMesh = SerializedTriMesh.SerializedTriMeshEnd(builder)

    # aggregate everything to the MModel
    SerializedMorphableModel.SerializedMorphableModelStart(builder)
    SerializedMorphableModel.SerializedMorphableModelAddVersion(builder, get_version())
    SerializedMorphableModel.SerializedMorphableModelAddMeanMesh(builder, builder_meanMesh)
    SerializedMorphableModel.SerializedMorphableModelAddComponents(builder, all_components)
    builder_mmodel = SerializedMorphableModel.SerializedMorphableModelEnd(builder)

    builder.Finish(builder_mmodel)

    return builder.Output()

def serialize_from_dictionary(data, uv_data=[]):
    """ Helper function to serialize the input dict data into the package flatbuffers format
    All keys are optional but at least there should be at least valid points OR components

    data -- expected input must be a dict with the following keys
    - points (optional) -- 3D vertices of the mean mesh (will get flattened)
    - trilist (optional) -- index triplets with the triangles in the main sub mesh of the mean mesh (will get flattened)
    - std (optional) -- standard deviation for each component (will get the sqrt of these values)
    - components (optional) -- components of the model as NumComponents x NumPoints
    - submeshes (optional) -- extract lists of triangle indices with further submeshes in the mean mesh (will get flattened)
    - uv_data -- dict with tcoords key with all UVs for the mean points (will get flattened)

    Return the serialized blob that can be stored into disk.
    """

    if not isinstance(data, dict):
        raise TypeError("non dict data passed to serialize_from_dictionary")
    if uv_data:
        if not isinstance(uv_data, dict):
            raise TypeError("non dict uv data passed to serialize_from_dictionary")

    if 'points' in data.keys():
        points = numpy.asarray(data['points'], dtype=numpy.float32).ravel()
    else:
        points = None
    
    if 'trilist' in data.keys():
        # this needs to be uint32 as uint16 doesn't seem to be natively supported by flatbuffers
        tri_indices = numpy.asarray(data['trilist'], dtype=numpy.uint32).ravel()
    else:
        tri_indices = None

    if 'submeshes' in data.keys() and tri_indices is not None:
        submesh_offsets = numpy.zeros(len(data['submeshes']), dtype=numpy.uint32)
        for (i, submesh) in enumerate(data['submeshes']):
            submesh_indices = numpy.asarray(submesh, dtype=numpy.uint32)
            if submesh_indices.size % 3 != 0:
                raise TypeError("invalid sub mesh size")
            # append the sub mesh indices            
            offset = tri_indices.size
            submesh_offsets[i] = offset
            tri_indices = numpy.append(tri_indices, submesh_indices.ravel())
    else:
        submesh_offsets = None

    # add std if in data
    if 'std' in data.keys():
        std = numpy.asarray(data['std'], dtype=numpy.float32)
        # need to take the square root of the std as scale factor
        std = numpy.sqrt(std)
    else:
        std = None
    
        # add components if needed
    if 'components' in data.keys():
        components = numpy.asarray(data['components'], dtype=numpy.float32)
    else:
        components = None

    uvs = None
    if uv_data:
        uvs = numpy.asarray(uv_data["tcoords"], dtype=numpy.float32).ravel()

    # check for missing data
    if components is None and points is None:
        raise TypeError("dictionary passed to serialize_from_dictionary does not contain either points nor components")

    # serialize into flatbuffers
    dataBlob = serialize_morphable_model_to_binary(points, tri_indices, components, std, uvs, submesh_offsets)

    return dataBlob

def serialize_from_dictionary_file(inFile, outFile, uvFile=[]):
    """Helper to serialize Morphable Model from a dictionary saved in a pickle file and write the result into the output file"""

    with open(inFile, 'rb') as file:
        d = pickle.load(file)

    uv_dict = []
    if uvFile:
        with open(uvFile) as f:
            uv_dict = json.load(f)

    dataBlob = serialize_from_dictionary(d, uv_dict)

    with open(outFile,'wb') as f:
        f.write(dataBlob)


import cv2
import os
import numpy as np
import pickle
import sys
import scipy.io as io
import glob
from scipy import optimize
from tqdm import tqdm
from helpers.multiface import fc_predictor
from helpers.avatars import serialize
from helpers.hephaestus import hephaestus_bindings as hephaestus
from helpers import transform

def _procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

class NMFCRenderer:
    def __init__(self, args):
        self.args = args

        MM_inpath ='files/all_all_all.mat'
        multiface_path = 'files/A3'
        self.handler = fc_predictor.Handler([multiface_path], 70, 192, args.gpu_id)

        # load sampling indices
        with open('files/sampler1035ver.pkl','rb') as f:
            sampler = pickle.load(f)
        idxs = sampler['idxs']

        # Load 3DMM
        with open('files/exp_30.dat','rb') as f:
            exp_30 = serialize.deserialize_binary_to_morphable_model(f.read())
        self.m_fit = exp_30['mean_points'].reshape((-1, 3))[idxs].astype(np.float32)
        
        MM = io.loadmat(MM_inpath)['fmod']
        self.id_basis = MM['id_basis'][0][0]
        self.exp_basis = MM['exp_basis'][0][0]
        self.mean_shape = MM['mean'][0][0]

        self.M_fit = self.mean_shape.reshape((-1, 3))[idxs]
        Bas = np.concatenate([self.id_basis, self.exp_basis], axis=1)
        Bas_use_3Darr = Bas.T.reshape((Bas.shape[1], -1, 3)).transpose((2, 1, 0))
        Bas_fit = Bas_use_3Darr[:, idxs, :]
        self.Bas_fit = Bas_fit.transpose((2, 1, 0)).reshape((Bas.shape[1], -1)).T

        # initialize hephaestus renderer
        self.width = 256  # NMFC width hardcoded
        self.height = 256  # NMFC height hardcoded
        shaderDir = 'helpers/shaders'   # needs to point to the directory with the hephaestus shaders
        hephaestus.init_system(self.width, self.height, shaderDir)
        hephaestus.set_clear_color(0, 0, 0, 0)

        # create a model from the mean mesh of the 3DMM
        self.model = hephaestus.create_NMFC(exp_30['mean_points'], exp_30['mean_indices'])
        hephaestus.setup_model(self.model)

    def fit_3DMM(self, points, Wbound_Cid=.8, Wbound_Cexp=1.5):
        # Compute optmisation bounds
        num_id = self.id_basis.shape[1]
        num_exp = self.exp_basis.shape[1]
        UBcoefs = np.vstack((Wbound_Cid * np.ones((num_id, 1)), 
                             Wbound_Cexp * np.ones((num_exp,1))))
        Bcoefs = np.hstack((-UBcoefs, UBcoefs))

        # Align points with mean shape
        _, points_aligned, tform = _procrustes(self.M_fit, points, reflection=False)
        b = points_aligned.ravel() - self.M_fit.ravel()

        coefs = optimize.lsq_linear(self.Bas_fit, b, bounds=(Bcoefs[:, 0], Bcoefs[:, 1]), method='trf',
                                    tol=1e-10, lsq_solver=None, lsmr_tol=None, max_iter=None, verbose=0)
        return coefs['x']

    def reconstruct(self, images):
        # Values to return
        cam_params = [] 
        id_params = []
        exp_params = []
        landmarks5 = []
        success = True

        handler_ret_prev = None
        n_consecutive_fails = 0
        # Perform 3D face reconstruction for each given frame.
        print('Running face reconstruction')
        for image in tqdm(images):
            if isinstance(image, str):
                # Read image
                frame = cv2.imread(image)
                if frame is None:
                    print('Failed to read %s' % image)
                    success = False
                    break
            else:
                # If we are given images, convert them to BGR
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run multiface detector
            handler_ret = self.handler.get(frame)

            # Check if dense landmarks were found and only one face exists in the image.
            if len(handler_ret) == 2:
                # Face(s) found in frame.
                n_consecutive_fails = 0
                landmarks, lands5 = handler_ret[0], handler_ret[1]
                if len(landmarks) > 1:
                    print('More than one faces were found')
                    landmarks, lands5 = landmarks[0:1], lands5[0:1]
            else:
                # Face not found in frame.
                n_consecutive_fails += 1
                print('Failed to find a face (%d times in a row)' % n_consecutive_fails)
                if handler_ret_prev is None or n_consecutive_fails > 5:
                    success = False
                    break
                else:
                    # Recover using previous landmarks
                    handler_ret = handler_ret_prev

            # Perform fitting.
            pos_lms = landmarks[0][:-68].astype(np.float32)
            shape = pos_lms.copy() * np.array([1, -1, -1], dtype=np.float32) # landmark mesh is in left-handed system
            coefs = self.fit_3DMM(shape)
            Pinv = transform.estimate_affine_matrix_3d23d(self.m_fit, pos_lms).astype(np.float32)

            # Gather results.
            cam_params.append(transform.P2sRt(Pinv)) # Scale, Rotation, Translation
            id_params.append(coefs[:157])            # Identity coefficients
            exp_params.append(coefs[157:])           # Expression coefficients
            landmarks5.append(lands5[0])             # Five facial landmarks
            handler_ret_prev = handler_ret

        # Return
        return success, cam_params, id_params, exp_params, landmarks5

    def computeNMFCs(self, cam_params, id_params, exp_params, return_RGB=False):
        nmfcs = []
        print('Computing NMFCs')
        for cam_param, id_param, exp_param in tqdm(zip(cam_params, id_params, exp_params), total=len(cam_params)):
            # Get Scale, Rotation, Translation
            S, R, T = cam_param

            # Compute face without pose.
            faceAll = self.mean_shape.ravel() + np.matmul(self.id_basis, id_param).ravel() + exp_param.dot(self.exp_basis.T)

            # Compute face with pose.
            T = (T / S).reshape(3,1)
            posed_face3d = R.dot(faceAll.reshape(-1, 3).T) + T

            # Use hephaestus to generate the NMFC image.
            hephaestus.update_positions(self.model, posed_face3d.astype(np.float32).T.ravel())

            # setup orthographic projection and place the camera
            viewportWidth = self.width / S
            viewportHeight = self.height / S

            # seems the viewport is inverted for Vulkan, handle this by inverting the ortho projection
            hephaestus.set_orthographics_projection(self.model, viewportWidth * 0.5, -viewportWidth * 0.5,
                                                    -viewportHeight * 0.5, viewportHeight * 0.5, -10, 10)

            # set the cameara to look at the center of the mesh
            target = hephaestus.vec4(viewportWidth * 0.5, viewportHeight * 0.5, 0, 1)
            camera = hephaestus.vec4(viewportWidth * 0.5, viewportHeight * 0.5, -3, 1)
            hephaestus.set_camera_lookat(self.model, camera, target)

            # Render NMFC
            data, channels, width, height = hephaestus.render_NMFC(self.model)

            data3D = data.reshape((height, width, channels))
            data3D = data3D[:,:,0:3]
            if not return_RGB:
                data3D = data3D[..., ::-1]
            nmfcs.append(data3D)
        return nmfcs

    def clear(self):
        # clean up
        hephaestus.clear_system()

import os
import argparse
import numpy as np
import collections
import cv2
import pandas as pd
from shutil import copyfile
from helpers.audio_features.audioFeaturesExtractor import get_mid_features
from helpers.audio_features.deepspeechFeaturesExtractor import get_logits
from util.util import *


def save_audio_features(audio_features, name, split, args):
    save_results = True
    if split == 'train':
        n_parts = len(audio_features) // args.train_seq_length
        n_audio_features = n_parts  * args.train_seq_length
    else:
        n_audio_features = len(audio_features)
    for i in range(n_audio_features):
        n_frame = "{:06d}".format(i)
        part = "_{:06d}".format(i // args.train_seq_length) if split == 'train' else ""
        save_dir = os.path.join(args.dataset_path, split, 'audio_features', name + part)

        # Check if corresponding images directory exists before saving the audio features
        # If it doesn't, don't save results
        if os.path.exists(save_dir.replace('audio_features', 'images')):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savetxt(os.path.join(save_dir, n_frame + '.txt'), audio_features[i])
        else:
            save_results = False
            break

    # If test split, save audio .wav file as well
    if split == 'test' and save_results:
        save_dir = os.path.join(args.dataset_path, split, 'audio')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, name + '.wav')
        copyfile('temp_audio.wav', save_path)


def is_path_processed(name, split, args):
    first_part = '_000000' if split == 'train' else ''
    path = os.path.join(args.dataset_path, split, 'audio_features', name + first_part)
    return os.path.isdir(path)


def get_split_dict(csv_file, args):
    if csv_file and os.path.exists(csv_file):
        csv = pd.read_csv(csv_file)
        names = list(csv['filename'])
        names = [os.path.splitext(name)[0] for name in names]
        splits = list(csv['split'])
        split_dict = dict(zip(names, splits))
        return split_dict, set(val for val in split_dict.values())
    else:
        print('No metadata file found. All samples will be saved in the %s split.' % args.split)
        return None, set([args.split])


def get_video_paths_dict(dir):
    # Returns dict: {video_name: path, ...}
    if os.path.exists(dir) and is_video_file(dir):
        # If path to single .mp4 file was given directly.
        # If '_' in file name remove it.
        video_files = {os.path.splitext(os.path.basename(dir))[0].replace('_', '') : dir}
    else:
        video_files = {}
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_video_file(fname):
                    path = os.path.join(root, fname)
                    video_name = os.path.splitext(fname)[0]
                    if video_name not in video_files:
                        video_files[video_name] = path
    return collections.OrderedDict(sorted(video_files.items()))


def get_video_info(mp4_path):
    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    reader.release()
    return fps, n_frames


def fix_audio_features_size(audio_feats, n_frames):
    # Check that we have one feature vector per frame
    if len(audio_feats) < n_frames:
        a = audio_feats[-1:] * (n_frames - len(audio_feats))
        audio_feats.extend(audio_feats[-1:] * (n_frames - len(audio_feats)))
    if len(audio_feats) > n_frames:
        audio_feats = audio_feats[:n_frames]
    return audio_feats


def extract_audio_features(mp4_path, audio_save_path='temp_audio.wav'):
    print('Extracting audio features')
    # Get video frame rate.
    fps, n_frames = get_video_info(mp4_path)

    # Use ffmpeg to get audio data in .wav format.
    ffmpeg_call = 'ffmpeg -y -i ' + mp4_path.replace(' ', '\ ') + ' ' + audio_save_path + ' > /dev/null 2>&1'
    os.system(ffmpeg_call)

    # Extract lower level audio features
    audio_feats = get_mid_features(audio_save_path, 8/fps, 1/fps, 8/(fps * 16), 1/(fps * 16))
    audio_feats = fix_audio_features_size(audio_feats, n_frames)

    # Extract deepspeech character one-hot vectors (higher level features)
    deepspeech_feats = get_logits(audio_save_path, fps, window=8)

    if len(deepspeech_feats) == 1:
        # If no deepspeech features detected, replicate empty token.
        deepspeech_feats = deepspeech_feats * len(audio_feats)
    deepspeech_feats = fix_audio_features_size(deepspeech_feats, n_frames)

    # Concatenate features
    feats = [np.concatenate((af, df)) for af, df in zip(audio_feats, deepspeech_feats)]
    return feats


def main():
    print('---- Extract audio features from .mp4 files ---- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_files_path', type=str, required=True, 
                        help='Path to video root directory.')
    parser.add_argument('--dataset_path', type=str, default='datasets/voxceleb', 
                        help='Path to save dataset.')
    parser.add_argument('--metadata_path', type=str, default=None, 
                        help='Path to metadata (train/test split information).')
    parser.add_argument('--train_seq_length', default=50, type=int, help='The number of frames for each training sequence.')
    parser.add_argument('--split', default='train', type=str, help='The default split for data if no metadata file is provided. [train|test]')
    args = parser.parse_args()

    # Read metadata files to create data split
    split_dict, splits = get_split_dict(args.metadata_path, args)

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    # Store video paths in dictionary.
    mp4_paths_dict = get_video_paths_dict(args.original_files_path)
    n_mp4s = len(mp4_paths_dict)
    print('Number of videos to process: %d \n' % n_mp4s)

    # Run audio feature extraction.
    n_completed = 0
    for name, mp4_path in mp4_paths_dict.items():
        n_completed += 1
        split = split_dict[name] if split_dict else args.split
        if not is_path_processed(name, split, args):
            # Extract features
            feats = extract_audio_features(mp4_path)

            # Save features
            save_audio_features(feats, name, split, args)
            
            os.remove('temp_audio.wav')
            print('(%d/%d) %s (%s file) [SUCCESS]' % (n_completed, n_mp4s, mp4_path, split))
        else:
            print('(%d/%d) %s (%s file) already processed!' % (n_completed, n_mp4s, mp4_path, split))

if __name__ == "__main__":
    main()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from dataloader.data_loader import CreateDataLoader
from models.headGAN import headGANModelG, headGANModelD 
from options.train_options import TrainOptions
from util.visualizer import Visualizer
import util.util as util
import torchvision
from models.segmenter_pytorch.segmenter import Segmenter

torch.autograd.set_detect_anomaly(True)

opt = TrainOptions().parse()

visualizer = Visualizer(opt)
segmenter = Segmenter(opt.gpu_ids[0])

modelG = headGANModelG()
modelG.initialize(opt)
modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)

modelD = headGANModelD()
modelD.initialize(opt)
modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
visualizer.vis_print('Number of identities in dataset %d' % data_loader.dataset.n_identities)
visualizer.vis_print('Number of sequences in dataset %d' % dataset_size)

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

if opt.continue_train:
    try:
        start_epoch, seen_seqs = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, seen_seqs = 1, 0

    if seen_seqs > 0:
        # initialize dataset again
        if opt.serial_batches:
            data_loader = CreateDataLoader(opt, seen_seqs)
            dataset = data_loader.load_data()
            dataset_size = len(data_loader)

    visualizer.vis_print('Resuming from epoch %d at iteration %d' % (start_epoch, seen_seqs))

    if start_epoch > opt.niter:
        modelG.module.update_learning_rate(start_epoch)
        modelD.module.update_learning_rate(start_epoch)
    if start_epoch >= opt.niter_start:
        data_loader.dataset.update_sequence_length((start_epoch - opt.niter_start + opt.niter_step) // opt.niter_step)
else:
    start_epoch, seen_seqs = 1, 0
    visualizer.vis_print('Initiating training.')

seen_seqs_start_time = None
n_steps_G = 0

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for idx, data in enumerate(dataset):
        # New batch of sequences
        if seen_seqs_start_time is None:
            seen_seqs_start_time = time.time()

        bs, n_frames_total, height, width = data['image'].size()
        n_frames_total = n_frames_total // opt.output_nc

        ref_input_A = Variable(data['ref_nmfc']).view(opt.batch_size, opt.input_nc, height, width).cuda(opt.gpu_ids[0])
        ref_input_B = Variable(data['ref_image']).view(opt.batch_size, opt.output_nc, height, width).cuda(opt.gpu_ids[0])

        ref_input_B_segmenter = Variable(data['ref_image_segmenter']).view(opt.batch_size, opt.output_nc, 512, 512).cuda(opt.gpu_ids[0])
        ref_masks_B = segmenter.get_masks(ref_input_B_segmenter, (height, width)).view(opt.batch_size, 1, height, width)

        # Go through sequences
        for i in range(0, n_frames_total-opt.n_frames_G+1):
            nmfc = Variable(data['nmfc'][:, i*opt.input_nc:(i+opt.n_frames_G)*opt.input_nc, ...])
            input_A = nmfc.view(opt.batch_size, opt.n_frames_G, opt.input_nc, height, width).cuda(opt.gpu_ids[0])

            image = Variable(data['image'][:, i*opt.output_nc:(i+opt.n_frames_G)*opt.output_nc, ...])
            input_B = image.view(opt.batch_size, opt.n_frames_G, opt.output_nc, height, width).cuda(opt.gpu_ids[0])

            image_segmenter = Variable(data['image_segmenter'][:, (i+opt.n_frames_G-1)*opt.output_nc:(i+opt.n_frames_G)*opt.output_nc, ...])
            image_segmenter = image_segmenter.view(opt.batch_size, opt.output_nc, 512, 512).cuda(opt.gpu_ids[0])

            masks_B = segmenter.get_masks(image_segmenter, (height, width)).view(opt.batch_size, 1, height, width)
            masks_union_B = segmenter.join_masks(masks_B, ref_masks_B)

            audio_feats = None
            if not opt.no_audio_input:
                audio_feats = Variable(data['audio_feats'][:, (i+opt.n_frames_G-1)*opt.naf:(i+opt.n_frames_G)*opt.naf])
                audio_feats = audio_feats.cuda(opt.gpu_ids[0])
                
            mouth_centers = Variable(data['mouth_centers'][:, i*2:(i+opt.n_frames_G)*2]).view(opt.batch_size, opt.n_frames_G, 2) if not opt.no_mouth_D else None

            input = input_A.view(opt.batch_size, -1, height, width)
            ref_input = torch.cat([ref_input_A, ref_input_B], dim=1)

            # Generator forward
            generated_B, feat_maps, warped, flows, masks = modelG(input, ref_input, audio_feats)

            real_A = input_A[:, opt.n_frames_G-1, :, :, :]
            real_B = input_B[:, opt.n_frames_G-1, :, :, :]

            mouth_centers = mouth_centers[:,opt.n_frames_G - 1,:] if not opt.no_mouth_D else None

            # Image (and Mouth) Discriminator forward
            losses = modelD(real_B, generated_B, warped, real_A, masks_union_B, masks, audio_feats, mouth_centers)

            losses = [torch.mean(loss) for loss in losses]
            loss_dict = dict(zip(modelD.module.loss_names, losses))

            # Losses
            loss_D = loss_dict['D_generated'] + loss_dict['D_real']
            loss_G = loss_dict['G_GAN'].clone()

            if not opt.no_ganFeat_loss:
                loss_G += loss_dict['G_GAN_Feat']

            if not opt.no_vgg_loss:
                loss_G += loss_dict['G_VGG']

            if not opt.no_maskedL1_loss:
                loss_G += loss_dict['G_MaskedL1']

            if not opt.no_flownetwork:
                if not opt.no_vgg_loss:
                    loss_G += loss_dict['G_VGG_w']

                if not opt.no_maskedL1_loss:
                    loss_G += loss_dict['G_MaskedL1_w']

                loss_G += loss_dict['G_L1_mask']

            if not opt.no_mouth_D:
                loss_G += loss_dict['Gm_GAN'] + loss_dict['Gm_GAN_Feat']
                loss_D += (loss_dict['Dm_generated'] + loss_dict['Dm_real']) * 0.5

            # Backward
            optimizer_G = modelG.module.optimizer_G
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            n_steps_G += 1
            if n_steps_G % opt.n_steps_update_D == 0:
                optimizer_D = modelD.module.optimizer_D
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

        # End of sequences
        seen_seqs += opt.batch_size

        # Print out errors
        if (seen_seqs / opt.batch_size) % opt.print_freq == 0:
            t = (time.time() - seen_seqs_start_time) / (opt.print_freq * opt.batch_size)
            seen_seqs_start_time = time.time()
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t_epoch = util.seconds_to_hours_mins(time.time() - epoch_start_time)
            visualizer.print_current_errors(epoch, seen_seqs, errors, t, t_epoch)
            visualizer.plot_current_errors(errors, seen_seqs)

        # Display output images
        if (seen_seqs / opt.batch_size) % opt.display_freq == 0:
            visual_dict = []
            for i in range(opt.batch_size):
                visual_dict += [('input_nmfc_image %d' % i, util.tensor2im(real_A[i, :opt.input_nc], normalize=False))]
                visual_dict += [('generated image %d' % i, util.tensor2im(generated_B[i])),
                                ('warped image %d' % i, util.tensor2im(warped[i])),
                                ('real image %d' % i, util.tensor2im(real_B[i]))]
                visual_dict += [('reference image %d' % i, util.tensor2im(ref_input_B[i]))]
                visual_dict += [('flow %d' % i, util.tensor2flow(flows[i]))]
                visual_dict += [('masks union %d' % i, util.tensor2im(masks_union_B[i], normalize=False))]
                visual_dict += [('masks %d' % i, util.tensor2im(masks[i], normalize=False))]

            visuals = OrderedDict(visual_dict)
            visualizer.display_current_results(visuals, epoch, seen_seqs)

        # Save latest model
        if (seen_seqs / opt.batch_size) % opt.save_latest_freq == 0:
            modelG.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, seen_seqs), delimiter=',', fmt='%d')
            visualizer.vis_print('Saved the latest model (epoch %d, seen sequences %d)' % (epoch, seen_seqs))

        # Break when we have gone through the entire dataset.
        if seen_seqs >= dataset_size:
            break

    # End of epoch
    seen_seqs = 0
    visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # Save model for this epoch, as latest
    modelG.module.save('latest')
    modelD.module.save('latest')
    np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
    visualizer.vis_print('Saved the model at the end of epoch %d (as latest)' % (epoch))

    if epoch % opt.save_epoch_freq == 0:
        modelG.module.save(epoch)
        modelD.module.save(epoch)
        visualizer.vis_print('Saved the model at the end of epoch %d' % (epoch))

    # Linearly decay learning rate after certain iterations
    if epoch + 1 > opt.niter:
        modelG.module.update_learning_rate(epoch + 1)
        modelD.module.update_learning_rate(epoch + 1)

    # Grow training sequence length
    if epoch + 1 >= opt.niter_start:
        data_loader.dataset.update_sequence_length((epoch + 1 - opt.niter_start + opt.niter_step) // opt.niter_step)

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from facenet_pytorch import MTCNN, extract_face
from PIL import Image

from models.headGAN import headGANModelG 
from options.reenact_options import ReenactOptions
from helpers.audio_features.audioFeaturesExtractor import get_mid_features
from helpers.audio_features.deepspeechFeaturesExtractor import get_logits
from helpers.reconstruction import NMFCRenderer
from detect_faces import get_faces
from extract_audio_features import extract_audio_features
from util.util import *
from dataloader.base_dataset import get_params, get_transform

opt = ReenactOptions().parse(save=False)

if not opt.no_crop:
    detector = MTCNN(opt.cropped_image_size, opt.margin, post_process=False, device='cuda:' + str(opt.gpu_id))

renderer = NMFCRenderer(opt)

modelG = headGANModelG()
modelG.initialize(opt)
modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
modelG.eval()

if is_video_file(opt.driving_path):
    driving_frames, fps = read_mp4(opt.driving_path, opt.n_frames_G - 1)
else:
    print('%s is not a video. Exit' % opt.driving_path)
    exit(0)
        
if is_image_file(opt.reference_path):
    reference_image = read_image(opt.reference_path)
else:
    print('%s is not an image. Exit' % opt.reference_path)
    exit(0)

driving_name = os.path.splitext(os.path.basename(opt.driving_path))[0]
reference_name = os.path.splitext(os.path.basename(opt.reference_path))[0]
save_name = driving_name + '_' + reference_name
save_dir = os.path.join(opt.results_dir, opt.name, opt.which_epoch, 'reenact', save_name)
mkdir(save_dir)

# Detect faces
if not opt.no_crop:
    driving_stat, driving_frames = get_faces(detector, driving_frames, opt)
    if driving_stat:
        driving_frames = tensor2npimage(driving_frames)
    else:
        print('%s Face detection failed. Exit' % opt.driving_path)
        exit(0)
    reference_stat, reference_image = get_faces(detector, reference_image, opt)
    if reference_stat:
        reference_image = tensor2npimage(reference_image)
    else:
        print('%s Face detection failed. Exit' % opt.reference_path)
        exit(0)
else:
    reference_image = [cv2.resize(make_image_square(reference_image[0]), (opt.crop_size, opt.crop_size))]
    driving_frames = [cv2.resize(make_image_square(driving_frame), (opt.crop_size, opt.crop_size))
                      for driving_frame in driving_frames]
    
if not opt.no_audio_input:
    # Extract audio features
    audio_save_path = os.path.join(save_dir, 'audio.wav')
    audio_features = extract_audio_features(opt.driving_path, audio_save_path)

# Run face reconstruction for reference image
ref_success, ref_cam_params, ref_id_params, ref_exp_params, _ = renderer.reconstruct(reference_image)
if not ref_success:
    print('%s Face reconstruction failed. Exit' % opt.reference_path)
    exit(0)
reference_nmfc = renderer.computeNMFCs(ref_cam_params, ref_id_params, ref_exp_params, return_RGB=True)

# Run face reconstruction for driving video
success, cam_params, _, exp_params, _  = renderer.reconstruct(driving_frames)
if not success:
    print('%s Face reconstruction failed. Exit' % opt.driving_path)
    exit(0)

# Adapt driving camera parameters
cam_params = adapt_cam_params(cam_params, ref_cam_params, opt)

# Use the reference identity parameters
id_params = ref_id_params * len(exp_params)

# Render driving nmfcs
nmfcs = renderer.computeNMFCs(cam_params, id_params, exp_params, return_RGB=True)
renderer.clear()

height, width = reference_image[0].shape[:2]
params = get_params(opt, (width, height))
transform_nmfc = get_transform(opt, params, normalize=False)
transform_rgb = get_transform(opt, params)

ref_nmfc = transform_nmfc(Image.fromarray(reference_nmfc[0]))
ref_input_A = ref_nmfc.view(opt.batch_size, opt.input_nc, height, width).cuda(opt.gpu_ids[0])
ref_rgb = transform_rgb(Image.fromarray(reference_image[0]))
ref_input_B = ref_rgb.view(opt.batch_size, opt.output_nc, height, width).cuda(opt.gpu_ids[0])

driving_nmfc = torch.stack([transform_nmfc(Image.fromarray(nmfc)) for nmfc in nmfcs[:opt.n_frames_G]], dim=0)

print('Running generative network')
result_frames = []
with torch.no_grad():
    for i, nmfc in enumerate(tqdm(nmfcs[opt.n_frames_G-1:])):
        driving_nmfc = torch.cat([driving_nmfc[1:,:,:,:], transform_nmfc(Image.fromarray(nmfc)).unsqueeze(0)], dim=0)
        input_A = driving_nmfc.view(opt.batch_size, -1, opt.input_nc, height, width).cuda(opt.gpu_ids[0])

        if not opt.no_audio_input:
            audio_feats = torch.tensor(audio_features[i]).float().view(opt.batch_size, -1).cuda(opt.gpu_ids[0])
        else:
            audio_feats = None

        input = input_A.view(opt.batch_size, -1, height, width)
        ref_input = torch.cat([ref_input_A, ref_input_B], dim=1)

        generated, warped, _ = modelG(input, ref_input, audio_feats)

        generated = tensor2im(generated[0])
        result_list = [reference_image[0], driving_frames[i + opt.n_frames_G - 1], generated]

        mkdirs([os.path.join(save_dir, 'driving'), 
                os.path.join(save_dir, 'generated')])

        save_image(driving_frames[i + opt.n_frames_G - 1], os.path.join(save_dir, 'driving', str(i).zfill(6) + '.png'))
        save_image(generated, os.path.join(save_dir, 'generated', str(i).zfill(6) + '.png'))

        if opt.show_warped:
            warped = tensor2im(warped[0])
            result_list += [warped]
            mkdir(os.path.join(save_dir, 'warped'))
            save_image(warped, os.path.join(save_dir, 'warped', str(i).zfill(6) + '.png'))

        result_frame = np.concatenate(result_list, axis=1)
        result_frames.append(result_frame)

video_save_path = os.path.join(save_dir, 'video.mp4')
save_video(result_frames, video_save_path, fps)

if not opt.no_audio_input:
    # Add audio to generated video
    save_path = os.path.join(save_dir, 'video+audio.mp4')
    call = 'ffmpeg -y -i ' + video_save_path + ' -i ' + audio_save_path + ' -c:v copy -c:a aac ' + save_path + ' > /dev/null 2>&1'
    os.system(call)
    print("Video saved to %s" % save_path)
else:
    print("Video saved to %s" % video_save_path)
import os

os.system('cd models/flownet2_pytorch/; bash install.sh; cd ../../')

import zipfile
import wget
import os
import argparse

class MyProgressBar():
    def __init__(self, message):
        self.message = message

    def get_bar(self, current, total, width=80):
        print(self.message + ": %d%%" % (current / total * 100), end="\r")

def unzip_file(file_name, unzip_path):
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(unzip_path)
    zip_ref.close()
    os.remove(file_name)

def main():
    # Download urls
    download_urls = ['https://www.dropbox.com/scl/fi/u5sopigekavoqvtloz44j/files.zip?rlkey=qtog3kpbauxg7foxbjmj9jeid&dl=1',
                     'https://www.dropbox.com/scl/fi/kyiilyxnwxswzq6gpuv69/checkpoints.zip?rlkey=pqtj5xk0dkfktlbwdk1ealqou&dl=1',
                     'https://www.dropbox.com/scl/fi/g4o2t3tvr3smkw75o1njt/retinaface_r50_v1.zip?rlkey=n9nwaf6sgg3jqq8ho37n8ec1b&dl=1',
                     'https://www.dropbox.com/scl/fi/kc7zjlg4w1innmiwslk6h/resample.zip?rlkey=wuijjsgn9vn5est43akdg4x8b&dl=1']
                     
    for i, path in enumerate(download_urls):
        fname = path.split('/')[-1].split('?')[0]
        if fname == 'retinaface_r50_v1.zip':
            dir = os.path.join(os.path.expanduser('~'), '.insightface/models/')
            if not os.path.exists(dir):
                os.makedirs(dir)
        elif fname == 'resample.zip':
            dir = os.path.join(os.path.expanduser('~'), '.local/lib/python3.7/site-packages')
            if not os.path.exists(dir):
                os.makedirs(dir)
        else:
            dir = './'
        fpath = os.path.join(dir, fname)
        if not os.path.exists(fpath):
            bar = MyProgressBar('Downloading file %d/%d' % (i+1,
                                len(download_urls)))
            wget.download(path, fpath, bar=bar.get_bar)
            print('\n')
            print('Unzipping file...')
            unzip_file(fpath, dir)
    print('DONE!')

if __name__ == "__main__":
    main()

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for data
        parser.add_argument('--max_seqs_per_identity', type=int, default=1000, help='How many short sequences (of 50 frames) to use per identity')
        # show, save frequencies
        parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
        parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        # for display
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--niter', type=int, default=6, help='# of iter at starting learning rate.')
        parser.add_argument('--niter_decay', type=int, default=4, help='# of iter to linearly decay learning rate to zero.')
        parser.add_argument('--niter_start', type=int, default=3, help='in which epoch do we start doubling the training sequences length')
        parser.add_argument('--niter_step', type=int, default=1, help='every how many epochs we double the training sequences length')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', default=True, help='Do not use TTUR training scheme')
        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        # for discriminators and losses
        parser.add_argument('--n_steps_update_D', type=int, default=4, help='')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_maskedL1', type=float, default=50.0, help='')
        parser.add_argument('--lambda_mask', type=float, default=10.0, help='')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_maskedL1_loss', action='store_true', help='')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        parser.add_argument('--n_frames_D', type=int, default=3, help='number of frames to feed into temporal discriminator')
        parser.add_argument('--no_mouth_D', action='store_true', help='if true, do not use mouth discriminator')
        parser.add_argument('--ROI_size', type=int, default=64, help='The size of the mouth area.')
        self.isTrain = True
        return parser

from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # Out directory.
        parser.add_argument('--time_fwd_pass', action='store_true', 
                            help='Show the forward pass time for synthesizing each frame.')
        # Default test arguments.
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(batch_size=1)
        parser.set_defaults(nThreads=0)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from .base_options import BaseOptions

class ReenactOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # Basic options
        parser.add_argument('--reference_path', type=str, default='assets/reference.png',
                            help='The path to reference image')
        parser.add_argument('--driving_path', type=str, default='assets/driving.mp4',
                            help='The path to driving .mp4 video')
        parser.add_argument('--no_crop', action='store_true', 
                            help='If set, do not perform face detection and cropping')
        parser.add_argument('--show_warped', action='store_true', 
                            help='If set, add the warped image to the results')
        # Face detection options
        parser.add_argument('--gpu_id', type=int, default=0, 
                            help='The gpu id for face detector and face reconstruction modules')
        parser.add_argument('--mtcnn_batch_size', default=1, type=int, 
                            help='The number of frames for face detection')
        parser.add_argument('--cropped_image_size', default=256, type=int, 
                            help='The size of frames after cropping the face')
        parser.add_argument('--margin', default=100, type=int, 
                            help='The margin around the face bounding box')
        parser.add_argument('--dst_threshold', default=0.35, type=float, 
                            help='Max L_inf distance between any bounding boxes in a video. (normalised by image size: (h+w)/2)')
        parser.add_argument('--height_recentre', default=0.0, type=float, 
                            help='The amount of re-centring bounding boxes lower on the face')
        # Reenactment options
        parser.add_argument('--no_scale_or_translation_adaptation', action='store_true',
                            help='Do not perform scale or translation adaptation using statistics from driving video')
        parser.add_argument('--no_translation_adaptation', action='store_true',
                            help='Do not perform translation adaptation using statistics from driving video')
        parser.add_argument('--standardize', action='store_true',
                            help='Perform adaptation using also std from driving videos')
        # Default reenact arguments
        parser.set_defaults(batch_size=1)
        self.isTrain = False
        return parser
import sys
import argparse
import os
from util import util
import torch
import random
import numpy as np

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='headGAN', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--results_dir', type=str, default='./results', help='saves results here.')
        parser.add_argument('--load_pretrain', type=str, default=None, help='Directory of model to load.')
        parser.add_argument('--model', type=str, default='headGAN', help='which model to use')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        parser.add_argument('--target_name', type=str, default=None, help='If given, use only this target identity.')

        # input/output sizes
        parser.add_argument('--use_landmarks_input', action='store_true', help='Use facial landmark sketches instead of NMFC images as conditional input.')
        parser.add_argument('--resize', action='store_true', help='')
        parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input NMFC channels')
        parser.add_argument('--n_frames_G', type=int, default=3, help='number of frames to look in the past (T) + current -> T + 1')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size')
        parser.add_argument('--n_frames_total', type=int, default=6, help='Starting number of frames to read for each sequence in the batch. Increases progressively.')
        parser.add_argument('--naf', type=int, default=300, help='number of audio features for each frame.')

        # for setting inputs
        parser.add_argument('--reference_frames_strategy', type=str, default='ref', help='[ref|previous]')
        parser.add_argument('--dataroot', type=str, default='datasets/voxceleb')
        parser.add_argument('--dataset_mode', type=str, default='video')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes sequences in sorted order for making batches, otherwise takes them randomly')
        parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

        # for generator
        parser.add_argument('--no_audio_input', action='store_true', help='')
        parser.add_argument('--no_pixelshuffle', action='store_true', help='Do not use PixelShuffle for upsampling.')
        parser.add_argument('--no_previousframesencoder', action='store_true', help='Do not condition synthesis of generator on previouly generated frames.')
        parser.add_argument('--no_flownetwork', action='store_true', help='')
        parser.add_argument('--netG', type=str, default='headGAN', help='selects model to use for netG.')
        parser.add_argument('--norm_G', type=str, default='spectralspadeinstance3x3', help='The type of adaptive normalization.')
        parser.add_argument('--norm_G_noadapt', type=str, default='spectralinstance', help='The type of non-adaptive normalization.')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--down_size', type=int, choices=(8, 16, 32, 64), default=64, help="The size of the bottleneck spatial dimension when encoding-decoding.")
        parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of encoder convolutions')
        parser.add_argument('--initial_kernel_size', type=int, default=7, help='kernel size of the first convolution of encoder')

        # initialization
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

        # visualization
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()
        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    def parse(self, save=False):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.dataroot = opt.dataroot.replace('\ ', ' ')
        # Remove '_' from target_name.
        if opt.target_name:
            opt.target_name = opt.target_name.replace('_', '')
        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.opt = opt
        return self.opt

import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def forward(self):
        pass

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if 'G' in network_label:
                raise('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y, nmfc):
        mask = torch.sum(nmfc, dim=1, keepdim=True)
        mask = (mask > (torch.ones_like(mask) * 0.01)).float()
        loss = self.criterion(x * mask, y * mask)
        return loss

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda(gpu_ids)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def forward(self):
        pass

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if 'G' in network_label:
                raise('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        pass
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks import BaseNetwork, get_norm_layer

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt, input_nc):
        super().__init__()
        self.opt = opt
        self.input_nc = input_nc
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, self.input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt, input_nc):
        super().__init__()
        self.opt = opt
        self.input_nc = input_nc
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.input_nc

        norm_layer = get_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
import re
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
from models.networks import BaseNetwork, get_norm_layer
from models.flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d


# The following class has been taken from https://github.com/NVlabs/SPADE and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.norm_nc = norm_nc
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 256

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # Make sure segmap has the same spatial size with input.
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


# The following class has been taken from https://github.com/NVlabs/SPADE and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class SPADEResnetBlock(nn.Module):
    def __init__(self, semantic_nc, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# The following class has been taken from https://github.com/clovaai/stargan-v2 and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


# The following class has been taken from https://github.com/clovaai/stargan-v2 and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class AdaINResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, opt):
        super().__init__()
        self.opt = opt
        self.actv = nn.LeakyReLU(0.2)
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        # apply spectral norm if specified
        if 'spectral' in self.opt.norm_G:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        return out


class FlowNetwork(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.opt = opt
        self.ngf = opt.ngf
        self.nl = nl
        # The flow application operator (warping function).
        self.resample = Resample2d()

        # Use average pool 2D to downsample predicted flow.
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

        # Encoder first layer
        enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                     norm_layer(nn.Conv2d(opt.output_nc+opt.input_nc,
                                          self.ngf,
                                          kernel_size=opt.initial_kernel_size,
                                          padding=0)),
                     activation]
        self.enc = [nn.Sequential(*enc_first)]

        # Encoder downsampling layers
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)

            # Conditional encoders
            enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                             self.ngf * mult_out,
                                             kernel_size=opt.kernel_size,
                                             stride=2,
                                             padding=1)),
                        activation]
            self.enc.append(nn.Sequential(*enc_down))
        self.enc = nn.ModuleList(self.enc)

        # Residual part of decoder
        fin = (base**self.nl) * self.ngf
        fout = fin
        self.dec_res = []
        for i in range(self.nl):
            self.dec_res.append(SPADEResnetBlock(opt.input_nc * opt.n_frames_G, fin, fout, opt))
        self.dec_res = nn.ModuleList(self.dec_res)

        # Upsampling part of decoder.
        self.dec_up = []
        self.dec_main = []
        for i in range(self.nl):
            fin = (base**(self.nl-i)) * self.ngf

            # In case of PixelShuffle, let it do the filters amount reduction.
            fout = (base**(self.nl-i-1)) * self.ngf if self.opt.no_pixelshuffle else fin
            if self.opt.no_pixelshuffle:
                self.dec_up.append(nn.Upsample(scale_factor=2))
            else:
                self.dec_up.append(nn.PixelShuffle(upscale_factor=2))
            self.dec_main.append(SPADEResnetBlock(opt.input_nc * opt.n_frames_G, fin, fout, opt))

        self.dec_up = nn.ModuleList(self.dec_up)
        self.dec_main = nn.ModuleList(self.dec_main)
        self.dec_flow = [nn.ReflectionPad2d(3),
                         nn.Conv2d(self.ngf, 2,
                                   kernel_size=opt.initial_kernel_size,
                                   padding=0)]
        self.dec_flow = nn.Sequential(*self.dec_flow)
        self.dec_mask = [nn.ReflectionPad2d(3),
                         nn.Conv2d(self.ngf, 1,
                                   kernel_size=opt.initial_kernel_size,
                                   padding=0)]
        self.dec_mask = nn.Sequential(*self.dec_mask)

    def forward(self, input, ref_input):
        # Get dimensions sizes
        NN_ref, _, H, W = ref_input.size()
        N = input.size()[0]
        N_ref = NN_ref // N

        # Repeat the conditional input for all reference frames
        seg = input.repeat(1, N_ref, 1, 1).view(NN_ref, -1, H, W)

        # Encode
        feats = []
        feat = ref_input
        for i in range(self.nl + 1):
            feat = self.enc[i](feat)
            feats.append(feat)

        # Decode
        for i in range(self.nl):
            feat = self.dec_res[i](feat, seg)
        for i in range(self.nl):
            feat = self.dec_main[i](feat, seg)
            feat = self.dec_up[i](feat)

        # Compute flow layer
        flow = self.dec_flow(feat)
        mask = self.dec_mask(feat)
        mask = (torch.tanh(mask) + 1) / 2
        flow = flow * mask
        down_flow = flow

        # Apply flow on features to match them spatially with the desired pose.
        flow_feats = []
        for i in range(self.nl + 1):
            flow_feats.append(self.resample(feats[i], down_flow))

            # Downsample flow and reduce its magnitude.
            down_flow = self.downsample(down_flow) / 2.0
        return flow, flow_feats, mask

class FramesEncoder(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.ngf = opt.ngf
        self.nl = nl
        cond_enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                          norm_layer(nn.Conv2d(opt.output_nc+opt.input_nc,
                                               self.ngf,
                                               kernel_size=opt.initial_kernel_size,
                                               padding=0)),
                          activation]
        self.cond_enc = [nn.Sequential(*cond_enc_first)]
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)
            cond_enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                                  self.ngf * mult_out,
                                                  kernel_size=opt.kernel_size,
                                                  stride=2,
                                                  padding=1)),
                             activation]
            self.cond_enc.append(nn.Sequential(*cond_enc_down))
        self.cond_enc = nn.ModuleList(self.cond_enc)

    def forward(self, ref_input):
        # Encode
        feats = []
        x_cond_enc = ref_input
        for i in range(self.nl + 1):
            x_cond_enc = self.cond_enc[i](x_cond_enc)
            feats.append(x_cond_enc)
        return feats


class RenderingNetwork(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.opt = opt
        self.ngf = opt.ngf
        self.naf = opt.naf
        self.nl = nl

        # Encode
        cond_enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                          norm_layer(nn.Conv2d(opt.input_nc * opt.n_frames_G, self.ngf,
                                               kernel_size=opt.initial_kernel_size,
                                               padding=0)),
                          activation]
        self.cond_enc = [nn.Sequential(*cond_enc_first)]
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)
            cond_enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                                  self.ngf * mult_out,
                                                  kernel_size=opt.kernel_size,
                                                  stride=2,
                                                  padding=1)),
                             activation]
            self.cond_enc.append(nn.Sequential(*cond_enc_down))
        self.cond_enc = nn.ModuleList(self.cond_enc)

        # Decode
        self.cond_dec = []
        if not self.opt.no_audio_input:
            self.cond_dec_audio = []
        self.cond_dec_up = []

        for i in range(self.nl):
            fin = (base**(self.nl-i)) * opt.ngf
            fout = (base**(self.nl-i-1)) * opt.ngf if opt.no_pixelshuffle else fin
            self.cond_dec.append(SPADEResnetBlock(fin, fin, fout, opt))
            if not self.opt.no_audio_input:
                self.cond_dec_audio.append(AdaINResnetBlock(fout, fout, self.naf, opt))
            if self.opt.no_pixelshuffle:
                self.cond_dec_up.append(nn.Upsample(scale_factor=2))
            else:
                self.cond_dec_up.append(nn.PixelShuffle(upscale_factor=2))

        self.cond_dec.append(SPADEResnetBlock(opt.ngf, opt.ngf, opt.ngf, opt))
        if not self.opt.no_audio_input:
            self.cond_dec_audio.append(AdaINResnetBlock(opt.ngf, opt.ngf, self.naf, opt))
        self.cond_dec.append(SPADEResnetBlock(opt.output_nc, opt.ngf, opt.ngf, opt))
        self.cond_dec = nn.ModuleList(self.cond_dec)
        if not self.opt.no_audio_input:
            self.cond_dec_audio = nn.ModuleList(self.cond_dec_audio)
        self.cond_dec_up = nn.ModuleList(self.cond_dec_up)

        self.conv_img = [nn.ReflectionPad2d(3), nn.Conv2d(self.ngf, 3, kernel_size=opt.initial_kernel_size, padding=0)]
        self.conv_img = nn.Sequential(*self.conv_img)

    def forward(self, input, feat_maps, warped, audio_feats):
        # Encode
        x_cond_enc = input
        for i in range(self.nl + 1):
            x_cond_enc = self.cond_enc[i](x_cond_enc)
        x = x_cond_enc

        # Decode
        for i in range(self.nl):
            x = self.cond_dec[i](x, feat_maps[-i-1])
            if not self.opt.no_audio_input:
                x = self.cond_dec_audio[i](x, audio_feats)
            x = self.cond_dec_up[i](x)

        x = self.cond_dec[self.nl](x, feat_maps[0])
        if not self.opt.no_audio_input:
            x = self.cond_dec_audio[self.nl](x, audio_feats)
        x = self.cond_dec[self.nl+1](x, warped)

        imgs = self.conv_img(F.leaky_relu(x, 2e-1))
        imgs = torch.tanh(imgs)
        return imgs

class headGANGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.resample = Resample2d()

        # Activation functions
        activation = nn.ReLU()
        leaky_activation = nn.LeakyReLU(2e-1)

        # Non-adaptive normalization layer
        norm_layer = get_norm_layer(opt, opt.norm_G_noadapt)

        # Number of times to (up/down)-sample spatial dimensions.

        nl = round(math.log(opt.crop_size // opt.down_size, 2))

        # If pixelshuffle is used, quadruple the number of filters when
        # upsampling, else simply double them.
        base = 2 if self.opt.no_pixelshuffle else 4

        if not self.opt.no_flownetwork:
            self.flow_network = FlowNetwork(opt, activation, norm_layer, nl, base)
        else:
            self.frames_encoder = FramesEncoder(opt, activation, norm_layer, nl, base)
        self.rendering_network = RenderingNetwork(opt, activation, norm_layer, nl, base)

    def forward(self, input, ref_input, audio_feats):
        if not self.opt.no_flownetwork:
            # Get flow and warped features.
            flow, flow_feats, mask = self.flow_network(input, ref_input)
        else:
            flow = torch.zeros_like(ref_input[:,:2,:,:])
            mask = torch.zeros_like(ref_input[:,:1,:,:])
            flow_feats = self.frames_encoder(ref_input)
        feat_maps = flow_feats

        # Apply flows on reference frame(s)
        ref_rgb_input = ref_input[:,-self.opt.output_nc:,:,:]
        if not self.opt.no_flownetwork:
            warped = self.resample(ref_rgb_input, flow)
        else:
            warped = ref_rgb_input
        imgs = self.rendering_network(input, feat_maps, warped, audio_feats)

        if self.opt.isTrain:
            return imgs, feat_maps, warped, flow, mask
        else:
            return imgs, warped, flow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models.networks as networks
import models.losses as losses
from models.base_model import BaseModel
import util.util as util
import torchvision


class headGANModelD(BaseModel):
    def name(self):
        return 'headGANModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gpu_ids = opt.gpu_ids
        self.n_frames_D = opt.n_frames_D
        self.output_nc = opt.output_nc
        self.input_nc = opt.input_nc

        # Image discriminator
        netD_input_nc = self.input_nc + self.output_nc
        self.netD = networks.define_D(opt, netD_input_nc)

        # Mouth discriminator
        if not opt.no_mouth_D:
            if not self.opt.no_audio_input:
                netDm_input_nc = opt.naf + self.output_nc
            else:
                netDm_input_nc = self.output_nc
            self.netDm = networks.define_D(opt, netDm_input_nc)

        # load networks
        if (opt.continue_train or opt.load_pretrain):
            self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)
            if not opt.no_mouth_D:
                self.load_network(self.netDm, 'Dm', opt.which_epoch, opt.load_pretrain)
            print('---------- Discriminators loaded -------------')
        else:
            print('---------- Discriminators initialized -------------')

        # set loss functions and optimizers
        self.old_lr = opt.lr
        self.criterionGAN = losses.GANLoss(opt.gan_mode, tensor=self.Tensor, opt=self.opt)
        self.criterionL1 = torch.nn.L1Loss()
        if not opt.no_vgg_loss:
            self.criterionVGG = losses.VGGLoss(self.opt.gpu_ids[0])
        if not opt.no_maskedL1_loss:
            self.criterionMaskedL1 = losses.MaskedL1Loss()

        self.loss_names = ['G_VGG', 'G_GAN', 'G_GAN_Feat', 'G_MaskedL1', 'D_real', 'D_generated']
        if not self.opt.no_flownetwork:
            self.loss_names += ['G_VGG_w', 'G_MaskedL1_w', 'G_L1_mask']
        if not opt.no_mouth_D:
            self.loss_names += ['Gm_GAN', 'Gm_GAN_Feat', 'Dm_real', 'Dm_generated']

        beta1, beta2 = opt.beta1, opt.beta2
        lr = opt.lr
        if opt.no_TTUR:
            D_lr = lr
        else:
            D_lr = lr * 2

        # initialize optimizers
        params = list(self.netD.parameters())
        if not opt.no_mouth_D:
            params += list(self.netDm.parameters())
        self.optimizer_D = torch.optim.Adam(params, lr=D_lr, betas=(beta1, beta2))

    def compute_D_losses(self, netD, real_A, real_B, generated_B):
        # Input
        if real_A is not None:
            real_AB = torch.cat((real_A, real_B), dim=1)
            generated_AB = torch.cat((real_A, generated_B), dim=1)
        else:
            real_AB = real_B
            generated_AB = generated_B
        # D losses
        pred_real = netD.forward(real_AB)
        pred_generated = netD.forward(generated_AB.detach())
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        loss_D_generated = self.criterionGAN(pred_generated, False, for_discriminator=True)
        # G losses
        pred_generated = netD.forward(generated_AB)
        loss_G_GAN = self.criterionGAN(pred_generated, True, for_discriminator=False)
        loss_G_GAN_Feat = self.FM_loss(pred_real, pred_generated)
        return loss_D_real, loss_D_generated, loss_G_GAN, loss_G_GAN_Feat

    def FM_loss(self, pred_real, pred_generated):
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(min(len(pred_generated), self.opt.num_D)):
                for j in range(len(pred_generated[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionL1(pred_generated[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        else:
            loss_G_GAN_Feat = torch.zeros(self.bs, 1).cuda()
        return loss_G_GAN_Feat

    def forward(self, real_B, generated_B, warped_B, real_A, masks_B, masks, audio_feats, mouth_centers):
        lambda_feat = self.opt.lambda_feat
        lambda_vgg = self.opt.lambda_vgg
        lambda_maskedL1 = self.opt.lambda_maskedL1
        lambda_mask = self.opt.lambda_mask
        
        self.bs , _, self.height, self.width = real_B.size()

        # VGG loss
        loss_G_VGG = (self.criterionVGG(generated_B, real_B) * lambda_vgg) if not self.opt.no_vgg_loss else torch.zeros(self.bs, 1).cuda()
     
        # GAN and FM loss for Generator
        loss_D_real, loss_D_generated, loss_G_GAN, loss_G_GAN_Feat = self.compute_D_losses(self.netD, real_A, real_B, generated_B)
      
        loss_G_MaskedL1 = torch.zeros(self.bs, 1).cuda()
        if not self.opt.no_maskedL1_loss:
            loss_G_MaskedL1 = self.criterionMaskedL1(generated_B, real_B, real_A) * lambda_maskedL1

        loss_list = [loss_G_VGG, loss_G_GAN, loss_G_GAN_Feat, loss_G_MaskedL1, loss_D_real, loss_D_generated]
        
        # Warp Losses
        if not self.opt.no_flownetwork:
            loss_G_VGG_w = (self.criterionVGG(warped_B, real_B) * lambda_vgg) if not self.opt.no_vgg_loss else torch.zeros(self.bs, 1).cuda()
            loss_G_MaskedL1_w = torch.zeros(self.bs, 1).cuda()
            if not self.opt.no_maskedL1_loss:
                loss_G_MaskedL1_w = self.criterionMaskedL1(warped_B, real_B, real_A) * lambda_maskedL1
            loss_G_L1_mask = self.criterionL1(masks, masks_B.detach()) * lambda_mask
            loss_list += [loss_G_VGG_w, loss_G_MaskedL1_w, loss_G_L1_mask]

        # Mouth discriminator losses
        if not self.opt.no_mouth_D:
            # Extract mouth region around the center
            real_B_mouth, generated_B_mouth = util.get_ROI([real_B, generated_B], mouth_centers, self.opt)

            if not self.opt.no_audio_input:
                # Repeat audio features spatially for conditional input to mouth discriminator
                real_A_mouth = audio_feats[:, -self.opt.naf:].view(audio_feats.size(0), self.opt.naf, 1, 1)
                real_A_mouth = real_A_mouth.repeat(1, 1, real_B_mouth.size(2), real_B_mouth.size(3))
            else:
                real_A_mouth = None

            # Losses for mouth discriminator
            loss_Dm_real, loss_Dm_generated, loss_Gm_GAN, loss_Gm_GAN_Feat = self.compute_D_losses(self.netDm, real_A_mouth, real_B_mouth, generated_B_mouth)
            mouth_weight = 1
            loss_Gm_GAN *= mouth_weight
            loss_Gm_GAN_Feat *= mouth_weight
            loss_list += [loss_Gm_GAN, loss_Gm_GAN_Feat, loss_Dm_real, loss_Dm_generated]

        loss_list = [loss.unsqueeze(0) for loss in loss_list]
        return loss_list

    def save(self, label):
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        if not self.opt.no_mouth_D:
            self.save_network(self.netDm, 'Dm', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        if self.opt.niter_decay > 0:
            lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
            print('Update learning rate for D: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr


class headGANModelG(BaseModel):
    def name(self):
        return 'headGANModelG'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.n_frames_G = opt.n_frames_G
        self.output_nc = opt.output_nc
        self.input_nc = opt.input_nc

        self.netG = networks.define_G(opt)

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.load_pretrain)
            print('---------- Generator loaded -------------')
        else:
            print('---------- Generator initialized -------------')

        # Otimizer for G
        if self.isTrain:
            self.old_lr = opt.lr
            
            # initialize optimizer G
            paramsG = list(self.netG.parameters())
            beta1, beta2 = opt.beta1, opt.beta2
            lr = opt.lr
            if opt.no_TTUR:
                G_lr = lr
            else:
                G_lr = lr / 2
            self.optimizer_G = torch.optim.Adam(paramsG, lr=G_lr, betas=(beta1, beta2))

    def forward(self, input, ref_input, audio_feats):
        ret = self.netG.forward(input, ref_input, audio_feats)
        return ret

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        if self.opt.niter_decay > 0:
            lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            print('Update learning rate for G: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr


def create_model_G(opt):
    modelG = headGANModelG()
    modelG.initialize(opt)
    modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
    return modelG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y, nmfc):
        mask = torch.sum(nmfc, dim=1, keepdim=True)
        mask = (mask > (torch.ones_like(mask) * 0.01)).float()
        loss = self.criterion(x * mask, y * mask)
        return loss

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda(gpu_ids)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
import importlib
import torch
import util.util as util
import torch.nn as nn
from torch.nn import init
import torch.nn.utils.spectral_norm as spectral_norm

def get_norm_layer(opt, norm_type='spectralinstance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=True)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj
    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)
    return cls

def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.' + filename
    network = find_class_in_module(target_class_name, module_name)
    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network
    return network

def create_network(cls, opt, input_nc=None):
    net = cls(opt, input_nc) if input_nc else cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(opt.gpu_ids[0])
    net.init_weights(opt.init_type, opt.init_variance)
    return net

def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)

def define_D(opt, input_nc):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt, input_nc)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              #'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss 

import torch
import torch.nn as nn
import math

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*EPE(output_, target_)
                lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]


#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models, losses, datasets
from utils import flow_utils, tools

# fp32 copy of parameters for update
global param_copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default = -1, help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default = 255.)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true', help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam', skip_params=['params'])
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training'})
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                        'replicates': 1})
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                        'replicates': 1})

    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE',  action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
        args.loss_class = tools.module_to_dict(losses)[args.loss]

        args.training_dataset_class = tools.module_to_dict(datasets)[args.training_dataset]
        args.validation_dataset_class = tools.module_to_dict(datasets)[args.validation_dataset]
        args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
        args.log_file = join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

        if args.inference:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)

    print('Source Code')
    print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers, 
                   'pin_memory': True, 
                   'drop_last' : True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        if exists(args.training_dataset_root):
            train_dataset = args.training_dataset_class(args, True, **tools.kwargs_from_args(args, 'training_dataset'))
            block.log('Training Dataset: {}'.format(args.training_dataset))
            block.log('Training Input: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][0]])))
            block.log('Training Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][1]])))
            train_loader = DataLoader(train_dataset, batch_size=args.effective_batch_size, shuffle=True, **gpuargs)

        if exists(args.validation_dataset_root):
            validation_dataset = args.validation_dataset_class(args, True, **tools.kwargs_from_args(args, 'validation_dataset'))
            block.log('Validation Dataset: {}'.format(args.validation_dataset))
            block.log('Validation Input: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][0]])))
            block.log('Validation Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][1]])))
            validation_loader = DataLoader(validation_dataset, batch_size=args.effective_batch_size, shuffle=False, **gpuargs)

        if exists(args.inference_dataset_root):
            inference_dataset = args.inference_dataset_class(args, False, **tools.kwargs_from_args(args, 'inference_dataset'))
            block.log('Inference Dataset: {}'.format(args.inference_dataset))
            block.log('Inference Input: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][0]])))
            block.log('Inference Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][1]])))
            inference_loader = DataLoader(inference_dataset, batch_size=args.effective_inference_batch_size, shuffle=False, **inf_gpuargs)

    # Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)
                
            def forward(self, data, target, inference=False ):
                output = self.model(data)

                loss_values = self.loss(output, target)

                if not inference :
                    return loss_values
                else :
                    return loss_values, output

        model_and_loss = ModelAndLoss(args)

        block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
        block.log('Number of parameters: {}'.format(sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

        # assing to cuda or wrap with dataparallel, model and loss 
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))

            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed) 
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model_and_loss.parameters()]

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed) 

        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.inference:
                args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_EPE']
            model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

        elif args.resume and args.inference:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()

        else:
            block.log("Random initialization")

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        train_logger = SummaryWriter(log_dir = os.path.join(args.save, 'train'), comment = 'training')
        validation_logger = SummaryWriter(log_dir = os.path.join(args.save, 'validation'), comment = 'validation')

    # Dynamically load the optimizer with parameters passed in via "--optimizer_[param]=[value]" arguments 
    with tools.TimerBlock("Initializing {} Optimizer".format(args.optimizer)) as block:
        kwargs = tools.kwargs_from_args(args, 'optimizer')
        if args.fp16:
            optimizer = args.optimizer_class([p for p in param_copy if p.requires_grad], **kwargs)
        else:
            optimizer = args.optimizer_class([p for p in model_and_loss.parameters() if p.requires_grad], **kwargs)
        for param, default in list(kwargs.items()):
            block.log("{} = {} ({})".format(param, default, type(default)))

    # Log all arguments to file
    for argument, value in sorted(vars(args).items()):
        block.log2file(args.log_file, '{}: {}'.format(argument, value))

    # Reusable function for training and validataion
    def train(args, epoch, start_iteration, data_loader, model, optimizer, logger, is_validate=False, offset=0):
        statistics = []
        total_loss = 0

        if is_validate:
            model.eval()
            title = 'Validating Epoch {}'.format(epoch)
            args.validation_n_batches = np.inf if args.validation_n_batches < 0 else args.validation_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=100, total=np.minimum(len(data_loader), args.validation_n_batches), leave=True, position=offset, desc=title)
        else:
            model.train()
            title = 'Training Epoch {}'.format(epoch)
            args.train_n_batches = np.inf if args.train_n_batches < 0 else args.train_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=np.minimum(len(data_loader), args.train_n_batches), smoothing=.9, miniters=1, leave=True, position=offset, desc=title)

        last_log_time = progress._time()
        for batch_idx, (data, target) in enumerate(progress):

            data, target = [Variable(d) for d in data], [Variable(t) for t in target]
            if args.cuda and args.number_gpus == 1:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]

            optimizer.zero_grad() if not is_validate else None
            losses = model(data[0], target[0])
            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0] # Collect first loss for weight update
            total_loss += loss_val.data[0]
            loss_values = [v.data[0] for v in losses]

            # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
            loss_labels = list(model.module.loss.loss_labels)

            assert not np.isnan(total_loss)

            if not is_validate and args.fp16:
                loss_val.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)

                params = list(model.parameters())
                for i in range(len(params)):
                   param_copy[i].grad = params[i].grad.clone().type_as(params[i]).detach()
                   param_copy[i].grad.mul_(1./args.loss_scale)
                optimizer.step()
                for i in range(len(params)):
                    params[i].data.copy_(param_copy[i].data)

            elif not is_validate:
                loss_val.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)
                optimizer.step()

            # Update hyperparameters if needed
            global_iteration = start_iteration + batch_idx
            if not is_validate:
                tools.update_hyperparameter_schedule(args, epoch, global_iteration, optimizer)
                loss_labels.append('lr')
                loss_values.append(optimizer.param_groups[0]['lr'])

            loss_labels.append('load')
            loss_values.append(progress.iterable.last_duration)

            # Print out statistics
            statistics.append(loss_values)
            title = '{} Epoch {}'.format('Validating' if is_validate else 'Training', epoch)

            progress.set_description(title + ' ' + tools.format_dictionary_of_losses(loss_labels, statistics[-1]))

            if ((((global_iteration + 1) % args.log_frequency) == 0 and not is_validate) or
                (is_validate and batch_idx == args.validation_n_batches - 1)):

                global_iteration = global_iteration if not is_validate else start_iteration

                logger.add_scalar('batch logs per second', len(statistics) / (progress._time() - last_log_time), global_iteration)
                last_log_time = progress._time()

                all_losses = np.array(statistics)

                for i, key in enumerate(loss_labels):
                    logger.add_scalar('average batch ' + str(key), all_losses[:, i].mean(), global_iteration)
                    logger.add_histogram(str(key), all_losses[:, i], global_iteration)

            # Reset Summary
            statistics = []

            if ( is_validate and ( batch_idx == args.validation_n_batches) ):
                break

            if ( (not is_validate) and (batch_idx == (args.train_n_batches)) ):
                break

        progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)

    # Reusable function for inference
    def inference(args, epoch, data_loader, model, offset=0):

        model.eval()
        
        if args.save_flow or args.render_validation:
            flow_folder = "{}/inference/{}.epoch-{}-flow-field".format(args.save,args.name.replace('/', '.'),epoch)
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)

        
        args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches), desc='Inferencing ', 
            leave=True, position=offset)

        statistics = []
        total_loss = 0
        for batch_idx, (data, target) in enumerate(progress):
            if args.cuda:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
            data, target = [Variable(d) for d in data], [Variable(t) for t in target]

            # when ground-truth flows are not available for inference_dataset, 
            # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows, 
            # depending on the type of loss norm passed in
            with torch.no_grad():
                losses, output = model(data[0], target[0], inference=True)

            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0] # Collect first loss for weight update
            total_loss += loss_val.data[0]
            loss_values = [v.data[0] for v in losses]

            # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
            loss_labels = list(model.module.loss.loss_labels)

            statistics.append(loss_values)
            # import IPython; IPython.embed()
            if args.save_flow or args.render_validation:
                for i in range(args.inference_batch_size):
                    _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                    flow_utils.writeFlow( join(flow_folder, '%06d.flo'%(batch_idx * args.inference_batch_size + i)),  _pflow)

            progress.set_description('Inference Averages for Epoch {}: '.format(epoch) + tools.format_dictionary_of_losses(loss_labels, np.array(statistics).mean(axis=0)))
            progress.update(1)

            if batch_idx == (args.inference_n_batches - 1):
                break

        progress.close()

        return

    # Primary epoch loop
    best_err = 1e8
    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100, desc='Overall Progress', leave=True, position=0)
    offset = 1
    last_epoch_time = progress._time()
    global_iteration = 0

    for epoch in progress:
        if args.inference or (args.render_validation and ((epoch - 1) % args.validation_frequency) == 0):
            stats = inference(args=args, epoch=epoch - 1, data_loader=inference_loader, model=model_and_loss, offset=offset)
            offset += 1

        if not args.skip_validation and ((epoch - 1) % args.validation_frequency) == 0:
            validation_loss, _ = train(args=args, epoch=epoch - 1, start_iteration=global_iteration, data_loader=validation_loader, model=model_and_loss, optimizer=optimizer, logger=validation_logger, is_validate=True, offset=offset)
            offset += 1

            is_best = False
            if validation_loss < best_err:
                best_err = validation_loss
                is_best = True

            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
            tools.save_checkpoint({   'arch' : args.model,
                                      'epoch': epoch,
                                      'state_dict': model_and_loss.module.model.state_dict(),
                                      'best_EPE': best_err}, 
                                      is_best, args.save, args.model)
            checkpoint_progress.update(1)
            checkpoint_progress.close()
            offset += 1

        if not args.skip_training:
            train_loss, iterations = train(args=args, epoch=epoch, start_iteration=global_iteration, data_loader=train_loader, model=model_and_loss, optimizer=optimizer, logger=train_logger, offset=offset)
            global_iteration += iterations
            offset += 1

            # save checkpoint after every validation_frequency number of epochs
            if ((epoch - 1) % args.validation_frequency) == 0:
                checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
                tools.save_checkpoint({   'arch' : args.model,
                                          'epoch': epoch,
                                          'state_dict': model_and_loss.module.model.state_dict(),
                                          'best_EPE': train_loss}, 
                                          False, args.save, args.model, filename = 'train-checkpoint.pth.tar')
                checkpoint_progress.update(1)
                checkpoint_progress.close()


        train_logger.add_scalar('seconds per epoch', progress._time() - last_epoch_time, epoch)
        last_epoch_time = progress._time()
    print("\n")

import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import utils.frame_utils as frame_utils

from scipy.misc import imread, imresize

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)

class FlyingChairs(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/FlyingChairs_release/data', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.ppm') ) )

    self.flow_list = sorted( glob( join(root, '*.flo') ) )

    assert (len(images)//2 == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = images[2*i]
        im2 = images[2*i + 1]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThings(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs = sorted(glob(join(root, dstype, 'TRAIN/*/*')))
    image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

    flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/TRAIN/*/*')))
    flow_dirs = sorted([join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

    assert (len(image_dirs) == len(flow_dirs))

    self.image_list = []
    self.flow_list = []

    for idir, fdir in zip(image_dirs, flow_dirs):
        images = sorted( glob(join(idir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.flo')) )
        for i in range(len(flows)):
            self.image_list += [ [ images[i], images[i+1] ] ]
            self.flow_list += [flows[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates)

class ChairsSDHom(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/chairssdhom/data', dstype = 'train', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.flo') ) )

    assert (len(image1) == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = image1[i]
        im2 = image2[i]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = flow[::-1,:,:]

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'train', replicates = replicates)

class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTest, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'test', replicates = replicates)

class ImagesFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.' + iext) ) )
    self.image_list = []
    for i in range(len(images)-1):
        im1 = images[i]
        im2 = images[i+1]
        self.image_list += [ [ im1, im2 ] ]

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    
    images = np.array(images).transpose(3,0,1,2)
    images = torch.from_numpy(images.astype(np.float32))

    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

  def __len__(self):
    return self.size * self.replicates

'''
import argparse
import sys, os
import importlib
from scipy.misc import imsave
import numpy as np

import datasets
reload(datasets)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.inference_size = [1080, 1920]
args.crop_size = [384, 512]
args.effective_batch_size = 1

index = 500
v_dataset = datasets.MpiSintelClean(args, True, root='../MPI-Sintel/flow/training')
a, b = v_dataset[index]
im1 = a[0].numpy()[:,0,:,:].transpose(1,2,0)
im2 = a[0].numpy()[:,1,:,:].transpose(1,2,0)
imsave('./img1.png', im1)
imsave('./img2.png', im2)
flow_utils.writeFlow('./flow.flo', b[0].numpy().transpose(1,2,0))

'''

#!/usr/bin/env python2.7

import caffe
from caffe.proto import caffe_pb2
import sys, os

import torch
import torch.nn as nn

import argparse, tempfile
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('caffe_model', help='input model in hdf5 or caffemodel format')
parser.add_argument('prototxt_template',help='prototxt template')
parser.add_argument('flownet2_pytorch', help='path to flownet2-pytorch')

args = parser.parse_args()

args.rgb_max = 255
args.fp16 = False
args.grads = {}

# load models
sys.path.append(args.flownet2_pytorch)

import models
from utils.param_utils import *

width = 256
height = 256
keys = {'TARGET_WIDTH': width, 
        'TARGET_HEIGHT': height,
        'ADAPTED_WIDTH':width,
        'ADAPTED_HEIGHT':height,
        'SCALE_WIDTH':1.,
        'SCALE_HEIGHT':1.,}

template = '\n'.join(np.loadtxt(args.prototxt_template, dtype=str, delimiter='\n'))
for k in keys:
    template = template.replace('$%s$'%(k),str(keys[k]))

prototxt = tempfile.NamedTemporaryFile(mode='w', delete=True)
prototxt.write(template)
prototxt.flush()

net = caffe.Net(prototxt.name, args.caffe_model, caffe.TEST)

weights = {}
biases = {}

for k, v in list(net.params.items()):
    weights[k] = np.array(v[0].data).reshape(v[0].data.shape)
    biases[k] = np.array(v[1].data).reshape(v[1].data.shape)
    print((k, weights[k].shape, biases[k].shape))

if 'FlowNet2/' in args.caffe_model:
    model = models.FlowNet2(args)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')
    parse_flownetsd(model.flownets_d.modules(), weights, biases, param_prefix='netsd_')
    parse_flownetfusion(model.flownetfusion.modules(), weights, biases, param_prefix='fuse_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2_checkpoint.pth.tar'))

elif 'FlowNet2-C/' in args.caffe_model:
    model = models.FlowNet2C(args)

    parse_flownetc(model.modules(), weights, biases)
    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-C_checkpoint.pth.tar'))

elif 'FlowNet2-CS/' in args.caffe_model:
    model = models.FlowNet2CS(args)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-CS_checkpoint.pth.tar'))

elif 'FlowNet2-CSS/' in args.caffe_model:
    model = models.FlowNet2CSS(args)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-CSS_checkpoint.pth.tar'))

elif 'FlowNet2-CSS-ft-sd/' in args.caffe_model:
    model = models.FlowNet2CSS(args)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-CSS-ft-sd_checkpoint.pth.tar'))

elif 'FlowNet2-S/' in args.caffe_model:
    model = models.FlowNet2S(args)

    parse_flownetsonly(model.modules(), weights, biases, param_prefix='')
    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-S_checkpoint.pth.tar'))

elif 'FlowNet2-SD/' in args.caffe_model:
    model = models.FlowNet2SD(args)

    parse_flownetsd(model.modules(), weights, biases, param_prefix='')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(args.flownet2_pytorch, 'FlowNet2-SD_checkpoint.pth.tar'))

else:
    print(('model type cound not be determined from input caffe model %s'%(args.caffe_model)))
    quit()
print(("done converting ", args.caffe_model))

from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import resample2d_cuda

class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()

        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size

        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()

        resample2d_cuda.forward(input1, input2, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()

        input1, input2 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())

        resample2d_cuda.backward(input1, input2, grad_output.data,
                                 grad_input1.data, grad_input2.data,
                                 ctx.kernel_size)

        return grad_input1, grad_input2, None

class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size)


#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='resample2d_cuda',
    ext_modules=[
        CUDAExtension('resample2d_cuda', [
            'resample2d_cuda.cc',
            'resample2d_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .submodules import *
'Parameter count = 581,226'

class FlowNetFusion(nn.Module):
    def __init__(self,args, batchNorm=True):
        super(FlowNetFusion,self).__init__()

        self.batchNorm = batchNorm
        self.conv0   = conv(self.batchNorm,  11,   64)
        self.conv1   = conv(self.batchNorm,  64,   64, stride=2)
        self.conv1_1 = conv(self.batchNorm,  64,   128)
        self.conv2   = conv(self.batchNorm,  128,  128, stride=2)
        self.conv2_1 = conv(self.batchNorm,  128,  128)

        self.deconv1 = deconv(128,32)
        self.deconv0 = deconv(162,16)

        self.inter_conv1 = i_conv(self.batchNorm,  162,   32)
        self.inter_conv0 = i_conv(self.batchNorm,  82,   16)

        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)

        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        flow2       = self.predict_flow2(out_conv2)
        flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        
        concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1       = self.predict_flow1(out_interconv1)
        flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        
        concat0 = torch.cat((out_conv0,out_deconv0,flow1_up),1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0       = self.predict_flow0(out_interconv0)

        return flow0


from torch.autograd import Function, Variable
from torch.nn.modules.module import Module
import channelnorm_cuda

class ChannelNormFunction(Function):

    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        assert input1.is_contiguous()
        b, _, h, w = input1.size()
        output = input1.new(b, 1, h, w).zero_()

        channelnorm_cuda.forward(input1, output, norm_deg)
        ctx.save_for_backward(input1, output)
        ctx.norm_deg = norm_deg

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, output = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())

        channelnorm.backward(input1, output, grad_output.data,
                                              grad_input1.data, ctx.norm_deg)

        return grad_input1, None


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)



#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']

nvcc_args = [
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='channelnorm_cuda',
    ext_modules=[
        CUDAExtension('channelnorm_cuda', [
            'channelnorm_cuda.cc',
            'channelnorm_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda

class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply
        # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, 
                self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return output

    def backward(self, grad_output):
        input1, input2 = self.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return grad_input1, grad_input2


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)(input1, input2)

        return result


#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


'''
Portions of this code copyright 2017, Clement Pinard
'''

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .submodules import *
'Parameter count : 38,676,504 '

class FlowNetS(nn.Module):
    def __init__(self, args, input_channels = 12, batchNorm=True):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,  input_channels,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2,


import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .submodules import *
'Parameter count = 45,371,666'

class FlowNetSD(nn.Module):
    def __init__(self, args, batchNorm=True):
        super(FlowNetSD,self).__init__()

        self.batchNorm = batchNorm
        self.conv0   = conv(self.batchNorm,  6,   64)
        self.conv1   = conv(self.batchNorm,  64,   64, stride=2)
        self.conv1_1 = conv(self.batchNorm,  64,   128)
        self.conv2   = conv(self.batchNorm,  128,  128, stride=2)
        self.conv2_1 = conv(self.batchNorm,  128,  128)
        self.conv3   = conv(self.batchNorm, 128,  256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.inter_conv5 = i_conv(self.batchNorm,  1026,   512)
        self.inter_conv4 = i_conv(self.batchNorm,  770,   256)
        self.inter_conv3 = i_conv(self.batchNorm,  386,   128)
        self.inter_conv2 = i_conv(self.batchNorm,  194,   64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')



    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5       = self.predict_flow5(out_interconv5)

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4       = self.predict_flow4(out_interconv4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3       = self.predict_flow3(out_interconv3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2,

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .correlation_package.correlation import Correlation

from .submodules import *
'Parameter count , 39,175,298 '

class FlowNetC(nn.Module):
    def __init__(self, args, batchNorm=True, div_flow = 20):
        super(FlowNetC,self).__init__()
        self.fp16 = args.fp16
        self.batchNorm = batchNorm
        self.div_flow = div_flow

        self.conv1   = conv(self.batchNorm,   3,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv_redir  = conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        """if args.fp16:
            self.corr = nn.Sequential(
                tofp32(),
                Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1),
                tofp16())
        else:"""
        self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)

        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = x[:,0:3,:,:]
        x2 = x[:,3::,:,:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        if self.fp16:                        
            out_corr = self.corr(out_conv3a.float(), out_conv3b.float()).half() # False            
        else:
            out_corr = self.corr(out_conv3a, out_conv3b) # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)

        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)

        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1,out_deconv3,flow4_up),1)

        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2,

# freda (todo) : 

import torch.nn as nn
import torch
import numpy as np 

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i,j,:,:] = torch.from_numpy(bilinear)


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook

'''
def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook
import torch
from channelnorm_package.modules.channelnorm import ChannelNorm 
model = ChannelNorm().cuda()
grads = {}
a = 100*torch.autograd.Variable(torch.randn((1,3,5,5)).cuda(), requires_grad=True)
a.register_hook(save_grad(grads, 'a'))
b = model(a)
y = torch.mean(b)
y.backward()

'''

import numpy as np
from os.path import *
from scipy.misc import imread
from . import flow_utils 

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []

import torch
import torch.nn as nn
import numpy as np

def parse_flownetc(modules, weights, biases):
    keys = [
    'conv1',
    'conv2',
    'conv3',
    'conv_redir',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
    
    'deconv5',
    'deconv4',
    'deconv3',
    'deconv2',
    
    'Convolution1',
    'Convolution2',
    'Convolution3',
    'Convolution4',
    'Convolution5',

    'upsample_flow6to5',
    'upsample_flow5to4',
    'upsample_flow4to3',
    'upsample_flow3to2',
    
    ]
    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == 'conv1':
                m.weight.data[:,:,:,:] = torch.from_numpy(np.flip(weight, axis=1).copy())
                m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                m.bias.data[:] = torch.from_numpy(bias)                    

            i = i + 1
    return

def parse_flownets(modules, weights, biases, param_prefix='net2_'):
    keys = [
    'conv1',
    'conv2',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
    
    'deconv5',
    'deconv4',
    'deconv3',
    'deconv2',
    
    'predict_conv6',
    'predict_conv5',
    'predict_conv4',
    'predict_conv3',
    'predict_conv2',

    'upsample_flow6to5',
    'upsample_flow5to4',
    'upsample_flow4to3',
    'upsample_flow3to2',
    ]
    for i, k in enumerate(keys):
        if 'upsample' in k:
            keys[i] = param_prefix + param_prefix + k
        else:
            keys[i] = param_prefix + k
    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == param_prefix+'conv1':
                m.weight.data[:,0:3,:,:] = torch.from_numpy(np.flip(weight[:,0:3,:,:], axis=1).copy())
                m.weight.data[:,3:6,:,:] = torch.from_numpy(np.flip(weight[:,3:6,:,:], axis=1).copy())
                m.weight.data[:,6:9,:,:] = torch.from_numpy(np.flip(weight[:,6:9,:,:], axis=1).copy())
                m.weight.data[:,9::,:,:] = torch.from_numpy(weight[:,9:,:,:].copy())
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            i = i + 1
    return

def parse_flownetsonly(modules, weights, biases, param_prefix=''):
    keys = [
    'conv1',
    'conv2',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
    
    'deconv5',
    'deconv4',
    'deconv3',
    'deconv2',
    
    'Convolution1',
    'Convolution2',
    'Convolution3',
    'Convolution4',
    'Convolution5',

    'upsample_flow6to5',
    'upsample_flow5to4',
    'upsample_flow4to3',
    'upsample_flow3to2',
    ]
    for i, k in enumerate(keys):
        if 'upsample' in k:
            keys[i] = param_prefix + param_prefix + k
        else:
            keys[i] = param_prefix + k
    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == param_prefix+'conv1':
                # print ("%s :"%(keys[i]), m.weight.size(), m.bias.size(), tf_w[keys[i]].shape[::-1])
                m.weight.data[:,0:3,:,:] = torch.from_numpy(np.flip(weight[:,0:3,:,:], axis=1).copy())
                m.weight.data[:,3:6,:,:] = torch.from_numpy(np.flip(weight[:,3:6,:,:], axis=1).copy())
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            i = i + 1
    return

def parse_flownetsd(modules, weights, biases, param_prefix='netsd_'):
    keys = [
    'conv0',
    'conv1',
    'conv1_1',
    'conv2',
    'conv2_1',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
    
    'deconv5',
    'deconv4',
    'deconv3',
    'deconv2',

    'interconv5',
    'interconv4',
    'interconv3',
    'interconv2',
    
    'Convolution1',
    'Convolution2',
    'Convolution3',
    'Convolution4',
    'Convolution5',

    'upsample_flow6to5',
    'upsample_flow5to4',
    'upsample_flow4to3',
    'upsample_flow3to2',
    ]
    for i, k in enumerate(keys):
        keys[i] = param_prefix + k

    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == param_prefix+'conv0':
                m.weight.data[:,0:3,:,:] = torch.from_numpy(np.flip(weight[:,0:3,:,:], axis=1).copy())
                m.weight.data[:,3:6,:,:] = torch.from_numpy(np.flip(weight[:,3:6,:,:], axis=1).copy())
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            i = i + 1

    return

def parse_flownetfusion(modules, weights, biases, param_prefix='fuse_'):
    keys = [
    'conv0',
    'conv1',
    'conv1_1',
    'conv2',
    'conv2_1',

    'deconv1',
    'deconv0',

    'interconv1',
    'interconv0',
    
    '_Convolution5',
    '_Convolution6',
    '_Convolution7',

    'upsample_flow2to1',
    'upsample_flow1to0',
    ]
    for i, k in enumerate(keys):
        keys[i] = param_prefix + k

    i = 0
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight = weights[keys[i]].copy()
            bias = biases[keys[i]].copy()
            if keys[i] == param_prefix+'conv0':
                m.weight.data[:,0:3,:,:] = torch.from_numpy(np.flip(weight[:,0:3,:,:], axis=1).copy())
                m.weight.data[:,3::,:,:] = torch.from_numpy(weight[:,3:,:,:].copy())
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            else:
                m.weight.data[:,:,:,:] = torch.from_numpy(weight)
                if m.bias is not None:
                    m.bias.data[:] = torch.from_numpy(bias)
            i = i + 1

    return


import numpy as np

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

# freda (todo) : 

import os, time, sys, math
import subprocess, shutil
from os.path import *
import numpy as np
from inspect import isclass
from pytz import timezone
from datetime import datetime
import inspect
import torch

def datestr():
    pacific = timezone('US/Pacific')
    now = datetime.now(pacific)
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)

def module_to_dict(module, exclude=[]):
        return dict([(x, getattr(module, x)) for x in dir(module)
                     if isclass(getattr(module, x))
                     and x not in exclude
                     and getattr(module, x) not in exclude])

class TimerBlock: 
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")


    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print(("  [{:.3f}{}] {}".format(duration, units, string)))
    
    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n"%(string))
        fid.close()

def add_arguments_for_module(parser, module, argument_for_class, default, skip_params=[], parameter_defaults={}):
    argument_group = parser.add_argument_group(argument_for_class.capitalize())

    module_dict = module_to_dict(module)
    argument_group.add_argument('--' + argument_for_class, type=str, default=default, choices=list(module_dict.keys()))
    
    args, unknown_args = parser.parse_known_args()
    class_obj = module_dict[vars(args)[argument_for_class]]

    argspec = inspect.getargspec(class_obj.__init__)

    defaults = argspec.defaults[::-1] if argspec.defaults else None

    args = argspec.args[::-1]
    for i, arg in enumerate(args):
        cmd_arg = '{}_{}'.format(argument_for_class, arg)
        if arg not in skip_params + ['self', 'args']:
            if arg in list(parameter_defaults.keys()):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(parameter_defaults[arg]), default=parameter_defaults[arg])
            elif (defaults is not None and i < len(defaults)):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(defaults[i]), default=defaults[i])
            else:
                print(("[Warning]: non-default argument '{}' detected on class '{}'. This argument cannot be modified via the command line"
                        .format(arg, module.__class__.__name__)))
            # We don't have a good way of dealing with inferring the type of the argument
            # TODO: try creating a custom action and using ast's infer type?
            # else:
            #     argument_group.add_argument('--{}'.format(cmd_arg), required=True)

def kwargs_from_args(args, argument_for_class):
    argument_for_class = argument_for_class + '_'
    return {key[len(argument_for_class):]: value for key, value in list(vars(args).items()) if argument_for_class in key and key != argument_for_class + 'class'}

def format_dictionary_of_losses(labels, values):
    try:
        string = ', '.join([('{}: {:' + ('.3f' if value >= 0.001 else '.1e') +'}').format(name, value) for name, value in zip(labels, values)])
    except (TypeError, ValueError) as e:
        print((list(zip(labels, values))))
        string = '[Log Error] ' + str(e)

    return string


class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = next(self.iterator)
        self.last_duration = (time.time() - start)
        return n

    next = __next__

def gpumemusage():
    gpu_mem = subprocess.check_output("nvidia-smi | grep MiB | cut -f 3 -d '|'", shell=True).replace(' ', '').replace('\n', '').replace('i', '')
    all_stat = [float(a) for a in gpu_mem.replace('/','').split('MB')[:-1]]

    gpu_mem = ''
    for i in range(len(all_stat)/2):
        curr, tot = all_stat[2*i], all_stat[2*i+1]
        util = "%1.2f"%(100*curr/tot)+'%'
        cmem = str(int(math.ceil(curr/1024.)))+'GB'
        gmem = str(int(math.ceil(tot/1024.)))+'GB'
        gpu_mem += util + '--' + join(cmem, gmem) + ' '
    return gpu_mem


def update_hyperparameter_schedule(args, epoch, global_iteration, optimizer):
    if args.schedule_lr_frequency > 0:
        for param_group in optimizer.param_groups:
            if (global_iteration + 1) % args.schedule_lr_frequency == 0:
                param_group['lr'] /= float(args.schedule_lr_fraction)
                param_group['lr'] = float(np.maximum(param_group['lr'], 0.000001))

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .networks.resample2d_package.resample2d import Resample2d
from .networks.channelnorm_package.channelnorm import ChannelNorm

from .networks import FlowNetC
from .networks import FlowNetS
from .networks import FlowNetSD
from .networks import FlowNetFusion

from .networks.submodules import *
'Parameter count = 162,518,834'

class MyDict(dict):
    pass

class fp16_resample2d(nn.Module):
    def __init__(self):
        super(fp16_resample2d, self).__init__()
        self.resample = Resample2d()

    def forward(self, input1, input2):
        return self.resample(input1.float(), input2.float()).half()

class FlowNet2(nn.Module):

    def __init__(self, args=None, batchNorm=False, div_flow = 20., fp16=False):
        super(FlowNet2,self).__init__()
        if args is None:
            args = MyDict()
            args.rgb_max = 1
            args.fp16 = fp16
            args.grads = {}
        self.fp16 = fp16
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')        

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD.FlowNetSD(args, batchNorm=self.batchNorm) 
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest') 
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest') 
        
        self.resample = Resample2d() if not args.fp16 else fp16_resample2d()

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion.FlowNetFusion(args, batchNorm=self.batchNorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        heigh, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.)
        for i in range(min_dim):
            weight.data[i,i,:,:] = torch.from_numpy(bilinear)
        return 

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.resample(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1 
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow) 

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample(x[:,3:,:,:], flownets1_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)
        
        diff_flownets2_flow = self.resample(x[:,3:,:,:], flownets2_flow)
        # if not diff_flownets2_flow.volatile:
        #     diff_flownets2_flow.register_hook(save_grad(self.args.grads, 'diff_flownets2_flow'))

        diff_flownets2_img1 = self.channelnorm((x[:,:3,:,:]-diff_flownets2_flow))
        # if not diff_flownets2_img1.volatile:
        #     diff_flownets2_img1.register_hook(save_grad(self.args.grads, 'diff_flownets2_img1'))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        
        diff_flownetsd_flow = self.resample(x[:,3:,:,:], flownetsd_flow)
        # if not diff_flownetsd_flow.volatile:
        #     diff_flownetsd_flow.register_hook(save_grad(self.args.grads, 'diff_flownetsd_flow'))

        diff_flownetsd_img1 = self.channelnorm((x[:,:3,:,:]-diff_flownetsd_flow))
        # if not diff_flownetsd_img1.volatile:
        #     diff_flownetsd_img1.register_hook(save_grad(self.args.grads, 'diff_flownetsd_img1'))

        # concat img1 flownetsd, flownets2, norm_flownetsd, norm_flownets2, diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat((x[:,:3,:,:], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)

        # if not flownetfusion_flow.volatile:
        #     flownetfusion_flow.register_hook(save_grad(self.args.grads, 'flownetfusion_flow'))

        return flownetfusion_flow

class FlowNet2C(FlowNetC.FlowNetC):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2C,self).__init__(args, batchNorm=batchNorm, div_flow=20)
        self.rgb_max = args.rgb_max

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b) # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)

        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)

        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1,out_deconv3,flow4_up),1)

        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2S(FlowNetS.FlowNetS):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2S,self).__init__(args, input_channels = 6, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow
        
    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2SD(FlowNetSD.FlowNetSD):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2SD,self).__init__(args, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5       = self.predict_flow5(out_interconv5)

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4       = self.predict_flow4(out_interconv4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3       = self.predict_flow3(out_interconv3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2CS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2CS,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')        
        self.resample1 = Resample2d() if not args.fp16 else fp16_resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1 
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow) 

        return flownets1_flow

class FlowNet2CSS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2CSS,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample1 = Resample2d() if not args.fp16 else fp16_resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample2 = Resample2d() if not args.fp16 else fp16_resample2d()

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest') 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1 
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow) 

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:,3:,:,:], flownets1_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        return flownets2_flow


import re
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
from models.networks import BaseNetwork, get_norm_layer
from models.flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d


# The following class has been taken from https://github.com/NVlabs/SPADE and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.norm_nc = norm_nc
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 256

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # Make sure segmap has the same spatial size with input.
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


# The following class has been taken from https://github.com/NVlabs/SPADE and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class SPADEResnetBlock(nn.Module):
    def __init__(self, semantic_nc, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# The following class has been taken from https://github.com/clovaai/stargan-v2 and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


# The following class has been taken from https://github.com/clovaai/stargan-v2 and modified. 
# The use of this code is subject to the terms and conditions set forth by the original code's license.
class AdaINResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, opt):
        super().__init__()
        self.opt = opt
        self.actv = nn.LeakyReLU(0.2)
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        # apply spectral norm if specified
        if 'spectral' in self.opt.norm_G:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        return out


class FlowNetwork(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.opt = opt
        self.ngf = opt.ngf
        self.nl = nl
        # The flow application operator (warping function).
        self.resample = Resample2d()

        # Use average pool 2D to downsample predicted flow.
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

        # Encoder first layer
        enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                     norm_layer(nn.Conv2d(opt.output_nc+opt.input_nc,
                                          self.ngf,
                                          kernel_size=opt.initial_kernel_size,
                                          padding=0)),
                     activation]
        self.enc = [nn.Sequential(*enc_first)]

        # Encoder downsampling layers
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)

            # Conditional encoders
            enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                             self.ngf * mult_out,
                                             kernel_size=opt.kernel_size,
                                             stride=2,
                                             padding=1)),
                        activation]
            self.enc.append(nn.Sequential(*enc_down))
        self.enc = nn.ModuleList(self.enc)

        # Residual part of decoder
        fin = (base**self.nl) * self.ngf
        fout = fin
        self.dec_res = []
        for i in range(self.nl):
            self.dec_res.append(SPADEResnetBlock(opt.input_nc * opt.n_frames_G, fin, fout, opt))
        self.dec_res = nn.ModuleList(self.dec_res)

        # Upsampling part of decoder.
        self.dec_up = []
        self.dec_main = []
        for i in range(self.nl):
            fin = (base**(self.nl-i)) * self.ngf

            # In case of PixelShuffle, let it do the filters amount reduction.
            fout = (base**(self.nl-i-1)) * self.ngf if self.opt.no_pixelshuffle else fin
            if self.opt.no_pixelshuffle:
                self.dec_up.append(nn.Upsample(scale_factor=2))
            else:
                self.dec_up.append(nn.PixelShuffle(upscale_factor=2))
            self.dec_main.append(SPADEResnetBlock(opt.input_nc * opt.n_frames_G, fin, fout, opt))

        self.dec_up = nn.ModuleList(self.dec_up)
        self.dec_main = nn.ModuleList(self.dec_main)
        self.dec_flow = [nn.ReflectionPad2d(3),
                         nn.Conv2d(self.ngf, 2,
                                   kernel_size=opt.initial_kernel_size,
                                   padding=0)]
        self.dec_flow = nn.Sequential(*self.dec_flow)
        self.dec_mask = [nn.ReflectionPad2d(3),
                         nn.Conv2d(self.ngf, 1,
                                   kernel_size=opt.initial_kernel_size,
                                   padding=0)]
        self.dec_mask = nn.Sequential(*self.dec_mask)

    def forward(self, input, ref_input):
        # Get dimensions sizes
        NN_ref, _, H, W = ref_input.size()
        N = input.size()[0]
        N_ref = NN_ref // N

        # Repeat the conditional input for all reference frames
        seg = input.repeat(1, N_ref, 1, 1).view(NN_ref, -1, H, W)

        # Encode
        feats = []
        feat = ref_input
        for i in range(self.nl + 1):
            feat = self.enc[i](feat)
            feats.append(feat)

        # Decode
        for i in range(self.nl):
            feat = self.dec_res[i](feat, seg)
        for i in range(self.nl):
            feat = self.dec_main[i](feat, seg)
            feat = self.dec_up[i](feat)

        # Compute flow layer
        flow = self.dec_flow(feat)
        mask = self.dec_mask(feat)
        mask = (torch.tanh(mask) + 1) / 2
        flow = flow * mask
        down_flow = flow

        # Apply flow on features to match them spatially with the desired pose.
        flow_feats = []
        for i in range(self.nl + 1):
            flow_feats.append(self.resample(feats[i], down_flow))

            # Downsample flow and reduce its magnitude.
            down_flow = self.downsample(down_flow) / 2.0
        return flow, flow_feats, mask

class FramesEncoder(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.ngf = opt.ngf
        self.nl = nl
        cond_enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                          norm_layer(nn.Conv2d(opt.output_nc+opt.input_nc,
                                               self.ngf,
                                               kernel_size=opt.initial_kernel_size,
                                               padding=0)),
                          activation]
        self.cond_enc = [nn.Sequential(*cond_enc_first)]
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)
            cond_enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                                  self.ngf * mult_out,
                                                  kernel_size=opt.kernel_size,
                                                  stride=2,
                                                  padding=1)),
                             activation]
            self.cond_enc.append(nn.Sequential(*cond_enc_down))
        self.cond_enc = nn.ModuleList(self.cond_enc)

    def forward(self, ref_input):
        # Encode
        feats = []
        x_cond_enc = ref_input
        for i in range(self.nl + 1):
            x_cond_enc = self.cond_enc[i](x_cond_enc)
            feats.append(x_cond_enc)
        return feats


class RenderingNetwork(nn.Module):
    def __init__(self, opt, activation, norm_layer, nl, base):
        super().__init__()
        self.opt = opt
        self.ngf = opt.ngf
        self.naf = opt.naf
        self.nl = nl

        # Encode
        cond_enc_first = [nn.ReflectionPad2d(opt.initial_kernel_size // 2),
                          norm_layer(nn.Conv2d(opt.input_nc * opt.n_frames_G, self.ngf,
                                               kernel_size=opt.initial_kernel_size,
                                               padding=0)),
                          activation]
        self.cond_enc = [nn.Sequential(*cond_enc_first)]
        for i in range(self.nl):
            mult_in = base**i
            mult_out = base**(i+1)
            cond_enc_down = [norm_layer(nn.Conv2d(self.ngf * mult_in,
                                                  self.ngf * mult_out,
                                                  kernel_size=opt.kernel_size,
                                                  stride=2,
                                                  padding=1)),
                             activation]
            self.cond_enc.append(nn.Sequential(*cond_enc_down))
        self.cond_enc = nn.ModuleList(self.cond_enc)

        # Decode
        self.cond_dec = []
        if not self.opt.no_audio_input:
            self.cond_dec_audio = []
        self.cond_dec_up = []

        for i in range(self.nl):
            fin = (base**(self.nl-i)) * opt.ngf
            fout = (base**(self.nl-i-1)) * opt.ngf if opt.no_pixelshuffle else fin
            self.cond_dec.append(SPADEResnetBlock(fin, fin, fout, opt))
            if not self.opt.no_audio_input:
                self.cond_dec_audio.append(AdaINResnetBlock(fout, fout, self.naf, opt))
            if self.opt.no_pixelshuffle:
                self.cond_dec_up.append(nn.Upsample(scale_factor=2))
            else:
                self.cond_dec_up.append(nn.PixelShuffle(upscale_factor=2))

        self.cond_dec.append(SPADEResnetBlock(opt.ngf, opt.ngf, opt.ngf, opt))
        if not self.opt.no_audio_input:
            self.cond_dec_audio.append(AdaINResnetBlock(opt.ngf, opt.ngf, self.naf, opt))
        self.cond_dec.append(SPADEResnetBlock(opt.output_nc, opt.ngf, opt.ngf, opt))
        self.cond_dec = nn.ModuleList(self.cond_dec)
        if not self.opt.no_audio_input:
            self.cond_dec_audio = nn.ModuleList(self.cond_dec_audio)
        self.cond_dec_up = nn.ModuleList(self.cond_dec_up)

        self.conv_img = [nn.ReflectionPad2d(3), nn.Conv2d(self.ngf, 3, kernel_size=opt.initial_kernel_size, padding=0)]
        self.conv_img = nn.Sequential(*self.conv_img)

    def forward(self, input, feat_maps, warped, audio_feats):
        # Encode
        x_cond_enc = input
        for i in range(self.nl + 1):
            x_cond_enc = self.cond_enc[i](x_cond_enc)
        x = x_cond_enc

        # Decode
        for i in range(self.nl):
            x = self.cond_dec[i](x, feat_maps[-i-1])
            if not self.opt.no_audio_input:
                x = self.cond_dec_audio[i](x, audio_feats)
            x = self.cond_dec_up[i](x)

        x = self.cond_dec[self.nl](x, feat_maps[0])
        if not self.opt.no_audio_input:
            x = self.cond_dec_audio[self.nl](x, audio_feats)
        x = self.cond_dec[self.nl+1](x, warped)

        imgs = self.conv_img(F.leaky_relu(x, 2e-1))
        imgs = torch.tanh(imgs)
        return imgs

class headGANGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.resample = Resample2d()

        # Activation functions
        activation = nn.ReLU()
        leaky_activation = nn.LeakyReLU(2e-1)

        # Non-adaptive normalization layer
        norm_layer = get_norm_layer(opt, opt.norm_G_noadapt)

        # Number of times to (up/down)-sample spatial dimensions.

        nl = round(math.log(opt.crop_size // opt.down_size, 2))

        # If pixelshuffle is used, quadruple the number of filters when
        # upsampling, else simply double them.
        base = 2 if self.opt.no_pixelshuffle else 4

        if not self.opt.no_flownetwork:
            self.flow_network = FlowNetwork(opt, activation, norm_layer, nl, base)
        else:
            self.frames_encoder = FramesEncoder(opt, activation, norm_layer, nl, base)
        self.rendering_network = RenderingNetwork(opt, activation, norm_layer, nl, base)

    def forward(self, input, ref_input, audio_feats):
        if not self.opt.no_flownetwork:
            # Get flow and warped features.
            flow, flow_feats, mask = self.flow_network(input, ref_input)
        else:
            flow = torch.zeros_like(ref_input[:,:2,:,:])
            mask = torch.zeros_like(ref_input[:,:1,:,:])
            flow_feats = self.frames_encoder(ref_input)
        feat_maps = flow_feats

        # Apply flows on reference frame(s)
        ref_rgb_input = ref_input[:,-self.opt.output_nc:,:,:]
        if not self.opt.no_flownetwork:
            warped = self.resample(ref_rgb_input, flow)
        else:
            warped = ref_rgb_input
        imgs = self.rendering_network(input, feat_maps, warped, audio_feats)

        if self.opt.isTrain:
            return imgs, feat_maps, warped, flow, mask
        else:
            return imgs, warped, flow

import importlib
import torch
import util.util as util
import torch.nn as nn
from torch.nn import init
import torch.nn.utils.spectral_norm as spectral_norm

def get_norm_layer(opt, norm_type='spectralinstance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=True)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj
    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)
    return cls

def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.' + filename
    network = find_class_in_module(target_class_name, module_name)
    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network
    return network

def create_network(cls, opt, input_nc=None):
    net = cls(opt, input_nc) if input_nc else cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(opt.gpu_ids[0])
    net.init_weights(opt.init_type, opt.init_variance)
    return net

def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)

def define_D(opt, input_nc):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt, input_nc)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              #'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

# from modules.bn import InPlaceABNSync as BatchNorm2d

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        #self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,  nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    net = Resnet18()
    x = torch.randn(16, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    net.get_params()

# Notice: This code has been taken from https://github.com/zllrunning/face-parsing.PyTorch#Demo and modified.

from .model import BiSeNet
import torch

class Segmenter():
    def __init__(self, gpu_id=0):
        self.net = BiSeNet(n_classes=19)
        self.net.cuda(gpu_id)
        model_path='files/79999_iter.pth'
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def get_masks(self, imgs, size):
        with torch.no_grad():
            out = self.net(imgs)[0]
            mask = torch.argmax(out, dim=1)
            mask[mask > 0] = 1
            # Add channel dimesion and make double
            mask = mask.unsqueeze(1).double()
            mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear')
            return mask

    def join_masks(self, masks, ref_masks):
        mask_union = torch.clamp(masks + ref_masks, 0.0, 1.0)
        return mask_union

#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2

from transform import *



class FaceMask(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', *args, **kwargs):
        super(FaceMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.rootpth = rootpth

        self.imgs = os.listdir(os.path.join(self.rootpth, 'CelebA-HQ-img'))

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        img = Image.open(osp.join(self.rootpth, 'CelebA-HQ-img', impth))
        img = img.resize((512, 512), Image.BILINEAR)
        label = Image.open(osp.join(self.rootpth, 'mask', impth[:-3]+'png')).convert('P')
        # print(np.unique(np.array(label)))
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    face_data = '/home/zll/data/CelebAMask-HQ/CelebA-HQ-img'
    face_sep_mask = '/home/zll/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    mask_path = '/home/zll/data/CelebAMask-HQ/mask'
    counter = 0
    total = 0
    for i in range(15):
        # files = os.listdir(osp.join(face_sep_mask, str(i)))

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        for j in range(i*2000, (i+1)*2000):

            mask = np.zeros((512, 512))

            for l, att in enumerate(atts, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name)

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))
                    # print(np.unique(sep_mask))

                    mask[sep_mask == 225] = l
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            print(j)

    print(counter, total)















#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))







if __name__ == "__main__":
    setup_logger('./res')
    evaluate()

#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np

class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    )


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
            #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

            flip_lb = np.array(lb)
            flip_lb[lb == 2] = 3
            flip_lb[lb == 3] = 2
            flip_lb[lb == 4] = 5
            flip_lb[lb == 5] = 4
            flip_lb[lb == 7] = 8
            flip_lb[lb == 8] = 7
            flip_lb = Image.fromarray(flip_lb)
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = flip_lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im = im,
                    lb = lb,
                )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb




if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')

#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import sys
import logging

import torch.distributed as dist


def setup_logger(logpth):
    logfile = 'BiSeNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())



#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .resnet import Resnet18
# from modules.bn import InPlaceABNSync as BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    net = BiSeNet(19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(16, 3, 640, 480).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)

    net.get_params()

import cv2
import os
import numpy as np
from skimage.filters import gaussian


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    # changed = cv2.resize(changed, (512, 512))
    return changed

#
# def lip(image, parsing, part=17, color=[230, 50, 20]):
#     b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
#     tar_color = np.zeros_like(image)
#     tar_color[:, :, 0] = b
#     tar_color[:, :, 1] = g
#     tar_color[:, :, 2] = r
#
#     image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
#     il, ia, ib = cv2.split(image_lab)
#
#     tar_lab = cv2.cvtColor(tar_color, cv2.COLOR_BGR2Lab)
#     tl, ta, tb = cv2.split(tar_lab)
#
#     image_lab[:, :, 0] = np.clip(il - np.mean(il) + tl, 0, 100)
#     image_lab[:, :, 1] = np.clip(ia - np.mean(ia) + ta, -127, 128)
#     image_lab[:, :, 2] = np.clip(ib - np.mean(ib) + tb, -127, 128)
#
#
#     changed = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)
#
#     if part == 17:
#         changed = sharpen(changed)
#
#     changed[parsing != part] = image[parsing != part]
#     # changed = cv2.resize(changed, (512, 512))
#     return changed


if __name__ == '__main__':
    # 1  face
    # 10 nose
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair
    num = 116
    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }
    image_path = '/home/zll/data/CelebAMask-HQ/test-img/{}.jpg'.format(num)
    parsing_path = 'res/test_res/{}.png'.format(num)

    image = cv2.imread(image_path)
    ori = image.copy()
    parsing = np.array(cv2.imread(parsing_path, 0))
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['hair'], table['upper_lip'], table['lower_lip']]
    # colors = [[20, 20, 200], [100, 100, 230], [100, 100, 230]]
    colors = [[100, 200, 100]]
    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)
    cv2.imwrite('res/makeup/116_ori.png', cv2.resize(ori, (512, 512)))
    cv2.imwrite('res/makeup/116_2.png', cv2.resize(image, (512, 512)))

    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    cv2.imshow('color', cv2.resize(image, (512, 512)))

    # cv2.imshow('image', ori)
    # cv2.imshow('color', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
















#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
from evaluate import evaluate
from optimizer import Optimizer
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse


respth = './res'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()


def train():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:33241',
                world_size = torch.cuda.device_count(),
                rank=args.local_rank
                )
    setup_logger(respth)

    # dataset
    n_classes = 19
    n_img_per_gpu = 16
    n_workers = 8
    cropsize = [448, 448]
    data_root = '/home/zll/data/CelebAMask-HQ/'

    ds = FaceMask(data_root, cropsize=cropsize, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)

    # model
    ignore_idx = -100
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank, ],
            output_device = args.local_rank
            )
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//16
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
            model = net.module,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = LossP(out, lb)
        loss2 = Loss2(out16, lb)
        loss3 = Loss3(out32, lb)
        loss = lossp + loss2 + loss3
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        #  print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
        if dist.get_rank() == 0:
            if (it+1) % 5000 == 0:
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state, './res/cp/{}_iter.pth'.format(it))
                evaluate(dspth='/home/zll/data/CelebAMask-HQ/test-img', cp='{}_iter.pth'.format(it))

    #  dump the final model
    save_pth = osp.join(respth, 'model_final_diss.pth')
    # net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    train()

#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import logging

logger = logging.getLogger()

class Optimizer(object):
    def __init__(self,
                model,
                lr0,
                momentum,
                wd,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power,
                *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        param_list = [
                {'params': wd_params},
                {'params': nowd_params, 'weight_decay': 0},
                {'params': lr_mul_wd_params, 'lr_mul': True},
                {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True}]
        self.optim = torch.optim.SGD(
                param_list,
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)


    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()


#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp
import os
import cv2
from transform import *
from PIL import Image

face_data = '/home/zll/data/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = '/home/zll/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/home/zll/data/CelebAMask-HQ/mask'
counter = 0
total = 0
for i in range(15):

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    for j in range(i * 2000, (i + 1) * 2000):

        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                # print(np.unique(sep_mask))

                mask[sep_mask == 225] = l
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
        print(j)

print(counter, total)
from collections import OrderedDict

import torch
import torch.nn as nn

from .bn import ABN


class DenseModule(nn.Module):
    def __init__(self, in_channels, growth, layers, bottleneck_factor=4, norm_act=ABN, dilation=1):
        super(DenseModule, self).__init__()
        self.in_channels = in_channels
        self.growth = growth
        self.layers = layers

        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        for i in range(self.layers):
            self.convs1.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(in_channels)),
                ("conv", nn.Conv2d(in_channels, self.growth * bottleneck_factor, 1, bias=False))
            ])))
            self.convs3.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.growth * bottleneck_factor)),
                ("conv", nn.Conv2d(self.growth * bottleneck_factor, self.growth, 3, padding=dilation, bias=False,
                                   dilation=dilation))
            ])))
            in_channels += self.growth

    @property
    def out_channels(self):
        return self.in_channels + self.growth * self.layers

    def forward(self, x):
        inputs = [x]
        for i in range(self.layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs1[i](x)
            x = self.convs3[i](x)
            inputs += [x]

        return torch.cat(inputs, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as functional

from models._util import try_index
from .bn import ABN


class DeeplabV3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=256,
                 dilations=(12, 24, 36),
                 norm_act=ABN,
                 pooling_size=None):
        super(DeeplabV3, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[0], padding=dilations[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[1], padding=dilations[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[2], padding=dilations[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.reset_parameters(self.map_bn.activation, self.map_bn.slope)

    def reset_parameters(self, activation, slope):
        gain = nn.init.calculate_gain(activation, slope)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ABN):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = functional.avg_pool2d(x, pooling_size, stride=1)
            pool = functional.pad(pool, pad=padding, mode="replicate")
        return pool

from collections import OrderedDict

import torch.nn as nn

from .bn import ABN


class IdentityResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=ABN,
                 dropout=None):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    groups=groups, dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out

from .bn import ABN, InPlaceABN, InPlaceABNSync
from .functions import ACT_RELU, ACT_LEAKY_RELU, ACT_ELU, ACT_NONE
from .misc import GlobalAvgPool2d, SingleGPU
from .residual import IdentityResidualBlock
from .dense import DenseModule

import torch.nn as nn
import torch
import torch.distributed as dist

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)

class SingleGPU(nn.Module):
    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module=module

    def forward(self, input):
        return self.module(input.cuda(non_blocking=True))


import torch
import torch.nn as nn
import torch.nn.functional as functional

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from .functions import *


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                  self.training, self.momentum, self.eps)

        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class InPlaceABN(ABN):
    """InPlace Activated Batch Normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABN, self).__init__(num_features, eps, momentum, affine, activation, slope)

    def forward(self, x):
        return inplace_abn(x, self.weight, self.bias, self.running_mean, self.running_var,
                           self.training, self.momentum, self.eps, self.activation, self.slope)


class InPlaceABNSync(ABN):
    """InPlace Activated Batch Normalization with cross-GPU synchronization
    This assumes that it will be replicated across GPUs using the same mechanism as in `nn.DistributedDataParallel`.
    """

    def forward(self, x):
        return inplace_abn_sync(x, self.weight, self.bias, self.running_mean, self.running_var,
                                   self.training, self.momentum, self.eps, self.activation, self.slope)

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)



from os import path
import torch 
import torch.distributed as dist
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

_src_path = path.join(path.dirname(path.abspath(__file__)), "src")
_backend = load(name="inplace_abn",
                extra_cflags=["-O3"],
                sources=[path.join(_src_path, f) for f in [
                    "inplace_abn.cpp",
                    "inplace_abn_cpu.cpp",
                    "inplace_abn_cuda.cu",
                    "inplace_abn_cuda_half.cu"
                ]],
                extra_cuda_cflags=["--expt-extended-lambda"])

# Activation names
ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError("CUDA Error encountered in {}".format(fn))


def _broadcast_shape(x):
    out_size = []
    for i, s in enumerate(x.size()):
        if i != 1:
            out_size.append(1)
        else:
            out_size.append(s)
    return out_size


def _reduce(x):
    if len(x.size()) == 2:
        return x.sum(dim=0)
    else:
        n, c = x.size()[0:2]
        return x.contiguous().view((n, c, -1)).sum(2).sum(0)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_forward(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_forward(x)
    elif ctx.activation == ACT_NONE:
        pass


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_backward(x, dx, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_backward(x, dx)
    elif ctx.activation == ACT_NONE:
        pass


class InPlaceABN(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)

        if ctx.training:
            mean, var = _backend.mean_var(x)

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * count / (count - 1))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward + activation
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
        else:
            # TODO: implement simplified CUDA backward for inference mode
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))

        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz * weight.sign() if ctx.affine else None
        dbias = edz if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None

class InPlaceABNSync(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01, equal_batches=True):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1

        #count = _count_samples(x)
        batch_size = x.new_tensor([x.shape[0]],dtype=torch.long)

        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)

        if ctx.training:
            mean, var = _backend.mean_var(x)
            if ctx.world_size>1:
                # get global batch size
                if equal_batches:
                    batch_size *= ctx.world_size
                else:
                    dist.all_reduce(batch_size, dist.ReduceOp.SUM)

                ctx.factor = x.shape[0]/float(batch_size.item())

                mean_all = mean.clone() * ctx.factor
                dist.all_reduce(mean_all, dist.ReduceOp.SUM)

                var_all = (var + (mean - mean_all) ** 2) * ctx.factor
                dist.all_reduce(var_all, dist.ReduceOp.SUM)

                mean = mean_all
                var = var_all

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            count = batch_size.item() * x.view(x.shape[0],x.shape[1],-1).shape[-1]
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * (float(count) / (count - 1)))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward + activation
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
            edz_local = edz.clone()
            eydz_local = eydz.clone()

            if ctx.world_size>1:
                edz *= ctx.factor
                dist.all_reduce(edz, dist.ReduceOp.SUM)

                eydz *= ctx.factor
                dist.all_reduce(eydz, dist.ReduceOp.SUM)
        else:
            edz_local = edz = dz.new_zeros(dz.size(1))
            eydz_local = eydz = dz.new_zeros(dz.size(1))

        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz_local * weight.sign() if ctx.affine else None
        dbias = edz_local if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None

inplace_abn = InPlaceABN.apply
inplace_abn_sync = InPlaceABNSync.apply

__all__ = ["inplace_abn", "inplace_abn_sync", "ACT_RELU", "ACT_LEAKY_RELU", "ACT_ELU", "ACT_NONE"]

#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks import BaseNetwork, get_norm_layer

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt, input_nc):
        super().__init__()
        self.opt = opt
        self.input_nc = input_nc
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, self.input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt, input_nc):
        super().__init__()
        self.opt = opt
        self.input_nc = input_nc
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.input_nc

        norm_layer = get_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models.networks as networks
import models.losses as losses
from models.base_model import BaseModel
import util.util as util
import torchvision


class headGANModelD(BaseModel):
    def name(self):
        return 'headGANModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gpu_ids = opt.gpu_ids
        self.n_frames_D = opt.n_frames_D
        self.output_nc = opt.output_nc
        self.input_nc = opt.input_nc

        # Image discriminator
        netD_input_nc = self.input_nc + self.output_nc
        self.netD = networks.define_D(opt, netD_input_nc)

        # Mouth discriminator
        if not opt.no_mouth_D:
            if not self.opt.no_audio_input:
                netDm_input_nc = opt.naf + self.output_nc
            else:
                netDm_input_nc = self.output_nc
            self.netDm = networks.define_D(opt, netDm_input_nc)

        # load networks
        if (opt.continue_train or opt.load_pretrain):
            self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)
            if not opt.no_mouth_D:
                self.load_network(self.netDm, 'Dm', opt.which_epoch, opt.load_pretrain)
            print('---------- Discriminators loaded -------------')
        else:
            print('---------- Discriminators initialized -------------')

        # set loss functions and optimizers
        self.old_lr = opt.lr
        self.criterionGAN = losses.GANLoss(opt.gan_mode, tensor=self.Tensor, opt=self.opt)
        self.criterionL1 = torch.nn.L1Loss()
        if not opt.no_vgg_loss:
            self.criterionVGG = losses.VGGLoss(self.opt.gpu_ids[0])
        if not opt.no_maskedL1_loss:
            self.criterionMaskedL1 = losses.MaskedL1Loss()

        self.loss_names = ['G_VGG', 'G_GAN', 'G_GAN_Feat', 'G_MaskedL1', 'D_real', 'D_generated']
        if not self.opt.no_flownetwork:
            self.loss_names += ['G_VGG_w', 'G_MaskedL1_w', 'G_L1_mask']
        if not opt.no_mouth_D:
            self.loss_names += ['Gm_GAN', 'Gm_GAN_Feat', 'Dm_real', 'Dm_generated']

        beta1, beta2 = opt.beta1, opt.beta2
        lr = opt.lr
        if opt.no_TTUR:
            D_lr = lr
        else:
            D_lr = lr * 2

        # initialize optimizers
        params = list(self.netD.parameters())
        if not opt.no_mouth_D:
            params += list(self.netDm.parameters())
        self.optimizer_D = torch.optim.Adam(params, lr=D_lr, betas=(beta1, beta2))

    def compute_D_losses(self, netD, real_A, real_B, generated_B):
        # Input
        if real_A is not None:
            real_AB = torch.cat((real_A, real_B), dim=1)
            generated_AB = torch.cat((real_A, generated_B), dim=1)
        else:
            real_AB = real_B
            generated_AB = generated_B
        # D losses
        pred_real = netD.forward(real_AB)
        pred_generated = netD.forward(generated_AB.detach())
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        loss_D_generated = self.criterionGAN(pred_generated, False, for_discriminator=True)
        # G losses
        pred_generated = netD.forward(generated_AB)
        loss_G_GAN = self.criterionGAN(pred_generated, True, for_discriminator=False)
        loss_G_GAN_Feat = self.FM_loss(pred_real, pred_generated)
        return loss_D_real, loss_D_generated, loss_G_GAN, loss_G_GAN_Feat

    def FM_loss(self, pred_real, pred_generated):
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(min(len(pred_generated), self.opt.num_D)):
                for j in range(len(pred_generated[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionL1(pred_generated[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        else:
            loss_G_GAN_Feat = torch.zeros(self.bs, 1).cuda()
        return loss_G_GAN_Feat

    def forward(self, real_B, generated_B, warped_B, real_A, masks_B, masks, audio_feats, mouth_centers):
        lambda_feat = self.opt.lambda_feat
        lambda_vgg = self.opt.lambda_vgg
        lambda_maskedL1 = self.opt.lambda_maskedL1
        lambda_mask = self.opt.lambda_mask
        
        self.bs , _, self.height, self.width = real_B.size()

        # VGG loss
        loss_G_VGG = (self.criterionVGG(generated_B, real_B) * lambda_vgg) if not self.opt.no_vgg_loss else torch.zeros(self.bs, 1).cuda()
     
        # GAN and FM loss for Generator
        loss_D_real, loss_D_generated, loss_G_GAN, loss_G_GAN_Feat = self.compute_D_losses(self.netD, real_A, real_B, generated_B)
      
        loss_G_MaskedL1 = torch.zeros(self.bs, 1).cuda()
        if not self.opt.no_maskedL1_loss:
            loss_G_MaskedL1 = self.criterionMaskedL1(generated_B, real_B, real_A) * lambda_maskedL1

        loss_list = [loss_G_VGG, loss_G_GAN, loss_G_GAN_Feat, loss_G_MaskedL1, loss_D_real, loss_D_generated]
        
        # Warp Losses
        if not self.opt.no_flownetwork:
            loss_G_VGG_w = (self.criterionVGG(warped_B, real_B) * lambda_vgg) if not self.opt.no_vgg_loss else torch.zeros(self.bs, 1).cuda()
            loss_G_MaskedL1_w = torch.zeros(self.bs, 1).cuda()
            if not self.opt.no_maskedL1_loss:
                loss_G_MaskedL1_w = self.criterionMaskedL1(warped_B, real_B, real_A) * lambda_maskedL1
            loss_G_L1_mask = self.criterionL1(masks, masks_B.detach()) * lambda_mask
            loss_list += [loss_G_VGG_w, loss_G_MaskedL1_w, loss_G_L1_mask]

        # Mouth discriminator losses
        if not self.opt.no_mouth_D:
            # Extract mouth region around the center
            real_B_mouth, generated_B_mouth = util.get_ROI([real_B, generated_B], mouth_centers, self.opt)

            if not self.opt.no_audio_input:
                # Repeat audio features spatially for conditional input to mouth discriminator
                real_A_mouth = audio_feats[:, -self.opt.naf:].view(audio_feats.size(0), self.opt.naf, 1, 1)
                real_A_mouth = real_A_mouth.repeat(1, 1, real_B_mouth.size(2), real_B_mouth.size(3))
            else:
                real_A_mouth = None

            # Losses for mouth discriminator
            loss_Dm_real, loss_Dm_generated, loss_Gm_GAN, loss_Gm_GAN_Feat = self.compute_D_losses(self.netDm, real_A_mouth, real_B_mouth, generated_B_mouth)
            mouth_weight = 1
            loss_Gm_GAN *= mouth_weight
            loss_Gm_GAN_Feat *= mouth_weight
            loss_list += [loss_Gm_GAN, loss_Gm_GAN_Feat, loss_Dm_real, loss_Dm_generated]

        loss_list = [loss.unsqueeze(0) for loss in loss_list]
        return loss_list

    def save(self, label):
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        if not self.opt.no_mouth_D:
            self.save_network(self.netDm, 'Dm', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        if self.opt.niter_decay > 0:
            lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
            print('Update learning rate for D: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr


class headGANModelG(BaseModel):
    def name(self):
        return 'headGANModelG'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.n_frames_G = opt.n_frames_G
        self.output_nc = opt.output_nc
        self.input_nc = opt.input_nc

        self.netG = networks.define_G(opt)

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.load_pretrain)
            print('---------- Generator loaded -------------')
        else:
            print('---------- Generator initialized -------------')

        # Otimizer for G
        if self.isTrain:
            self.old_lr = opt.lr
            
            # initialize optimizer G
            paramsG = list(self.netG.parameters())
            beta1, beta2 = opt.beta1, opt.beta2
            lr = opt.lr
            if opt.no_TTUR:
                G_lr = lr
            else:
                G_lr = lr / 2
            self.optimizer_G = torch.optim.Adam(paramsG, lr=G_lr, betas=(beta1, beta2))

    def forward(self, input, ref_input, audio_feats):
        ret = self.netG.forward(input, ref_input, audio_feats)
        return ret

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        if self.opt.niter_decay > 0:
            lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            print('Update learning rate for G: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr


def create_model_G(opt):
    modelG = headGANModelG()
    modelG.initialize(opt)
    modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
    return modelG

