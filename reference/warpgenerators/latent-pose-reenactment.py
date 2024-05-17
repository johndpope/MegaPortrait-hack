import torch
from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        pass

    @staticmethod
    def get_net(args):
        net = Embedder()
        return net.to(args.device)


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

    def enable_finetuning(self, _=None):
        pass

    def get_identity_embedding(self, _):
        pass

    def get_pose_embedding(self, _):
        pass

    def forward(self, data_dict):
        self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

import logging # standard Python logging
logger = logging.getLogger('embedder')

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--average_function', type=str, default='sum', help='sum|max')

    @staticmethod
    def get_net(args):
        net = Embedder(
            args.embed_channels, args.average_function)
        return net.to(args.device)


class Embedder(nn.Module):
    def __init__(self, identity_embedding_size, average_function):
        super().__init__()

        self.identity_embedding_size = identity_embedding_size

        import torchvision
        self.identity_encoder = torchvision.models.resnext50_32x4d(num_classes=identity_embedding_size)
        
        import sys
        FABNET_ROOT_DIR = "embedders/FAb-Net"
        sys.path.append(f"{FABNET_ROOT_DIR}/FAb-Net/code/")
        try:
            from models_multiview import FrontaliseModelMasks_wider
            MODEL_PATH = f"{FABNET_ROOT_DIR}/FAb-Net/models/"
            classifier_model = MODEL_PATH + '/release/300w_4views.pt'
            checkpoint = torch.load(classifier_model, map_location='cpu')
        except (ImportError, FileNotFoundError):
            logger.critical(
                f"Please initialize submodules, then download FAb-Net models and put them "
                f"into {FABNET_ROOT_DIR}/FAb-Net/models/release/")
            raise

        self.pose_encoder = FrontaliseModelMasks_wider(3, inner_nc=256, num_additional_ids=32)
        self.pose_encoder.load_state_dict(checkpoint['state_dict_model'])
        self.pose_encoder = self.pose_encoder.encoder
        self.pose_encoder.eval()

        # Forbid doing .train(), .eval() and .parameters()
        def train_noop(self, mode=True): pass
        def parameters_noop(self, recurse=True): return []
        self.pose_encoder.train = train_noop.__get__(self.pose_encoder, nn.Module)
        self.pose_encoder.parameters = parameters_noop.__get__(self.pose_encoder, nn.Module)

        self.average_function = average_function

        self.finetuning = False

    def enable_finetuning(self, data_dict=None):
        self.finetuning = True

    def get_identity_embedding(self, data_dict):
        inputs = data_dict['enc_rgbs']

        batch_size, num_faces, c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        identity_embeddings = self.identity_encoder(inputs).view(batch_size, num_faces, -1)
        assert identity_embeddings.shape[2] == self.identity_embedding_size

        if self.average_function == 'sum':
            identity_embeddings_aggregated = identity_embeddings.mean(1)
        elif self.average_function == 'max':
            identity_embeddings_aggregated = identity_embeddings.max(1)[0]
        else:
            raise ValueError("Incorrect `average_function` argument, expected `sum` or `max`")

        data_dict['embeds'] = identity_embeddings_aggregated
        data_dict['embeds_elemwise'] = identity_embeddings

    def get_pose_embedding(self, data_dict):
        x = data_dict['pose_input_rgbs'][:, 0]
        with torch.no_grad():
            data_dict['pose_embedding'] = self.pose_encoder(x)[:, :, 0, 0]

    def forward(self, data_dict):
        if not self.finetuning:
            self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

import logging # standard Python logging
logger = logging.getLogger('embedder')

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--average_function', type=str, default='sum', help='sum|max')

    @staticmethod
    def get_net(args):
        net = Embedder(
            args.embed_channels, args.average_function)
        return net.to(args.device)


class Embedder(nn.Module):
    def __init__(self, identity_embedding_size, average_function):
        super().__init__()

        self.identity_embedding_size = identity_embedding_size

        import torchvision
        self.identity_encoder = torchvision.models.resnext50_32x4d(num_classes=identity_embedding_size)
        
        import sys
        X2FACE_ROOT_DIR = "embedders/X2Face"
        sys.path.append(f"{X2FACE_ROOT_DIR}/UnwrapMosaic/")
        try:
            from UnwrappedFace import UnwrappedFaceWeightedAverage
            state_dict = torch.load(
                f"{X2FACE_ROOT_DIR}/models/x2face_model_forpython3.pth", map_location='cpu')
        except (ImportError, FileNotFoundError):
            logger.critical(
                f"Please initialize submodules, then download 'x2face_model_forpython3.pth' from "
                f"http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_x2face_eccv_withpy3.zip"
                f" and put it into {X2FACE_ROOT_DIR}/models/")
            raise

        self.pose_encoder = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128, sampler_only=True)
        self.pose_encoder.load_state_dict(state_dict['state_dict'], strict=False)
        self.pose_encoder.eval()

        # Forbid doing .train(), .eval() and .parameters()
        def train_noop(self, mode=True): pass
        def parameters_noop(self, recurse=True): return []
        self.pose_encoder.train = train_noop.__get__(self.pose_encoder, nn.Module)
        self.pose_encoder.parameters = parameters_noop.__get__(self.pose_encoder, nn.Module)

        self.average_function = average_function

        self.finetuning = False

    def enable_finetuning(self, data_dict=None):
        self.finetuning = True

    def get_identity_embedding(self, data_dict):
        inputs = data_dict['enc_rgbs']

        batch_size, num_faces, c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        identity_embeddings = self.identity_encoder(inputs).view(batch_size, num_faces, -1)
        assert identity_embeddings.shape[2] == self.identity_embedding_size

        if self.average_function == 'sum':
            identity_embeddings_aggregated = identity_embeddings.mean(1)
        elif self.average_function == 'max':
            identity_embeddings_aggregated = identity_embeddings.max(1)[0]
        else:
            raise ValueError("Incorrect `average_function` argument, expected `sum` or `max`")

        data_dict['embeds'] = identity_embeddings_aggregated
        data_dict['embeds_elemwise'] = identity_embeddings

    def get_pose_embedding(self, data_dict):
        x = data_dict['pose_input_rgbs'][:, 0]
        with torch.no_grad():
            data_dict['pose_embedding'] = self.pose_encoder.get_sampler(x, latent_pose_vector_only=True)[:, :, 0, 0]

    def forward(self, data_dict):
        if not self.finetuning:
            self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--average_function', type=str, default='sum', help='sum|max')

    @staticmethod
    def get_net(args):
        net = Embedder(
            args.embed_channels, args.pose_embedding_size, args.average_function)
        return net.to(args.device)


class Embedder(nn.Module):
    def __init__(self, identity_embedding_size, pose_embedding_size, average_function):
        super().__init__()

        self.identity_embedding_size = identity_embedding_size
        self.pose_embedding_size = pose_embedding_size

        import torchvision
        self.identity_encoder = torchvision.models.resnext50_32x4d(num_classes=identity_embedding_size)
        self.pose_encoder = torchvision.models.mobilenet_v2(num_classes=pose_embedding_size)

        self.average_function = average_function

        self.finetuning = False

    def enable_finetuning(self, data_dict=None):
        self.finetuning = True

    def get_identity_embedding(self, data_dict):
        inputs = data_dict['enc_rgbs']

        batch_size, num_faces, c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        identity_embeddings = self.identity_encoder(inputs).view(batch_size, num_faces, -1)
        assert identity_embeddings.shape[2] == self.identity_embedding_size

        if self.average_function == 'sum':
            identity_embeddings_aggregated = identity_embeddings.mean(1)
        elif self.average_function == 'max':
            identity_embeddings_aggregated = identity_embeddings.max(1)[0]
        else:
            raise ValueError("Incorrect `average_function` argument, expected `sum` or `max`")

        data_dict['embeds'] = identity_embeddings_aggregated
        data_dict['embeds_elemwise'] = identity_embeddings

    def get_pose_embedding(self, data_dict):
        x = data_dict['pose_input_rgbs'][:, 0]
        data_dict['pose_embedding'] = self.pose_encoder(x)

    def forward(self, data_dict):
        if not self.finetuning:
            self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--embed_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--embed_num_blocks', type=int, default=6)
        parser.add('--average_function', type=str, default='sum', help='sum|max')

    @staticmethod
    def get_net(args):
        net = Embedder(
            args.embed_padding, args.in_channels, args.out_channels,
            args.num_channels, args.max_num_channels, args.embed_channels,
            args.embed_num_blocks, args.average_function)
        return net.to(args.device)

class Embedder(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, embed_channels,
                 embed_num_blocks, average_function):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer='none')

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d

        self.out_channels = embed_channels

        self.down_block = nn.Sequential(
            padding(1),
            spectral_norm(
                nn.Conv2d(in_channels + out_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.ReLU(),
            padding(1),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels + out_channels, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))

        layers = []
        in_channels = num_channels
        for i in range(1, embed_num_blocks - 1):
            out_channels = min(in_channels * 2, max_num_channels)
            layers.append(get_down_block(in_channels, out_channels, padding))
            in_channels = out_channels
        layers.append(get_down_block(out_channels, embed_channels, padding))
        self.down_blocks = nn.Sequential(*layers)

        self.average_function = average_function

        self.finetuning = False

    def enable_finetuning(self, data_dict=None):
        self.finetuning = True

    def get_identity_embedding(self, data_dict):
        enc_stickmen = data_dict['enc_stickmen']
        enc_rgbs = data_dict['enc_rgbs']

        inputs = torch.cat([enc_stickmen, enc_rgbs], 2)

        b, n, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)
        out = self.down_block(inputs)
        out = out + self.skip(inputs)
        out = self.down_blocks(out)
        out = torch.relu(out)
        embeds_elemwise = out.view(b, n, self.out_channels, -1).sum(3)

        if self.average_function == 'sum':
            embeds = embeds_elemwise.mean(1)
        elif self.average_function == 'max':
            embeds = embeds_elemwise.max(1)[0]
        else:
            raise Exception('Incorrect `average_function` argument, expected `sum` or `max`')

        data_dict['embeds'] = embeds
        data_dict['embeds_elemwise'] = embeds_elemwise

    def get_pose_embedding(self, data_dict):
        pass

    def forward(self, data_dict):
        if not self.finetuning:
            self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--embed_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--embed_num_blocks', type=int, default=6)
        parser.add('--average_function', type=str, default='sum', help='sum|max')

    @staticmethod
    def get_net(args):
        net = Embedder(
            args.embed_padding, args.in_channels, args.out_channels,
            args.num_channels, args.max_num_channels, args.embed_channels,
            args.embed_num_blocks, args.average_function)
        return net.to(args.device)

class Embedder(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, embed_channels,
                 embed_num_blocks, average_function):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer='none')

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d

        in_channels = 3
        self.out_channels = embed_channels

        self.down_block = nn.Sequential(
            padding(1),
            spectral_norm(
                nn.Conv2d(3, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.ReLU(),
            padding(1),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(3, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))

        layers = []
        in_channels = num_channels
        for i in range(1, embed_num_blocks - 1):
            out_channels = min(in_channels * 2, max_num_channels)
            layers.append(get_down_block(in_channels, out_channels, padding))
            in_channels = out_channels
        layers.append(get_down_block(out_channels, embed_channels, padding))
        self.down_blocks = nn.Sequential(*layers)

        self.average_function = average_function

        self.finetuning = False

    def enable_finetuning(self, data_dict=None):
        self.finetuning = True

    def get_identity_embedding(self, data_dict):
        enc_rgbs = data_dict['enc_rgbs']
        inputs = enc_rgbs

        b, n, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)
        out = self.down_block(inputs)
        out = out + self.skip(inputs)
        out = self.down_blocks(out)
        out = torch.relu(out)
        embeds_elemwise = out.view(b, n, self.out_channels, -1).sum(3)

        if self.average_function == 'sum':
            embeds = embeds_elemwise.mean(1)
        elif self.average_function == 'max':
            embeds = embeds_elemwise.max(1)[0]
        else:
            raise Exception('Incorrect `average_function` argument, expected `sum` or `max`')

        data_dict['embeds'] = embeds
        data_dict['embeds_elemwise'] = embeds_elemwise

    def get_pose_embedding(self, data_dict):
        pass

    def forward(self, data_dict):
        if not self.finetuning:
            self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)

"""
Compute metrics (pose error and identity error) as defined in the "Neural Head Reenactment
with Latent Pose Descriptors" paper.

When running as a script, the reenactment results should be first obtained by
'batched_finetune.py' followed by 'batched_drive.py'.

Usage:
    First, change (or adapt your directories to) these paths below:
        `model=...`, `DATASET_ROOT`, `RESULTS_ROOT`, `DESCRIPTORS_GT_FILE`, `LANDMARKS_GT_FILE`
    Then:
    python3 compute_pose_identity_error.py <model-name>

Example usage:
    python3 compute_pose_identity_error.py MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_02492466

Args:
    <model-name>:
        A directory name where reenactment results are stored, as e.g.
        './puppeteering/VoxCeleb2_30Test/<model-name>/id03839_LhI_8AWX_Mg_identity/driving-results/*.mp4'.
"""
import face_alignment

from tqdm import tqdm

import numpy as np
import cv2

from pathlib import Path
import sys

############### ArcFace ###############

FACE_DESCRIPTOR_DIM = 512

arcface_model = None

def get_default_bbox(kind):
    """
    Get default rough estimate of face bounding box for `get_identity_descriptor()`.

    Args:
        kind:
            `str`
            Crop type that your model's pose encoder consumes.
            One of: 'ffhq', 'x2face', 'latentpose'.

    Returns:
        bbox:
            `tuple` of `int`, length == 4
            The bounding box in pixels. Defines how much pixels to clip from top, left,
            bottom and right in a 256 x 256 image.
    """
    if kind == 'ffhq':
        return (0, 30, 60, 30)
    elif kind == 'x2face':
        return (37, (37+45)//2, 45, (37+45)//2)
    elif kind == 'latentpose':
        return (42, (42+64)//2, 64, (42+64)//2)
    else:
        raise ValueError(f"Wrong crop type: {kind}")

def get_identity_descriptor(images, default_bbox):
    """
    Compute an identity vector (by the ArcFace face recognition system) for each image in `images`.

    Args:
        images:
            iterable of `numpy.ndarray`, dtype == uint8, shape == (256, 256, 3)
            Images to compute identity descriptors for.
        default_bbox:
            `tuple` of `int`, length == 4
            See `get_default_bbox()`.

    Returns:
        descriptors:
            `numpy.ndarray`, dtype == float32, shape == (`len(images)`, `FACE_DESCRIPTOR_DIM`)
        num_bad_images:
            int
            For how many images face detection failed.
    """
    global arcface_model

    # Load the model if it hasn't been loaded yet
    if arcface_model is None:
        from insightface import face_model

        arcface_model = face_model.FaceModel(
            image_size='112,112',
            model="/Vol0/user/e.burkov/Projects/insightface/models/model-r100-ii/model,0000",
            ga_model="",
            det=0,
            flip=1,
            threshold=1.24,
            gpu=0)

    num_bad_images = 0
    images_cropped = []

    for image in images:
        image_cropped = arcface_model.get_input(image)
        if image_cropped is None: # no faces found
            num_bad_images += 1
            t, l, b, r = default_bbox
            image_cropped = cv2.resize(image[t:256-b, l:256-r], (112, 112), interpolation=cv2.INTER_CUBIC)
            image_cropped = image_cropped.transpose(2, 0, 1)

        images_cropped.append(image_cropped)

    return arcface_model.get_feature(np.stack(images_cropped)), num_bad_images


############## Landmark detector ###############

MEAN_FACE = np.array([
    [74.0374984741211, 115.65937805175781],
    [74.81562805175781, 130.58021545410156],
    [77.2906265258789, 143.63853454589844],
    [80.5406265258789, 156.11041259765625],
    [85.6812515258789, 170.04791259765625],
    [93.36354064941406, 181.28541564941406],
    [101.20833587646484, 188.8718719482422],
    [110.51457977294922, 195.19479370117188],
    [126.53229522705078, 199.7687530517578],
    [142.9031219482422, 194.9875030517578],
    [154.76771545410156, 187.64999389648438],
    [163.98646545410156, 179.6666717529297],
    [172.2624969482422, 167.578125],
    [177.1437530517578, 152.93020629882812],
    [179.59478759765625, 139.87396240234375],
    [181.76145935058594, 125.9468765258789],
    [182.359375, 110.66458129882812],
    [84.17292022705078, 101.70625305175781],
    [89.2249984741211, 97.9437484741211],
    [96.4124984741211, 96.10104370117188],
    [103.30208587646484, 96.92916870117188],
    [109.55416870117188, 98.98958587646484],
    [135.68959045410156, 98.4749984741211],
    [142.27499389648438, 96.1500015258789],
    [149.71978759765625, 94.640625],
    [158.04896545410156, 95.68020629882812],
    [164.90728759765625, 99.32499694824219],
    [122.91041564941406, 114.76145935058594],
    [122.50416564941406, 125.12395477294922],
    [122.07604217529297, 134.3125],
    [122.16354370117188, 142.02915954589844],
    [115.19271087646484, 146.9250030517578],
    [118.640625, 148.04270935058594],
    [123.62187194824219, 149.28125],
    [128.79896545410156, 147.8489532470703],
    [132.8333282470703, 146.4479217529297],
    [94.09166717529297, 113.77291870117188],
    [98.35832977294922, 111.75],
    [104.53020477294922, 111.42916870117188],
    [110.55937194824219, 114.43645477294922],
    [105.203125, 116.39167022705078],
    [98.70207977294922, 116.40520477294922],
    [137.22084045410156, 113.53020477294922],
    [143.1770782470703, 110.64583587646484],
    [149.63645935058594, 110.56145477294922],
    [154.83749389648438, 112.0625],
    [149.82186889648438, 115.09479522705078],
    [142.86146545410156, 115.31041717529297],
    [107.09062194824219, 165.00416564941406],
    [112.30104064941406, 161.16354370117188],
    [119.99166870117188, 158.30313110351562],
    [124.18228912353516, 159.046875],
    [128.3802032470703, 158.02708435058594],
    [137.22084045410156, 160.6906280517578],
    [144.14688110351562, 164.3625030517578],
    [137.1770782470703, 170.67604064941406],
    [131.06353759765625, 174.26145935058594],
    [124.75104522705078, 175.1281280517578],
    [118.46145629882812, 174.7604217529297],
    [113.23645782470703, 171.27499389648438],
    [108.41666412353516, 164.7708282470703],
    [119.25729370117188, 163.55624389648438],
    [124.46979522705078, 163.3625030517578],
    [129.99583435058594, 163.53854370117188],
    [142.75416564941406, 164.22604370117188],
    [130.0520782470703, 167.13958740234375],
    [124.57083129882812, 167.7864532470703],
    [119.16666412353516, 167.3072967529297]], dtype=np.float32)

landmark_detector = None

def get_landmarks(image):
    """
    Compute 68 facial landmarks (2D ones!) by the Bulat et al. 2017 system from `image`.

    Args:
        image:
            `numpy.ndarray`, dtype == uint8, shape == (H, W, 3)

    Returns:
        landmarks:
            `numpy.ndarray`, dtype == float32, shape == (68, 2)
        success:
            bool
            False if no faces were detected.
    """
    global landmark_detector

    # Load the landmark detector if it hasn't been loaded yet
    if landmark_detector is None:
        landmark_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    landmarks = landmark_detector.get_landmarks(image)
    try:
        return landmarks[0], True
    except TypeError: # zero faces detected
        return MEAN_FACE, False

########################## List identities ###############################

# The ones used in the paper
IDENTITIES = [
    "id00061/cAT9aR8oFx0",
    "id00061/Df_m1slf_hY",
    "id00812/XoAi2n4S2wo",
    "id01106/B08yOvYMF7Y",
    "id01228/7qHTvs0VO68",
    "id01333/9kgJaduwKkY",
    "id01437/4lFDvxXzYWY",
    "id02057/s5VqJY7DDEE",
    "id02548/x2LUQEUXdz4",
    "id03127/uiRiyK8Qlic",
    "id03178/cCoNRuzAL-A",
    "id03178/fnARFfUwf2s",
    "id03524/GkvScYvOJ7o",
    "id03839/LhI_8AWX_Mg",
    "id03839/PUwanP-C5qg",
    "id03862/fsCqKQb9Rdg",
    "id04094/JUYMzfVp8zI",
    "id04950/PQEAck-3wcA",
    "id05459/3TI6dVmEwzw",
    "id05714/wFGNufaMbDY",
    "id06104/7UnGAS5-jpU",
    "id06811/KmvEwL3fP9Q",
    "id07312/h1dszoDi1E8",
    "id07663/54qlJ2HZ08s",
    "id07802/BfQUBDw7TiM",
    "id07868/JC0QT4oXh2Y",
    "id07961/464OHFffwjI",
    "id07961/hROZwL8pbGg",
    "id08149/vxBFGKGXSFA",
    "id08701/UeUyLqpLz70",
]

NUM_VIDEO_FRAMES = 32

########################## Define metrics ###############################

def identity_error(gt_descriptors, our_descriptors):
    assert gt_descriptors.shape == (len(IDENTITIES), FACE_DESCRIPTOR_DIM)
    assert our_descriptors.shape == (len(IDENTITIES), len(IDENTITIES), NUM_VIDEO_FRAMES, FACE_DESCRIPTOR_DIM)

    cosine_distances = (gt_descriptors[:, None, None] * our_descriptors).sum(-1).astype(np.float64)
    # Don't include self-driving
    for driver_idx in range(len(IDENTITIES)):
        cosine_distances[driver_idx][driver_idx] = 0

    return 1.0 - cosine_distances.sum() / (len(IDENTITIES) * (len(IDENTITIES) - 1) * NUM_VIDEO_FRAMES)

def pose_reconstruction_error(gt_landmarks, our_landmarks, apply_optimal_alignment=False):
    assert gt_landmarks.shape == (len(IDENTITIES), NUM_VIDEO_FRAMES, 68, 2)
    assert our_landmarks.shape == gt_landmarks.shape

    if apply_optimal_alignment:
        # The 3 variables here are scale, shift_x, shift_y.
        # Find a transform that optimizes || scale * our_landmarks + shift - gt_landmarks ||^2.
        alignments = np.empty(gt_landmarks.shape[:2] + (3,), dtype=np.float32)

        all_lhs = np.empty(gt_landmarks.shape + (3,), dtype=np.float64)
        all_lhs[:, :, :, :, 0] = our_landmarks
        all_lhs[:, :, :, 0, 1:] = [1, 0]
        all_lhs[:, :, :, 1, 1:] = [0, 1]
        all_lhs = all_lhs.reshape(len(IDENTITIES), NUM_VIDEO_FRAMES, -1, 3)

        all_rhs = gt_landmarks.astype(np.float64).reshape(len(IDENTITIES), NUM_VIDEO_FRAMES, -1)

        for i in range(len(IDENTITIES)):
            for f in range(NUM_VIDEO_FRAMES):
                alignments[i, f] = np.linalg.lstsq(all_lhs[i, f], all_rhs[i, f], rcond=None)[0]

        scale = alignments[:, :, 0, None, None] # `None` for proper broadcasting
        shift = alignments[:, :, None, 1:]
        our_landmarks = our_landmarks * scale + shift

    interocular = np.linalg.norm(gt_landmarks[:, :, 36] - gt_landmarks[:, :, 45], axis=-1).clip(min=1e-2)
    normalized_distances = np.linalg.norm(gt_landmarks - our_landmarks, axis=-1) / interocular[:, :, None]
    return normalized_distances.mean()


########################## The script ###############################

if __name__ == "__main__":

    # Where the "ground truth" driver/identity images are
    DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/VoxCeleb2_30TestIdentities")

    MODEL = sys.argv[1]
    # Where are cross-driving/self-driving outputs are, and where to cache the computed descriptors and landmarks
    RESULTS_ROOT = Path(f"puppeteering/VoxCeleb2_30Test/{MODEL}")
    assert RESULTS_ROOT.is_dir()

    ############  GT ArcFace  ##############

    if MODEL.startswith("Zakharov_0"):
        crop_type = 'ffhq'
    elif MODEL.startswith("X2Face_vanilla"):
        crop_type = 'x2face'
    else:
        crop_type = 'latentpose'
    print(f"Assuming the crop type is '{crop_type}'")
    DEFAULT_BBOX = get_default_bbox(crop_type)

    erase_background = not ('noSegm' in MODEL or MODEL.startswith("Zakharov_0") or MODEL.startswith("X2Face_vanilla"))

    if erase_background:
        DESCRIPTORS_GT_FILE = Path("puppeteering/VoxCeleb2_30Test/true_average_identity_descriptors_noBackground.npy")
    else:
        DESCRIPTORS_GT_FILE = Path("puppeteering/VoxCeleb2_30Test/true_average_identity_descriptors.npy")

    COMPUTE_GT_DESCRIPTORS = False # hardcode to `True` if you want to recompute
    try:
        gt_average_descriptors = np.load(DESCRIPTORS_GT_FILE)
        print(f"Loaded the cached target descriptors from {DESCRIPTORS_GT_FILE}")
    except FileNotFoundError:
        print(f"Could not load the target descriptors from {DESCRIPTORS_GT_FILE}")
        gt_average_descriptors = np.empty((len(IDENTITIES), FACE_DESCRIPTOR_DIM), dtype=np.float32)
        COMPUTE_GT_DESCRIPTORS = True

    if COMPUTE_GT_DESCRIPTORS:
        print(f"Recomputing target descriptors into {DESCRIPTORS_GT_FILE}")
        for identity, gt_average_descriptor in zip(tqdm(IDENTITIES), gt_average_descriptors):
            images_folder       = DATASET_ROOT /       'images-cropped' / identity / 'identity'
            segmentation_folder = DATASET_ROOT / 'segmentation-cropped' / identity / 'identity'
            identity_images = []
            for image_path in images_folder.iterdir():
                image = cv2.imread(str(image_path))
                if erase_background:
                    segmentation = cv2.imread(str(segmentation_folder / image_path.with_suffix('.png').name))
                    image = cv2.multiply(image, segmentation, dst=image, scale=1/255)

                identity_images.append(image)

            gt_descriptors, num_bad_images = get_identity_descriptor(identity_images, DEFAULT_BBOX)
            if num_bad_images > 0:
                print(f"===== WARNING: couldn't detect {num_bad_images} faces in {images_folder}")

            gt_average_descriptor[:] = gt_descriptors.mean(0)

        np.save(DESCRIPTORS_GT_FILE, gt_average_descriptors)

    ############# GT landmarks ##############

    def string_to_valid_filename(x):
        return x.replace('/', '_')

    COMPUTE_GT_LANDMARKS = False # hardcode to `True` if you want to recompute
    LANDMARKS_GT_FILE = Path("puppeteering/VoxCeleb2_30Test/target_landmarks.npy")
    try:
        gt_landmarks = np.load(LANDMARKS_GT_FILE)
        print(f"Loaded the cached target landmarks from {LANDMARKS_GT_FILE}")
    except FileNotFoundError:
        print(f"Couldn't load the cached target landmarks from {LANDMARKS_GT_FILE}")
        gt_landmarks = np.empty((len(IDENTITIES), NUM_VIDEO_FRAMES, 68, 2), dtype=np.float32)
        COMPUTE_GT_LANDMARKS = True

    if COMPUTE_GT_LANDMARKS:
        print(f"Recomputing target landmarks into {LANDMARKS_GT_FILE}")

        for identity_idx, identity in enumerate(IDENTITIES):
            images_folder = DATASET_ROOT / 'images-cropped' / identity / 'driver'
            for frame_idx, image_path in enumerate(sorted(images_folder.iterdir())):
                driver_image = cv2.imread(str(image_path))

                landmarks, success = get_landmarks(driver_image)
                if not success:
                    print(f"Failed to detect driver's landmarks in {image_path}")

                gt_landmarks[identity_idx, frame_idx] = landmarks

        np.save(LANDMARKS_GT_FILE, gt_landmarks)

    ############### cross-driving ArcFace and self-driving landmarks ###############

    our_landmarks = np.empty((len(IDENTITIES), NUM_VIDEO_FRAMES, 68, 2), dtype=np.float32)
    # identities x drivers x frames x 512
    our_descriptors = np.empty((len(IDENTITIES), len(IDENTITIES), NUM_VIDEO_FRAMES, FACE_DESCRIPTOR_DIM), dtype=np.float32)

    for identity_idx, identity in enumerate(IDENTITIES):
        IDENTITY_RESULTS_PATH = RESULTS_ROOT / (string_to_valid_filename(identity) + '_identity')

        (IDENTITY_RESULTS_PATH / "our_identity_descriptors").mkdir(parents=True, exist_ok=True)
        (IDENTITY_RESULTS_PATH / "our_landmarks").mkdir(parents=True, exist_ok=True)

        LANDMARKS_OUR_FILE = IDENTITY_RESULTS_PATH / "our_landmarks" / f"{string_to_valid_filename(identity)}.npy"
        COMPUTE_OUR_LANDMARKS = False # hardcode to `True` if you want to recompute
        try:
            our_landmarks[identity_idx] = np.load(LANDMARKS_OUR_FILE)
            print(f"Loaded the cached landmarks from {LANDMARKS_OUR_FILE}")
        except FileNotFoundError:
            print(f"Could not load our landmarks from {LANDMARKS_OUR_FILE}, recomputing")
            COMPUTE_OUR_LANDMARKS = True

        DESCRIPTORS_OUR_FILE = IDENTITY_RESULTS_PATH / "our_identity_descriptors" / f"{string_to_valid_filename(identity)}.npy"
        COMPUTE_OUR_DESCRIPTORS = False # hardcode to `True` if you want to recompute
        try:
            our_descriptors[identity_idx] = np.load(DESCRIPTORS_OUR_FILE)
            print(f"Loaded the cached face recognition descriptors from {DESCRIPTORS_OUR_FILE}")
        except FileNotFoundError:
            print(f"Could not load our descriptors from {DESCRIPTORS_OUR_FILE}, recomputing")
            COMPUTE_OUR_DESCRIPTORS = True

        if not COMPUTE_OUR_LANDMARKS and not COMPUTE_OUR_DESCRIPTORS:
            continue

        for driver_idx, driver in enumerate(tqdm(IDENTITIES)):
            video_path = IDENTITY_RESULTS_PATH / 'driving-results' / (string_to_valid_filename(driver) + '_driver.mp4')
            video_reader = cv2.VideoCapture(str(video_path))

            driver_images, reenacted_images = [], []

            for frame_idx in range(NUM_VIDEO_FRAMES):
                ok, image = video_reader.read()
                assert ok, video_path
                reenacted_images.append(image[:, 256:])

            if COMPUTE_OUR_DESCRIPTORS:
                identity_descriptors, num_bad_images = get_identity_descriptor(reenacted_images, DEFAULT_BBOX)
                if num_bad_images > 0:
                    print(f"===== WARNING: couldn't detect {num_bad_images} faces in {video_path}")

                our_descriptors[identity_idx, driver_idx] = identity_descriptors

            if COMPUTE_OUR_LANDMARKS and driver_idx == identity_idx:
                for frame_idx, reenacted_image in enumerate(reenacted_images):
                    landmarks, success = get_landmarks(reenacted_image)
                    if not success:
                        print(f"===== WARNING: failed to detect reenactment's landmarks in frame #{frame_idx} of {video_path}")

                    our_landmarks[identity_idx, frame_idx] = landmarks

        if COMPUTE_OUR_LANDMARKS:
            np.save(LANDMARKS_OUR_FILE, our_landmarks[identity_idx])
        if COMPUTE_OUR_DESCRIPTORS:
            np.save(DESCRIPTORS_OUR_FILE, our_descriptors[identity_idx])

    print(f"Identity error: {identity_error(gt_average_descriptors, our_descriptors)}")
    print(f"Pose reconstruction error: {pose_reconstruction_error(gt_landmarks, our_landmarks)}")
    print(f"Pose reconstruction error (with optimal alignment): {pose_reconstruction_error(gt_landmarks, our_landmarks, apply_optimal_alignment=True)}")

import torch
import torch.utils.data
import numpy as np
import cv2

from .common import voxceleb, augmentation

import copy
import math
from pathlib import Path

import logging
logger = logging.getLogger('dataloader')

class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default='', type=Path)
        parser.add('--img_dir', default='Img', type=Path)
        parser.add('--kp_dir', default='landmarks', type=Path)
        parser.add('--segm_dir', default='segm', type=Path)
        parser.add('--bboxes_dir', default="/Vol0/user/e.burkov/Shared/VoxCeleb2.1_bounding-boxes-SingleBBox.npy", type=Path)

        parser.add('--draw_oval', default=True, action="store_bool")

        parser.add('--n_frames_for_encoder', default=8, type=int)

        parser = augmentation.get_args(parser)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = SampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, segm_dir=args.segm_dir, bboxes_dir=args.bboxes_dir,
            deterministic=part != 'train')

        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2SegmDataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size, augmenter)

        return dataset


class SampleLoader(voxceleb.SampleLoader):
    """
        Extends `voxceleb.SampleLoader` with segmentation masks for each sample.
    """
    def __init__(
        self, data_root, img_dir=None, kp_dir=None,
        draw_oval=True, segm_dir=None, bboxes_dir=None,
        deterministic=False):

        super().__init__(
            data_root, img_dir, kp_dir,
            draw_oval=draw_oval, deterministic=deterministic)

        self.segm_dir = segm_dir
        
        try:
            self.bboxes = np.load(bboxes_dir, allow_pickle=True).item()
        except FileNotFoundError:
            self.bboxes = {}
            logger.warning(
                f"Could not find the '.npy' file with bboxes, will assume the "
                f"images are already cropped")

    def load_segm(self, path, i):
        segm_path = Path(self.data_root) / self.segm_dir / path / (i + '.png')
        segm_path_np = Path(self.data_root) / self.segm_dir / path / (i + '.png.npy')

        if segm_path.exists():
            # Pick the second channel:
            # with PGN, it denotes head+body; with Graphonomy, all 3 channels are identical.
            segm = cv2.imread(str(segm_path))[:, :, 1]
            if segm is None:
                logger.critical(f"Couldn't load segmentation for {self.segm_dir}/{path}/{i}")
                segm = np.ones((1, 1), dtype=np.uint8)
        elif segm_path_np.exists():
            segm = np.load(str(segm_path_np))
            segm = segm[:, :, 0]
        else:
            raise FileNotFoundError(f'Sample {segm_path} not found')

        return segm

    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_stickman=False,
                    load_keypoints=False,
                    load_bounding_box=False,
                    load_segmentation=False):
        retval = {}

        # Get bounding box
        try:
            identity, sequence = path.split('/')
            bbox = self.bboxes[identity][sequence][int(i)]
            l, t, r, b = (bbox / 256.0).tolist() # bbox coordinates in a space where image takes [0; 1] x [0; 1]

            # Make bbox square and scale it
            SCALE = 1.8

            center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
            height, width = b - t, r - l
            new_box_size = max(height, width)
            l = center_x - new_box_size / 2 * SCALE
            r = center_x + new_box_size / 2 * SCALE
            t = center_y - new_box_size / 2 * SCALE
            b = center_y + new_box_size / 2 * SCALE
        except:
            # Could not find that bbox in `self.bboxes`, so assume this image is already cropped
            l, t, r, b = 0.0, 0.0, 1.0, 1.0

        # Load and crop image
        if load_image:
            image_original = self.load_rgb(path, i) # np.uint8, H x W x 3

            t_img, l_img, b_img, r_img = bbox_to_integer_coords(t, l, b, r, *image_original.shape[:2])
            # cv2.rectangle(image, (l_img, t_img), (r_img, b_img), (255,0,0), 2)

            # In VoxCeleb2.1, images have weird gray borders which are useless
            image = image_original[1:-1, 1:-1]
            t_img -= 1
            l_img -= 1
            r_img -= 1
            b_img -= 1

            # Crop
            image = crop_with_padding(image, t_img, l_img, b_img, r_img)

            # Resize to the target resolution
            image = cv2.resize(image, (imsize, imsize),
                interpolation=cv2.INTER_CUBIC if imsize > b_img - t_img else cv2.INTER_AREA)

            retval['image'] = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_keypoints:
            assert load_image, \
                "Please also require the image if you need keypoints, because we need to know what are H and W"
            keypoints = self.load_keypoints(path, i)

            keypoints /= image_original.shape[1]
            keypoints -= [[l, t]]
            keypoints /= [[r-l, b-t]]

            retval['keypoints'] = torch.tensor(keypoints.astype(np.float32).flatten())

        if load_stickman:
            assert load_keypoints
            stickman = self.draw_stickman(image.shape[:2], keypoints * image.shape[0])
            retval['stickman'] = torch.tensor(stickman.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_segmentation:
            segmentation = self.load_segm(path, i)

            t_img, l_img, b_img, r_img = bbox_to_integer_coords(t, l, b, r, *segmentation.shape[:2])

            # In VoxCeleb2.1, images have weird gray borders which are useless
            segmentation = segmentation[1:-1, 1:-1]
            t_img -= 1
            l_img -= 1
            r_img -= 1
            b_img -= 1

            segmentation = crop_with_padding(segmentation, t_img, l_img, b_img, r_img, segmentation=True)

            segmentation = cv2.resize(segmentation, (imsize, imsize))
            segmentation = torch.tensor(segmentation.astype(np.float32) / 255.0)[None]
            segmentation = segmentation.expand((3,) + segmentation.shape[1:])
            retval['segmentation'] = segmentation

        return retval


class VoxCeleb2SegmDataset(voxceleb.VoxCeleb2Dataset):
    def __init__(self, dirlist, loader, inference, n_frames_for_encoder, imsize, augmenter):
        super().__init__(dirlist, loader, inference, n_frames_for_encoder, imsize, augmenter)

    def __getitem__(self, index):
        data_dict, target_dict = {}, {}
        row = self.dirlist.iloc[index]
        path = row['path']

        finetuning = 'file' in self.dirlist
        if finetuning:
            # We are doing fine-tuning (`self.dirlist` enumerates all images, not just identities)
            dec_ids = [row['file']]

            features_to_load = {
                'load_image': True,
                'load_segmentation': not self.inference
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['enc_rgbs'] = dec_dict['image']
            # data_dict['dec_keypoints'] = dec_dict['keypoints']
            data_dict['pose_input_rgbs'] = dec_dict['image']

            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']

                target_dict['real_segm'] = dec_dict['segmentation']

            target_dict['label'] = 0
        else:
            # We are doing normal training, in which "one sample" is one video
            ids = self.loader.list_ids(path, self.n_frames_for_encoder+1)

            enc_ids = ids[:-1]
            dec_ids = ids[-1:]

            features_to_load = {
                'load_image': True
            }
            enc_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in enc_ids]
            enc_dict = torch.utils.data.dataloader.default_collate(enc_dicts)
            del enc_dicts

            features_to_load = {
                'load_image': not self.inference,
                'load_segmentation': not self.inference
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['enc_rgbs'] = enc_dict['image']
            # data_dict['dec_keypoints'] = dec_dict['keypoints']
            data_dict['pose_input_rgbs'] = dec_dict['image']

            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']

                target_dict['real_segm'] = dec_dict['segmentation']

            target_dict['label'] = self.dirlist.index[index]

        if not self.inference:
            data_dict['pose_input_rgbs'], data_dict['target_rgbs'], target_dict['real_segm'] = \
                self.augmenter.augment_triplet(
                    data_dict['pose_input_rgbs'], data_dict['target_rgbs'], target_dict['real_segm'])

        return data_dict, target_dict

    def deterministic_(self, seed=0):
        return self.augmenter.deterministic_(seed)

def bbox_to_integer_coords(t, l, b, r, image_h, image_w):
    """
        t, l, b, r:
            float
            Bbox coordinates in a space where image takes [0; 1] x [0; 1].
        image_h, image_w:
            int

        return: t, l, b, r
            int
            Bbox coordinates in given image's pixel space.
            C-style indices (i.e. `b` and `r` are exclusive).
    """
    t *= image_h
    l *= image_h
    b *= image_h
    r *= image_h

    l, t = map(math.floor, (l, t))
    r, b = map(math.ceil, (r, b))

    # After rounding, make *exactly* square again
    b += (r - l) - (b - t)
    assert b - t == r - l

    # Make `r` and `b` C-style (=exclusive) indices
    r += 1
    b += 1
    return t, l, b, r

def crop_with_padding(image, t, l, b, r, segmentation=False):
    """
        image:
            numpy, H x W x 3
        t, l, b, r:
            int
        segmentation:
            bool
            Affects padding.

        return:
            numpy, (b-t) x (r-l) x 3
    """
    t_clamp, b_clamp = max(0, t), min(b, image.shape[0])
    l_clamp, r_clamp = max(0, l), min(r, image.shape[1])
    image = image[t_clamp:b_clamp, l_clamp:r_clamp]

    # If the bounding box went outside of the image, restore those areas by padding
    padding = [t_clamp - t, b - b_clamp, l_clamp - l, r - r_clamp]
    if sum(padding) == 0: # = if the bbox fully fit into image
        return image

    if segmentation:
        padding_top = [(x if i == 0 else 0) for i, x in enumerate(padding)]
        padding_others = [(x if i != 0 else 0) for i, x in enumerate(padding)]
        image = cv2.copyMakeBorder(image, *padding_others, cv2.BORDER_REPLICATE)
        image = cv2.copyMakeBorder(image, *padding_top, cv2.BORDER_CONSTANT)
    else:
        image = cv2.copyMakeBorder(image, *padding, cv2.BORDER_REFLECT101)
    assert image.shape[:2] == (b - t, r - l)

    # We will blur those padded areas
    h, w = image.shape[:2]
    y, x = map(np.float32, np.ogrid[:h, :w]) # meshgrids

    mask_l = np.full_like(x, np.inf) if padding[2] == 0 else (x / padding[2])
    mask_t = np.full_like(y, np.inf) if padding[0] == 0 else (y / padding[0])
    mask_r = np.full_like(x, np.inf) if padding[3] == 0 else ((w-1-x) / padding[3])
    mask_b = np.full_like(y, np.inf) if padding[1] == 0 else ((h-1-y) / padding[1])

    # The farther from the original image border, the more blur will be applied
    mask = np.maximum(
        1.0 - np.minimum(mask_l, mask_r),
        1.0 - np.minimum(mask_t, mask_b))

    # Do blur
    sigma = h * 0.016
    kernel_size = 0
    image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Now we'd like to do alpha blending math, so convert to float32
    def to_float32(x):
        x = x.astype(np.float32)
        x /= 255.0
        return x
    image = to_float32(image)
    image_blurred = to_float32(image_blurred)

    # Support 2-dimensional images (e.g. segmentation maps)
    if image.ndim < 3:
        image.shape += (1,)
        image_blurred.shape += (1,)
    mask.shape += (1,)

    # Replace padded areas with their blurred versions, and apply
    # some quickly fading blur to the inner part of the image
    image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)

    # Make blurred borders fade to edges
    if segmentation:
        fade_color = np.zeros_like(image)
        fade_color[:, :padding[2]] = 0.0
        fade_color[:, -padding[3]:] = 0.0
        mask = (1.0 - np.minimum(mask_l, mask_r))[:, :, None]
    else:
        fade_color = np.median(image, axis=(0,1))
    image += (fade_color - image) * np.clip(mask, 0.0, 1.0) 

    # Convert back to uint8 for interface consistency
    image *= 255.0
    image.round(out=image)
    image.clip(0, 255, out=image)
    image = image.astype(np.uint8)

    return image

import torch
import torch.utils.data
import numpy as np
from .common import voxceleb, augmentation

from pathlib import Path
import cv2

import logging
logger = logging.getLogger('dataloader')

class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default='', type=Path)
        parser.add('--img_dir', default='Img', type=Path)
        parser.add('--kp_dir', default='landmarks', type=Path)
        parser.add('--segm_dir', default='segm', type=Path)
        parser.add('--draw_oval', default=True, action="store_bool")

        parser.add('--n_frames_for_encoder', default=8, type=int)

        parser = augmentation.get_args(parser)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = SampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, segm_dir=args.segm_dir,
            deterministic=part != 'train')
        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2SegmDataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size)

        return dataset


class SampleLoader(voxceleb.SampleLoader):
    """
        Extends `voxceleb.SampleLoader` with segmentation masks for each sample.
    """
    def __init__(
        self, data_root, img_dir=None, kp_dir=None,
        draw_oval=True, segm_dir=None, deterministic=False):

        super().__init__(
            data_root, img_dir, kp_dir,
            draw_oval=draw_oval, deterministic=deterministic)

        self.segm_dir = segm_dir

    def load_segm(self, path, i):
        segm_path = Path(self.data_root) / self.segm_dir / path / (i + '.png')
        segm_path_np = Path(self.data_root) / self.segm_dir / path / (i + '.png.npy')

        if segm_path.exists():
            segm = cv2.imread(str(segm_path))
            if segm is None:
                logger.critical(f"Couldn't load segmentation for {self.segm_dir}/{path}/{i}")
                segm = np.ones((1, 1, 3), dtype=np.float32)
            segm = segm[:, :, 1].astype(np.float32) / 255.
        elif segm_path_np.exists():
            segm = np.load(str(segm_path_np))
            segm = segm[:, :, 0]
        else:
            raise FileNotFoundError(f'Sample {segm_path} not found')

        return segm

    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_stickman=False,
                    load_keypoints=False,
                    load_segmentation=False):
        retval = super().load_sample(
            path, i, imsize,
            load_image=load_image,
            load_stickman=load_stickman,
            load_keypoints=load_keypoints)

        if load_segmentation:
            segmentation = self.load_segm(path, i)
            segmentation = cv2.resize(segmentation, (imsize, imsize))
            segmentation = torch.tensor(segmentation)[None]
            segmentation = segmentation.expand((3,) + segmentation.shape[1:])
            retval['segmentation'] = segmentation

        return retval


class VoxCeleb2SegmDataset(voxceleb.VoxCeleb2Dataset):
    def __getitem__(self, index):
        data_dict, target_dict = {}, {}

        row = self.dirlist.iloc[index]
        path = row['path']

        finetuning = 'file' in self.dirlist
        if finetuning:
            # We are doing fine-tuning (`self.dirlist` enumerates all images, not just identities)
            dec_ids = [row['file']]

            features_to_load = {
                'load_image': True,
                'load_stickman': True,
                'load_keypoints': True,
                'load_segmentation': not self.inference
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']
                data_dict['real_segm'] = dec_dict['segmentation']

            data_dict['pose_input_rgbs'] = dec_dict['image']
            data_dict['dec_stickmen'] = dec_dict['stickman']
            data_dict['dec_keypoints'] = dec_dict['keypoints']
            # Also putting `enc_*` stuff for embedding pre-calculation before fune-tuning
            data_dict['enc_stickmen'] = dec_dict['stickman']
            data_dict['enc_rgbs'] = dec_dict['image']

            target_dict['label'] = 0
        else:
            # We are doing normal training, in which "one sample" is one video
            ids = self.loader.list_ids(path, self.n_frames_for_encoder+1)

            enc_ids = ids[:-1]
            dec_ids = ids[-1:]

            features_to_load = {
                'load_image': True,
                'load_stickman': True
            }
            enc_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in enc_ids]
            enc_dict = torch.utils.data.dataloader.default_collate(enc_dicts)
            del enc_dicts

            features_to_load = {
                'load_image': True,
                'load_stickman': True,
                'load_keypoints': True,
                'load_segmentation': not self.inference
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']
                
                target_dict['real_segm'] = dec_dict['segmentation']
                
            data_dict['enc_stickmen'] = enc_dict['stickman']
            data_dict['enc_rgbs'] = enc_dict['image']
            data_dict['dec_keypoints'] = dec_dict['keypoints']
            data_dict['dec_stickmen'] = dec_dict['stickman']
            data_dict['pose_input_rgbs'] = dec_dict['image']

            target_dict['label'] = self.dirlist.index[index]

        if not self.augmenter.is_empty():
            raise NotImplementedError("Keypoints augmentation is NYI")
        data_dict['pose_input_rgbs'] = self.augmenter.augment_tensor(data_dict['pose_input_rgbs'])

        return data_dict, target_dict

import torch
import torch.utils.data
import numpy as np
import cv2

from .common import voxceleb, augmentation

import copy
import math
from pathlib import Path

import logging
logger = logging.getLogger('dataloader')

class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default='', type=Path)
        parser.add('--img_dir', default='Img', type=Path)
        parser.add('--kp_dir', default='landmarks', type=Path)
        parser.add('--segm_dir', default='segm', type=Path)
        parser.add('--bboxes_dir', default="/Vol0/user/e.burkov/Shared/VoxCeleb2.1_bounding-boxes-SingleBBox.npy", type=Path)
        
        parser.add('--draw_oval', default=True, action="store_bool")
        parser.add('--n_frames_for_encoder', default=8, type=int)

        parser.add('--voxceleb1_crop_type', choices=['x2face', 'fabnet'], default='x2face')

        parser = augmentation.get_args(parser)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = SampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, segm_dir=args.segm_dir, bboxes_dir=args.bboxes_dir,
            deterministic=part != 'train', voxceleb1_crop_type=args.voxceleb1_crop_type)
        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2SegmDataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size, augmenter)

        return dataset


class SampleLoader(voxceleb.SampleLoader):
    """
        Extends `voxceleb.SampleLoader` with segmentation masks for each sample.
    """
    def __init__(
        self, data_root, img_dir=None, kp_dir=None,
        draw_oval=True, segm_dir=None, bboxes_dir=None,
        deterministic=False, voxceleb1_crop_type='x2face'):

        super().__init__(
            data_root, img_dir, kp_dir,
            draw_oval=draw_oval, deterministic=deterministic)

        self.segm_dir = segm_dir
        
        try:
            self.bboxes = np.load(bboxes_dir, allow_pickle=True).item()
        except FileNotFoundError:
            self.bboxes = {}
            logger.warning(
                f"Could not find the '.npy' file with bboxes, will assume the "
                f"images are already cropped")

        self.voxceleb1_crop_type = voxceleb1_crop_type

    def load_segm(self, path, i):
        segm_path = Path(self.data_root) / self.segm_dir / path / (i + '.png')
        segm_path_np = Path(self.data_root) / self.segm_dir / path / (i + '.png.npy')

        if segm_path.exists():
            # Pick the second channel (denotes head+body)
            segm = cv2.imread(str(segm_path))[:, :, 1]
            if segm is None:
                logger.critical(f"Couldn't load segmentation for {self.segm_dir}/{path}/{i}")
                segm = np.ones((1, 1), dtype=np.uint8)
        elif segm_path_np.exists():
            segm = np.load(str(segm_path_np))
            segm = segm[:, :, 0]
        else:
            raise FileNotFoundError(f'Sample {segm_path} not found')

        return segm

    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_bounding_box=False,
                    load_keypoints=False,
                    load_segmentation=False,
                    load_voxceleb1_crop=False):
        retval = {}

        # Get bounding box
        try:
            identity, sequence = path.split('/')
            bbox = self.bboxes[identity][sequence][int(i)]
            l, t, r, b = (bbox / 256.0).tolist() # bbox coordinates in a space where image takes [0; 1] x [0; 1]

            # Make bbox square and scale it
            SCALE = 1.8

            center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
            height, width = b - t, r - l
            new_box_size = max(height, width)
            l = center_x - new_box_size / 2 * SCALE
            r = center_x + new_box_size / 2 * SCALE
            t = center_y - new_box_size / 2 * SCALE
            b = center_y + new_box_size / 2 * SCALE
        except:
            # Could not find that bbox in `self.bboxes`, so assume this image is already cropped
            l, t, r, b = 0.0, 0.0, 1.0, 1.0

        def bbox_to_integer_coords(t, l, b, r, image_h, image_w):
            """
                t, l, b, r:
                    float
                    Bbox coordinates in a space where image takes [0; 1] x [0; 1].
                image_h, image_w:
                    int

                return: t, l, b, r
                    int
                    Bbox coordinates in given image's pixel space.
                    C-style indices (i.e. `b` and `r` are exclusive).
            """
            t *= image_h
            l *= image_h
            b *= image_h
            r *= image_h

            l, t = map(math.floor, (l, t))
            r, b = map(math.ceil, (r, b))

            # After rounding, make *exactly* square again
            b += (r - l) - (b - t)
            assert b - t == r - l

            # Make `r` and `b` C-style (=exclusive) indices
            r += 1
            b += 1
            return t, l, b, r

        def crop_with_padding(image, t, l, b, r, segmentation=False):
            """
                image:
                    numpy, H x W x 3
                t, l, b, r:
                    int
                segmentation:
                    bool
                    Affects padding.

                return:
                    numpy, (b-t) x (r-l) x 3
            """
            t_clamp, b_clamp = max(0, t), min(b, image.shape[0])
            l_clamp, r_clamp = max(0, l), min(r, image.shape[1])
            image = image[t_clamp:b_clamp, l_clamp:r_clamp]

            # If the bounding box went outside of the image, restore those areas by padding
            padding = [t_clamp - t, b - b_clamp, l_clamp - l, r - r_clamp]
            if sum(padding) == 0: # = if the bbox fully fit into image
                return image

            if segmentation:
                padding_top = [(x if i == 0 else 0) for i, x in enumerate(padding)]
                padding_others = [(x if i != 0 else 0) for i, x in enumerate(padding)]
                image = cv2.copyMakeBorder(image, *padding_others, cv2.BORDER_REPLICATE)
                image = cv2.copyMakeBorder(image, *padding_top, cv2.BORDER_CONSTANT)
            else:
                image = cv2.copyMakeBorder(image, *padding, cv2.BORDER_REFLECT101)
            assert image.shape[:2] == (b - t, r - l)

            # We will blur those padded areas
            h, w = image.shape[:2]
            y, x = map(np.float32, np.ogrid[:h, :w]) # meshgrids
            
            mask_l = np.full_like(x, np.inf) if padding[2] == 0 else (x / padding[2])
            mask_t = np.full_like(y, np.inf) if padding[0] == 0 else (y / padding[0])
            mask_r = np.full_like(x, np.inf) if padding[3] == 0 else ((w-1-x) / padding[3])
            mask_b = np.full_like(y, np.inf) if padding[1] == 0 else ((h-1-y) / padding[1])

            # The farther from the original image border, the more blur will be applied
            mask = np.maximum(
                1.0 - np.minimum(mask_l, mask_r),
                1.0 - np.minimum(mask_t, mask_b))
            
            # Do blur
            sigma = h * 0.016
            kernel_size = 0
            image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

            # Now we'd like to do alpha blending math, so convert to float32
            def to_float32(x):
                x = x.astype(np.float32)
                x /= 255.0
                return x
            image = to_float32(image)
            image_blurred = to_float32(image_blurred)

            # Support 2-dimensional images (e.g. segmentation maps)
            if image.ndim < 3:
                image.shape += (1,)
                image_blurred.shape += (1,)
            mask.shape += (1,)

            # Replace padded areas with their blurred versions, and apply
            # some quickly fading blur to the inner part of the image
            image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)

            # Make blurred borders fade to edges
            if segmentation:
                fade_color = np.zeros_like(image)
                fade_color[:, :padding[2]] = 0.0
                fade_color[:, -padding[3]:] = 0.0
                mask = (1.0 - np.minimum(mask_l, mask_r))[:, :, None]
            else:
                fade_color = np.median(image, axis=(0,1))
            image += (fade_color - image) * np.clip(mask, 0.0, 1.0) 
            
            # Convert back to uint8 for interface consistency
            image *= 255.0
            image.round(out=image)
            image.clip(0, 255, out=image)
            image = image.astype(np.uint8)

            return image

        # Load and crop image
        if load_image:
            image_original = self.load_rgb(path, i) # np.uint8, H x W x 3

            t_img, l_img, b_img, r_img = bbox_to_integer_coords(t, l, b, r, *image_original.shape[:2])
            # cv2.rectangle(image, (l_img, t_img), (r_img, b_img), (255,0,0), 2)

            # In VoxCeleb2.1, images have weird gray borders which are useless
            image = image_original[1:-1, 1:-1]
            t_img -= 1
            l_img -= 1
            r_img -= 1
            b_img -= 1

            # Crop
            image = crop_with_padding(image, t_img, l_img, b_img, r_img)

            # Resize to the target resolution
            image = cv2.resize(image, (imsize, imsize),
                interpolation=cv2.INTER_CUBIC if imsize > b_img - t_img else cv2.INTER_AREA)

            retval['image'] = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_voxceleb1_crop:
            # Crop as in VoxCeleb1
            SCALE = 1.4
            try:
                bbox = self.bboxes[identity][sequence][int(i)]
                l_bbox, t_bbox, r_bbox, b_bbox = (bbox / 256.0).tolist() # bbox coordinates in a space where image takes [0; 1] x [0; 1]

                # Make bbox square and scale it
                center_x, center_y = (l_bbox + r_bbox) * 0.5, (t_bbox + b_bbox) * 0.5
                height, width = b_bbox - t_bbox, r_bbox - l_bbox
                new_box_size = max(height, width)
                l_bbox = center_x - new_box_size / 2 * SCALE
                r_bbox = center_x + new_box_size / 2 * SCALE
                t_bbox = center_y - new_box_size / 2 * SCALE
                b_bbox = center_y + new_box_size / 2 * SCALE
            except:
                # Could not find that bbox in `self.bboxes`, so assume this image is already cropped
                cutoff = (1 - SCALE / 1.8) / 2
                l_bbox, t_bbox, r_bbox, b_bbox = cutoff, cutoff, 1 - cutoff, 1 - cutoff

            if self.voxceleb1_crop_type == 'fabnet':
                cutoff_l = 43 / 256
                cutoff_t = 66 / 256
                cutoff_r = 43 / 256
                cutoff_b = 20 / 256

                h_bbox = b_bbox - t_bbox
                w_bbox = r_bbox - l_bbox

                l_bbox += w_bbox * cutoff_l
                r_bbox -= w_bbox * cutoff_r
                t_bbox += h_bbox * cutoff_t
                b_bbox -= h_bbox * cutoff_b

            t_crop, l_crop, b_crop, r_crop = bbox_to_integer_coords(t_bbox, l_bbox, b_bbox, r_bbox, *image_original.shape[:2])

            image_cropped_voxceleb = crop_with_padding(image_original, t_crop, l_crop, b_crop, r_crop)
            image_cropped_voxceleb = cv2.resize(image_cropped_voxceleb, (256, 256),
                interpolation=cv2.INTER_CUBIC if 256 > b_crop - r_crop else cv2.INTER_AREA)

            retval['image_cropped_voxceleb1'] = torch.tensor(image_cropped_voxceleb.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_keypoints:
            assert load_image, \
                "Please also require the image if you need keypoints, because we need to know what are H and W"
            keypoints = self.load_keypoints(path, i)

            keypoints /= image_original.shape[1]
            keypoints -= [[l, t]]
            keypoints /= [[r-l, b-t]]

            retval['keypoints'] = torch.tensor(keypoints.astype(np.float32).flatten())

        if load_segmentation:
            segmentation = self.load_segm(path, i)

            t_img, l_img, b_img, r_img = bbox_to_integer_coords(t, l, b, r, *segmentation.shape[:2])

            # In VoxCeleb2.1, images have weird gray borders which are useless
            segmentation = segmentation[1:-1, 1:-1]
            t_img -= 1
            l_img -= 1
            r_img -= 1
            b_img -= 1

            segmentation = crop_with_padding(segmentation, t_img, l_img, b_img, r_img, segmentation=True)

            segmentation = cv2.resize(segmentation, (imsize, imsize))
            segmentation = torch.tensor(segmentation.astype(np.float32) / 255.0)[None]
            segmentation = segmentation.expand((3,) + segmentation.shape[1:])
            retval['segmentation'] = segmentation

        return retval


class VoxCeleb2SegmDataset(voxceleb.VoxCeleb2Dataset):
    def __getitem__(self, index):
        data_dict, target_dict = {}, {}

        row = self.dirlist.iloc[index]
        path = row['path']

        finetuning = 'file' in self.dirlist
        if finetuning:
            # We are doing fine-tuning (`self.dirlist` enumerates all images, not just identities)
            dec_ids = [row['file']]

            features_to_load = {
                'load_image': True,
                'load_segmentation': not self.inference,
                'load_voxceleb1_crop': True
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            if not self.inference:
                data_dict['enc_rgbs'] = dec_dict['image']
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']
                
                target_dict['real_segm'] = dec_dict['segmentation']

            # data_dict['dec_keypoints'] = dec_dict['keypoints']
            data_dict['pose_input_rgbs'] = dec_dict['image_cropped_voxceleb1']

            target_dict['label'] = 0
        else:
            # We are doing normal training, in which "one sample" is one video
            ids = self.loader.list_ids(path, self.n_frames_for_encoder+1)

            enc_ids = ids[:-1]
            dec_ids = ids[-1:]

            features_to_load = {
                'load_image': True
            }
            enc_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in enc_ids]
            enc_dict = torch.utils.data.dataloader.default_collate(enc_dicts)
            del enc_dicts

            features_to_load = {
                'load_image': True,
                'load_segmentation': not self.inference,
                'load_voxceleb1_crop': True
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            if not self.inference:
                data_dict['enc_rgbs'] = enc_dict['image']
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']

                target_dict['real_segm'] = dec_dict['segmentation']
            
            # data_dict['dec_keypoints'] = dec_dict['keypoints']
            data_dict['pose_input_rgbs'] = dec_dict['image_cropped_voxceleb1']

            target_dict['label'] = self.dirlist.index[index]

        data_dict['pose_input_rgbs'] = self.augmenter.augment_tensor(data_dict['pose_input_rgbs'])

        return data_dict, target_dict

import torch
import torch.utils.data
import numpy as np
import cv2

from .common import voxceleb, augmentation

import copy
import math
from pathlib import Path

import logging
logger = logging.getLogger('dataloader')

class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default='', type=Path)
        parser.add('--img_dir', default='Img', type=Path)
        parser.add('--kp_dir', default='landmarks', type=Path)
        parser.add('--segm_dir', default='segm', type=Path)
        parser.add('--bboxes_dir', default="/Vol0/user/e.burkov/Shared/VoxCeleb2.1_bounding-boxes-SingleBBox.npy", type=Path)
        
        parser.add('--draw_oval', default=True, action="store_bool")
        parser.add('--n_frames_for_encoder', default=8, type=int)

        parser.add('--voxceleb1_crop_type', choices=['x2face', 'fabnet'], default='x2face')

        parser = augmentation.get_args(parser)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = SampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, segm_dir=args.segm_dir, bboxes_dir=args.bboxes_dir,
            deterministic=part != 'train', voxceleb1_crop_type=args.voxceleb1_crop_type)
        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2SegmDataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size, augmenter)

        return dataset


class SampleLoader(voxceleb.SampleLoader):
    """
        Extends `voxceleb.SampleLoader` with segmentation masks for each sample.
    """
    def __init__(
        self, data_root, img_dir=None, kp_dir=None,
        draw_oval=True, segm_dir=None, bboxes_dir=None,
        deterministic=False, voxceleb1_crop_type='x2face'):

        super().__init__(
            data_root, img_dir, kp_dir,
            draw_oval=draw_oval, deterministic=deterministic)

        self.segm_dir = segm_dir
        
        try:
            self.bboxes = np.load(bboxes_dir, allow_pickle=True).item()
        except FileNotFoundError:
            self.bboxes = {}
            logger.warning(
                f"Could not find the '.npy' file with bboxes, will assume the "
                f"images are already cropped")

        self.voxceleb1_crop_type = voxceleb1_crop_type

    def load_segm(self, path, i):
        segm_path = Path(self.data_root) / self.segm_dir / path / (i + '.png')
        segm_path_np = Path(self.data_root) / self.segm_dir / path / (i + '.png.npy')

        if segm_path.exists():
            # Pick the second channel (denotes head+body)
            segm = cv2.imread(str(segm_path))[:, :, 1]
            if segm is None:
                logger.critical(f"Couldn't load segmentation for {self.segm_dir}/{path}/{i}")
                segm = np.ones((1, 1), dtype=np.uint8)
        elif segm_path_np.exists():
            segm = np.load(str(segm_path_np))
            segm = segm[:, :, 0]
        else:
            raise FileNotFoundError(f'Sample {segm_path} not found')

        return segm

    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_bounding_box=False,
                    load_keypoints=False,
                    load_segmentation=False,
                    load_voxceleb1_crop=False):
        retval = {}

        # Get bounding box
        try:
            identity, sequence = path.split('/')
            bbox = self.bboxes[identity][sequence][int(i)]
            l, t, r, b = (bbox / 256.0).tolist() # bbox coordinates in a space where image takes [0; 1] x [0; 1]

            # Make bbox square and scale it
            SCALE = 1.8

            center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
            height, width = b - t, r - l
            new_box_size = max(height, width)
            l = center_x - new_box_size / 2 * SCALE
            r = center_x + new_box_size / 2 * SCALE
            t = center_y - new_box_size / 2 * SCALE
            b = center_y + new_box_size / 2 * SCALE
        except:
            # Could not find that bbox in `self.bboxes`, so assume this image is already cropped
            l, t, r, b = 0.0, 0.0, 1.0, 1.0

        def bbox_to_integer_coords(t, l, b, r, image_h, image_w):
            """
                t, l, b, r:
                    float
                    Bbox coordinates in a space where image takes [0; 1] x [0; 1].
                image_h, image_w:
                    int

                return: t, l, b, r
                    int
                    Bbox coordinates in given image's pixel space.
                    C-style indices (i.e. `b` and `r` are exclusive).
            """
            t *= image_h
            l *= image_h
            b *= image_h
            r *= image_h

            l, t = map(math.floor, (l, t))
            r, b = map(math.ceil, (r, b))

            # After rounding, make *exactly* square again
            b += (r - l) - (b - t)
            assert b - t == r - l

            # Make `r` and `b` C-style (=exclusive) indices
            r += 1
            b += 1
            return t, l, b, r

        def crop_with_padding(image, t, l, b, r, segmentation=False):
            """
                image:
                    numpy, H x W x 3
                t, l, b, r:
                    int
                segmentation:
                    bool
                    Affects padding.

                return:
                    numpy, (b-t) x (r-l) x 3
            """
            t_clamp, b_clamp = max(0, t), min(b, image.shape[0])
            l_clamp, r_clamp = max(0, l), min(r, image.shape[1])
            image = image[t_clamp:b_clamp, l_clamp:r_clamp]

            # If the bounding box went outside of the image, restore those areas by padding
            padding = [t_clamp - t, b - b_clamp, l_clamp - l, r - r_clamp]
            if sum(padding) == 0: # = if the bbox fully fit into image
                return image

            if segmentation:
                padding_top = [(x if i == 0 else 0) for i, x in enumerate(padding)]
                padding_others = [(x if i != 0 else 0) for i, x in enumerate(padding)]
                image = cv2.copyMakeBorder(image, *padding_others, cv2.BORDER_REPLICATE)
                image = cv2.copyMakeBorder(image, *padding_top, cv2.BORDER_CONSTANT)
            else:
                image = cv2.copyMakeBorder(image, *padding, cv2.BORDER_REFLECT101)
            assert image.shape[:2] == (b - t, r - l)

            # We will blur those padded areas
            h, w = image.shape[:2]
            y, x = map(np.float32, np.ogrid[:h, :w]) # meshgrids
            
            mask_l = np.full_like(x, np.inf) if padding[2] == 0 else (x / padding[2])
            mask_t = np.full_like(y, np.inf) if padding[0] == 0 else (y / padding[0])
            mask_r = np.full_like(x, np.inf) if padding[3] == 0 else ((w-1-x) / padding[3])
            mask_b = np.full_like(y, np.inf) if padding[1] == 0 else ((h-1-y) / padding[1])

            # The farther from the original image border, the more blur will be applied
            mask = np.maximum(
                1.0 - np.minimum(mask_l, mask_r),
                1.0 - np.minimum(mask_t, mask_b))
            
            # Do blur
            sigma = h * 0.016
            kernel_size = 0
            image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

            # Now we'd like to do alpha blending math, so convert to float32
            def to_float32(x):
                x = x.astype(np.float32)
                x /= 255.0
                return x
            image = to_float32(image)
            image_blurred = to_float32(image_blurred)

            # Support 2-dimensional images (e.g. segmentation maps)
            if image.ndim < 3:
                image.shape += (1,)
                image_blurred.shape += (1,)
            mask.shape += (1,)

            # Replace padded areas with their blurred versions, and apply
            # some quickly fading blur to the inner part of the image
            image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)

            # Make blurred borders fade to edges
            if segmentation:
                fade_color = np.zeros_like(image)
                fade_color[:, :padding[2]] = 0.0
                fade_color[:, -padding[3]:] = 0.0
                mask = (1.0 - np.minimum(mask_l, mask_r))[:, :, None]
            else:
                fade_color = np.median(image, axis=(0,1))
            image += (fade_color - image) * np.clip(mask, 0.0, 1.0) 
            
            # Convert back to uint8 for interface consistency
            image *= 255.0
            image.round(out=image)
            image.clip(0, 255, out=image)
            image = image.astype(np.uint8)

            return image

        # Load and crop image
        if load_image:
            image_original = self.load_rgb(path, i) # np.uint8, H x W x 3

            t_img, l_img, b_img, r_img = bbox_to_integer_coords(t, l, b, r, *image_original.shape[:2])
            # cv2.rectangle(image, (l_img, t_img), (r_img, b_img), (255,0,0), 2)

            # In VoxCeleb2.1, images have weird gray borders which are useless
            image = image_original[1:-1, 1:-1]
            t_img -= 1
            l_img -= 1
            r_img -= 1
            b_img -= 1

            # Crop
            image = crop_with_padding(image, t_img, l_img, b_img, r_img)

            # Resize to the target resolution
            image = cv2.resize(image, (imsize, imsize),
                interpolation=cv2.INTER_CUBIC if imsize > b_img - t_img else cv2.INTER_AREA)

            retval['image'] = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_voxceleb1_crop:
            # Crop as in VoxCeleb1
            SCALE = 1.4
            try:
                bbox = self.bboxes[identity][sequence][int(i)]
                l_bbox, t_bbox, r_bbox, b_bbox = (bbox / 256.0).tolist() # bbox coordinates in a space where image takes [0; 1] x [0; 1]

                # Make bbox square and scale it
                center_x, center_y = (l_bbox + r_bbox) * 0.5, (t_bbox + b_bbox) * 0.5
                height, width = b_bbox - t_bbox, r_bbox - l_bbox
                new_box_size = max(height, width)
                l_bbox = center_x - new_box_size / 2 * SCALE
                r_bbox = center_x + new_box_size / 2 * SCALE
                t_bbox = center_y - new_box_size / 2 * SCALE
                b_bbox = center_y + new_box_size / 2 * SCALE
            except:
                # Could not find that bbox in `self.bboxes`, so assume this image is already cropped
                cutoff = (1 - SCALE / 1.8) / 2
                l_bbox, t_bbox, r_bbox, b_bbox = cutoff, cutoff, 1 - cutoff, 1 - cutoff

            if self.voxceleb1_crop_type == 'fabnet':
                cutoff_l = 43 / 256
                cutoff_t = 66 / 256
                cutoff_r = 43 / 256
                cutoff_b = 20 / 256

                h_bbox = b_bbox - t_bbox
                w_bbox = r_bbox - l_bbox

                l_bbox += w_bbox * cutoff_l
                r_bbox -= w_bbox * cutoff_r
                t_bbox += h_bbox * cutoff_t
                b_bbox -= h_bbox * cutoff_b

            t_crop, l_crop, b_crop, r_crop = bbox_to_integer_coords(t_bbox, l_bbox, b_bbox, r_bbox, *image_original.shape[:2])

            image_cropped_voxceleb = crop_with_padding(image_original, t_crop, l_crop, b_crop, r_crop)
            image_cropped_voxceleb = cv2.resize(image_cropped_voxceleb, (256, 256),
                interpolation=cv2.INTER_CUBIC if 256 > b_crop - r_crop else cv2.INTER_AREA)

            retval['image_cropped_voxceleb1'] = torch.tensor(image_cropped_voxceleb.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_keypoints:
            assert load_image, \
                "Please also require the image if you need keypoints, because we need to know what are H and W"
            keypoints = self.load_keypoints(path, i)

            keypoints /= image_original.shape[1]
            keypoints -= [[l, t]]
            keypoints /= [[r-l, b-t]]

            retval['keypoints'] = torch.tensor(keypoints.astype(np.float32).flatten())

        if load_segmentation:
            segmentation = self.load_segm(path, i)

            t_img, l_img, b_img, r_img = bbox_to_integer_coords(t, l, b, r, *segmentation.shape[:2])

            # In VoxCeleb2.1, images have weird gray borders which are useless
            segmentation = segmentation[1:-1, 1:-1]
            t_img -= 1
            l_img -= 1
            r_img -= 1
            b_img -= 1

            segmentation = crop_with_padding(segmentation, t_img, l_img, b_img, r_img, segmentation=True)

            segmentation = cv2.resize(segmentation, (imsize, imsize))
            segmentation = torch.tensor(segmentation.astype(np.float32) / 255.0)[None]
            segmentation = segmentation.expand((3,) + segmentation.shape[1:])
            retval['segmentation'] = segmentation

        return retval


class VoxCeleb2SegmDataset(voxceleb.VoxCeleb2Dataset):
    def __getitem__(self, index):
        data_dict, target_dict = {}, {}

        row = self.dirlist.iloc[index]
        path = row['path']

        finetuning = 'file' in self.dirlist
        if finetuning:
            # We are doing fine-tuning (`self.dirlist` enumerates all images, not just identities)
            dec_ids = [row['file']]

            features_to_load = {
                'load_image': True, # but not used, needed only for computing 'load_voxceleb1_crop'
                'load_voxceleb1_crop': True
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['pose_input_rgbs'] = dec_dict['image_cropped_voxceleb1']
            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image_cropped_voxceleb1']

            target_dict['label'] = 0
        else:
            # We are doing normal training, in which "one sample" is one video
            ids = self.loader.list_ids(path, self.n_frames_for_encoder+1)

            enc_ids = ids[:-1]
            dec_ids = ids[-1:]

            features_to_load = {
                'load_image': True, # but not used, needed only for computing 'load_voxceleb1_crop'
                'load_voxceleb1_crop': True
            }
            enc_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in enc_ids]
            enc_dict = torch.utils.data.dataloader.default_collate(enc_dicts)
            del enc_dicts

            features_to_load = {
                'load_image': True, # but not used, needed only for computing 'load_voxceleb1_crop'
                'load_voxceleb1_crop': True
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['enc_rgbs'] = enc_dict['image_cropped_voxceleb1']
            data_dict['pose_input_rgbs'] = dec_dict['image_cropped_voxceleb1']
            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image_cropped_voxceleb1']

            target_dict['label'] = self.dirlist.index[index]

        data_dict['pose_input_rgbs'] = self.augmenter.augment_tensor(data_dict['pose_input_rgbs'])

        return data_dict, target_dict

import torch
import torch.utils.data
import numpy as np
from .common import voxceleb, augmentation

from pathlib import Path

class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default='', type=Path)
        parser.add('--img_dir', default='Img', type=Path)
        parser.add('--kp_dir', default='landmarks', type=Path)
        parser.add('--draw_oval', default=True, action="store_bool")

        parser.add('--n_frames_for_encoder', default=8, type=int)

        parser = augmentation.get_args(parser)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = voxceleb.SampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, deterministic=part != 'train')

        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2Dataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size, augmenter)

        return dataset


class VoxCeleb2Dataset(voxceleb.VoxCeleb2Dataset):
    def __getitem__(self, index):
        data_dict, target_dict = {}, {}

        row = self.dirlist.iloc[index]
        path = row['path']

        finetuning = 'file' in self.dirlist
        if finetuning:
            # We are doing fine-tuning (`self.dirlist` enumerates all images, not just identities)
            dec_ids = [row['file']]

            features_to_load = {
                'load_image': True,
                'load_stickman': True,
                'load_keypoints': True,
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['target_rgbs'] = dec_dict['image']
            data_dict['pose_input_rgbs'] = dec_dict['image']
            data_dict['dec_stickmen'] = dec_dict['stickman']
            data_dict['dec_keypoints'] = dec_dict['keypoints']
            # Also putting `enc_*` stuff for embedding pre-calculation before fune-tuning
            data_dict['enc_stickmen'] = dec_dict['stickman']
            data_dict['enc_rgbs'] = dec_dict['image']

            target_dict['label'] = 0
        else:
            # We are doing normal training, in which "one sample" is one video
            ids = self.loader.list_ids(path, self.n_frames_for_encoder+1)

            enc_ids = ids[:-1]
            dec_ids = ids[-1:]

            features_to_load = {
                'load_image': True,
                'load_stickman': True
            }
            enc_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in enc_ids]
            enc_dict = torch.utils.data.dataloader.default_collate(enc_dicts)
            del enc_dicts

            features_to_load = {
                'load_image': True,
                'load_stickman': True,
                'load_keypoints': True,
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['enc_stickmen'] = enc_dict['stickman']
            data_dict['enc_rgbs'] = enc_dict['image']
            data_dict['target_rgbs'] = dec_dict['image']
            data_dict['pose_input_rgbs'] = dec_dict['image']
            data_dict['dec_stickmen'] = dec_dict['stickman']
            data_dict['dec_keypoints'] = dec_dict['keypoints']

            target_dict['label'] = self.dirlist.index[index]

        if not self.augmenter.is_empty():
            raise NotImplementedError("Keypoints augmentation is NYI")
        data_dict['pose_input_rgbs'] = self.augmenter.augment_tensor(data_dict['pose_input_rgbs'])

        return data_dict, target_dict

import imgaug
import imgaug.augmenters as iaa
import torch
import numpy as np

from contextlib import contextmanager

# Fix library version conflict
if hasattr(np.random, "_bit_generator"):
    np.random.bit_generator = np.random._bit_generator

import logging
logger = logging.getLogger('dataloaders.augmentation')

def get_args(parser):
    parser.add('--use_pixelwise_augs', action='store_bool', default=False)
    parser.add('--use_affine_scale', action='store_bool', default=False)
    parser.add('--use_affine_shift', action='store_bool', default=False)

    return parser


def get_augmentation_seq(args):
    return ParametricAugmenter(args)

class ParametricAugmenter:
    def is_empty(self):
        return not self.seq and not self.shift_seq

    def __init__(self, args):
        if args.inference:
            logger.info(f"`args.inference` is set, so switching off all augmentations")
            self.seq = self.shift_seq = None
            return

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        logger.info(f"Pixelwise augmentation: {args.use_pixelwise_augs}")
        logger.info(f"Affine scale augmentation: {args.use_affine_scale}")
        logger.info(f"Affine shift augmentation: {args.use_affine_shift}")

        total_augs = []

        if args.use_pixelwise_augs:
            pixelwise_augs = [
                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 0.25), n_segments=(150, 200))),
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(1, 3)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(1, 3)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(1.0, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.5)),  # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.BlendAlphaSimplexNoise(
                                   iaa.EdgeDetect(alpha=(0.0, 0.15)),
                               ),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=False),
                               # add gaussian noise to images
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToSaturation((-20, 20)),  # change hue and saturation
                               iaa.JpegCompression((70, 99)),

                               iaa.Multiply((0.5, 1.5), per_channel=False),

                               iaa.OneOf([
                                   iaa.LinearContrast((0.75, 1.25), per_channel=False),
                                   iaa.SigmoidContrast(cutoff=0.5, gain=(3.0, 11.0))
                               ]),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.15)),
                               # move pixels locally around (with random strengths)
                           ],
                           random_order=True
                           )
            ]
            total_augs.extend(pixelwise_augs)

        if args.use_affine_scale:
            affine_augs_scale = [sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                order=[1],  # use  bilinear interpolation (fast)
                mode=["reflect"]
            ))]
            total_augs.extend(affine_augs_scale)

        if args.use_affine_shift:
            affine_augs_shift = [sometimes(iaa.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                order=[1],  # use bilinear interpolation (fast)
                mode=["reflect"]
            ))]
        else:
            affine_augs_shift = []

        self.shift_seq = iaa.Sequential(affine_augs_shift)
        self.seq = iaa.Sequential(total_augs, random_order=True)

    def tensor2image(self, image, norm = 255.0):
        return (np.expand_dims(image.squeeze().permute(1, 2, 0).numpy(), 0) * norm)

    def image2tensor(self, image, norm = 255.0):
        image = image.astype(np.float32) / norm
        image = torch.tensor(np.squeeze(image)).permute(2, 0, 1).unsqueeze(0)
        return image


    def augment_tensor(self, image):
        if self.seq or self.shift_seq:
            image = self.tensor2image(image).astype(np.uint8)
            image = self.seq(images=image)
            image = self.shift_seq(images=image,)
            image = self.image2tensor(image)

        return image

    def augment_triplet(self, image1, image2, segmentation):
        if self.seq or self.shift_seq:
            image1 = self.tensor2image(image1).astype(np.uint8)

            image1 = self.seq(images=image1,)

            if self.shift_seq:
                image2 = self.tensor2image(image2).astype(np.uint8)
                segmentation = self.tensor2image(segmentation, 1.0).astype(np.float32)

                shift_seq_deterministic = self.shift_seq.to_deterministic()
                image1 = shift_seq_deterministic(images=image1,)
                image2, segmentation = shift_seq_deterministic(images=image2, heatmaps=segmentation)

                image2 = self.image2tensor(image2)
                segmentation = self.image2tensor(segmentation, 1.0)

            image1 = self.image2tensor(image1)

        return image1, image2, segmentation

    @contextmanager
    def deterministic_(self, seed):
        """
        A context manager to pre-define the random state of all augmentations.

        seed:
            `int`
        """
        # Backup the random states
        old_seq = self.seq.deepcopy()
        old_shift_seq = self.shift_seq.deepcopy()
        self.seq.seed_(seed)
        self.shift_seq.seed_(seed)
        yield
        # Restore the backed up random states
        self.seq = old_seq
        self.shift_seq = old_shift_seq

import torch
import torch.utils.data

import numpy as np
import cv2
import scipy

import random
from pathlib import Path

import logging # standard Python logging
logger = logging.getLogger('dataloader')

def get_part_data(args, part):
    """
    Load a list of VoxCeleb identities as a pandas dataframe (an identity is currently defined
    by a folder with images). Or, if `args.finetuning` is `True`, load a list of images for
    that identity.

    args:
        `argparse.Namespace` or any namespace
        Configuration arguments from 'train.py' launch.
    part:
        `str`
        'train' or 'val'.

    return:
    part_data:
        `pandas.DataFrame`, columns == ('path'[, 'file'])
    """
    logger = logging.getLogger(f"dataloaders.common.voxceleb.get_part_data ({part})")

    import pandas
    assert part in ('train', 'val'), "`.get_part_data()`'s `part` argument must be 'train' or 'val'"
    split_path = args.train_split_path if part == 'train' else args.val_split_path
    
    logger.info(f"Determining the '{part}' data source")

    def check_data_source_one_identity_path():
        logger.info(f"Checking if '{args.data_root / args.img_dir / split_path}' is a directory...")
        if (args.data_root / args.img_dir / split_path).is_dir():
            logger.info(f"Yes, it is; the only {part} identity will be '{split_path}'")
            return pandas.DataFrame({'path': [str(split_path)]})
        else:
            logger.info(f"No, it isn't")
            return None

    def check_data_source_identity_list_file():
        logger.info(f"Checking if '{split_path}' is a file...")
        if split_path.is_file():
            logger.info(f"Yes, it is; reading {part} identity list from it")
            return pandas.read_csv(split_path)
        else:
            logger.info(f"No, it isn't")
            return None

    def check_data_source_folder_with_identities():
        logger.info(f"Checking if '{args.data_root / args.img_dir}' is a directory...")
        if (args.data_root / args.img_dir).is_dir():
            identity_list = pandas.DataFrame({'path':
                sorted(str(x.relative_to(args.data_root)) for x in (args.data_root / args.img_dir).iterdir() if x.is_dir())
            })
            logger.info(f"Yes, it is; found {len(identity_list)} {part} identities in it")
            return identity_list
        else:
            logger.info(f"No, it isn't")
            return None

    check_data_source_functions = [
        check_data_source_one_identity_path,
        check_data_source_identity_list_file,
        check_data_source_folder_with_identities,
    ]
    for check_data_source in check_data_source_functions:
        identity_list = check_data_source()
        if identity_list is not None:
            break
    else:
        raise ValueError(
            f"Could not determine input data source, check `args.data_root`" \
            f", `args.img_dir` and `args.{part}_split_path")

    if args.finetune:
        if len(identity_list) > 1:
            raise NotImplementedError("Sorry, fine-tuning to multiple identities is not yet available")

        # In fine-tuning, let the dataframe hold paths to all images instead of identities
        from itertools import chain
        image_list = ((args.data_root / args.img_dir / x).iterdir() for x in identity_list['path'])
        image_list = sorted(chain(*image_list))
        logger.info(f"This dataset has {len(image_list)} images")

        args.num_labels = 1
        logger.info(f"Setting `args.num_labels` to 1 because we are fine-tuning or the model has been fine-tuned")

        retval = pandas.DataFrame({
            'path': [str(path.parent.relative_to(args.data_root / args.img_dir)) for path in image_list],
            'file': [path.stem                                                   for path in image_list]
        })
    else:
        # Make identity_list length exactly divisible by `world_size` so that epochs are synchronized among processes
        if args.checkpoint_path != "":
            logger.info(f"Truncating the identity list as in the checkpoint, to {args.num_labels} samples")
            identity_list = identity_list.iloc[:args.num_labels]
        else:
            if part == 'train':
                args.num_labels = len(identity_list)
                logger.info(f"Setting `args.num_labels` to {args.num_labels}")

        # Append some `identity_list`'s items to itself so that it's divisible by world size
        num_samples_to_add = (args.world_size - len(identity_list) % args.world_size) % args.world_size
        logger.info(
            f"Making dataset length divisible by world size: was {len(identity_list)}"
            f", became {len(identity_list) + num_samples_to_add}")
        retval = identity_list.append(identity_list.iloc[:num_samples_to_add])
    
    return retval

class SampleLoader:
    def __init__(
        self, data_root, img_dir=None, kp_dir=None,
        draw_oval=True, deterministic=False):

        self.data_root = data_root
        self.img_dir = img_dir
        self.kp_dir = kp_dir

        self.edges_parts, self.closed_parts, self.colors_parts = [], [], []

        # For drawing stickman    
        if draw_oval:
            self.edges_parts.append(list(range(0, 17)))
            self.closed_parts.append(False)
            self.colors_parts.append((255, 255, 255))

        self.edges_parts.extend([
            list(range(17, 22)),
            list(range(22, 27)),
            list(range(27, 31)),
            list(range(31, 36)),
            list(range(36, 42)),
            list(range(42, 48)),
            list(range(48, 60))])
        self.closed_parts.extend([False, False, False, False, True, True, True])
        self.colors_parts.extend([
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 0, 255),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 0)])

        self.deterministic = deterministic

    def list_ids(self, path, k):
        """
            path:
                str
                "{person_id}/{video_hash_string}/"
            k:
                int
                how many frames to sample from this video
        """
        full_path = self.data_root / self.img_dir / path
        id_list = list(full_path.iterdir())
        random_generator = random.Random(666) if self.deterministic else random

        while k > len(id_list):
            # just in case (unlikely) when we need to sample more frames than there are in this video
            id_list += list(full_path.iterdir())

        return [i.stem for i in random_generator.sample(id_list, k=k)]

    @staticmethod
    def calc_qsize(lm):
        lm_eye_left = lm[36: 42, :2]  # left-clockwise
        lm_eye_right = lm[42: 48, :2]  # left-clockwise
        lm_mouth_outer = lm[48: 60, :2]  # left-clockwise

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        qsize = np.hypot(*x) * 2
        return qsize

    @staticmethod
    def pad_img(image, keypoints):
        if keypoints is None:
            return image, keypoints

        h, w, _ = image.shape
        qsize = SampleLoader.calc_qsize(keypoints)

        if w > h:
            pad_h = w - h
            pad_w = 0
        else:
            pad_h = 0
            pad_w = h - w

        image = np.pad(np.float32(image), ((pad_h, 0), (pad_w, 0), (0, 0)), 'reflect')
        keypoints[:, 1] += pad_h
        keypoints[:, 0] += pad_w

        h, w, _ = image.shape
        y, x, _ = np.ogrid[:h, :w, :1]

        pad = np.array([pad_w, pad_h, 0, 0]).astype(np.float32)
        pad[pad == 0] = 1e-10

        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        image += (np.median(image, axis=(0, 1)) - image) * np.clip(mask, 0.0, 1.0)

        return image, keypoints

    @staticmethod
    def make_image_square(image, keypoints):
        h, w, _ = image.shape

        if abs(h - w) > 1:
            image, keypoints = SampleLoader.pad_img(image, keypoints)

        if h - w == 1:
            image = image[:-1]
        elif h - w == -1:
            image = image[:, :-1]

        assert image.shape[0] == image.shape[1]
        return image, keypoints

    def load_rgb(self, path, i):
        img_path = self.data_root / self.img_dir / path / (i + '.jpg')
        image = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Couldn't load image {img_path}")
            image = np.zeros((1, 1), dtype=np.uint8)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)

        return image

    def load_keypoints(self, path, i):
        keypoints_path = self.data_root / self.kp_dir / path / (i + '.npy')
        keypoints = np.load(keypoints_path)[:, :2]

        return keypoints

    def draw_stickman(self, image_shape, keypoints):
        stickman = np.zeros(image_shape + (3,), np.uint8)

        for edges, closed, color in zip(self.edges_parts, self.closed_parts, self.colors_parts):
            cv2.polylines(stickman, [keypoints.round()[edges].astype(np.int32)], closed, color, thickness=2)

        return stickman

    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_stickman=False,
                    load_keypoints=False):

        retval = {}

        if load_image:
            image = self.load_rgb(path, i)
            resize_ratio = imsize / image.shape[1]

        if load_stickman or load_keypoints:
            assert load_image, \
                "Please also require the image if you need keypoints, because we need to know what are H and W"
            keypoints = self.load_keypoints(path, i)

            keypoints *= resize_ratio

        if load_image:
            image = cv2.resize(image, (imsize, imsize),
                interpolation=cv2.INTER_CUBIC if resize_ratio > 1.0 else cv2.INTER_AREA)
            retval['image'] = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        # image, keypoints = self.make_image_square(image, keypoints) # temporarily disabled

        if load_stickman:
            stickman = self.draw_stickman(image.shape[:2], keypoints)
            retval['stickman'] = torch.tensor(stickman.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_keypoints:
            retval['keypoints'] = torch.tensor(keypoints.astype(np.float32).flatten() / imsize)

        return retval


class VoxCeleb2Dataset(torch.utils.data.Dataset):
    def __init__(self, dirlist, loader, inference, n_frames_for_encoder, imsize, augmenter):
        self.loader = loader
        self.inference = inference
        self.dirlist = dirlist
        self.imsize = imsize
        self.n_frames_for_encoder = n_frames_for_encoder
        self.augmenter = augmenter

        # Temporary code for taking identity and pose from different people for visualization
        self.identity_to_labels = {}
        for record in self.dirlist.itertuples():
            identity = record.path[:7]
            if identity not in self.identity_to_labels:
                self.identity_to_labels[identity] = []
            self.identity_to_labels[identity].append(record.Index)

    # Temporary code for taking identity and pose from different people for visualization
    def get_other_sample_by_label(self, label, same_identity=False, deterministic=True):
        """
            label:
                `int`
                pandas index (="label") of a sample in dataset (e.g. the one
                from `data_dict['label']`).
            same_identity:
                `bool`
                See "return".
            deterministic:
                `bool`
                Return the index of the next sample, not a random one.

            return:
                `int`
                dataset (!) index of a random sample that has the SAME person but in a DIFFERENT
                video sequence. If `same_identity` is `False`, the person will also be DIFFERENT.
        """
        identity = self.dirlist.loc[label].path[:7]
        # All frames of the given person, including other videos
        labels_for_this_identity = self.identity_to_labels[identity]
        retval_index = 0
        if same_identity:
            while True:
                if not deterministic:
                    # Pick a random frame, but other than the given one
                    retval_label = random.choice(labels_for_this_identity)
                else:
                    # Pick next sutable frame, but other than the given one
                    retval_label = labels_for_this_identity[retval_index]
                    retval_index = retval_index + 1
                    
                if retval_label != label or len(labels_for_this_identity) == 1:
                    break

            return self.dirlist.index.get_loc(retval_label)
        else:
            retval_label = labels_for_this_identity[0]
            retval_index = self.dirlist.index.get_loc(retval_label)
            while True:
                if not deterministic:
                    # Pick a random frame, making sure there is other person in it
                    retval_index = random.randint(0, len(self) - 1)
                else:
                    # Pick next sutable frame, but other than the given one
                    if retval_index < self.dirlist.shape[0] - 1:
                        retval_index = retval_index + 1
                    else:
                        retval_index = 0
                        
                if self.dirlist.iloc[retval_index].path[:7] != identity or len(labels_for_this_identity) == len(self):
                    break
                
            return retval_index

    def __len__(self):
        return self.dirlist.shape[0]

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from utils.utils import load_module

import logging
logger = logging.getLogger('dataloaders.dataloader')

class Dataloader:
    def __init__(self, dataset_name):
        self.dataset = self.find_definition(dataset_name)

    def find_definition(self, dataset_name):
        m = load_module('dataloaders', dataset_name)
        return m.__dict__['Dataset']

    def get_args(self, parser):
        parser.add('--num_workers', type=int, default=4, help='Number of data loading workers.')
        parser.add('--prefetch_size', type=int, default=16, help='Prefetch queue size')
        parser.add('--batch_size', type=int, default=64, help='Batch size')

        return self.dataset.get_args(parser)

    def get_dataloader(self, args, part, phase):
        if hasattr(self.dataset, 'get_dataloader'):
            return self.dataset.get_dataloader(args, part)
        else:
            dataset = self.dataset.get_dataset(args, part)
            # Get a split for this process in distributed training
            assert len(dataset) % args.world_size == 0, \
                "`dataset.get_dataset()` was expected to return a dataset equally divisible by `args.world_size`"
            dataset = torch.utils.data.Subset(dataset, range(args.rank, len(dataset), args.world_size))

            logger.info(f"This process will receive a dataset with {len(dataset)} samples")

            if len(dataset) < args.batch_size: # can happen at fine-tuning
                logger.warning(
                    f"Dataset length is smaller than batch size ({len(dataset)} < {args.batch_size})" \
                    f", reducing the latter to {len(dataset)}")
                args.batch_size = len(dataset)

            return DataLoaderWithPrefetch(
                dataset,
                prefetch_size=args.prefetch_size,
                batch_size=args.batch_size // args.num_gpus,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True if phase == 'train' else False,
                shuffle=True if part == 'train' else False)


class DataLoaderWithPrefetch(DataLoader):
    def __init__(self, *args, prefetch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_size = prefetch_size if prefetch_size is not None else 2 * kwargs.get('num_workers', 0)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIterWithPrefetch(self)


class _MultiProcessingDataLoaderIterWithPrefetch(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self.prefetch_size = loader.prefetch_size

        super().__init__(loader)

        # Prefetch more items than the default 2 * self._num_workers
        assert self.prefetch_size >= 2 * self._num_workers
        for _ in range(loader.prefetch_size - 2 * self._num_workers):
            self._try_put_index()

    def _try_put_index(self):
        assert self._tasks_outstanding < self.prefetch_size
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

import torch
import torch.utils.data
import numpy as np
import cv2

from .common import voxceleb, augmentation
from .voxceleb2 import VoxCeleb2Dataset, Dataset as ParentDataset

from pathlib import Path

class Dataset(ParentDataset):
    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = SmallCropSampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, deterministic=part != 'train')

        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2Dataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size, augmenter)

        return dataset

class SmallCropSampleLoader(voxceleb.SampleLoader):
    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_stickman=False,
                    load_keypoints=False):

        retval = {}

        if load_image:
            image = self.load_rgb(path, i)
            original_image_size = image.shape

            cut_t, cut_b = 0.2, 1.0
            cut_l = (1.0 - (cut_b - cut_t)) / 2
            cut_r = 1.0 - cut_l

            cut_t = min(image.shape[0]-1, round(cut_t * image.shape[0]))
            cut_l = min(image.shape[1]-1, round(cut_l * image.shape[1]))
            cut_b = max(cut_t+1, round(cut_b * image.shape[0]))
            cut_r = max(cut_l+1, round(cut_r * image.shape[1]))

            image = image[cut_t:cut_b, cut_l:cut_r]

        if load_stickman or load_keypoints:
            assert load_image, \
                "Please also require the image if you need keypoints, because we need to know what are H and W"
            keypoints = self.load_keypoints(path, i)

            keypoints -= [[cut_l, cut_t]]
            keypoints *= [[imsize / (cut_r - cut_l), imsize / (cut_b - cut_t)]]

        if load_image:
            image = cv2.resize(image, (imsize, imsize),
                interpolation=cv2.INTER_CUBIC if imsize > image.shape[0] else cv2.INTER_AREA)
            retval['image'] = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        # image, keypoints = self.make_image_square(image, keypoints) # temporarily disabled

        if load_stickman:
            stickman = self.draw_stickman(image.shape[:2], keypoints)
            retval['stickman'] = torch.tensor(stickman.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_keypoints:
            retval['keypoints'] = torch.tensor(keypoints.astype(np.float32).flatten() / imsize)

        return retval

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

import logging # standard Python logging
logger = logging.getLogger('embedder')

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--X2Face_num_identity_images', type=int, default=3)

    @staticmethod
    def get_net(args):
        assert not args.weights_running_average, "Please set `weights_running_average: false` with X2Face"
        net = Generator(args.X2Face_num_identity_images)
        return net.to(args.device)

class Generator(nn.Module):
    def __init__(self, num_identity_images):
        super().__init__()

        self.identity_images = nn.Parameter(torch.empty(num_identity_images, 3, 256, 256))

        import sys
        X2FACE_ROOT_DIR = "embedders/X2Face"
        sys.path.append(f"{X2FACE_ROOT_DIR}/UnwrapMosaic/")
        try:
            from UnwrappedFace import UnwrappedFaceWeightedAverage
            state_dict = torch.load(
                f"{X2FACE_ROOT_DIR}/models/x2face_model_forpython3.pth", map_location='cpu')
        except (ImportError, FileNotFoundError):
            logger.critical(
                f"Please initialize submodules, then download 'x2face_model_forpython3.pth' from "
                f"http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_x2face_eccv_withpy3.zip"
                f" and put it into {X2FACE_ROOT_DIR}/models/")
            raise

        self.x2face_model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
        self.x2face_model.load_state_dict(state_dict['state_dict'])
        self.x2face_model.eval()

        # Forbid doing .train(), .eval() and .parameters()
        def train_noop(self, *args, **kwargs): pass
        def parameters_noop(self, *args, **kwargs): return []
        self.x2face_model.train = train_noop.__get__(self.x2face_model, nn.Module)
        self.x2face_model.parameters = parameters_noop.__get__(self.x2face_model, nn.Module)

        # Disable saving weights
        def state_dict_empty(self, *args, **kwargs): return {}
        self.x2face_model.state_dict = state_dict_empty.__get__(self.x2face_model, nn.Module)
        # Forbid loading weights after we have done that
        def _load_from_state_dict_noop(self, *args, **kwargs): pass
        for module in self.x2face_model.modules():
            module._load_from_state_dict = _load_from_state_dict_noop.__get__(module, nn.Module)

        self.finetuning = False

    @torch.no_grad()
    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to generator architecture to allow fine-tuning.
            For `vanilla` generator, initialize AdaIN parameters from `data_dict['embeds']`
            and flag them as trainable parameters.
            Will require re-initializing optimizer, but only after the first call.

            data_dict:
                dict
                Required contents depend on the specific generator. For `vanilla` generator,
                it is `'embeds'` (1 x `args.embed_channels`).
                If `None`, the module's new parameters will be initialized randomly.
        """
        if data_dict is not None:
            self.identity_images = nn.Parameter(data_dict['enc_rgbs'][0]) # N x C x H x W

        self.finetuning = True

    @torch.no_grad()
    def forward(self, data_dict):
        batch_size = len(data_dict['pose_input_rgbs'])
        outputs = torch.empty_like(data_dict['pose_input_rgbs'][:, 0])

        for batch_idx in range(batch_size):
            # N x C x H x W
            identity_images = self.identity_images if self.finetuning else data_dict['enc_rgbs'][batch_idx]
            identity_images_list = []
            for identity_image in identity_images:
                identity_images_list.append(identity_image[None])
                
            # C x H x W
            pose_driver = data_dict['pose_input_rgbs'][batch_idx, 0]
            driver_images = pose_driver[None]

            result = self.x2face_model(driver_images, *identity_images_list)
            result = result.clamp(min=0, max=1)

            outputs[batch_idx].copy_(result[0])

        data_dict['fake_rgbs'] = outputs
        outputs.requires_grad_()
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

import math

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gen_constant_input_size', type=int, default=4)
        parser.add('--gen_num_residual_blocks', type=int, default=2)

        parser.add('--gen_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--norm_layer', type=str, default='in')

    @staticmethod
    def get_net(args):
        # backward compatibility
        if 'gen_constant_input_size' not in args:
            args.gen_constant_input_size = 4

        net = Generator(
            args.gen_padding, args.in_channels, args.out_channels+1,
            args.num_channels, args.max_num_channels, args.embed_channels, args.pose_embedding_size,
            args.norm_layer, args.gen_constant_input_size, args.gen_num_residual_blocks,
            args.image_size)
        return net.to(args.device)


class Constant(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(1, *shape))

    def forward(self, batch_size):
        return self.constant.expand((batch_size,) + self.constant.shape[1:])


class Generator(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, identity_embedding_size,
        pose_embedding_size, norm_layer, gen_constant_input_size, gen_num_residual_blocks, output_image_size):
        super().__init__()

        def get_res_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer=norm_layer)

        def get_up_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=True, downsample=False,
                                   norm_layer=norm_layer)

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d
        else:
            raise Exception('Incorrect `padding` argument, required `zero` or `reflection`')

        assert math.log2(output_image_size / gen_constant_input_size).is_integer(), \
            "`gen_constant_input_size` must be `image_size` divided by a power of 2"
        num_upsample_blocks = int(math.log2(output_image_size / gen_constant_input_size))
        out_channels_block_nonclamped = num_channels * (2 ** num_upsample_blocks)
        out_channels_block = min(out_channels_block_nonclamped, max_num_channels)

        self.constant = Constant(out_channels_block, gen_constant_input_size, gen_constant_input_size)
        current_image_size = gen_constant_input_size

        # Decoder
        layers = []
        for i in range(gen_num_residual_blocks):
            layers.append(get_res_block(out_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        
        for _ in range(num_upsample_blocks):
            in_channels_block = out_channels_block
            out_channels_block_nonclamped //= 2
            out_channels_block = min(out_channels_block_nonclamped, max_num_channels)
            layers.append(get_up_block(in_channels_block, out_channels_block, padding, 'ada' + norm_layer))

        layers.extend([
            blocks.AdaptiveNorm2d(out_channels_block, norm_layer),
            nn.ReLU(True),
            # padding(1),
            spectral_norm(
                nn.Conv2d(out_channels_block, out_channels, 3, 1, 1),
                eps=1e-4),
            nn.Tanh()
        ])
        self.decoder_blocks = nn.Sequential(*layers)

        self.adains = [module for module in self.modules() if module.__class__.__name__ == 'AdaptiveNorm2d']

        self.identity_embedding_size = identity_embedding_size
        self.pose_embedding_size = pose_embedding_size

        joint_embedding_size = identity_embedding_size + pose_embedding_size
        self.affine_params_projector = nn.Sequential(
            spectral_norm(nn.Linear(joint_embedding_size, max(joint_embedding_size, 512))),
            nn.ReLU(True),
            spectral_norm(nn.Linear(max(joint_embedding_size, 512), self.get_num_affine_params()))
        )

        self.finetuning = False

    def get_num_affine_params(self):
        return sum(2*module.num_features for module in self.adains)

    def assign_affine_params(self, affine_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                new_bias = affine_params[:, :m.num_features]
                new_weight = affine_params[:, m.num_features:2 * m.num_features]

                if m.bias is None: # to keep m.bias being `nn.Parameter`
                    m.bias = new_bias.contiguous()
                else:
                    m.bias.copy_(new_bias)

                if m.weight is None: # to keep m.weight being `nn.Parameter`
                    m.weight = new_weight.contiguous()
                else:
                    m.weight.copy_(new_weight)

                if affine_params.size(1) > 2 * m.num_features:
                    affine_params = affine_params[:, 2 * m.num_features:]

    def assign_embeddings(self, data_dict):
        if self.finetuning:
            identity_embedding = self.identity_embedding.expand(len(data_dict['pose_embedding']), -1)
        else:
            identity_embedding = data_dict['embeds']
            
        pose_embedding = data_dict['pose_embedding']
        joint_embedding = torch.cat((identity_embedding, pose_embedding), dim=1)

        affine_params = self.affine_params_projector(joint_embedding)
        self.assign_affine_params(affine_params)

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to generator architecture to allow fine-tuning.
            For `vanilla` generator, initialize AdaIN parameters from `data_dict['embeds']`
            and flag them as trainable parameters.
            Will require re-initializing optimizer, but only after the first call.

            data_dict:
                dict
                Required contents depend on the specific generator. For `vanilla` generator,
                it is `'embeds'` (1 x `args.embed_channels`).
                If `None`, the module's new parameters will be initialized randomly.
        """
        if data_dict is None:
            some_parameter = next(iter(self.parameters())) # to know target device and dtype
            identity_embedding = torch.rand(1, self.identity_embedding_size).to(some_parameter)
        else:
            identity_embedding = data_dict['embeds']

        if self.finetuning:
            with torch.no_grad():
                self.identity_embedding.copy_(identity_embedding)
        else:
            self.identity_embedding = nn.Parameter(identity_embedding)
            self.finetuning = True

    def forward(self, data_dict):
        self.assign_embeddings(data_dict)

        batch_size = len(data_dict['pose_embedding'])
        outputs = self.decoder_blocks(self.constant(batch_size))
        rgb, segmentation = outputs[:, :-1], outputs[:, -1:]

        # Move tanh's output from (-1; 1) to (-0.25; 1.25)
        rgb = rgb * 0.75
        rgb += 0.5

        # Same, but to (0; 1)
        segmentation = segmentation * 0.5
        segmentation += 0.5

        data_dict['fake_rgbs'] = rgb * segmentation
        data_dict['fake_segm'] = segmentation

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class AdaptiveNorm2d(nn.Module):
    def __init__(self, num_features, norm_layer='in', eps=1e-4):
        super(AdaptiveNorm2d, self).__init__()
        self.num_features = num_features
        self.weight = self.bias = None
        if 'in' in norm_layer:
            self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)
        elif 'bn' in norm_layer:
            self.norm_layer = SyncBatchNorm(num_features, momentum=1.0, eps=eps, affine=False)

        self.delete_weight_on_forward = True

    def forward(self, input):
        out = self.norm_layer(input)
        output = out * self.weight[:, :, None, None] + self.bias[:, :, None, None]

        # To save GPU memory
        if self.delete_weight_on_forward:
            self.weight = self.bias = None

        return output


class AdaptiveNorm2dTrainable(nn.Module):
    def __init__(self, num_features, norm_layer='in', eps=1e-4):
        super(AdaptiveNorm2dTrainable, self).__init__()
        self.num_features = num_features
        if 'in' in norm_layer:
            self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)

    def forward(self, input):
        out = self.norm_layer(input)
        t = out.shape[0] // self.weight.shape[0]
        output = out * self.weight + self.bias
        return output

    def assign_params(self, weight, bias):
        self.weight = torch.nn.Parameter(weight.view(1, -1, 1, 1))
        self.bias = torch.nn.Parameter(bias.view(1, -1, 1, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU, gated=False):
        super(ResBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        # if norm_layer == 'bn':
        #     # norm0 = SyncBatchNorm(in_channels, momentum=1.0, eps=1e-4)
        #     # norm1 = SyncBatchNorm(out_channels, momentum=1.0, eps=1e-4)
        #     pass
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')

        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            nn.Sequential() if padding is nn.ZeroPad2d else padding(1),
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1 if padding is nn.ZeroPad2d else 0, bias=bias),
                eps=1e-4)])
        if normalize:
            layers.append(norm1)
        layers.extend([
            activation(inplace=True),
            nn.Sequential() if padding is nn.ZeroPad2d else padding(1),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1 if padding is nn.ZeroPad2d else 0, bias=bias),
                eps=1e-4)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1),
                eps=1e-4))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output

class channelShuffle(nn.Module):
    def __init__(self,groups):
        super(channelShuffle, self).__init__()
        self.groups=groups
        
    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()

#         batchsize = x.shape[0]
#         num_channels = x.shape[1]
#         height = x.shape[2]
#         width = x.shape[3]

        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    

class shuffleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(shuffleConv, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        self.padding=padding
        groups=4
        block=[]
        if (in_channels%groups==0) and (out_channels%groups==0):
            block.append(spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,padding=0, groups=groups),eps=1e-4))
            block.append(nn.ReLU6(inplace=True))
            block.append(channelShuffle(groups=groups))
            block.append(spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,padding=1, groups=groups),eps=1e-4))
            block.append(nn.ReLU6(inplace=True))
            block.append(spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=1,padding=0, groups=groups),eps=1e-4))
        else:
            block.append(spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1),eps=1e-4))
        self.block=nn.Sequential(*block)
            
    def forward(self,x):
        x=self.block(x)
        return x    
    
    
class ResBlockShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU, gated=False):
        super(ResBlockShuffle, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        # if norm_layer == 'bn':
        #     # norm0 = SyncBatchNorm(in_channels, momentum=1.0, eps=1e-4)
        #     # norm1 = SyncBatchNorm(out_channels, momentum=1.0, eps=1e-4)
        #     pass
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')

        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            #padding(1),
            #spectral_norm(
                shuffleConv(in_channels, out_channels, 3, 1, 0, bias=bias)#,
            #    eps=1e-4)
        ])
        if normalize:
            layers.append(norm1)
        layers.extend([
            activation(inplace=True),
            #padding(1),
            #spectral_norm(
                shuffleConv(out_channels, out_channels, 3, 1, 0, bias=bias)#,
            #    eps=1e-4)
        ])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(
                #spectral_norm(
                shuffleConv(in_channels, out_channels, 1)#,
            #    eps=1e-4)
            )
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output
    
    

class ResBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, 
                 resize_layer, norm_layer, activation):
        super(ResBlockV2, self).__init__()
        upsampling_layers = {
            'nearest': lambda: nn.Upsample(scale_factor=stride, mode='nearest')
        }
        downsampling_layers = {
            'avgpool': lambda: nn.AvgPool2d(stride)
        }
        norm_layers = {
            'bn': lambda num_features: SyncBatchNorm(num_features, momentum=1.0, eps=1e-4),
            'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=1e-4, affine=True),
            'adabn': lambda num_features: AdaptiveNorm2d(num_features, 'bn'),
            'adain': lambda num_features: AdaptiveNorm2d(num_features, 'in')
        }
        normalize = norm_layer != 'none'
        bias = not normalize
        upsample = resize_layer in upsampling_layers
        downsample = resize_layer in downsampling_layers
        if normalize: 
            norm_layer = norm_layers[norm_layer]

        layers = []
        if normalize:
            layers.append(norm_layer(in_channels))
        layers.append(activation())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias),
                eps=1e-4)])
        if normalize:
            layers.append(norm_layer(out_channels))
        layers.extend([
            activation(),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias),
                eps=1e-4)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1),
                eps=1e-4))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output

class ResBlockV2Shuffle(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, 
                 resize_layer, norm_layer, activation):
        super(ResBlockV2Shuffle, self).__init__()
        upsampling_layers = {
            'nearest': lambda: nn.Upsample(scale_factor=stride, mode='nearest')
        }
        downsampling_layers = {
            'avgpool': lambda: nn.AvgPool2d(stride)
        }
        norm_layers = {
            'bn': lambda num_features: SyncBatchNorm(num_features, momentum=1.0, eps=1e-4),
            'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=1e-4, affine=True),
            'adabn': lambda num_features: AdaptiveNorm2d(num_features, 'bn'),
            'adain': lambda num_features: AdaptiveNorm2d(num_features, 'in')
        }
        normalize = norm_layer != 'none'
        bias = not normalize
        upsample = resize_layer in upsampling_layers
        downsample = resize_layer in downsampling_layers
        if normalize: 
            norm_layer = norm_layers[norm_layer]

        layers = []
        if normalize:
            layers.append(norm_layer(in_channels))
        layers.append(activation())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            #spectral_norm(
                shuffleConv(in_channels, out_channels, 3, 1, 1, bias=bias)#,
            #    eps=1e-4)
        ])
        if normalize:
            layers.append(norm_layer(out_channels))
        layers.extend([
            activation(),
            #spectral_norm(
                shuffleConv(out_channels, out_channels, 3, 1, 1, bias=bias)#,
            #    eps=1e-4)
        ])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(#spectral_norm(
                shuffleConv(in_channels, out_channels, 1)#,
            #    eps=1e-4)
            )
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output
    
    

class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_fun, kernel_size, stride=1, padding=0, bias=True):
        super(GatedBlock, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                  eps=1e-4)
        self.gate = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                  eps=1e-4)
        self.act_fun = act_fun()
        self.gate_act_fun = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.act_fun(out)

        mask = self.gate(x)
        mask = self.gate_act_fun(mask)

        out_masked = out * mask
        return out_masked


class GatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU):
        super(GatedResBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')

        main_layers = []

        if normalize:
            main_layers.append(norm0)
        if upsample:
            main_layers.append(nn.Upsample(scale_factor=2))

        main_layers.extend([
            padding(1),
            GatedBlock(in_channels, out_channels, activation, 3, 1, 0, bias=bias)])

        if normalize:
            main_layers.append(norm1)
        main_layers.extend([
            padding(1),
            GatedBlock(out_channels, out_channels, activation, 3, 1, 0, bias=bias)])
        if downsample:
            main_layers.append(nn.AvgPool2d(2))

        self.main_pipe = nn.Sequential(*main_layers)

        self.skip_pipe = None
        if in_channels != out_channels or upsample or downsample:
            skip_layers = []
            
            if upsample:
                skip_layers.append(nn.Upsample(scale_factor=2))

            skip_layers.append(GatedBlock(in_channels, out_channels, activation, 1))

            if downsample:
                skip_layers.append(nn.AvgPool2d(2))
            self.skip_pipe = nn.Sequential(*skip_layers)

    def forward(self, input):
        mp_out = self.main_pipe(input)
        if self.skip_pipe is not None:
            output = mp_out + self.skip_pipe(input)
        else:
            output = mp_out + input
        return output


class ResBlockWithoutSpectralNorms(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU):
        super(ResBlockWithoutSpectralNorms, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        # if norm_layer == 'bn':
        #     # norm0 = SyncBatchNorm(in_channels, momentum=1.0, eps=1e-4)
        #     # norm1 = SyncBatchNorm(out_channels, momentum=1.0, eps=1e-4)
        #     pass
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')

        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            padding(1),
            # spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=bias)  # ,
            #    eps=1e-4)
        ])
        if normalize:
            layers.append(norm1)
        layers.extend([
            activation(inplace=True),
            padding(1),
            # spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, 1, 0, bias=bias)  # ,
            #    eps=1e-4)
        ])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(  # spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1)  # ,
                #    eps=1e-4)
            )
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU6, expansion_factor=6):
        super(MobileNetBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        conv0 = nn.Conv2d(in_channels, int(in_channels * expansion_factor), 1)
        dwise = nn.Conv2d(int(in_channels * expansion_factor), int(in_channels * expansion_factor), 3,
                          2 if downsample else 1, 1, groups=int(in_channels * expansion_factor))
        conv1 = nn.Conv2d(int(in_channels * expansion_factor), out_channels, 1)

        if norm_layer == 'bn':
            # norm0 = SyncBatchNorm(in_channels, momentum=1.0, eps=1e-4)
            # norm1 = SyncBatchNorm(out_channels, momentum=1.0, eps=1e-4)
            pass
        if 'in' in norm_layer:
            norm0 = nn.InstanceNorm2d(int(in_channels * expansion_factor), eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(int(in_channels * expansion_factor), eps=1e-4, affine=True)
            norm2 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        if 'ada' in norm_layer:
            norm2 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm2 = AdaptiveNorm2dTrainable(out_channels, norm_layer)

        # layers = [spectral_norm(conv0, eps=1e-4)]
        layers = [conv0]
        if normalize: layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample: layers.append(nn.Upsample(scale_factor=2))
        # layers.append(spectral_norm(dwise, eps=1e-4))
        layers.append(dwise)
        if normalize: layers.append(norm1)
        layers.extend([
            activation(inplace=True),
            # spectral_norm(
            conv1  # ,
            # eps=1e-4)
        ])
        if normalize: layers.append(norm2)
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample: layers.append(nn.Upsample(scale_factor=2))
            layers.append(
                # spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1)  # ,
                #    eps=1e-4)
            )
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(-1)

    def forward(self, input):
        b, c, h, w = input.shape
        query = self.query_conv(input).view(b, -1, h * w).permute(0, 2, 1)  # B x HW x C/8
        key = self.key_conv(input).view(b, -1, h * w)  # B x C/8 x HW
        energy = torch.bmm(query, key)  # B x HW x HW
        attention = self.softmax(energy)  # B x HW x HW
        value = self.value_conv(input).view(b, -1, h * w)  # B x C x HW

        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)
        output = self.gamma * out + input
        return output

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gen_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--gen_num_downsample_blocks', type=int, default=4)
        parser.add('--gen_num_residual_blocks', type=int, default=4)
        parser.add('--norm_layer', type=str, default='in')

    @staticmethod
    def get_net(args):
        net = Generator(
            args.gen_padding, args.in_channels, args.out_channels,
            args.num_channels, args.max_num_channels, args.embed_channels,
            args.norm_layer, args.gen_num_downsample_blocks, args.gen_num_residual_blocks)
        return net.to(args.device)


class Generator(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, embed_channels, norm_layer,
                 gen_num_downsample_blocks, gen_num_residual_blocks):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer=norm_layer)

        def get_res_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer=norm_layer)

        def get_up_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=True, downsample=False,
                                   norm_layer=norm_layer)

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d
        else:
            raise Exception('Incorrect `padding` argument, required `zero` or `reflection`')

        in_channels_block = in_channels

        # Encoder of inputs
        self.down_block = nn.Sequential(
            padding(1),
            spectral_norm(
                nn.Conv2d(in_channels_block, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.ReLU(),
            padding(1),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels_block, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))
        layers = []
        in_channels_block = num_channels
        for i in range(1, gen_num_downsample_blocks):
            out_channels_block = min(in_channels_block * 2, max_num_channels)
            layers.append(get_down_block(in_channels_block, out_channels_block, padding, norm_layer))
            in_channels_block = out_channels_block
        self.down_blocks = nn.Sequential(*layers)

        # Decoder
        layers = []
        for i in range(gen_num_residual_blocks):
            layers.append(get_res_block(out_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        for i in range(gen_num_downsample_blocks - 1, -1, -1):
            in_channels_block = out_channels_block
            out_channels_block = min(int(num_channels * 2 ** i), max_num_channels)
            layers.append(get_up_block(in_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        layers.extend([
            blocks.AdaptiveNorm2d(out_channels_block, norm_layer),
            nn.ReLU(True),
            padding(1),
            spectral_norm(
                nn.Conv2d(out_channels_block, out_channels, 3, 1, 0),
                eps=1e-4),
            nn.Tanh()])
        self.decoder_blocks = nn.Sequential(*layers)

        # self.project moved from embedder
        num_affine_params = self.get_num_affine_params()

        self.project = spectral_norm(
            nn.Linear(embed_channels, num_affine_params),
            eps=1e-4)

        self.finetuning = False

    def get_num_affine_params(self):
        num_affine_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                num_affine_params += 2 * m.num_features
        return num_affine_params

    def assign_affine_params(self, affine_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                new_bias = affine_params[:, :m.num_features]
                new_weight = affine_params[:, m.num_features:2 * m.num_features]

                if m.bias is None: # to keep m.bias being `nn.Parameter`
                    m.bias = new_bias.contiguous()
                else:
                    m.bias.copy_(new_bias)

                if m.weight is None: # to keep m.weight being `nn.Parameter`
                    m.weight = new_weight.contiguous()
                else:
                    m.weight.copy_(new_weight)

                if affine_params.size(1) > 2 * m.num_features:
                    affine_params = affine_params[:, 2 * m.num_features:]

    def assign_embeddings(self, data_dict):
        embedding = data_dict['embeds']
        affine_params = self.project(embedding)
        self.assign_affine_params(affine_params)

    def make_affine_params_trainable(self):
        """
            Used prior to fine-tuning.

            Flag `.weight` and `.bias` of all `AdaptiveNorm2d` layers as trainable parameters.
            After calling this function, the said tensors will be returned by `.parameters()`.
            Their values are set to those present at the time of calling this function, i.e.
            `.assign_embeddings()` (or `.assign_affine_params`) must be called beforehand.
        """
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                for field in m.weight, m.bias:
                    assert torch.is_tensor(field) and field.numel() == m.num_features, \
                        "One of `AdaptiveNorm2d`'s parameters is of wrong size or is None. " \
                        "Did you forget to call `.assign_embeddings()` (or `.assign_affine_params()`)?"

                m.weight = nn.Parameter(m.weight)
                m.bias   = nn.Parameter(m.bias)
                m.delete_weight_on_forward = False

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to generator architecture to allow fine-tuning.
            For `vanilla` generator, initialize AdaIN parameters from `data_dict['embeds']`
            and flag them as trainable parameters.
            Will require re-initializing optimizer, but only after the first call.

            data_dict:
                dict, optional
                Required contents depend on the specific generator. For `vanilla` generator,
                it is `'embeds'` (1 x `args.embed_channels`).
                If `None`, the module's new parameters will be initialized randomly.
        """
        was_training = self.training
        self.eval() # TODO use `args.set_eval_mode_in_test` instead of hard `True`

        if data_dict is None:
            some_parameter = next(iter(self.parameters())) # to know target device and dtype
            data_dict = {
                'embeds': torch.rand(1, self.project.in_features).to(some_parameter)
            }

        with torch.no_grad():
            self.assign_embeddings(data_dict)
        
        if not self.finetuning:
            self.make_affine_params_trainable()
            self.finetuning = True

        self.train(was_training)

    def forward(self, data_dict):
        if not self.finetuning: # made `True` in `.make_affine_params_trainable()` (e.g. for fine-tuning)
            self.assign_embeddings(data_dict)
        
        inputs = data_dict['dec_stickmen']
        if len(inputs.shape) > 4:
            inputs = inputs[:, 0]

        out = self.down_block(inputs)
        out = out + self.skip(inputs)
        out = self.down_blocks(out)
        # Decode
        outputs = self.decoder_blocks(out)

        data_dict['fake_rgbs'] = outputs

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

import math

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gen_constant_input_size', type=int, default=4)
        parser.add('--gen_num_residual_blocks', type=int, default=2)

        parser.add('--gen_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--norm_layer', type=str, default='in')

    @staticmethod
    def get_net(args):
        # backward compatibility
        if 'gen_constant_input_size' not in args:
            args.gen_constant_input_size = 4

        net = Generator(
            args.gen_padding, args.in_channels, args.out_channels+1,
            args.num_channels, args.max_num_channels, args.embed_channels, args.pose_embedding_size,
            args.norm_layer, args.gen_constant_input_size, args.gen_num_residual_blocks,
            args.image_size)
        return net.to(args.device)


class Constant(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(1, *shape))

    def forward(self, batch_size):
        return self.constant.expand((batch_size,) + self.constant.shape[1:])


class Generator(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, identity_embedding_size,
        pose_embedding_size, norm_layer, gen_constant_input_size, gen_num_residual_blocks, output_image_size):
        super().__init__()

        def get_res_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer=norm_layer)

        def get_up_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=True, downsample=False,
                                   norm_layer=norm_layer)

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d
        else:
            raise Exception('Incorrect `padding` argument, required `zero` or `reflection`')

        assert math.log2(output_image_size / gen_constant_input_size).is_integer(), \
            "`gen_constant_input_size` must be `image_size` divided by a power of 2"
        num_upsample_blocks = int(math.log2(output_image_size / gen_constant_input_size))
        out_channels_block_nonclamped = num_channels * (2 ** num_upsample_blocks)
        out_channels_block = min(out_channels_block_nonclamped, max_num_channels)

        self.constant = Constant(out_channels_block, gen_constant_input_size, gen_constant_input_size)
        current_image_size = gen_constant_input_size

        # Decoder
        layers = []
        for i in range(gen_num_residual_blocks):
            layers.append(get_res_block(out_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        
        for _ in range(num_upsample_blocks):
            in_channels_block = out_channels_block
            out_channels_block_nonclamped //= 2
            out_channels_block = min(out_channels_block_nonclamped, max_num_channels)
            layers.append(get_up_block(in_channels_block, out_channels_block, padding, 'ada' + norm_layer))

        layers.extend([
            blocks.AdaptiveNorm2d(out_channels_block, norm_layer),
            nn.ReLU(True),
            # padding(1),
            spectral_norm(
                nn.Conv2d(out_channels_block, out_channels, 3, 1, 1),
                eps=1e-4),
            nn.Tanh()
        ])
        self.decoder_blocks = nn.Sequential(*layers)

        self.adains = [module for module in self.modules() if module.__class__.__name__ == 'AdaptiveNorm2d']

        self.identity_embedding_size = identity_embedding_size
        self.pose_embedding_size = pose_embedding_size

        hidden_layer_size = max(512, pose_embedding_size + identity_embedding_size)
        self.affine_params_projector = nn.Sequential(
            nn.Linear(pose_embedding_size + identity_embedding_size, hidden_layer_size),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_layer_size, self.get_num_affine_params())
        )

        self.finetuning = False

    def get_num_affine_params(self):
        return sum(2*module.num_features for module in self.adains)

    def assign_affine_params(self, affine_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                new_bias = affine_params[:, :m.num_features]
                new_weight = affine_params[:, m.num_features:2 * m.num_features]

                if m.bias is None: # to keep m.bias being `nn.Parameter`
                    m.bias = new_bias.contiguous()
                else:
                    m.bias.copy_(new_bias)

                if m.weight is None: # to keep m.weight being `nn.Parameter`
                    m.weight = new_weight.contiguous()
                else:
                    m.weight.copy_(new_weight)

                if affine_params.size(1) > 2 * m.num_features:
                    affine_params = affine_params[:, 2 * m.num_features:]

    def assign_embeddings(self, data_dict):
        if self.finetuning:
            identity_embedding = self.identity_embedding.expand(len(data_dict['dec_keypoints']), -1)
        else:
            identity_embedding = data_dict['embeds']

        pose_embedding = data_dict['dec_keypoints'][:, 0] - 0.5
        joint_embedding = torch.cat((identity_embedding, pose_embedding), dim=1)

        affine_params = self.affine_params_projector(joint_embedding)
        self.assign_affine_params(affine_params)

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to generator architecture to allow fine-tuning.
            For `vanilla` generator, initialize AdaIN parameters from `data_dict['embeds']`
            and flag them as trainable parameters.
            Will require re-initializing optimizer, but only after the first call.

            data_dict:
                dict
                Required contents depend on the specific generator. For `vanilla` generator,
                it is `'embeds'` (1 x `args.embed_channels`).
                If `None`, the module's new parameters will be initialized randomly.
        """
        if data_dict is None:
            some_parameter = next(iter(self.parameters())) # to know target device and dtype
            identity_embedding = torch.rand(1, self.identity_embedding_size).to(some_parameter)
        else:
            identity_embedding = data_dict['embeds']

        if self.finetuning:
            with torch.no_grad():
                self.identity_embedding.copy_(identity_embedding)
        else:
            self.identity_embedding = nn.Parameter(identity_embedding)
            self.finetuning = True

    def forward(self, data_dict):
        self.assign_embeddings(data_dict)

        batch_size = len(data_dict['dec_keypoints'])
        outputs = self.decoder_blocks(self.constant(batch_size))
        rgb, segmentation = outputs[:, :-1], outputs[:, -1:]

        # Move tanh's output from (-1; 1) to (-0.25; 1.25)
        rgb = rgb * 0.75
        rgb += 0.5

        # Same, but to (0; 1)
        segmentation = segmentation * 0.5
        segmentation += 0.5

        data_dict['fake_rgbs'] = rgb * segmentation
        data_dict['fake_segm'] = segmentation

import os, sys
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from torch.optim import Adam
from torch import nn
from pathlib import Path

from utils import utils
from utils.argparse_utils import MyArgumentParser
from utils.utils import setup, get_args_and_modules, save_model, load_model_from_checkpoint
from utils.tensorboard_logging import setup_logging
from utils.visualize import Saver

import logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="PID %(process)d - %(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger('train.py')

parser = MyArgumentParser(conflict_handler='resolve') # TODO: allow_abbrev=False
parser.add = parser.add_argument

parser.add('--config_name', type=str, default="")

parser.add('--generator', type=str, default="", help='')
parser.add('--embedder', type=str, default="", help='')
parser.add('--discriminator', type=str, default="", help='')
parser.add('--criterions', type=str, default="", help='')
parser.add('--metrics', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')
parser.add('--runner', type=str, default="", help='')

parser.add('--args-to-ignore', type=str,
           default="checkpoint,splits_dir,experiments_dir,extension,"
                   "experiment_name,rank,local_rank,world_size")
parser.add('--experiments_dir', type=Path, default="data/experiments", help='')
parser.add('--experiment_name', type=str, default="", help='')
parser.add('--train_split_path', default="data/splits/train.csv", type=Path,
    help="Enumerates identities from the dataset to be used in training. Resolution order: " \
         "if '`--data_root`/`--img_dir`/`--train_split_path`' is a valid directory, use that; " \
         "if '`--train_split_path`' points to an existing file, use it as a CSV identity list; " \
         "else, use sorted list of directories '`--data_root`/`--img_dir`/*'.")
parser.add('--val_split_path', default="data/splits/val.csv", type=Path,
    help="See `--train_split_path`.")

# directory with vgg weights for perceptual losses
parser.add('--vgg_weights_dir', default="criterions/common/", type=str)

# Training process
parser.add('--num_epochs', type=int, default=10**9)
parser.add('--set_eval_mode_in_train', action='store_bool', default=False)
parser.add('--set_eval_mode_in_test', action='store_bool', default=True)
parser.add('--save_frequency', type=int, default=1,
    help="Save checkpoint every X epochs. If 0, save only at the end of training")
parser.add('--logging', action='store_bool', default=True)
parser.add('--skip_eval', action='store_bool', default=True)
parser.add('--profile_flops', action='store_bool', default=False)
parser.add('--weights_running_average', action='store_bool', default=True)
parser.add('--finetune', action='store_bool', default=False)
parser.add('--inference', action='store_bool', default=False)

# Model
parser.add('--in_channels', type=int, default=3)
parser.add('--out_channels', type=int, default=3)
parser.add('--num_channels', type=int, default=64)
parser.add('--max_num_channels', type=int, default=512)
parser.add('--embed_channels', type=int, default=512)
parser.add('--pose_embedding_size', type=int, default=136)
parser.add('--image_size', type=int, default=256)

# Optimizer
parser.add('--optimizer', default='Adam', type=str, choices=['Adam', 'RAdam'])
parser.add('--lr_gen', default=5e-5, type=float)
parser.add('--beta1', default=0.0, type=float, help='beta1 for Adam')

# Hardware
parser.add('--device', type=str, default='cuda')
parser.add('--num_gpus', type=int, default=1, help='requires apex if > 1, requires horovod if > 8')
parser.add('--hvd_fp16_allreduce', action='store_true')
parser.add('--hvd_batches_per_allreduce', default=1, help='number of batches processed locally before allreduce')
parser.add('--rank', type=int, default=0, help='global rank, DO NOT SET')
parser.add('--local_rank', type=int, default=0, help='"rank" within a machine, DO NOT SET')
parser.add('--world_size', type=int, default=1, help='number of devices, DO NOT SET')

# Misc
parser.add('--random_seed', type=int, default=123, help='')
parser.add('--checkpoint_path', type=str, default='')
parser.add('--saver', type=str, default='')

args, default_args, m, checkpoint_object = get_args_and_modules(parser, use_checkpoint_args=True)

# Set random seed, number of threads etc.
setup(args)

# In case of distributed training we first initialize rank and world_size
if args.num_gpus == 1:
    args.rank = args.local_rank = 0
    args.world_size = 1
elif args.num_gpus > 1 and args.num_gpus <= 8:
    # use distributed data parallel
    # `args.local_rank` is the automatic command line argument, input by `python -m torch.distributed.launch ...`;
    # `args.rank` is the actual rank value we rely on
    args.rank = args.local_rank
    args.world_size = args.num_gpus
    torch.cuda.set_device(args.local_rank)
    args.device = f'cuda:{args.local_rank}'
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
elif args.num_gpus > 8:
    # use horovod
    import horovod.torch as hvd
    hvd.init()
    # set options
    args.rank = hvd.rank()
    args.world_size = hvd.size()

logger.info(f"Initialized the process group, my rank is {args.rank}")

if args.finetune and args.num_gpus > 1:
    if args.local_rank == 0:
        logger.warning("Sorry, multi-GPU fine-tuning is NYI, setting `--num_gpus=1`")
        args.num_gpus = 1
    else:
        logger.warning("Sorry, multi-GPU fine-tuning is NYI, shutting down all processes but one")
        exit()

logger.info(f"Loading dataloader '{args.dataloader}'")
dataloader_train = m['dataloader'].get_dataloader(args, part='train', phase='train')
if not args.skip_eval:
    if args.num_gpus > 1:
        raise NotImplementedError("Multi-GPU validation not implemented")
    dataloader_val = m['dataloader'].get_dataloader(args, part='val', phase='val')

runner = m['runner']

if args.checkpoint_path != "":
    if checkpoint_object is not None:
        logger.info(f"Starting from checkpoint {args.checkpoint_path}")
        embedder, generator, discriminator, \
        running_averages, saved_args, optimizer_G, optimizer_D = \
            load_model_from_checkpoint(checkpoint_object, args)

        logger.info(f"Starting from iteration #{args.iteration}")
    else:
        raise FileNotFoundError(f"Checkpoint `{args.checkpoint_path}` not found")
else:
    if args.finetune:
        logger.error("`--finetune` is set, but `--checkpoint_path` isn't. This has to be a mistake.")

    discriminator = m['discriminator'].get_net(args)
    generator = m['generator'].get_net(args)
    embedder = m['embedder'].get_net(args)
    running_averages = {}

    optimizer_G = runner.get_optimizer(embedder, generator, args)
    optimizer_D = m['discriminator'].get_optimizer(discriminator, args)

criterion_list = [crit.get_net(args) for crit in m['criterion_list']]

if not args.weights_running_average:
    running_averages = None

writer = None
if args.logging and args.rank == 0:
    args.experiment_dir, writer = setup_logging(
        args, default_args, args.args_to_ignore.split(','))
    args.experiment_dir = Path(args.experiment_dir)
    metric_list = [metric.get_net(args) for metric in m['metric_list']]
else:
    metric_list = []

training_module = runner.TrainingModule(embedder, generator, discriminator, criterion_list, metric_list, running_averages)

# If someone tries to terminate the program, let us save the weights first
model_already_saved = False
if args.rank == 0:
    import signal, sys, os
    parent_pid = os.getpid()
    def save_last_model_and_exit(_1, _2):
        global model_already_saved
        if model_already_saved:
            return
        model_already_saved = True
        if os.getpid() == parent_pid: # otherwise, dataloader workers will try to save the model too!
            logger.info("Interrupted, saving the current model")
            save_model(training_module, optimizer_G, optimizer_D, args)
            # protect from Tensorboard's "Unable to get first event timestamp
            # for run `...`: No event timestamp could be found"
            if writer is not None:
                writer.close()
            sys.exit()
    signal.signal(signal.SIGINT , save_last_model_and_exit)
    signal.signal(signal.SIGTERM, save_last_model_and_exit)

if args.device.startswith('cuda') and torch.cuda.device_count() > 1:
    if args.num_gpus > 1 and args.num_gpus <= 8:
        from apex.parallel import Reducer
        training_module.reducer = Reducer(training_module)
        training_module.__dict__['module'] = training_module # do not register self as a nested module
    elif args.num_gpus > 8:
        optimizer_G = hvd.DistributedOptimizer(optimizer_G,
                                               named_parameters=runner.get_named_parameters(embedder, generator, args),
                                               compression=hvd.Compression.fp16 if args.hvd_fp16_allreduce else hvd.Compression.none,
                                               backward_passes_per_step=args.hvd_batches_per_allreduce)
        optimizer_D = hvd.DistributedOptimizer(optimizer_G,
                                               named_parameters=m['discriminator'].get_named_parameters(discriminator, args),
                                               compression=Compression.fp16 if args.hvd_fp16_allreduce else hvd.Compression.none,
                                               backward_passes_per_step=args.hvd_batches_per_allreduce)
        hvd.broadcast_optimizer_state(optimizer_G, root_rank=0)
        hvd.broadcast_optimizer_state(optimizer_D, root_rank=0)

# Optional results saver
saver = None
if args.saver and args.rank == 0:
    saver = Saver(save_dir=f'{args.experiment_dir}/validation_results/', save_fn=args.saver)

if args.finetune:
    # A dirty hack (sorry) for reproducing X2Face within our pipeline
    if args.generator == 'X2Face':
        MAX_IDENTITY_IMAGES = 8
        identity_images = []
        for data_dict, _ in dataloader_train:
            identity_images.append(data_dict['pose_input_rgbs'][:, 0]) # B x C x H x W
            total_identity_images = sum(map(len, identity_images))
            if total_identity_images >= MAX_IDENTITY_IMAGES:
                break

        total_identity_images = min(MAX_IDENTITY_IMAGES, total_identity_images)

        logger.info(f"Saving X2Face model with {total_identity_images} identity images")
        args.X2Face_num_identity_images = total_identity_images
        data_dict = {'enc_rgbs': torch.cat(identity_images)[:total_identity_images][None]}
        training_module.generator.enable_finetuning(data_dict)

        print(training_module.generator.identity_images.shape)
        save_model(training_module, optimizer_G, optimizer_D, args)
        exit()

    logger.info(f"For fine-tuning, computing an averaged identity embedding from {len(dataloader_train.dataset)} frames")

    training_module.eval()
    identity_embeddings = []

    with torch.no_grad():
        # Precompute identity embedding $\hat{e}_{NEW}$
        for data_dict, _ in dataloader_train:
            try:
                embedder = training_module.running_averages['embedder']
            except:
                logger.warning(f"Couldn't get embedder's running average, computing the embedding with the original embedder")
                embedder = training_module.embedder

            utils.dict_to_device(data_dict, args.device)
            embedder.get_identity_embedding(data_dict)
            identity_embeddings.append(data_dict['embeds_elemwise'].view(-1, args.embed_channels))

        identity_embedding = torch.cat(identity_embeddings).mean(0)
        del identity_embeddings

    # Initialize person-specific generator parameters $\psi'$ and flag them as trainable
    data_dict = {'embeds': identity_embedding[None]}
    training_module.generator.enable_finetuning(data_dict)
    # Put the embedding $\hat{e}_{NEW}$ into discriminator's matrix W
    training_module.discriminator.enable_finetuning(data_dict)

    if args.weights_running_average:
        # Do the same for running averages
        if 'generator' in training_module.running_averages:
            training_module.running_averages['generator'].enable_finetuning(data_dict)
        if 'discriminator' in training_module.running_averages:
            training_module.running_averages['discriminator'].enable_finetuning(data_dict)
    else:
        # Remove running averages
        training_module.initialize_running_averages(None)

    # Re-initialize optimizers
    optimizer_G = runner.get_optimizer(training_module.embedder, training_module.generator, args)
    optimizer_D = m['discriminator'].get_optimizer(discriminator, args)

logger.info(f"Entering training loop")

# Main loop
for epoch in range(0, args.num_epochs):
    # ===================
    #       Train
    # ===================
    training_module.train(not args.set_eval_mode_in_train)
    torch.set_grad_enabled(True)
    runner.run_epoch(dataloader_train, training_module, optimizer_G, optimizer_D,
                     epoch, args, phase='train', writer=writer, saver=saver)

    if not args.skip_eval:
        raise NotImplementedError("NYI: validation")
        # ===================
        #       Validate
        # ===================
        training_module.train(not args.set_eval_mode_in_test)
        torch.set_grad_enabled(False)
        with training_module.set_use_running_averages(), training_module.set_compute_losses(False):
            runner.run_epoch(dataloader_val, training_module, None, None,
                             epoch, args, phase='val', writer=writer, saver=saver)

    if args.rank == 0:
        will_save_checkpoint = epoch == args.num_epochs-1
        if args.save_frequency != 0:
            will_save_checkpoint |= epoch % args.save_frequency == 0

        if will_save_checkpoint:
            save_model(training_module, optimizer_G, optimizer_D, args)

import torch
import torch.utils.data
import numpy as np
import cv2

import math
import os
from abc import ABC, abstractmethod

try:
    import face_alignment
    from face_alignment.detection.sfd import FaceDetector
except ImportError:
    raise ImportError(
        "Please install face alignment package from "
        "https://github.com/1adrianb/face-alignment")

def load_landmark_detector():
    return face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

def load_face_detector():
    return FaceDetector(device='cuda')

class FaceCropper(ABC):
    @abstractmethod
    def __init__(self, output_size=(256, 256)):
        """
            output_size
                tuple, (width x height)
        """
        pass

    @abstractmethod
    def crop_image(self, image, bbox=None, compute_landmarks=True):
        """
            image
                numpy.ndarray, np.uint8, H x W x 3, RGB
            bbox
                list, int, len == 4 or 5, LTRB
                If provided, don't run face detector.
            compute_landmarks:
                bool

            return:
                numpy.ndarray, np.uint8, `self.output_size` x 3 (e.g. 256 x 256 x 3), RGB
                    cropped image
                object
                    any crop auxiliary information (for now, facial landmarks)
        """
        pass

class FFHQFaceCropper(FaceCropper):
    """
        Yields "FFHQ-style" crops. Based on landmarks, which are detected
        using https://github.com/1adrianb/face-alignment.
    """
    def __init__(self, output_size=(256, 256)):
        self.landmark_detector = load_landmark_detector()
        self.output_size = output_size

    def crop_image(self, image, bbox=None, compute_landmarks=True):
        """
            image
                numpy.ndarray, np.uint8, H x W x 3, RGB
            bbox
                None
                No effect.
            compute_landmarks
                bool

            return:
                numpy.ndarray, np.uint8, `self.output_size` x 3 (e.g. 256 x 256 x 3), RGB
                    cropped image
                numpy.ndarray, np.float32, 68 x 3
                    cropped 3D landmarks
        """
        assert bbox is None, "NYI: custom bbox for FFHQFaceCropper"

        landmarks = self.landmark_detector.get_landmarks(image)
        try:
            landmarks = landmarks[0]
        except TypeError: # zero faces detected
            landmarks = np.random.rand(68, 3).astype(np.float32)

        image, landmarks = self.crop_from_landmarks(image, landmarks)
        
        h_resize_ratio = self.output_size[1] / image.shape[0]
        w_resize_ratio = self.output_size[0] / image.shape[1]
        landmarks[:, 0 ] *= h_resize_ratio
        landmarks[:, 1:] *= w_resize_ratio # scale Z too
        image = cv2.resize(image, self.output_size,
            interpolation=cv2.INTER_CUBIC if h_resize_ratio > 1.0 else cv2.INTER_AREA)

        return image, landmarks if compute_landmarks else None

    @staticmethod
    def crop_from_landmarks(image, landmarks, only_landmarks=False):
        """
            Crops an image as in VoxCeleb2 dataset, with blurred reflection
            padding, given pixel coordinates of 68 facial landmarks.

            image
                numpy.ndarray, np.uint8, H x W x 3
            landmarks
                numpy.ndarray, np.float32, 68 x {2|3}, float
            only_landmarks
                bool
                if True, image will not be cropped and returned

            return:
                numpy.ndarray, np.uint8, h x w x 3 (optional)
                    cropped image
                numpy.ndarray, np.float32, 68 x {2|3}
                    cropped landmarks
        """
        lm_chin          = landmarks[0  : 17, :2]  # left-right
        lm_eyebrow_left  = landmarks[17 : 22, :2]  # left-right
        lm_eyebrow_right = landmarks[22 : 27, :2]  # left-right
        lm_nose          = landmarks[27 : 31, :2]  # top-down
        lm_nostrils      = landmarks[31 : 36, :2]  # top-down
        lm_eye_left      = landmarks[36 : 42, :2]  # left-clockwise
        lm_eye_right     = landmarks[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = landmarks[48 : 60, :2]  # left-clockwise
        lm_mouth_inner   = landmarks[60 : 68, :2]  # left-clockwise

        lm_cropped = landmarks.copy()

        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x).item() * 2

        # Crop.
        border = max(round(qsize * 0.1), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (crop[0] - border, crop[1] - border, crop[2] + border, crop[3] + border)

        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.shape[1] + border, 0), max(pad[3] - image.shape[0] + border, 0))

        lm_cropped[:, 0] -= crop[0]
        lm_cropped[:, 1] -= crop[1]

        if not only_landmarks:
            def crop_from_bbox(img, bbox):
                """
                    bbox: tuple, (x1, y1, x2, y2)
                        x: horizontal, y: vertical, exclusive
                """
                x1, y1, x2, y2 = bbox
                if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                    img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
                return img[y1:y2, x1:x2]

            def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
                img = cv2.copyMakeBorder(img,
                    -min(0, y1), max(y2 - img.shape[0], 0),
                    -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REFLECT)
                y2 += -min(0, y1)
                y1 += -min(0, y1)
                x2 += -min(0, x1)
                x1 += -min(0, x1)
                return img, x1, x2, y1, y2

            image = crop_from_bbox(image, crop).astype(np.float32)

            h, w, _ = image.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            y, x = y.astype(np.float32), x.astype(np.float32)
            pad = np.array(pad, dtype=np.float32)
            pad[pad == 0] = 1e-10
            mask = np.maximum(1.0 - np.minimum(x / pad[0], (w-1-x) / pad[2]), 1.0 - np.minimum(y / pad[1], (h-1-y) / pad[3]))
            
            sigma = qsize * 0.02
            kernel_size = 0 #round(sigma * 4)
            image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_REFLECT)
            image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            image += (np.median(image, axis=(0,1)) - image) * np.clip(mask, 0.0, 1.0)
            image.round(out=image)
            image.clip(0, 255, out=image)
            image = image.astype(np.uint8)
            
        if only_landmarks:
            return lm_cropped
        else:
            return np.array(image), lm_cropped


class LatentPoseFaceCropper(FaceCropper):
    """
        Yields "latent pose style" crops. Based on face detections from the S^3FD detector.
    """
    def __init__(self, output_size=(256, 256)):
        """
            output_size
                tuple, (width x height)
        """
        self.face_detector = load_face_detector()
        self.landmark_detector = None # only loaded when `compute_landmarks=True` is requested
        self.output_size = output_size

    def crop_image(self, image, bbox=None, compute_landmarks=True):
        """
            image
                numpy.ndarray, np.uint8, H x W x 3, RGB
            bbox
                list, int, len == 4 or 5, LTRB
                If provided, don't run face detector.
            compute_landmarks
                bool

            return:
                numpy.ndarray, np.uint8, `self.output_size` x 3 (e.g. 256 x 256 x 3), RGB
                    cropped image
                numpy.ndarray, (np.float32, 68 x 3) or None
                    if `compute_landmarks` was `True`: cropped 3D facial landmark coordinates (x, y, z)
        """
        if bbox is None:
            bboxes = self.detect_faces([image])[0]
            bbox = self.choose_one_detection(bboxes)[:4]

        if compute_landmarks:
            if self.landmark_detector is None:
                self.landmark_detector = load_landmark_detector()
            landmarks = self.landmark_detector.get_landmarks_from_image(image, [bbox])[0]

        # Make bbox square and scale it
        l, t, r, b = bbox
        SCALE = 1.8

        center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
        height, width = b - t, r - l
        new_box_size = max(height, width)
        l = center_x - new_box_size / 2 * SCALE
        r = center_x + new_box_size / 2 * SCALE
        t = center_y - new_box_size / 2 * SCALE
        b = center_y + new_box_size / 2 * SCALE

        # Make floats integers
        l, t = map(math.floor, (l, t))
        r, b = map(math.ceil, (r, b))

        # After rounding, make *exactly* square again
        b += (r - l) - (b - t)
        assert b - t == r - l

        # Make `r` and `b` C-style (=exclusive) indices
        r += 1
        b += 1

        # Crop
        image_cropped = self.crop_with_padding(image, t, l, b, r)

        # "Crop" landmarks
        if compute_landmarks:
            landmarks[:, 0] -= l
            landmarks[:, 1] -= t

            h_resize_ratio = self.output_size[1] / image_cropped.shape[0]
            w_resize_ratio = self.output_size[0] / image_cropped.shape[1]
            landmarks[:, 0 ] *= h_resize_ratio
            landmarks[:, 1:] *= w_resize_ratio # scale Z too

        # Resize to the target resolution
        image_cropped = cv2.resize(image_cropped, self.output_size,
            interpolation=cv2.INTER_CUBIC if self.output_size[1] > bbox[3] - bbox[1] else cv2.INTER_AREA)

        return image_cropped, landmarks if compute_landmarks else None

    def detect_faces(self, images):
        """
            images
                list of numpy.ndarray, any dtype 0-255, H x W x 3, RGB
                OR
                torch.tensor, any dtype 0-255, B x 3 x H x W, RGB

                A batch of images.

            return:
                list of lists of lists of length 5
                Bounding boxes for each image in batch.
        """
        if type(images) is list:
            images = np.stack(images).transpose(0,3,1,2).astype(np.float32)
            images_torch = torch.tensor(images)
        else:
            assert torch.is_tensor(images)
            images_torch = images.to(torch.float32)

        return self.face_detector.detect_from_batch(images_torch.cuda())

    @staticmethod
    def choose_one_detection(frame_faces):
        """
            frame_faces
                list of lists of length 5
                several face detections from one image

            return:
                list of 5 floats
                one of the input detections: `(l, t, r, b, confidence)`
        """
        if len(frame_faces) == 0:
            retval = [0, 0, 200, 200, 0.0]
        elif len(frame_faces) == 1:
            retval = frame_faces[0]
        else:
            # sort by area, find the largest box
            largest_area, largest_idx = -1, -1
            for idx, face in enumerate(frame_faces):
                area = abs(face[2]-face[0]) * abs(face[1]-face[3])
                if area > largest_area:
                    largest_area = area
                    largest_idx = idx

            retval = frame_faces[largest_idx]

        return np.array(retval).tolist()

    @staticmethod
    def crop_with_padding(image, t, l, b, r, segmentation=False):
        """
            image:
                numpy, np.uint8, (H x W x 3) or (H x W)
            t, l, b, r:
                int
            segmentation:
                bool
                Affects padding.

            return:
                numpy, (b-t) x (r-l) x 3
        """
        t_clamp, b_clamp = max(0, t), min(b, image.shape[0])
        l_clamp, r_clamp = max(0, l), min(r, image.shape[1])
        image = image[t_clamp:b_clamp, l_clamp:r_clamp]

        # If the bounding box went outside of the image, restore those areas by padding
        padding = [t_clamp - t, b - b_clamp, l_clamp - l, r - r_clamp]
        if sum(padding) == 0: # = if the bbox fully fit into image
            return image

        if segmentation:
            padding_top = [(x if i == 0 else 0) for i, x in enumerate(padding)]
            padding_others = [(x if i != 0 else 0) for i, x in enumerate(padding)]
            image = cv2.copyMakeBorder(image, *padding_others, cv2.BORDER_REPLICATE)
            image = cv2.copyMakeBorder(image, *padding_top, cv2.BORDER_CONSTANT)
        else:
            image = cv2.copyMakeBorder(image, *padding, cv2.BORDER_REFLECT101)
        assert image.shape[:2] == (b - t, r - l)

        # We will blur those padded areas
        h, w = image.shape[:2]
        y, x = map(np.float32, np.ogrid[:h, :w]) # meshgrids
        
        mask_l = np.full_like(x, np.inf) if padding[2] == 0 else (x / padding[2])
        mask_t = np.full_like(y, np.inf) if padding[0] == 0 else (y / padding[0])
        mask_r = np.full_like(x, np.inf) if padding[3] == 0 else ((w-1-x) / padding[3])
        mask_b = np.full_like(y, np.inf) if padding[1] == 0 else ((h-1-y) / padding[1])

        # The farther from the original image border, the more blur will be applied
        mask = np.maximum(
            1.0 - np.minimum(mask_l, mask_r),
            1.0 - np.minimum(mask_t, mask_b))
        
        # Do blur
        sigma = h * 0.016
        kernel_size = 0
        image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # Now we'd like to do alpha blending math, so convert to float32
        def to_float32(x):
            x = x.astype(np.float32)
            x /= 255.0
            return x
        image = to_float32(image)
        image_blurred = to_float32(image_blurred)

        # Support 2-dimensional images (e.g. segmentation maps)
        if image.ndim < 3:
            image.shape += (1,)
            image_blurred.shape += (1,)
        mask.shape += (1,)

        # Replace padded areas with their blurred versions, and apply
        # some quickly fading blur to the inner part of the image
        image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)

        # Make blurred borders fade to edges
        if segmentation:
            fade_color = np.zeros_like(image)
            fade_color[:, :padding[2]] = 0.0
            fade_color[:, -padding[3]:] = 0.0
            mask = (1.0 - np.minimum(mask_l, mask_r))[:, :, None]
        else:
            fade_color = np.median(image, axis=(0,1))
        image += (fade_color - image) * np.clip(mask, 0.0, 1.0) 
        
        # Convert back to uint8 for interface consistency
        image *= 255.0
        image.round(out=image)
        image.clip(0, 255, out=image)
        image = image.astype(np.uint8)

        return image


VIDEO_EXTENSIONS = ('.avi', '.mpg', '.mov', '.mkv', '.mp4')
IMAGE_EXTENSIONS = ('.jpg', '.png')

class ImageReader(ABC):
    """
        An abstract iterator to read images from a folder, video, webcam, ...
    """
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __next__(self):
        """
            return:
                numpy, H x W x 3, uint8
        """
        pass

    def __getitem__(self, _):
        """
            A hack for using `ImageReader` in `torch.utils.data.DataLoader`.
        """
        return next(self)

    def __iter__(self):
        return self

    @staticmethod
    def get_image_reader(source):
        """
            source:
                `Path` or `str`
                See `crop_as_in_dataset.py`'s command line documentation for `SOURCE`.

            return:
                `ImageReader`
                A concrete `ImageReader` instance guessed from `source`.
        """
        source = str(source)

        if source.startswith('WEBCAM_'):
            return OpencvVideoCaptureReader(int(source[7:]))
        elif source[-4:].lower() in VIDEO_EXTENSIONS:
            return OpencvVideoCaptureReader(source)
        elif source[-4:].lower() in IMAGE_EXTENSIONS:
            return SingleImageReader(source)
        elif os.path.isdir(source):
            return FolderReader(source)
        else:
            raise ValueError(f"Invalid `source` argument: {source}")

class ImageWriter(ABC):
    """
        An abstract class to write images into (folder, video, screen, ...)
    """
    @abstractmethod
    def add(self, image, extra_data=None):
        """
            image:
                numpy.ndarray, H x W x 3, uint8
            extra_data:
                object
        """
        pass

    @staticmethod
    def get_image_writer(destination, fourcc=None, fps=None):
        """
            destination:
                `Path` or `str`
                See command line documentation for `DESTINATION`.
            fps:
                `float` (optional)

            return:
                `ImageWriter`
                An `ImageReader` instance guessed from `destination`.
        """
        destination = str(destination)

        if destination == 'SCREEN':
            return ScreenWriter()
        elif destination[-4:].lower() in IMAGE_EXTENSIONS:
            return SingleImageWriter(destination)
        elif destination[-4:].lower() in VIDEO_EXTENSIONS:
            return VideoWriter(destination, fourcc, fps)
        else:
            return FolderWriter(destination)

###### Image readers ######

class FolderReader(ImageReader):
    def __init__(self, path):
        self.path = str(path)
        self.files = sorted(os.listdir(self.path))
        self.index = 0

    def __len__(self):
        return len(self.files)

    def __next__(self):
        if self.index == len(self.files):
            raise StopIteration

        image = cv2.imread(os.path.join(self.path, self.files[self.index]))

        self.index += 1
        return image

class OpencvVideoCaptureReader(ImageReader):
    def __init__(self, source):
        self.video_capture = cv2.VideoCapture(str(source))
        assert self.video_capture.isOpened()

    def __len__(self):
        retval = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return None if retval < 0 else retval

    def __next__(self):
        success, image = self.video_capture.read()
        if not success:
            raise StopIteration
        else:
            return image

class SingleImageReader(ImageReader):
    def __init__(self, path):
        self.image = cv2.imread(str(path))

    def __len__(self):
        return 1

    def __next__(self):
        if self.image is None:
            raise StopIteration

        try:
            return self.image
        finally:
            self.image = None

###### Image writers ######

class FolderWriter(ImageWriter):
    def __init__(self, path):
        self.path = str(path)
        if os.path.exists(path):
            num_files = len(os.listdir(path))
            print(f"WARNING: {path} already exists, contains {num_files} files")

        os.makedirs(path, exist_ok=True)

        self.index = 0

    def add(self, image, extra_data=None):
        cv2.imwrite(os.path.join(self.path, '%05d.jpg' % self.index), image)

        if extra_data is not None:
            np.save(os.path.join(self.path, '%05d.npy' % self.index), extra_data)

        self.index += 1

class VideoWriter(ImageWriter):
    def __init__(self, path, fourcc=None, fps=None):
        """
            path: str
                Where to save the video. Extension matters (see below).
            fourcc: str, length 4 (optional)
                Codec to use.
                - 'MJPG' with ".avi" extension is very safe platform agnostic, works everywhere.
                - 'avc1' or 'mp4v' with ".mp4" extension, on the other hand,
                  is good for sending to Telegram. Be careful: this most likely won't work
                  with pip's `opencv-python`, but you can use ffmpeg to convert to manually,
                  e.g. `ffmpeg -i input.avi -vcodec libx264 -f mp4 output.mp4`.
            fps: float (optional)
                Output video framerate. Default: 25.
        """
        self.path = str(path)

        if fourcc is None:
            default_codecs = {
                '.avi': 'MJPG',
                '.mp4': 'avc1',
            }
            fourcc = default_codecs.get(self.path[-4:], 'XVID')
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        self.fps = 25.0 if fps is None else fps
        self.video_writer = None

    def add(self, image, extra_data=None):
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(
                self.path, self.fourcc, self.fps, image.shape[1::-1])
            assert self.video_writer.isOpened(), "Couldn't initialize video writer"

        self.video_writer.write(image)

class SingleImageWriter(ImageWriter):
    def __init__(self, path):
        self.path = str(path)

    def add(self, image, extra_data=None):
        cv2.imwrite(self.path, image)

        if extra_data:
            np.save(os.path.splitext(self.path)[0] + '.npy', extra_data)

class ScreenWriter(ImageWriter):
    def add(self, image):
        cv2.imshow('Cropped image', image)
        cv2.waitKey(1)


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Crop images as in training dataset",
        formatter_class=argparse.RawTextHelpFormatter)
    
    arg_parser.add_argument('source', metavar='SOURCE', type=str,
        help="Where to take images from: can be\n"
             "- path to a folder with images,\n"
             "- path to a single image,\n"
             "- path to a video file,\n"
             "- 'WEBCAM_`N`', N=0,1,2... .")
    arg_parser.add_argument('destination', metavar='DESTINATION', type=str,
        help="Where to put cropped images: can be\n"
             "- path to a non-existent folder (will be created and filled with images),\n"
             "- path to a maybe existing image (guessed by extension),\n"
             "- path to a maybe existing video file (guessed by extension),\n"
             "- 'SCREEN'.")
    arg_parser.add_argument('--crop-style', type=str, default='latentpose', choices=['ffhq', 'latentpose'],
        help="Which crop style to use.")
    arg_parser.add_argument('--image-size', type=int, default=256,
        help="Size of square output images.")
    arg_parser.add_argument('--save-extra-data', action='store_true',
        help="If set, will save '.npy' files with keypoints alongside.")
    args = arg_parser.parse_args()

    # Initialize image reader
    image_reader = ImageReader.get_image_reader(args.source)
    image_loader = torch.utils.data.DataLoader(image_reader,
        num_workers=1 if isinstance(image_reader, FolderReader) else 0)

    # Initialize image writer
    fps = None
    if isinstance(image_reader, OpencvVideoCaptureReader):
        fps = image_reader.video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = None
    image_writer = ImageWriter.get_image_writer(args.destination, fps=fps)

    # Initialize cropper
    ChosenFaceCropper = {
        'ffhq':       FFHQFaceCropper,
        'latentpose': LatentPoseFaceCropper,
    }[args.crop_style]
    cropper = ChosenFaceCropper((args.image_size, args.image_size))

    # Main loop
    from tqdm import tqdm
    for input_image in tqdm(image_loader):
        input_image = input_image[0].numpy()

        if max(input_image.shape) > 1152:
            resize_ratio = 1152 / max(input_image.shape)
            input_image = cv2.resize(input_image, dsize=None, fx=resize_ratio, fy=resize_ratio)

        image_cropped, extra_data = cropper.crop_image(input_image)
        if not args.save_extra_data:
            extra_data = None
        image_writer.add(image_cropped, extra_data)

import importlib
import logging
import os
import random
import time
from argparse import Namespace
from collections import defaultdict
from typing import List

import cv2
import numpy as np
import torch
import yamlenv


def setup(args):
    logger = logging.getLogger('utils.setup')

    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.random_seed is None:
        args.random_seed = int(time.time() * 2)

    logger.info(f"Random Seed: {args.random_seed}")
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.random_seed)


def dict_to_device(d, device):
    for key in d:
        if torch.is_tensor(d[key]):
            d[key] = d[key].to(device)


def get_args_and_modules(parser, use_checkpoint_args=True, custom_args={}):
    """
        Load modules ("embedder", "generator", "discriminator", "runner", "dataloader", criteria)
        and initialize program arguments based on various sources of those arguments. Optionally,
        also load a checkpoint file.

        The preference (i.e. the inverse resolution order) is:
            1. command line
            2. `custom_args`
            3. .yaml file
            4. `args` from a checkpoint file
            5. argparse defaults

        parser:
            `argparse.ArgumentParser`
        use_checkpoint_args:
            `bool`
            If `True`: check if `--checkpoint_path` is defined either
            in .yaml file or on the command line; if yes, use
            `saved_args` from that checkpoint.
        custom_args:
            `dict`
            Defines any custom default values for `parser`'s parameters.

        return:
            args:
                namespace
            <deprecated>:
                namespace
            m:
                `dict`
                A mapping from module names (e.g. 'generator') to actual loaded modules
            checkpoint_object:
                `dict` or `None`
                If `use_checkpoint_args` was `True`, the return value of
                `torch.load(args.checkpoint_path, map_location='cpu')`.
                Otherwise, `None`.
    """
    logger = logging.getLogger('utils.get_args_and_modules')

    # Parse arguments supplied on the command line
    # (and in `custom_args`) to obtain "--config_name" value
    parser.set_defaults(**custom_args)
    args, _ = parser.parse_known_args()

    # Read the .yaml config
    try:
        if args.config_name == '':
            logger.warning(f"Not using any .yaml config file")
            config_args = {}
        else:
            config_args = load_config_file(args.config_name)
    except FileNotFoundError:
        logger.warning(f"Could not load config {args.config_name}")
        config_args = {}

    # Let the parser update `args` with values from there.
    # We do this to obtain the "--checkpoint_path" value
    parser.set_defaults(**config_args)
    parser.set_defaults(**custom_args)
    args, _ = parser.parse_known_args()

    # If "--checkpoint_path" is defined, load the checkpoint to merge its args
    if use_checkpoint_args:
        if args.checkpoint_path:
            logger.info(f"Loading checkpoint file {args.checkpoint_path}")
            checkpoint_object = torch.load(args.checkpoint_path, map_location='cpu')
            checkpoint_args = vars(checkpoint_object['args'])
        else:
            logger.info(f"`args.checkpoint_path` isn't defined, so not using args from a checkpoint")
            checkpoint_object, checkpoint_args = None, {}
    else:
        checkpoint_object, checkpoint_args = None, {}

    # Go through the config resolution order (see docstring), but just to determine
    # module names, i.e. the following config arguments:
    # "embedder", "generator", "discriminator", "runner", "dataloader", "criterions"
    parser.set_defaults(**checkpoint_args)
    parser.set_defaults(**config_args)
    parser.set_defaults(**custom_args)
    args, _ = parser.parse_known_args()

    # Now when the module names are known, load them and update the argument parser accordingly,
    # since those modules may define their own variables with their own argument parsers:
    m = {}

    m['generator'] = load_module('generators', args.generator).Wrapper
    m['generator'].get_args(parser)

    m['embedder'] = load_module('embedders', args.embedder).Wrapper
    m['embedder'].get_args(parser)

    m['runner'] = load_module('runners', args.runner)
    m['runner'].get_args(parser)

    m['discriminator'] = load_module('discriminators', args.discriminator).Wrapper
    m['discriminator'].get_args(parser)

    m['criterion_list'] = load_wrappers_for_module_list(args.criterions, 'criterions')
    for crit in m['criterion_list']:
        crit.get_args(parser)

    m['metric_list']= load_wrappers_for_module_list(args.metrics, 'metrics')
    for metric in m['metric_list']:
        metric.get_args(parser)

    m['dataloader'] = load_module('dataloaders', 'dataloader').Dataloader(args.dataloader)
    m['dataloader'].get_args(parser)

    # Finally, `parser` is aware of all the possible parameters.
    # Go through the resolution order again to establish the values for all of them:
    # TODO make overriding verbose
    parser.set_defaults(**checkpoint_args)
    parser.set_defaults(**config_args)
    parser.set_defaults(**custom_args)
    args, default_args = parser.parse_args(), parser.parse_args([])

    # Dynamic defaults
    if not args.experiment_name:
        logger.info(f"`args.experiment_name` is missing, so setting it to `args.config_name` (\"{args.config_name}\")")
        args.experiment_name = args.config_name

    return args, default_args, m, checkpoint_object


def load_config_file(config_name):
    logger = logging.getLogger('utils.load_config_file')

    config_path = f'configs/{config_name}.yaml'

    logger.info(f"Using config {config_path}")
    with open(config_path, 'r') as stream:
        return yamlenv.load(stream)


def load_module(module_type, module_name):
    return importlib.import_module(f'{module_type}.{module_name}')


def load_wrappers_for_module_list(module_name_list: str, parent_module: str):
    """Import a comma-separated list of Python module names (e.g. "perceptual, adversarial,dice").
    For each of those modules, create the respective 'Wrapper' class and return those classes in
    a list."""
    module_names = module_name_list.split(',')
    module_names = [c.strip() for c in module_names if c.strip()]

    wrappers = []
    for module_name in module_names:
        module = importlib.import_module(f'{parent_module}.{module_name}')
        wrappers.append(module.Wrapper)

    return wrappers


class Meter:
    """
    Tracks average and last values of several metrics.
    """
    def __init__(self):
        super().__init__()
        self.sum = defaultdict(float)
        self.num_measurements = defaultdict(int)
        self.last_value = {}

    def add(self, name, value, num_measurements=1):
        """
        Add `num_measurements` measurements for metric `name`, given their average (`value`).
        To add just one measurement, call with `num_measurements = 1` (default).

        name:
            `str`
        value:
            convertible to `float`
        num_measurements:
            `int`
        """
        assert num_measurements >= 0
        if num_measurements == 0:
            return

        value = float(value)
        if value != value: # i.e. if value is NaN
            # add 0 in case dict keys don't exist yet
            self.sum[name] += 0
            self.num_measurements[name] += 0
        else:
            self.sum[name] += value * num_measurements
            self.num_measurements[name] += num_measurements
        self.last_value[name] = value

    def keys(self):
        return self.sum.keys()

    def get_average(self, name):
        return self.sum[name] / max(1, self.num_measurements[name])

    def get_last(self, name):
        return self.last_value[name]

    def get_num_measurements(self, name):
        return self.num_measurements[name]

    def __iadd__(self, other_meter):
        for name in other_meter.sum:
            self.add(name, other_meter.get_average(name), other_meter.get_num_measurements(name))
            self.last_value[name] = other_meter.last_value[name]
        return self


def save_model(training_module, optimizer_G, optimizer_D, args):
    logger = logging.getLogger('utils.save_model')

    if args.rank != 0:
        return

    if args.num_gpus > 1:
        training_module = training_module.module

    save_dict = {}
    if training_module.embedder is not None:
        save_dict['embedder'] = training_module.embedder.state_dict()
    if training_module.generator is not None:
        save_dict['generator'] = training_module.generator.state_dict()
    if training_module.discriminator is not None:
        save_dict['discriminator'] = training_module.discriminator.state_dict()
    if optimizer_G is not None:
        save_dict['optimizer_G'] = optimizer_G.state_dict()
    if optimizer_D is not None:
        save_dict['optimizer_D'] = optimizer_D.state_dict()
    if training_module.running_averages is not None:
        save_dict['running_averages'] = \
            {name: module.state_dict() for name, module in training_module.running_averages.items()}
    if args is not None:
        save_dict['args'] = args

    epoch_string = f'{args.iteration:08}'
    save_path = f'{args.experiment_dir}/checkpoints/model_{epoch_string}.pth'
    logger.info(f"Trying to save checkpoint at {save_path}")

    while os.path.exists(save_path): # ugly temporary solution, sorry
        epoch_string += '_0'
        save_path = f'{args.experiment_dir}/checkpoints/model_{epoch_string}.pth'
        logger.info(f"That path already exists, so augmenting it with '_0': {save_path}")

    try:
        logger.info(f"Finally saving checkpoint at {save_path}")
        torch.save(save_dict, save_path, pickle_protocol=-1)
        logger.info(f"Done saving checkpoint")
    except RuntimeError as err: # disk full?
        logger.error(f"Could not write to {save_path}: {err}; removing that file")
        try:
            os.remove(save_path)
        except:
            pass


def load_model_from_checkpoint(checkpoint_object, args=Namespace()):
    logger = logging.getLogger('utils.load_model_from_checkpoint')

    saved_args = checkpoint_object['args']
    saved_args_device_backup, saved_args.device = saved_args.device, 'cpu'

    # Determine
    # (1) if we will set the model to "fine-tuning mode" and
    # (2) if we have loaded a fine-tuned model
    finetune = 'finetune' in args and args.finetune
    already_finetuned = 'finetune' in saved_args and saved_args.finetune
    assert not (already_finetuned and 'finetune' in args and not finetune), \
         "NYI: using fine-tuned checkpoint for meta-learning"

    # TODO: move the below to `get_args_and_modules()`
    """
    # Load the command line arguments supplied when that model was trained
    # and combine them with the current arguments (if there are any)

    # Always prioritize values in checkpoint for these arguments
    ARGS_TO_IGNORE = \
        'iteration', 'num_labels'
    # Do not log the replacement of values for these arguments
    SILENT_ARGS = \
        ('device', 'fixed_val_ids', 'experiment_dir', 'local_rank', \
        'rank', 'world_size', 'num_workers', '__module__', '__dict__', '__weakref__', \
        'log_frequency_images', 'log_frequency_fixed_images', 'checkpoint_path', \
        'save_frequency', 'experiment_name', 'config_name')

    differing_args = []
    if args is not None:
        for arg_name, arg_value in vars(args).items():
            arg_value_saved = saved_args.__dict__.get(arg_name)
            if arg_value != arg_value_saved and arg_name not in ARGS_TO_IGNORE:
                if arg_name not in SILENT_ARGS:
                    logger.info(
                        f"Values for the config argument `{arg_name}` differ! Checkpoint has " \
                        f"`{arg_value_saved}`, replacing it with `{arg_value}`")

                differing_args.append(arg_name)
                saved_args.__dict__[arg_name] = args.__dict__[arg_name]
    """
    differing_args = []
    for arg_name, arg_value in vars(args).items():
        if arg_name in saved_args and arg_value != saved_args.__dict__.get(arg_name):
            differing_args.append(arg_name)

    # Load weights' running averages
    running_averages = checkpoint_object.get('running_averages', {})

    # Load embedder, generator, discriminator
    modules = {}
    for module_name in 'embedder', 'generator', 'discriminator':
        module_kind = getattr(args, module_name)
        logger.info(f"Loading {module_name} '{module_kind}'")

        module = load_module(f'{module_name}s', module_kind).Wrapper.get_net(args)

        module_old = load_module(f'{module_name}s', module_kind).Wrapper.get_net(saved_args)
        if already_finetuned:
            module_old.enable_finetuning()
        module_old.load_state_dict(checkpoint_object[module_name])

        # Change module structure to match the structure of the checkpointed module
        if finetune:
            module.enable_finetuning()
            if not already_finetuned:
                module_old.enable_finetuning()

        if module_name in differing_args:
            logger.warning(f"{module_name} has changed in config, so not loading weights")
        else:
            module.load_state_dict(module_old.state_dict())

        modules[module_name] = module

    # Load optimizer states, runner
    if 'inference' in args and args.inference:
        optimizer_G = optimizer_D = None
    else:
        optimizer_D = \
            load_module('discriminators', args.discriminator).Wrapper \
            .get_optimizer(modules['discriminator'], args)
        if 'discriminator' in differing_args or optimizer_D is None or finetune and not already_finetuned:
            logger.warning(f"Discriminator has changed in config (maybe due to finetuning), so not loading `optimizer_D`")
        else:
            optimizer_D.load_state_dict(checkpoint_object['optimizer_D'])

        logger.info(f"Loading runner {args.runner}")
        runner = load_module('runners', args.runner)
        optimizer_G = runner.get_optimizer(modules['embedder'], modules['generator'], args)
        if 'generator' in differing_args or 'embedder' in differing_args or finetune and not already_finetuned:
            logger.warning(f"Embedder or generator has changed in config, so not loading `optimizer_G`")
        else:
            optimizer_G.load_state_dict(checkpoint_object['optimizer_G'])

    saved_args.device = saved_args_device_backup

    return \
        modules['embedder'], modules['generator'], modules['discriminator'], \
        running_averages, saved_args, optimizer_G, optimizer_D


def torch_image_to_numpy(image_torch, inplace=False):
    """Convert PyTorch tensor to Numpy array.
    :param image_torch: PyTorch float CHW Tensor in range [0..1].
    :param inplace: modify the tensor in-place.
    :returns: Numpy uint8 HWC array in range [0..255]."""
    if not inplace:
        image_torch = image_torch.clone()
    return image_torch.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


import argparse


class ActionNoYes(argparse.Action):
    def __init__(self,
                option_strings,
                dest,
                nargs=0,
                const=None,
                default=None,
                type=None,
                choices=None,
                required=False,
                help="",
                metavar=None):

        assert len(option_strings) == 1
        assert option_strings[0][:2] == '--'

        name= option_strings[0][2:]
        help += f'Use "--{name}" for True, "--no-{name}" for False'
        super(ActionNoYes, self).__init__(['--' + name, '--no-' + name],
                                          dest=dest,
                                          nargs=nargs,
                                          const=const,
                                          default=default,
                                          type=type,
                                          choices=choices,
                                          required=required,
                                          help=help,
                                          metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


class MyArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(MyArgumentParser, self).__init__(**kwargs)
        self.register('action', 'store_bool', ActionNoYes)

    def add(self, *args, **kwargs):
        return self.add_argument(*args, **kwargs)

import torch
from pathlib import Path
import shutil
import sys
import os
import cv2
import numpy as np


def make_visual(data, n_samples=2):
    output_images = []

    if 'enc_rgbs' in data:
        enc_rgb = data['enc_rgbs'][:n_samples, 0]
        output_images.append(("Identity src", enc_rgb))

    def add_one_driver(suffix, annotation):
        if 'dec_stickmen' + suffix in data:
            dec_stickmen = data['dec_stickmen' + suffix][:n_samples]
            if len(dec_stickmen.shape) > 4:
                dec_stickmen = dec_stickmen[:, 0]
            output_images.append((f"Pose src ({annotation})", dec_stickmen))
        elif 'pose_input_rgbs_cropped_voxceleb1' + suffix in data:
            real_rgb = data['pose_input_rgbs_cropped_voxceleb1' + suffix][:n_samples]
            if len(real_rgb.shape) > 4:
                real_rgb = real_rgb[:, 0]
            output_images.append((f"Pose src ({annotation})", real_rgb))
        elif 'target_rgbs' + suffix in data:
            real_rgb = data['target_rgbs' + suffix][:n_samples]
            if len(real_rgb.shape) > 4:
                real_rgb = real_rgb[:, 0]
            output_images.append((f"Pose target ({annotation})", real_rgb))
        if 'pose_input_rgbs' + suffix in data:
            real_rgb = data['pose_input_rgbs' + suffix][:n_samples]
            if len(real_rgb.shape) > 4:
                real_rgb = real_rgb[:, 0]
            output_images.append((f"Pose input ({annotation})", real_rgb))
        if 'fake_rgbs' + suffix in data:
            fake_rgb = data['fake_rgbs' + suffix][:n_samples]
            if len(fake_rgb.shape) > 4:
                fake_rgb = fake_rgb[:, 0]
            output_images.append(("Generator output", fake_rgb))

    add_one_driver('', 'same video')

    if 'real_segm' in data:
        real_segm = data['real_segm'][:n_samples]
        if len(real_segm.shape) > 4:
            real_segm = real_segm[:, 0]
        output_images.append(("True segmentation", real_segm))
    if 'fake_segm' in data:
        fake_segm = data['fake_segm'][:n_samples]
        if len(fake_segm.shape) > 4:
            fake_segm = fake_segm[:, 0]
        fake_segm = torch.cat([fake_segm]*3, dim=1)
        output_images.append(("Predicted segmentation", fake_segm))

    add_one_driver('_other_video', 'other video')
    add_one_driver('_other_person', 'other person')

    assert len(set(image.shape for _, image in output_images)) == 1 # all images are of same size
    with torch.no_grad():
        output_image_rows = torch.cat([image.cpu() for _, image in output_images], dim=3)

    captions_height = 38
    captions = [np.ones((captions_height, image.shape[3], 3), np.float32) for _, image in output_images]
    for caption_image, (text, _) in zip(captions, output_images):
        cv2.putText(caption_image, text, (1, captions_height-4), cv2.FONT_HERSHEY_PLAIN, 1.25, (0,0,0), 2)
    captions = np.concatenate(captions, axis=1)
    captions = torch.tensor(captions).permute(2,0,1).contiguous()

    return output_image_rows, captions


# TODO obsolete, remove
class Saver:
    def __init__(self, save_dir, save_fn='npz_per_batch', clean_dir=False):
        super(Saver, self).__init__()
        self.save_dir = Path(str(save_dir))
        self.need_save = True

        if clean_dir and os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

        os.makedirs(self.save_dir, exist_ok=True)

        self.save_fn = sys.modules[__name__].__dict__[save_fn]

    def save(self, epoch, **kwargs):
        self.save_fn(save_dir=self.save_dir, epoch=epoch, **kwargs)

# Source: https://github.com/LiyuanLucasLiu/RAdam
import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
                    
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)


                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

import torch
import os.path
import datetime
import shutil
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import logging # standard python logging
logger = logging.getLogger('tensorboard_logging')

class MySummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disk_space_available = True

    def check_if_disk_space_available(self):
        free_space_on_disk_MB = shutil.disk_usage(self.log_dir).free / 1024**2
        actually_disk_space_available = free_space_on_disk_MB > 1024
        if self.disk_space_available != actually_disk_space_available: # change of state, let's log it
            self.disk_space_available = actually_disk_space_available
            if actually_disk_space_available:
                logger.info("Disk space has freed up! Resuming tensorboard logging")
            else:
                logger.error("Stopping tensorboard logging: disk low on space")
        return actually_disk_space_available

    def add_scalar(self, *args):
        if self.check_if_disk_space_available():
            super().add_scalar(*args)

    def add_image(self, name, images_minibatch, captions, iteration):
        """
            images_minibatch: B x 3 x H x (k*W), float
            captions: 3 x h x (k*W), float
        """
        if self.check_if_disk_space_available():
            grid = make_grid(images_minibatch.detach().clamp(0, 1).data.cpu(), nrow=1)
            # Pad captions because `make_grid` also adds side padding
            captions = torch.nn.functional.pad(captions, () if len(images_minibatch) == 1 else (2,2))
            grid = torch.cat((captions, grid), dim=1) # Add a header with captions on top

            super().add_image(name, grid, iteration)


def get_postfix(args, default_args, args_to_ignore, delimiter='__'):
    s = []

    for arg in sorted(args.keys()):
        if not isinstance(arg, Path) and arg not in args_to_ignore and default_args[arg] != args[arg]:
            s += [f"{arg}^{args[arg]}"]

    return delimiter.join(s).replace('/', '+')  # .replace(';', '+')


def setup_logging(args, default_args, args_to_ignore, exp_name_use_date=True, tensorboard=True):
    if not args.experiment_name:
        args.experiment_name = get_postfix(vars(args), vars(default_args), args_to_ignore)
    
        if exp_name_use_date:
            time = datetime.datetime.now()
            args.experiment_name = time.strftime(f"%m-%d_%H-%M___{args.experiment_name}")

    save_dir = os.path.join(args.experiments_dir, args.experiment_name)
    os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)

    writer = MySummaryWriter(save_dir, filename_suffix='_train') if tensorboard else None

    return save_dir, writer

import numpy as np
import cv2
import torch
torch.set_grad_enabled(False)

from utils.crop_as_in_dataset import ImageWriter
from utils import utils

from pathlib import Path

from tqdm import tqdm

def string_to_valid_filename(x):
    return str(x).replace('/', '_')

if __name__ == '__main__':
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger('drive')

    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Render 'puppeteering' videos, given a fine-tuned model and driving images.\n"
                    "Be careful: inputs have to be preprocessed by 'utils/preprocess_dataset.sh'.",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('checkpoint_path', type=Path,
        help="Path to the *.pth checkpoint of a fine-tuned neural renderer model.")
    arg_parser.add_argument('data_root', type=Path,
        help="Driving images' source: \"root path\" that contains folders\n"
             "like 'images-cropped', 'segmentation-cropped-ffhq', or 'keypoints-cropped'.")
    arg_parser.add_argument('--images_paths', type=Path, nargs='+',
        help="Driving images' sources: paths to folders with driving images, relative to "
             "'`--data_root`/images-cropped' (note: here 'images-cropped' is the "
             "checkpoint's `args.img_dir`). Example: \"id01234/q2W3e4R5t6Y monalisa\".")
    arg_parser.add_argument('--destination', type=Path, required=True,
        help="Where to put the resulting videos: path to an existing folder.")
    args = arg_parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Will run on device '{device}'")

    # Initialize the model
    logger.info(f"Loading checkpoint from '{args.checkpoint_path}'")
    checkpoint_object = torch.load(args.checkpoint_path, map_location='cpu')

    import copy
    saved_args = copy.copy(checkpoint_object['args'])
    saved_args.finetune = True
    saved_args.inference = True
    saved_args.data_root = args.data_root
    saved_args.world_size = 1
    saved_args.num_workers = 1
    saved_args.batch_size = 1
    saved_args.device = device
    saved_args.bboxes_dir = Path("/non/existent/file")
    saved_args.prefetch_size = 4

    embedder, generator, _, running_averages, _, _, _ = \
        utils.load_model_from_checkpoint(checkpoint_object, saved_args)

    if 'embedder' in running_averages:
        embedder.load_state_dict(running_averages['embedder'])
    if 'generator' in running_averages:
        generator.load_state_dict(running_averages['generator'])

    embedder.train(not saved_args.set_eval_mode_in_test)
    generator.train(not saved_args.set_eval_mode_in_test)

    for driver_images_path in args.images_paths:
        # Initialize the data loader
        saved_args.val_split_path = driver_images_path
        from dataloaders.dataloader import Dataloader
        logger.info(f"Loading dataloader '{saved_args.dataloader}'")
        dataloader = Dataloader(saved_args.dataloader).get_dataloader(saved_args, part='val', phase='val')

        current_output_path = (args.destination / string_to_valid_filename(driver_images_path)).with_suffix('.mp4')
        current_output_path.parent.mkdir(parents=True, exist_ok=True)
        image_writer = ImageWriter.get_image_writer(current_output_path)

        for data_dict, _ in tqdm(dataloader):
            utils.dict_to_device(data_dict, device)

            embedder.get_pose_embedding(data_dict)
            generator(data_dict)

            def torch_to_opencv(image):
                image = image.permute(1,2,0).clamp_(0, 1).mul_(255).cpu().byte().numpy()
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR, dst=image)

            result = torch_to_opencv(data_dict['fake_rgbs'][0])
            pose_driver = torch_to_opencv(data_dict['pose_input_rgbs'][0, 0])

            frame_grid = np.concatenate((pose_driver, result), axis=1)
            image_writer.add(frame_grid)

import time
import torch
from torch import nn
from tqdm import tqdm
import utils.radam
torch.optim.RAdam = utils.radam.RAdam

from utils.visualize import make_visual
from utils.utils import Meter
from utils import utils

import itertools
import copy

import logging # standard python logging
logger = logging.getLogger('runner')

def get_args(parser):
    parser.add('--iteration', type=int, default=0, help="Optional iteration number to start from")
    parser.add('--log_frequency_loss', type=int, default=1)
    parser.add('--log_frequency_images', type=int, default=100)
    parser.add('--log_frequency_fixed_images', type=int, default=2500)
    parser.add('--detailed_metrics', action='store_bool', default=True)
    # parser.add('--num_steps_dis_per_gen', default=2, type=int)
    parser.add('--num_visuals_per_img', default=2, type=int)
    parser.add('--fixed_val_ids', action='append', type=int, default=[50, 100, 200, 250, 300])
    parser.add('--batch_size_inference', default=5, type=int,
        help="Batch size for processing 'fixed_val_ids' during visualization. Different from 'batch_size', "
             "this number is for one GPU (visualization inference is currently done on 1 GPU anyway).")

    return parser


def get_optimizer(embedder, generator, args):
    model_parameters = list(generator.parameters())
    if 'finetune' not in args or not args.finetune: # TODO backward compatibility, remove `'finetune' not in args or`
        model_parameters += list(embedder.parameters())

    Optimizer = torch.optim.__dict__[args.optimizer]
    optimizer = Optimizer(model_parameters, lr=args.lr_gen, betas=(args.beta1, 0.999), eps=1e-5)
    return optimizer


class TrainingModule(torch.nn.Module):
    def __init__(self, embedder, generator, discriminator, criterion_list, metric_list, running_averages={}):
        """
            `embedder`, `generator`, `discriminator`: `nn.Module`s
            `criterion_list`, `metric_list`: a list of `nn.Module`s
            `running_averages`: `None` or a dict of {`str`: `nn.Module`}
                Optional initial states of weights' running averages (useful when resuming training).
                Can provide running averages just for some modules (e.g. for none or only for generator).
                If `None`, don't track running averages at all.
        """
        super().__init__()
        self.embedder = embedder
        self.generator = generator
        self.discriminator = discriminator
        self.criterion_list = nn.ModuleList(criterion_list)
        self.metric_list = nn.ModuleList(metric_list)

        self.compute_losses = True
        self.use_running_averages = False
        self.initialize_running_averages(running_averages)

    def initialize_running_averages(self, initial_values={}):
        """
            Set up weights' running averages for generator and discriminator.

            initial_values: `dict` of `nn.Module`, or `None`
                `None` means do not use running averages at all.

                Otherwise, `initial_values['embedder']` will be used as
                the initla value for embedder's running average. Same for generator.
                If `initial_values['embedder']` is missing, then embedder's running average
                will be initialized to embedder's current weights.
        """
        self.running_averages = {}

        if initial_values is not None:
            for name in 'embedder', 'generator':
                model = getattr(self, name)
                self.running_averages[name] = copy.deepcopy(model)
                try:
                    initial_value = initial_values[name]
                    self.running_averages[name].load_state_dict(initial_value)
                except KeyError:
                    logger.info(
                        f"No initial value of weights' running averages provided for {name}. Initializing by cloning")
                except:
                    logger.warning(
                        f"Parameters mismatch in {name} and the initial value of weights' "
                        f"running averages. Initializing by cloning")
                    self.running_averages[name].load_state_dict(model.state_dict())

        for module in self.running_averages.values():
            module.eval()
            module.requires_grad_(False)

    def update_running_average(self, alpha=0.999):
        with torch.no_grad():
            for model_name, model_running_avg in self.running_averages.items():
                model_current = getattr(self, model_name)

                for p_current, p_running_average in zip(model_current.parameters(), model_running_avg.parameters()):
                    p_running_average *= alpha
                    p_running_average += p_current * (1-alpha)

                for p_current, p_running_average in zip(model_current.buffers(), model_running_avg.buffers()):
                    p_running_average.copy_(p_current)

    def set_use_running_averages(self, use_running_averages=True):
        """
            Changes `training_module.use_running_averages` to the specified value.
            Can be used either as a context manager or as a separate method call.

            TODO: migrate to contextlib
        """
        class UseRunningAveragesContextManager:
            def __init__(self, training_module, use_running_averages):
                self.training_module = training_module
                self.old_value = self.training_module.use_running_averages
                self.training_module.use_running_averages = use_running_averages
            
            def __enter__(self):
                pass

            def __exit__(self, *args):
                self.training_module.use_running_averages = self.old_value

        return UseRunningAveragesContextManager(self, use_running_averages)

    def set_compute_losses(self, compute_losses=True):
        """
            Changes `training_module.compute_losses` to the specified value.
            Can be used either as a context manager or as a separate method call.

            TODO: migrate to contextlib
        """
        class ComputeLossesContextManager:
            def __init__(self, training_module, compute_losses):
                self.training_module = training_module
                self.old_value = self.training_module.compute_losses
                self.training_module.compute_losses = compute_losses
            
            def __enter__(self):
                pass

            def __exit__(self, *args):
                self.training_module.compute_losses = self.old_value

        return ComputeLossesContextManager(self, compute_losses)

    def forward(self, data_dict, target_dict):
        if self.running_averages and self.use_running_averages:
            embedder = self.running_averages['embedder']
            generator = self.running_averages['generator']
        else:
            embedder = self.embedder
            generator = self.generator

        # First, let the new `data_dict` (the return value) hold only inputs.
        # As we run modules, `data_dict` will gradually get filled with outputs too.
        data_dict = copy.copy(data_dict)

        embedder(data_dict)
        generator(data_dict)

        # Now add target data to `data_dict` too, to run discriminator and calculate losses if needed
        data_dict.update(target_dict)
        if self.compute_losses:
            self.discriminator(data_dict)
        
        losses_G_dict = {}
        losses_D_dict = {}

        for criterion in self.criterion_list:
            try:
                crit_out = criterion(data_dict)
            except:
                # couldn't compute this loss; if validating, skip it
                if self.compute_losses:
                    raise
                else:
                    continue

            if isinstance(crit_out, tuple):
                if len(crit_out) != 2:
                    raise TypeError(
                        f'Unexpected number of outputs in criterion {type(criterion)}: '
                        f'expected 2, got {len(crit_out)}')
                crit_out_G, crit_out_D = crit_out
                losses_G_dict.update(crit_out_G)
                losses_D_dict.update(crit_out_D)
            elif isinstance(crit_out, dict):
                losses_G_dict.update(crit_out)
            else:
                raise TypeError(
                    f'Unexpected type of {type(criterion)} output: '
                    f'expected dict or tuple of two dicts, got {type(crit_out)}')

        return data_dict, losses_G_dict, losses_D_dict

    def compute_metrics(self, data_dict):
        metrics_meter = Meter()
        for metric in self.metric_list:
            metric_out, num_errors = metric(data_dict)
            for metric_output_name, metric_value in metric_out.items():
                metrics_meter.add(metric_output_name, metric_value, num_errors[metric_output_name])

        return metrics_meter

def run_epoch(dataloader, training_module, optimizer_G, optimizer_D, epoch, args,
              phase,
              writer=None,
              saver=None):
    meter = Meter()

    if phase == 'train':
        optimizer_G.zero_grad()
        if optimizer_D:
            optimizer_D.zero_grad()

    end = time.time()
    for it, (data_dict, target_dict) in enumerate(dataloader):
        meter.add('Data_time', time.time() - end)

        utils.dict_to_device(data_dict  , args.device)
        utils.dict_to_device(target_dict, args.device)

        all_data_dict, losses_G_dict, losses_D_dict = training_module(data_dict, target_dict)

        # The isinstance() check is a workaround. We have some metrics that are implemented as losses
        # (currently gaze angle metric), and those are not differentiable, and occasionally take NaN values.
        # We don't need to include them in the cumulative loss_{G,D} values, as those are used for backprop only.
        loss_G = sum(v for v in losses_G_dict.values() if isinstance(v, torch.Tensor))
        loss_D = sum(v for v in losses_D_dict.values() if isinstance(v, torch.Tensor))

        if phase == 'train':
            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            if args.num_gpus > 1 and args.num_gpus <= 8:
                training_module.reducer.reduce()

            optimizer_G.step()
            
            if losses_D_dict:
                optimizer_D.zero_grad()
                loss_D.backward()
                if args.num_gpus > 1 and args.num_gpus <= 8:
                    training_module.reducer.reduce()

                optimizer_D.step()

        if phase == 'val' and saver is not None:
            saver.save(epoch=epoch, data=all_data_dict)

        training_module.update_running_average(0.972 if args.finetune else 0.999)

        # Log
        if args.detailed_metrics:
            for loss_name, loss_ in itertools.chain(losses_G_dict.items(), losses_D_dict.items()):
                meter.add(f'Loss_{loss_name}', float(loss_))

        del loss_G, loss_D, losses_G_dict, losses_D_dict

        def try_other_driving_images(data_dict, suffix, same_identity=False, deterministic=False):
            """
            For each sample in the given `data_dict`, pick a different driving image
            ('pose_input_rgbs'), run the model again with those drivers and save its new outputs
            ('fake_rgbs', 'fake_segm' etc.) -- as well as new inputs (e.g. 'pose_input_rgbs') --
            to `data_dict` with a given `suffix` (e.g. 'pose_input_rgbs_cross_driving' and
            'fake_rgbs_cross_driving' if `suffix` is '_cross_driving').

            data_dict:
                `dict`
                As returned by dataloaders.
            suffix:
                `str`
                See above.
            same_identity:
                `bool`
                If `True`, pick new drivers from other (if possible) videos of the same person.
                Else, pick them from videos of other people.
            deterministic:
                `bool`
                Whether to choose fixed (`True`) or random (`False`) drivers.
            """
            given_samples_labels = data_dict['label'].tolist()

            # Why double `.dataset`? Because dataloader links to a `torch.utils.data.Subset`, which
            # in turn links to our full VoxCeleb dataset
            other_samples_indices = \
                [dataloader.dataset.dataset.get_other_sample_by_label(l, same_identity=same_identity,\
                    deterministic=deterministic) for l in given_samples_labels]

            other_samples = [dataloader.dataset.dataset[i][0] for i in other_samples_indices]
            other_samples = dataloader.collate_fn(other_samples)

            # First, backup original inputs for visualization
            keys_to_backup = \
                'pose_input_rgbs', 'target_rgbs', '3dmm_pose', \
                'fake_rgbs', 'real_segm', 'fake_segm', 'dec_stickmen', 'dec_keypoints'
            backup = {key: data_dict[key] for key in keys_to_backup if key in data_dict}
            # Then replace that data with new inputs
            for key in keys_to_backup:
                if key in other_samples:
                    data_dict[key] = other_samples[key].to(args.device)

            updated_data_dict, _, _ = training_module(data_dict, {})    
            data_dict.update(updated_data_dict)

            # Finally, save new inputs and outputs by a new key, and restore backup
            for key in backup:
                if key in data_dict:
                    data_dict[key + suffix] = data_dict[key]
                    data_dict[key] = backup[key]


        if writer is not None and phase == 'train':
            if args.iteration % args.log_frequency_loss == 0:
                for metric in meter.keys():
                    writer.add_scalar(f'Metrics/{phase}/{metric}', meter.get_last(metric), args.iteration)

            if args.iteration % args.log_frequency_images == 0:
                # Visualize how a person drives self but from other video

                # Re-evaluate embedding because embedder can behave differently in .train() and .eval()
                training_module.train(not args.set_eval_mode_in_test)
                with torch.no_grad():
                    with training_module.set_use_running_averages(), training_module.set_compute_losses(False):
                        all_data_dict, _, _ = training_module(data_dict, {'label': target_dict['label']})
                        if not args.finetune:
                            try_other_driving_images(all_data_dict, suffix='_other_video', same_identity=True)
                            try_other_driving_images(all_data_dict, suffix='_other_person', same_identity=False)
                training_module.train(not args.set_eval_mode_in_train)

                try:
                    del all_data_dict['dec_stickmen']
                except KeyError:
                    pass
                logging_images, captions = make_visual(all_data_dict, n_samples=args.num_visuals_per_img)
                writer.add_image(f'Images/{phase}/visual', logging_images, captions, args.iteration)
                
            if args.iteration % args.log_frequency_fixed_images == 0 and args.fixed_val_ids:
                # Make data loading functions deterministic to make sure same images are sampled from a directory
                was_deterministic = dataloader.dataset.dataset.loader.deterministic
                dataloader.dataset.dataset.loader.deterministic = True

                metrics_meter = Meter()

                # Also, make augmentations always same for each sample in batch
                with dataloader.dataset.dataset.deterministic_(666):
                    # Iterate over `args.fixed_data_dict` in batches
                    for first_sample_idx in range(0, len(args.fixed_val_ids), args.batch_size_inference):
                        batch_sample_ids = args.fixed_val_ids[first_sample_idx:first_sample_idx+args.batch_size_inference]
                        fixed_data_dict = [dataloader.dataset.dataset[i] for i in batch_sample_ids]
                        fixed_data_dict, fixed_target_dict = dataloader.collate_fn(fixed_data_dict)
                        fixed_data_dict.update(fixed_target_dict)
                        utils.dict_to_device(fixed_data_dict, args.device)

                        training_module.train(not args.set_eval_mode_in_test)
                        with torch.no_grad():
                            with training_module.set_use_running_averages(), training_module.set_compute_losses(False):
                                fixed_all_data_dict, _, _ = training_module(fixed_data_dict, {})
                                if not args.finetune:
                                    try_other_driving_images(fixed_all_data_dict, suffix='_other_video', same_identity=True, deterministic=True)
                                    try_other_driving_images(fixed_all_data_dict, suffix='_other_person', same_identity=False, deterministic=True)
                        training_module.train(not args.set_eval_mode_in_train)

                        try:
                            del fixed_all_data_dict['dec_stickmen']
                        except KeyError:
                            pass

                        # Visualize the first batch in TensorBoard/WandB
                        if first_sample_idx == 0:
                            logging_images, captions = make_visual(fixed_all_data_dict, n_samples=len(batch_sample_ids))
                            writer.add_image(f'Fixed_images/{phase}/visual', logging_images, captions, args.iteration)

                        with torch.no_grad():
                            metrics_meter += training_module.compute_metrics(fixed_all_data_dict)

                for metric_name in metrics_meter.keys():
                    writer.add_scalar(
                        f'Fixed_metrics/{phase}/{metric_name}', metrics_meter.get_average(metric_name), args.iteration)

                dataloader.dataset.dataset.loader.deterministic = was_deterministic

            if phase == 'train':
                args.iteration += 1

        # Measure elapsed time
        meter.add('Batch_time', time.time() - end)
        end = time.time()

    if writer is not None and phase == 'val':
        for metric in meter.keys():
            writer.add_scalar(f'Metrics/{phase}/{metric}', meter.get_average(metric), args.iteration)
        logging_images, captions = make_visual(all_data_dict, n_samples=args.num_visuals_per_img * 3)
        writer.add_image(f'Images/{phase}/visual', logging_images, captions, args.iteration)

    logger.info(f"Epoch {epoch} {phase.capitalize()} finished")

import torch
from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        pass

    @staticmethod
    def get_net(args):
        return Discriminator().to(args.device)

    @staticmethod
    def get_optimizer(discriminator, args):
        return None

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.finetuning = False

    def enable_finetuning(self, _=None):
        self.finetuning = True

    def forward(self, _):
        pass

import torch
from torch import nn
from torch.nn.utils import spectral_norm

import utils.radam
torch.optim.RAdam = utils.radam.RAdam

from generators.common import blocks
import math

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--dis_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--dis_num_blocks', type=int, default=7)
        parser.add('--lr_dis', type=float, default=2e-4)

    @staticmethod
    def get_net(args):
        net = Discriminator(args.dis_padding, args.in_channels, args.out_channels, args.num_channels,
                            args.max_num_channels, args.embed_channels, args.dis_num_blocks, args.image_size,
                            args.num_labels).to(args.device)
        return net

    @staticmethod
    def get_optimizer(discriminator, args):
        Optimizer = torch.optim.__dict__[args.optimizer]
        return Optimizer(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999), eps=1e-5)


class Discriminator(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, embed_channels,
                 dis_num_blocks, image_size,
                 num_labels):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer='none')

        def get_res_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer='none')

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d

        self.out_channels = embed_channels
        in_channels = (in_channels + out_channels)

        self.down_block = nn.Sequential(
            padding(1),
            spectral_norm(
                nn.Conv2d(in_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.ReLU(),
            padding(1),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))

        self.blocks = nn.ModuleList()
        num_down_blocks = min(int(math.log(image_size, 2)) - 2, dis_num_blocks)
        in_channels = num_channels
        for i in range(1, num_down_blocks):
            out_channels = min(in_channels * 2, max_num_channels)
            if i == dis_num_blocks - 1: out_channels = self.out_channels
            self.blocks.append(get_down_block(in_channels, out_channels, padding))
            in_channels = out_channels
        for i in range(num_down_blocks, dis_num_blocks):
            if i == dis_num_blocks - 1: out_channels = self.out_channels
            self.blocks.append(get_res_block(in_channels, out_channels, padding))

        self.linear = spectral_norm(nn.Linear(self.out_channels, 1), eps=1e-4)

        # Embeddings for identities
        embed = nn.Embedding(num_labels, self.out_channels)
        embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(embed, eps=1e-4)

        self.finetuning = False

    def pass_inputs(self, input, embed=None):
        scores = []
        feats = []

        out = self.down_block(input)
        out = out + self.skip(input)
        feats.append(out)
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        out = torch.relu(out)
        out = out.view(out.shape[0], self.out_channels, -1).sum(2)
        out_linear = self.linear(out)[:, 0]

        if embed is not None:
            scores.append((out * embed).sum(1) + out_linear)
        else:
            scores.append(out_linear)
        return scores[0], feats

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to discriminator architecture to allow fine-tuning.
            For `vanilla` discriminator, replace embedding matrix W (`self.embed`) with one
            vector `data_dict['embeds']`.

            data_dict:
                dict
                Required contents depend on the specific discriminator. For `vanilla` discriminator,
                it is `'embeds'` (1 x `args.embed_channels`).
        """
        some_parameter = next(iter(self.parameters())) # to know target device and dtype

        if data_dict is None:
            data_dict = {
                'embeds': torch.rand(1, self.out_channels).to(some_parameter)
            }

        with torch.no_grad():
            if self.finetuning:
                self.embed.weight_orig.copy_(data_dict['embeds'])
            else:
                new_embedding_matrix = nn.Embedding(1, self.out_channels).to(some_parameter)
                new_embedding_matrix.weight.copy_(data_dict['embeds'])
                self.embed = spectral_norm(new_embedding_matrix)
                
                self.finetuning = True

    def forward(self, data_dict):
        fake_rgbs = data_dict['fake_rgbs']
        target_rgbs = data_dict['target_rgbs']
        dec_stickmen = data_dict['dec_stickmen']
        label = data_dict['label']

        if len(fake_rgbs.shape) > 4:
            fake_rgbs = fake_rgbs[:, 0]
        if len(target_rgbs.shape) > 4:
            target_rgbs = target_rgbs[:, 0]
        if len(dec_stickmen.shape) > 4:
            dec_stickmen = dec_stickmen[:, 0]

        b, c_in, h, w = dec_stickmen.shape

        embed = None
        if hasattr(self, 'embed'):
            embed = self.embed(label)

        disc_common = dec_stickmen

        fake_in = torch.cat([disc_common, fake_rgbs], dim=2).view(b, -1, h, w)
        fake_score_G, fake_features = self.pass_inputs(fake_in, embed)
        fake_score_D, _ = self.pass_inputs(fake_in.detach(), embed.detach())

        real_in = torch.cat([disc_common, target_rgbs], dim=2).view(b, -1, h, w)
        real_score, real_features = self.pass_inputs(real_in, embed)

        data_dict['fake_features'] = fake_features
        data_dict['real_features'] = real_features
        data_dict['real_embedding'] = embed
        data_dict['fake_score_G'] = fake_score_G
        data_dict['fake_score_D'] = fake_score_D
        data_dict['real_score'] = real_score

import torch
from torch import nn
from torch.nn.utils import spectral_norm

import utils.radam
torch.optim.RAdam = utils.radam.RAdam

from generators.common import blocks
import math

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--dis_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--dis_num_blocks', type=int, default=7)
        parser.add('--lr_dis', type=float, default=2e-4)

    @staticmethod
    def get_net(args):
        net = Discriminator(args.dis_padding, args.in_channels, args.out_channels, args.num_channels,
                            args.max_num_channels, args.embed_channels, args.dis_num_blocks, args.image_size,
                            args.num_labels).to(args.device)
        return net

    @staticmethod
    def get_optimizer(discriminator, args):
        Optimizer = torch.optim.__dict__[args.optimizer]
        return Optimizer(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999), eps=1e-5)


class Discriminator(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, embed_channels,
                 dis_num_blocks, image_size,
                 num_labels):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer='none')

        def get_res_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer='none')

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d

        self.out_channels = embed_channels

        self.down_block = nn.Sequential(
            # padding(1),
            spectral_norm(
                nn.Conv2d(in_channels, num_channels, 3, 1, 1),
                eps=1e-4),
            nn.ReLU(),
            # padding(1),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 1),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))

        self.blocks = nn.ModuleList()
        num_down_blocks = min(int(math.log(image_size, 2)) - 2, dis_num_blocks)
        in_channels = num_channels
        for i in range(1, num_down_blocks):
            out_channels = min(in_channels * 2, max_num_channels)
            if i == dis_num_blocks - 1: out_channels = self.out_channels
            self.blocks.append(get_down_block(in_channels, out_channels, padding))
            in_channels = out_channels
        for i in range(num_down_blocks, dis_num_blocks):
            if i == dis_num_blocks - 1: out_channels = self.out_channels
            self.blocks.append(get_res_block(in_channels, out_channels, padding))

        self.linear = spectral_norm(nn.Linear(self.out_channels, 1), eps=1e-4)

        # Embeddings for identities
        embed = nn.Embedding(num_labels, self.out_channels)
        embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(embed, eps=1e-4)

        self.finetuning = False

    def pass_inputs(self, input, embed=None):
        scores = []
        feats = []

        out = self.down_block(input)
        out = out + self.skip(input)
        feats.append(out)
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        out = torch.relu(out)
        out = out.view(out.shape[0], self.out_channels, -1).sum(2)
        out_linear = self.linear(out)[:, 0]

        if embed is not None:
            scores.append((out * embed).sum(1) + out_linear)
        else:
            scores.append(out_linear)
        return scores[0], feats

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to discriminator architecture to allow fine-tuning.
            For `vanilla` discriminator, replace embedding matrix W (`self.embed`) with one
            vector `data_dict['embeds']`.

            data_dict:
                dict
                Required contents depend on the specific discriminator. For `vanilla` discriminator,
                it is `'embeds'` (1 x `args.embed_channels`).
        """
        some_parameter = next(iter(self.parameters())) # to know target device and dtype

        if data_dict is None:
            data_dict = {
                'embeds': torch.rand(1, self.out_channels).to(some_parameter)
            }

        with torch.no_grad():
            if self.finetuning:
                self.embed.weight_orig.copy_(data_dict['embeds'])
            else:
                new_embedding_matrix = nn.Embedding(1, self.out_channels).to(some_parameter)
                new_embedding_matrix.weight.copy_(data_dict['embeds'])
                self.embed = spectral_norm(new_embedding_matrix)
                
                self.finetuning = True

    def forward(self, data_dict):
        fake_rgbs = data_dict['fake_rgbs']
        target_rgbs = data_dict['target_rgbs']
        label = data_dict['label']

        if len(fake_rgbs.shape) > 4:
            fake_rgbs = fake_rgbs[:, 0]
        if len(target_rgbs.shape) > 4:
            target_rgbs = target_rgbs[:, 0]
        
        b, c_in, h, w = target_rgbs.shape

        embed = None
        if hasattr(self, 'embed'):
            embed = self.embed(label)

        fake_in = fake_rgbs
        fake_score_G, fake_features = self.pass_inputs(fake_in, embed)
        fake_score_D, _ = self.pass_inputs(fake_in.detach(), embed.detach())

        real_in = target_rgbs
        real_score, real_features = self.pass_inputs(real_in, embed)

        data_dict['fake_features'] = fake_features
        data_dict['real_features'] = real_features
        data_dict['real_embedding'] = embed
        data_dict['fake_score_G'] = fake_score_G
        data_dict['fake_score_D'] = fake_score_D
        data_dict['real_score'] = real_score

import torch
from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gan_type', type=str, default='gan', help='gan|rgan|ragan')

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.gan_type)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, gan_type):
        super().__init__()
        self.gan_type = gan_type

    def get_dis_preds(self, real_score, fake_score):
        if self.gan_type == 'gan':
            real_pred = real_score
            fake_pred = fake_score
        elif self.gan_type == 'rgan':
            real_pred = real_score - fake_score
            fake_pred = fake_score - real_score
        elif self.gan_type == 'ragan':
            real_pred = real_score - fake_score.mean()
            fake_pred = fake_score - real_score.mean()
        else:
            raise Exception('Incorrect `gan_type` argument')
        return real_pred, fake_pred

    def forward(self, data_dict):
        fake_score_G = data_dict['fake_score_G']
        fake_score_D = data_dict['fake_score_D']
        real_score = data_dict['real_score']

        real_pred, fake_pred_D = self.get_dis_preds(real_score, fake_score_D)
        _, fake_pred_G = self.get_dis_preds(real_score, fake_score_G)

        loss_D = torch.relu(1. - real_pred).mean() + torch.relu(1. + fake_pred_D).mean()  # TODO: check correctness

        if self.gan_type == 'gan':
            loss_G = -fake_pred_G.mean()
        elif 'r' in self.gan_type:
            loss_G = torch.relu(1. + real_pred).mean() + torch.relu(1. - fake_pred_G).mean()
        else:
            raise Exception('Incorrect `gan_type` argument')

        loss_G_dict = {}
        loss_G_dict['adversarial_G'] = loss_G

        loss_D_dict = {}
        loss_D_dict['adversarial_D'] = loss_D

        return loss_G_dict, loss_D_dict

from torch import nn
import torch.nn.functional as F

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--fm_weight', type=float, default=10.0)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.fm_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, fm_weight):
        super().__init__()
        self.fm_crit = lambda inputs, targets: sum(
            [F.l1_loss(input, target.detach()) for input, target in zip(inputs, targets)]) / len(
            inputs) * fm_weight

    def forward(self, data_dict):
        fake_feats = data_dict['fake_features']
        real_feats = data_dict['real_features']

        loss_G_dict = {}
        loss_G_dict['feature_matching'] = self.fm_crit(fake_feats, real_feats)

        return loss_G_dict

from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--l1_weight', type=float, default=30.0)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.l1_weight)
        return criterion.to(args.device)

class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, inputs):
        fake_rgb = inputs['fake_rgbs']
        real_rgb = inputs['target_rgbs']

        loss_G_dict = {}
        loss_G_dict['l1_rgb'] = self.weight * nn.functional.l1_loss(fake_rgb, real_rgb[:, 0])

        return loss_G_dict

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1)


class PerceptualLoss(nn.Module):
    def __init__(self, weight, vgg_weights_dir, net='caffe', normalize_grad=False):
        super().__init__()
        self.weight = weight
        self.normalize_grad = normalize_grad

        if net == 'pytorch':
            model = torchvision.models.vgg19(pretrained=True).features

            mean = torch.tensor([0.485, 0.456, 0.406])
            std  = torch.tensor([0.229, 0.224, 0.225])

            num_layers = 30

        elif net == 'caffe':
            vgg_weights = torch.load(os.path.join(vgg_weights_dir, 'vgg19-d01eb7cb.pth'))

            map = {'classifier.6.weight': u'classifier.7.weight', 'classifier.6.bias': u'classifier.7.bias'}
            vgg_weights = OrderedDict([(map[k] if k in map else k, v) for k, v in vgg_weights.items()])

            model = torchvision.models.vgg19()
            model.classifier = nn.Sequential(Flatten(), *model.classifier._modules.values())

            model.load_state_dict(vgg_weights)

            model = model.features

            mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
            std = torch.tensor([1., 1., 1.]) / 255.

            num_layers = 30

        elif net == 'face':
            # Load caffe weights for VGGFace, converted from
            # https://media.githubusercontent.com/media/yzhang559/vgg-face/master/VGG_FACE.caffemodel.pth
            # The base model is VGG16, not VGG19.
            model = torchvision.models.vgg16().features
            model.load_state_dict(torch.load(os.path.join(vgg_weights_dir, 'vgg_face_weights.pth')))

            mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
            std = torch.tensor([1., 1., 1.]) / 255.

            num_layers = 30

        else:
            raise ValueError(f"Unknown type of PerceptualLoss: expected '{{pytorch,caffe,face}}', got '{net}'")

        self.register_buffer('mean', mean[None, :, None, None])
        self.register_buffer('std' ,  std[None, :, None, None])

        layers_avg_pooling = []

        for weights in model.parameters():
            weights.requires_grad = False

        for module in model.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                layers_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers_avg_pooling.append(module)

            if len(layers_avg_pooling) >= num_layers:
                break

        layers_avg_pooling = nn.Sequential(*layers_avg_pooling)

        self.model = layers_avg_pooling

    def normalize_inputs(self, x):
        return (x - self.mean) / self.std

    def forward(self, input, target):
        input = (input + 1) / 2
        target = (target.detach() + 1) / 2

        loss = 0

        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)

        for layer in self.model:
            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                if self.normalize_grad:
                    pass
                else:
                    loss = loss + F.l1_loss(features_input, features_target)

        return loss * self.weight

from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--perc_weight', type=float, default=1e-2)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.perc_weight, args.vgg_weights_dir)
        return criterion.to(args.device)

class Criterion(nn.Module):
    def __init__(self, perc_weight, vgg_weights_dir):
        super().__init__()

        self.perceptual_crit = PerceptualLoss(perc_weight, vgg_weights_dir).eval()

    def forward(self, data_dict):
        fake_rgb = data_dict['fake_rgbs']
        real_rgb = data_dict['target_rgbs']

        if len(fake_rgb.shape) > 4:
            fake_rgb = fake_rgb[:, 0]

        if len(real_rgb.shape) > 4:
            real_rgb = real_rgb[:, 0]

        loss_G_dict = {}
        loss_G_dict['VGG'] = self.perceptual_crit(fake_rgb, real_rgb)

        return loss_G_dict

import torch
from torch import nn
import torch.nn.functional as F

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--dis_embed_weight', type=float, default=1e-2)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.dis_embed_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, dis_embed_weight):
        super().__init__()
        self.dis_embed_crit = lambda input, target: F.l1_loss(input, target.detach()) * dis_embed_weight

    def forward(self, data_dict):
        fake_embed = data_dict['embeds_elemwise']
        real_embed = data_dict['real_embedding']

        if len(fake_embed.shape) > 2:
            fake_embed = fake_embed[:, 0]

        if len(real_embed.shape) > 2:
            real_embed = real_embed[:, 0]

        loss_G_dict = {}
        loss_G_dict['embedding_matching'] = self.dis_embed_crit(fake_embed, real_embed)

        return loss_G_dict

import torch
from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--dice_weight', type=float, default=1)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.dice_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, dice_weight):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, data_dict):
        fake_segm = data_dict['fake_segm']
        real_segm = data_dict['real_segm']

        if len(fake_segm.shape) > 4:
            fake_segm = fake_segm[:, 0]

        if len(real_segm.shape) > 4:
            real_segm = real_segm[:, 0]

        numer = (2*fake_segm*real_segm).sum()
        denom =  ((fake_segm**2).sum() + (real_segm**2).sum())

        dice = numer / denom
        loss = -torch.log(dice) * self.dice_weight

        loss_G_dict = {}
        loss_G_dict['segmentation_dice'] = loss

        return loss_G_dict

import torch
from criterions.common.perceptual_loss import PerceptualLoss

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--idt_embed_weight', type=float, default=2e-3)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.idt_embed_weight, args.vgg_weights_dir)
        return criterion.to(args.device)

class Criterion(torch.nn.Module):
    def __init__(self, idt_embed_weight, vgg_weights_dir):
        super().__init__()
        self.idt_embed_crit = PerceptualLoss(idt_embed_weight, vgg_weights_dir, net='face').eval()

    def forward(self, data_dict):
        fake_rgb = data_dict['fake_rgbs']
        real_rgb = data_dict['target_rgbs']

        if len(fake_rgb.shape) > 4:
            fake_rgb = fake_rgb[:, 0]

        if len(real_rgb.shape) > 4:
            real_rgb = real_rgb[:, 0]

        if 'dec_keypoints' in data_dict:
            keypoints = data_dict['dec_keypoints']

            bboxes_estimate = compute_bboxes_from_keypoints(keypoints)

            # convert bboxes from [0; 1] to pixel coordinates
            h, w = real_rgb.shape[2:]
            bboxes_estimate[:, 0:2] *= h
            bboxes_estimate[:, 2:4] *= w
        else:
            crop_factor = 1 / 1.8
            h, w = real_rgb.shape[2:]

            t = h * (1 - crop_factor) / 2
            l = w * (1 - crop_factor) / 2
            b = h - t
            r = w - l
            
            bboxes_estimate = torch.empty((1, 4), dtype=torch.float32, device=real_rgb.device)
            bboxes_estimate[0].copy_(torch.tensor([t, b, l, r]))
            bboxes_estimate = bboxes_estimate.expand(len(real_rgb), 4)

        fake_rgb_cropped = crop_and_resize(fake_rgb, bboxes_estimate)
        real_rgb_cropped = crop_and_resize(real_rgb, bboxes_estimate)

        loss_G_dict = {}
        loss_G_dict['VGGFace'] = self.idt_embed_crit(fake_rgb_cropped, real_rgb_cropped)
        return loss_G_dict

def crop_and_resize(images, bboxes, target_size=None):
    """
    images: B x C x H x W
    bboxes: B x 4; [t, b, l, r], in pixel coordinates
    target_size (optional): tuple (h, w)

    return value: B x C x h x w

    Crop i-th image using i-th bounding box, then resize all crops to the
    desired shape (default is the original images' size, H x W).
    """
    t, b, l, r = bboxes.t().float()
    batch_size, num_channels, h, w = images.shape

    affine_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=images.device)
    affine_matrix[:, 0, 0] = (r-l) / w
    affine_matrix[:, 1, 1] = (b-t) / h
    affine_matrix[:, 0, 2] = (l+r) / w - 1
    affine_matrix[:, 1, 2] = (t+b) / h - 1

    output_shape = (batch_size, num_channels) + (target_size or (h, w))
    try:
        grid = torch.affine_grid_generator(affine_matrix, output_shape, False)
    except TypeError: # PyTorch < 1.4.0
        grid = torch.affine_grid_generator(affine_matrix, output_shape)
    return torch.nn.functional.grid_sample(images, grid, 'bilinear', 'reflection')

def compute_bboxes_from_keypoints(keypoints):
    """
    keypoints: B x 68*2

    return value: B x 4 (t, b, l, r)

    Compute a very rough bounding box approximate from 68 keypoints.
    """
    x, y = keypoints.float().view(-1, 68, 2).transpose(0, 2)

    face_height = y[8] - y[27]
    b = y[8] + face_height * 0.2
    t = y[27] - face_height * 0.47

    midpoint_x = (x.min() + x.max()) / 2
    half_height = (b - t) * 0.5
    l = midpoint_x - half_height
    r = midpoint_x + half_height

    return torch.stack([t, b, l, r], dim=1)

# Feel totally free to modify any variables with capital names
import subprocess
from pathlib import Path
import os
import socket

def string_to_valid_filename(x):
    return x.replace('/', '_')

MODELS = [
    ("X2Face_vanilla", "00000009"),
    ("X2FacePretrained_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBug_Graphonomy", "01497847"),
    ("ExpressionNet_ResNeXt_3xVGGLossWeight_256_bboxes_noBottleneck_Graphonomy", "01080152"),
    ("FAbNetPretrained_ResNeXt_3xVGGLossWeight_bboxes_augScaleXNoShift_Graphonomy_smallerCrop", "01327623"),
    ("Zakharov", "01529383"),
    ("Zakharov_bboxes_vectorPose_noLandmarks_FineTune7xWeightNewMLP", "01464169"),

    # ("Zakharov_bboxes_vectorPose_noLandmarks", "01363326"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02275845"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShiftReally_noBottleneck", "02023609"),
    # ("X2FacePretrained_ResNeXt_3xVGGLossWeight_256_bboxes_noBug_Graphonomy", "01337493"),
    # ("FAbNetPretrained_ResNeXt_3xVGGLossWeight_bboxes_Graphonomy_smallerCrop", "01222652"),

    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck", "02492466"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_cleanData", "01303150"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02227532"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck", "02444553"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "01361933"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "01613709"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck", "02713859"),

    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_FineTune7xWeight", "02714183"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "02359800"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_64_bboxes_aug_noBottleneck", "02191987"),
    ("MobileNetV2_ResNeXt_7xVGGLossWeight_256_bboxes_SAIC0.02_FromLearned", "02742693"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_cleanData", "01607204"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck", "02737273"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02467652"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck_FineTune7xWeight", "02730111"),
]

for MODEL_NAME, ITERATION in MODELS:
    CHECKPOINT_PATH = f"experiments/{MODEL_NAME}/checkpoints/model_{ITERATION}.pth"
    assert Path(CHECKPOINT_PATH).is_file(), CHECKPOINT_PATH

    output_dir = Path(f"puppeteering/{MODEL_NAME}_{ITERATION}")

    DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/VoxCeleb2_30TestIdentities")
    # DATASET_ROOT = Path("/Vol0/user/e.burkov/Datasets/violet/VoxCeleb2_test_finetuning")
    # DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/Identity sources")

    IMAGES_DIR = DATASET_ROOT / "images-cropped"

    IDENTITIES = [
        "id00061/cAT9aR8oFx0/identity",
        "id00061/Df_m1slf_hY/identity",
        "id00812/XoAi2n4S2wo/identity",
        "id01106/B08yOvYMF7Y/identity",
        "id01228/7qHTvs0VO68/identity",
        "id01333/9kgJaduwKkY/identity",
        "id01437/4lFDvxXzYWY/identity",
        "id02057/s5VqJY7DDEE/identity",
        "id02548/x2LUQEUXdz4/identity",
        "id03127/uiRiyK8Qlic/identity",
        "id03178/cCoNRuzAL-A/identity",
        "id03178/fnARFfUwf2s/identity",
        "id03524/GkvScYvOJ7o/identity",
        "id03839/LhI_8AWX_Mg/identity",
        "id03839/PUwanP-C5qg/identity",
        "id03862/fsCqKQb9Rdg/identity",
        "id04094/JUYMzfVp8zI/identity",
        "id04950/PQEAck-3wcA/identity",
        "id05459/3TI6dVmEwzw/identity",
        "id05714/wFGNufaMbDY/identity",
        "id06104/7UnGAS5-jpU/identity",
        "id06811/KmvEwL3fP9Q/identity",
        "id07312/h1dszoDi1E8/identity",
        "id07663/54qlJ2HZ08s/identity",
        "id07802/BfQUBDw7TiM/identity",
        "id07868/JC0QT4oXh2Y/identity",
        "id07961/464OHFffwjI/identity",
        "id07961/hROZwL8pbGg/identity",
        "id08149/vxBFGKGXSFA/identity",
        "id08701/UeUyLqpLz70/identity",
    ]

    for identity in IDENTITIES:
        experiment_name = string_to_valid_filename(identity)
        checkpoint_output_dir = output_dir / experiment_name
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        if (checkpoint_output_dir / 'checkpoints').is_dir() and len(list((checkpoint_output_dir / 'checkpoints').iterdir())) > 0:
            print(f"Skipping {checkpoint_output_dir}")
            continue

        num_images = sum(1 for _ in (IMAGES_DIR / identity).iterdir())
        MAX_BATCH_SIZE = 7 # 8 is memory limit for MobileNetV2+ResNeXt50 on P100
        batch_size = min(num_images, MAX_BATCH_SIZE)

        TARGET_NUM_ITERATIONS = 560
        iterations_in_epoch = num_images // batch_size
        num_epochs = (TARGET_NUM_ITERATIONS + iterations_in_epoch - 1) // iterations_in_epoch

        command = [
            "python3",
            "train.py",
            "--config", "finetuning-base",
            "--checkpoint_path", str(CHECKPOINT_PATH),
            "--data_root", str(DATASET_ROOT),
            "--train_split_path", str(identity),
            "--batch_size", str(batch_size),
            "--num_epochs", str(num_epochs),
            "--experiments_dir", str(output_dir),
            "--experiment_name", str(experiment_name),
            "--criterions", "adversarial, featmat, idt_embed, perceptual" + ", dice" * ('noSegm' not in MODEL_NAME and MODEL_NAME != "Zakharov"),
        ]

        if MODEL_NAME == "Zakharov":
            command += [
            "--img_dir", "images-cropped-ffhq",
            "--kp_dir", "keypoints-cropped-ffhq",
        ]

        if socket.gethostname() == 'airulsf01':
            # Submit to LSF
            job_name = f"{experiment_name}_{MODEL_NAME}_{ITERATION}"

            if os.getenv('AIRUGPUB') is None or os.getenv('AIRUGPUA') is None:
                exec_hosts = ""
            else:
                exec_hosts = f"{os.getenv('AIRUGPUB')} {os.getenv('AIRUGPUA')}"
            # exec_hosts = " ".join("airugpua%02d" % i for i in (1,3,4,5,6,7,8,9,10,11,12,13)) + " airugpub01 airugpub02 airugpub03"
            command = [
                "bsub", "-J", str(job_name), "-gpu", "num=1:mode=exclusive_process",
                "-o", f"logs/{job_name}.txt", "-m", str(exec_hosts),
            ] + command

        subprocess.run(command)

# Feel totally free to modify any variables with capital names
import subprocess
from pathlib import Path
import os
import socket

def string_to_valid_filename(x):
    return x.replace('/', '_')

MODELS = [
    ("X2Face_vanilla", "00000009"),
    ("X2FacePretrained_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBug_Graphonomy", "01497847"),
    ("ExpressionNet_ResNeXt_3xVGGLossWeight_256_bboxes_noBottleneck_Graphonomy", "01080152"),
    ("FAbNetPretrained_ResNeXt_3xVGGLossWeight_bboxes_augScaleXNoShift_Graphonomy_smallerCrop", "01327623"),
    ("Zakharov", "01529383"),
    ("Zakharov_bboxes_vectorPose_noLandmarks_FineTune7xWeightNewMLP", "01464169"),

    # ("Zakharov_bboxes_vectorPose_noLandmarks", "01363326"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02275845"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShiftReally_noBottleneck", "02023609"),
    # ("X2FacePretrained_ResNeXt_3xVGGLossWeight_256_bboxes_noBug_Graphonomy", "01337493"),
    # ("FAbNetPretrained_ResNeXt_3xVGGLossWeight_bboxes_Graphonomy_smallerCrop", "01222652"),

    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck", "02492466"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_cleanData", "01303150"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02227532"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck", "02444553"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "01361933"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "01613709"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck", "02713859"),

    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_FineTune7xWeight", "02714183"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "02359800"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_64_bboxes_aug_noBottleneck", "02191987"),
    ("MobileNetV2_ResNeXt_7xVGGLossWeight_256_bboxes_SAIC0.02_FromLearned", "02742693"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_cleanData", "01607204"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck", "02737273"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02467652"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck_FineTune7xWeight", "02730111"),
]

for MODEL_NAME, ITERATION in MODELS:
    # The directory where all fine-tuned checkpoints reside
    output_dir = Path(f"puppeteering/{MODEL_NAME}_{ITERATION}")
    assert output_dir.is_dir()

    # Identities to drive
    identities_to_drive = list(output_dir.iterdir()) # change this if you want, e.g.:
    identities_to_drive = [output_dir / string_to_valid_filename(x) for x in [
        "id00061/cAT9aR8oFx0/identity",
        "id00061/Df_m1slf_hY/identity",
        "id00812/XoAi2n4S2wo/identity",
        "id01106/B08yOvYMF7Y/identity",
        "id01228/7qHTvs0VO68/identity",
        "id01333/9kgJaduwKkY/identity",
        "id01437/4lFDvxXzYWY/identity",
        "id02057/s5VqJY7DDEE/identity",
        "id02548/x2LUQEUXdz4/identity",
        "id03127/uiRiyK8Qlic/identity",
        "id03178/cCoNRuzAL-A/identity",
        "id03178/fnARFfUwf2s/identity",
        "id03524/GkvScYvOJ7o/identity",
        "id03839/LhI_8AWX_Mg/identity",
        "id03839/PUwanP-C5qg/identity",
        "id03862/fsCqKQb9Rdg/identity",
        "id04094/JUYMzfVp8zI/identity",
        "id04950/PQEAck-3wcA/identity",
        "id05459/3TI6dVmEwzw/identity",
        "id05714/wFGNufaMbDY/identity",
        "id06104/7UnGAS5-jpU/identity",
        "id06811/KmvEwL3fP9Q/identity",
        "id07312/h1dszoDi1E8/identity",
        "id07663/54qlJ2HZ08s/identity",
        "id07802/BfQUBDw7TiM/identity",
        "id07868/JC0QT4oXh2Y/identity",
        "id07961/464OHFffwjI/identity",
        "id07961/hROZwL8pbGg/identity",
        "id08149/vxBFGKGXSFA/identity",
        "id08701/UeUyLqpLz70/identity",
    ]]

    # Drivers
    DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/VoxCeleb2_30TestIdentities")
    # DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/Custom avatar drivers")
    # DATASET_ROOT = Path("/Vol0/user/e.burkov/Datasets/violet/VoxCeleb2_test_finetuning")

    IMAGES_DIR = DATASET_ROOT / ("images-cropped-ffhq" if MODEL_NAME == "Zakharov" else "images-cropped")
    DRIVERS = [
        "id00061/cAT9aR8oFx0/driver",
        "id00061/Df_m1slf_hY/driver",
        "id00812/XoAi2n4S2wo/driver",
        "id01106/B08yOvYMF7Y/driver",
        "id01228/7qHTvs0VO68/driver",
        "id01333/9kgJaduwKkY/driver",
        "id01437/4lFDvxXzYWY/driver",
        "id02057/s5VqJY7DDEE/driver",
        "id02548/x2LUQEUXdz4/driver",
        "id03127/uiRiyK8Qlic/driver",
        "id03178/cCoNRuzAL-A/driver",
        "id03178/fnARFfUwf2s/driver",
        "id03524/GkvScYvOJ7o/driver",
        "id03839/LhI_8AWX_Mg/driver",
        "id03839/PUwanP-C5qg/driver",
        "id03862/fsCqKQb9Rdg/driver",
        "id04094/JUYMzfVp8zI/driver",
        "id04950/PQEAck-3wcA/driver",
        "id05459/3TI6dVmEwzw/driver",
        "id05714/wFGNufaMbDY/driver",
        "id06104/7UnGAS5-jpU/driver",
        "id06811/KmvEwL3fP9Q/driver",
        "id07312/h1dszoDi1E8/driver",
        "id07663/54qlJ2HZ08s/driver",
        "id07802/BfQUBDw7TiM/driver",
        "id07868/JC0QT4oXh2Y/driver",
        "id07961/464OHFffwjI/driver",
        "id07961/hROZwL8pbGg/driver",
        "id08149/vxBFGKGXSFA/driver",
        "id08701/UeUyLqpLz70/driver",
    ]

    for identity_to_drive in identities_to_drive:
        # Get fine-tuned checkpoint
        checkpoint_path = identity_to_drive / "checkpoints"
        assert checkpoint_path.is_dir()
        all_checkpoints = sorted(checkpoint_path.iterdir())
        if len(all_checkpoints) > 1:
            print(
                f"WARNING: there are {len(all_checkpoints)} checkpoints in" \
                f"{checkpoint_path}, using the latest one ({all_checkpoints[-1]})")
        checkpoint_path = all_checkpoints[-1]

        command = [
            "python3",
            "drive.py",
            str(checkpoint_path),
            str(DATASET_ROOT),
            "--destination", str(identity_to_drive / "driving-results"),
            "--images_paths"] + DRIVERS

        if socket.gethostname() == 'airulsf01':
            # Submit to LSF
            job_name = f"driving_{identity_to_drive.name}_{MODEL_NAME}_{ITERATION}"

            if os.getenv('AIRUGPUB') is None or os.getenv('AIRUGPUA') is None:
                exec_hosts = ""
            else:
                exec_hosts = f"{os.getenv('AIRUGPUB')} {os.getenv('AIRUGPUA')}"
            # exec_hosts = "airugpub01 airugpub02 airugpub03 " + " ".join("airugpua%02d" % i for i in (1,3,4,5,6,7,8,9,10,11,12,13))
            command = [
                "bsub", "-J", str(job_name), "-gpu", "num=1:mode=exclusive_process",
                "-o", f"logs/{job_name}.txt", "-m", str(exec_hosts),
            ] + command

        subprocess.run(command)

