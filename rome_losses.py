import torch
from torch import nn
from torch.nn import functional as F

from typing import List



class AdversarialLoss(nn.Module):
    def __init__(self, loss_type = 'hinge'):
        super(AdversarialLoss, self).__init__()
        # TODO: different adversarial loss types
        self.loss_type = loss_type

    def forward(self, 
                fake_scores: List[List[torch.Tensor]], 
                real_scores: List[List[torch.Tensor]] = None, 
                mode: str = 'gen') -> torch.Tensor:
        """
        scores: a list of lists of scores (the second layer corresponds to a
                separate input to each of these discriminators)
        """
        loss = 0

        if mode == 'dis':
            for real_scores_net, fake_scores_net in zip(real_scores, fake_scores):
                # *_scores_net corresponds to outputs of a separate discriminator
                loss_real = 0
                
                for real_scores_net_i in real_scores_net:
                    if self.loss_type == 'hinge':
                        loss_real += torch.relu(1.0 - real_scores_net_i).mean()
                    else:
                        raise # not implemented
                
                loss_real /= len(real_scores_net)

                loss_fake = 0
                
                for fake_scores_net_i in fake_scores_net:
                    if self.loss_type == 'hinge':
                        loss_fake += torch.relu(1.0 + fake_scores_net_i).mean()
                    else:
                        raise # not implemented
                
                loss_fake /= len(fake_scores_net)

                loss_net = loss_real + loss_fake
                loss += loss_net

        elif mode == 'gen':
            for fake_scores_net in fake_scores:
                assert isinstance(fake_scores_net, list), 'Expect a list of fake scores per discriminator'

                loss_net = 0

                for fake_scores_net_i in fake_scores_net:
                    if self.loss_type == 'hinge':
                        # *_scores_net_i corresponds to outputs for separate inputs
                        loss_net -= fake_scores_net_i.mean()

                    else:
                        raise # not implemented

                loss_net /= len(fake_scores_net) # normalize by the number of inputs
                loss += loss_net
        
        loss /= len(fake_scores) # normalize by the nubmer of discriminators

        return loss
import torch



class PSNR(object):
    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        mse = ((y_pred - y_true) ** 2).mean()
        return 10 * torch.log10(1 / mse)
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
# try:
#     from pytorch3d.loss.mesh_laplacian_smoothing import cot_laplacian
# except:
#     from pytorch3d.loss.mesh_laplacian_smoothing import laplacian_cot as cot_laplacian


def make_grid(h, w, device, dtype):
    grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    v, u = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([u, v], dim=2).view(1, h * w, 2)

    return grid


class Transform(nn.Module):
    def __init__(self, sigma_affine, sigma_tps, points_tps):
        super(Transform, self).__init__()
        self.sigma_affine = sigma_affine
        self.sigma_tps = sigma_tps
        self.points_tps = points_tps

    def transform_img(self, img):
        b, _, h, w = img.shape
        device = img.device
        dtype = img.dtype

        if not hasattr(self, 'identity_grid'):
            identity_grid = make_grid(h, w, device, dtype)
            self.register_buffer('identity_grid', identity_grid, persistent=False)

        if not hasattr(self, 'control_grid'):
            control_grid = make_grid(self.points_tps, self.points_tps, device, dtype)
            self.register_buffer('control_grid', control_grid, persistent=False)

        # Sample transform
        noise = torch.normal(
            mean=0,
            std=self.sigma_affine,
            size=(b, 2, 3),
            device=device,
            dtype=dtype)

        self.theta = (noise + torch.eye(2, 3, device=device, dtype=dtype)[None])[:, None]  # b x 1 x 2 x 3

        self.control_params = torch.normal(
            mean=0,
            std=self.sigma_tps,
            size=(b, 1, self.points_tps ** 2),
            device=device,
            dtype=dtype)

        grid = self.warp_pts(self.identity_grid).view(-1, h, w, 2)

        return F.grid_sample(img, grid, padding_mode="reflection")

    def warp_pts(self, pts):
        b = self.theta.shape[0]
        n = pts.shape[1]
 
        pts_transformed = torch.matmul(self.theta[:, :, :, :2], pts[..., None]) + self.theta[:, :, :, 2:]
        pts_transformed = pts_transformed[..., 0]

        pdists = pts[:, :, None] - self.control_grid[:, None]
        pdists = (pdists).abs().sum(dim=3)

        result = pdists**2 * torch.log(pdists + 1e-5) * self.control_params
        result = result.sum(dim=2).view(b, n, 1)

        pts_transformed = pts_transformed + result

        return pts_transformed

    def jacobian(self, pts):
        new_pts = self.warp_pts(pts)
        grad_x = grad(new_pts[..., 0].sum(), pts, create_graph=True)
        grad_y = grad(new_pts[..., 1].sum(), pts, create_graph=True)
        jac = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        
        return jac


class EquivarianceLoss(nn.Module):
    def __init__(self, sigma_affine, sigma_tps, points_tps):
        super(EquivarianceLoss, self).__init__()
        self.transform = Transform(sigma_affine, sigma_tps, points_tps)

    def forward(self, img, kp, jac, kp_detector):
        img_transformed = self.transform.transform_img(img)
        kp_transformed, jac_transformed = kp_detector(img_transformed)
        kp_recon = self.transform.warp_pts(kp_transformed)

        loss_kp = (kp - kp_recon).abs().mean()

        jac_recon = torch.matmul(self.transform.jacobian(kp_transformed), jac_transformed)
        inv_jac = torch.linalg.inv(jac)

        loss_jac = (torch.matmul(inv_jac, jac_recon) - torch.eye(2)[None, None].type(inv_jac.type())).abs().mean()

        return loss_kp, loss_jac, img_transformed, kp_transformed, kp_recon


class LaplaceMeshLoss(nn.Module):
    def __init__(self, type='uniform', use_vector_constant=False):
        super(LaplaceMeshLoss, self).__init__()
        self.method = type
        self.precomputed_laplacian = None
        self.use_vector_constant = use_vector_constant

    def _compute_loss(self, L, verts_packed, inv_areas=None):
        if self.method == "uniform":
            loss = L.mm(verts_packed)
        
        elif self.method == "cot":
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            idx = norm_w > 0
            norm_w[idx] = 1.0 / norm_w[idx]
            loss = L.mm(verts_packed) * norm_w - verts_packed
        
        elif self.method == "cotcurv":
            L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            norm_w = 0.25 * inv_areas
            loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
        
        return loss.norm(dim=1)

    def forward(self, meshes, coefs=None):
        if meshes.isempty():
            return torch.tensor(
                [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
            )

        N = len(meshes)
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
        num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
        verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
        weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
        weights = 1.0 / weights.float()
        norm_w, inv_areas = None, None
        with torch.no_grad():
            if self.method == "uniform":
                if self.precomputed_laplacian is None or self.precomputed_laplacian.shape[0] != verts_packed.shape[0]:
                    L = meshes.laplacian_packed()
                    self.precomputed_laplacian = L
                else:
                    L = self.precomputed_laplacian
            elif self.method in ["cot", "cotcurv"]:
                L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            else:
                raise ValueError("Method should be one of {uniform, cot, cotcurv}")

        loss = self._compute_loss(L, verts_packed,
                                  inv_areas=inv_areas)
        loss = loss * weights
        if coefs is not None:
            loss = loss * coefs.view(-1)

        return loss.sum() / N
import torch
from torch import nn
import torch.nn.functional as F

from typing import List



class FeatureMatchingLoss(nn.Module):
    def __init__(self, loss_type = 'l1', ):
        super(FeatureMatchingLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, 
                real_features: List[List[List[torch.Tensor]]], 
                fake_features: List[List[List[torch.Tensor]]]
        ) -> torch.Tensor:
        """
        features: a list of features of different inputs (the third layer corresponds to
                  features of a separate input to each of these discriminators)
        """
        loss = 0

        for real_feats_net, fake_feats_net in zip(real_features, fake_features):
            # *_feats_net corresponds to outputs of a separate discriminator
            loss_net = 0

            for real_feats_layer, fake_feats_layer in zip(real_feats_net, fake_feats_net):
                assert len(real_feats_layer) == 1 or len(real_feats_layer) == len(fake_feats_layer), 'Wrong number of real inputs'
                if len(real_feats_layer) == 1:
                    real_feats_layer = [real_feats_layer[0]] * len(fake_feats_layer)

                for real_feats_layer_i, fake_feats_layer_i in zip(real_feats_layer, fake_feats_layer):
                    if self.loss_type == 'l1':
                        loss_net += F.l1_loss(fake_feats_layer_i, real_feats_layer_i)
                    elif self.loss_type == 'l2':
                        loss_net += F.mse_loss(fake_feats_layer_i, real_feats_layer_i)

            loss_net /= len(fake_feats_layer) # normalize by the number of inputs
            loss_net /= len(fake_feats_net) # normalize by the number of layers
            loss += loss_net

        loss /= len(real_features) # normalize by the number of networks

        return loss
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class KeypointsMatchingLoss(nn.Module):
    def __init__(self):
        super(KeypointsMatchingLoss, self).__init__()
        self.register_buffer('weights', torch.ones(68), persistent=False)
        self.weights[5:7] = 2.0
        self.weights[10:12] = 2.0
        self.weights[27:36] = 1.5
        self.weights[30] = 3.0
        self.weights[31] = 3.0
        self.weights[35] = 3.0
        self.weights[60:68] = 1.5
        self.weights[48:60] = 1.5
        self.weights[48] = 3
        self.weights[54] = 3

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff = pred_keypoints - keypoints

        loss = (diff.abs().mean(-1) * self.weights[None] / self.weights.sum()).sum(-1).mean()

        return loss
import torch
from torch import nn
import lpips



class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.metric = lpips.LPIPS(net='alex')

        for m in self.metric.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

            names = [name for name, _ in m.named_buffers()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    @torch.no_grad()
    def __call__(self, inputs, targets):
        return self.metric(inputs, targets, normalize=True).mean()

    def train(self, mode: bool = True):
        return self
# from .adversarial import AdversarialLoss
# from .feature_matching import FeatureMatchingLoss
# from .keypoints_matching import KeypointsMatchingLoss
# from .eye_closure import EyeClosureLoss
# from .lip_closure import LipClosureLoss
# from .head_pose_matching import HeadPoseMatchingLoss
# from .perceptual import PerceptualLoss

# from .segmentation import SegmentationLoss, MultiScaleSilhouetteLoss
# from .chamfer_silhouette import ChamferSilhouetteLoss
# from .equivariance import EquivarianceLoss, LaplaceMeshLoss
# from .vgg2face import VGGFace2Loss
# from .gaze import GazeLoss

# from .psnr import PSNR
# from .lpips import LPIPS
from pytorch_msssim import SSIM, MS_SSIM
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from typing import Union

# from src.utils import misc

def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.
    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [0, 1].
    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input using the ImageNet mean and std
    mean = input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (input - mean) / std
    return output


class PerceptualLoss(nn.Module):
    r"""Perceptual loss initialization.
    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the inputsut images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
        use_fp16 (bool) : If ``True``, use cast networks and inputs to FP16
    """

    def __init__(
            self, 
            network='vgg19', 
            layers=('relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'), 
            weights=(0.03125, 0.0625, 0.125, 0.25, 1.0),
            criterion='l1', 
            resize=False, 
            resize_mode='bilinear',
            instance_normalized=False,
            replace_maxpool_with_avgpool=False,
            num_scales=1,
            use_fp16=False
        ) -> None:
        super(PerceptualLoss, self).__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        elif network == 'alexnet':
            self.model = _alexnet(layers)
        elif network == 'inception_v3':
            self.model = _inception_v3(layers)
        elif network == 'resnet50':
            self.model = _resnet50(layers)
        elif network == 'robust_resnet50':
            self.model = _robust_resnet50(layers)
        elif network == 'vgg_face_dag':
            self.model = _vgg_face_dag(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        if replace_maxpool_with_avgpool:
	        for k, v in self.model.network._modules.items():
	        	if isinstance(v, nn.MaxPool2d):
	        		self.model.network._modules[k] = nn.AvgPool2d(2)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSEloss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized
        self.fp16 = use_fp16
        if self.fp16:
            self.model.half()

    @torch.cuda.amp.autocast(True)
    def forward(self, 
                inputs: Union[torch.Tensor, list], 
                target: torch.Tensor) -> Union[torch.Tensor, list]:
        r"""Perceptual loss forward.
        Args:
           inputs (4D tensor or list of 4D tensors) : inputsut tensor.
           target (4D tensor) : Ground truth tensor, same shape as the inputsut.
        Returns:
           (scalar tensor or list of tensors) : The perceptual loss.
        """
        if isinstance(inputs, list):
            # Concat alongside the batch axis
            input_is_a_list = True
            num_chunks = len(inputs)
            inputs = torch.cat(inputs)
        else:
            input_is_a_list = False

        # Perceptual loss should operate in eval mode by default.
        self.model.eval()
        inputs, target = \
            apply_imagenet_normalization(inputs), \
            apply_imagenet_normalization(target)
        if self.resize:
            inputs = F.interpolate(
                inputs, mode=self.resize_mode, size=(224, 224),
                align_corners=False)
            target = F.interpolate(
                target, mode=self.resize_mode, size=(224, 224),
                align_corners=False)

        # Evaluate perceptual loss at each scale.
        loss = 0

        for scale in range(self.num_scales):
            if self.fp16:
                input_features = self.model(inputs.half())
                with torch.no_grad():
                    target_features = self.model(target.half())
            else:
                input_features = self.model(inputs)
                with torch.no_grad():
                    target_features = self.model(target)

            for layer, weight in zip(self.layers, self.weights):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)

                if input_is_a_list:
                    target_feature = torch.cat([target_feature] * num_chunks)

                loss += weight * self.criterion(input_feature,
                                                target_feature)

            # Downsample the inputsut and target.
            if scale != self.num_scales - 1:
                inputs = F.interpolate(
                    inputs, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        loss /= self.num_scales

        return loss

    def train(self, mode: bool = True):
        return self


class _PerceptualNetwork(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.
    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), \
            'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers

        for m in self.network.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output


def _vgg19(layers):
    r"""Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg16(layers):
    r"""Get vgg16 layers"""
    network = torchvision.models.vgg16(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          18: 'relu_4_1',
                          20: 'relu_4_2',
                          22: 'relu_4_3',
                          25: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _alexnet(layers):
    r"""Get alexnet layers"""
    network = torchvision.models.alexnet(pretrained=True).features
    layer_name_mapping = {0: 'conv_1',
                          1: 'relu_1',
                          3: 'conv_2',
                          4: 'relu_2',
                          6: 'conv_3',
                          7: 'relu_3',
                          8: 'conv_4',
                          9: 'relu_4',
                          10: 'conv_5',
                          11: 'relu_5'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _inception_v3(layers):
    r"""Get inception v3 layers"""
    inception = torchvision.models.inception_v3(pretrained=True)
    network = nn.Sequential(inception.Conv2d_1a_3x3,
                            inception.Conv2d_2a_3x3,
                            inception.Conv2d_2b_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Conv2d_3b_1x1,
                            inception.Conv2d_4a_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Mixed_5b,
                            inception.Mixed_5c,
                            inception.Mixed_5d,
                            inception.Mixed_6a,
                            inception.Mixed_6b,
                            inception.Mixed_6c,
                            inception.Mixed_6d,
                            inception.Mixed_6e,
                            inception.Mixed_7a,
                            inception.Mixed_7b,
                            inception.Mixed_7c,
                            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    layer_name_mapping = {3: 'pool_1',
                          6: 'pool_2',
                          14: 'mixed_6e',
                          18: 'pool_3'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _resnet50(layers):
    r"""Get resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=True)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _robust_resnet50(layers):
    r"""Get robust resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.utils.model_zoo.load_url(
        'http://andrewilyas.com/ImageNet.pt')
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        if k.startswith('module.model.'):
            new_state_dict[k[13:]] = v
    resnet50.load_state_dict(new_state_dict)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face_dag(layers):
    r"""Get vgg face layers"""
    network = torchvision.models.vgg16(num_classes=2622).features
    state_dict = torch.utils.model_zoo.load_url(
        'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/'
        'vgg_face_dag.pth')
    layer_name_mapping = {
        0: 'conv1_1',
        2: 'conv1_2',
        5: 'conv2_1',
        7: 'conv2_2',
        10: 'conv3_1',
        12: 'conv3_2',
        14: 'conv3_3',
        17: 'conv4_1',
        19: 'conv4_2',
        21: 'conv4_3',
        24: 'conv5_1',
        26: 'conv5_2',
        28: 'conv5_3'}
    new_state_dict = {}
    for k, v in layer_name_mapping.items():
        new_state_dict[str(k) + '.weight'] =\
            state_dict[v + '.weight']
        new_state_dict[str(k) + '.bias'] = \
            state_dict[v + '.bias']

    return _PerceptualNetwork(network, layer_name_mapping, layers)
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
import math


import torch
import torch.nn as nn


class Resnet50_scratch_dag(nn.Module):

    def __init__(self):
        super(Resnet50_scratch_dag, self).__init__()
        self.meta = {'mean': [131.0912, 103.8827, 91.4953],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_relu_7x7_s2 = nn.ReLU()
        self.pool1_3x3_s2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        self.conv2_1_1x1_reduce = nn.Conv2d(64, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_1x1_reduce_relu = nn.ReLU()
        self.conv2_1_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_1_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_3x3_relu = nn.ReLU()
        self.conv2_1_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_1x1_proj = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_proj_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_relu = nn.ReLU()
        self.conv2_2_1x1_reduce = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_2_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_1x1_reduce_relu = nn.ReLU()
        self.conv2_2_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_2_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_3x3_relu = nn.ReLU()
        self.conv2_2_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_2_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_relu = nn.ReLU()
        self.conv2_3_1x1_reduce = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_3_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_1x1_reduce_relu = nn.ReLU()
        self.conv2_3_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_3_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_3x3_relu = nn.ReLU()
        self.conv2_3_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_3_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_relu = nn.ReLU()
        self.conv3_1_1x1_reduce = nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv3_1_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_1x1_reduce_relu = nn.ReLU()
        self.conv3_1_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_1_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_3x3_relu = nn.ReLU()
        self.conv3_1_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_1_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_1x1_proj = nn.Conv2d(256, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv3_1_1x1_proj_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_relu = nn.ReLU()
        self.conv3_2_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_2_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_1x1_reduce_relu = nn.ReLU()
        self.conv3_2_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_2_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_3x3_relu = nn.ReLU()
        self.conv3_2_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_2_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_relu = nn.ReLU()
        self.conv3_3_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_3_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_1x1_reduce_relu = nn.ReLU()
        self.conv3_3_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_3_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_3x3_relu = nn.ReLU()
        self.conv3_3_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_3_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_relu = nn.ReLU()
        self.conv3_4_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_4_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_1x1_reduce_relu = nn.ReLU()
        self.conv3_4_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_4_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_3x3_relu = nn.ReLU()
        self.conv3_4_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_4_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_relu = nn.ReLU()
        self.conv4_1_1x1_reduce = nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv4_1_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_1x1_reduce_relu = nn.ReLU()
        self.conv4_1_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_1_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_3x3_relu = nn.ReLU()
        self.conv4_1_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_1_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_1x1_proj = nn.Conv2d(512, 1024, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv4_1_1x1_proj_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_relu = nn.ReLU()
        self.conv4_2_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_2_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_1x1_reduce_relu = nn.ReLU()
        self.conv4_2_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_2_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_3x3_relu = nn.ReLU()
        self.conv4_2_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_2_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_relu = nn.ReLU()
        self.conv4_3_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_3_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_1x1_reduce_relu = nn.ReLU()
        self.conv4_3_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_3_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_3x3_relu = nn.ReLU()
        self.conv4_3_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_3_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_relu = nn.ReLU()
        self.conv4_4_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_4_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_1x1_reduce_relu = nn.ReLU()
        self.conv4_4_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_4_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_3x3_relu = nn.ReLU()
        self.conv4_4_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_4_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_relu = nn.ReLU()
        self.conv4_5_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_5_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_1x1_reduce_relu = nn.ReLU()
        self.conv4_5_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_5_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_3x3_relu = nn.ReLU()
        self.conv4_5_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_5_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_relu = nn.ReLU()
        self.conv4_6_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_6_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_1x1_reduce_relu = nn.ReLU()
        self.conv4_6_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_6_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_3x3_relu = nn.ReLU()
        self.conv4_6_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_6_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_relu = nn.ReLU()
        self.conv5_1_1x1_reduce = nn.Conv2d(1024, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv5_1_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_1x1_reduce_relu = nn.ReLU()
        self.conv5_1_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_1_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_3x3_relu = nn.ReLU()
        self.conv5_1_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_1_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_1x1_proj = nn.Conv2d(1024, 2048, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv5_1_1x1_proj_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_relu = nn.ReLU()
        self.conv5_2_1x1_reduce = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_2_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_1x1_reduce_relu = nn.ReLU()
        self.conv5_2_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_2_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_3x3_relu = nn.ReLU()
        self.conv5_2_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_2_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_relu = nn.ReLU()
        self.conv5_3_1x1_reduce = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_3_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_1x1_reduce_relu = nn.ReLU()
        self.conv5_3_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_3_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_3x3_relu = nn.ReLU()
        self.conv5_3_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_3_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_relu = nn.ReLU()
        self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.classifier = nn.Conv2d(2048, 8631, kernel_size=[1, 1], stride=(1, 1))

    def forward(self, data):
        conv1_7x7_s2 = self.conv1_7x7_s2(data)
        conv1_7x7_s2_bn = self.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.conv1_relu_7x7_s2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_7x7_s2_bnxx)
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, conv2_1_1x1_increase_bn)
        conv2_1x = self.conv2_1_relu(conv2_1)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        conv2_2 = torch.add(conv2_1x, 1, conv2_2_1x1_increase_bn)
        conv2_2x = self.conv2_2_relu(conv2_2)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        conv2_3 = torch.add(conv2_2x, 1, conv2_3_1x1_increase_bn)
        conv2_3x = self.conv2_3_relu(conv2_3)
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, conv3_1_1x1_increase_bn)
        conv3_1x = self.conv3_1_relu(conv3_1)
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        conv3_2 = torch.add(conv3_1x, 1, conv3_2_1x1_increase_bn)
        conv3_2x = self.conv3_2_relu(conv3_2)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        conv3_3 = torch.add(conv3_2x, 1, conv3_3_1x1_increase_bn)
        conv3_3x = self.conv3_3_relu(conv3_3)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        conv3_4 = torch.add(conv3_3x, 1, conv3_4_1x1_increase_bn)
        conv3_4x = self.conv3_4_relu(conv3_4)
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, conv4_1_1x1_increase_bn)
        conv4_1x = self.conv4_1_relu(conv4_1)
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        conv4_2 = torch.add(conv4_1x, 1, conv4_2_1x1_increase_bn)
        conv4_2x = self.conv4_2_relu(conv4_2)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        conv4_3 = torch.add(conv4_2x, 1, conv4_3_1x1_increase_bn)
        conv4_3x = self.conv4_3_relu(conv4_3)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        conv4_4 = torch.add(conv4_3x, 1, conv4_4_1x1_increase_bn)
        conv4_4x = self.conv4_4_relu(conv4_4)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        conv4_5 = torch.add(conv4_4x, 1, conv4_5_1x1_increase_bn)
        conv4_5x = self.conv4_5_relu(conv4_5)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        conv4_6 = torch.add(conv4_5x, 1, conv4_6_1x1_increase_bn)
        conv4_6x = self.conv4_6_relu(conv4_6)
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, conv5_1_1x1_increase_bn)
        conv5_1x = self.conv5_1_relu(conv5_1)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        conv5_2 = torch.add(conv5_1x, 1, conv5_2_1x1_increase_bn)
        conv5_2x = self.conv5_2_relu(conv5_2)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        conv5_3 = torch.add(conv5_2x, 1, conv5_3_1x1_increase_bn)
        conv5_3x = self.conv5_3_relu(conv5_3)
        pool5_7x7_s1 = self.pool5_7x7_s1(conv5_3x)
        classifier_preflatten = self.classifier(pool5_7x7_s1)
        classifier = classifier_preflatten.view(classifier_preflatten.size(0), -1)
        return classifier, pool5_7x7_s1

def resnet50_scratch_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Resnet50_scratch_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

class VGGFace2Loss(object):
    def __init__(self, pretrained_model, pretrained_data='vggface2', device='cuda'):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50_scratch_dag(pretrained_model).eval().cuda()
        # self.reg_model.load_state_dict(torch.load(pretrained_model), strict=False)
        # self.reg_model = self.reg_model.eval().cuda()
        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).cuda()
        self.mean_rgb = torch.tensor((131.0912, 103.8827, 91.4953)).cuda()

    def reg_features(self, x):
        # out = []
        margin = 10
        x = x[:, :, margin:224 - margin, margin:224 - margin]
        x = F.interpolate(x * 2. - 1., [224, 224], mode='bilinear')
        feature = self.reg_model(x)[1]
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # import ipdb;ipdb.set_trace()
        img = img[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) * 255 - self.mean_rgb
        img = img.permute(0, 3, 1, 2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        # loss = ((gen_out - tar_out)**2).mean()
        loss = self._cos_metric(gen_out, tar_out).mean()
        return loss

import torch
import torch.nn.functional as F
from torch import nn
from typing import Union

import torch
import torch.nn.functional as F
# from pytorch3d.ops.knn import knn_gather, knn_points
# from pytorch3d.structures.pointclouds import Pointclouds

from typing import Union



# class ChamferSilhouetteLoss(nn.Module):
#     def __init__(
#         self, 
#         num_neighbours=1, 
#         use_same_number_of_points=False, 
#         sample_outside_of_silhouette=False,
#         use_visibility=True
#     ):
#         super(ChamferSilhouetteLoss, self).__init__()
#         self.num_neighbours = num_neighbours
#         self.use_same_number_of_points = use_same_number_of_points
#         self.sample_outside_of_silhouette = sample_outside_of_silhouette
#         self.use_visibility = use_visibility

#     def forward(self, 
#                 pred_points: torch.Tensor,
#                 points_visibility: torch.Tensor,
#                 target_silhouette: torch.Tensor,
#                 target_segs: torch.Tensor) -> torch.Tensor:        
#         target_points, target_lengths, weight = self.get_pointcloud(target_segs, target_silhouette)

#         if self.use_visibility:
#             pred_points, pred_lengths = self.get_visible_points(pred_points, points_visibility)
                
#         if self.use_same_number_of_points:
#             target_points = target_points[:, :pred_points.shape[1]]    

#             target_lengths = pred_lengths = torch.minimum(target_lengths, pred_lengths)
            
#             if self.sample_outside_of_silhouette:
#                 target_lengths = (target_lengths.clone() * weight).long()

#             for i in range(target_points.shape[0]):
#                 target_points[i, target_lengths[i]:] = -100.0

#             for i in range(pred_points.shape[0]):
#                 pred_points[i, pred_lengths[i]:] = -100.0

#         visible_batch = target_lengths > 0
#         if self.use_visibility:
#             visible_batch *= pred_lengths > 0

#         if self.use_visibility:
#             loss = chamfer_distance(
#                 pred_points[visible_batch], 
#                 target_points[visible_batch], 
#                 x_lengths=pred_lengths[visible_batch], 
#                 y_lengths=target_lengths[visible_batch],
#                 num_neighbours=self.num_neighbours
#             )        
#         else:
#             loss = chamfer_distance(
#                 pred_points[visible_batch], 
#                 target_points[visible_batch], 
#                 y_lengths=target_lengths[visible_batch],
#                 num_neighbours=self.num_neighbours
#             )

#         if isinstance(loss, tuple):
#             loss = loss[0]
        
#         return loss, pred_points, target_points
    
#     @torch.no_grad()
#     def get_pointcloud(self, seg, silhouette):
#         if self.sample_outside_of_silhouette:
#             silhouette = (silhouette > 0.0).type(seg.type())

#             old_area = seg.view(seg.shape[0], -1).sum(1)
#             seg = seg * (1 - silhouette)
#             new_area = seg.view(seg.shape[0], -1).sum(1)

#             weight = new_area / (old_area + 1e-7)
        
#         else:
#             weight = torch.ones(seg.shape[0], dtype=seg.dtype, device=seg.device)

#         batch, coords = torch.nonzero(seg[:, 0] > 0.5).split([1, 2], dim=1)
#         batch = batch[:, 0]
#         coords = coords.float()
#         coords[:, 0] = (coords[:, 0] / seg.shape[2] - 0.5) * 2
#         coords[:, 1] = (coords[:, 1] / seg.shape[3] - 0.5) * 2

#         pointcloud = -100.0 * torch.ones(seg.shape[0], seg.shape[2]*seg.shape[3], 2).to(seg.device)
#         length = torch.zeros(seg.shape[0]).to(seg.device).long()
#         for i in range(seg.shape[0]):
#             pt = coords[batch == i]
#             pt = pt[torch.randperm(pt.shape[0])] # randomly permute the points
#             pointcloud[i][:pt.shape[0]] = torch.cat([pt[:, 1:], pt[:, :1]], dim=1)
#             length[i] = pt.shape[0]
        
#         return pointcloud, length, weight
    
#     @staticmethod
#     def get_visible_points(points, visibility):
#         batch, indices = torch.nonzero(visibility > 0.0).split([1, 1], dim=1)
#         batch = batch[:, 0]
#         indices = indices[:, 0]

#         length = torch.zeros(points.shape[0]).to(points.device).long()
#         for i in range(points.shape[0]):
#             batch_i = batch == i
#             indices_i = indices[batch_i]
#             points[i][:indices_i.shape[0]] = points[i][indices_i]
#             points[i][indices_i.shape[0]:] = -100.0
#             length[i] = indices_i.shape[0]

#         return points, length


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# def _validate_chamfer_reduction_inputs(
#     batch_reduction: Union[str, None], point_reduction: str
# ):
#     """Check the requested reductions are valid.
#     Args:
#         batch_reduction: Reduction operation to apply for the loss across the
#             batch, can be one of ["mean", "sum"] or None.
#         point_reduction: Reduction operation to apply for the loss across the
#             points, can be one of ["mean", "sum"].
#     """
#     if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
#         raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
#     if point_reduction not in ["mean", "sum"]:
#         raise ValueError('point_reduction must be one of ["mean", "sum"]')


# def _handle_pointcloud_input(
#     points: Union[torch.Tensor, Pointclouds],
#     lengths: Union[torch.Tensor, None],
#     normals: Union[torch.Tensor, None],
# ):
#     """
#     If points is an instance of Pointclouds, retrieve the padded points tensor
#     along with the number of points per batch and the padded normals.
#     Otherwise, return the input points (and normals) with the number of points per cloud
#     set to the size of the second dimension of `points`.
#     """
#     if isinstance(points, Pointclouds):
#         X = points.points_padded()
#         lengths = points.num_points_per_cloud()
#         normals = points.normals_padded()  # either a tensor or None
#     elif torch.is_tensor(points):
#         if points.ndim != 3:
#             raise ValueError("Expected points to be of shape (N, P, D)")
#         X = points
#         if lengths is not None and (
#             lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
#         ):
#             raise ValueError("Expected lengths to be of shape (N,)")
#         if lengths is None:
#             lengths = torch.full(
#                 (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
#             )
#         if normals is not None and normals.ndim != 3:
#             raise ValueError("Expected normals to be of shape (N, P, 3")
#     else:
#         raise ValueError(
#             "The input pointclouds should be either "
#             + "Pointclouds objects or torch.Tensor of shape "
#             + "(minibatch, num_points, 3)."
#         )
#     return X, lengths, normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    num_neighbours=1,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.
    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    Returns:
        2-element tuple containing
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=num_neighbours)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=num_neighbours)

    cham_x = x_nn.dists.mean(-1)  # (N, P1)
    cham_y = y_nn.dists.mean(-1)  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals
import torch
import torch.nn.functional as F
from torch import nn

from typing import Union



class SegmentationLoss(nn.Module):
    def __init__(self, loss_type = 'bce_with_logits'):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, 
                pred_seg_logits: Union[torch.Tensor, list], 
                target_segs: Union[torch.Tensor, list]) -> torch.Tensor:
        if isinstance(pred_seg_logits, list):
            # Concat alongside the batch axis
            pred_seg_logits = torch.cat(pred_seg_logits)
            target_segs = torch.cat(target_segs)

        if target_segs.shape[2] != pred_seg_logits.shape[2]:
            target_segs = F.interpolate(target_segs, size=pred_seg_logits.shape[2:], mode='bilinear')

        if self.loss_type == 'bce_with_logits':
            loss = self.criterion(pred_seg_logits, target_segs)
        
        elif self.loss_type == 'dice':
            pred_segs = torch.sigmoid(pred_seg_logits)

            intersection = (pred_segs * target_segs).view(pred_segs.shape[0], -1)
            cardinality = (pred_segs**2 + target_segs**2).view(pred_segs.shape[0], -1)
            loss = 1 - ((2. * intersection.mean(1)) / (cardinality.mean(1) + 1e-7)).mean(0)

        return loss


class MultiScaleSilhouetteLoss(nn.Module):
    def __init__(self, num_scales: int = 1, loss_type: str = 'bce'):
        super().__init__()
        self.num_scales = num_scales
        self.loss_type = loss_type
        if self.loss_type == 'bce':
            self.loss = nn.BCELoss()
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        original_size = targets.size()[-1]
        loss = 0.0
        for i in range(self.num_scales):
            if i > 0:
                x = F.interpolate(inputs, size=original_size // (2 ** i))
                gt = F.interpolate(targets, size=original_size // (2 ** i))
            else:
                x = inputs
                gt = targets
            
            if self.loss_type == 'iou':
                intersection = (x * gt).view(x.shape[0], -1)
                union = (x + gt).view(x.shape[0], -1)
                loss += 1 - (intersection.mean(1) / (union - intersection).mean(1)).mean(0)
            
            elif self.loss_type == 'mse':
                loss += ((x - gt)**2).mean() * 0.5

            elif self.loss_type == 'bce':
                loss += self.loss(x, gt.float())
            elif self.loss_type == 'mse':
                loss += self.loss(x, gt.float())
        return loss / self.num_scales
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class LipClosureLoss(nn.Module):
    def __init__(self):
        super(LipClosureLoss, self).__init__()
        self.register_buffer('upper_lips', torch.LongTensor([61, 62, 63]), persistent=False)
        self.register_buffer('lower_lips', torch.LongTensor([67, 66, 65]), persistent=False)

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff_pred = pred_keypoints[:, self.upper_lips] - pred_keypoints[:, self.lower_lips]
        diff = keypoints[:, self.upper_lips] - keypoints[:, self.lower_lips]

        loss = (diff_pred.abs().sum(-1) - diff.abs().sum(-1)).abs().mean()

        return loss
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class EyeClosureLoss(nn.Module):
    def __init__(self):
        super(EyeClosureLoss, self).__init__()
        self.register_buffer('upper_lids', torch.LongTensor([37, 38, 43, 44]), persistent=False)
        self.register_buffer('lower_lids', torch.LongTensor([41, 40, 47, 46]), persistent=False)

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff_pred = pred_keypoints[:, self.upper_lids] - pred_keypoints[:, self.lower_lids]
        diff = keypoints[:, self.upper_lids] - keypoints[:, self.lower_lids]

        loss = (diff_pred.abs().sum(-1) - diff.abs().sum(-1)).abs().mean()

        return loss
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from typing import Union



class HeadPoseMatchingLoss(nn.Module):
    def __init__(self, loss_type = 'l2'):
        super(HeadPoseMatchingLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, 
                pred_thetas: Union[torch.Tensor, list], 
                target_thetas: Union[torch.Tensor, list]) -> torch.Tensor:
        loss = 0

        if isinstance(pred_thetas, torch.Tensor):
            pred_thetas = [pred_thetas]
            target_thetas = [target_thetas]

        for pred_theta, target_theta in zip(pred_thetas, target_thetas):
            if self.loss_type == 'l1':
                loss += (pred_theta - target_theta).abs().mean()
            elif self.loss_type == 'l2':
                loss += ((pred_theta - target_theta)**2).mean()

        return loss
import torch
from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
import cv2
import numpy as np

# from typing import Union
# from typing import Tuple, List
# from rt_gene.estimate_gaze_pytorch import GazeEstimator
# from rt_gene import FaceBox



# class GazeLoss(object):
#     def __init__(self,
#                  device: str,
#                  gaze_model_types: Union[List[str], str] = ['vgg16',],
#                  criterion: str = 'l1',
#                  interpolate: bool = False,
#                  layer_indices: tuple = (1, 6, 11, 18, 25),
# #                  layer_indices: tuple = (4, 5, 6, 7), # for resnet 
# #                  weights: tuple = (2.05625e-3, 2.78125e-4, 5.125e-5, 6.575e-8, 9.67e-10)
# #                  weights: tuple = (1.0, 1e-1, 4e-3, 2e-6, 1e-8),
# #                  weights: tuple = (0.0625, 0.125, 0.25, 1.0),
#                  weights: tuple = (0.03125, 0.0625, 0.125, 0.25, 1.0),
#                  ) -> None:
#         super(GazeLoss, self).__init__()
#         self.len_features = len(layer_indices)
#         # checkpoints_paths_dict = {'vgg16':'/Vol0/user/n.drobyshev/latent-texture-avatar/losses/gaze_models/vgg_16_2_forward_sum.pt', 'resnet18':'/Vol0/user/n.drobyshev/latent-texture-avatar/losses/gaze_models/resnet_18_2_forward_sum.pt'}
#         # if interpolate:
#         checkpoints_paths_dict = {'vgg16': '/group-volume/orc_srr/multimodal/t.khakhulin/pretrained/gaze_net.pt',
#                                 'resnet18': '/group-volume/orc_srr/multimodal/t.khakhulin/pretrained/gaze_net.pt'}
            
#         self.gaze_estimator = GazeEstimator(device=device,
#                                               model_nets_path=[checkpoints_paths_dict[m] for m in gaze_model_types],
#                                               gaze_model_types=gaze_model_types,
#                                               interpolate = interpolate,
#                                               align_face=True)

#         if criterion == 'l1':
#             self.criterion = nn.L1Loss()
#         elif criterion == 'l2':
#             self.criterion = nn.MSELoss()

#         self.layer_indices = layer_indices
#         self.weights = weights

#     @torch.cuda.amp.autocast(False)
#     def forward(self,
#                 inputs: Union[torch.Tensor, list],
#                 target: torch.Tensor,
#                 keypoints: torch.Tensor = None,
#                 interpolate=True) -> Union[torch.Tensor, list]:
#         if isinstance(inputs, list):
#             # Concat alongside the batch axis
#             input_is_a_list = True
#             num_chunks = len(inputs)
#             chunk_size = inputs[0].shape[0]
#             inputs = torch.cat(inputs)

#         else:
#             input_is_a_list = False
            
#         if interpolate:   
#             inputs = F.interpolate(inputs, (224, 224), mode='bicubic', align_corners=False)
#             target = F.interpolate(target, (224, 224), mode='bicubic', align_corners=False)
        
#         if keypoints is not None:
#             keypoints_np = [(kp[:, :2].cpu().numpy() + 1) / 2 * tgt.shape[2] for kp, tgt in zip(keypoints, target)]
# #             keypoints_np = [(kp[:, :2].cpu().numpy() + 1) / 2 * tgt.shape[2] for kp, tgt in zip(keypoints, target)]
# #             keypoints_np = [(kp[:, :2].cpu().numpy()/tgt.shape[2]*224).astype(np.int32) for kp, tgt in zip(keypoints, target)]
            
#             faceboxes = [FaceBox(left=kp[:, 0].min(),
#                                                top=kp[:, 1].min(),
#                                                right=kp[:, 0].max(),
#                                                bottom=kp[:, 1].max()) for kp in keypoints_np]
#         else:
#             faceboxes = None
#             keypoints_np = None

#         target = target.float()
#         inputs = inputs.float()

#         with torch.no_grad():
#             target_subjects = self.gaze_estimator.get_eye_embeddings(target,
#                                                                      self.layer_indices,
#                                                                      faceboxes,
#                                                                      keypoints_np)

#         # Filter subjects with visible eyes
#         visible_eyes = [subject is not None and subject.eye_embeddings is not None for subject in target_subjects]

#         if not any(visible_eyes):
#             return torch.zeros(1).to(target.device)

#         target_subjects = self.select_by_mask(target_subjects, visible_eyes)

#         faceboxes = [subject.box for subject in target_subjects]
#         keypoints_np = [subject.landmarks for subject in target_subjects]

#         target_features = [[] for i in range(self.len_features)]
#         for subject in target_subjects:
#             for k in range(self.len_features):
#                 target_features[k].append(subject.eye_embeddings[k])
#         target_features = [torch.cat(feats) for feats in target_features]

#         eye_masks = self.draw_eye_masks(keypoints_np, target.shape[2], target.device)

#         if input_is_a_list:
#             visible_eyes *= num_chunks
#             faceboxes *= num_chunks
#             keypoints_np *= num_chunks
#             eye_masks = torch.cat([eye_masks] * num_chunks)

#         # Grads are masked
#         inputs = inputs[visible_eyes]
# #         inputs.retain_grad() # turn it on while debugging
#         inputs_ = inputs * eye_masks + inputs.detach() * (1 - eye_masks)
# #         inputs_.retain_grad() # turn it on while debugging
        
#         # In order to apply eye masks for gradients, first calc the grads
# #         inputs_ = inputs.detach().clone().requires_grad_()
# #         inputs_ = inputs_ * eye_masks + inputs_.detach() * (1 - eye_masks)
#         input_subjects = self.gaze_estimator.get_eye_embeddings(inputs_,
#                                                                 self.layer_indices,
#                                                                 faceboxes,
#                                                                 keypoints_np)

#         input_features = [[] for i in range(self.len_features)]
#         for subject in input_subjects:
#             for k in range(self.len_features):
#                 input_features[k].append(subject.eye_embeddings[k])
#         input_features = [torch.cat(feats) for feats in input_features]

#         loss = 0

#         for input_feature, target_feature, weight in zip(input_features, target_features, self.weights):
#             if input_is_a_list:
#                 target_feature = torch.cat([target_feature.detach()] * num_chunks)

#             loss += weight * self.criterion(input_feature, target_feature)
        
#         return loss

#     @staticmethod
#     def select_by_mask(a, mask):
#         return [v for (is_true, v) in zip(mask, a) if is_true]

#     @staticmethod
#     def draw_eye_masks(keypoints_np, image_size, device):
#         ### Define drawing options ###
#         edges_parts = [list(range(36, 42)), list(range(42, 48))]

#         mask_kernel = np.ones((5, 5), np.uint8)

#         ### Start drawing ###
#         eye_masks = []

#         for xy in keypoints_np:
#             xy = xy[None, :, None].astype(np.int32)

#             eye_mask = np.zeros((image_size, image_size, 3), np.uint8)

#             for edges in edges_parts:
#                 eye_mask = cv2.fillConvexPoly(eye_mask, xy[0, edges], (255, 255, 255))

#             eye_mask = cv2.dilate(eye_mask, mask_kernel, iterations=1)
#             eye_mask = cv2.blur(eye_mask, mask_kernel.shape)
#             eye_mask = torch.FloatTensor(eye_mask[:, :, [0]].transpose(2, 0, 1)) / 255.
#             eye_masks.append(eye_mask)

#         eye_masks = torch.stack(eye_masks).to(device)

#         return eye_masks

