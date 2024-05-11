import torch
from torch import nn
from argparse import ArgumentParser
from pytorch3d.structures import Meshes

import src.networks as networks
from src.parametric_avatar import ParametricAvatar
from src.utils import args as args_utils
from src.utils import harmonic_encoding
from src.utils.visuals import mask_errosion


class ROME(nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("model")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--model_image_size', default=256, type=int)

        parser.add_argument('--align_source', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--align_target', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--align_scale', default=1.25, type=float)

        parser.add_argument('--use_mesh_deformations', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--subdivide_mesh', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--renderer_sigma', default=1e-8, type=float)
        parser.add_argument('--renderer_zfar', default=100.0, type=float)
        parser.add_argument('--renderer_type', default='soft_mesh')
        parser.add_argument('--renderer_texture_type', default='texture_uv')
        parser.add_argument('--renderer_normalized_alphas', default='False', type=args_utils.str2bool,
                            choices=[True, False])

        parser.add_argument('--deca_path', default='')
        parser.add_argument('--rome_data_dir', default='')


        parser.add_argument('--autoenc_cat_alphas', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_align_inputs', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_use_warp', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_num_channels', default=64, type=int)
        parser.add_argument('--autoenc_max_channels', default=512, type=int)
        parser.add_argument('--autoenc_num_groups', default=4, type=int)
        parser.add_argument('--autoenc_num_bottleneck_groups', default=0, type=int)
        parser.add_argument('--autoenc_num_blocks', default=2, type=int)
        parser.add_argument('--autoenc_num_layers', default=4, type=int)
        parser.add_argument('--autoenc_block_type', default='bottleneck')

        parser.add_argument('--neural_texture_channels', default=8, type=int)
        parser.add_argument('--num_harmonic_encoding_funcs', default=6, type=int)

        parser.add_argument('--unet_num_channels', default=64, type=int)
        parser.add_argument('--unet_max_channels', default=512, type=int)
        parser.add_argument('--unet_num_groups', default=4, type=int)
        parser.add_argument('--unet_num_blocks', default=1, type=int)
        parser.add_argument('--unet_num_layers', default=2, type=int)
        parser.add_argument('--unet_block_type', default='conv')
        parser.add_argument('--unet_skip_connection_type', default='cat')
        parser.add_argument('--unet_use_normals_cond', default=True, action='store_true')
        parser.add_argument('--unet_use_vertex_cond', action='store_true')
        parser.add_argument('--unet_use_uvs_cond', action='store_true')
        parser.add_argument('--unet_pred_mask', action='store_true')
        parser.add_argument('--use_separate_seg_unet', default='True', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--norm_layer_type', default='gn', type=str, choices=['bn', 'sync_bn', 'in', 'gn'])
        parser.add_argument('--activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--conv_layer_type', default='ws_conv', type=str, choices=['conv', 'ws_conv'])

        parser.add_argument('--deform_norm_layer_type', default='gn', type=str, choices=['bn', 'sync_bn', 'in', 'gn'])
        parser.add_argument('--deform_activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--deform_conv_layer_type', default='ws_conv', type=str, choices=['conv', 'ws_conv'])
        parser.add_argument('--unet_seg_weight', default=0.0, type=float)
        parser.add_argument('--unet_seg_type', default='bce_with_logits', type=str, choices=['bce_with_logits', 'dice'])
        parser.add_argument('--deform_face_tightness', default=0.0, type=float)

        parser.add_argument('--use_whole_segmentation', action='store_true')
        parser.add_argument('--mask_hair_for_neck', action='store_true')
        parser.add_argument('--use_hair_from_avatar', action='store_true')

        # Basis deformations
        parser.add_argument('--use_scalp_deforms', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--use_neck_deforms', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--use_basis_deformer', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--use_unet_deformer', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')

        parser.add_argument('--pretrained_encoder_basis_path', default='')
        parser.add_argument('--pretrained_vertex_basis_path', default='')
        parser.add_argument('--num_basis', default=50, type=int)
        parser.add_argument('--basis_init', default='pca', type=str, choices=['random', 'pca'])
        parser.add_argument('--num_vertex', default=5023, type=int)
        parser.add_argument('--train_basis', default=True, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--path_to_deca', default='DECA')

        parser.add_argument('--path_to_linear_hair_model',
                            default='data/linear_hair.pth')
        parser.add_argument('--path_to_mobile_model',
                            default='data/disp_model.pth')
        parser.add_argument('--n_scalp', default=60, type=int)

        parser.add_argument('--use_distill', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_mobile_version', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--deformer_path', default='data/rome.pth')

        parser.add_argument('--output_unet_deformer_feats', default=32, type=int,
                            help='output features in the UNet')

        parser.add_argument('--use_deca_details', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_flametex', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--upsample_type', default='nearest', type=str,
                            choices=['nearest', 'bilinear', 'bicubic'])

        parser.add_argument('--num_frequencies', default=6, type=int, help='frequency for harmonic encoding')
        parser.add_argument('--deform_face_scale_coef', default=0.0, type=float)
        parser.add_argument('--device', default='cpu', type=str)

        return parser_out

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.init_networks(args)

    def init_networks(self, args):
        self.autoencoder = networks.Autoencoder(
            args.autoenc_num_channels,
            args.autoenc_max_channels,
            args.autoenc_num_groups,
            args.autoenc_num_bottleneck_groups,
            args.autoenc_num_blocks,
            args.autoenc_num_layers,
            args.autoenc_block_type,
            input_channels=3 + 1,  # cat alphas
            input_size=args.model_image_size,
            output_channels=args.neural_texture_channels,
            norm_layer_type=args.norm_layer_type,
            activation_type=args.activation_type,
            conv_layer_type=args.conv_layer_type,
            use_psp=False,
        ).eval()

        self.basis_deformer = None
        self.vertex_deformer = None
        self.mask_hard_threshold = 0.6

        deformer_input_ch = args.neural_texture_channels

        deformer_input_ch += 3

        deformer_input_ch += 3 * args.num_frequencies * 2

        output_channels = self.args.output_unet_deformer_feats

        if self.args.use_unet_deformer:
            self.mesh_deformer = networks.UNet(
                args.unet_num_channels,
                args.unet_max_channels,
                args.unet_num_groups,
                args.unet_num_blocks,
                args.unet_num_layers,
                args.unet_block_type,
                input_channels=deformer_input_ch,
                output_channels=output_channels,
                skip_connection_type=args.unet_skip_connection_type,
                norm_layer_type=args.deform_norm_layer_type,
                activation_type=args.deform_activation_type,
                conv_layer_type=args.deform_conv_layer_type,
                downsampling_type='maxpool',
                upsampling_type='nearest',
            )

        input_mlp_feat = self.args.output_unet_deformer_feats + 2 * (1 + args.num_frequencies * 2)

        self.mlp_deformer = networks.MLP(
            num_channels=256,
            num_layers=8,
            skip_layer=4,
            input_channels=input_mlp_feat,
            output_channels=3,
            activation_type=args.activation_type,
            last_bias=False,
        )

        if self.args.use_basis_deformer:
            print('Create and load basis deformer.')
            self.basis_deformer = networks.EncoderResnet(
                pretrained_encoder_basis_path=args.pretrained_encoder_basis_path,
                norm_type='gn+ws',
                num_basis=args.num_basis)

            self.vertex_deformer = networks.EncoderVertex(
                path_to_deca_lib=args.path_to_deca,
                pretrained_vertex_basis_path=args.pretrained_vertex_basis_path,
                norm_type='gn+ws',
                num_basis=args.num_basis,
                basis_init=args.basis_init,
                num_vertex=args.num_vertex)

        self.parametric_avatar = ParametricAvatar(
            args.model_image_size,
            args.deca_path,
            args.use_scalp_deforms,
            args.use_neck_deforms,
            args.subdivide_mesh,
            args.use_deca_details,
            args.use_flametex,
            args,
            device=args.device,
        )

        self.unet = networks.UNet(
            args.unet_num_channels,
            args.unet_max_channels,
            args.unet_num_groups,
            args.unet_num_blocks,
            args.unet_num_layers,
            args.unet_block_type,
            input_channels=args.neural_texture_channels + 3 * (
                    1 + args.unet_use_vertex_cond) * (1 + 6 * 2),  # unet_use_normals_cond
            output_channels=3 + 1,
            skip_connection_type=args.unet_skip_connection_type,
            norm_layer_type=args.norm_layer_type,
            activation_type=args.activation_type,
            conv_layer_type=args.conv_layer_type,
            downsampling_type='maxpool',
            upsampling_type='nearest',
        ).eval()

    @torch.no_grad()
    def forward(self, data_dict, neutral_pose: bool = False, source_information=None):
        if source_information is None:
            source_information = dict()

        parametric_output = self.parametric_avatar.forward(
            data_dict['source_img'],
            data_dict['source_mask'],
            data_dict['source_keypoints'],
            data_dict['target_img'],
            data_dict['target_keypoints'],
            deformer_nets={
                'neural_texture_encoder': self.autoencoder,
                'unet_deformer': self.mesh_deformer,
                'mlp_deformer': self.mlp_deformer,
                'basis_deformer': self.basis_deformer,
            },
            neutral_pose=neutral_pose,
            neural_texture=source_information.get('neural_texture'),
            source_information=source_information,
        )
        result_dict = {}
        rendered_texture = parametric_output.pop('rendered_texture')

        for key, value in parametric_output.items():
            result_dict[key] = value

        unet_inputs = rendered_texture * result_dict['pred_target_hard_mask']

        normals = result_dict['pred_target_normal'].permute(0, 2, 3, 1)
        normal_inputs = harmonic_encoding.harmonic_encoding(normals, 6).permute(0, 3, 1, 2)
        unet_inputs = torch.cat([unet_inputs, normal_inputs], dim=1)
        unet_outputs = self.unet(unet_inputs)

        pred_img = torch.sigmoid(unet_outputs[:, :3])
        pred_soft_mask = torch.sigmoid(unet_outputs[:, 3:])

        return_mesh = False
        if return_mesh:
            verts = result_dict['vertices_target'].cpu()
            faces = self.parametric_avatar.render.faces.expand(verts.shape[0], -1, -1).long()
            result_dict['mesh'] = Meshes(verts=verts, faces=faces)

        result_dict['pred_target_unet_mask'] = pred_soft_mask
        result_dict['pred_target_img'] = pred_img
        mask_pred = (result_dict['pred_target_unet_mask'][0].cpu() > self.mask_hard_threshold).float()
        mask_pred = mask_errosion(mask_pred.float().numpy() * 255)
        result_dict['render_masked'] = result_dict['pred_target_img'][0].cpu() * (mask_pred) + (1 - mask_pred)

        return result_dict
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

        return lossimport torch



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
        return 10 * torch.log10(1 / mse)import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
try:
    from pytorch3d.loss.mesh_laplacian_smoothing import cot_laplacian
except:
    from pytorch3d.loss.mesh_laplacian_smoothing import laplacian_cot as cot_laplacian


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

        return loss.sum() / Nimport torch
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

        return lossimport torch
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

        return lossimport torch
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
        return selffrom .adversarial import AdversarialLoss
from .feature_matching import FeatureMatchingLoss
from .keypoints_matching import KeypointsMatchingLoss
from .eye_closure import EyeClosureLoss
from .lip_closure import LipClosureLoss
from .head_pose_matching import HeadPoseMatchingLoss
from .perceptual import PerceptualLoss

from .segmentation import SegmentationLoss, MultiScaleSilhouetteLoss
from .chamfer_silhouette import ChamferSilhouetteLoss
from .equivariance import EquivarianceLoss, LaplaceMeshLoss
from .vgg2face import VGGFace2Loss
from .gaze import GazeLoss

from .psnr import PSNR
from .lpips import LPIPS
from pytorch_msssim import SSIM, MS_SSIM# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from typing import Union

from src.utils import misc



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
            misc.apply_imagenet_normalization(inputs), \
            misc.apply_imagenet_normalization(target)
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

    return _PerceptualNetwork(network, layer_name_mapping, layers)import torch.nn as nn
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
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

from typing import Union



class ChamferSilhouetteLoss(nn.Module):
    def __init__(
        self, 
        num_neighbours=1, 
        use_same_number_of_points=False, 
        sample_outside_of_silhouette=False,
        use_visibility=True
    ):
        super(ChamferSilhouetteLoss, self).__init__()
        self.num_neighbours = num_neighbours
        self.use_same_number_of_points = use_same_number_of_points
        self.sample_outside_of_silhouette = sample_outside_of_silhouette
        self.use_visibility = use_visibility

    def forward(self, 
                pred_points: torch.Tensor,
                points_visibility: torch.Tensor,
                target_silhouette: torch.Tensor,
                target_segs: torch.Tensor) -> torch.Tensor:        
        target_points, target_lengths, weight = self.get_pointcloud(target_segs, target_silhouette)

        if self.use_visibility:
            pred_points, pred_lengths = self.get_visible_points(pred_points, points_visibility)
                
        if self.use_same_number_of_points:
            target_points = target_points[:, :pred_points.shape[1]]    

            target_lengths = pred_lengths = torch.minimum(target_lengths, pred_lengths)
            
            if self.sample_outside_of_silhouette:
                target_lengths = (target_lengths.clone() * weight).long()

            for i in range(target_points.shape[0]):
                target_points[i, target_lengths[i]:] = -100.0

            for i in range(pred_points.shape[0]):
                pred_points[i, pred_lengths[i]:] = -100.0

        visible_batch = target_lengths > 0
        if self.use_visibility:
            visible_batch *= pred_lengths > 0

        if self.use_visibility:
            loss = chamfer_distance(
                pred_points[visible_batch], 
                target_points[visible_batch], 
                x_lengths=pred_lengths[visible_batch], 
                y_lengths=target_lengths[visible_batch],
                num_neighbours=self.num_neighbours
            )        
        else:
            loss = chamfer_distance(
                pred_points[visible_batch], 
                target_points[visible_batch], 
                y_lengths=target_lengths[visible_batch],
                num_neighbours=self.num_neighbours
            )

        if isinstance(loss, tuple):
            loss = loss[0]
        
        return loss, pred_points, target_points
    
    @torch.no_grad()
    def get_pointcloud(self, seg, silhouette):
        if self.sample_outside_of_silhouette:
            silhouette = (silhouette > 0.0).type(seg.type())

            old_area = seg.view(seg.shape[0], -1).sum(1)
            seg = seg * (1 - silhouette)
            new_area = seg.view(seg.shape[0], -1).sum(1)

            weight = new_area / (old_area + 1e-7)
        
        else:
            weight = torch.ones(seg.shape[0], dtype=seg.dtype, device=seg.device)

        batch, coords = torch.nonzero(seg[:, 0] > 0.5).split([1, 2], dim=1)
        batch = batch[:, 0]
        coords = coords.float()
        coords[:, 0] = (coords[:, 0] / seg.shape[2] - 0.5) * 2
        coords[:, 1] = (coords[:, 1] / seg.shape[3] - 0.5) * 2

        pointcloud = -100.0 * torch.ones(seg.shape[0], seg.shape[2]*seg.shape[3], 2).to(seg.device)
        length = torch.zeros(seg.shape[0]).to(seg.device).long()
        for i in range(seg.shape[0]):
            pt = coords[batch == i]
            pt = pt[torch.randperm(pt.shape[0])] # randomly permute the points
            pointcloud[i][:pt.shape[0]] = torch.cat([pt[:, 1:], pt[:, :1]], dim=1)
            length[i] = pt.shape[0]
        
        return pointcloud, length, weight
    
    @staticmethod
    def get_visible_points(points, visibility):
        batch, indices = torch.nonzero(visibility > 0.0).split([1, 1], dim=1)
        batch = batch[:, 0]
        indices = indices[:, 0]

        length = torch.zeros(points.shape[0]).to(points.device).long()
        for i in range(points.shape[0]):
            batch_i = batch == i
            indices_i = indices[batch_i]
            points[i][:indices_i.shape[0]] = points[i][indices_i]
            points[i][indices_i.shape[0]:] = -100.0
            length[i] = indices_i.shape[0]

        return points, length


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


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

    return cham_dist, cham_normalsimport torch
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
        return loss / self.num_scalesimport torch
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

        return lossimport torch
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

        return lossimport torch
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

from typing import Union
from typing import Tuple, List
from rt_gene.estimate_gaze_pytorch import GazeEstimator
from rt_gene import FaceBox


class GazeLoss(object):
    def __init__(self,
                 device: str,
                 gaze_model_types: Union[List[str], str] = ['vgg16',],
                 criterion: str = 'l1',
                 interpolate: bool = False,
                 layer_indices: tuple = (1, 6, 11, 18, 25),
#                  layer_indices: tuple = (4, 5, 6, 7), # for resnet 
#                  weights: tuple = (2.05625e-3, 2.78125e-4, 5.125e-5, 6.575e-8, 9.67e-10)
#                  weights: tuple = (1.0, 1e-1, 4e-3, 2e-6, 1e-8),
#                  weights: tuple = (0.0625, 0.125, 0.25, 1.0),
                 weights: tuple = (0.03125, 0.0625, 0.125, 0.25, 1.0),
                 ) -> None:
        super(GazeLoss, self).__init__()
        self.len_features = len(layer_indices)
        # checkpoints_paths_dict = {'vgg16':'/Vol0/user/n.drobyshev/latent-texture-avatar/losses/gaze_models/vgg_16_2_forward_sum.pt', 'resnet18':'/Vol0/user/n.drobyshev/latent-texture-avatar/losses/gaze_models/resnet_18_2_forward_sum.pt'}
        # if interpolate:
        checkpoints_paths_dict = {'vgg16': '/group-volume/orc_srr/multimodal/t.khakhulin/pretrained/gaze_net.pt',
                                'resnet18': '/group-volume/orc_srr/multimodal/t.khakhulin/pretrained/gaze_net.pt'}
            
        self.gaze_estimator = GazeEstimator(device=device,
                                              model_nets_path=[checkpoints_paths_dict[m] for m in gaze_model_types],
                                              gaze_model_types=gaze_model_types,
                                              interpolate = interpolate,
                                              align_face=True)

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()

        self.layer_indices = layer_indices
        self.weights = weights

    @torch.cuda.amp.autocast(False)
    def forward(self,
                inputs: Union[torch.Tensor, list],
                target: torch.Tensor,
                keypoints: torch.Tensor = None,
                interpolate=True) -> Union[torch.Tensor, list]:
        if isinstance(inputs, list):
            # Concat alongside the batch axis
            input_is_a_list = True
            num_chunks = len(inputs)
            chunk_size = inputs[0].shape[0]
            inputs = torch.cat(inputs)

        else:
            input_is_a_list = False
            
        if interpolate:   
            inputs = F.interpolate(inputs, (224, 224), mode='bicubic', align_corners=False)
            target = F.interpolate(target, (224, 224), mode='bicubic', align_corners=False)
        
        if keypoints is not None:
            keypoints_np = [(kp[:, :2].cpu().numpy() + 1) / 2 * tgt.shape[2] for kp, tgt in zip(keypoints, target)]
#             keypoints_np = [(kp[:, :2].cpu().numpy() + 1) / 2 * tgt.shape[2] for kp, tgt in zip(keypoints, target)]
#             keypoints_np = [(kp[:, :2].cpu().numpy()/tgt.shape[2]*224).astype(np.int32) for kp, tgt in zip(keypoints, target)]
            
            faceboxes = [FaceBox(left=kp[:, 0].min(),
                                               top=kp[:, 1].min(),
                                               right=kp[:, 0].max(),
                                               bottom=kp[:, 1].max()) for kp in keypoints_np]
        else:
            faceboxes = None
            keypoints_np = None

        target = target.float()
        inputs = inputs.float()

        with torch.no_grad():
            target_subjects = self.gaze_estimator.get_eye_embeddings(target,
                                                                     self.layer_indices,
                                                                     faceboxes,
                                                                     keypoints_np)

        # Filter subjects with visible eyes
        visible_eyes = [subject is not None and subject.eye_embeddings is not None for subject in target_subjects]

        if not any(visible_eyes):
            return torch.zeros(1).to(target.device)

        target_subjects = self.select_by_mask(target_subjects, visible_eyes)

        faceboxes = [subject.box for subject in target_subjects]
        keypoints_np = [subject.landmarks for subject in target_subjects]

        target_features = [[] for i in range(self.len_features)]
        for subject in target_subjects:
            for k in range(self.len_features):
                target_features[k].append(subject.eye_embeddings[k])
        target_features = [torch.cat(feats) for feats in target_features]

        eye_masks = self.draw_eye_masks(keypoints_np, target.shape[2], target.device)

        if input_is_a_list:
            visible_eyes *= num_chunks
            faceboxes *= num_chunks
            keypoints_np *= num_chunks
            eye_masks = torch.cat([eye_masks] * num_chunks)

        # Grads are masked
        inputs = inputs[visible_eyes]
#         inputs.retain_grad() # turn it on while debugging
        inputs_ = inputs * eye_masks + inputs.detach() * (1 - eye_masks)
#         inputs_.retain_grad() # turn it on while debugging
        
        # In order to apply eye masks for gradients, first calc the grads
#         inputs_ = inputs.detach().clone().requires_grad_()
#         inputs_ = inputs_ * eye_masks + inputs_.detach() * (1 - eye_masks)
        input_subjects = self.gaze_estimator.get_eye_embeddings(inputs_,
                                                                self.layer_indices,
                                                                faceboxes,
                                                                keypoints_np)

        input_features = [[] for i in range(self.len_features)]
        for subject in input_subjects:
            for k in range(self.len_features):
                input_features[k].append(subject.eye_embeddings[k])
        input_features = [torch.cat(feats) for feats in input_features]

        loss = 0

        for input_feature, target_feature, weight in zip(input_features, target_features, self.weights):
            if input_is_a_list:
                target_feature = torch.cat([target_feature.detach()] * num_chunks)

            loss += weight * self.criterion(input_feature, target_feature)
        
        return loss

    @staticmethod
    def select_by_mask(a, mask):
        return [v for (is_true, v) in zip(mask, a) if is_true]

    @staticmethod
    def draw_eye_masks(keypoints_np, image_size, device):
        ### Define drawing options ###
        edges_parts = [list(range(36, 42)), list(range(42, 48))]

        mask_kernel = np.ones((5, 5), np.uint8)

        ### Start drawing ###
        eye_masks = []

        for xy in keypoints_np:
            xy = xy[None, :, None].astype(np.int32)

            eye_mask = np.zeros((image_size, image_size, 3), np.uint8)

            for edges in edges_parts:
                eye_mask = cv2.fillConvexPoly(eye_mask, xy[0, edges], (255, 255, 255))

            eye_mask = cv2.dilate(eye_mask, mask_kernel, iterations=1)
            eye_mask = cv2.blur(eye_mask, mask_kernel.shape)
            eye_mask = torch.FloatTensor(eye_mask[:, :, [0]].transpose(2, 0, 1)) / 255.
            eye_masks.append(eye_mask)

        eye_masks = torch.stack(eye_masks).to(device)

        return eye_masks
from src.networks import *
from src.utils import *
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from .common import layers


class UNet(nn.Module):
    def __init__(self, 
                 num_channels: int,
                 max_channels: int,
                 num_groups: int,
                 num_blocks: int,
                 num_layers: int,
                 block_type: str,
                 input_channels: int,
                 output_channels: int,
                 skip_connection_type: str,
                 norm_layer_type: str,
                 activation_type: str,
                 conv_layer_type: str,
                 downsampling_type: str,
                 upsampling_type: str,
                 multiscale_outputs: bool = False,
                 pretrained_model_path: str = '',
                 pretrained_model_name: str = 'unet'):
        super(UNet, self).__init__()
        self.skip_connection_type = skip_connection_type
        self.multiscale_outputs = multiscale_outputs
        expansion_factor = 4 if block_type == 'bottleneck' else 1

        if block_type != 'conv':
            self.from_inputs = nn.Conv2d(input_channels, num_channels * expansion_factor, 7, 1, 3, bias=False)

        out_channels = num_channels if block_type != 'conv' else input_channels

        if downsampling_type == 'maxpool':
            self.downsample = nn.MaxPool2d(kernel_size=2)
        elif downsampling_type == 'avgpool':
            self.downsample = nn.AvgPool2d(kernel_size=2)

        self.encoder = nn.ModuleList()

        for i in range(num_groups):
            layers_ = []

            in_channels = out_channels
            out_channels = min(num_channels * 2**(i+1), max_channels)

            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=in_channels, 
                    out_channels=in_channels if j < num_blocks - 1 else out_channels,
                    mid_channels=in_channels if in_channels != input_channels else num_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

            self.encoder.append(nn.Sequential(*layers_))

        if in_channels != out_channels:
            self.bottleneck = nn.Conv2d(out_channels, in_channels, 1, bias=False)
            out_channels = in_channels

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder = nn.ModuleList()

        for i in reversed(range(num_groups - 1)):
            in_channels = out_channels
            out_channels = min(num_channels * 2**i, max_channels)

            layers_ = []

            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=in_channels * 2 if j == 0 and skip_connection_type == 'cat' else in_channels, 
                    out_channels=in_channels if j < num_blocks - 1 else out_channels,
                    mid_channels=in_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

            self.decoder.append(nn.Sequential(*layers_))

        if block_type == 'conv':
            self.to_outputs = nn.Conv2d(out_channels * expansion_factor, output_channels, 1)

        else:
            self.to_outputs = nn.Sequential(
                layers.norm_layers[norm_layer_type](out_channels * expansion_factor, affine=True),
                layers.activations[activation_type](inplace=True),
                nn.Conv2d(out_channels * expansion_factor, output_channels, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode=upsampling_type)

        if pretrained_model_path:
            state_dict_full = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in state_dict_full.items():
                if pretrained_model_name in k:
                    state_dict[k.replace(f'{pretrained_model_name}.', '')] = v
            self.load_state_dict(state_dict, strict=False)
            print(f'Loaded {pretrained_model_name} state dict')

    def forward(self, x):
        if hasattr(self, 'from_inputs'):
            x = self.from_inputs(x)

        feats = []

        for i, block in enumerate(self.encoder):
            x = block(x)

            if i < len(self.encoder) - 1:
                feats.append(x)
                x = self.downsample(x)

        outputs = []

        if hasattr(self, 'bottleneck'):
            x = self.bottleneck(x)

        if self.multiscale_outputs:
            outputs.append(x)

        for j, block in zip(reversed(range(len(self.decoder))), self.decoder):
            x = self.upsample(x)

            if self.skip_connection_type == 'cat':
                x = torch.cat([x, feats[j]], dim=1)
            elif self.skip_connection_type == 'sum':
                x = x + feats[j]

            x = block(x)

            if self.multiscale_outputs:
                outputs.append(x)

        if self.multiscale_outputs:
            outputs[-1] = self.to_outputs(outputs[-1])
            return outputs

        else:
            return self.to_outputs(x)import torch
from torch import nn

from typing import Union, List
from src.networks.common import layers


class Discriminator(nn.Module):
    def __init__(self,
                 num_channels: int,
                 max_channels: int,
                 num_blocks: int,
                 input_channels: int):
        super(Discriminator, self).__init__()
        self.num_blocks = num_blocks

        self.in_channels = [min(num_channels * 2**(i-1), max_channels) for i in range(self.num_blocks)]
        self.in_channels[0] = input_channels
        
        self.out_channels = [min(num_channels * 2**i, max_channels) for i in range(self.num_blocks)]

        self.init_networks()

    def init_networks(self) -> None:
        self.blocks = nn.ModuleList()

        for i in range(self.num_blocks):
            self.blocks.append(
                layers.blocks['conv'](
                    in_channels=self.in_channels[i], 
                    out_channels=self.out_channels[i],
                    kernel_size=3,
                    padding=1,
                    stride=2 if i < self.num_blocks - 1 else 1,
                    norm_layer_type='in',
                    activation_type='lrelu'))

        self.to_scores = nn.Conv2d(
            in_channels=self.out_channels[-1],
            out_channels=1,
            kernel_size=1)

    def forward(self, inputs):
        outputs = inputs
        features = []

        for block in self.blocks:
            outputs = block(outputs)
            features.append(outputs)

        scores = self.to_scores(outputs)

        return scores, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self,
                 min_channels: int,
                 max_channels: int,
                 num_blocks: int,
                 input_channels: int,
                 input_size: int,
                 num_scales: int) -> None:
        super(MultiScaleDiscriminator, self).__init__()
        self.input_size = input_size
        self.num_scales = num_scales

        spatial_size = input_size
        self.nets = []

        for i in range(num_scales):
            net = Discriminator(min_channels, max_channels, num_blocks, input_channels)

            setattr(self, 'net_%04d' % spatial_size, net)
            self.nets.append(net)

            spatial_size //= 2

        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self, inputs: torch.Tensor):
        spatial_size = self.input_size
        scores, features = [], []

        for i in range(self.num_scales):
            net = getattr(self, 'net_%04d' % spatial_size)

            scores_i, features_i = net(inputs)

            scores.append([scores_i])
            features.append([[features_i_block] for features_i_block in features_i])

            spatial_size //= 2
            inputs = self.down(inputs)

        return scores, featuresimport torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from src.networks.common import layers


class Autoencoder(nn.Module):
    def __init__(self,
                 num_channels: int,
                 max_channels: int,
                 num_groups: int,
                 num_bottleneck_groups: int,
                 num_blocks: int,
                 num_layers: int,
                 block_type: str,
                 input_channels: int,
                 input_size: int,
                 output_channels: int,
                 norm_layer_type: str,
                 activation_type: str,
                 conv_layer_type: str,
                 use_psp: bool,
                 pretrained_model_path: str = '',
                 pretrained_model_name: str = 'autoencoder'):
        super(Autoencoder, self).__init__()
        # Encoder from inputs to latents
        expansion_factor = 4 if block_type == 'bottleneck' else 1

        layers_ = [nn.Conv2d(input_channels, num_channels * expansion_factor, 7, 1, 3, bias=False)]
        in_channels = num_channels
        out_channels = num_channels

        for i in range(num_groups):
            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=in_channels if j == 0 else out_channels,
                    out_channels=out_channels,
                    mid_channels=out_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

            in_channels = out_channels
            out_channels = min(num_channels * 2 ** (i + 1), max_channels)

            if i < num_groups - 1:
                layers_.append(nn.MaxPool2d(kernel_size=2))

        for i in range(num_bottleneck_groups):
            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=out_channels,
                    out_channels=out_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

        if use_psp:
            layers_.append(PSP(
                levels=4,
                num_channels=out_channels * expansion_factor,
                conv_layer=layers.conv_layers[conv_layer_type],
                norm_layer=layers.norm_layers[norm_layer_type],
                activation=layers.activations[activation_type]))

        for i in reversed(range(num_groups)):
            in_channels = out_channels
            out_channels = min(num_channels * 2 ** max(i - 1, 0), max_channels)

            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=in_channels,
                    out_channels=in_channels if j < num_blocks - 1 else out_channels,
                    mid_channels=in_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

            if i > 0:
                layers_.append(nn.Upsample(scale_factor=2, mode='nearest'))

        layers_ += [
            layers.norm_layers[norm_layer_type](out_channels * expansion_factor, affine=True),
            layers.activations[activation_type](inplace=True),
            nn.Conv2d(out_channels * expansion_factor, output_channels, 1)]

        self.net = nn.Sequential(*layers_)

        if pretrained_model_path:
            state_dict_full = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in state_dict_full.items():
                if pretrained_model_name in k:
                    state_dict[k.replace(f'{pretrained_model_name}.', '')] = v
            self.load_state_dict(state_dict)
            print('Loaded autoencoder state dict')

    def forward(self, x, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.net(x)
        else:
            return self.net(x)


class PSP(nn.Module):
    def __init__(self, levels, num_channels, conv_layer, norm_layer, activation):
        super(PSP, self).__init__()
        self.blocks = nn.ModuleList()

        for i in range(1, levels + 1):
            self.blocks.append(nn.Sequential(
                nn.AvgPool2d(2 ** i),
                norm_layer(num_channels),
                activation(inplace=True),
                conv_layer(num_channels, num_channels // levels, 1),
                nn.Upsample(scale_factor=2 ** i, mode='bilinear')))

        self.squish = conv_layer(num_channels * 2, num_channels, 1)

    def forward(self, x):
        out = [x]
        for block in self.blocks:
            out.append(block(x))
        out = torch.cat(out, dim=1)

        return self.squish(out)
import sys
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50

from src.networks.common.conv_layers import WSConv2d


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class EncoderResnet(nn.Module):
    def __init__(self, 
                 pretrained_encoder_basis_path='', 
                 norm_type='bn', 
                 num_basis=10,
                 head_init_gain=1e-3):
        super(EncoderResnet, self).__init__()
        if norm_type == 'gn+ws':
            self.backbone = resnet50(num_classes=159, norm_layer=lambda x: nn.GroupNorm(32, x))
            self.backbone = patch_conv_to_wsconv(self.backbone)

        elif norm_type == 'bn':
            self.backbone = resnet50(num_classes=159)

        self.backbone.fc = nn.Linear(in_features=2048, out_features=num_basis, bias=True)
        nn.init.zeros_(self.backbone.fc.bias)
        nn.init.xavier_normal_(self.backbone.fc.weight, gain=head_init_gain)

        if pretrained_encoder_basis_path:
            print('Load checkpoint in Encoder Resnet!')
            state_dict = torch.load(pretrained_encoder_basis_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
        
    def forward(self, x):
        y = self.backbone(x)
        return y


def patch_conv_to_wsconv(module):
    new_module = module
    
    if isinstance(module, nn.Conv2d):
        # Split affine part of instance norm into separable 1x1 conv
        new_module = WSConv2d(module.in_channels, module.out_channels, module.kernel_size,
                              module.stride, module.padding, module.dilation, module.groups, module.bias)
        
        new_module.weight.data = module.weight.data.clone()
        if module.bias:
            new_module.bias.data = module.bias.data.clone()
        
    else:
        for name, child in module.named_children():
            new_module.add_module(name, patch_conv_to_wsconv(child))

    return new_module


class EncoderVertex(nn.Module):
    def __init__(self, 
                 path_to_deca_lib='DECA/decalib',
                 pretrained_vertex_basis_path='', 
                 norm_type='bn', 
                 num_basis=10,
                 num_vertex=5023,
                 basis_init='pca'): 
        super(EncoderVertex, self).__init__()
        self.num_vertex = num_vertex
                                    
        if basis_init == 'pca':
            path = os.path.join(path_to_deca_lib, 'data', 'generic_model.pkl')
            with open(path, 'rb') as f:
                ss = pickle.load(f, encoding='latin1')
                flame_model = Struct(**ss)
            shapedirs = to_tensor(to_np(flame_model.shapedirs[:, :, :num_basis]), dtype=torch.float32)
            self.vertex = torch.nn.parameter.Parameter(shapedirs)
            del flame_model
        else:
            self.vertex = torch.nn.parameter.Parameter(torch.normal(mean=3.7647e-12, std=0.0003, size=[self.num_vertex, 3, num_basis]))
                                    
        if pretrained_vertex_basis_path:
            state_dict = torch.load(pretrained_vertex_basis_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
from .autoencoder import Autoencoder
from .unet import UNet
from .mlp import MLP
from .regress_encoder import EncoderVertex, EncoderResnet
from .multiscale_discriminator import MultiScaleDiscriminator
import torch
from torch import nn
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), 
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = False):
        super(AdaptiveConv, self).__init__()
        # Set options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert not bias, 'bias == True is not supported for AdaptiveConv'
        self.bias = None

        self.kernel_numel = kernel_size[0] * kernel_size[1]
        if len(kernel_size) == 3:
            self.kernel_numel *= kernel_size[2]

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.ada_weight = None # assigned externally

        if len(kernel_size) == 2:
            self.conv_func = F.conv2d
        elif len(kernel_size) == 3:
            self.conv_func = F.conv3d

    def forward(self, inputs):
        # Cast parameters into inputs.dtype
        if inputs.type() != self.ada_weight.type():
            weight = self.ada_weight.type(inputs.type())
        else:
            weight = self.ada_weight

        # Conv is applied to the inputs grouped by t frames
        B = weight.shape[0]
        T = inputs.shape[0] // B
        assert inputs.shape[0] == B*T, 'Wrong shape of weight'

        if self.kernel_numel > 1:
            if weight.shape[0] == 1:
                # No need to iterate through batch, can apply conv to the whole batch
                outputs = self.conv_func(inputs, weight[0], None, self.stride, self.padding, self.dilation, self.groups)

            else:
                outputs = []
                for b in range(B):
                    outputs += [self.conv_func(inputs[b*T:(b+1)*T], weight[b], None, self.stride, self.padding, self.dilation, self.groups)]
                outputs = torch.cat(outputs, 0)

        else:
            if weight.shape[0] == 1:
                if len(inputs.shape) == 5:
                    weight = weight[..., None, None, None]
                else:
                    weight = weight[..., None, None]

                outputs = self.conv_func(inputs, weight[0], None, self.stride, self.padding, self.dilation, self.groups)
            else:
                # 1x1(x1) adaptive convolution is a simple bmm
                if len(weight.shape) == 6:
                    weight = weight[..., 0, 0, 0]
                else:
                    weight = weight[..., 0, 0]

                outputs = torch.bmm(weight, inputs.view(B*T, inputs.shape[1], -1)).view(B, -1, *inputs.shape[2:])

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
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class AdaptiveConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, 
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = False):
        kernel_size = (kernel_size,) * 3
        super(AdaptiveConv, self).__init__(in_channels, out_channels, kernel_size,
                                           stride, padding, dilation, groups, bias)import torch
from torch import nn

import math
import itertools
from typing import Union



class NormParamsPredictor(nn.Module):
    def __init__(self, 
                 net_or_nets: Union[list, nn.Module], 
                 embed_channels: int) -> None:
        super(NormParamsPredictor, self).__init__()        
        self.mappers = nn.ModuleList()

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            m_name = m.__class__.__name__
            if (m_name == 'AdaptiveBatchNorm' or m_name == 'AdaptiveSyncBatchNorm' or 
                m_name == 'AdaptiveInstanceNorm' or m_name == 'AdaptiveGroupNorm'):

                self.mappers.append(nn.Linear(
                    in_features=embed_channels, 
                    out_features=m.num_features * 2,
                    bias=False))

    def forward(self, embed):
        params = []

        for mapper in self.mappers:
            param = mapper(embed)
            weight, bias = param.split(param.shape[1] // 2, dim=1)
            params += [(weight, bias)]

        return params


class SPADEParamsPredictor(nn.Module):
    def __init__(self, 
                 net_or_nets: Union[list, nn.Module], 
                 embed_channels: int,
                 spatial_dims=2) -> None:
        super(SPADEParamsPredictor, self).__init__()        
        self.mappers = nn.ModuleList()

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            m_name = m.__class__.__name__
            if (m_name == 'AdaptiveBatchNorm' or m_name == 'AdaptiveSyncBatchNorm' or 
                m_name == 'AdaptiveInstanceNorm' or m_name == 'AdaptiveGroupNorm'):

                if spatial_dims == 2:
                    self.mappers.append(nn.Conv2d(
                        in_channels=embed_channels, 
                        out_channels=m.num_features * 2,
                        kernel_size=1,
                        bias=False))
                else:
                    self.mappers.append(nn.Conv3d(
                        in_channels=embed_channels, 
                        out_channels=m.num_features * 2,
                        kernel_size=1,
                        bias=False))

    def forward(self, embed):
        params = []

        for mapper in self.mappers:
            param = mapper(embed)
            weight, bias = param.split(param.shape[1] // 2, dim=1)
            params += [(weight, bias)]

        return params


class ConvParamsPredictor(nn.Module):
    def __init__(self, 
                 net_or_nets: Union[list, nn.Module], 
                 embed_channels: int) -> None:
        super(ConvParamsPredictor, self).__init__()        
        # Matrices that perform a lowrank matrix decomposition W = U E V
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()
        self.kernel_size = []

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            if m.__class__.__name__ == 'AdaptiveConv' or m.__class__.__name__ == 'AdaptiveConv3d':
                # Assumes that adaptive conv layers have no bias
                kernel_numel = m.kernel_size[0] * m.kernel_size[1]
                if len(m.kernel_size) == 3:
                    kernel_numel *= m.kernel_size[2]

                if kernel_numel == 1:
                    self.u += [nn.Parameter(torch.empty(m.out_channels, embed_channels))]
                    self.v += [nn.Parameter(torch.empty(embed_channels, m.in_channels))]

                elif kernel_numel > 1:
                    self.u += [nn.Parameter(torch.empty(m.out_channels, embed_channels))]
                    self.v += [nn.Parameter(torch.empty(m.in_channels, embed_channels))]

                self.kernel_size += [m.kernel_size]

                nn.init.xavier_normal_(self.u[-1], gain=0.02)
                nn.init.xavier_normal_(self.v[-1], gain=0.02)

    def forward(self, embed):       
        params = []

        for u, v, kernel_size in zip(self.u, self.v, self.kernel_size):
            kernel_numel = kernel_size[0] * kernel_size[1]
            if len(kernel_size) == 3:
                kernel_numel *= kernel_size[2]

            embed_ = embed

            if kernel_numel == 1:
                # AdaptiveConv with kernel size = 1
                weight = u[None].matmul(embed_).matmul(v[None])
                weight = weight.view(*weight.shape, *kernel_size) # B x C_out x C_in x 1 ...

            else:
                embed_ = embed_[..., None]

                kernel_numel_ = 1
                kernel_size_ = (1,)*len(kernel_size)

                param = embed_.view(*embed_.shape[:2], -1)
                param = u[None].matmul(param) # B x C_out x C_emb/2
                b, c_out = param.shape[:2]
                param = param.view(b, c_out, -1, kernel_numel_)
                param = v[None].matmul(param) # B x C_out x C_in x kernel_numel
                weight = param.view(*param.shape[:3], *kernel_size_)

            params += [weight]

        return params


def assign_adaptive_norm_params(net_or_nets, params, alpha=1.0):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if (m_name == 'AdaptiveBatchNorm' or m_name == 'AdaptiveSyncBatchNorm' or 
            m_name == 'AdaptiveInstanceNorm' or m_name == 'AdaptiveGroupNorm'):
            ada_weight, ada_bias = params.pop(0)

            if len(ada_weight.shape) == 2:
                m.ada_weight = m.weight[None] + ada_weight * alpha
                m.ada_bias = m.bias[None] + ada_bias * alpha
            elif len(ada_weight.shape) == 4:
                m.ada_weight = m.weight[None, :, None, None] + ada_weight * alpha
                m.ada_bias = m.bias[None, :, None, None] + ada_bias + alpha
            elif len(ada_weight.shape) == 5:
                m.ada_weight = m.weight[None, :, None, None, None] + ada_weight * alpha
                m.ada_bias = m.bias[None, :, None, None, None] + ada_bias + alpha

def assign_adaptive_conv_params(net_or_nets, params, alpha=1.0):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if m_name == 'AdaptiveConv' or m_name == 'AdaptiveConv3d':
            attr_name = 'weight_orig' if hasattr(m, 'weight_orig') else 'weight'

            weight = getattr(m, attr_name)
            ada_weight = params.pop(0)

            ada_weight = weight[None] + ada_weight * alpha
            setattr(m, 'ada_' + attr_name, ada_weight)import torch
from torch import nn
import torch.nn.functional as F
import functools
from typing import Union, Tuple, List

from . import norm_layers as norms
from . import conv_layers as convs


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        batch_size, channels, in_height, in_width = inputs.size()

        out_height = in_height // self.upscale_factor
        out_width = in_width // self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, out_height, self.upscale_factor,
            out_width, self.upscale_factor)

        channels *= self.upscale_factor ** 2
        unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffle_out.view(batch_size, channels, out_height, out_width)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        mid_channels: int = -1,
        spade_channels: int = -1,
        num_layers: int = 2,
        expansion_factor: int = None,
        kernel_size: Union[int, Tuple[int], List[int]] = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer_type: str = 'conv',
        norm_layer_type: str = 'bn',
        activation_type: str = 'relu',
        skip_layer_type: str = 'conv',
        resize_layer_type: str = 'none'):
        super(BasicBlock, self).__init__()
        if stride > 1 and resize_layer_type in ['nearest', 'bilinear']:
            self.upsample = lambda inputs: F.interpolate(input=inputs, 
                                                         scale_factor=stride, 
                                                         mode=resize_layer_type, 
                                                         align_corners=None if resize_layer_type == 'nearest' else False)

        if mid_channels == -1:
            mid_channels = out_channels

        if norm_layer_type != 'none':
            norm_layer = norm_layers[norm_layer_type]
        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]
        skip_layer = conv_layers[skip_layer_type]

        ### Initialize the layers of the first half of the block ###
        layers_ = []

        for i in range(num_layers):
            if norm_layer_type != 'none':
                if spade_channels != -1:
                    layers_ += [norm_layer(in_channels if i == 0 else mid_channels, spade_channels, affine=True)]
                else:
                    layers_ += [norm_layer(in_channels if i == 0 else mid_channels, affine=True)]

            layers_ += [
                activation(inplace=True),
                conv_layer(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=out_channels if i == num_layers - 1 else mid_channels,
                    kernel_size=kernel_size, 
                    stride=stride if resize_layer_type == 'none' and i == num_layers - 1 else 1,
                    padding=padding, 
                    dilation=dilation, 
                    groups=groups,
                    bias=norm_layer_type == 'none')]

        self.main = nn.Sequential(*layers_)

        if in_channels != out_channels:
            self.skip = skip_layer(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=1,
                bias=norm_layer_type == 'none')
        else:
            self.skip = nn.Identity()

        if stride > 1 and resize_layer_type in downsampling_layers:
            self.downsample = downsampling_layers[resize_layer_type](stride)

    def forward(self, x):
        if hasattr(self, 'upsample'):
            x = self.upsample(x)

        x = self.main(x) + self.skip(x)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return x


class BottleneckBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        mid_channels: int = -1,
        spade_channels: int = -1,
        num_layers: int = 3,
        expansion_factor: int = 4,
        kernel_size: Union[int, Tuple[int], List[int]] = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer_type: str = 'conv',
        norm_layer_type: str = 'bn',
        activation_type: str = 'relu',
        skip_layer_type: str = 'conv',
        resize_layer_type: str = 'none'):
        """This is a base module for a residual bottleneck block"""
        super(BottleneckBlock, self).__init__()
        if stride > 1 and resize_layer_type in ['nearest', 'bilinear']:
            self.upsample = lambda inputs: F.interpolate(input=inputs, 
                                                         scale_factor=stride, 
                                                         mode=resize_layer_type, 
                                                         align_corners=None if resize_layer_type == 'nearest' else False)

        if mid_channels == -1:
            mid_channels = out_channels

        if norm_layer_type != 'none':
            norm_layer = norm_layers[norm_layer_type]
        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]
        skip_layer = conv_layers[skip_layer_type]

        layers_ = []

        if norm_layer_type != 'none':
            if spade_channels != -1:
                layers_ += [norm_layer(in_channels * expansion_factor, spade_channels, affine=True)]
            else:
                layers_ += [norm_layer(in_channels * expansion_factor, affine=True)]

        layers_ += [
            activation(inplace=True),
            conv_layer(
                in_channels=in_channels * expansion_factor,
                out_channels=mid_channels,
                kernel_size=1,
                bias=norm_layer_type == 'none')]

        if norm_layer_type != 'none':
            if spade_channels != -1:
                layers_ += [norm_layer(mid_channels, spade_channels, affine=True)]
            else:
                layers_ += [norm_layer(mid_channels, affine=True)]
        layers_ += [activation(inplace=True)]

        assert num_layers > 2, 'Minimum number of layers is 3'
        for i in range(num_layers - 2):
            layers_ += [
                conv_layer(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size, 
                    stride=stride if resize_layer_type == 'none' and i == num_layers - 3 else 1,
                    padding=padding, 
                    dilation=dilation, 
                    groups=groups,
                    bias=norm_layer_type == 'none')]

            if norm_layer_type != 'none':
                if spade_channels != -1:
                    layers_ += [norm_layer(mid_channels, spade_channels, affine=True)]
                else:
                    layers_ += [norm_layer(mid_channels, affine=True)]
            layers_ += [activation(inplace=True)]

        layers_ += [
            skip_layer(
                in_channels=mid_channels,
                out_channels=out_channels * expansion_factor,
                kernel_size=1,
                bias=norm_layer_type == 'none')]

        self.main = nn.Sequential(*layers_)

        if in_channels != out_channels:
            self.skip = skip_layer(
                in_channels=in_channels * expansion_factor,
                out_channels=out_channels * expansion_factor, 
                kernel_size=1,
                bias=norm_layer_type == 'none')
        else:
            self.skip = nn.Identity()

        if stride > 1 and resize_layer_type in downsampling_layers:
            self.downsample = downsampling_layers[resize_layer_type](stride)

    def forward(self, x):
        if hasattr(self, 'upsample'):
            x = self.upsample(x)

        x = self.main(x) + self.skip(x)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        mid_channels: int = -1,
        spade_channels: int = -1,
        num_layers: int = 1,
        expansion_factor: int = None,
        kernel_size: Union[int, Tuple[int], List[int]] = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer_type: str = 'conv',
        norm_layer_type: str = 'none',
        activation_type: str = 'relu',
        skip_layer_type: str = 'conv',
        resize_layer_type: str = 'none'):
        """This is a base module for residual blocks"""
        super(ConvBlock, self).__init__()
        if stride > 1 and resize_layer_type in ['nearest', 'bilinear']:
            self.upsample = lambda inputs: F.interpolate(input=inputs, 
                                                         scale_factor=stride, 
                                                         mode=resize_layer_type, 
                                                         align_corners=None if resize_layer_type == 'nearest' else False)

        if mid_channels == -1:
            mid_channels = out_channels

        if norm_layer_type != 'none':
            norm_layer = norm_layers[norm_layer_type]
        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]
        skip_layer = conv_layers[skip_layer_type]

        ### Initialize the layers of the first half of the block ###
        layers_ = []

        for i in range(num_layers):
            layers_ += [
                conv_layer(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=out_channels if i == num_layers - 1 else mid_channels,
                    kernel_size=kernel_size,
                    stride=stride if resize_layer_type == 'none' and i == num_layers - 1 else 1,
                    padding=padding, 
                    dilation=dilation, 
                    groups=groups,
                    bias=norm_layer_type == 'none')]

            if norm_layer_type != 'none':
                if spade_channels != -1:
                    layers_ += [norm_layer(out_channels if i == num_layers - 1 else mid_channels, spade_channels, affine=True)]
                else:
                    layers_ += [norm_layer(out_channels if i == num_layers - 1 else mid_channels, affine=True)]
            layers_ += [activation(inplace=True)]

        self.main = nn.Sequential(*layers_)

        if stride > 1 and resize_layer_type in downsampling_layers:
            self.downsample = downsampling_layers[resize_layer_type](stride)

    def forward(self, x):
        if hasattr(self, 'upsample'):
            x = self.upsample(x)

        x = self.main(x)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)
        
        return x


############################################################
#                Definitions for the layers                #
############################################################

# Supported blocks
blocks = {
    'basic': BasicBlock,
    'bottleneck': BottleneckBlock,
    'conv': ConvBlock
}

# Supported conv layers
conv_layers = {
    'conv': nn.Conv2d,
    'ws_conv': convs.WSConv2d,
    'conv_3d': nn.Conv3d,
    'ada_conv': convs.AdaptiveConv,
    'ada_conv_3d': convs.AdaptiveConv3d}

# Supported activations
activations = {
    'relu': nn.ReLU,
    'lrelu': functools.partial(nn.LeakyReLU, negative_slope=0.2)}

# Supported normalization layers
norm_layers = {
    'in': nn.InstanceNorm2d,
    'in_3d': nn.InstanceNorm3d,
    'bn': nn.BatchNorm2d,
    'gn': lambda num_features, affine=True: nn.GroupNorm(num_groups=min(32, num_features), num_channels=num_features, affine=affine),
    'ada_in': norms.AdaptiveInstanceNorm,
    'ada_gn': lambda num_features, affine=True: norms.AdaptiveGroupNorm(num_groups=min(32, num_features), num_features=num_features, affine=affine),
}

# Supported downsampling layers
downsampling_layers = {
    'avgpool': nn.AvgPool2d,
    'maxpool': nn.MaxPool2d,
    'avgpool_3d': nn.AvgPool3d,
    'maxpool_3d': nn.MaxPool3d,
    'pixelunshuffle': PixelUnShuffle}import torch
from torch import nn


def init_parameters(self, num_features):
    self.weight = nn.Parameter(torch.ones(num_features))
    self.bias = nn.Parameter(torch.zeros(num_features))
    
    # These tensors are assigned externally
    self.ada_weight = None
    self.ada_bias = None

def init_spade_parameters(self, num_features, num_spade_features):
    self.conv_weight = nn.Conv2d(num_spade_features, num_features, 1, bias=False)
    self.conv_bias = nn.Conv2d(num_spade_features, num_features, 1, bias=False)

    nn.init.xavier_normal_(self.conv_weight.weight, gain=0.02)
    nn.init.xavier_normal_(self.conv_bias.weight, gain=0.02)

    # These tensors are assigned externally
    self.spade_features = None

def common_forward(x, weight, bias):
    B = weight.shape[0]
    T = x.shape[0] // B

    x = x.view(B, T, *x.shape[1:])

    if len(weight.shape) == 2:
        # Broadcast weight and bias accross T and spatial size of outputs
        if len(x.shape) == 5:
            x = x * weight[:, None, :, None, None] + bias[:, None, :, None, None]
        elif len(x.shape) == 6:
            x = x * weight[:, None, :, None, None, None] + bias[:, None, :, None, None, None]
    else:
        x = x * weight[:, None] + bias[:, None]

    x = x.view(B*T, *x.shape[2:])

    return x


class AdaptiveInstanceNorm(nn.modules.instancenorm._InstanceNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveInstanceNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveInstanceNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class AdaptiveBatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveBatchNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveBatchNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class AdaptiveGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        super(AdaptiveGroupNorm, self).__init__(num_groups, num_features, eps, False)
        self.num_features = num_features
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveGroupNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self) -> str:
        return '{num_groups}, {num_features}, eps={eps}, ' \
            'affine=True'.format(**self.__dict__)


class SPADEGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, num_spade_features, eps=1e-5, affine=True):
        super(SPADEGroupNorm, self).__init__(num_groups, num_features, eps, False)
        self.num_features = num_features
        self.num_spade_features = num_spade_features

        init_spade_parameters(self, num_features, num_spade_features)

    def forward(self, x):
        x = super(SPADEGroupNorm, self).forward(x)

        weight = self.conv_weight(self.spade_features) + 1.0
        bias = self.conv_bias(self.spade_features)
        
        x = common_forward(x, weight, bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self) -> str:
        return '{num_groups}, {num_features}, eps={eps}, ' \
            'affine=True, spade_features={num_spade_features}'.format(**self.__dict__)


def assign_spade_features(net_or_nets, features):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if (m_name == 'SPADEBatchNorm' or m_name == 'SPADESyncBatchNorm' or 
            m_name == 'SPADEInstanceNorm' or m_name == 'SPADEGroupNorm'):
            m.spade_features = features.pop()import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import sys
import os


class FaceParsing(object):
    def __init__(self,
                 path_to_face_parsing,
                 device='cuda'):
        super(FaceParsing, self).__init__()
        import sys
        sys.path.append(path_to_face_parsing)

        from model import BiSeNet

        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes).to(device)
        save_pth = os.path.join(f'{path_to_face_parsing}/res/cp/79999_iter.pth')
        self.net.load_state_dict(torch.load(save_pth, map_location='cpu'))
        self.net.eval()

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)

        self.mask_types = {
            'face': [1, 2, 3, 4, 5, 6, 10, 11, 12, 13],
            'ears': [7, 8, 9],
            'neck': [14, 15],
            'cloth': [16],
            'hair': [17, 18],
        }

    @torch.no_grad()
    def forward(self, x):
        h, w = x.shape[2:]
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        x = F.interpolate(x, size=(512, 512), mode='bilinear')
        y = self.net(x)[0]
        y = F.interpolate(y, size=(h, w), mode='bilinear')

        labels = y.argmax(1)

        mask = torch.zeros(x.shape[0], len(self.mask_types.keys()), h, w, dtype=x.dtype, device=x.device)

        for i, indices in enumerate(self.mask_types.values()):
            for j in indices:
                mask[:, i] += labels == j

        return maskimport torch
import torch.nn as nn
from collections import OrderedDict

from .common import layers


class MLP(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_layers: int,
                 skip_layer: int,
                 input_channels: int,
                 output_channels: int,
                 activation_type: str,
                 pretrained_model_path: str = '',
                 pretrained_model_name: str = 'mlp',
                 last_bias=False):
        super(MLP, self).__init__()
        assert num_layers > 1
        layers_ = [
            nn.Linear(input_channels, num_channels),
            layers.activations[activation_type](inplace=True)]

        for i in range(skip_layer - 1):
            layers_ += [
                nn.Linear(num_channels, num_channels),
                layers.activations[activation_type](inplace=True)]

        self.block_1 = nn.Sequential(*layers_)

        layers_ = [
            nn.Linear(num_channels + input_channels, num_channels),
            layers.activations[activation_type](inplace=True)]

        for i in range(num_layers - skip_layer - 1):
            layers_ += [
                nn.Linear(num_channels, num_channels),
                layers.activations[activation_type](inplace=True)]

        layers_ += [
            nn.Linear(num_channels, output_channels, bias=last_bias)]

        self.block_2 = nn.Sequential(*layers_)

        if pretrained_model_path:
            state_dict_full = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in state_dict_full.items():
                if pretrained_model_name in k:
                    state_dict[k.replace(f'{pretrained_model_name}.', '')] = v
            self.load_state_dict(state_dict)

    def forward(self, x):
        if len(x.shape) == 4:
            # Input is a 4D tensor
            b, c, h, w = x.shape
            x_ = x.permute(0, 2, 3, 1).reshape(-1, c)
        else:
            x_ = x

        y = self.block_1(x_)
        y = torch.cat([x_, y], dim=1)
        y = self.block_2(y)

        if len(x.shape) == 4:
            y = y.view(b, h, w, -1).permute(0, 3, 1, 2)

        return yimport torch
import os
import pickle as pkl
import tensorboardX
from torchvision import transforms
import copy
from tqdm import tqdm


class Logger(object):
    def __init__(self, args, experiment_dir, rank):
        super(Logger, self).__init__()
        self.ddp = args.num_gpus > 1
        self.logging_freq = args.logging_freq
        self.visuals_freq = args.visuals_freq
        self.batch_size = args.batch_size
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        self.rank = rank

        self.train_iter = 0
        self.epoch = 0
        self.output_train_logs = not (self.train_iter + 1) % self.logging_freq
        self.output_train_visuals = self.visuals_freq > 0 and not (self.train_iter + 1) % self.visuals_freq

        self.to_image = transforms.ToPILImage()
        self.losses_buffer = {'train': {}, 'test': {}}

        if self.rank == 0:
            for phase in ['train', 'test']:
                os.makedirs(self.experiment_dir / 'images' / phase, exist_ok=True)

            self.losses = {'train': {}, 'test': {}}
            self.writer = tensorboardX.SummaryWriter(self.experiment_dir)

    def log(self, phase, losses_dict=None, histograms_dict=None, visuals=None, epoch_end=False):
        if losses_dict is not None:
            for name, loss in losses_dict.items():
                if name in self.losses_buffer[phase].keys():
                    self.losses_buffer[phase][name].append(loss)
                else:
                    self.losses_buffer[phase][name] = [loss]

        if phase == 'train':
            self.train_iter += 1

            if self.output_train_logs:
                self.output_logs(phase)
                self.output_histograms(phase, histograms_dict)

            if self.output_train_visuals and visuals is not None:
                self.output_visuals(phase, visuals)

            self.output_train_logs = not (self.train_iter + 1) % self.logging_freq
            self.output_train_visuals = self.visuals_freq > 0 and not (self.train_iter + 1) % self.visuals_freq

        elif phase == 'test' and epoch_end:
            self.epoch += 1
            self.output_logs(phase)

            if visuals is not None:
                self.output_visuals(phase, visuals)

    def output_logs(self, phase):
        # Average the buffers and flush
        names = list(self.losses_buffer[phase].keys())
        losses = []
        for losses_ in self.losses_buffer[phase].values():
            losses.append(torch.stack(losses_).mean())

        if not losses:
            return

        losses = torch.stack(losses)

        self.losses_buffer[phase] = {}

        if self.ddp:
            # Synchronize buffers across GPUs
            losses_ = torch.zeros(size=(torch.distributed.get_world_size(), len(losses)), dtype=losses.dtype,
                                  device=losses.device)
            losses_[self.rank] = losses
            torch.distributed.reduce(losses_, dst=0, op=torch.distributed.ReduceOp.SUM)

            if self.rank == 0:
                losses = losses_.mean(0)

        if self.rank == 0:
            for name, loss in zip(names, losses):
                loss = loss.item()
                if name in self.losses[phase].keys():
                    self.losses[phase][name].append(loss)
                else:
                    self.losses[phase][name] = [loss]

                self.writer.add_scalar(name, loss, self.train_iter)

            tqdm.write(f'Iter {self.train_iter:06d} ' + ', '.join(
                f'{name}: {losses[-1]:.3f}' for name, losses in self.losses[phase].items()))

    def output_histograms(self, phase, histograms):
        if self.rank == 0:
            for key, value in histograms.items():
                value = value.reshape(-1).clone().cpu().data.numpy()
                self.writer.add_histogram(f'{phase}_{key}_hist', value, self.train_iter)

    def output_visuals(self, phase, visuals):
        device = str(visuals.device)

        if self.ddp and device != 'cpu':
            # Synchronize visuals across GPUs
            c, h, w = visuals.shape[1:]
            b = self.batch_size if phase == 'train' else 1
            visuals_ = torch.zeros(size=(torch.distributed.get_world_size(), b, c, h, w), dtype=visuals.dtype,
                                   device=visuals.device)
            visuals_[self.rank, :visuals.shape[0]] = visuals
            torch.distributed.reduce(visuals_, dst=0, op=torch.distributed.ReduceOp.SUM)

            if self.rank == 0:
                visuals = visuals_.view(-1, c, h, w)

        if device != 'cpu':
            # All visuals are reduced, save only one image
            name = f'{self.train_iter:06d}.png'
        else:
            # Save all images
            name = f'{self.train_iter:06d}_{self.rank}.png'

        if self.rank == 0 or device == 'cpu':
            visuals = torch.cat(visuals.split(1, 0), 2)[0]  # cat batch dim in lines w.r.t. height
            visuals = visuals.cpu()

            # Save visuals
            image = self.to_image(visuals)
            image.save(self.experiment_dir / 'images' / phase / name)

            if self.rank == 0:
                self.writer.add_image(f'{phase}_images', visuals, self.train_iter)

    def state_dict(self):
        state_dict = {
            'losses': self.losses,
            'train_iter': self.train_iter,
            'epoch': self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.losses = state_dict['losses']
        self.train_iter = state_dict['train_iter']
        self.epoch = state_dict['epoch']import torch
from torch import nn
from torch.nn import functional as F

from typing import Union, Tuple



class GradScaler(nn.Module):
    def __init__(self, size: Union[int, Tuple[int]]):
        super(GradScaler, self).__init__()

    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        if isinstance(self.size, tuple):
            for i in range(len(self.size)):
                grad_output[..., i] /= self.size[i]

        elif isinstance(self.size, int):
            grad_output /= self.size

        return grad_output


class GridSample(nn.Module):
    def __init__(self, size: Union[int, Tuple[int]]):
        super(GridSample, self).__init__()
        self.scaler = GradScaler(size)

    def forward(self, input, grid, padding_mode='reflection', align_corners=False):
        return F.grid_sample(input, self.scaler(grid), padding_mode=padding_mode, align_corners=align_corners)


def make_grid(h, w, device=torch.device('cpu'), dtype=torch.float32):
    grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    v, u = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([u, v], dim=2).view(1, h, w, 2)

    return grid


def grid_sampler_backward(grad_out, grid, h=None, w=None, padding_mode='zeros', align_corners=False):
    with torch.no_grad():
        b, c = grad_out.shape[:2]
        if h is None or w is None:
            h, w = grad_out.shape[2:]
        size = torch.FloatTensor([w, h]).to(grad_out.device)
        grad_in = torch.zeros(b, c, h, w, device=grad_out.device)

        if align_corners:
            grid_ = (grid + 1) / 2 * (size - 1)
        else:
            grid_ = ((grid + 1) * size - 1) / 2

        if padding_mode == 'border':
            assert False, 'TODO'

        elif padding_mode == 'reflection':
            assert False, 'TODO'

        grid_nw = grid_.floor().long()
        
        grid_ne = grid_nw.clone()
        grid_ne[..., 0] += 1
        
        grid_sw = grid_nw.clone()
        grid_sw[..., 1] += 1
        
        grid_se = grid_nw.clone() + 1
        
        nw = (grid_se - grid_).prod(3)
        ne = (grid_ - grid_sw).abs().prod(3)
        sw = (grid_ne - grid_).abs().prod(3)
        se = (grid_ - grid_nw).prod(3)

        indices_ = torch.cat([
            (
                (
                    g[:, None, ..., 0] + g[:, None,..., 1] * w
                ).repeat_interleave(c, dim=1) 
                + torch.arange(c, device=g.device)[None, :, None, None] * (h*w) # add channel shifts
                + torch.arange(b, device=g.device)[:, None, None, None] * (c*h*w) # add batch size shifts
            ).view(-1) 
            for g in [grid_nw, grid_ne, grid_sw, grid_se]
        ])

        masks = torch.cat([
            (
                (g[..., 0] >= 0) & (g[..., 0] < w) & (g[..., 1] >= 0) & (g[..., 1] < h)
            )[:, None].repeat_interleave(c, dim=1).view(-1)
            for g in [grid_nw, grid_ne, grid_sw, grid_se]
        ])
    
    values_ = torch.cat([
        (m[:, None].repeat_interleave(c, dim=1) * grad_out).view(-1)
        for m in [nw, ne, sw, se]
    ])

    indices = indices_[masks]
    values = values_[masks]
    
    grad_in.put_(indices, values, accumulate=True)

    return grad_inimport torch
import torch.nn.functional as F


def batch_cont2matrix(module_input):
    ''' Decoder for transforming a latent representation to rotation matrices
        Implements the decoding method described in:
        "On the Continuity of Rotation Representations in Neural Networks"
        Code from https://github.com/vchoutas/expose
    '''
    batch_size = module_input.shape[0]
    reshaped_input = module_input.reshape(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(reshaped_input[:, :, 0].clone(), dim=1)

    dot_prod = torch.sum(
        b1 * reshaped_input[:, :, 1].clone(), dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=1)
    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats.view(batch_size, -1, 3, 3)
import argparse



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
    elif isfloat(v):
        v_type = float
        v = float(v)
    elif v == 'True':
        v = True
    elif v == 'False':
        v = False

    return k, v, v_type

def parse_args(args_path):
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    with open(args_path, 'rt') as args_file:
        lines = args_file.readlines()
        for line in lines:
            k, v, v_type = parse_args_line(line)
            parser.add('--%s' % k, type=v_type, default=v)

    args, _ = parser.parse_known_args()

    return args"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from torch.nn import Module
from torch import nn



def apply_spectral_norm(module, name='weight', apply_to=['conv2d'], n_power_iterations=1, eps=1e-12):
    # Apply only to modules in apply_to list
    module_name = module.__class__.__name__.lower()
    if module_name not in apply_to:
        return module

    if isinstance(module, nn.ConvTranspose2d):
        dim = 1
    else:
        dim = 0

    SpectralNorm.apply(module, name, n_power_iterations, dim, eps, adaptive='adaptive' in module_name)

    return module

def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break

    return module


class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12, adaptive: bool = False) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.adaptive = adaptive

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight

        if self.adaptive:
            assert self.dim == 0

            height = weight_mat.size(1)
            return weight_mat.reshape(weight_mat.shape[0], height, -1)

        else:
            if self.dim != 0:
                # permute dim to front
                weight_mat = weight_mat.permute(self.dim,
                                                *[d for d in range(weight_mat.dim()) if d != self.dim])
            height = weight_mat.size(0)
            return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        if self.adaptive:
            weight = getattr(module, 'ada_' + self.name + '_orig')
        else:
            weight = getattr(module, self.name + '_orig')

        weight_mat = self.reshape_weight_to_matrix(weight)

        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    if self.adaptive:
                        v = normalize(torch.matmul(u[None, None, :], weight_mat)[:, 0].mean(0), dim=0, eps=self.eps, out=v)
                        u = normalize(torch.matmul(weight_mat, v[None, :, None])[..., 0].mean(0), dim=0, eps=self.eps, out=u)

                    else:
                        v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                        u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)

                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        if self.adaptive:
            sigma = torch.mv(torch.matmul(weight_mat, v[None, :, None])[..., 0], u)

            if len(weight.shape) == 6:
                sigma = sigma[:, None, None, None, None, None]
            else:
                sigma = sigma[:, None, None, None, None]

        else:
            sigma = torch.dot(u, torch.mv(weight_mat, v))

        weight = weight / sigma
        
        return weight

    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, 'ada_' + self.name if self.adaptive else self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float, adaptive: bool) -> 'SpectralNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        fn.adaptive = adaptive

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormLoadStateDictPreHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs) -> None:
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if version is None and all(weight_key + s in state_dict for s in ('_orig', '_u', '_v')) and \
                    weight_key not in state_dict:
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ('_orig', '', '_u'):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + '_orig']
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + '_v'] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None) -> T_module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return moduleimport torch
from torch import optim
import math



def parse_3dmm_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp

def world_to_camera(pts_world, params):
    R, offset, roi_box, size = params['R'], params['offset'], params['roi_box'], params['size']
    crop_box = params['crop_box'] if 'crop_box' in params.keys() and len(params['crop_box']) else None

    if pts_world.shape[0] < R.shape[0]:
        pts_camera = pts_world.repeat_interleave(R.shape[0] // pts_world.shape[0], dim=0)
    
    elif pts_world.shape[0] > R.shape[0]:
        num_repeats = pts_world.shape[0] // R.shape[0]

        R = R.repeat_interleave(num_repeats, dim=0)
        offset = offset.repeat_interleave(num_repeats, dim=0)
        roi_box = roi_box.repeat_interleave(num_repeats, dim=0)
        size = size.repeat_interleave(num_repeats, dim=0)
        if crop_box is not None:
            crop_box = crop_box.repeat_interleave(num_repeats, dim=0)

        pts_camera = pts_world.clone()

    else:
        pts_camera = pts_world.clone()

    pts_camera[..., 2] += 0.5
    pts_camera *= 2e5

    pts_camera = pts_camera @ R.transpose(1, 2) + offset.transpose(1, 2)

    pts_camera[..., 0] -= 1
    pts_camera[..., 2] -= 1
    pts_camera[..., 1] = 120 - pts_camera[..., 1]
    
    sx, sy, ex, ey = [chunk[..., 0] for chunk in roi_box.split(1, dim=2)]
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    scale_z = (scale_x + scale_y) / 2
    
    pts_camera[..., 0] = pts_camera[..., 0] * scale_x + sx
    pts_camera[..., 1] = pts_camera[..., 1] * scale_y + sy
    pts_camera[..., 2] = pts_camera[..., 2] * scale_z

    pts_camera /= size
    pts_camera[..., 0] -= 0.5
    pts_camera[..., 1] -= 0.5
    pts_camera[..., :2] *= 2

    if crop_box is not None:
        crop_shift_x = (crop_box[..., 0] + crop_box[..., 2]) / 2
        crop_shift_y = (crop_box[..., 1] + crop_box[..., 3]) / 2
        
        pts_camera[..., 0] -= crop_shift_x
        pts_camera[..., 1] -= crop_shift_y
    
        crop_scale_x = (crop_box[..., 2] - crop_box[..., 0]) / 2
        crop_scale_y = (crop_box[..., 3] - crop_box[..., 1]) / 2
        crop_scale_z = (crop_scale_x + crop_scale_y) / 2

        pts_camera[..., 0] /= crop_scale_x
        pts_camera[..., 1] /= crop_scale_y
        pts_camera[..., 2] /= crop_scale_z
    
    return pts_camera

def camera_to_world(pts_camera, params):
    R, offset, roi_box, size = params['R'], params['offset'], params['roi_box'], params['size']
    crop_box = params['crop_box'] if 'crop_box' in params.keys() and len(params['crop_box']) else None

    if pts_camera.shape[0] < R.shape[0]:
        pts_world = pts_camera.repeat_interleave(R.shape[0] // pts_camera.shape[0], dim=0)
    
    elif pts_camera.shape[0] > R.shape[0]:
        num_repeats = pts_camera.shape[0] // R.shape[0]

        R = R.repeat_interleave(num_repeats, dim=0)
        offset = offset.repeat_interleave(num_repeats, dim=0)
        roi_box = roi_box.repeat_interleave(num_repeats, dim=0)
        size = size.repeat_interleave(num_repeats, dim=0)
        if crop_box is not None:
            crop_box = crop_box.repeat_interleave(num_repeats, dim=0)

        pts_world = pts_camera.clone()

    else:
        pts_world = pts_camera.clone()

    if crop_box is not None:
        crop_scale_x = (crop_box[..., 2] - crop_box[..., 0]) / 2
        crop_scale_y = (crop_box[..., 3] - crop_box[..., 1]) / 2
        crop_scale_z = (crop_scale_x + crop_scale_y) / 2

        pts_world[..., 0] *= crop_scale_x
        pts_world[..., 1] *= crop_scale_y
        pts_world[..., 2] *= crop_scale_z

        crop_shift_x = (crop_box[..., 0] + crop_box[..., 2]) / 2
        crop_shift_y = (crop_box[..., 1] + crop_box[..., 3]) / 2

        pts_world[..., 0] += crop_shift_x
        pts_world[..., 1] += crop_shift_y
        
    pts_world[..., :2] /= 2
    pts_world[..., 0] += 0.5
    pts_world[..., 1] += 0.5
    pts_world *= size

    sx, sy, ex, ey = [chunk[..., 0] for chunk in roi_box.split(1, dim=2)]
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    scale_z = (scale_x + scale_y) / 2
    
    pts_world[..., 0] = (pts_world[..., 0] - sx) / scale_x
    pts_world[..., 1] = (pts_world[..., 1] - sy) / scale_y
    pts_world[..., 2] = pts_world[..., 2] / scale_z

    pts_world[..., 0] += 1
    pts_world[..., 2] += 1
    pts_world[..., 1] = -(pts_world[..., 1] - 120)
    
    pts_world = (pts_world - offset.transpose(1, 2)) @ torch.linalg.inv(R.transpose(1, 2))
    
    pts_world /= 2e5
    pts_world[..., 2] -= 0.5
    
    return pts_world

###############################################################################

def align_ffhq_with_zoom(pts_camera, params, zoom_factor=0.6):
    R, offset = params['theta'].split([2, 1], dim=2)
    crop_box = params['crop_box'] if 'crop_box' in params.keys() and len(params['crop_box']) else None

    if pts_camera.shape[0] != R.shape[0]:
        pts_camera = pts_camera.repeat_interleave(R.shape[0], dim=0)
    else:
        pts_camera = pts_camera.clone()
    
    pts_camera = pts_camera @ R.transpose(1, 2) + offset.transpose(1, 2)

    # Zoom into face
    pts_camera *= zoom_factor

    if crop_box is not None:
        crop_shift_x = (crop_box[..., 0] + crop_box[..., 2]) / 2
        crop_shift_y = (crop_box[..., 1] + crop_box[..., 3]) / 2

        pts_camera[..., 0] -= crop_shift_x
        pts_camera[..., 1] -= crop_shift_y
        
        crop_scale_x = (crop_box[..., 2] - crop_box[..., 0]) / 2
        crop_scale_y = (crop_box[..., 3] - crop_box[..., 1]) / 2
        
        pts_camera[..., 0] /= crop_scale_x
        pts_camera[..., 1] /= crop_scale_y
    
    return pts_camera

###############################################################################

def get_transform_matrix(scale, rotation, translation):
    b = scale.shape[0]
    dtype = scale.dtype
    device = scale.device

    eye_matrix = torch.eye(4, dtype=dtype, device=device)[None].repeat_interleave(b, dim=0)

    # Scale transform
    S = eye_matrix.clone()

    if scale.shape[1] == 3:
        S[:, 0, 0] = scale[:, 0]
        S[:, 1, 1] = scale[:, 1]
        S[:, 2, 2] = scale[:, 2]
    else:
        S[:, 0, 0] = scale[:, 0]
        S[:, 1, 1] = scale[:, 0]
        S[:, 2, 2] = scale[:, 0]

    # Rotation transform
    R = eye_matrix.clone()

    rotation = rotation.clamp(-math.pi/2, math.pi)

    yaw, pitch, roll = rotation.split(1, dim=1)
    yaw, pitch, roll = yaw[:, 0], pitch[:, 0], roll[:, 0] # squeeze angles
    yaw_cos = yaw.cos()
    yaw_sin = yaw.sin()
    pitch_cos = pitch.cos()
    pitch_sin = pitch.sin()
    roll_cos = roll.cos()
    roll_sin = roll.sin()

    R[:, 0, 0] = yaw_cos * pitch_cos
    R[:, 0, 1] = yaw_cos * pitch_sin * roll_sin - yaw_sin * roll_cos
    R[:, 0, 2] = yaw_cos * pitch_sin * roll_cos + yaw_sin * roll_sin

    R[:, 1, 0] = yaw_sin * pitch_cos
    R[:, 1, 1] = yaw_sin * pitch_sin * roll_sin + yaw_cos * roll_cos
    R[:, 1, 2] = yaw_sin * pitch_sin * roll_cos - yaw_cos * roll_sin

    R[:, 2, 0] = -pitch_sin
    R[:, 2, 1] = pitch_cos * roll_sin
    R[:, 2, 2] = pitch_cos * roll_cos

    # Translation transform
    T = eye_matrix.clone()

    T[:, 0, 3] = translation[:, 0]
    T[:, 1, 3] = translation[:, 1]
    T[:, 2, 3] = translation[:, 2]

    theta = S @ R @ T

    return theta

def estimate_transform_from_keypoints(keypoints, aligned_keypoints, dilation=True, shear=False):
    b, n = keypoints.shape[:2]
    device = keypoints.device
    dtype = keypoints.dtype

    keypoints = keypoints.to(device)
    aligned_keypoints = aligned_keypoints.to(device)

    keypoints = torch.cat([keypoints, torch.ones(b, n, 1, device=device, dtype=dtype)], dim=2)

    if not dilation and not shear:
        # scale, yaw, pitch, roll, dx, dy, dz
        param = torch.tensor([[1,   0, 0, 0,   0, 0, 0]], device=device, dtype=dtype)

        scale, rotation, translation = param.repeat_interleave(b, dim=0).split([1, 3, 3], dim=1)
        params = [scale, rotation, translation]

    elif dilation and not shear:
        # scale_x, scale_y, scale_z, yaw, pitch, roll, dx, dy, dz
        param = torch.tensor([[1, 1, 1,   0, 0, 0,   0, 0, 0]], device=device, dtype=dtype)

        scale, rotation, translation = param.repeat_interleave(b, dim=0).split([3, 3, 3], dim=1)
        params = [scale, rotation, translation]

    elif dilation and shear:
        # full affine matrix
        theta = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], device=device, dtype=dtype)
        theta = theta[None].repeat_interleave(b, dim=0)
        params = [theta]

    # Solve for a given transform
    params = [p.clone().requires_grad_() for p in params]

    opt = optim.LBFGS(params)

    def closure():
        opt.zero_grad()

        if not shear:
            theta = get_transform_matrix(*params)[:, :3]
        else:
            theta = params[0]
        
        pred_aligned_keypoints = keypoints @ theta.transpose(1, 2)

        loss = ((pred_aligned_keypoints - aligned_keypoints)**2).mean()
        loss.backward()

        return loss

    for i in range(5):
        opt.step(closure)

    if not shear:
        theta = get_transform_matrix(*params).detach()
    else:
        theta = params[0].detach()

        eye = torch.zeros(b, 4, device=device, dtype=dtype)
        eye[:, 2] = 1

        theta = torch.cat([theta, eye], dim=1)

    return theta, paramsimport torch
import numpy as np
import cv2
from torchvision import transforms
import apex
from torch import nn



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

@torch.no_grad()
def grid_sampler_backward(grad_out, grid, h=None, w=None, padding_mode='zeros', align_corners=False):
    b, c = grad_out.shape[:2]
    if h is None or w is None:
        h, w = grad_out.shape[2:]
    size = torch.FloatTensor([w, h]).to(grad_out.device)
    grad_in = torch.zeros(b, c, h, w, device=grad_out.device)

    if align_corners:
        grid_ = (grid + 1) / 2 * (size - 1)
    else:
        grid_ = ((grid + 1) * size - 1) / 2

    if padding_mode == 'border':
        assert False, 'TODO'

    elif padding_mode == 'reflection':
        assert False, 'TODO'

    grid_nw = grid_.floor().long()
    
    grid_ne = grid_nw.clone()
    grid_ne[..., 0] += 1
    
    grid_sw = grid_nw.clone()
    grid_sw[..., 1] += 1
    
    grid_se = grid_nw.clone() + 1
    
    nw = (grid_se - grid_).prod(3)
    ne = (grid_ - grid_sw).abs().prod(3)
    sw = (grid_ne - grid_).abs().prod(3)
    se = (grid_ - grid_nw).prod(3)

    indices_ = torch.cat([
        (
            (
                g[:, None, ..., 0] + g[:, None,..., 1] * w
            ).repeat_interleave(c, dim=1) 
            + torch.arange(c, device=g.device)[None, :, None, None] * (h*w) # add channel shifts
            + torch.arange(b, device=g.device)[:, None, None, None] * (c*h*w) # add batch size shifts
        ).view(-1) 
        for g in [grid_nw, grid_ne, grid_sw, grid_se]
    ])

    masks = torch.cat([
        (
            (g[..., 0] >= 0) & (g[..., 0] < w) & (g[..., 1] >= 0) & (g[..., 1] < h)
        )[:, None].repeat_interleave(c, dim=1).view(-1)
        for g in [grid_nw, grid_ne, grid_sw, grid_se]
    ])
    
    values_ = torch.cat([
        (m[:, None].repeat_interleave(c, dim=1) * grad_out).view(-1)
        for m in [nw, ne, sw, se]
    ])

    indices = indices_[masks]
    values = values_[masks]
    
    grad_in.put_(indices, values, accumulate=True)

    return grad_in

def replace_bn_with_in(module):
    mod = module
    
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, apex.parallel.SyncBatchNorm):
        mod = nn.InstanceNorm2d(module.num_features, affine=True)

        gamma = module.weight.data.squeeze().detach().clone()
        beta = module.bias.data.squeeze().detach().clone()
        
        mod.weight.data = gamma
        mod.bias.data = beta

    else:
        for name, child in module.named_children():
            mod.add_module(name, replace_bn_with_in(child))

    del module
    return mod

@torch.no_grad()
def keypoints_to_heatmaps(keypoints, img):
    HEATMAPS_VAR = 1e-2
    s = img.shape[2]

    keypoints = keypoints[..., :2] # use 2D projection of keypoints

    return kp2gaussian(keypoints, img.shape[2:], HEATMAPS_VAR)

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

def prepare_visual(data_dict, tensor_name, preprocessing_op=None):
    visuals = []

    if tensor_name in data_dict.keys():
        tensor = data_dict[tensor_name].detach().cpu()

        if preprocessing_op is not None:
            tensor = preprocessing_op(tensor)

        if tensor.shape[1] == 1:
            tensor = torch.cat([tensor] * 3, dim=1)

        elif tensor.shape[1] == 2:
            b, _, h, w = tensor.shape

            tensor = torch.cat([tensor, torch.empty(b, 1, h, w, dtype=tensor.dtype).fill_(-1)], dim=1)

        visuals += [tensor]

    return visuals

def draw_stickman(keypoints, image_size):
    ### Define drawing options ###
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

    for kp in keypoints:
        if kp is None:
            stickmen.append(torch.zeros(3, image_size, image_size))
            continue

        if isinstance(kp, torch.Tensor):
            xy = (kp[:, :2].detach().cpu().numpy() + 1) / 2 * image_size
        
        elif kp.max() < 1.0:
            xy = kp[:, :2] * image_size

        else:
            xy = kp[:, :2]

        xy = xy[None, :, None].astype(np.int32)

        stickman = np.ones((image_size, image_size, 3), np.uint8)

        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
            stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=2)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2. 

    return stickmen

def draw_keypoints(img, kp):
    to_image = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    
    h, w = img.shape[-2:]
    kp = (kp + 1) / 2
    kp[..., 0] *= w
    kp[..., 1] *= h
    kp = kp.detach().cpu().numpy().astype(int)

    img_out = []
    for i in range(kp.shape[0]):
        img_i = np.asarray(to_image(img[i].cpu())).copy()
        for j in range(kp.shape[1]):
            cv2.circle(img_i, tuple(kp[i, j]), radius=2, color=(255, 0, 0), thickness=-1)
        img_out.append(to_tensor(img_i))
    img_out = torch.stack(img_out)
    
    return img_out


def vis_parsing_maps(parsing_annotations, im, with_image=False, stride=1):
    # Colors for all 20 parts
    part_colors = [[255, 140, 255], [0, 30, 255],
                   [255, 0, 85],[0, 255, 255],  [255, 0, 170],
                   [170, 255, 0],
                   [0, 255, 85],   [255, 0, 0],
                   [0, 255, 0], [85, 255, 0], [0, 255, 170],
                   [255, 85, 0], [0, 255, 255],
                  [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [255, 170, 0],[0, 85, 255],
                   [255, 255, 0], [255, 255, 170],
                   [255, 85, 255],[255, 255, 35],
                   [255, 0, 255], [80, 215, 255], [140, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_annotations.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.ones((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    if with_image:
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    else:
        vis_im = vis_parsing_anno_color
    return vis_im# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from torch.nn import init



def weight_init(init_type='normal', gain=0.02, bias=None):
    def init_func(m):
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or
                                     class_name.find('Linear') != -1 or
                                     class_name.find('Embedding') != -1):
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
            
            elif init_type == 'none':
                m.reset_parameters()
            
            else:
                raise NotImplementedError(
                    'initialization method [%s] is '
                    'not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                if bias is not None:
                    bias_type = getattr(bias, 'type', 'normal')
                    if bias_type == 'normal':
                        bias_gain = getattr(bias, 'gain', 0.5)
                        init.normal_(m.bias.data, 0.0, bias_gain)

                    else:
                        raise NotImplementedError(
                            'initialization method [%s] is '
                            'not implemented' % bias_type)
                
                else:
                    init.constant_(m.bias.data, 0.0)
    
    return init_funcimport torch
import numpy as np
import torch.nn as nn
import torchvision.models as models


def create_regressor(model_name, num_params):
    models_mapping = {
                'resnet50': lambda x : models.resnet50(pretrained=True),
                'resnet18': lambda x : models.resnet18(pretrained=True),
                'efficientnet_b0': lambda x : models.efficientnet_b0(pretrained=True),
                'efficientnet_b3': lambda x : models.efficientnet_b3(pretrained=True),
                'mobilenet_v3_large': lambda x : models.mobilenet_v3_large(pretrained=True),
                'mobilenet_v3_small': lambda x : models.mobilenet_v3_small(pretrained=True),
                'mobilenet_v2': lambda x: models.mobilenet_v2(pretrained=True),
                 }
    model = models_mapping[model_name](True)
    if model_name == 'resnet50':
        model.fc = nn.Linear(in_features=2048, out_features=num_params, bias=True)
    elif model_name == 'resnet18':
        model.fc = nn.Linear(in_features=512, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v3_large':
        model.classifier[3] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v3_small':
        model.classifier[3] = nn.Linear(in_features=1024, out_features=num_params, bias=True)    
    elif model_name == 'efficientnet_b3':
        model.classifier[1] = nn.Linear(in_features=1536, out_features=num_params, bias=True)
    elif model_name == 'efficientnet_b0':
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    else:
        model.fc = nn.Linear(in_features=10, out_features=num_params, bias=True)
    return model


def process_black_shape(shape_img):
    black_mask = shape_img == 0.0
    shape_img[black_mask] = 1.0
    return shape_img


def prepare_input_data(data_dict):
        for k, v in data_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    v_ = v_.cuda()
                    v[k_] = v_.view(-1, *v_.shape[2:])
                data_dict[k] = v
            else:
                v = v.cuda()
                data_dict[k] = v.view(-1, *v.shape[2:])

        return data_dict


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[2,1,0]]
    return image.astype(np.uint8).copy()

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import io
from torchvision import transforms
import cv2
import yaml

from torch.utils.data.dataloader import default_collate


def create_batch(indexes, target_dataset):
    res = []
    for idx in indexes:
        res.append(target_dataset[idx])
    return default_collate(res)


def swap_source_target(target_idx_in_dataset, batch, dataset):
    assert len(target_idx_in_dataset) == len(batch['target_img'])
    for i, target_idx in enumerate(target_idx_in_dataset):
        for key in ['target_img', 'target_mask', 'target_keypoints']:
            batch[key][i] = dataset[target_idx][key]
    return batch


def process_black_shape(shape_img):
    black_mask = shape_img == 0.0
    shape_img[black_mask] = 1.0
    shape_img_opa = torch.cat([shape_img, (black_mask.float()).mean(-1, keepdim=True)], dim=-1)
    return shape_img_opa[..., :256, :256]


def process_white_shape(shape_img):
    black_mask = shape_img == 1.0
    shape_img[black_mask] = 1.0
    shape_img_opa = torch.cat([shape_img, (black_mask.float()).mean(-1, keepdim=True)], dim=-1)
    return shape_img_opa[..., :256, :256]


def mask_update_by_vec(masks, thres_list):
    for i, th in enumerate(thres_list):
        masks[i] = masks[i] > th
        masks[i] = mask_errosion(masks[i].numpy() * 255)
    return masks


def obtain_modnet_mask(im: torch.tensor, modnet: nn.Module,
                       ref_size=512, ):
    transes = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    im_transform = transforms.Compose(transes)
    im = im_transform(im)
    im = im[None, :, :, :]

    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
    _, _, matte = modnet(im, True)
    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte[None]


def mask_errosion(mask):
    kernel = np.ones((9, 9), np.uint8)
    resmask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return torch.from_numpy(resmask / 255)


def save_kp(img_path, kp_save_path, fa):
    kp_input = io.imread(img_path)
    l = fa.get_landmarks(kp_input, return_bboxes=True)
    if l is not None and l[-1] is not None:
        keypoints, _, bboxes = l
        areas = []
        for bbox in bboxes:
            areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        i = np.argmax(areas)

        np.savetxt(kp_save_path, keypoints[i])
        return True
    return False


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    return image.astype(np.uint8).copy()
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import math



def harmonic_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True) -> torch.Tensor:
    r"""Apply harmonic encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be harmonically encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a harmonic encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            harmonic encoding (default: True).
    Returns:
    (torch.Tensor): harmonic encoding of the input tensor.

    Source: https://github.com/krrish94/nerf-pytorch
    """
    # TESTED
    # Trivially, the input tensor is added to the harmonic encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no harmonic encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
from pytorch3d.structures import Meshes
import pickle as pkl

from DECA.decalib.utils import util
from DECA.decalib.models.lbs import blend_shapes, batch_rodrigues
from DECA.decalib.deca import DECA
import DECA.decalib.utils.config as config

from src.utils import harmonic_encoding
from src.utils.params import batch_cont2matrix
from src.utils.processing import create_regressor
from src.parametric_avatar import ParametricAvatar


class ParametricAvatarTrainable(ParametricAvatar):

    def estimate_texture(self, source_image: torch.Tensor, source_mask: torch.Tensor,
                         texture_encoder: torch.nn.Module) -> torch.Tensor:
        autoenc_inputs = torch.cat([source_image, source_mask], dim=1)
        neural_texture = texture_encoder(autoenc_inputs)
        if neural_texture.shape[-1] != 256:
            neural_texture = F.interpolate(neural_texture, (256, 256))

        return neural_texture

    def deform_source_mesh(self, verts_parametric, neural_texture, deformer_nets):
        unet_deformer = deformer_nets['unet_deformer']
        vertex_deformer = deformer_nets['mlp_deformer']
        batch_size = verts_parametric.shape[0]

        verts_uvs = self.true_uvcoords[:, :, None, :2]  # 1 x V x 1 x 2

        verts_uvs = verts_uvs.repeat_interleave(batch_size, dim=0)

        # bs x 3 x H x W
        verts_texture = self.render.world2uv(verts_parametric) * 5

        enc_verts_texture = harmonic_encoding.harmonic_encoding(verts_texture.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        deform_unet_inputs = torch.cat([neural_texture.detach(), enc_verts_texture.detach()], dim=1)

        uv_deformations_codes = unet_deformer(deform_unet_inputs)

        mlp_input_uv_z = F.grid_sample(uv_deformations_codes, verts_uvs, align_corners=False)[..., 0].permute(0, 2, 1)

        mlp_input_uv = F.grid_sample(self.uv_grid.repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2),
                                     verts_uvs, align_corners=False)[..., 0]
        mlp_input_uv = harmonic_encoding.harmonic_encoding(mlp_input_uv.permute(0, 2, 1), 6, )

        mlp_input_uv_deformations = torch.cat([mlp_input_uv_z, mlp_input_uv], dim=-1)

        if self.mask_for_face is None:
            self.mask_for_face = F.grid_sample((F.interpolate(self.uv_face_eye_mask.repeat(batch_size, 1, 1, 1)
                                                              , uv_deformations_codes.shape[-2:])),
                                               verts_uvs, align_corners=False)[..., 0].permute(0, 2, 1) > 0.5

        bs, v, ch = mlp_input_uv_deformations.shape
        deformation_project = vertex_deformer(mlp_input_uv_deformations.view(-1, ch))
        predefined_mask = None
        if predefined_mask is not None:
            deforms = torch.tanh(deformation_project.view(bs, -1, 3).contiguous())
            verts_empty_deforms = torch.zeros(batch_size, verts_uvs.shape[1], 3,
                                              dtype=verts_uvs.dtype, device=verts_uvs.device)
            verts_empty_deforms = verts_empty_deforms.scatter_(1, predefined_mask[None, :, None].expand(bs, -1, 3),
                                                               deforms)
            # self.deforms_mask.nonzero()[None].repeat(bs, 1, 1), deforms)
            verts_deforms = verts_empty_deforms
        else:
            verts_deforms = torch.tanh(deformation_project.view(bs, v, 3).contiguous())

        if self.mask_for_face is not None and self.external_params.get('deform_face_tightness', 0.0) > 0.0:
            #      We slightly deform areas along the face
            self.deforms_mask[self.mask_for_face[[0]]] = self.external_params.get('deform_face_tightness', 0.0)

        vert_texture_codes = torch.tanh(uv_deformations_codes[:, :3])
        vert_texture_coord_inp = torch.tanh(enc_verts_texture[:, :3])

        verts_deforms = verts_deforms * self.deforms_mask
        return verts_deforms, vert_texture_codes, vert_texture_coord_inp, uv_deformations_codes

    def add_details(self, target_codedict, verts, verts_target, uvs, shape_target):
        bs = verts.shape[0]
        uv_z = self.D_detail(
            torch.cat(
                [
                    target_codedict['pose_vec'][:, 3:],
                    target_codedict['exp'],
                    target_codedict['detail']
                ], dim=1)
        )

        vertex_normals = util.vertex_normals(verts, self.render.faces.expand(bs, -1, -1))
        uv_detail_normals = self.displacement2normal(uv_z, verts, vertex_normals)
        detail_normal_images = F.grid_sample(uv_detail_normals, uvs, align_corners=False)

        face_mask_deca = F.grid_sample(self.uv_face_mask.repeat_interleave(bs, dim=0), uvs)
        detail_shape_target = self.render.render_shape(verts, verts_target,
                                                       detail_normal_images=detail_normal_images)
        detail_shape_target = detail_shape_target * face_mask_deca + shape_target * (1 - face_mask_deca)
        return detail_shape_target

    def decode(self, target_codedict, neutral_pose,
               deformer_nets=None, verts_deforms=None, neural_texture=None):
        images = target_codedict['images']
        batch_size = images.shape[0]

        # Visualize shape
        default_cam = torch.zeros_like(target_codedict['cam'])[:, :3]  # default cam has orthogonal projection
        default_cam[:, :1] = 5.0

        cam_rot_mats, root_joint, verts_template, \
        shape_neutral_frontal, shape_parametric_frontal = self.get_parametric_vertices(target_codedict, neutral_pose)

        if deformer_nets['mlp_deformer']:
            verts_deforms, vert_texture_codes, \
            vert_texture_coord_inp, uv_deformations_codes = self.deform_source_mesh(verts_template, neural_texture, deformer_nets)

            # Obtain visualized frontal vertices
            faces = self.render.faces.expand(batch_size, -1, -1)

            vertex_normals = util.vertex_normals(verts_template, faces)
            verts_deforms = verts_deforms * vertex_normals

        verts_final = verts_template + verts_deforms
        verts_deforms_texture = self.render.world2uv(verts_deforms) * 5

        _, verts_final_frontal, _ = util.batch_orth_proj(verts_final, default_cam, flame=self.flame)
        shape_final_frontal = self.render.render_shape(verts_final, verts_final_frontal)

        verts_target_hair, verts_final_hair = None, None
        verts_final_neck, verts_target_neck = None, None
        soft_alphas_hair_only, soft_alphas_hair = None, None
        soft_alphas_neck_only, soft_alphas_neck = None, None
        self.detach_silhouettes = True

        use_deformations = True
        soft_alphas_detach_hair, soft_alphas_detach_neck = None, None
        # Obtain visualized frontal vertices

        if self.use_scalp_deforms or self.external_params.get('predict_hair_silh', False):
            verts_final_hair = verts_final.clone().detach()
            verts_final_hair_only = verts_final[:, self.hair_list].clone()
            # if not self.external_params.get('detach_neck', True):
            verts_final_hair[:, self.hair_list] = verts_final_hair_only

            _, verts_target_hair, _ = util.batch_orth_proj(
                verts_final_hair,
                target_codedict['cam'],
                root_joint,
                cam_rot_mats,
                self.flame
            )

        if self.use_neck_deforms or self.external_params.get('predict_hair_silh'):
            verts_final_neck = verts_final.clone()
            verts_final_neck[:, self.hair_list] = verts_final_neck[:, self.hair_list].detach()

            _, verts_target_neck, _ = util.batch_orth_proj(
                verts_final_neck,
                target_codedict['cam'],
                root_joint,
                cam_rot_mats,
                self.flame
            )

        if self.external_params.get('use_laplace_vector_coef', False):
            vertices_laplace_list = torch.ones_like(verts_final[..., 0]) * self.external_params.get(
                'laplacian_reg_weight', 0.0)
            vertices_laplace_list[:, self.hair_list] = self.external_params.get('laplacian_reg_hair_weight', 0.01)
            vertices_laplace_list[:, self.neck_list] = self.external_params.get('laplacian_reg_neck_weight', 10.0)

        # Project verts into target camera
        _, verts_target, landmarks_target = util.batch_orth_proj(
            verts_final.clone(),
            target_codedict['cam'],
            root_joint,
            cam_rot_mats,
            self.flame
        )
        shape_target = self.render.render_shape(verts_final, verts_target)

        with torch.no_grad():
            _, verts_final_posed, _ = util.batch_orth_proj(verts_final.clone(), default_cam, flame=self.flame)

            shape_final_posed = self.render.render_shape(verts_final, verts_final_posed)

        # Render and parse the outputs
        hair_neck_face_mesh_faces = torch.cat([self.faces_hair_mask, self.faces_neck_mask, self.faces_face_mask],
                                              dim=-1)

        ops = self.render(verts_final, verts_target, face_masks=hair_neck_face_mesh_faces)

        alphas = ops['alpha_images']
        soft_alphas = ops['soft_alpha_images']
        uvs = ops['uvcoords_images'].permute(0, 2, 3, 1)[..., :2]
        normals = ops['normal_images']
        coords = ops['vertice_images']

        if self.detach_silhouettes:
            verts_final_detach_hair = verts_final.clone()
            verts_final_detach_neck = verts_final.clone()

            verts_final_detach_hair[:, self.hair_list] = verts_final_detach_hair[:, self.hair_list].detach()
            verts_final_detach_neck[:, self.neck_list] = verts_final_detach_neck[:, self.neck_list].detach()

            verts_target_detach_hair = verts_target.clone()
            verts_target_detach_neck = verts_target.clone()

            verts_target_detach_hair[:, self.hair_list] = verts_target_detach_hair[:, self.hair_list].detach()
            verts_target_detach_neck[:, self.neck_list] = verts_target_detach_neck[:, self.neck_list].detach()

            ops_detach_hair = self.render(
                verts_final_detach_hair,
                verts_target_detach_hair,
                faces=self.faces_subdiv if self.subdivide_mesh else None,
                render_only_soft_silhouette=True,
            )

            ops_detach_neck = self.render(
                verts_final_detach_neck,
                verts_target_detach_neck,
                faces=self.faces_subdiv if self.subdivide_mesh else None,
                render_only_soft_silhouette=True,
            )

            soft_alphas_detach_hair = ops_detach_hair['soft_alpha_images']
            soft_alphas_detach_neck = ops_detach_neck['soft_alpha_images']

        verts_vis_mask = ops['vertices_visibility']
        verts_hair_vis_mask = ops['vertices_visibility'][:, self.hair_list]
        verts_neck_vis_mask = ops['vertices_visibility'][:, self.neck_list]
        hard_alphas_hair_only = (ops['area_alpha_images'][:, 0:1] == 1.0).float()
        hard_alphas_neck_only = (ops['area_alpha_images'][:, 1:2] == 1.0).float()
        hard_alphas_face_only = (ops['area_alpha_images'][:, 2:3] == 1.0).float()

        if self.use_scalp_deforms or self.external_params.get('predict_hair_silh'):
            ops_hair = self.render(
                    verts_final_hair[:, self.hair_edge_list],
                    verts_target_hair[:, self.hair_edge_list],
                    faces=self.hair_faces,
                    render_only_soft_silhouette=True,
            )

            soft_alphas_hair = ops_hair.get('soft_alpha_images')

        if self.use_neck_deforms or self.external_params.get('predict_hair_silh'):
            # Render whole
            ops_neck = self.render(
                    verts_final_neck[:, self.neck_edge_list],
                    verts_target_neck[:, self.neck_edge_list],
                    faces=self.neck_faces,
                    render_only_soft_silhouette=True,
            )

            soft_alphas_neck = ops_neck['soft_alpha_images']

            if self.external_params.get('use_whole_segmentation', False):
                soft_alphas_neck = soft_alphas

        # Grid sample outputs
        rendered_texture = None
        rendered_texture_detach_geom = None
        if neural_texture is not None:
            rendered_texture = F.grid_sample(neural_texture, uvs, mode='bilinear')
            rendered_texture_detach_geom = F.grid_sample(neural_texture, uvs.detach(), mode='bilinear')

        dense_vert_tensor = None
        dense_face_tensor = None
        dense_shape = None

        opdict = {
            'rendered_texture': rendered_texture,
            'rendered_texture_detach_geom': rendered_texture_detach_geom,
            'vertices': verts_final,
            'vertices_target': verts_target,
            'vertices_target_hair': verts_target_hair,
            'vertices_target_neck': verts_target_neck,
            'vertices_deforms': verts_deforms,
            'vertices_vis_mask': verts_vis_mask,
            'vertices_hair_vis_mask': verts_hair_vis_mask,
            'vertices_neck_vis_mask': verts_neck_vis_mask,
            'landmarks': landmarks_target,
            'alphas': alphas,
            'alpha_hair': None,
            'alpha_neck': None,
            'soft_alphas': soft_alphas,
            'soft_alphas_detach_hair': soft_alphas_detach_hair,
            'soft_alphas_detach_neck': soft_alphas_detach_neck,
            'soft_alphas_hair': soft_alphas_hair,
            'soft_alphas_neck': soft_alphas_neck,
            'hard_alphas_hair_only': hard_alphas_hair_only,
            'hard_alphas_neck_only': hard_alphas_neck_only,
            'hard_alphas_face_only': hard_alphas_face_only,
            'coords': coords,
            'normals': normals,
            'uvs': uvs,
            'source_uvs': None,
            'dense_verts': dense_vert_tensor,
            'dense_faces': dense_face_tensor,
            'dense_shape': dense_shape,
            'uv_deformations_codes': uv_deformations_codes,
            'source_warped_img': None
        }

        if self.use_tex:
            opdict['flametex_images'] = ops.get('images')

        if use_deformations:
            vert_texture_inp = torch.tanh(neural_texture[:, :3])

            opdict['deformations_out'] = verts_deforms_texture
            opdict['deformations_inp_texture'] = vert_texture_inp
            opdict['deformations_inp_coord'] = vert_texture_coord_inp
            opdict['deformations_inp_orig'] = vert_texture_codes
            opdict['vertices_laplace_list'] = None
            if self.external_params.get('fp_visualize_uv', True):
                opdict['face_parsing_uvs'] = None
            if self.neck_mask is not None:
                opdict['uv_neck_mask'] = self.render.world2uv(self.neck_mask).detach()[:, 0].repeat(batch_size, 1, 1, 1)

        visdict = {
            'shape_neutral_frontal_images': shape_neutral_frontal,
            'shape_parametric_frontal_images': shape_parametric_frontal,
            'shape_final_frontal_images': shape_final_frontal,
            'shape_final_posed_images': shape_final_posed,
            'shape_images': shape_target,
        }

        return opdict, visdict

    def encode_by_distill(self, target_image):
        delta_blendshapes = blend_shapes(self.hair_basis_reg(target_image),
                                         self.u_full.reshape(5023, 3, -1)) + self.mean_deforms
        return delta_blendshapes

    def forward(
            self,
            source_image,
            source_mask,
            source_keypoints,
            target_image,
            target_keypoints,
            neutral_pose=False,
            deformer_nets=None,
            neural_texture=None,
            source_information: dict = {},
    ) -> dict:
        source_image_crop, source_warp_to_crop, source_crop_bbox = self.preprocess_image(source_image, source_keypoints)
        target_image_crop, target_warp_to_crop, target_crop_bbox = self.preprocess_image(target_image, target_keypoints)

        if neural_texture is None:
            source_codedict = self.encode(source_image_crop, source_crop_bbox)
            source_information = {}
            source_information['shape'] = source_codedict['shape']
            source_information['codedict'] = source_codedict

        target_codedict = self.encode(target_image_crop, target_crop_bbox)

        target_codedict['shape'] = source_information.get('shape')
        target_codedict['batch_size'] = target_image.shape[0]
        delta_blendshapes = None

        if neural_texture is None:
            neural_texture = self.estimate_texture(source_image, source_mask, deformer_nets['neural_texture_encoder'])
            source_information['neural_texture'] = neural_texture

        if self.external_params['use_distill']:
            delta_blendshapes = self.encode_by_distill(source_image)

            if self.external_params.get('use_mobile_version', False):
                output2 = self.online_regressor(target_image)
                codedict_ = {}
                full_online = self.flame_config.model.n_exp + self.flame_config.model.n_pose + self.flame_config.model.n_cam
                codedict_['shape'] = target_codedict['shape']
                codedict_['batch_size'] = target_codedict['batch_size']
                codedict_['exp'] = output2[:, : self.flame_config.model.n_exp]
                codedict_['pose'] = output2[:,
                                    self.flame_config.model.n_exp: self.flame_config.model.n_exp + self.flame_config.model.n_pose]
                codedict_['cam'] = output2[:, full_online - self.flame_config.model.n_cam:full_online]
                pose = codedict_['pose'].view(codedict_['batch_size'], -1, 3)
                angle = torch.norm(pose + 1e-8, dim=2, keepdim=True)
                rot_dir = pose / angle
                codedict_['pose_rot_mats'] = batch_rodrigues(
                    torch.cat([angle, rot_dir], dim=2).view(-1, 4)
                ).view(codedict_['batch_size'], pose.shape[1], 3, 3)
                target_codedict = codedict_

            deformer_nets = {
                'unet_deformer': None,
                'mlp_deformer': None,
            }

        opdict, visdict = self.decode(
            target_codedict,
            neutral_pose,
            deformer_nets,
            neural_texture=neural_texture,
            verts_deforms=delta_blendshapes
        )

        posed_final_shapes = F.interpolate(visdict['shape_final_posed_images'], size=self.image_size, mode='bilinear')
        frontal_final_shapes = F.interpolate(visdict['shape_final_frontal_images'], size=self.image_size, mode='bilinear')
        frontal_parametric_shapes = F.interpolate(visdict['shape_parametric_frontal_images'], size=self.image_size, mode='bilinear')
        frontal_neutral_shapes = F.interpolate(visdict['shape_neutral_frontal_images'], size=self.image_size, mode='bilinear')

        outputs = {
            'rendered_texture' : opdict['rendered_texture'],
            'rendered_texture_detach_geom': opdict['rendered_texture_detach_geom'],
            'pred_target_coord': opdict['coords'],
            'pred_target_normal': opdict['normals'],
            'pred_target_uv': opdict['uvs'],
            'pred_source_coarse_uv': opdict['source_uvs'],
            'pred_target_shape_img': visdict['shape_images'],
            'pred_target_hard_mask': opdict['alphas'],
            'pred_target_hard_hair_mask': opdict['alpha_hair'],
            'pred_target_hard_neck_mask': opdict['alpha_neck'],
            'pred_target_soft_mask': opdict['soft_alphas'],
            'pred_target_soft_detach_hair_mask': opdict['soft_alphas_detach_hair'],
            'pred_target_soft_detach_neck_mask': opdict['soft_alphas_detach_neck'],
            'pred_target_soft_hair_mask': opdict['soft_alphas_hair'],
            'pred_target_soft_neck_mask': opdict['soft_alphas_neck'],
            'pred_target_hard_hair_only_mask':opdict['hard_alphas_hair_only'],
            'pred_target_hard_neck_only_mask':opdict['hard_alphas_neck_only'],
            'pred_target_hard_face_only_mask':opdict['hard_alphas_face_only'],
            'pred_target_keypoints': opdict['landmarks'],
            'vertices': opdict['vertices'],
            'vertices_target': opdict['vertices_target'],
            'vertices_target_hair':opdict['vertices_target_hair'],
            'vertices_target_neck':opdict['vertices_target_neck'],
            'vertices_deforms':opdict['vertices_deforms'],
            'vertices_vis_mask':opdict['vertices_vis_mask'],
            'vertices_hair_vis_mask':opdict['vertices_hair_vis_mask'],
            'vertices_neck_vis_mask':opdict['vertices_neck_vis_mask'],
            'target_shape_final_posed_img': posed_final_shapes,
            'target_shape_final_frontal_img': frontal_final_shapes,
            'target_shape_parametric_frontal_img': frontal_parametric_shapes,
            'target_shape_neutral_frontal_img': frontal_neutral_shapes,
            'deformations_out': opdict.get('deformations_out'),
            'deformations_inp_texture': opdict.get('deformations_inp_texture'),
            'deformations_inp_coord': opdict.get('deformations_inp_coord'),
            'deformations_inp_orig': opdict.get('deformations_inp_orig'),
            'laplace_coefs': opdict.get('vertices_laplace_list'),
            'target_render_face_mask': opdict.get('predicted_segmentation'),
            'uv_neck_mask': opdict.get('uv_neck_mask'),
            'target_visual_faceparsing': opdict.get('face_parsing_uvs'),
            'target_warp_to_crop': target_warp_to_crop,
            'dense_verts': opdict.get('dense_verts'),
            'dense_faces': opdict.get('dense_faces'),
            'flametex_images': opdict.get('flametex_images'),
            'dense_shape': opdict.get('dense_shape'),
            'uv_deformations_codes': opdict['uv_deformations_codes'],
            'source_warped_img': opdict['source_warped_img'],
            'source_image_crop': source_image_crop,
        }

        return outputs
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
from pytorch3d.structures import Meshes
import pickle as pkl

from DECA.decalib.utils import util
from DECA.decalib.models.lbs import blend_shapes, batch_rodrigues
from DECA.decalib.deca import DECA
import DECA.decalib.utils.config as config

from src.utils import harmonic_encoding
from src.utils.params import batch_cont2matrix
from src.utils.processing import create_regressor


class ParametricAvatar(DECA):
    def __init__(self,
                 image_size,
                 deca_path=None,
                 use_scalp_deforms=False,
                 use_neck_deforms=False,
                 subdivide_mesh=False,
                 use_details=False,
                 use_tex=False,
                 external_params=None,
                 device=torch.device('cpu'),
                 ):

        self.flame_config = cfg = config.cfg
        config.cfg.deca_dir = deca_path

        cfg.model.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template.obj')
        cfg.model.addfiles_path = external_params.rome_data_dir
        cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl')
        cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy')
        cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png')
        cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png')
        cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
        cfg.model.use_tex = use_tex
        cfg.dataset.image_size = image_size
        cfg.model.uv_size = image_size

        sys.path.append(os.path.join(deca_path, 'data'))
        super().__init__(cfg, device=device)

        self.device = device
        self.image_size = image_size
        self.use_scalp_deforms = use_scalp_deforms
        self.use_neck_deforms = use_neck_deforms
        self.subdivide_mesh = subdivide_mesh
        self.external_params = external_params.__dict__
        self.use_tex = use_tex
        self.use_details = use_details
        self.finetune_flame_encoder = False
        self.train_flame_encoder_from_scratch = False
        self.mask_for_face = None
        # Modify default FLAME config
        flame_config = cfg
        flame_config.model.uv_size = image_size

        self.cfg = flame_config

        grid_s = torch.linspace(0, 1, 224)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('identity_grid_deca', torch.stack([u, v], dim=2)[None], persistent=False)

        grid_s = torch.linspace(-1, 1, image_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('uv_grid', torch.stack([u, v], dim=2)[None])
        harmonized_uv_grid = harmonic_encoding.harmonic_encoding(self.uv_grid, num_encoding_functions=6)
        self.register_buffer('harmonized_uv_grid', harmonized_uv_grid[None])

        # Load scalp-related data
        external_params.data_dir = deca_path

        self.hair_list = pkl.load(open(f'{external_params.rome_data_dir}/hair_list.pkl', 'rb'))
        self.neck_list = pkl.load(open(f'{external_params.rome_data_dir}/neck_list.pkl', 'rb'))

        self.hair_edge_list = pkl.load(open(f'{external_params.rome_data_dir}/hair_edge_list.pkl', 'rb'))
        self.neck_edge_list = pkl.load(open(f'{external_params.rome_data_dir}/neck_edge_list.pkl', 'rb'))

        true_uvcoords = torch.load(f'{external_params.rome_data_dir}/vertex_uvcoords.pth')
        self.render = self.render.to(device)

        self.register_buffer('hair_faces', torch.load(f'{external_params.rome_data_dir}/hair_faces.pth'))
        self.register_buffer('neck_faces', torch.load(f'{external_params.rome_data_dir}/neck_faces.pth'))

        self.deforms_mask = torch.zeros(1, 5023, 1, device=device)

        self.hair_mask = torch.zeros(1, 5023, 1, device=device)
        self.neck_mask = torch.zeros(1, 5023, 1, device=device)
        self.face_mask = torch.zeros(1, 5023, 1, device=device)

        if self.use_scalp_deforms:
            self.deforms_mask[:, self.hair_list] = 1.0
        if self.use_neck_deforms:
            self.deforms_mask[:, self.neck_list] = 1.0

        self.hair_mask[:, self.hair_edge_list] = 1.0
        self.neck_mask[:, self.neck_edge_list] = 1.0
        self.true_uvcoords = true_uvcoords.to(device)

        def rm_from_list(a, b):
            return list(set(a) - set(b))

        # TODO save list to pickle
        hard_not_deform_list = [3587, 3594, 3595, 3598, 3600, 3630, 3634,
                                3635, 3636, 3637, 3643, 3644, 3646, 3649,
                                3650, 3652, 3673, 3676, 3677, 3678, 3679,
                                3680, 3681, 3685, 3691, 3693, 3695, 3697,
                                3698, 3701, 3703, 3707, 3709, 3713, 3371,
                                3372, 3373, 3374, 3375, 3376, 3377, 3378,
                                3379, 3382, 3383, 3385, 3387, 3389, 3392,
                                3393, 3395, 3397, 3399, 3413, 3414, 3415,
                                3416, 3417, 3418, 3419, 3420, 3421, 3422,
                                3423, 3424, 3441, 3442, 3443, 3444, 3445,
                                3446, 3447, 3448, 3449, 3450, 3451, 3452,
                                3453, 3454, 3455, 3456, 3457, 3458, 3459,
                                3460, 3461, 3462, 3463, 3494, 3496, 3510,
                                3544, 3562, 3578, 3579, 3581, 3583]
        exclude_list = [3382, 3377, 3378, 3379, 3375, 3374, 3544, 3494, 3496,
                        3462, 3463, 3713, 3510, 3562, 3372, 3373, 3376, 3371]

        hard_not_deform_list = list(rm_from_list(hard_not_deform_list, exclude_list))

        # if self.use_neck_deforms and self.external_params.get('updated_neck_mask', False):
        self.deforms_mask[:, hard_not_deform_list] = 0.0
        self.face_mask[:, self.hair_edge_list] = 0.0
        self.face_mask[:, self.neck_edge_list] = 0.0

        self.register_buffer('faces_hair_mask', util.face_vertices(self.hair_mask, self.render.faces))
        self.register_buffer('faces_neck_mask', util.face_vertices(self.neck_mask, self.render.faces))
        self.register_buffer('faces_face_mask', util.face_vertices(self.face_mask, self.render.faces))

        if self.external_params.get('deform_face_scale_coef', 0.0) > 0.0:
            self.face_deforms_mask = torch.ones_like(self.deforms_mask).cpu() / \
                                     self.external_params.get('deform_face_scale_coef')
            self.face_deforms_mask[:, self.neck_list] = 1.0
            self.face_deforms_mask[:, self.hair_list] = 1.0

            if self.external_params.get('deform_face'):
                # put original deformation ofr face zone, scaling applied only for ears & eyes
                verts_uvs = self.true_uvcoords
                face_vertices = F.grid_sample(self.uv_face_mask, verts_uvs[None]).squeeze() > 0.0
                self.face_deforms_mask[:, face_vertices] = 1.0
        else:
            self.face_deforms_mask = None

        # Create distill model
        if self.external_params.get('use_distill', False):
            self._setup_linear_model()

    def _setup_linear_model(self):
        n_online_params = self.flame_config.model.n_exp + self.flame_config.model.n_pose + self.flame_config.model.n_cam
        self.hair_basis_reg = create_regressor('resnet50', self.external_params.get('n_scalp', 60))
        state_dict = torch.load(self.external_params['path_to_linear_hair_model'], map_location='cpu')

        self.hair_basis_reg.load_state_dict(state_dict)
        self.hair_basis_reg.eval()

        if self.external_params.get('use_mobile_version', False):
            self.online_regressor = create_regressor('mobilenet_v2', n_online_params)
            state_dict = torch.load(self.external_params['path_to_mobile_model'], map_location='cpu')

            self.online_regressor.load_state_dict(state_dict)
            self.online_regressor.eval()

        # Load basis
        self.u_full = torch.load(os.path.join(f'{self.external_params["rome_data_dir"]}',
                                              'u_full.pt'), map_location='cpu').to(self.device)
        # Create mean deforms
        self.mean_deforms = torch.load(
            os.path.join(f'{self.external_params["rome_data_dir"]}', 'mean_deform.pt'),
            map_location='cpu').to(self.device)

        self.mean_hair = torch.zeros(5023, 3, device=self.mean_deforms.device)
        self.mean_neck = torch.zeros(5023, 3, device=self.mean_deforms.device)
        self.mean_hair[self.hair_list] = self.mean_deforms[self.hair_list]
        self.mean_neck[self.neck_list] = self.mean_deforms[self.neck_list]

    @staticmethod
    def calc_crop(l_old, r_old, t_old, b_old, scale):
        size = (r_old - l_old + b_old - t_old) / 2 * 1.1
        size *= scale

        center = torch.stack([r_old - (r_old - l_old) / 2.0,
                              b_old - (b_old - t_old) / 2.0], dim=1)

        l_new = center[:, 0] - size / 2
        r_new = center[:, 0] + size / 2
        t_new = center[:, 1] - size / 2
        b_new = center[:, 1] + size / 2

        l_new = l_new[:, None, None]
        r_new = r_new[:, None, None]
        t_new = t_new[:, None, None]
        b_new = b_new[:, None, None]

        return l_new, r_new, t_new, b_new

    def preprocess_image(self, image, keypoints) -> torch.Tensor:
        old_size = image.shape[2]

        keypoints = (keypoints + 1) / 2

        l_old = torch.min(keypoints[..., 0], dim=1)[0]
        r_old = torch.max(keypoints[..., 0], dim=1)[0]
        t_old = torch.min(keypoints[..., 1], dim=1)[0]
        b_old = torch.max(keypoints[..., 1], dim=1)[0]

        l_new, r_new, t_new, b_new = self.calc_crop(l_old, r_old, t_old, b_old, scale=1.25)

        warp_to_crop = self.identity_grid_deca.clone().repeat_interleave(image.shape[0], dim=0)

        warp_to_crop[..., 0] = warp_to_crop[..., 0] * (r_new - l_new) + l_new
        warp_to_crop[..., 1] = warp_to_crop[..., 1] * (b_new - t_new) + t_new
        warp_to_crop = (warp_to_crop - 0.5) * 2

        if not hasattr(self, 'identity_grid_input'):
            grid_s = torch.linspace(0, 1, old_size)
            v, u = torch.meshgrid(grid_s, grid_s)
            device = warp_to_crop.device
            dtype = warp_to_crop.type()
            self.register_buffer('identity_grid_input', torch.stack([u, v], dim=2)[None].cpu().type(dtype),
                                 persistent=False)

        crop_bbox = [l_new[..., 0], t_new[..., 0], r_new[..., 0], b_new[..., 0]]

        return F.grid_sample(image, warp_to_crop.float()), warp_to_crop.float(), crop_bbox

    def encode(self, images, crop_bbox):
        batch_size = images.shape[0]

        if self.finetune_flame_encoder or self.train_flame_encoder_from_scratch:
            e_flame_code_parameters = self.E_flame(images)
        else:
            with torch.no_grad():
                e_flame_code_parameters = self.E_flame(images)

        if not self.train_flame_encoder_from_scratch:
            codedict = self.decompose_code(e_flame_code_parameters, self.param_dict)
        else:
            codedict = e_flame_code_parameters

        codedict['images'] = images

        codedict['pose_vec'] = codedict['pose']

        pose = codedict['pose'].view(batch_size, -1, 3)
        angle = torch.norm(pose + 1e-8, dim=2, keepdim=True)
        rot_dir = pose / angle
        codedict['pose_rot_mats'] = batch_rodrigues(
            torch.cat([angle, rot_dir], dim=2).view(-1, 4)
        ).view(batch_size, pose.shape[1], 3, 3)  # cam & jaw | jaw | jaw & eyes

        if 'cont_pose' in codedict.keys():
            pose_rot_mats = batch_cont2matrix(codedict['cont_pose'])
            codedict['pose_rot_mats'] = torch.cat([pose_rot_mats, codedict['pose_rot_mats']], dim=1)  # cam | cam & neck

        if self.use_details:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode

        # Modify camera params to include uncrop using crop_bbox
        if crop_bbox is not None:
            crop_w = crop_bbox[2] - crop_bbox[0]
            crop_h = crop_bbox[3] - crop_bbox[1]
            crop_s = (crop_w + crop_h) / 2
            crop_s = crop_s[:, 0]

            cam = codedict['cam']
            scale_orig = cam[..., 0]
            scale_crop = scale_orig * crop_s
            crop_y = cam[..., 1] + 1 / scale_orig + (2 * crop_bbox[0][:, 0] - 1) / scale_crop
            crop_z = cam[..., 2] - 1 / scale_orig - (2 * crop_bbox[1][:, 0] - 1) / scale_crop
            cam_crop = torch.stack([scale_crop, crop_y, crop_z], dim=-1)
            codedict['cam'] = cam_crop

        return codedict

    def get_parametric_vertices(self, codedict, neutral_pose):
        cam_rot_mats = codedict['pose_rot_mats'][:, :1]
        batch_size = cam_rot_mats.shape[0]

        eye_rot_mats = neck_rot_mats = None

        if codedict['pose_rot_mats'].shape[1] >= 3:
            neck_rot_mats = codedict['pose_rot_mats'][:, 1:2]
            jaw_rot_mats = codedict['pose_rot_mats'][:, 2:3]
        else:
            jaw_rot_mats = codedict['pose_rot_mats'][:, 1:2]

        if codedict['pose_rot_mats'].shape[1] == 4:
            eye_rot_mats = codedict['pose_rot_mats'][:, 3:]

        # Use zero global camera pose inside FLAME fitting class
        cam_rot_mats_ = torch.eye(3).to(cam_rot_mats.device).expand(batch_size, 1, 3, 3)
        # Shaped vertices
        verts_neutral = self.flame.reconstruct_shape(codedict['shape'])

        # Visualize shape
        default_cam = torch.zeros_like(codedict['cam'])[:, :3]  # default cam has orthogonal projection
        default_cam[:, :1] = 5.0

        _, verts_neutral_frontal, _ = util.batch_orth_proj(verts_neutral, default_cam, flame=self.flame)
        shape_neutral_frontal = self.render.render_shape(verts_neutral, verts_neutral_frontal)

        # Apply expression and pose
        if neutral_pose:
            verts_parametric, rot_mats, root_joint, neck_joint = self.flame.reconstruct_exp_and_pose(
                verts_neutral, torch.zeros_like(codedict['exp']))
        else:
            verts_parametric, rot_mats, root_joint, neck_joint = self.flame.reconstruct_exp_and_pose(
                verts_neutral,
                codedict['exp'],
                cam_rot_mats_,
                neck_rot_mats,
                jaw_rot_mats,
                eye_rot_mats
            )

        # Add neck rotation
        if neck_rot_mats is not None:
            neck_rot_mats = neck_rot_mats.repeat_interleave(verts_parametric.shape[1], dim=1)
            verts_parametric = verts_parametric - neck_joint[:, None]
            verts_parametric = torch.matmul(neck_rot_mats.transpose(2, 3), verts_parametric[..., None])[..., 0]
            verts_parametric = verts_parametric + neck_joint[:, None]

        # Visualize exp verts
        _, verts_parametric_frontal, _ = util.batch_orth_proj(verts_parametric, default_cam, flame=self.flame)
        shape_parametric_frontal = self.render.render_shape(verts_parametric, verts_parametric_frontal)

        return cam_rot_mats, root_joint, verts_parametric, shape_neutral_frontal, shape_parametric_frontal

    def estimate_texture(self, source_image: torch.Tensor, source_mask: torch.Tensor,
                         texture_encoder: torch.nn.Module) -> torch.Tensor:
        autoenc_inputs = torch.cat([source_image, source_mask], dim=1)
        neural_texture = texture_encoder(autoenc_inputs)
        if neural_texture.shape[-1] != 256:
            neural_texture = F.interpolate(neural_texture, (256, 256))

        return neural_texture

    @torch.no_grad()
    def deform_source_mesh(self, verts_parametric, neural_texture, deformer_nets):
        unet_deformer = deformer_nets['unet_deformer']
        vertex_deformer = deformer_nets['mlp_deformer']
        batch_size = verts_parametric.shape[0]

        verts_uvs = self.true_uvcoords[:, :, None, :2]  # 1 x V x 1 x 2

        verts_uvs = verts_uvs.repeat_interleave(batch_size, dim=0)

        # bs x 3 x H x W
        verts_texture = self.render.world2uv(verts_parametric) * 5

        enc_verts_texture = harmonic_encoding.harmonic_encoding(verts_texture.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        deform_unet_inputs = torch.cat([neural_texture.detach(), enc_verts_texture.detach()], dim=1)

        uv_deformations_codes = unet_deformer(deform_unet_inputs)

        mlp_input_uv_z = F.grid_sample(uv_deformations_codes, verts_uvs, align_corners=False)[..., 0].permute(0, 2, 1)

        mlp_input_uv = F.grid_sample(self.uv_grid.repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2),
                                     verts_uvs, align_corners=False)[..., 0]
        mlp_input_uv = harmonic_encoding.harmonic_encoding(mlp_input_uv.permute(0, 2, 1), 6, )

        mlp_input_uv_deformations = torch.cat([mlp_input_uv_z, mlp_input_uv], dim=-1)

        if self.mask_for_face is None:
            self.mask_for_face = F.grid_sample((F.interpolate(self.uv_face_eye_mask.repeat(batch_size, 1, 1, 1)
                                                              , uv_deformations_codes.shape[-2:])),
                                               verts_uvs, align_corners=False)[..., 0].permute(0, 2, 1) > 0.5

        bs, v, ch = mlp_input_uv_deformations.shape
        deformation_project = vertex_deformer(mlp_input_uv_deformations.view(-1, ch))
        predefined_mask = None
        if predefined_mask is not None:
            deforms = torch.tanh(deformation_project.view(bs, -1, 3).contiguous())
            verts_empty_deforms = torch.zeros(batch_size, verts_uvs.shape[1], 3,
                                              dtype=verts_uvs.dtype, device=verts_uvs.device)
            verts_empty_deforms = verts_empty_deforms.scatter_(1, predefined_mask[None, :, None].expand(bs, -1, 3),
                                                               deforms)
            # self.deforms_mask.nonzero()[None].repeat(bs, 1, 1), deforms)
            verts_deforms = verts_empty_deforms
        else:
            verts_deforms = torch.tanh(deformation_project.view(bs, v, 3).contiguous())

        if self.mask_for_face is not None and self.external_params.get('deform_face_tightness', 0.0) > 0.0:
            #      We slightly deform areas along the face
            self.deforms_mask[self.mask_for_face[[0]]] = self.external_params.get('deform_face_tightness', 0.0)

        verts_deforms = verts_deforms * self.deforms_mask
        return verts_deforms

    def add_details(self, target_codedict, verts, verts_target, uvs, shape_target):
        bs = verts.shape[0]
        uv_z = self.D_detail(
            torch.cat(
                [
                    target_codedict['pose_vec'][:, 3:],
                    target_codedict['exp'],
                    target_codedict['detail']
                ], dim=1)
        )

        vertex_normals = util.vertex_normals(verts, self.render.faces.expand(bs, -1, -1))
        uv_detail_normals = self.displacement2normal(uv_z, verts, vertex_normals)
        detail_normal_images = F.grid_sample(uv_detail_normals, uvs, align_corners=False)

        face_mask_deca = F.grid_sample(self.uv_face_mask.repeat_interleave(bs, dim=0), uvs)
        detail_shape_target = self.render.render_shape(verts, verts_target,
                                                       detail_normal_images=detail_normal_images)
        detail_shape_target = detail_shape_target * face_mask_deca + shape_target * (1 - face_mask_deca)
        return detail_shape_target

    @torch.no_grad()
    def decode(self, target_codedict, neutral_pose,
               deformer_nets=None, verts_deforms=None, neural_texture=None):
        batch_size = target_codedict['batch_size']

        # Visualize shape
        default_cam = torch.zeros_like(target_codedict['cam'])[:, :3]  # default cam has orthogonal projection
        default_cam[:, :1] = 5.0

        cam_rot_mats, root_joint, verts_template, \
        shape_neutral_frontal, shape_parametric_frontal = self.get_parametric_vertices(target_codedict, neutral_pose)

        if deformer_nets['mlp_deformer']:
            verts_deforms = self.deform_source_mesh(verts_template, neural_texture, deformer_nets)

            # Obtain visualized frontal vertices
            faces = self.render.faces.expand(batch_size, -1, -1)

            vertex_normals = util.vertex_normals(verts_template, faces)
            verts_deforms = verts_deforms * vertex_normals

        verts_final = verts_template + verts_deforms

        _, verts_final_frontal, _ = util.batch_orth_proj(verts_final, default_cam, flame=self.flame)
        shape_final_frontal = self.render.render_shape(verts_final, verts_final_frontal)

        _, verts_target, landmarks_target = util.batch_orth_proj(
            verts_final.clone(), target_codedict['cam'],
            root_joint, cam_rot_mats, self.flame)

        shape_target = self.render.render_shape(verts_final, verts_target)
        _, verts_final_posed, _ = util.batch_orth_proj(verts_final.clone(), default_cam, flame=self.flame)
        shape_final_posed = self.render.render_shape(verts_final, verts_final_posed)
        hair_neck_face_mesh_faces = torch.cat([self.faces_hair_mask, self.faces_neck_mask, self.faces_face_mask],
                                              dim=-1)

        ops = self.render(verts_final, verts_target, face_masks=hair_neck_face_mesh_faces)

        alphas = ops['alpha_images']
        soft_alphas = ops['soft_alpha_images']
        uvs = ops['uvcoords_images'].permute(0, 2, 3, 1)[..., :2]
        normals = ops['normal_images']
        coords = ops['vertice_images']

        # Grid sample outputs
        rendered_texture = None
        if neural_texture is not None:
            rendered_texture = F.grid_sample(neural_texture, uvs, mode='bilinear')

        if self.use_details:
            detail_shape_target = self.add_details(target_codedict, verts_final, verts_target, uvs, shape_target)

        opdict = {
            'rendered_texture': rendered_texture,
            'rendered_texture_detach_geom': None,
            'vertices': verts_final,
            'vertices_target': verts_target,
            'vertices_deforms': verts_deforms,
            'landmarks': landmarks_target,
            'alphas': alphas,
            'coords': coords,
            'normals': normals,
            'neural_texture': neural_texture,
        }

        if self.use_tex:
            opdict['flametex_images'] = ops.get('images')

        visdict = {
            'shape_neutral_frontal_images': shape_neutral_frontal,
            'shape_parametric_frontal_images': shape_parametric_frontal,
            'shape_final_frontal_images': shape_final_frontal,
            'shape_final_posed_images': shape_final_posed,
            'shape_images': shape_target,
            'shape_target_displ_images': detail_shape_target if self.use_details else None
        }
        for k, v in visdict.items():
            if v is None: continue
            visdict[k] = F.interpolate(v, size=self.image_size, mode='bilinear')

        return opdict, visdict

    def encode_by_distill(self, target_image):
        delta_blendshapes = blend_shapes(self.hair_basis_reg(target_image),
                                         self.u_full.reshape(5023, 3, -1)) + self.mean_deforms
        return delta_blendshapes

    def forward(
            self,
            source_image,
            source_mask,
            source_keypoints,
            target_image,
            target_keypoints,
            neutral_pose=False,
            deformer_nets=None,
            neural_texture=None,
            source_information: dict = {},
    ) -> dict:
        source_image_crop, _, source_crop_bbox = self.preprocess_image(source_image, source_keypoints)
        target_image_crop, _, target_crop_bbox = self.preprocess_image(target_image, target_keypoints)

        if neural_texture is None:
            source_codedict = self.encode(source_image_crop, source_crop_bbox)
            source_information = {}
            source_information['shape'] = source_codedict['shape']
            source_information['codedict'] = source_codedict

        target_codedict = self.encode(target_image_crop, target_crop_bbox)

        target_codedict['shape'] = source_information.get('shape')
        target_codedict['batch_size'] = target_image.shape[0]
        delta_blendshapes = None

        if neural_texture is None:
            neural_texture = self.estimate_texture(source_image, source_mask, deformer_nets['neural_texture_encoder'])
            source_information['neural_texture'] = neural_texture

        if self.external_params['use_distill']:
            delta_blendshapes = self.encode_by_distill(source_image)

            if self.external_params.get('use_mobile_version', False):
                output2 = self.online_regressor(target_image)
                codedict_ = {}
                full_online = self.flame_config.model.n_exp + self.flame_config.model.n_pose + self.flame_config.model.n_cam
                codedict_['shape'] = target_codedict['shape']
                codedict_['batch_size'] = target_codedict['batch_size']
                codedict_['exp'] = output2[:, : self.flame_config.model.n_exp]
                codedict_['pose'] = output2[:,
                                    self.flame_config.model.n_exp: self.flame_config.model.n_exp + self.flame_config.model.n_pose]
                codedict_['cam'] = output2[:, full_online - self.flame_config.model.n_cam:full_online]
                pose = codedict_['pose'].view(codedict_['batch_size'], -1, 3)
                angle = torch.norm(pose + 1e-8, dim=2, keepdim=True)
                rot_dir = pose / angle
                codedict_['pose_rot_mats'] = batch_rodrigues(
                    torch.cat([angle, rot_dir], dim=2).view(-1, 4)
                ).view(codedict_['batch_size'], pose.shape[1], 3, 3)
                target_codedict = codedict_

            deformer_nets = {
                'unet_deformer': None,
                'mlp_deformer': None,
            }

        opdict, visdict = self.decode(
            target_codedict,
            neutral_pose,
            deformer_nets,
            neural_texture=neural_texture,
            verts_deforms=delta_blendshapes
        )

        outputs = {
            'rendered_texture': opdict['rendered_texture'],
            'source_neural_texture': opdict['neural_texture'],
            'pred_target_normal': opdict['normals'],
            'pred_target_shape_img': visdict['shape_images'],
            'pred_target_shape_displ_img': visdict.get('shape_target_displ_images'),
            'pred_target_keypoints': opdict['landmarks'],
            'vertices': opdict['vertices'],
            'pred_target_hard_mask': opdict['alphas'],
            'vertices_target': opdict['vertices_target'],
            'target_shape_final_posed_img': visdict['shape_final_posed_images'],
            'target_shape_final_frontal_img': visdict['shape_final_frontal_images'],
            'target_shape_parametric_frontal_img': visdict['shape_parametric_frontal_images'],
            'target_shape_neutral_frontal_img': visdict['shape_neutral_frontal_images'],
            'source_information': source_information,
        }

        return outputs
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from argparse import ArgumentParser
from pytorch3d.structures import Meshes

from src.networks.face_parsing import FaceParsing
from src.parametric_avatar_trainable import ParametricAvatarTrainable
from src.rome import ROME
from src.utils import args as args_utils
from src.utils import harmonic_encoding
from src.utils.visuals import mask_errosion
from src.losses import *
from src.networks import MultiScaleDiscriminator
from src.utils import misc, spectral_norm, weight_init


class TrainableROME(ROME):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_source_frames = args.num_source_frames
        self.num_target_frames = args.num_target_frames

        self.weights = {
            'adversarial': args.adversarial_weight,
            'feature_matching': args.feature_matching_weight,
            'l1': args.l1_weight,
            'vgg19': args.vgg19_weight,
            'vggface': args.vggface_weight,
            'vgggaze': args.vgggaze_weight,
            'unet_seg': args.unet_seg_weight,
            'seg': args.seg_weight,
            'seg_hair': args.seg_hair_weight,
            'seg_neck': args.seg_neck_weight,
            'seg_hard': args.seg_hard_weight,
            'seg_hard_neck': args.seg_hard_neck_weight,
            'chamfer': args.chamfer_weight,
            'chamfer_hair': args.chamfer_hair_weight,
            'chamfer_neck': args.chamfer_neck_weight,
            'keypoints_matching': args.keypoints_matching_weight,
            'eye_closure': args.eye_closure_weight,
            'lip_closure': args.lip_closure_weight,
            'shape_reg': args.shape_reg_weight,
            'exp_reg': args.exp_reg_weight,
            'tex_reg': args.tex_reg_weight,
            'light_reg': args.light_reg_weight,
            'laplacian_reg': args.laplacian_reg_weight,
            'edge_reg': args.edge_reg_weight,
            'l1_hair': args.l1_hair_weight,
            'repulsion_hair': args.repulsion_hair_weight,
            'repulsion': args.repulsion_weight,
            'normal_reg': args.normal_reg_weight}

        self.init_networks(args)
        self.discriminator = None
        self.init_losses(args)

        self.fp_masks_setup = ['cloth_neck', 'hair_face_ears', 'neck_cloth_face']
        self.face_parsing = FaceParsing(
            args.face_parsing_path,
            device='cuda' if args.num_gpus else 'cpu'
        )

        self.parametric_avatar = ParametricAvatarTrainable(
            args.model_image_size,
            args.deca_path,
            args.use_scalp_deforms,
            args.use_neck_deforms,
            args.subdivide_mesh,
            args.use_deca_details,
            args.use_flametex,
            args,
            device=args.device,
        )
        if args.adversarial_weight > 0:
            self.init_disc(args)

    def init_disc(self, args):
        if args.spn_layers:
            spn_layers = args_utils.parse_str_to_list(args.spn_layers, sep=',')

        self.discriminator = MultiScaleDiscriminator(
            min_channels=args.dis_num_channels,
            max_channels=args.dis_max_channels,
            num_blocks=args.dis_num_blocks,
            input_channels=3,
            input_size=args.model_image_size,
            num_scales=args.dis_num_scales)

        self.discriminator.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))
        if args.spn_apply_to_dis:
            self.discriminator.apply(lambda module: spectral_norm.apply_spectral_norm(module, apply_to=spn_layers))

        if args.deferred_neural_rendering_path and not args.reset_dis_weights:
            state_dict_full = torch.load(args.deferred_neural_rendering_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in state_dict_full.items():
                if 'discriminator' in k:
                    state_dict[k.replace(f'discriminator.', '')] = v
            self.discriminator.load_state_dict(state_dict)
            print('Loaded discriminator state dict')

    def init_losses(self, args):
        if self.weights['adversarial']:
            self.adversarial_loss = AdversarialLoss()

        if self.weights['feature_matching']:
            self.feature_matching_loss = FeatureMatchingLoss()

        if self.weights['vgg19']:
            class PerceptualLossWrapper(object):
                def __init__(self, num_scales, use_gpu, use_fp16):
                    self.loss = PerceptualLoss(
                        num_scales=num_scales,
                        use_fp16=use_fp16)

                    if use_gpu:
                        self.loss = self.loss.cuda()

                    self.forward = self.loss.forward

            self.vgg19_loss = PerceptualLossWrapper(
                num_scales=args.vgg19_num_scales,
                use_gpu=args.num_gpus > 0,
                use_fp16=True if args.use_amp and args.amp_opt_level != 'O0' else False)

        if self.weights['vggface']:
            self.vggface_loss = VGGFace2Loss(pretrained_model=args.vggface_path,
                                             device='cuda' if args.num_gpus else 'cpu')

        if self.weights['vgggaze']:
            self.vgggaze_loss = GazeLoss('cuda' if args.num_gpus else 'cpu')

        if self.weights['keypoints_matching']:
            self.keypoints_matching_loss = KeypointsMatchingLoss()

        if self.weights['eye_closure']:
            self.eye_closure_loss = EyeClosureLoss()

        if self.weights['lip_closure']:
            self.lip_closure_loss = LipClosureLoss()

        if self.weights['unet_seg']:
            self.unet_seg_loss = SegmentationLoss(loss_type=args.unet_seg_type)

        if (self.weights['seg'] or self.weights['seg_hair'] or self.weights['seg_neck'] or
                self.weights['seg_hard']):
            self.seg_loss = MultiScaleSilhouetteLoss(num_scales=args.seg_num_scales, loss_type=args.seg_type)

        if self.weights['chamfer'] or self.weights['chamfer_hair'] or self.weights['chamfer_neck']:
            self.chamfer_loss = ChamferSilhouetteLoss(
                args.chamfer_num_neighbours,
                args.chamfer_same_num_points,
                args.chamfer_sample_outside_of_silhouette
            )

        if self.weights['laplacian_reg']:
            self.laplace_loss = LaplaceMeshLoss(args.laplace_reg_type)
        if self.args.predict_face_parsing_mask:
            self.face_parsing_loss = nn.CrossEntropyLoss(ignore_index=255,
                                                         reduction='mean')

        self.ssim = SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.psnr = PSNR()
        self.lpips = LPIPS()

    def train(self, mode=False):
        if self.args.train_deferred_neural_rendering:
            self.autoencoder.train(mode)
            self.unet.train(mode)
            if self.args.unet_pred_mask and self.args.use_separate_seg_unet:
                self.unet_seg.train(mode)
            if self.discriminator is not None:
                self.discriminator.train(mode)

        elif self.args.train_texture_encoder:
            self.autoencoder.train(mode)

        if self.args.use_mesh_deformations:
            if self.args.use_unet_deformer:
                self.mesh_deformer.train(mode)
            if self.args.use_mlp_deformer:
                self.mlp_deformer.train(mode)
            if self.args.use_basis_deformer:
                self.basis_deformer.train(mode)

    def _forward(self, data_dict):
        deca_results = self.parametric_avatar.forward(
            data_dict['source_img'],
            data_dict['source_mask'],
            data_dict['source_keypoints'],
            data_dict['target_img'],
            data_dict['target_keypoints'],
            deformer_nets={
                'neural_texture_encoder': self.autoencoder,
                'unet_deformer': self.mesh_deformer,
                'mlp_deformer': self.mlp_deformer,
                'basis_deformer': self.basis_deformer,
            },
        )

        rendered_texture = deca_results['rendered_texture']
        rendered_texture_detach_geom = deca_results.pop('rendered_texture_detach_geom')

        for key, value in deca_results.items():
            data_dict[key] = value

        if self.args.predict_face_parsing_mask:
            target_fp_mask = self.face_parsing.forward(data_dict['target_img'])  # face, ears, neck, cloth, hair
            data_dict['target_hair_mask'] = (
                                                    target_fp_mask[:, 0] +
                                                    target_fp_mask[:, 1] +
                                                    target_fp_mask[:, 4]
                                            )[:, None]

            if self.args.include_neck_to_hair_mask:
                data_dict['target_hair_mask'] += target_fp_mask[:, 2][:, None]

            data_dict['target_face_mask'] = target_fp_mask[:, 0][:, None]
            data_dict['target_hair_only_mask'] = target_fp_mask[:, 4][:, None]

            neck_mask = target_fp_mask[:, 0] + target_fp_mask[:, 1] + \
                        target_fp_mask[:, 2] + target_fp_mask[:, 3] + \
                        target_fp_mask[:, 4]

            data_dict['target_neck_mask'] = neck_mask[:, None]

            data_dict['target_neck_only_mask'] = (
                                                         target_fp_mask[:, 2] +
                                                         target_fp_mask[:, 3]
                                                 )[:, None]

        if self.args.use_graphonomy_mask:
            graphonomy_mask = self.graphonomy.forward(data_dict['target_img'])
            data_dict['target_face_mask'] = graphonomy_mask[:, 1:2]

        if self.unet is not None:
            unet_inputs = rendered_texture * data_dict['pred_target_hard_mask']
            # hard mask the rendered texture to make consistent padding

            if self.args.unet_use_normals_cond:
                normals = data_dict['pred_target_normal'].permute(0, 2, 3, 1)
                normal_inputs = harmonic_encoding.harmonic_encoding(normals,
                                                                    self.args.num_harmonic_encoding_funcs).permute(0, 3,
                                                                                                                   1, 2)
                unet_inputs = torch.cat([unet_inputs, normal_inputs], dim=1)
                if self.args.reg_positive_z_normals:
                    data_dict['normals_z'] = normals[..., [-1]]
                if self.args.mask_according_to_normal:
                    normal_z_mask = normals[..., [-1]] > -0.3
                    unet_inputs = normal_z_mask.permute(0, 3, 1, 2) * unet_inputs
                    normal_z_mask = normals[..., [-1]] > -0.2
                    data_dict['pred_target_soft_detach_hair_mask'] = data_dict[
                                                                         'pred_target_soft_detach_hair_mask'] * normal_z_mask.permute(
                        0, 3, 1, 2)
                    data_dict['pred_target_soft_neck_mask'] = data_dict[
                                                                  'pred_target_soft_neck_mask'] * normal_z_mask.permute(
                        0, 3, 1, 2)

            if self.args.unet_use_uvs_cond:
                uvs = data_dict['pred_target_uv'][..., :2]
                uvs_inputs = harmonic_encoding.harmonic_encoding(uvs, self.args.num_harmonic_encoding_funcs).permute(0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     2)
                unet_inputs = torch.cat([unet_inputs, uvs_inputs], dim=1)

            if self.args.use_separate_seg_unet:
                data_dict['pred_target_img'] = torch.sigmoid(self.unet(unet_inputs))

                if self.args.unet_pred_mask:
                    data_dict['pred_target_unet_logits'] = self.unet_seg(unet_inputs)
                    data_dict['pred_target_unet_mask'] = torch.sigmoid(data_dict['pred_target_unet_logits']).detach()

            else:
                unet_outputs = self.unet(unet_inputs)

                data_dict['pred_target_img'] = torch.sigmoid(unet_outputs[:, :3])

                if self.args.unet_pred_mask:
                    data_dict['pred_target_unet_logits'] = unet_outputs[:, 3:]
                    data_dict['pred_target_unet_mask'] = torch.sigmoid(data_dict['pred_target_unet_logits']).detach()

            if self.args.adv_only_for_rendering:
                unet_inputs = rendered_texture_detach_geom * data_dict[
                    'pred_target_hard_mask']  # hard mask the rendered texture to make consistent padding
                if self.args.unet_use_normals_cond:
                    unet_inputs = torch.cat([unet_inputs, normal_inputs.detach()], dim=1)

                data_dict['pred_target_img_detach_geom'] = torch.sigmoid(self.unet(unet_inputs)[:, :3])

        else:
            data_dict['pred_target_img'] = data_dict['flametex_images']

        if self.args.train_only_face:
            data_dict['pred_target_img'] = data_dict['pred_target_img'] * data_dict['target_face_mask']
            data_dict['target_img'] = data_dict['target_img'] * data_dict['target_face_mask']

        if self.args.use_mesh_deformations:
            if self.args.laplacian_reg_only_deforms:
                verts = data_dict['vertices_deforms']
            else:
                verts = data_dict['vertices']

            faces = self.parametric_avatar.render.faces.expand(verts.shape[0], -1, -1).long()

            data_dict['mesh'] = Meshes(
                verts=verts,
                faces=faces
            )

        # Apply masks
        if not self.args.unet_pred_mask:
            target_mask = data_dict['target_mask']
        else:
            target_mask = data_dict['pred_target_unet_mask'].detach()

        if self.args.use_random_uniform_background:
            random_bg_color = torch.rand(target_mask.shape[0], 3, 1, 1, dtype=target_mask.dtype,
                                         device=target_mask.device)
            data_dict['pred_target_img'] = data_dict['pred_target_img'] * target_mask + random_bg_color * (
                    1 - target_mask)
            data_dict['target_img'] = data_dict['target_img'] * target_mask + random_bg_color * (1 - target_mask)
        # else:
        #     random_bg_color = torch.zeros(target_mask.shape[0], 3, 1, 1, dtype=target_mask.dtype, device=target_mask.device)
        return data_dict

    def calc_train_losses(self, data_dict: dict, mode: str = 'gen'):
        losses_dict = {}

        if mode == 'dis':
            losses_dict['dis_adversarial'] = (
                    self.weights['adversarial'] *
                    self.adversarial_loss(
                        real_scores=data_dict['real_score_dis'],
                        fake_scores=data_dict['fake_score_dis'],
                        mode='dis'))

        if mode == 'gen':
            if self.weights['adversarial']:
                losses_dict['gen_adversarial'] = (
                        self.weights['adversarial'] *
                        self.adversarial_loss(
                            fake_scores=data_dict['fake_score_gen'],
                            mode='gen'))

                losses_dict['feature_matching'] = (
                        self.weights['feature_matching'] *
                        self.feature_matching_loss(
                            real_features=data_dict['real_feats_gen'],
                            fake_features=data_dict['fake_feats_gen']))

            if self.weights['l1']:
                losses_dict['l1'] = self.weights['l1'] * F.l1_loss(data_dict['pred_target_img'],
                                                                   data_dict['target_img'])

            if self.weights['vgg19']:
                losses_dict['vgg19'] = self.weights['vgg19'] * self.vgg19_loss.forward(
                    data_dict['pred_target_img'],
                    data_dict['target_img']
                )

            if self.weights['vggface']:
                pred_target_warped_img = F.grid_sample(data_dict['pred_target_img'], data_dict['target_warp_to_crop'])
                target_warped_img = F.grid_sample(data_dict['target_img'], data_dict['target_warp_to_crop'])

                losses_dict['vggface'] = self.weights['vggface'] * self.vggface_loss.forward(
                    pred_target_warped_img,
                    target_warped_img
                )

                # For vis
                with torch.no_grad():
                    data_dict['pred_target_warped_img'] = F.interpolate(
                        pred_target_warped_img,
                        size=256,
                        mode='bilinear'
                    )

                    data_dict['target_warped_img'] = F.interpolate(
                        target_warped_img,
                        size=256,
                        mode='bilinear'
                    )

            if self.weights['vgggaze']:
                try:
                    losses_dict['vgggaze'] = self.weights['vgggaze'] * self.vgggaze_loss.forward(
                        data_dict['pred_target_img'],
                        data_dict['target_img'],
                        data_dict['target_keypoints']
                    )

                except:
                    losses_dict['vgggaze'] = torch.zeros(1).to(data_dict['target_img'].device).mean()

            if self.weights['keypoints_matching']:
                losses_dict['keypoints_matching'] = self.weights['keypoints_matching'] * self.keypoints_matching_loss(
                    data_dict['pred_target_keypoints'],
                    data_dict['target_keypoints'])

                if self.weights['eye_closure']:
                    losses_dict['eye_closure'] = self.weights['eye_closure'] * self.eye_closure_loss(
                        data_dict['pred_target_keypoints'],
                        data_dict['target_keypoints'])

                if self.weights['lip_closure']:
                    losses_dict['lip_closure'] = self.weights['lip_closure'] * self.lip_closure_loss(
                        data_dict['pred_target_keypoints'],
                        data_dict['target_keypoints'])

            if self.args.finetune_flame_encoder or self.args.train_flame_encoder_from_scratch:

                if self.args.flame_encoder_reg:
                    losses_dict['shape_reg'] = (torch.sum(data_dict['shape'] ** 2) / 2) * self.weights['shape_reg']
                    losses_dict['exp_reg'] = (torch.sum(data_dict['exp'] ** 2) / 2) * self.weights['exp_reg']
                    if 'flame_tex_params' in data_dict.keys():
                        losses_dict['tex_reg'] = (torch.sum(data_dict['flame_tex_params'] ** 2) / 2) * self.weights[
                            'tex_reg']
                    # losses_dict['light_reg'] = ((torch.mean(data_dict['flame_light_params'], dim=2)[:, :, None] - data_dict[
                    #     'flame_light_params']) ** 2).mean() * self.weights['light_reg']

            if self.args.train_deferred_neural_rendering and self.args.unet_pred_mask:
                losses_dict['seg_unet'] = (
                        self.weights['unet_seg'] *
                        self.unet_seg_loss(
                            data_dict['pred_target_unet_logits'],
                            data_dict['target_mask']
                        )
                )

            if self.args.use_mesh_deformations:
                if self.weights['seg']:
                    losses_dict['seg'] = self.seg_loss(
                        data_dict['pred_target_soft_mask'],
                        data_dict['target_mask']
                    ) * self.weights['seg']

                if self.weights['seg_hair']:
                    losses_dict['seg_hair'] = self.seg_loss(
                        data_dict['pred_target_soft_detach_neck_mask'],
                        data_dict['target_hair_mask']
                    ) * self.weights['seg_hair']

                if self.weights['seg_neck']:
                    losses_dict['seg_neck'] = self.seg_loss(
                        data_dict['pred_target_soft_detach_hair_mask'],
                        data_dict['target_neck_mask']
                    ) * self.weights['seg_neck']

                if self.weights['seg_hard']:
                    data_dict['pred_target_soft_hair_only_mask'] = (
                            data_dict['pred_target_soft_hair_mask'] *
                            data_dict['pred_target_hard_hair_only_mask']
                    )

                    batch_indices = torch.nonzero(
                        (
                                data_dict['target_hair_only_mask'].mean([1, 2, 3]) /
                                data_dict['target_face_mask'].mean([1, 2, 3])
                        ) > 0.3
                    )[:, 0]

                    if len(batch_indices) > 0:
                        losses_dict['seg_only_hair'] = self.seg_loss(
                            data_dict['pred_target_soft_hair_only_mask'][batch_indices],
                            data_dict['target_hair_only_mask'][batch_indices]
                        ) * self.weights['seg_hard']
                    else:
                        losses_dict['seg_only_hair'] = torch.zeros(1, device=batch_indices.device,
                                                                   dtype=data_dict['target_mask'].dtype).mean()

                    # Zero values for visualization
                    tmp = data_dict['target_hair_only_mask'].clone()
                    data_dict['target_hair_only_mask'] = torch.zeros_like(tmp)
                    if len(batch_indices) > 0:
                        data_dict['target_hair_only_mask'][batch_indices] = tmp[batch_indices]

                if self.weights['seg_hard_neck']:
                    pred_target_soft_neck_mask = (
                            data_dict['pred_target_soft_neck_mask'] *
                            data_dict['pred_target_hard_neck_only_mask']
                    )

                    losses_dict['seg_only_neck'] = self.seg_loss(
                        pred_target_soft_neck_mask,
                        data_dict['target_neck_only_mask']
                    ) * self.weights['seg_hard_neck']

                if self.weights['chamfer']:
                    (
                        losses_dict['chamfer_loss'],
                        data_dict['chamfer_pred_vertices'],
                        data_dict['chamfer_target_vertices']
                    ) = self.chamfer_loss(
                        data_dict['vertices_target'][..., :2],
                        data_dict['vertices_vis_mask'],
                        data_dict['pred_target_hard_mask'],
                        data_dict['target_mask']
                    )
                    losses_dict['chamfer_loss'] = losses_dict['chamfer_loss'] * self.weights['chamfer']

                if self.weights['repulsion_hair']:
                    points = data_dict['vertices_target_hair'][:, self.deca.hair_list, :]
                    valid_dists = knn_points(points, points, K=5)[0]
                    losses_dict['repulsion_hair_loss'] = self.weights['repulsion_hair'] * (
                        torch.exp((-valid_dists / 10))).mean()

                if self.weights['repulsion']:
                    points = data_dict['vertices_target']
                    valid_dists = knn_points(points, points, K=5)[0]
                    losses_dict['repulsion_loss'] = self.weights['repulsion'] * (
                        torch.exp((-valid_dists / 10))).mean()

                if self.weights['chamfer_hair']:
                    batch_indices = torch.nonzero(
                        (
                                data_dict['target_hair_only_mask'].mean([1, 2, 3]) /
                                data_dict['target_face_mask'].mean([1, 2, 3])
                        ) > 0.3
                    )[:, 0]

                    data_dict['chamfer_pred_hair_vertices'] = torch.ones_like(
                        data_dict['vertices_target_hair'][:, :len(self.deca.hair_list), :2]) * -100.0
                    data_dict['chamfer_target_hair_vertices'] = data_dict['chamfer_pred_hair_vertices'].clone()

                    if len(batch_indices) > 0:
                        (
                            losses_dict['chamfer_hair_loss'],
                            chamfer_pred_hair_vertices,
                            chamfer_target_hair_vertices
                        ) = self.chamfer_loss(
                            data_dict['vertices_target_hair'][batch_indices][:, self.deca.hair_list, :2],
                            data_dict['vertices_hair_vis_mask'][batch_indices],
                            data_dict['pred_target_hard_hair_only_mask'][batch_indices],
                            data_dict['target_hair_only_mask'][batch_indices]
                        )

                        data_dict['chamfer_pred_hair_vertices'][batch_indices] = chamfer_pred_hair_vertices
                        data_dict['chamfer_target_hair_vertices'][batch_indices] = chamfer_target_hair_vertices

                    else:
                        losses_dict['chamfer_hair_loss'] = torch.zeros(1, device=batch_indices.device,
                                                                       dtype=data_dict['target_mask'].dtype).mean()

                    chamfer_weight = self.chamfer_hair_scheduler.step() if self.chamfer_hair_scheduler is not None else \
                        self.weights['chamfer_hair']
                    losses_dict['chamfer_hair_loss'] = losses_dict['chamfer_hair_loss'] * chamfer_weight

                if self.weights['chamfer_neck']:
                    (
                        losses_dict['chamfer_neck_loss'],
                        data_dict['chamfer_pred_neck_vertices'],
                        data_dict['chamfer_target_neck_vertices']
                    ) = self.chamfer_loss(
                        data_dict['vertices_target_neck'][:, self.deca.neck_list, :2],
                        data_dict['vertices_neck_vis_mask'],
                        data_dict['pred_target_hard_neck_only_mask'],
                        data_dict['target_neck_only_mask']
                    )
                    chamfer_weight = self.weights['chamfer_neck']
                    losses_dict['chamfer_neck_loss'] = losses_dict['chamfer_neck_loss'] * chamfer_weight

                if self.weights['laplacian_reg']:
                    laplacian_weight = self.weights['laplacian_reg']
                    losses_dict['laplacian_reg'] = laplacian_weight * \
                                                   self.laplace_loss(data_dict['mesh'],
                                                                     data_dict.get('laplace_coefs'))

                if self.weights['edge_reg']:
                    losses_dict['edge_reg'] = self.weights['edge_reg'] * mesh_edge_loss(data_dict['mesh'])

                if self.weights['normal_reg']:
                    losses_dict['normal_reg'] = self.weights['normal_reg'] * mesh_normal_consistency(data_dict['mesh'])

                if self.args.reg_positive_z_normals:
                    losses_dict['normal_reg'] = torch.pow(torch.relu(-data_dict['normals_z']), 2).mean()
        loss = 0
        for k, v in losses_dict.items():
            loss += v

        return loss, losses_dict

    def calc_test_losses(self, data_dict: dict):
        losses_dict = {}

        if self.args.pretrain_global_encoder:
            losses_dict['cam'] = ((data_dict['pred_cam'] - data_dict['flame_cam_params']) ** 2).mean(0).sum()
            losses_dict['pose_rot'] = ((data_dict['pred_pose'][:, 0] - data_dict['flame_pose_params'][:, 0]) ** 2).mean(
                0)
            losses_dict['pose_dir'] = 1 - (
                    data_dict['pred_pose'][:, 1:4] *
                    data_dict['flame_pose_params'][:, 1:4]
            ).sum(-1).mean(0)

        if 'pred_target_img' in data_dict.keys() and data_dict['pred_target_img'] is not None:
            losses_dict['ssim'] = self.ssim(data_dict['pred_target_img'], data_dict['target_img']).mean()
            losses_dict['psnr'] = self.psnr(data_dict['pred_target_img'], data_dict['target_img'])
            losses_dict['lpips'] = self.lpips(data_dict['pred_target_img'], data_dict['target_img'])

            if self.args.model_image_size > 160:
                losses_dict['ms_ssim'] = self.ms_ssim(data_dict['pred_target_img'], data_dict['target_img']).mean()

        return losses_dict

    def prepare_input_data(self, data_dict):
        for k, v in data_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    if self.args.num_gpus:
                        v_ = v_.cuda()
                    v[k_] = v_.view(-1, *v_.shape[2:])
                data_dict[k] = v
            else:
                if self.args.num_gpus:
                    v = v.cuda()
                data_dict[k] = v.view(-1, *v.shape[2:])

        return data_dict

    def forward(self,
                data_dict: dict,
                phase: str = 'test',
                optimizer_idx: int = 0,
                visualize: bool = False):
        assert phase in ['train', 'test']
        mode = self.optimizer_idx_to_mode[optimizer_idx]

        if mode == 'gen':
            data_dict = self.prepare_input_data(data_dict)
            data_dict = self._forward(data_dict)

            if phase == 'train':
                if self.args.adversarial_weight > 0:
                    self.discriminator.eval()

                    with torch.no_grad():
                        _, data_dict['real_feats_gen'] = self.discriminator(data_dict['target_img'])

                    if self.args.adv_only_for_rendering:
                        data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(
                            data_dict['pred_target_img_detach_geom'])
                    else:
                        data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(
                            data_dict['pred_target_img'])

                loss, losses_dict = self.calc_train_losses(data_dict, mode='gen')

            elif phase == 'test':
                loss = None
                losses_dict = self.calc_test_losses(data_dict)

            histograms_dict = {}

        elif mode == 'dis':
            # Backward through dis
            self.discriminator.train()

            data_dict['real_score_dis'], _ = self.discriminator(data_dict['target_img'])
            data_dict['fake_score_dis'], _ = self.discriminator(data_dict['pred_target_img'].detach().clone())

            loss, losses_dict = self.calc_train_losses(data_dict, mode='dis')

            histograms_dict = {}

        visuals = None
        if visualize:
            visuals = self.get_visuals(data_dict)

        return loss, losses_dict, histograms_dict, visuals, data_dict

    @torch.no_grad()
    def get_visuals(self, data_dict):
        data_dict['target_stickman'] = misc.draw_stickman(data_dict['target_keypoints'], self.args.model_image_size)
        data_dict['pred_target_stickman'] = misc.draw_stickman(data_dict['pred_target_keypoints'],
                                                               self.args.model_image_size)
        # This function creates an output grid of visuals
        visuals_data_dict = {}

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            visuals_data_dict[k] = v

        if 'pred_source_shape_img' in data_dict.keys():
            visuals_data_dict['pred_source_shape_overlay_img'] = (
                                                                         data_dict['source_img'] +
                                                                         data_dict['pred_source_shape_img']
                                                                 ) * 0.5

        visuals_data_dict['pred_target_shape_overlay_img'] = (
                                                                     data_dict['target_img'] +
                                                                     data_dict['pred_target_shape_img']
                                                             ) * 0.5

        if 'chamfer_pred_vertices' in data_dict.keys():
            visuals_data_dict['chamfer_vis_pred_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_pred_vertices']
            )

            visuals_data_dict['chamfer_vis_target_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_target_vertices']
            )

        if 'chamfer_pred_hair_vertices' in data_dict.keys():
            visuals_data_dict['chamfer_vis_pred_hair_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_pred_hair_vertices']
            )

            visuals_data_dict['chamfer_vis_target_hair_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_target_hair_vertices']
            )

        if 'chamfer_pred_neck_vertices' in data_dict.keys():
            visuals_data_dict['chamfer_vis_pred_neck_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_pred_neck_vertices']
            )

            visuals_data_dict['chamfer_vis_target_neck_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_target_neck_vertices']
            )

        if 'pred_texture_warp' in data_dict.keys():
            visuals_data_dict['vis_pred_texture_warp'] = F.grid_sample(
                data_dict['source_img'],
                data_dict['pred_texture_warp'],
                mode='bilinear'
            )

        visuals = []

        uv_prep = lambda x: (x.permute(0, 3, 1, 2) + 1) / 2
        coords_prep = lambda x: (x + 1) / 2
        seg_prep = lambda x: torch.cat([x] * 3, dim=1)
        score_prep = lambda x: (x + 1) / 2

        visuals_list = [
            ['source_img', None],
            ['source_mask', seg_prep],
            ['source_warped_img', None],
            ['pred_source_coarse_uv', uv_prep],
            ['pred_texture_warp', uv_prep],
            ['vis_pred_texture_warp', None],
            ['target_img', None],
            ['pred_target_img', None],
            ['target_stickman', None],
            ['pred_target_stickman', None],
            ['pred_target_coord', coords_prep],
            ['pred_target_uv', uv_prep],
            ['pred_target_shape_img', None],
            ['pred_target_shape_displ_img', None],
            ['pred_target_shape_overlay_img', None],

            ['target_mask', seg_prep],
            ['pred_target_unet_mask', seg_prep],
            ['pred_target_hard_mask', seg_prep],
            ['pred_target_soft_mask', seg_prep],
            ['target_face_mask', seg_prep],
            ['pred_target_hard_face_only_mask', seg_prep],
            ['chamfer_vis_pred_verties', None],
            ['chamfer_vis_target_verties', None],

            ['target_hair_mask', seg_prep],
            ['target_hair_only_mask', seg_prep],
            ['pred_target_soft_hair_mask', seg_prep],
            ['pred_target_hard_hair_only_mask', seg_prep],
            ['pred_target_soft_hair_only_mask', seg_prep],
            ['chamfer_vis_pred_hair_verties', None],
            ['chamfer_vis_target_hair_verties', None],

            ['target_neck_mask', seg_prep],
            ['target_neck_only_mask', seg_prep],
            ['pred_target_soft_neck_mask', seg_prep],
            ['pred_target_hard_neck_only_mask', seg_prep],
            ['pred_target_soft_neck_only_mask', seg_prep],
            ['chamfer_vis_pred_neck_verties', None],
            ['chamfer_vis_target_neck_verties', None],

            ['pred_target_normal', coords_prep],
            ['pred_target_shading', None if self.args.shading_channels == 3 else seg_prep],
            ['pred_target_albedo', None],
            ['target_vertices_texture', coords_prep],
            ['target_shape_final_posed_img', None],
            ['target_shape_final_frontal_img', None],
            ['target_shape_parametric_frontal_img', None],
            ['target_shape_neutral_frontal_img', None],

            ['pred_target_warped_img', None],
            ['target_warped_img', None],

            ['dummy_vis', seg_prep]
        ]

        if (
                self.args.finetune_flame_encoder or self.args.train_flame_encoder_from_scratch) and not self.args.use_mesh_deformations:
            visuals_list = [
                ['source_img', None],
                ['source_mask', seg_prep],
                ['pred_texture_warp', uv_prep],
                ['vis_pred_texture_warp', None],
                ['target_img', None],
                ['pred_target_img', None],
                ['target_stickman', None],
                ['pred_target_stickman', None],
                ['pred_target_coord', coords_prep],
                ['pred_target_uv', uv_prep],
                ['pred_target_shape_img', None],
                ['pred_target_shape_displ_img', None],
                ['pred_target_shape_overlay_img', None],
                ['pred_target_warped_img', None],
                ['target_warped_img', None],
            ]

        if self.args.use_mlp_deformer:
            visuals_list.append(['deformations_out', coords_prep])
            visuals_list.append(['deformations_inp_orig', None])
            visuals_list.append(['deformations_inp_texture', None])
            visuals_list.append(['deformations_inp_coord', coords_prep])

        max_h = max_w = 0

        for tensor_name, preprocessing_op in visuals_list:
            if tensor_name in visuals_data_dict.keys() and visuals_data_dict[tensor_name] is not None:
                visuals += misc.prepare_visual(visuals_data_dict, tensor_name, preprocessing_op)

            if len(visuals):
                h, w = visuals[-1].shape[2:]
                max_h = max(h, max_h)
                max_w = max(w, max_w)

        visuals = torch.cat(visuals, 3)  # cat w.r.t. width
        visuals = visuals.clamp(0, 1)

        return visuals

    def gen_parameters(self):
        params = iter([])

        if self.args.train_deferred_neural_rendering or self.args.train_texture_encoder:
            print('Training autoencoder')
            params = itertools.chain(params, self.autoencoder.parameters())

        if self.args.train_deferred_neural_rendering:
            print('Training rendering unet')
            params = itertools.chain(params, self.unet.parameters())
            if self.args.unet_pred_mask and self.args.use_separate_seg_unet:
                print('Training seg unet')
                params = itertools.chain(params, self.unet_seg.parameters())

        if self.basis_deformer is not None:
            print('Training basis deformer')
            params = itertools.chain(params, self.basis_deformer.parameters())
            if self.args.train_basis:
                params = itertools.chain(params, self.vertex_deformer.parameters())

        if self.mesh_deformer is not None:
            print('Training mesh deformer')
            params = itertools.chain(params, self.mesh_deformer.parameters())

            if self.mlp_deformer is not None:
                print('Training MLP deformer')
                params = itertools.chain(params, self.mlp_deformer.parameters())

        if self.args.finetune_flame_encoder or self.args.train_flame_encoder_from_scratch:
            print('Training FLAME encoder')
            params = itertools.chain(params, self.deca.E_flame.parameters())

        for param in params:
            yield param

    def configure_optimizers(self):
        self.optimizer_idx_to_mode = {0: 'gen', 1: 'dis'}

        opts = {
            'adam': lambda param_groups, lr, beta1, beta2: torch.optim.Adam(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2)),
            'adamw': lambda param_groups, lr, beta1, beta2: torch.optim.AdamW(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2))}

        opt_gen = opts[self.args.gen_opt_type](
            self.gen_parameters(),
            self.args.gen_lr,
            self.args.gen_beta1,
            self.args.gen_beta2)

        if self.args.adversarial_weight > 0:
            opt_dis = opts[self.args.dis_opt_type](
                self.discriminator.parameters(),
                self.args.dis_lr,
                self.args.dis_beta1,
                self.args.dis_beta2)

            return [opt_gen, opt_dis]

        else:
            return [opt_gen]

    def configure_schedulers(self, opts):
        shds = {
            'step': lambda optimizer, lr_max, lr_min, max_iters: torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=max_iters,
                gamma=lr_max / lr_min),
            'cosine': lambda optimizer, lr_max, lr_min, max_iters: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max_iters,
                eta_min=lr_min)}

        if self.args.gen_shd_type != 'none':
            shd_gen = shds[self.args.gen_shd_type](
                opts[0],
                self.args.gen_lr,
                self.args.gen_shd_lr_min,
                self.args.gen_shd_max_iters)

        if self.args.adversarial_weight > 0 and self.args.dis_shd_type != 'none':
            shd_dis = shds[self.args.dis_shd_type](
                opts[1],
                self.args.dis_lr,
                self.args.dis_shd_lr_min,
                self.args.dis_shd_max_iters)

            return [shd_gen, shd_dis], [self.args.gen_shd_max_iters, self.args.dis_shd_max_iters], []

        elif self.args.gen_shd_type != 'none':
            return [shd_gen], [self.args.gen_shd_max_iters], []

        else:
            return [], [], []

    @torch.no_grad()
    def forward_infer(self, data_dict, neutral_pose: bool = False, source_information=None):
        if source_information is None:
            source_information = dict()

        parametric_output = self.parametric_avatar.forward(
            data_dict['source_img'],
            data_dict['source_mask'],
            data_dict['source_keypoints'],
            data_dict['target_img'],
            data_dict['target_keypoints'],
            deformer_nets={
                'neural_texture_encoder': self.autoencoder,
                'unet_deformer': self.mesh_deformer,
                'mlp_deformer': self.mlp_deformer,
                'basis_deformer': self.basis_deformer,
            },
            neutral_pose=neutral_pose,
            neural_texture=source_information.get('neural_texture'),
            source_information=source_information,
        )
        result_dict = {}
        rendered_texture = parametric_output.pop('rendered_texture')

        for key, value in parametric_output.items():
            result_dict[key] = value

        unet_inputs = rendered_texture * result_dict['pred_target_hard_mask']

        normals = result_dict['pred_target_normal'].permute(0, 2, 3, 1)
        normal_inputs = harmonic_encoding.harmonic_encoding(normals, 6).permute(0, 3, 1, 2)
        unet_inputs = torch.cat([unet_inputs, normal_inputs], dim=1)
        unet_outputs = self.unet(unet_inputs)

        pred_img = torch.sigmoid(unet_outputs[:, :3])
        pred_soft_mask = torch.sigmoid(unet_outputs[:, 3:])

        return_mesh = False
        if return_mesh:
            verts = result_dict['vertices_target'].cpu()
            faces = self.parametric_avatar.render.faces.expand(verts.shape[0], -1, -1).long()
            result_dict['mesh'] = Meshes(verts=verts, faces=faces)

        result_dict['pred_target_unet_mask'] = pred_soft_mask
        result_dict['pred_target_img'] = pred_img
        mask_pred = (result_dict['pred_target_unet_mask'][0].cpu() > self.mask_hard_threshold).float()
        mask_pred = mask_errosion(mask_pred.float().numpy() * 255)
        result_dict['render_masked'] = result_dict['pred_target_img'][0].cpu() * (mask_pred) + (1 - mask_pred)

        return result_dict

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("model")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--model_image_size', default=256, type=int)

        parser.add_argument('--predict_face_parsing_mask', action='store_true')
        parser.add_argument('--compute_face_parsing_mask', action='store_true')
        parser.add_argument('--face_parsing_path', default='')
        parser.add_argument('--face_parsing_mask_type', default='face')
        parser.add_argument('--include_neck_to_hair_mask', action='store_true')

        parser.add_argument('--use_graphonomy_mask', action='store_true')
        parser.add_argument('--graphonomy_path', default='')

        parser.add_argument('--segm_classes', default=0, type=int)
        parser.add_argument('--fp_visualize_uv', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_fp_merge_classes', action='store_true')
        parser.add_argument('--update_silh_with_segm', action='store_true')
        parser.add_argument('--mask_silh_cloth', action='store_true')

        parser.add_argument('--adv_only_for_rendering', default='False', type=args_utils.str2bool,
                            choices=[True, False])

        parser.add_argument('--use_mesh_deformations', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--subdivide_mesh', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--detach_silhouettes', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--train_deferred_neural_rendering', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--train_only_autoencoder', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--train_texture_encoder', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--train_extra_flame_parameters', action='store_true')
        parser.add_argument('--train_flametex', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--use_cam_encoder', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_neck_pose_encoder', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_light_encoder', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--light_channels', default=8, type=int)

        parser.add_argument('--pretrain_global_encoder', default='False', type=args_utils.str2bool,
                            choices=[True, False],
                            help='fit the encoder from DECA')
        parser.add_argument('--train_global_encoder', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--renderer_sigma', default=1e-8, type=float)
        parser.add_argument('--renderer_zfar', default=100.0, type=float)
        parser.add_argument('--renderer_type', default='soft_mesh')
        parser.add_argument('--renderer_texture_type', default='texture_uv')
        parser.add_argument('--renderer_normalized_alphas', default='False', type=args_utils.str2bool,
                            choices=[True, False])

        parser.add_argument('--deca_path', default='')
        parser.add_argument('--global_encoder_path', default='')
        parser.add_argument('--deferred_neural_rendering_path', default='')
        parser.add_argument('--deca_neutral_pose', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--autoenc_cat_alphas', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_align_inputs', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_use_warp', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_num_channels', default=64, type=int)
        parser.add_argument('--autoenc_max_channels', default=512, type=int)
        parser.add_argument('--autoenc_num_warp_groups', default=4, type=int)
        parser.add_argument('--autoenc_num_warp_blocks', default=1, type=int)
        parser.add_argument('--autoenc_num_warp_layers', default=3, type=int)
        parser.add_argument('--autoenc_num_groups', default=4, type=int)
        parser.add_argument('--autoenc_num_bottleneck_groups', default=0, type=int)
        parser.add_argument('--autoenc_num_blocks', default=2, type=int)
        parser.add_argument('--autoenc_num_layers', default=4, type=int)
        parser.add_argument('--autoenc_block_type', default='bottleneck')
        parser.add_argument('--autoenc_use_psp', action='store_true')

        parser.add_argument('--neural_texture_channels', default=16, type=int)

        parser.add_argument('--finetune_flame_encoder', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--flame_encoder_reg', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--train_flame_encoder_from_scratch', default='False', type=args_utils.str2bool,
                            choices=[True, False])

        parser.add_argument('--flame_num_shape_params', default=-1, type=int)
        parser.add_argument('--flame_num_exp_params', default=-1, type=int)
        parser.add_argument('--flame_num_tex_params', default=-1, type=int)

        parser.add_argument('--mesh_deformer_gain', default=0.001, type=float)

        parser.add_argument('--backprop_adv_only_into_unet', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--replace_bn_with_in', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--num_harmonic_encoding_funcs', default=6, type=int)

        parser.add_argument('--unet_num_channels', default=64, type=int)
        parser.add_argument('--unet_max_channels', default=1024, type=int)
        parser.add_argument('--unet_num_groups', default=4, type=int)
        parser.add_argument('--unet_num_blocks', default=1, type=int)
        parser.add_argument('--unet_num_layers', default=2, type=int)
        parser.add_argument('--unet_block_type', default='conv')
        parser.add_argument('--unet_skip_connection_type', default='cat')
        parser.add_argument('--unet_use_normals_cond', action='store_true')
        parser.add_argument('--unet_use_vertex_cond', action='store_true')
        parser.add_argument('--unet_use_uvs_cond', action='store_true')
        parser.add_argument('--unet_pred_mask', action='store_true')
        parser.add_argument('--use_separate_seg_unet', default='True', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--use_mlp_renderer', action='store_true')

        parser.add_argument('--use_shading_renderer', action='store_true')
        parser.add_argument('--shading_channels', default=1, type=int)

        parser.add_argument('--norm_layer_type', default='gn', type=str, choices=['bn', 'sync_bn', 'in', 'gn'])
        parser.add_argument('--activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--conv_layer_type', default='ws_conv', type=str, choices=['conv', 'ws_conv'])

        parser.add_argument('--deform_norm_layer_type', default='gn', type=str, choices=['bn', 'sync_bn', 'in', 'gn'])
        parser.add_argument('--deform_activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--deform_conv_layer_type', default='ws_conv', type=str, choices=['conv', 'ws_conv'])

        parser.add_argument('--dis_num_channels', default=64, type=int)
        parser.add_argument('--dis_max_channels', default=512, type=int)
        parser.add_argument('--dis_num_blocks', default=4, type=int)
        parser.add_argument('--dis_num_scales', default=1, type=int)

        parser.add_argument('--dis_init_type', default='xavier')
        parser.add_argument('--dis_init_gain', default=0.02, type=float)

        parser.add_argument('--adversarial_weight', default=0.0, type=float)
        parser.add_argument('--gen_adversarial_weight', default=-1.0, type=float)
        parser.add_argument('--feature_matching_weight', default=0.0, type=float)

        parser.add_argument('--vgg19_weight', default=0.0, type=float)
        parser.add_argument('--vgg19_num_scales', default=1, type=int)
        parser.add_argument('--vggface_weight', default=0.0, type=float)
        parser.add_argument('--vgggaze_weight', default=0.0, type=float)

        parser.add_argument('--unet_seg_weight', default=0.0, type=float)
        parser.add_argument('--unet_seg_type', default='bce_with_logits', type=str, choices=['bce_with_logits', 'dice'])

        parser.add_argument('--l1_weight', default=0.0, type=float)
        parser.add_argument('--l1_hair_weight', default=0.0, type=float)
        parser.add_argument('--repulsion_hair_weight', default=0.0, type=float)
        parser.add_argument('--repulsion_weight', default=0.0, type=float)

        parser.add_argument('--keypoints_matching_weight', default=1.0, type=float)
        parser.add_argument('--eye_closure_weight', default=1.0, type=float)
        parser.add_argument('--lip_closure_weight', default=0.5, type=float)
        parser.add_argument('--seg_weight', default=0.0, type=float)
        parser.add_argument('--seg_type', default='bce', type=str, choices=['bce', 'iou', 'mse'])
        parser.add_argument('--seg_num_scales', default=1, type=int)

        parser.add_argument('--seg_hard_weight', default=0.0, type=float)
        parser.add_argument('--seg_hair_weight', default=0.0, type=float)
        parser.add_argument('--seg_neck_weight', default=0.0, type=float)
        parser.add_argument('--seg_hard_neck_weight', default=0.0, type=float)
        parser.add_argument('--seg_ignore_face', action='store_true')

        parser.add_argument('--chamfer_weight', default=0.0, type=float)
        parser.add_argument('--chamfer_hair_weight', default=0.0, type=float)
        parser.add_argument('--chamfer_neck_weight', default=0.0, type=float)
        parser.add_argument('--chamfer_num_neighbours', default=1, type=int)
        parser.add_argument('--chamfer_same_num_points', action='store_true')
        parser.add_argument('--chamfer_remove_face', action='store_true')
        parser.add_argument('--chamfer_sample_outside_of_silhouette', action='store_true')

        parser.add_argument('--shape_reg_weight', default=1e-4, type=float)
        parser.add_argument('--exp_reg_weight', default=1e-4, type=float)
        parser.add_argument('--tex_reg_weight', default=1e-4, type=float)
        parser.add_argument('--light_reg_weight', default=1.0, type=float)

        parser.add_argument('--laplacian_reg_weight', default=0.0, type=float)
        parser.add_argument('--laplacian_reg_only_deforms', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--laplacian_reg_apply_to_hair_only', action='store_true')
        parser.add_argument('--laplacian_reg_hair_weight', default=0.0, type=float)
        parser.add_argument('--laplacian_reg_neck_weight', default=0.0, type=float)

        parser.add_argument('--laplacian_reg_weight_start', default=0.0, type=float)
        parser.add_argument('--laplacian_reg_weight_end', default=0.0, type=float)
        parser.add_argument('--chamfer_hair_weight_start', default=0.0, type=float)
        parser.add_argument('--chamfer_hair_weight_end', default=0.0, type=float)
        parser.add_argument('--chamfer_neck_weight_start', default=0.0, type=float)
        parser.add_argument('--chamfer_neck_weight_end', default=0.0, type=float)
        parser.add_argument('--scheduler_total_iter', default=50000, type=int)

        parser.add_argument('--deform_face_tightness', default=0.0, type=float)

        parser.add_argument('--use_whole_segmentation', action='store_true')
        parser.add_argument('--mask_hair_for_neck', action='store_true')
        parser.add_argument('--use_hair_from_avatar', action='store_true')

        # Basis deformations
        parser.add_argument('--use_basis_deformer', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--use_unet_deformer', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--pretrained_encoder_basis_path', default='')
        parser.add_argument('--pretrained_vertex_basis_path', default='')
        parser.add_argument('--num_basis', default=50, type=int)
        parser.add_argument('--basis_init', default='pca', type=str, choices=['random', 'pca'])
        parser.add_argument('--num_vertex', default=5023, type=int)
        parser.add_argument('--train_basis', default=True, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--path_to_deca', default='/Vol0/user/v.sklyarova/cvpr/latent-texture-avatar/utils')

        parser.add_argument('--deformer_path', default=None)

        parser.add_argument('--edge_reg_weight', default=0.0, type=float)
        parser.add_argument('--normal_reg_weight', default=0.0, type=float)

        # Deformation Block arguments
        parser.add_argument('--use_scalp_deforms', default=False, action='store_true')
        parser.add_argument('--use_neck_deforms', default=False, action='store_true')

        parser.add_argument('--use_gaze_dir', default=True, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_neck_dir', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--harmonize_deform_input', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='harmonize input in the deformation Unet module')
        parser.add_argument('--detach_deformation_vertices', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='detach textured vertices')
        parser.add_argument('--predict_deformed_vertices', default=False, action='store_true',
                            help='predict new vertices')
        parser.add_argument('--output_unet_deformer_feats', default=3, type=int,
                            help='output features in the UNet')
        parser.add_argument('--use_mlp_deformer', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--harmonize_uv_mlp_input', default=True, action='store_true',
                            help='harmonize uv positional encoding')
        parser.add_argument('--mask_render_inputs', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='mask resampled texture for rendering')
        parser.add_argument('--per_vertex_deformation', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_complex_mesh', default=False, action='store_true',
                            help='two mesh for faces with blending weights')
        parser.add_argument('--invert_opacity', default=False, action='store_true',
                            help='instead of use opacity direct use 1 - p')
        parser.add_argument('--predict_sep_opacity', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='mask resampled texture for rendering')
        parser.add_argument('--multi_texture', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='use the second autoencoder for separate texture predicting')
        parser.add_argument('--detach_deformations', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='detach_deformations for training weight map')
        parser.add_argument('--use_extended_flame', default=False, action='store_true',
                            help='use extended flame template')
        parser.add_argument('--use_deca_details', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_flametex', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--mask_uv_opacity', default=False, action='store_true',
                            help='mask opacity for faces')
        parser.add_argument('--mask_uv_face', default=False, action='store_true',
                            help='mask uvs for faces')

        parser.add_argument('--train_only_face', default=False, action='store_true')
        parser.add_argument('--deform_face', default=False, action='store_true')
        parser.add_argument('--deform_along_normals', default=False, action='store_true')
        parser.add_argument('--deform_hair_along_normals', default=False, action='store_true')
        parser.add_argument('--deform_nothair_along_normals', default=False, action='store_true')

        parser.add_argument('--reg_scalp_only', default=False, action='store_true')
        parser.add_argument('--mask_according_to_normal', default=False, action='store_true')

        parser.add_argument('--mask_neck_deformation_uvs', default=False, action='store_true')
        parser.add_argument('--mask_ear_deformation_uvs', default=False, action='store_true',
                            help='mask uvs for faces')
        parser.add_argument('--mask_deformation_uvs', default=False, action='store_true',
                            help='mask uvs for input in deformation network')
        parser.add_argument('--mask_eye_deformation_uvs', default=False, action='store_true',
                            help='mask eye in uvs for input in deformation network')
        parser.add_argument('--mask_hair_deformation_uvs', default=False, action='store_true',
                            help='mask hair according ot segmentation in uvs for input in deformation network')
        parser.add_argument('--use_hard_mask', default=False, action='store_true',
                            help='use_hard masking procedure')
        parser.add_argument('--use_updated_vertices', default=False, action='store_true',
                            help='use updated vertices')

        parser.add_argument('--detach_deforms_neural_texture', default=False, action='store_true')
        parser.add_argument('--hard_masking_deformations', default=False, action='store_true')
        parser.add_argument('--double_subdivide', default=False, action='store_true')

        parser.add_argument('--use_post_rendering_augs', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--use_random_uniform_background', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--reset_dis_weights', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--predict_hair_silh', default=False, action='store_true')
        parser.add_argument('--detach_neck', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--updated_neck_mask', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='')

        parser.add_argument('--reg_positive_z_normals', default=False, action='store_true',
                            help='use defomations for mesh regularization')
        parser.add_argument('--use_deformation_reg', default=False, action='store_true',
                            help='use defomations for mesh regularization')
        parser.add_argument('--use_laplace_vector_coef', default=False, action='store_true',
                            help='use defomations for mesh regularization')
        parser.add_argument('--mask_eye_things_all', default=False, action='store_true',
                            help='')
        parser.add_argument('--mask_hair_face_soft', default=False, action='store_true')

        parser.add_argument('--mlp_input_camera_conditioned', default=False, action='store_true')

        parser.add_argument('--lambda_diffuse_reg', default=0.0, type=float)
        parser.add_argument('--num_frequencies', default=6, type=int, help='frequency for harmonic encoding')

        parser.add_argument('--laplace_reg_type', default='uniform', type=str, choices=['uniform', 'cot', 'cotcurv'])
        parser.add_argument('--update_laplace_weight_every', default=0, type=int)
        parser.add_argument('--vggface_path', default='data/resnet50_scratch_dag.pth', type=str)
        parser.add_argument('--use_gcn', default=False, action='store_true', help='')
        parser.add_argument('--dump_mesh', default=False, action='store_true',
                            help='dump batch of meshes')
        parser.add_argument('--deform_face_scale_coef', default=0.0, type=float)

        parser.add_argument('--spn_apply_to_gen', default=False, action='store_true')
        parser.add_argument('--spn_apply_to_dis', default=False, action='store_true')
        parser.add_argument('--spn_layers', default='conv2d, linear')

        # Optimization options
        parser.add_argument('--gen_opt_type', default='adam')
        parser.add_argument('--gen_lr', default=1e-4, type=float)
        parser.add_argument('--gen_beta1', default=0.0, type=float)
        parser.add_argument('--gen_beta2', default=0.999, type=float)

        parser.add_argument('--gen_weight_decay', default=1e-4, type=float)
        parser.add_argument('--gen_weight_decay_layers', default='conv2d')
        parser.add_argument('--gen_weight_decay_params', default='weight')

        parser.add_argument('--gen_shd_type', default='none')
        parser.add_argument('--gen_shd_max_iters', default=2.5e5, type=int)
        parser.add_argument('--gen_shd_lr_min', default=1e-6, type=int)

        parser.add_argument('--dis_opt_type', default='adam')
        parser.add_argument('--dis_lr', default=4e-4, type=float)
        parser.add_argument('--dis_beta1', default=0.0, type=float)
        parser.add_argument('--dis_beta2', default=0.999, type=float)

        parser.add_argument('--dis_shd_type', default='none')
        parser.add_argument('--dis_shd_max_iters', default=2.5e5, type=int)
        parser.add_argument('--dis_shd_lr_min', default=4e-6, type=int)
        parser.add_argument('--device', default='cuda', type=str)
        parser.add_argument('--deca_path', default='')
        parser.add_argument('--rome_data_dir', default='')

        parser.add_argument('--use_distill', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_mobile_version', default=False, type=args_utils.str2bool, choices=[True, False])

        return parser_out
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import albumentations as A
from argparse import ArgumentParser
import io
from PIL import Image, ImageOps
import random
import cv2
import pickle

from src.utils import args as args_utils
from src.utils.point_transforms import parse_3dmm_param


class LMDBDataset(data.Dataset):
    def __init__(self,
                 data_root,
                 image_size,
                 keys,
                 phase,
                 align_source=False,
                 align_target=False,
                 align_scale=1.0,
                 augment_geometric_source=False,
                 augment_geometric_target=False,
                 augment_color=False,
                 output_aug_warp=False,
                 aug_warp_size=-1,
                 epoch_len=-1,
                 return_keys=False,
                 ):
        super(LMDBDataset, self).__init__()
        self.envs = []
        for i in range(128):
            self.envs.append(lmdb.open(f'{data_root}/chunks/{i}_lmdb', max_readers=1, readonly=True,
                                       lock=False, readahead=False, meminit=False))

        self.keys = keys
        self.phase = phase

        self.image_size = image_size

        self.return_keys = return_keys
        self.align_source = align_source
        self.align_target = align_target
        self.align_scale = align_scale
        self.augment_geometric_source = augment_geometric_source
        self.augment_geometric_target = augment_geometric_target
        self.augment_color = augment_color
        self.output_aug_warp = output_aug_warp

        self.epoch_len = epoch_len

        # Transforms
        if self.augment_color:
            self.aug = A.Compose(
                [A.ColorJitter(hue=0.02, p=0.8)],
                additional_targets={f'image{k}': 'image' for k in range(1, 2)})
        self.to_tensor = transforms.ToTensor()

        if self.align_source:
            grid = torch.linspace(-1, 1, self.image_size)
            v, u = torch.meshgrid(grid, grid)
            self.identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)

        if self.output_aug_warp:
            self.aug_warp_size = aug_warp_size

            # Greate a uniform meshgrid, which is used for warping calculation from deltas
            tick = torch.linspace(0, 1, self.aug_warp_size)
            v, u = torch.meshgrid(tick, tick)
            grid = torch.stack([u, v, torch.zeros(self.aug_warp_size, self.aug_warp_size)], dim=2)

            self.grid = (grid * 255).numpy().astype('uint8')  # aug_warp_size x aug_warp_size x 3

    @staticmethod
    def to_tensor_keypoints(keypoints, size):
        keypoints = torch.from_numpy(keypoints).float()
        keypoints /= size
        keypoints[..., :2] -= 0.5
        keypoints *= 2

        return keypoints

    def __getitem__(self, index):
        n = 1
        t = 1

        chunk, keys_ = self.keys[index]
        env = self.envs[chunk]

        if self.phase == 'train':
            indices = torch.randperm(len(keys_))[:2]
            keys = [keys_[i] for i in indices]

        else:
            keys = keys_

        data_dict = {
            'image': [],
            'mask': [],
            'size': [],
            'face_scale': [],
            'keypoints': [],
            'params_3dmm': {'R': [], 'offset': [], 'roi_box': [], 'size': []},
            'params_ffhq': {'theta': []},
            'crop_box': []}

        with env.begin(write=False) as txn:
            for key in keys:
                item = pickle.loads(txn.get(key.encode()))

                image = Image.open(io.BytesIO(item['image'])).convert('RGB')
                mask = Image.open(io.BytesIO(item['mask']))

                data_dict['image'].append(image)
                data_dict['mask'].append(mask)

                data_dict['size'].append(item['size'])
                data_dict['face_scale'].append(item['face_scale'])
                data_dict['keypoints'].append(item['keypoints_2d'])

                R, offset, _, _ = parse_3dmm_param(item['3dmm']['param'])

                data_dict['params_3dmm']['R'].append(R)
                data_dict['params_3dmm']['offset'].append(offset)
                data_dict['params_3dmm']['roi_box'].append(item['3dmm']['bbox'])
                data_dict['params_3dmm']['size'].append(item['size'])

                data_dict['params_ffhq']['theta'].append(item['transform_ffhq']['theta'])

        # Geometric augmentations and resize
        data_dict = self.preprocess_data(data_dict)
        data_dict['image'] = [np.asarray(img).copy() for img in data_dict['image']]

        # Augment color
        if self.augment_color:
            imgs_dict = {(f'image{k}' if k > 0 else 'image'): img for k, img in enumerate(data_dict['image'])}
            data_dict['image'] = list(self.aug(**imgs_dict).values())

        # Augment with local warpings
        if self.output_aug_warp:
            warp_aug = self.augment_via_warp([self.grid] * (n + t), self.aug_warp_size)
            warp_aug = torch.stack([self.to_tensor(w) for w in warp_aug], dim=0)
            warp_aug = (warp_aug.permute(0, 2, 3, 1)[..., :2] - 0.5) * 2

        imgs = torch.stack([self.to_tensor(img) for img in data_dict['image']])
        masks = torch.stack([self.to_tensor(mask) for mask in data_dict['mask']])
        keypoints = torch.FloatTensor(data_dict['keypoints'])

        R = torch.FloatTensor(data_dict['params_3dmm']['R'])
        offset = torch.FloatTensor(data_dict['params_3dmm']['offset'])
        roi_box = torch.FloatTensor(data_dict['params_3dmm']['roi_box'])[:, None]
        size = torch.FloatTensor(data_dict['params_3dmm']['size'])[:, None, None]
        theta = torch.FloatTensor(data_dict['params_ffhq']['theta'])
        crop_box = torch.FloatTensor(data_dict['crop_box'])[:, None]
        face_scale = torch.FloatTensor(data_dict['face_scale'])

        if self.align_source or self.align_target:
            # Align input images using theta
            eye_vector = torch.zeros(theta.shape[0], 1, 3)
            eye_vector[:, :, 2] = 1

            theta_ = torch.cat([theta, eye_vector], dim=1).float()

            # Perform 2x zoom-in compared to default theta
            scale = torch.zeros_like(theta_)
            scale[:, [0, 1], [0, 1]] = self.align_scale
            scale[:, 2, 2] = 1

            theta_ = torch.bmm(theta_, scale)[:, :2]

            align_warp = self.identity_grid.repeat_interleave(theta_.shape[0], dim=0)
            align_warp = align_warp.bmm(theta_.transpose(1, 2)).view(theta_.shape[0], self.image_size, self.image_size,
                                                                     2)

            if self.align_source:
                source_imgs_aligned = F.grid_sample(imgs[:n], align_warp[:n])
                source_masks_aligned = F.grid_sample(masks[:n], align_warp[:n])

            if self.align_target:
                target_imgs_aligned = F.grid_sample(imgs[-t:], align_warp[-t:])
                target_masks_aligned = F.grid_sample(masks[-t:], align_warp[-t:])

        output_data_dict = {
            'source_img': source_imgs_aligned if self.align_source else F.interpolate(imgs[:n], size=self.image_size,
                                                                                      mode='bilinear'),
            'source_mask': source_masks_aligned if self.align_source else F.interpolate(masks[:n], size=self.image_size,
                                                                                        mode='bilinear'),
            'source_keypoints': keypoints[:n],

            'target_img': target_imgs_aligned if self.align_target else F.interpolate(imgs[-t:], size=self.image_size,
                                                                                      mode='bilinear'),
            'target_mask': target_masks_aligned if self.align_target else F.interpolate(masks[-t:],
                                                                                        size=self.image_size,
                                                                                        mode='bilinear'),
            'target_keypoints': keypoints[-t:]
        }
        if self.return_keys:
            output_data_dict['keys'] = keys

        if self.output_aug_warp:
            output_data_dict['source_warp_aug'] = warp_aug[:n]
            output_data_dict['target_warp_aug'] = warp_aug[-t:]

        return output_data_dict

    def preprocess_data(self, data_dict):
        MIN_SCALE = 0.67
        n = 1
        t = 1

        for i in range(len(data_dict['image'])):
            image = data_dict['image'][i]
            mask = data_dict['mask'][i]
            size = data_dict['size'][i]
            face_scale = data_dict['face_scale'][i]
            keypoints = data_dict['keypoints'][i]

            use_geometric_augs = (i < n) and self.augment_geometric_source or (i == n) and self.augment_geometric_target

            if use_geometric_augs and face_scale >= MIN_SCALE:
                # Random sized crop
                min_scale = MIN_SCALE / face_scale
                seed = random.random()
                scale = seed * (1 - min_scale) + min_scale
                translate_x = random.random() * (1 - scale)
                translate_y = random.random() * (1 - scale)

            elif i > n:
                pass  # use params of the previous frame

            else:
                translate_x = 0
                translate_y = 0
                scale = 1

            crop_box = (size * translate_x,
                        size * translate_y,
                        size * (translate_x + scale),
                        size * (translate_y + scale))

            size_box = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])

            keypoints[..., 0] = (keypoints[..., 0] - crop_box[0]) / size_box[0] - 0.5
            keypoints[..., 1] = (keypoints[..., 1] - crop_box[1]) / size_box[1] - 0.5
            keypoints *= 2

            data_dict['keypoints'][i] = keypoints

            image = image.crop(crop_box)
            image = image.resize((self.image_size * 2, self.image_size * 2), Image.BICUBIC)

            mask = mask.crop(crop_box)
            mask = mask.resize((self.image_size * 2, self.image_size * 2), Image.BICUBIC)

            data_dict['image'][i] = image
            data_dict['mask'][i] = mask

            # Normalize crop_box to work with coords in [-1, 1]
            crop_box = ((translate_x - 0.5) * 2,
                        (translate_y - 0.5) * 2,
                        (translate_x + scale - 0.5) * 2,
                        (translate_y + scale - 0.5) * 2)

            data_dict['crop_box'].append(crop_box)

        return data_dict

    @staticmethod
    def augment_via_warp(images, image_size):
        # Implementation is based on DeepFaceLab repo
        # https://github.com/iperov/DeepFaceLab
        #
        # Performs an elastic-like transform for a uniform grid accross the image
        image_aug = []

        for image in images:
            cell_count = 8 + 1
            cell_size = image_size // (cell_count - 1)

            grid_points = np.linspace(0, image_size, cell_count)
            mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
            mapy = mapx.T

            mapx[1:-1, 1:-1] = mapx[1:-1, 1:-1] + np.random.normal(
                size=(cell_count - 2, cell_count - 2)) * cell_size * 0.1
            mapy[1:-1, 1:-1] = mapy[1:-1, 1:-1] + np.random.normal(
                size=(cell_count - 2, cell_count - 2)) * cell_size * 0.1

            half_cell_size = cell_size // 2

            mapx = cv2.resize(mapx, (image_size + cell_size,) * 2)[half_cell_size:-half_cell_size,
                   half_cell_size:-half_cell_size].astype(np.float32)
            mapy = cv2.resize(mapy, (image_size + cell_size,) * 2)[half_cell_size:-half_cell_size,
                   half_cell_size:-half_cell_size].astype(np.float32)

            image_aug += [cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)]

        return image_aug

    def __len__(self):
        if self.epoch_len == -1:
            return len(self.keys)
        else:
            return self.epoch_len


class DataModule(object):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("dataset")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--test_batch_size', default=1, type=int)
        parser.add_argument('--num_workers', default=16, type=int)
        parser.add_argument('--data_root', type=str)
        parser.add_argument('--image_size', default=256, type=int)
        parser.add_argument('--augment_geometric_source', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--augment_geometric_target', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--augment_color', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--return_keys', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--num_source_frames', default=1, type=int)
        parser.add_argument('--num_target_frames', default=1, type=int)
        parser.add_argument('--keys_name', default='keys_diverse_pose')

        parser.add_argument('--align_source', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--align_target', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--align_scale', default=1.0, type=float)

        parser.add_argument('--output_aug_warp', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--aug_warp_size', default=128, type=int)

        # These parameters can be used for debug
        parser.add_argument('--train_epoch_len', default=-1, type=int)
        parser.add_argument('--test_epoch_len', default=-1, type=int)

        return parser_out

    def __init__(self, args):
        super(DataModule, self).__init__()
        self.ddp = args.num_gpus > 1
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.num_workers
        self.data_root = args.data_root
        self.image_size = args.image_size
        self.align_source = args.align_source
        self.align_target = args.align_target
        self.align_scale = args.align_scale
        self.augment_geometric_source = args.augment_geometric_source
        self.augment_geometric_target = args.augment_geometric_target
        self.augment_color = args.augment_color
        self.return_keys = args.return_keys
        self.output_aug_warp = args.output_aug_warp
        self.aug_warp_size = args.aug_warp_size
        self.train_epoch_len = args.train_epoch_len
        self.test_epoch_len = args.test_epoch_len

        self.keys = {
            'test': pickle.load(open(f'{self.data_root}/lists/test_keys.pkl', 'rb')),
            'train': pickle.load(open(f'{self.data_root}/lists/train_keys.pkl', 'rb'))}

    def train_dataloader(self):
        train_dataset = LMDBDataset(self.data_root,
                                    self.image_size,
                                    self.keys['train'],
                                    'train',
                                    self.align_source,
                                    self.align_target,
                                    self.align_scale,
                                    self.augment_geometric_source,
                                    self.augment_geometric_target,
                                    self.augment_color,
                                    self.output_aug_warp,
                                    self.aug_warp_size,
                                    self.train_epoch_len,
                                    self.return_keys)

        shuffle = True
        sampler = None
        if self.ddp:
            shuffle = False
            sampler = data.distributed.DistributedSampler(train_dataset)

        return (
            data.DataLoader(train_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=True,
                            shuffle=shuffle,
                            sampler=sampler,
                            drop_last=True),
            sampler
        )

    def test_dataloader(self):
        test_dataset = LMDBDataset(self.data_root,
                                   self.image_size,
                                   self.keys['test'],
                                   'test',
                                   self.align_source,
                                   self.align_target,
                                   self.align_scale,
                                   return_keys=self.return_keys,
                                   epoch_len=self.test_epoch_len)

        sampler = None
        if self.ddp:
            sampler = data.distributed.DistributedSampler(test_dataset, shuffle=False)

        return data.DataLoader(test_dataset,
                               batch_size=self.test_batch_size,
                               num_workers=self.num_workers,
                               pin_memory=True,
                               sampler=sampler)