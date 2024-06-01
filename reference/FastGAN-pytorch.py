import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import inception_v3, Inception3
from torchvision.utils import save_image

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import numpy as np
from scipy import linalg
from tqdm import tqdm
import pickle
import os

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008,
                                    aux_logits=False,
                                    pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features



@torch.no_grad()
def extract_feature_from_samples(generator, inception, device='cuda'):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


@torch.no_grad()
def extract_feature_from_generator_fn(generator_fn, inception, device='cuda', total=1000):
    features = []
    for batch in tqdm(generator_fn, total=total):
        feat = inception(batch)[0].view(batch.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0).detach()
    return features.numpy()


def calc_fid(sample_features, real_features=None, real_mean=None, real_cov=None, eps=1e-6):
    sample_mean = np.mean(sample_features, 0)
    sample_cov = np.cov(sample_features, rowvar=False)

    if real_features is not None:
        real_mean = np.mean(real_features, 0)
        real_cov = np.cov(real_features, rowvar=False)

    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == "__main__":
    #from utils import PairedMultiDataset, InfiniteSamplerWrapper, make_folders, AverageMeter
    from torch.utils.data import DataLoader
    from torchvision import utils as vutils
    
    IM_SIZE = 1024
    BATCH_SIZE = 16
    DATALOADER_WORKERS = 8
    NBR_CLS = 2000
    TRIAL_NAME = 'trial_vae_512_1'
    SAVE_FOLDER = './'

    from torchvision.datasets import ImageFolder

    '''
    data_root_colorful = '../images/celebA/CelebA_512/img'
    data_root_sketch_1 = './sketch_simplification/vggadin_iter_700'
    data_root_sketch_2 = './sketch_simplification/vggadin_iter_1900'
    data_root_sketch_3 = './sketch_simplification/vggadin_iter_2300'

    dataset = PairedMultiDataset(data_root_colorful, data_root_sketch_1, data_root_sketch_2, data_root_sketch_3, im_size=IM_SIZE, rand_crop=False)
    dataloader = iter(DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKERS, pin_memory=True))


    from pretrain_ae import StyleEncoder, ContentEncoder, Decoder
    import pickle
    from refine_ae_as_gan import AE, RefineGenerator
    from utils import load_params

    net_ig = RefineGenerator().cuda()
    net_ig = nn.DataParallel(net_ig)

    ckpt = './train_results/trial_refine_ae_as_gan_1024_2/models/4.pth'
    if ckpt is not None:
        ckpt = torch.load(ckpt)
        #net_ig.load_state_dict(ckpt['ig'])
        #net_id.load_state_dict(ckpt['id'])
        net_ig_ema = ckpt['ig_ema']
        load_params(net_ig, net_ig_ema)
    net_ig = net_ig.module
    #net_ig.eval()

    net_ae = AE()
    net_ae.load_state_dicts('./train_results/trial_vae_512_1/models/176000.pth')
    net_ae.cuda()
    net_ae.eval()

    #style_encoder = StyleEncoder(nbr_cls=NBR_CLS).cuda()
    #content_encoder = ContentEncoder().cuda()
    #decoder = Decoder().cuda()
    '''

    def real_image_loader(dataloader, n_batches=10):
        counter = 0
        while counter < n_batches:
            counter += 1
            rgb_img, _ = next(dataloader)
            if counter == 1:
                vutils.save_image(0.5*(rgb_img+1), 'tmp_real.jpg')  
            yield rgb_img.cuda()

    '''
    @torch.no_grad()
    def image_generator_1(dataloader, n_batches=10):
        counter = 0
        while counter < n_batches:
            counter += 1
            rgb_img, _, _, skt_img = next(dataloader)
            rgb_img = rgb_img.cuda()
            skt_img = skt_img.cuda()

            style_feat, _ = style_encoder(rgb_img)
            content_feats = content_encoder( F.interpolate( skt_img , size=512 ) )
            gimg = decoder(content_feats, style_feat)

            vutils.save_image(0.5*(gimg+1), 'tmp.jpg')        
            yield gimg

    from utils import true_randperm
    @torch.no_grad()
    def image_generator(dataset, net_ae, net_ig, n_batches=500):
        counter = 0
        dataloader = iter(DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKERS, pin_memory=False))

        while counter < n_batches:
            counter += 1
            rgb_img, _, _, skt_img = next(dataloader)
            rgb_img = F.interpolate( rgb_img, size=512 ).cuda()
            skt_img = F.interpolate( skt_img, size=512 ).cuda()

            #perm = true_randperm(rgb_img.shape[0], device=rgb_img.device)

            gimg_ae, style_feat = net_ae(skt_img, rgb_img)
            g_image = net_ig(gimg_ae, style_feat, skt_img)
            if counter == 1:
                vutils.save_image(0.5*(g_image+1), 'tmp.jpg')        
            yield g_image
    '''
    inception = load_patched_inception_v3().cuda()
    inception.eval()
    
    path_a = '../../../database/images/celebaMask/CelebA_1024'
    path_b = '../../stylegan/celebahq_samples'

    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize( (299, 299) ),
            #transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset_a = ImageFolder(path_a, transform)
    loader_a = iter(DataLoader(dset_a, batch_size=16, num_workers=4))

    real_features = extract_feature_from_generator_fn( 
        real_image_loader(loader_a, n_batches=900), inception )
    real_mean = np.mean(real_features, 0)
    real_cov = np.cov(real_features, rowvar=False)
    
    #pickle.dump({'feats': real_features, 'mean': real_mean, 'cov': real_cov}, open('celeba_fid_feats.npy','wb') ) 

    #real_features = pickle.load( open('celeba_fid_feats.npy', 'rb') )
    #real_mean = real_features['mean']
    #real_cov = real_features['cov']
    #sample_features = extract_feature_from_generator_fn( real_image_loader(dataloader, n_batches=100), inception )
    
    dset_b = ImageFolder(path_b, transform)
    loader_b = iter(DataLoader(dset_b, batch_size=16, num_workers=4))

    sample_features = extract_feature_from_generator_fn( 
        real_image_loader(loader_b, n_batches=900), inception )
    #sample_features = extract_feature_from_generator_fn( 
    #        image_generator(dataset, net_ae, net_ig, n_batches=1800), inception,
    #         total=1800 )

        #fid = calc_fid(sample_features, real_mean=real_features['mean'], real_cov=real_features['cov'])
    fid = calc_fid(sample_features, real_mean=real_mean, real_cov=real_cov)
        
    print(fid)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008,
                                    aux_logits=False,
                                    pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from calc_inception import load_patched_inception_v3
import os

@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img,_ in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--path_a', type=str)
    parser.add_argument('--path_b', type=str)
    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--end', type=int, default=13)

    args = parser.parse_args()

    inception = load_patched_inception_v3().eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize( (args.size, args.size) ),
            #transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset_a = ImageFolder(args.path_a, transform)
    loader_a = DataLoader(dset_a, batch_size=args.batch, num_workers=4)

    features_a = extract_features(loader_a, inception, device).numpy()
    print(f'extracted {features_a.shape[0]} features')

    real_mean = np.mean(features_a, 0)
    real_cov = np.cov(features_a, rowvar=False)
    
    #for folder in os.listdir(args.path_b):
    for folder in range(args.iter,args.end+1):
        folder = 'eval_%d'%(folder*10000)
        if os.path.exists(os.path.join( args.path_b, folder )):
            print(folder)
            dset_b = ImageFolder( os.path.join( args.path_b, folder ), transform)
            loader_b = DataLoader(dset_b, batch_size=args.batch, num_workers=4)

            features_b = extract_features(loader_b, inception, device).numpy()
            print(f'extracted {features_b.shape[0]} features')

            sample_mean = np.mean(features_b, 0)
            sample_cov = np.cov(features_b, rowvar=False)

            fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

            print(folder, ' fid:', fid)

import argparse
import pickle
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception3
import numpy as np
from tqdm import tqdm

from inception import InceptionV3
from torchvision.datasets import ImageFolder

class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img,_ in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description='Calculate Inception v3 features for datasets'
    )
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('path', metavar='PATH', help='path to datset lmdb file')

    args = parser.parse_args()

    inception = load_patched_inception_v3().eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize( (args.size, args.size) ),
            transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset = ImageFolder(args.path, transform)
    loader = DataLoader(dset, batch_size=args.batch, num_workers=4)

    features = extract_features(loader, inception, device).numpy()

    features = features[: args.n_sample]

    print(f'extracted {features.shape[0]} features')

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    name = os.path.splitext(os.path.basename(args.path))[0]

    print({'mean': mean.mean(), 'cov': cov.mean()})
    with open(f'inception_{name}.pkl', 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': args.size, 'path': args.path}, f)

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
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')



from __future__ import absolute_import

import sys
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from scipy.ndimage import zoom
import fractions
import functools
import skimage.transform
from tqdm import tqdm

from IPython import embed

from . import networks_basic as networks
import lpips as util

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
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,
                use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize), '..', 'weights/v%s/%s.pth'%(version,net)))

            if(not is_train):
                print('Loading model from: %s'%model_path)
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)

        elif(self.model=='net'): # pretrained network
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = networks.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = networks.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = networks.BCERankingLoss()
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
            networks.print_network(self.net)
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

        ref_img = util.tensor2im(self.var_ref.data)
        p0_img = util.tensor2im(self.var_p0.data)
        p1_img = util.tensor2im(self.var_p1.data)

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
    score = util.voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))

from collections import namedtuple
import torch
from torchvision import models as tv
from IPython import embed
from torchvision.models import VGG16_Weights,SqueezeNet1_1_Weights,AlexNet_Weights,ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights

class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT).features
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
        alexnet_pretrained_features = tv.alexnet(weights=AlexNet_Weights.DEFAULT).features
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
        vgg_pretrained_features = tv.vgg16(weights=VGG16_Weights.DEFAULT).features
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
            self.net = tv.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif(num==34):
            self.net = tv.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif(num==50):
            self.net = tv.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif(num==101):
            self.net = tv.resnet101(weights=ResNet101_Weights.DEFAULT)
        elif(num==152):
            self.net = tv.resnet152(weights=ResNet152_Weights.DEFAULT)
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage
import torch
from torch.autograd import Variable

from lpips import dist_model


if skimage.__version__ == '0.14.3':
    from skimage.measure import compare_ssim
else:
    from skimage.metrics import structural_similarity as compare_ssim



class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0]): # VGG using our perceptually-learned weights (LPIPS metric)
    # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG as a perceptual loss
        super(PerceptualLoss, self).__init__()
        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace, spatial=self.spatial, gpu_ids=gpu_ids)
        print('...[%s] initialized'%self.model.name())
        print('...Done')

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


from __future__ import absolute_import

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as st
from skimage import color
from IPython import embed
from . import pretrained_networks as pn

import lpips as util

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
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
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
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
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
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = util.dssim(1.*util.tensor2im(in0.data), 1.*util.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
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

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    save_interval = args.save_interval
    saved_model_folder, saved_image_folder = get_dir(args)

    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

   
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))
          
        if iteration % (save_interval*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--output_path', type=str, default='./', help='Output path for the train results')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--save_interval', type=int, default=100, help='number of iterations to save model')

    args = parser.parse_args()
    print(args)

    train(args)

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from models import Generator


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img,size=256):
    return F.interpolate(img, size=size)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--artifacts', type=str, default=".", help='path to artifacts.')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=6)
    parser.add_argument('--end_iter', type=int, default=10)

    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=1024)
    parser.add_argument('--multiplier', type=int, default=10000, help='multiplier for model number')
    parser.set_defaults(big=False)
    args = parser.parse_args()

    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)#, big=args.big )
    net_ig.to(device)

    for epoch in [args.multiplier*i for i in range(args.start_iter, args.end_iter+1)]:
        ckpt = f"{args.artifacts}/models/{epoch}.pth"
        checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
        # Remove prefix `module`.
        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net_ig.load_state_dict(checkpoint['g'])
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print('load checkpoint success, epoch %d'%epoch)

        net_ig.to(device)

        del checkpoint

        dist = 'eval_%d'%(epoch)
        dist = os.path.join(dist, 'img')
        os.makedirs(dist, exist_ok=True)

        with torch.no_grad():
            for i in tqdm(range(args.n_sample//args.batch)):
                noise = torch.randn(args.batch, noise_dim).to(device)
                g_imgs = net_ig(noise)[0]
                g_imgs = resize(g_imgs,args.im_size) # resize the image using given dimension
                for j, g_img in enumerate( g_imgs ):
                    vutils.save_image(g_img.add(1).mul(0.5), 
                        os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))

# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
from eval import load_params
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torchvision import transforms
import os
import random
import argparse
from tqdm import tqdm

from models import Generator
from operation import load_params, InfiniteSamplerWrapper

noise_dim = 256
device = torch.device('cuda:%d'%(0))

im_size = 512  
net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=im_size)#, big=args.big )
net_ig.to(device)

epoch = 50000
ckpt = './models/all_%d.pth'%(epoch)
checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
net_ig.load_state_dict(checkpoint['g'])
load_params(net_ig, checkpoint['g_ema'])

batch = 8
noise = torch.randn(batch, noise_dim).to(device)
g_imgs = net_ig(noise)[0]

vutils.save_image(g_imgs.add(1).mul(0.5), 
                    os.path.join('./', '%d.png'%(2)))


transform_list = [
            transforms.Resize((int(256),int(256))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
trans = transforms.Compose(transform_list)
data_root = '/media/database/images/first_1k'
dataset = ImageFolder(root=data_root, transform=trans)

import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

the_image = g_imgs[0].unsqueeze(0)
def find_closest(the_image):
    the_image = F.interpolate(the_image, size=256)
    small = 100
    close_image = None
    for i in tqdm(range(len(dataset))):
        real_iamge = dataset[i][0].unsqueeze(0).to(device)

        dis = percept(the_image, real_iamge).sum()
        if dis < small:
            small = dis
            close_image = real_iamge
    return close_image, small

all_dist = []
batch = 8
result_path = 'nn_track'
import os
os.makedirs(result_path, exist_ok=True)
for j in range(8):
    with torch.no_grad():
        noise = torch.randn(batch, noise_dim).to(device)
        g_imgs = net_ig(noise)[0]

    for n in range(batch):
        the_image = g_imgs[n].unsqueeze(0)

        close_0, dis = find_closest(the_image)
        
        vutils.save_image(torch.cat([F.interpolate(the_image,256), close_0]).add(1).mul(0.5), \
            result_path+'/nn_%d.jpg'%(j*batch+n))
        all_dist.append(dis.view(1))

new_all_dist = []
for v in all_dist:
    new_all_dist.append(v.view(1))
print(torch.cat(new_all_dist).mean())
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator, SimpleDecoder
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        #pred, [rec_all, rec_small, rec_part], part = net(data, label)
        pred = net(data, label)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() #+ \
            #percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            #percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            #percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item()#, rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        
@torch.no_grad()
def interpolate(z1, z2, netG, img_name, step=8):
    z = [  a*z2 + (1-a)*z1 for a in torch.linspace(0, 1, steps=step)  ]
    z = torch.cat(z).view(step, -1)
    g_image = netG(z)[0]
    vutils.save_image( g_image.add(1).mul(0.5), img_name , nrow=step)


def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    dataset = ImageFolder(root=data_root, transform=trans)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))


    
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)

    
    ckpt = torch.load(checkpoint)
    load_params( netG , ckpt['g_ema'] )
    #netG.eval()
    netG.to(device)

    fixed_noise = torch.randn(batch_size, nz, requires_grad=True, device=device)
    optimizerG = optim.Adam([fixed_noise], lr=0.1, betas=(nbeta1, 0.999))
    
    real_image = next(dataloader).to(device)

    log_rec_loss = 0

    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        
        optimizerG.zero_grad()
        
        g_image = netG(fixed_noise)[0]

        rec_loss = percept( F.avg_pool2d( g_image, 2, 2), F.avg_pool2d(real_image,2,2) ).sum() + 0.2*F.mse_loss(g_image, real_image)

        rec_loss.backward()

        optimizerG.step()

        log_rec_loss += rec_loss.item()

        if iteration % 100 == 0:
            print("lpips loss g: %.5f"%(log_rec_loss/100))
            log_rec_loss = 0

        if iteration % (save_interval*2) == 0:
            
            with torch.no_grad():
                vutils.save_image( torch.cat([
                        real_image, g_image]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )

                interpolate(fixed_noise[0], fixed_noise[1], netG, saved_image_folder+'/interpolate_0_1_%d.jpg'%iteration)
        
        if iteration % (save_interval*5) == 0 or iteration == total_iterations:
            torch.save(fixed_noise, saved_model_folder+'/%d.pth'%iteration)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path')


    args = parser.parse_args()
    print(args)

    train(args)
import torch
from torch import nn, real, select
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator, SimpleDecoder
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        #pred, [rec_all, rec_small, rec_part], part = net(data, label)
        pred = net(data, label)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() #+ \
            #percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            #percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            #percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item()#, rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        
@torch.no_grad()
def interpolate(z1, z2, netG, img_name, step=8):
    z = [  a*z2 + (1-a)*z1 for a in torch.linspace(0, 1, steps=step)  ]
    z = torch.cat(z).view(step, -1)
    g_image = netG(z)[0]
    vutils.save_image( g_image.add(1).mul(0.5), img_name , nrow=step)


def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    dataset = ImageFolder(root=data_root, transform=trans, return_idx=True)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))

    total_iterations = int(len(dataset)*100/batch_size)
    
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)

    
    ckpt = torch.load(checkpoint)
    load_params( netG , ckpt['g_ema'] )
    #netG.eval()
    netG.to(device)

    fixed_noise = torch.randn(len(dataset), nz, requires_grad=True, device=device)
    optimizerG = optim.Adam([fixed_noise], lr=0.1, betas=(nbeta1, 0.999))

    log_rec_loss = 0


    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image, noise_idx = next(dataloader)
        real_image = real_image.to(device)

        optimizerG.zero_grad()
        
        select_noise = fixed_noise[noise_idx]
        g_image = netG(select_noise)[0]

        rec_loss = percept( F.avg_pool2d( g_image, 2, 2), F.avg_pool2d(real_image,2,2) ).sum() + 0.2*F.mse_loss(g_image, real_image)

        rec_loss.backward()

        optimizerG.step()

        log_rec_loss += rec_loss.item()

        if iteration % 100 == 0:
            print("lpips loss g: %.5f"%(log_rec_loss/100))
            log_rec_loss = 0

        if iteration % (save_interval*10) == 0:
            
            with torch.no_grad():
                vutils.save_image( torch.cat([
                        real_image, g_image]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration , nrow=batch_size)

                interpolate(fixed_noise[0], fixed_noise[1], netG, saved_image_folder+'/interpolate_0_1_%d.jpg'%iteration)
        
        if iteration % (save_interval*10) == 0 or iteration == total_iterations:
            torch.save(fixed_noise, saved_model_folder+'/%d.pth'%iteration)
            
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers, pin_memory=True)
    
    mean_lpips = 0
    for idx, data in enumerate(dataloader):
        real_image, noise_idx = data
        real_image = real_image.to(device)

        select_noise = fixed_noise[noise_idx]
        g_image = netG(select_noise)[0]

        rec_loss = percept( F.avg_pool2d( g_image, 2, 2), F.avg_pool2d(real_image,2,2) ).sum() 
        mean_lpips += rec_loss.sum()
    mean_lpips /= len(dataset)
    print(mean_lpips)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=4, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path')


    args = parser.parse_args()
    print(args)

    train(args)
from easing_functions.easing import LinearInOut
import torch
import pandas as pd 
from torchvision import utils as vutils
import os
import cv2
from tqdm import tqdm
from scipy import io
import numpy as np
import argparse

from easing_functions import QuadEaseInOut
from easing_functions import SineEaseIn, SineEaseInOut, SineEaseOut
from easing_functions import ElasticEaseIn, ElasticEaseInOut, ElasticEaseOut

ease_fn_dict = {'QuadEaseInOut': QuadEaseInOut,
                'SineEaseIn': SineEaseIn, 
                'SineEaseInOut': SineEaseInOut, 
                'SineEaseOut': SineEaseOut,
                'ElasticEaseIn': ElasticEaseIn, 
                'ElasticEaseInOut': ElasticEaseInOut, 
                'ElasticEaseOut': ElasticEaseOut,
                'Linear': LinearInOut}

def interpolate(z1, z2, num_interp):
    # this is a "first frame included, last frame excluded" interpolation
    w = torch.linspace(0, 1, num_interp+1)
    interp_zs = []
    for n in range(num_interp):
        interp_zs.append( (z2*w[n].item() + z1*(1-w[n].item())).unsqueeze(0) )
    return torch.cat(interp_zs)



def interpolate_ease_inout(z1, z2, num_interp, ease_fn, model_type='freeform'):
    # this is a "first frame included, last frame excluded" interpolation
    w = ease_fn(start=0, end=1, duration=num_interp+1)
    interp_zs = []

    # just to make sure the latent vectors in the right shape
    if model_type == 'freeform':
        z1 = z1.view(1, -1)
        z2 = z2.view(1, -1)
    if model_type == 'stylegan2':
        if type(z1) is list:
            z1 = [z1[0].view(1, -1), z1[1].view(1, -1)]
        else:
            z1 = [z1.view(1, -1), z1.view(1, -1)]
        if type(z2) is list:
            z2 = [z2[0].view(1, -1), z2[1].view(1, -1)]
        else:
            z2 = [z2.view(1, -1), z2.view(1, -1)]

    for n in range(num_interp):
        if model_type == 'freeform':
            interp_zs.append( z2*w.ease(n) + z1*(1-w.ease(n)) )
        if model_type == 'stylegan2':
            interp_zs.append( [ z2[0]*w.ease(n) + z1[0]*(1-w.ease(n)),
                                z2[1]*w.ease(n) + z1[1]*(1-w.ease(n)) ] )
    return interp_zs

@torch.no_grad()
def net_generate(netG, z, model_type='freeform', im_size=1024):
    
    if model_type == 'stylegan2':
        z_contents = []
        z_styles = []
        for zidx in range(len(z)):
            z_contents.append(z[zidx][0])
            z_styles.append(z[zidx][1])
        z = [ torch.cat(z_contents), torch.cat(z_styles) ]
        gimg = netG( z, inject_index=8, input_is_latent=True, randomize_noise=False )[0].cpu()
    elif model_type == 'freeform':
        z = torch.cat(z)
        gimg = netG(z)[0].cpu()

    return torch.nn.functional.interpolate(gimg, im_size)

def batch_generate_and_save(netG, zs, folder_name, batch_size=8, model_type='freeform', im_size=1024):
    # zs is a list of vectors if model is freeform
    # zs is a list of lists, each list is 2 vectors, if model is stylegan
    t = 0
    num = 0
    if len(zs) < batch_size:
        gimgs = net_generate(netG, zs, model_type, im_size=im_size).cpu()
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.jpg"%(num) )
            num += 1

    for k in tqdm(range(len(zs)//batch_size)):
        gimgs = net_generate(netG, zs[k*batch_size:(k+1)*batch_size], model_type, im_size=im_size)
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.jpg"%(num) )
            num += 1
        t = k

    if len(zs)%batch_size>0:
        gimgs = net_generate(netG, zs[(t+1)*batch_size:], model_type, im_size=im_size)        
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.jpg"%(num) )
            num += 1



def batch_save(images, folder_name, start_num=0):
    os.makedirs(folder_name, exist_ok=True)
    num = start_num
    for image in images:
        vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.jpg"%(num) )
        num += 1


def read_img_and_make_video(dist, video_name, fps):
    img_array = []
    for i in tqdm(range(len(os.listdir(dist)))):
        try:
            filename = dist+'/%d.jpg'%(i)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        except:
            print('error at: %d'%i)
    
    if '.mp4' not in video_name:
        video_name += '.mp4'
    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

from shutil import rmtree

def make_video_from_latents(net, selected_latents, frames_dist_folder, video_name, fps, video_length, ease_fn, model_type, im_size=1024):
    # selected_latents: the latent noise of user selected key-frame images, it is a list
    # each item in the list is a vector if the model is freeform, 
    # each item in the list is a list of two vectors if the model is stylegan2
    # frames_dist_folder: the folder path to save the generated images to make the video
    # fps: is the frames we generate per second
    # video_length: is the time of the video, in seconds. For example: 30 means a video length of 30 seconds
    # ease_fn: user selected type of transitions between each key-frame

    # first calculate how many images need to generate
    try:
        rmtree(frames_dist_folder)
    except:
        pass
    os.makedirs(frames_dist_folder, exist_ok=True)

    nbr_generate = fps*video_length
    nbr_keyframe = len(selected_latents)
    nbr_interpolation = 1 + nbr_generate // (nbr_keyframe - 1)
    

    main_zs = []
    for idx in range(nbr_keyframe-1):
        main_zs += interpolate_ease_inout(selected_latents[idx], 
                            selected_latents[idx+1], nbr_interpolation, ease_fn, model_type)
    

    print('generating images ...')
    batch_generate_and_save(net, main_zs, folder_name=frames_dist_folder, batch_size=8, model_type=model_type, im_size=im_size)
    print('making videos ...')
    read_img_and_make_video(frames_dist_folder, video_name, fps=fps)


if __name__ == "__main__":


    device = torch.device('cuda:%d'%(0))

    load_model_err = 0

    from models import Generator as Generator_freeform
    
    frames_dist_folder = 'project_video_frames' # a folder to save generated images
    ckpt_path = './time_1024_1/models/180000.pth' # path to the checkpoint
    video_name = 'videl_keyframe_15'  # name of the generated video

    model_type = 'freeform'
    net = Generator_freeform(ngf=64, nz=100)
    net.load_state_dict(torch.load(ckpt_path)['g'])
    net.to(device)
    net.eval()

    
    try:
        rmtree(frames_dist_folder)
    except:
        pass
    os.makedirs(frames_dist_folder, exist_ok=True)

    fps = 30
    minutes = 1
    im_size = 1024
    
    ease_fn=ease_fn_dict['SineEaseInOut']

    init_kf_nbr = 15
    nbr_key_frames_per_minute = [init_kf_nbr-i for i in range(minutes)]
    nbr_key_frames_total = sum(nbr_key_frames_per_minute)
    noises = torch.randn( nbr_key_frames_total , 100).to(device)
    user_selected_noises = [n for n in noises]
    nbr_interpolation_list = [[fps*60//nbr_kf]*nbr_kf for nbr_kf in nbr_key_frames_per_minute]
    nbl = []
    for nb in nbr_interpolation_list:
        nbl += nb

    print(len(nbl)) 
    print(len(user_selected_noises))# , print("mismatch size")
    main_zs = []
    for idx in range(len(user_selected_noises)-1):
        main_zs += interpolate_ease_inout(user_selected_noises[idx], 
                            user_selected_noises[idx+1], nbl[idx], ease_fn, model_type)
    for idx in range(100):
        main_zs.append(main_zs[-1])
    print('generating images ...')
    batch_generate_and_save(net, main_zs, folder_name=frames_dist_folder, batch_size=8, model_type=model_type, im_size=im_size)
    print('making videos ...')
    read_img_and_make_video(frames_dist_folder, video_name, fps=fps)


import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment



ndf = 64
ngf = 64
nz = 256
nlr = 0.0002
nbeta1 = 0.5
use_cuda = True
multi_gpu = False
dataloader_workers = 8
current_iteration = 0
save_interval = 100
device = 'cuda:0'
im_size = 256


netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
netG.apply(weights_init)

netD = Discriminator(ndf=ndf, im_size=im_size)
netD.apply(weights_init)

netG.to(device)
netD.to(device)

avg_param_G = copy_G_params(netG)

fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

j = 4
checkpoint = "./models/all_%d.pth"%(j*10000)
ckpt = torch.load(checkpoint)
netG.load_state_dict(ckpt['g'])
netD.load_state_dict(ckpt['d'])
avg_param_G = ckpt['g_ema']
load_params(netG, avg_param_G)

bs = 8
noise_a = torch.randn(bs, nz).to(device)
noise_b = torch.randn(bs, nz).to(device)

def get_early_features(net, noise):
    feat_4 = net.init(noise)
    feat_8 = net.feat_8(feat_4)
    feat_16 = net.feat_16(feat_8)
    feat_32 = net.feat_32(feat_16)
    feat_64 = net.feat_64(feat_32)
    return feat_8, feat_16, feat_32, feat_64

def get_late_features(net, im_size, feat_64, feat_8, feat_16, feat_32):
    feat_128 = net.feat_128(feat_64)
    feat_128 = net.se_128(feat_8, feat_128)

    feat_256 = net.feat_256(feat_128)
    feat_256 = net.se_256(feat_16, feat_256)
    if im_size==256:
        return net.to_big(feat_256)
    
    feat_512 = net.feat_512(feat_256)
    feat_512 = net.se_512(feat_32, feat_512)
    if im_size==512:
        return net.to_big(feat_512)
    
    feat_1024 = net.feat_1024(feat_512)
    return net.to_big(feat_1024)


feat_8_a, feat_16_a, feat_32_a, feat_64_a = get_early_features(netG, noise_a)
feat_8_b, feat_16_b, feat_32_b, feat_64_b = get_early_features(netG, noise_b)

images_b = get_late_features(netG, im_size, feat_64_b, feat_8_b, feat_16_b, feat_32_b)
images_a = get_late_features(netG, im_size, feat_64_a, feat_8_a, feat_16_a, feat_32_a)

imgs = [ torch.ones(1, 3, im_size, im_size) ]
imgs.append(images_b.cpu())
for i in range(bs):
    imgs.append(images_a[i].unsqueeze(0).cpu())

    gimgs = get_late_features(netG, im_size, feat_64_a[i].unsqueeze(0).repeat(bs, 1, 1, 1), feat_8_b, feat_16_b, feat_32_b)
    imgs.append(gimgs.cpu())

imgs = torch.cat(imgs)
vutils.save_image(imgs.add(1).mul(0.5), 'style_mix_1.jpg', nrow=bs+1)
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(Generator, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])
                                
        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])  
        self.feat_256 = UpBlock(nfc[128], nfc[256]) 

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False) 
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False) 
        
        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512]) 
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])  
        
    def forward(self, input):
        
        feat_4   = self.init(input)
        feat_8   = self.feat_8(feat_4)
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)

        feat_64  = self.se_64( feat_4, self.feat_64(feat_32) )

        feat_128 = self.se_128( feat_8, self.feat_128(feat_64) )

        feat_256 = self.se_256( feat_16, self.feat_256(feat_128) )

        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]
        
        feat_512 = self.se_512( feat_32, self.feat_512(feat_256) )
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                                    batchNorm2d(nfc[512]),
                                    nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )

        self.down_4  = DownBlockComp(nfc[512], nfc[256])
        self.down_8  = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
                            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
                            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
                            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])
        
        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)
        
    def forward(self, imgs, label, part=None):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2 = self.down_from_big(imgs[0])        
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)
        
        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)
        
        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        #rf_0 = torch.cat([self.rf_big_1(feat_last).view(-1),self.rf_big_2(feat_last).view(-1)])
        #rff_big = torch.sigmoid(self.rf_factor_big)
        rf_0 = self.rf_big(feat_last).view(-1)

        feat_small = self.down_from_small(imgs[1])
        #rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        rf_1 = self.rf_small(feat_small).view(-1)

        if label=='real':    
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,8:])

            return torch.cat([rf_0, rf_1]) , [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1]) 


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

from random import randint
def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h-size-1)
    cw = randint(0, w-size-1)
    return image[:,:,ch:ch+size,cw:cw+size]

class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )
        self.rf_small = nn.Sequential(
                            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)
        
    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)
        
        if label=='real':    
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf
import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import shutil
import json

def InfiniteSampler(n):
    """Data sampler"""
    # check if the number of samples is valid
    if n <= 0:
        raise ValueError(f"Invalid number of samples: {n}.\nMake sure that images are present in the given path.")
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
    

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def get_dir(args):

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    task_name = os.path.join(args.output_path, 'train_results', args.name)
    saved_model_folder = os.path.join(task_name, 'models')
    saved_image_folder = os.path.join(task_name, 'images')
    
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    for f in os.listdir('./'):
        if '.py' in f:
            shutil.copy(f, os.path.join(task_name, f))
    
    with open(os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder




class  ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None):
        super( ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg': 
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
            
        if self.transform:
            img = self.transform(img) 

        return img



from io import BytesIO
import lmdb
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            #key_asp = f'aspect_ratio-{str(index).zfill(5)}'.encode('utf-8')
            #aspect_ratio = float(txn.get(key_asp).decode())

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


