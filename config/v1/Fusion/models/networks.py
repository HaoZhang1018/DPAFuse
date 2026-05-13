import logging

import torch

from models import modules as M

logger = logging.getLogger("base")

# Generator
def define_RE(opt):
    opt_net = opt["network_RE"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netRE = getattr(M, which_model)(**setting)
    return netRE


def define_AE(opt):
    opt_net = opt["network_AE"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netAE = getattr(M, which_model)(**setting)
    return netAE

def define_Fusion(opt):
    opt_net = opt["network_Fusion"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netFusion = getattr(M, which_model)(**setting)
    return netFusion


# Discriminator
def define_D_fea(opt):
    opt_net = opt["network_D_fea"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netDfea = getattr(M, which_model)(**setting)
    return netDfea

def define_D_img(opt):
    opt_net = opt["network_D_img"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netimg = getattr(M, which_model)(**setting)
    return netimg


# Perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = M.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train
    return netF
