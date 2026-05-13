import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm

from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss, Pur_loss

from .base_model import BaseModel

logger = logging.getLogger("base")


class ReModel(BaseModel):
    def __init__(self, opt):
        super(ReModel, self).__init__(opt)

        os.makedirs('image', exist_ok=True)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.re_model = networks.define_RE(opt).to(self.device)
        self.ae_model = networks.define_AE(opt).to(self.device)

        for param in self.ae_model.parameters():
            param.requires_grad = False
                

        if opt["dist"]:
            self.re_model = DistributedDataParallel(
                self.re_model, device_ids=[torch.cuda.current_device()]
            )

        self.load()

        self.encode = self.ae_model.encode
        self.decode = self.ae_model.decode

        if self.is_train:
            self.re_model.train()

            self.loss_fn = Pur_loss().to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.re_model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.log_dict = OrderedDict()

    def feed_data(self, LQ_fea, LQ, GT_fea=None, GT=None):
        # LQ_fea & GT_fea is a list
        self.LQ_image = LQ.to(self.device)
        self.latent_LQ = LQ_fea[0].to(self.device) # latent
        self.hidden_LQ = []
        if GT is not None and GT_fea is not None:
            self.GT_image = GT.to(self.device)
            self.latent_GT = GT_fea[0].to(self.device) # latent
            self.hidden_GT = []
                
        for i in range(1, len(LQ_fea)):
            self.hidden_LQ.append(LQ_fea[i].to(self.device))
            if GT is not None and GT_fea is not None:
                self.hidden_GT.append(GT_fea[i].to(self.device))
        

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        if self.opt["dist"]:
            RE_fn = self.re_model.module
        else:
            RE_fn = self.re_model

        latent_pur = RE_fn(self.latent_LQ)
        
        # first decode latent to image
        LQ_recon = self.decode(latent_pur,self.hidden_LQ)

                       
        # caculate the mse between Fuse image embeddings and text embeddings
        total_loss, loss_fea, loss_image = self.loss_fn(latent_pur, self.latent_GT, LQ_recon, self.GT_image)

             
        total_loss.backward()
        self.optimizer.step()
        
        # set log
        self.log_dict["total_loss"] = total_loss.item()
        self.log_dict["loss_fea"] = loss_fea.item()
        self.log_dict["loss_image"] = loss_image.item()

    def test(self):
        self.re_model.eval()

        if self.opt["dist"]:
            RE_fn = self.re_model.module
        else:
            RE_fn = self.re_model

        with torch.no_grad():
            latent_pur = RE_fn(self.latent_LQ)
            self.output = self.decode(latent_pur, self.hidden_LQ)
            self.Rec_GT_img = self.decode(self.latent_GT, self.hidden_GT)
            self.Rec_LQ_img  = self.decode(self.latent_LQ, self.hidden_LQ)
            
                   
        self.re_model.train()
        
        tvutils.save_image(self.LQ_image.data, f'image/LQ.png', normalize=False)
        tvutils.save_image(self.GT_image.data, f'image/GT.png', normalize=False)
        tvutils.save_image(self.output.data, f'image/Enhanced.png', normalize=False)
        tvutils.save_image(self.Rec_GT_img.data, f'image/Recon_GT.png', normalize=False)
        tvutils.save_image(self.Rec_LQ_img.data, f'image/Recon_LQ.png', normalize=False)
             

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.LQ_image.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.GT_image.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        print("start to load pretrained model...")

        load_path_AE = self.opt["path"]["pretrain_model_AE"]
        if load_path_AE is not None:
            logger.info("Loading model for AutoEncoder [{:s}] ...".format(load_path_AE))
            self.load_network(load_path_AE, self.ae_model, self.opt["path"]["strict_load"])

        load_path_RE = self.opt["path"]["pretrain_model_RE"]
        if load_path_RE is not None:
            logger.info("Loading model for REModel [{:s}] ...".format(load_path_RE))
            self.load_network(load_path_RE, self.re_model, self.opt["path"]["strict_load"])


    def save(self, iter_label):
        self.save_network(self.re_model, "RE", iter_label)


