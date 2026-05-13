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

from models.modules.loss import Fusion_loss, Dis_Fea_loss, Dis_Img_loss

from .base_model import BaseModel

logger = logging.getLogger("base")


class FIGAN_Model(BaseModel):
    def __init__(self, opt):
        super(FIGAN_Model, self).__init__(opt)

        os.makedirs('image', exist_ok=True)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.ae_model = networks.define_AE(opt).to(self.device)
        self.re_model_X = networks.define_RE(opt).to(self.device)
        self.re_model_Y = networks.define_RE(opt).to(self.device)
        self.fusion_model = networks.define_Fusion(opt).to(self.device)
        self.dis_fea = networks.define_D_fea(opt).to(self.device)
        self.dis_img = networks.define_D_img(opt).to(self.device)

        for param in self.ae_model.parameters():
            param.requires_grad = False
            
        for param in self.re_model_X.parameters():
            param.requires_grad = False    
                        
        for param in self.re_model_Y.parameters():
            param.requires_grad = False
            
            
            
        if opt["dist"]:
            self.fusion_model = DistributedDataParallel(
                self.fusion_model, device_ids=[torch.cuda.current_device()]
            )
            self.dis_fea = DistributedDataParallel(
                self.dis_fea, device_ids=[torch.cuda.current_device()]
            )
            self.dis_img = DistributedDataParallel(
                self.dis_img, device_ids=[torch.cuda.current_device()]
            )            
  
        self.load()

        self.ae_model.eval()
        self.re_model_X.eval()
        self.re_model_Y.eval()

        self.encode = self.ae_model.encode
        self.decode = self.ae_model.decode
        

        if self.is_train:
            self.fusion_model.train()
            self.dis_fea.train()
            self.dis_img.train()

            self.loss_fusion_fn = Fusion_loss().to(self.device)
            self.loss_dis_fea_fn = Dis_Fea_loss().to(self.device)
            self.loss_dis_img_fn = Dis_Img_loss().to(self.device)
            
            
            
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0

            optim_params_fusion = []
            for (
                k,
                v,
            ) in self.fusion_model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params_fusion.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))


            optim_params_dis_fea = []
            for (
                k,
                v,
            ) in self.dis_fea.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params_dis_fea.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))


            optim_params_dis_img = []
            for (
                k,
                v,
            ) in self.dis_img.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params_dis_img.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))



            if train_opt['optimizer'] == 'Adam':
                self.optimizer_fusion  = torch.optim.Adam(
                    optim_params_fusion,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_dis_fea = torch.optim.Adam(
                    optim_params_dis_fea,
                    lr=train_opt.get("lr_D", train_opt["lr_G"]),  
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )                
                self.optimizer_dis_img = torch.optim.Adam(
                    optim_params_dis_img,
                    lr=train_opt.get("lr_D", train_opt["lr_G"]),
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )                            

            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer_fusion  = torch.optim.AdamW(
                    optim_params_fusion,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_dis_fea  = torch.optim.AdamW(
                    optim_params_dis_fea,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )                                
                self.optimizer_dis_img  = torch.optim.AdamW(
                    optim_params_dis_img,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )                

            elif train_opt['optimizer'] == 'Lion':
                self.optimizer_fusion = Lion(
                    optim_params_fusion, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_dis_fea = Lion(
                    optim_params_dis_fea, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )                
                self.optimizer_dis_img = Lion(
                    optim_params_dis_img, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )                       
                                
            else:
                print('Not implemented optimizer, default using Adam!')
                self.optimizer_fusion = torch.optim.Adam(
                    optim_params_fusion,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_dis_fea = torch.optim.Adam(
                    optim_params_dis_fea,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_dis_img = torch.optim.Adam(
                    optim_params_dis_img,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer_fusion)
            self.optimizers.append(self.optimizer_dis_fea)
            self.optimizers.append(self.optimizer_dis_img)

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

    def feed_data(self, X_fea, Y_fea):
        # LQ_fea & GT_fea is a list
        # self.latent_X = self.re_model_X(X_fea[0]).to(self.device) # latent
        self.latent_X = X_fea[0].to(self.device)
        # print("X shape: ", self.latent_X.shape)
        # self.latent_Y = self.re_model_Y(Y_fea[0]).to(self.device) # latent
        self.latent_Y = Y_fea[0].to(self.device) # latent
        
        # get hidden state using Mean Fusion Rule
        self.hidden_X = []
        self.hidden_Y = []
        for i in range(1, len(X_fea)):
            self.hidden_X.append(X_fea[i].to(self.device))
            self.hidden_Y.append(Y_fea[i].to(self.device))

    def optimize_parameters(self, step):

        if self.opt["dist"]:
            Fuse_fn = self.fusion_model.module
            Dis_fea_fn = self.dis_fea.module
            Dis_img_fn = self.dis_img.module 
        else:
            Fuse_fn = self.fusion_model
            Dis_fea_fn = self.dis_fea
            Dis_img_fn = self.dis_img

        latent_fuse = Fuse_fn(self.latent_X,self.latent_Y)
        hidden_fuse = [(x + y) * 0.5 for x, y in zip(self.hidden_X, self.hidden_Y)]
        # first decode latent to image
        Fuse_recon = self.decode(latent_fuse, hidden_fuse)
        X_img = self.decode(self.latent_X, self.hidden_X)
        Y_img = self.decode(self.latent_Y, self.hidden_Y)        
        
        if step % 2 == 0:
                #### First phase: Optimize the Discriminator in Feature Space #### 
            self.optimizer_dis_fea.zero_grad()      
            P_fea_X_out = Dis_fea_fn(self.latent_X)        
            P_fea_Y_out = Dis_fea_fn(self.latent_Y)        
            P_fea_Fuse_out_detach = Dis_fea_fn(latent_fuse.detach())  
            
            total_dis_fea_loss, dis_fea_loss_X, dis_fea_loss_Y, dis_fea_loss_Fuse = self.loss_dis_fea_fn(P_fea_X_out, P_fea_Y_out, P_fea_Fuse_out_detach)

            total_dis_fea_loss.backward()
            self.optimizer_dis_fea.step()


            #### Second phase: Optimize the Discriminator in Image Space ####   
            self.optimizer_dis_img.zero_grad()
            P_img_X_out = Dis_img_fn(X_img)
            P_img_Y_out = Dis_img_fn(Y_img)
            P_img_Fuse_out_detach = Dis_img_fn(Fuse_recon.detach()) 
            
            total_dis_img_loss, dis_img_loss_X, dis_img_loss_Y, dis_img_loss_Fuse = self.loss_dis_img_fn(P_img_X_out, P_img_Y_out, P_img_Fuse_out_detach)
            
            total_dis_img_loss.backward()
            self.optimizer_dis_img.step()
            
            self.log_dict["dis_fea_loss"] = total_dis_fea_loss.item()
            self.log_dict["dis_fea_loss_X"] = dis_fea_loss_X.item()
            self.log_dict["dis_fea_loss_Y"] = dis_fea_loss_Y.item()
            self.log_dict["dis_fea_loss_Fuse"] = dis_fea_loss_Fuse.item()
            self.log_dict["dis_img_loss"] = total_dis_img_loss.item()
            self.log_dict["dis_img_loss_X"] = dis_img_loss_X.item()
            self.log_dict["dis_img_loss_Y"] = dis_img_loss_Y.item()
            self.log_dict["dis_img_loss_Fuse"] = dis_img_loss_Fuse.item()
        #### Third phase: Optimize the Fusion Model ####  
        # if step % 2 == 0:
    
        self.optimizer_fusion.zero_grad()
        P_fea_Fuse_out = Dis_fea_fn(latent_fuse)
        P_img_Fuse_out = Dis_img_fn(Fuse_recon) 
        
        total_fusion_loss, fusion_loss_adv_fea, fusion_loss_adv_img,fusion_loss_color,loss_max_luminance,loss_gradient = self.loss_fusion_fn(X_img,Y_img,Fuse_recon,self.latent_Y, latent_fuse,P_fea_Fuse_out, P_img_Fuse_out)
        total_fusion_loss.backward()
        self.optimizer_fusion.step()
    
        # set log
        # self.log_dict["loss_rgb"]= loss_rgb.item()
        self.log_dict["fusion_loss_color"] = fusion_loss_color.item()
        self.log_dict["fusion_loss_adv_fea"] = fusion_loss_adv_fea.item()
        self.log_dict["fusion_loss_adv_img"] = fusion_loss_adv_img.item()
        self.log_dict["total_fusion_loss"] = total_fusion_loss.item()
        self.log_dict["loss_max_luminance"] = loss_max_luminance.item()
        self.log_dict["loss_gradient"] = loss_gradient.item()
        # self.log_dict["frob_loss"] = frob_loss.item()
        # self.log_dict["tv_loss"] = tv.item()




    def test(self,current_step,idx=0):
        self.ae_model.eval()
        self.re_model_X.eval()
        self.re_model_Y.eval()
        self.fusion_model.eval()
        
        with torch.no_grad():
            rec_X_img = self.decode(self.latent_X, self.hidden_X)
            rec_Y_img = self.decode(self.latent_Y, self.hidden_Y)
        
            if self.opt["dist"]:
                fusion_fn = self.fusion_model.module
            else:
                fusion_fn = self.fusion_model

        
            latent_fuse = fusion_fn(self.latent_X, self.latent_Y)
            hidden_fuse = [(x + y) * 0.5 for x, y in zip(self.hidden_X, self.hidden_Y)]
            fuse_recon = self.decode(latent_fuse, hidden_fuse)
            
            
            self.output = fuse_recon
            self.Rec_X_img = rec_X_img
            self.Rec_Y_img = rec_Y_img
            
                   
        self.fusion_model.train()
        self.dis_fea.train()
        self.dis_img.train()
        
        save_dir = os.path.join('image', f'step_{current_step}')
        os.makedirs(save_dir, exist_ok=True)

        batch_size = self.output.shape[0]
        for i in range(batch_size):
            index = idx + i
            tvutils.save_image(self.output[i].data, os.path.join(save_dir, f'{index}_Fused.png'), normalize=False)
            tvutils.save_image(self.Rec_X_img[i].data, os.path.join(save_dir, f'{index}_Rec_X.png'), normalize=False)
            tvutils.save_image(self.Rec_Y_img[i].data, os.path.join(save_dir, f'{index}_Rec_Y.png'), normalize=False)
    
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Fused"] = self.output.detach()[0].float().cpu()
        out_dict["Rec_X_img"] = self.Rec_X_img.detach()[0].float().cpu()
        out_dict["Rec_Y_img"] = self.Rec_Y_img.detach()[0].float().cpu()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.fusion_model)
        if isinstance(self.fusion_model, nn.DataParallel) or isinstance(
            self.fusion_model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.fusion_model.__class__.__name__, self.fusion_model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.fusion_model.__class__.__name__)
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

        load_path_RE_X = self.opt["path"]["pretrain_model_RE_X"]
        if load_path_RE_X is not None:
            logger.info("Loading model for REModel_X [{:s}] ...".format(load_path_RE_X))
            self.load_network(load_path_RE_X, self.re_model_X, self.opt["path"]["strict_load"])

        load_path_RE_Y = self.opt["path"]["pretrain_model_RE_Y"]
        if load_path_RE_Y is not None:
            logger.info("Loading model for REModel_Y [{:s}] ...".format(load_path_RE_Y))
            self.load_network(load_path_RE_Y, self.re_model_Y, self.opt["path"]["strict_load"])

        load_path_Fusion = self.opt["path"]["pretrain_model_Fusion"]
        if load_path_Fusion is not None:
            logger.info("Loading model for FusionModel [{:s}] ...".format(load_path_Fusion))
            self.load_network(load_path_Fusion, self.fusion_model, self.opt["path"]["strict_load"])


    def save(self, iter_label):
        self.save_network(self.fusion_model, "Fusion", iter_label)


