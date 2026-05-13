import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys




class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


class Pur_loss(nn.Module):
    def __init__(self,loss_type='l1'):
        super(Pur_loss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, fea_pur, fea_GT, image_recon, image_GT):
        loss_fea=self.loss_fn(fea_pur, fea_GT)
        loss_image=self.loss_fn(image_recon, image_GT)

        total_loss = loss_fea + loss_image
        
        return total_loss, loss_fea, loss_image
        
class Fusion_loss(nn.Module):
    def __init__(self,loss_type='l2'):
        super(Fusion_loss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, X_GT_Img,Y_GT_Img,Fuse_Img, Y_Fea,F_Fea, P_Fea=None, P_Img=None):
    
        Y_GT_Img_y, Y_GT_Img_cbcr = self.rgb_to_y(Y_GT_Img)
        X_GT_Img_y, X_GT_Img_cbcr = self.rgb_to_y(X_GT_Img)
        Fuse_Img_y, Fuse_Img_cbcr = self.rgb_to_y(Fuse_Img)
        loss_color=self.loss_fn(Fuse_Img_cbcr, X_GT_Img_cbcr)
        
        max_luminance = torch.max(X_GT_Img_y, Y_GT_Img_y)
        loss_max_luminance = self.loss_fn(Fuse_Img_y, max_luminance)
        grad_X = self.compute_gradient(X_GT_Img_y)
        grad_Y = self.compute_gradient(Y_GT_Img_y)
        grad_Fuse = self.compute_gradient(Fuse_Img_y)
        max_gradient = torch.max(grad_X, grad_Y)
        loss_gradient = self.loss_fn(grad_Fuse, max_gradient)
        total_loss =   4*loss_gradient
        Fusion_adv_loss_fea = torch.tensor(0.0, device=X_GT_Img.device)
        Fusion_adv_loss_img = torch.tensor(0.0, device=X_GT_Img.device)

        if P_Fea is not None:
            with torch.no_grad():
                target_fea_0 = torch.empty_like(P_Fea[:, 0]).uniform_(0.4, 0.6)
                target_fea_1 = torch.empty_like(P_Fea[:, 1]).uniform_(0.4, 0.6)
            Fusion_adv_loss_fea=self.loss_fn(P_Fea[:, 0], target_fea_0) + self.loss_fn(P_Fea[:, 1], target_fea_1)
            total_loss += 0.5*Fusion_adv_loss_fea

        if P_Img is not None:
            with torch.no_grad():
                target_img_0 = torch.empty_like(P_Img[:, 0]).uniform_(0.4, 0.6)
                target_img_1 = torch.empty_like(P_Img[:, 1]).uniform_(0.4, 0.6)
            Fusion_adv_loss_img=self.loss_fn(P_Img[:, 0], target_img_0) + self.loss_fn(P_Img[:, 1], target_img_1)
            total_loss += 0.5*Fusion_adv_loss_img

        return total_loss, Fusion_adv_loss_fea, Fusion_adv_loss_img,loss_color,loss_max_luminance,loss_gradient
        
    def compute_gradient(self,img):
    # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)

        gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return gradient
    
    def tv_loss(self,x):
        loss = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        return loss

    def rgb_to_y(self, image):
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y)
        cr = 0.713 * (r - y)
        cbcr = torch.cat((cb, cr), dim=1)
        
        return y, cbcr       


class Dis_Fea_loss(nn.Module):
    def __init__(self,loss_type='l2'):
        super(Dis_Fea_loss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, P_Fea_X, P_Fea_Y, P_Fea_fus):
        with torch.no_grad():
            target_X_0 = torch.empty_like(P_Fea_X[:, 0]).uniform_(0.9, 1.1)
            target_X_1 = torch.empty_like(P_Fea_X[:, 1]).uniform_(0.0, 0.1)
            target_Y_0 = torch.empty_like(P_Fea_Y[:, 0]).uniform_(0.0, 0.1)
            target_Y_1 = torch.empty_like(P_Fea_Y[:, 1]).uniform_(0.9, 1.1)        
            target_fus_0 = torch.empty_like(P_Fea_fus[:, 0]).uniform_(0.0, 0.1)
            target_fus_1 = torch.empty_like(P_Fea_fus[:, 1]).uniform_(0.0, 0.1)      
        
        dis_loss_X_fea = self.loss_fn(P_Fea_X[:, 0], target_X_0) + self.loss_fn(P_Fea_X[:, 1], target_X_1)
        dis_loss_Y_fea = self.loss_fn(P_Fea_Y[:, 0], target_Y_0) + self.loss_fn(P_Fea_Y[:, 1], target_Y_1)
        dis_loss_fus_fea = self.loss_fn(P_Fea_fus[:, 0], target_fus_0) + self.loss_fn(P_Fea_fus[:, 1], target_fus_1)

        total_loss = 0.25*dis_loss_X_fea + 0.25*dis_loss_Y_fea + 0.5*dis_loss_fus_fea
        
        return total_loss, dis_loss_X_fea, dis_loss_Y_fea, dis_loss_fus_fea


class Dis_Img_loss(nn.Module):
    def __init__(self,loss_type='l2'):
        super(Dis_Img_loss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, P_Img_X, P_Img_Y, P_Img_fus):
        with torch.no_grad():    
            target_X_0 = torch.empty_like(P_Img_X[:, 0]).uniform_(0.9, 1.1)
            target_X_1 = torch.empty_like(P_Img_X[:, 1]).uniform_(0.0, 0.1)
            target_Y_0 = torch.empty_like(P_Img_Y[:, 0]).uniform_(0.0, 0.1)
            target_Y_1 = torch.empty_like(P_Img_Y[:, 1]).uniform_(0.9, 1.1)       
            target_fus_0 = torch.empty_like(P_Img_fus[:, 0]).uniform_(0.0, 0.1)
            target_fus_1 = torch.empty_like(P_Img_fus[:, 1]).uniform_(0.0, 0.1)      
        
        dis_loss_X_img = self.loss_fn(P_Img_X[:, 0], target_X_0) + self.loss_fn(P_Img_X[:, 1], target_X_1)
        dis_loss_Y_img = self.loss_fn(P_Img_Y[:, 0], target_Y_0) + self.loss_fn(P_Img_Y[:, 1], target_Y_1)
        dis_loss_fus_img = self.loss_fn(P_Img_fus[:, 0], target_fus_0) + self.loss_fn(P_Img_fus[:, 1], target_fus_1)

        total_loss = 0.25*dis_loss_X_img + 0.25*dis_loss_Y_img + 0.5*dis_loss_fus_img
        
        return total_loss, dis_loss_X_img, dis_loss_Y_img, dis_loss_fus_img         
        
        
        
        