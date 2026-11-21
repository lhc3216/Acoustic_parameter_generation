import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_dynamic_lambdas(current_epoch, total_epochs=200):
    end_l1 = 1
    end_kl = 1e-4
    end_grad = 0.5
    end_imp = 0.2

    lam_l1 = end_l1

    kl_start_epoch = 20
    kl_end_epoch = 80
    if current_epoch < kl_start_epoch:
        lam_kl = 0.0
    elif current_epoch > kl_end_epoch:
        lam_kl = end_kl
    else:
        progress = min(1.0, (current_epoch - kl_start_epoch) / (kl_end_epoch - kl_start_epoch))
        lam_kl = end_kl * progress
    
    grad_start = 40
    grad_end = 80
    if current_epoch < grad_start:
        lam_grad = 0.0
    elif current_epoch > grad_end:
        lam_grad = end_grad
    else:
        progress = min(1.0, (current_epoch - grad_start) / (grad_end - grad_start))
        lam_grad = end_grad * progress
        
    imp_start = 80
    imp_end = 120
    if current_epoch < imp_start:
        lam_imp = 0.0
    elif current_epoch > imp_end:
        lam_imp = end_imp
    else:
        progress = min(1.0, (current_epoch - imp_start) / (imp_end - imp_start))
        lam_imp = end_imp * progress
        
    return lam_l1, lam_kl, lam_grad, lam_imp

class PhysicsInformedVAELoss(nn.Module):
    def __init__(
        self, 
        lambda_l1=1.0, 
        lambda_grad=2.0, 
        lambda_imp=0.5,
        lambda_kl=1e-5,
        rho_min=1000.0, rho_max=1950.0,
        c_min=1482.3, c_max=3100.0,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.lambda_imp = lambda_imp
        self.lambda_kl = lambda_kl
        
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.c_min = c_min
        self.c_max = c_max

    def update_lambdas(self, lam_l1, lam_kl, lam_grad, lam_imp):
        self.lambda_l1 = lam_l1
        self.lambda_kl = lam_kl
        self.lambda_grad = lam_grad
        self.lambda_imp = lam_imp
        

    def denormalize(self, tensor_norm, min_val, max_val):
        tensor_01 = (tensor_norm + 1.0) / 2.0
        tensor_01 = torch.clamp(tensor_01, 0.0, 1.0)
        
        physical_tensor = tensor_01 * (max_val - min_val) + min_val
        return torch.clamp(physical_tensor, min_val, max_val)

    def compute_gradient_loss(self, pred, target):
        pred_dx = torch.diff(pred, dim=-1)
        pred_dy = torch.diff(pred, dim=-2)
        target_dx = torch.diff(target, dim=-1)
        target_dy = torch.diff(target, dim=-2)
        
        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        return loss_dx + loss_dy

    def compute_impedance_loss(self, rho_pred, c_pred, rho_gt, c_gt):
        Z_pred = rho_pred * c_pred
        Z_gt = rho_gt * c_gt
        loss_imp = torch.mean(torch.abs(Z_pred - Z_gt) / (Z_gt + 1e-8))
        return loss_imp

   

    def forward(self, recon_x, x, mu, logvar, p_gt=None):
        loss_l1 = F.l1_loss(recon_x, x)
        
        loss_grad = torch.tensor(0.0, device=recon_x.device)
        loss_imp  = torch.tensor(0.0, device=recon_x.device)
        loss_kl = torch.tensor(0.0, device=recon_x.device)
        # ==========================================
        # ==========================================
        
        if self.lambda_kl > 0:
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if self.lambda_grad > 0:
            loss_grad = self.compute_gradient_loss(recon_x, x)
            
        if self.lambda_imp > 0:
            rho_pred = self.denormalize(recon_x[:, 1:2, :, :], self.rho_min, self.rho_max)
            c_pred   = self.denormalize(recon_x[:, 0:1, :, :], self.c_min, self.c_max)
            rho_gt   = self.denormalize(x[:, 1:2, :, :], self.rho_min, self.rho_max)
            c_gt     = self.denormalize(x[:, 0:1, :, :], self.c_min, self.c_max)

            loss_imp = self.compute_impedance_loss(rho_pred, c_pred, rho_gt, c_gt)
                
          
        

        total_loss = (self.lambda_l1 * loss_l1) + \
                     (self.lambda_grad * loss_grad) + \
                     (self.lambda_imp * loss_imp) + \
                     (self.lambda_kl * loss_kl)
        loss_dict = {
            "loss_l1": loss_l1.item(),
            "loss_grad": loss_grad.item(),
            "loss_imp": loss_imp.item(),
            "loss_kl": loss_kl.item(),
        }
                     
        return total_loss, loss_dict