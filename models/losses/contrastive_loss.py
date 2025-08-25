import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def all_gather_with_grad(tensor):
    """all_gather that supports gradients (differentiable)"""
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, class_weights=None, need_norm=True, use_distributed=True):
        super().__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        self.need_norm = need_norm
        self.use_distributed = use_distributed
        print(f"ContrastiveLoss initialized with temperature={temperature}, class_weights={class_weights}, need_norm={need_norm}, use_distributed={use_distributed}")

    def forward(self, f_img, f_txt, labels):
        """
        Soft contrastive loss between image and text features.
        
        Args:
            f_img: Tensor of shape [B, D] - image features
            f_txt: Tensor of shape [B, D] - text features
            label: Tensor of shape [B] - labels, like a mask for positive pairs
            need_norm: bool - whether to normalize features before calculating loss
            temperature: float - temperature parameter
        Returns:
            Scalar loss
        """
        if self.need_norm:
            f_img = F.normalize(f_img, dim=1)
            f_txt = F.normalize(f_txt, dim=1)
        
        logits = torch.matmul(f_img, f_txt.T) / self.temperature

        if self.class_weights is not None:
            loss_i2t = F.cross_entropy(logits, labels, weight=self.class_weights.to(f_img.device))
        else:
            loss_i2t = F.cross_entropy(logits, labels)

        # Text-to-image loss (symmetrical)
        # transfer label from image cls to text cls
        # labels = labels.view(-1, 1)
        # loss_t2i = F.cross_entropy(logits.T, labels)

        return loss_i2t 
        # Average the two directions
        # return (loss_i2t + loss_t2i) / 2

class ContrastiveALLgatherLoss(nn.Module):
    def __init__(self, temperature=0.07, class_weights=None, need_norm=True, use_distributed=True):
        super().__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        self.need_norm = need_norm
        self.use_distributed = use_distributed

    def forward(self, f_img, f_txt, labels):
        """
        Soft contrastive loss between image and text features.
        
        Args:
            f_img: Tensor of shape [B, D] - image features
            f_txt: Tensor of shape [B, D] - text features
            label: Tensor of shape [B] - labels, like a mask for positive pairs
            need_norm: bool - whether to normalize features before calculating loss
            temperature: float - temperature parameter
        Returns:
            Scalar loss
        """
        if self.need_norm:
            f_img = F.normalize(f_img, dim=1)
            f_txt = F.normalize(f_txt, dim=1)
        world_size = 1
        if self.use_distributed and dist.is_available() and dist.is_initialized():
            # All-gather across all GPUs
            world_size = dist.get_world_size()
            f_txt_all = all_gather_with_grad(f_txt)
        else:
            f_txt_all = f_txt
        
        C = f_txt.shape[0]
        
        logits = torch.matmul(f_img, f_txt_all.T) / self.temperature
        print(f'world_size: {world_size}, logits: {logits.shape}')
        positive_indices = labels.unsqueeze(1) + torch.arange(world_size, device=labels.device) * C
        target = torch.zeros_like(logits)
        target.scatter_(1, positive_indices, 1)

        # Contrastive Loss
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss



def soft_contrastive_loss(f_img, f_txt, temperature=0.07):
    """
    Soft contrastive loss between image and text features.
    
    Args:
        f_img: Tensor of shape [B, D] - image features
        f_txt: Tensor of shape [B, D] - text features
        temperature: float - temperature parameter
    Returns:
        Scalar loss
    """
    # Normalize features to unit vectors
    f_img = F.normalize(f_img, dim=1)
    f_txt = F.normalize(f_txt, dim=1)

    # Similarity matrix: [B, B]
    logits = torch.matmul(f_img, f_txt.T) / temperature

    # Targets: soft match (identity matrix as soft labels)
    labels = torch.arange(f_img.size(0)).to(f_img.device)

    # Soft cross entropy loss: image-to-text
    loss_i2t = F.cross_entropy(logits, labels)

    # Text-to-image loss (symmetrical)
    loss_t2i = F.cross_entropy(logits.T, labels)

    # Average the two directions
    return (loss_i2t + loss_t2i) / 2