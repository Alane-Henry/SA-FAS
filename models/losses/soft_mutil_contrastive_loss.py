import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftMultiContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, class_weights=None, need_norm=True):
        super().__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        self.need_norm = need_norm

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
        # Soft cross entropy loss: image-to-text

        B, C = logits.shape

        # One-hot 标签
        soft_labels = F.one_hot(labels, num_classes=C).float()  # [B, C]

        # 找出 label > 0 的样本
        mask = (labels > 0)  # [B]

        # 对这些样本加上最后一类（即 index = C - 1）
        soft_labels[mask, C - 1] = 1.0  # 添加最后一类为正样本

        # 将有多个正样本的行进行归一化（变成概率分布）
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)

        # log softmax
        log_probs = F.log_softmax(logits, dim=1)  # [B, C]

        if self.class_weights is not None:
            weights = self.class_weights.unsqueeze(0).to(f_img.device)  # [1, C]
            weighted_loss = - (soft_labels * log_probs * weights).sum(dim=1)
        else:
            weighted_loss = - (soft_labels * log_probs).sum(dim=1)

        return weighted_loss.mean() 
        # Average the two directions
        # return (loss_i2t + loss_t2i) / 2



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