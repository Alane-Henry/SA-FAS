import torch
from torch import nn
import torch.nn.functional as F


# class OneCenterLoss(nn.Module):

#     def __init__(self, size=(1, 512)):
#         super(OneCenterLoss, self).__init__()
#         self.margin = 0.5
#         self.alpha = 0.1
#         self.center = None

#     def forward(self, x, labels):
#         ones_idx = labels==0  # 求label==1的索引
#         if self.center is None:  # init center
#             self.center = nn.Parameter(x.detach()[ones_idx].mean(0, keepdim=True), requires_grad=False)  # 以数据初始特征初始化
#         batch_size = x.size(0)
#         ones = torch.ones(batch_size).cuda()
#         zeros = torch.zeros(batch_size).cuda()
#         labels = torch.where(labels == 0, ones, -ones)
#         margin = torch.where(labels == 0, zeros, self.margin * ones)

#         centers = self.center.expand(batch_size, -1)
#         disinbatch = torch.norm(x / x.norm(2) - centers / centers.norm(2), 2, dim=(1))  # 计算每一个样本到中心的L2归一化的欧式距离
#         loss = F.relu(disinbatch * labels + margin).mean(0, keepdim=True)  # max(d(ap), 0), max(m-d(an), 0)

#         if ones_idx.size(0) != 0:
#             self.center.add_((x.detach() - centers)[ones_idx].mean(0, keepdim=True), alpha=self.alpha)  # update center
#         return loss


class OneCenterLoss(nn.Module):
    #Learning one class representations for face presentation attack detection using multi-channel convolutional neural networks
    def __init__(self, size=(1, 512)):
        super(OneCenterLoss, self).__init__()
        self.margin = 3.0
        self.alpha = 0.1
        self.center = None

    def forward(self, x, labels):
        ones_idx = labels==0  # 求label==0的索引
        if self.center is None:  # init center
            self.center = nn.Parameter(x.detach()[ones_idx].mean(0, keepdim=True), requires_grad=False)  # 以数据初始特征初始化

        batch_size = x.size(0)
        centers = self.center.expand(batch_size, -1)
        
        disinbatch = torch.norm(x - centers, 2, dim=(1))  # 计算每一个样本到中心的L2归一化的欧式距离
        bonafide_loss=(1-labels) * torch.pow(disinbatch, 2) /2
        attack_loss =labels * torch.pow(torch.clamp(self.margin - disinbatch, min=0.0), 2)
        loss = torch.mean(bonafide_loss+attack_loss, 0)

        if ones_idx.size(0) != 0:
           self.center.add_((x.detach() - centers)[ones_idx].mean(0, keepdim=True), alpha=self.alpha)  # update center
        return loss

class Centerloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.center = nn.Parameter(torch.randn(10,2),requires_grad=True)

    def forward(self,features,ys,lambdas=2):
        center_exp = self.center.index_select(dim=0,index=ys.long())
        count = torch.histc(ys,bins=int(max(ys).item()+1),min=0,max=int(max(ys).item()))
        count_exp = count.index_select(dim=0,index=ys.long())
        loss = lambdas/2*torch.mean(torch.div(torch.sum(torch.pow(features-center_exp,2),dim=1),count_exp))
        return loss

if __name__ == '__main__':#测试
    # a = Centerloss()
    a = OneCenterLoss()
    feature = torch.randn(5, 2, dtype=torch.float32).cuda()
    ys = torch.tensor([0, 0, 1, 0, 1,], dtype=torch.float32).cuda()
    b = a(feature, ys)