import torch
import torch.nn as nn
import torch.nn.functional as F

class Fore_Back_CLoss_New(nn.Module):
    """
        FoBa Contrastive Loss
        =============================================

        对应论文中的：
        Foreground-Background Contrastive Learning (FoBa-CL)

        输入:
            fore: 前景全局特征向量 (B, C)
            back: 背景全局特征向量 (B, C)

        目标:
            1. 同一样本的前景与背景 → 拉远 (push apart)
            2. 不同样本之间 → 作为负样本

        This loss encourages:
            - Intra-image foreground and background separation
            - Inter-image contrastive discrimination
        """
    def __init__(self, temperature1, temperature2):
        """
        temperature1: 正样本温度
        temperature2: 负样本温度
        """
        super().__init__()

        self.temperature1 = temperature1
        self.temperature2 = temperature2

    def forward(self,fore,back):
        batch_size = fore.size(0)
        # =====================================================
        # 构建单位矩阵（用于提取对角线正样本）
        # Identity matrix for positive pairs
        # =====================================================
        with torch.no_grad():
            ones = torch.eye(batch_size, device=fore.device)

        # =====================================================
        # L2 Normalize 特征
        # cosine similarity 准备
        # =====================================================
        fore_ = F.normalize(fore,p=2,dim=1)
        back_ = F.normalize(back,p=2,dim=1)

        # =====================================================
        # 计算前景-背景相似度矩阵
        # dot1[i,j] = cos(fore_i, back_j)
        # =====================================================
        dot1 = torch.matmul(fore_, back_.T)

        # =====================================================
        # 正样本项 (i == j)
        # 论文公式中使用 1 - sim(f_i, b_i)
        # =====================================================
        diag = 1-torch.diag(dot1)

        # 构造对角矩阵
        diag_T = torch.diag_embed(diag)

        # =====================================================
        # 负样本项 (i != j)
        # =====================================================
        dot = dot1 * (1-ones)

        # 将正样本放回对角
        dot = dot + diag_T

        # =====================================================
        # 负样本指数项
        # exp(sim / T2)
        # =====================================================
        exp_logits1 = torch.exp(torch.div(dot,self.temperature2))

        # =====================================================
        # 正样本 logits
        # =====================================================
        pos_logits1 = torch.div(dot ,self.temperature1)

        # 只保留对角线（正样本）
        pos_logits1 = pos_logits1 * ones

        # =====================================================
        # InfoNCE 风格损失
        # =====================================================
        prob1  =  pos_logits1 - torch.log(exp_logits1.sum(dim=1, keepdim=True)+1e-12)
        prob1  =  prob1 * ones

        mask_sum1 = ones.sum(dim=1)

        loss1 = prob1.sum(dim=1) / mask_sum1.detach()
        loss1 = -loss1.mean()

        return loss1


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict_, target):
    predict = torch.sigmoid(predict_)
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def cross_entropy2d_edge(input, target, reduction='mean'):
    assert (input.size()) == target.size()
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha *pos + beta*neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)







