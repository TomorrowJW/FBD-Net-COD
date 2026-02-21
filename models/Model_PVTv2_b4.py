import torch
import torch.nn.functional as F
import torch.nn as nn
from .PVTv2 import pvt_v2_b4
from .Align import BasicConv2d
from .HM import  HAIM_New
from einops import rearrange

class Disentangler(nn.Module):
    """
    DisHead 模块
    -----------------------------------------
    功能：
    1. 生成边界预测（Edge Objective）
    2. 生成前景全局向量（FoBa-CL）
    3. 生成背景全局向量（FoBa-CL）

    This module implements:
    - Edge prediction branch
    - Foreground representation head
    - Background representation head
    """

    def __init__(self, cin):
        super(Disentangler, self).__init__()

        # Edge prediction branch
        # 边界预测分支

        self.edge_stream = nn.Sequential(BasicConv2d(cin, cin, 1, 1, 0),
                                         nn.Conv2d(cin, 1, 3, 1, 1))

        # Foreground embedding head (for contrastive learning)
        # 前景特征嵌入头
        self.fore_head = nn.Sequential(nn.Linear(cin, cin * 2), nn.ReLU(), nn.Linear(cin * 2, cin))

        # Background embedding head
        # 背景特征嵌入头
        self.back_head = nn.Sequential(nn.Linear(cin, cin * 2), nn.ReLU(), nn.Linear(cin * 2, cin))

    def forward(self, x, prediction, mask):
        Pre = prediction

        # prediction -> sigmoid
        # 当前阶段预测图
        pre = torch.sigmoid(Pre)
        N, C, H, W = x.size()

        # Edge enhancement using foreground-background difference
        # 利用前景-背景差分增强边界
        edge = self.edge_stream((x * pre) - (x * (1 - pre)))

        # ==========================
        # Foreground / Background masks
        # 前景/背景掩码（不参与梯度）
        # ==========================

        with torch.no_grad():
            fore_mask = mask
            back_mask = torch.ones_like(mask) - fore_mask
            fore_mask = F.interpolate(fore_mask, x.size()[2:], mode='bilinear', align_corners=False)
            back_mask = F.interpolate(back_mask, x.size()[2:], mode='bilinear', align_corners=False)
            fore_mask = rearrange(fore_mask, "b c h w -> b c (h w)")
            back_mask = rearrange(back_mask, "b c h w -> b c (h w)")

        x = rearrange(x, "b c h w -> b (h w) c")

        # Foreground feature aggregation
        # 前景特征聚合
        fg_feats = fore_mask @ x
        # Background feature aggregation
        # 背景特征聚合
        bg_feats = back_mask @ x

        fg_feats = self.fore_head(fg_feats.reshape(N, -1))
        bg_feats = self.back_head(bg_feats.reshape(N, -1))

        return fg_feats, bg_feats, edge

class Fore_Back(nn.Module):
    """
    EFBD: Edge-aware Foreground-Background Disentanglement

    使用上一层预测和边界图
    进行前景流与背景流分离
    """

    def __init__(self, inchannel):
        super().__init__()

        self.fore_stream = nn.ModuleList(
            [BasicConv2d(inchannel, inchannel, 1, 1, 0), BasicConv2d(inchannel, inchannel, 3, 1, 1)])
        self.back_stream = nn.ModuleList(
            [BasicConv2d(inchannel, inchannel, 1, 1, 0), BasicConv2d(inchannel, inchannel, 3, 1, 1)])
        self.all = nn.Sequential(  # BasicConv2d(inchannel*3, inchannel, 1, 1, 0),
            BasicConv2d(inchannel * 3, inchannel, 3, 1, 1))

    def forward(self, Feature_Map, Prediction, edge_prediction):
        Pre = Prediction  # .detach()
        pre = torch.sigmoid(Pre)

        Edge = edge_prediction  # .detach()
        edge = torch.sigmoid(Edge)

        # Edge-aware modulation
        fore = self.fore_stream[0](Feature_Map) * edge + Feature_Map
        back = self.back_stream[0](Feature_Map) * edge + Feature_Map

        # Foreground flow
        fore = self.fore_stream[1](fore * pre)
        # Background flow
        back = self.back_stream[1](back * (1 - pre))

        # Fusion + residual
        all = self.all(torch.cat((fore, back, Feature_Map), 1)) + Feature_Map
        return all


class fusion_layer(nn.Module):
    """
        CAM: Cross-level Aggregation Module
        融合4个backbone阶段特征
        """
    def __init__(self, ):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv = BasicConv2d(64 * 4, 128, 3, 1, 1)

    def forward(self, o1, o2, o3, o4):
        o4 = self.upsample8(self.conv4(o4))
        o3 = self.upsample4(self.conv3(o3))
        o2 = self.upsample2(self.conv2(o2))
        o1 = self.conv1(o1)
        o = torch.cat((o1, o2, o3, o4), 1)
        o = self.conv(o)

        return o

class FeaFusion(nn.Module):
    """
    CSAF: Cross-Scale Adaptive Fusion
    自适应跨尺度融合模块
    """
    def __init__(self, channels):
        self.init__ = super(FeaFusion, self).__init__()

        self.relu = nn.ReLU()
        self.layer1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.layer2_1 = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1)

        self.layer_fu = nn.Conv2d(channels // 4, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        ###
        wweight = nn.Sigmoid()(self.layer1(x1 + x2))

        ###
        xw_resid_1 = x1 + x1.mul(wweight)
        xw_resid_2 = x2 + x2.mul(wweight)

        ###
        x1_2 = self.layer2_1(xw_resid_1)
        x2_2 = self.layer2_2(xw_resid_2)

        out = self.relu(self.layer_fu(x1_2 + x2_2))

        return out

'''
定义decoder
'''

class Decoder(nn.Module):
    """
        Decoder结构：

        CAM
          ↓
        SIEP
          ↓
        EFBD
          ↓
        CSAF
          ↓
        DisHead
        """
    def __init__(self):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.HM5 = HAIM_New(128)
        self.HM4 = HAIM_New(128)
        self.HM3 = HAIM_New(128)
        self.HM2 = HAIM_New(128)
        self.HM1 = HAIM_New(128)

        self.fusion = fusion_layer()

        self.d_out5 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))

        self.f4 = FeaFusion(128)
        self.d_out4 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.e_f_b4 = Fore_Back(128)


        self.f3 = FeaFusion(128)
        self.d_out3 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.e_f_b3 = Fore_Back(128)


        self.f2 = FeaFusion(128)
        self.d_out2 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.e_f_b2 = Fore_Back(128)


        self.f1 = FeaFusion(128)
        self.d_out1 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.e_f_b1 = Fore_Back(128)

        self.Disentangler5 = Disentangler(128)
        self.Disentangler4 = Disentangler(128)
        self.Disentangler3 = Disentangler(128)
        self.Disentangler2 = Disentangler(128)
        self.Disentangler1 = Disentangler(128)

    def forward(self, o1, o2, o3, o4, mask):
        o = self.fusion(o1, o2, o3, o4)
        o = self.HM5(o)
        s5_out = self.d_out5(o)
        f_c5, b_c5, edge_5 = self.Disentangler5(o, s5_out, mask)
        s5 = self.upsample4(s5_out)
        e5 = self.upsample4(edge_5)

        s5_4 = F.interpolate(s5_out, o4.size()[2:], mode='bilinear', align_corners=True)
        e5_4 = F.interpolate(edge_5, o4.size()[2:], mode='bilinear', align_corners=True)

        s54_up = F.interpolate(o, o4.size()[2:], mode='bilinear', align_corners=True)

        o4 = self.HM4(o4)
        o4 = self.e_f_b4(o4, s5_4, e5_4)
        s4_up = self.f4(s54_up, o4)
        s4_out = self.d_out4(s4_up)
        f_c4, b_c4, edge_4 = self.Disentangler4(s4_up, s4_out, mask)
        s4 = self.upsample32(s4_out)
        e4 = self.upsample32(edge_4)

        o3 = self.HM3(o3)
        o3 = self.e_f_b3(o3, self.upsample2(s4_out), self.upsample2(edge_4))
        s3_up = self.f3(self.upsample2(s4_up), o3)
        s3_out = self.d_out3(s3_up)
        f_c3, b_c3, edge_3 = self.Disentangler3(s3_up, s3_out, mask)
        s3 = self.upsample16(s3_out)
        e3 = self.upsample16(edge_3)

        o2 = self.HM2(o2)
        o2 = self.e_f_b2(o2, self.upsample2(s3_out), self.upsample2(edge_3))
        s2_up = self.f2(self.upsample2(s3_up), o2)
        s2_out = self.d_out2(s2_up)
        f_c2, b_c2, edge_2 = self.Disentangler2(s2_up, s2_out, mask)
        s2 = self.upsample8(s2_out)
        e2 = self.upsample8(edge_2)

        o1 = self.HM1(o1)
        o1 = self.e_f_b1(o1, self.upsample2(s2_out), self.upsample2(edge_2))
        s1_up = self.f1(self.upsample2(s2_up), o1)
        s1_out = self.d_out1(s1_up)
        f_c1, b_c1, e1 = self.Disentangler1(s1_up, s1_out, mask)
        s1 = self.upsample4(s1_out)
        e1 = self.upsample4(e1)

        return s1, s2, s3, s4, s5, e1, e2, e3, e4, e5, f_c1, b_c1, f_c2, b_c2, f_c3, b_c3, f_c4, b_c4, f_c5, b_c5


class CL_Model(nn.Module):
    """
     Full FBD-Net Model
     完整FBD-Net模型
     """

    def __init__(self):
        super().__init__()
        self.pvtv2 = pvt_v2_b4(pretrained=False)
        save_model = torch.load('./pre_train/pvt_v2_b4.pth')
        model_dict = self.pvtv2.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.pvtv2.load_state_dict(model_dict)

        self.sigmoid = nn.Sigmoid()

        self.conv1 = BasicConv2d(64, 128, 1, stride=1, padding=0)
        self.conv2 = BasicConv2d(128, 128, 1, stride=1, padding=0)
        self.conv3 = BasicConv2d(320, 128, 1, stride=1, padding=0)
        self.conv4 = BasicConv2d(512, 128, 1, stride=1, padding=0)

        self.decoder = Decoder()

    def forward(self, rgb, mask):
        stage_rgb = self.pvtv2(rgb)

        s1 = stage_rgb[0]
        s2 = stage_rgb[1]
        s3 = stage_rgb[2]
        s4 = stage_rgb[3]

        c1 = self.conv1(s1)
        c2 = self.conv2(s2)
        c3 = self.conv3(s3)
        c4 = self.conv4(s4)


        f1 = c1
        f2 = c2
        f3 = c3
        f4 = c4

        s1, s2, s3, s4, s5, e1, e2, e3, e4, e5, f_c1, b_c1, f_c2, b_c2, f_c3, b_c3, f_c4, b_c4, f_c5, b_c5 = self.decoder(
            f1, f2, f3, f4, mask)

        return s1, s2, s3, s4, s5, e1, e2, e3, e4, e5, f_c1, b_c1, f_c2, b_c2, f_c3, b_c3, f_c4, b_c4, f_c5, b_c5

