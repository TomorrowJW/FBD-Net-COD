import torch
import torch.nn as nn
from models.Align import BasicConv2d, ChannelAttention, SpatialAttention

class HAIM_New(nn.Module):
    """
        HAIM_New 模块
        ==========================================
        对应论文模块：SIEP (Scale-Interaction Enhancement Path)

        结构说明：
        1. 四个并行分支（多尺度）
        2. 每个分支：
           - 1×1 降维
           - CRE（Channel + Spatial Attention）
           - 3×3 空洞卷积
        3. 采用递进式交互（F2 += F1, F3 += F2, F4 += F3）
        4. 拼接 + Channel Attention
        5. Residual connection

        This module implements:
        - Multi-scale dilation branches
        - Cross-scale progressive interaction
        - Channel + Spatial attention refinement
        - Residual enhancement
        """
    def __init__(self, in_channel, rate=[1,6,12,18]):
        super().__init__()

        # ===============================
        # 四个多尺度分支
        # Each branch: 1x1 -> 3x3 dilated conv
        # ===============================

        self.rgb_b1 = nn.ModuleList([BasicConv2d(in_channel, in_channel//4, 1,1,0),
                                    BasicConv2d(in_channel//4, in_channel//4, 3, padding=rate[0], dilation=rate[0])])
        self.rgb_b2 = nn.ModuleList([BasicConv2d(in_channel, in_channel//4, 1,1,0),
                                    BasicConv2d(in_channel//4, in_channel//4, 3, padding=rate[0], dilation=rate[0])])
        self.rgb_b3 = nn.ModuleList([BasicConv2d(in_channel, in_channel//4, 1,1,0),
                                    BasicConv2d(in_channel//4, in_channel//4, 3, padding=rate[0], dilation=rate[0])])
        self.rgb_b4 = nn.ModuleList([BasicConv2d(in_channel, in_channel//4, 1,1,0),
                                    BasicConv2d(in_channel//4, in_channel//4, 3, padding=rate[0], dilation=rate[0])])

        # ===============================
        # Attention modules (CRE)
        # Channel Attention + Spatial Attention
        # ===============================
        self.rgb_b1_sa = SpatialAttention()
        self.rgb_b2_sa = SpatialAttention()
        self.rgb_b3_sa = SpatialAttention()
        self.rgb_b4_sa = SpatialAttention()

        self.rgb_b1_ca = ChannelAttention(in_channel//4)
        self.rgb_b2_ca = ChannelAttention(in_channel//4)
        self.rgb_b3_ca = ChannelAttention(in_channel//4)
        self.rgb_b4_ca = ChannelAttention(in_channel//4)

        # ===============================
        # Final fusion + channel attention
        # ===============================
        self.ca = nn.ModuleList([ChannelAttention((in_channel//4)*4),
                                BasicConv2d((in_channel//4)*4,in_channel,3,1,1)])


    def forward(self, RGB):
        """
                RGB: 输入特征 (S_k)

                Returns:
                    z: 增强后的特征
                """

        rgb = RGB

        # =====================================
        # Step 1: 1×1降维
        # =====================================

        x1_rgb = self.rgb_b1[0](rgb)
        x2_rgb = self.rgb_b2[0](rgb)
        x3_rgb = self.rgb_b3[0](rgb)
        x4_rgb = self.rgb_b4[0](rgb)

        # =====================================
        # Step 2: CRE增强 + 空洞卷积
        # Progressive Interaction
        # =====================================

        x1_rgb_f = self.rgb_b1[1](x1_rgb + x1_rgb*self.rgb_b1_sa(x1_rgb*self.rgb_b1_ca(x1_rgb)))

        x2_rgb_f = self.rgb_b2[1]((x2_rgb+x1_rgb_f) + (x2_rgb+x1_rgb_f) * self.rgb_b2_sa((x2_rgb+x1_rgb_f) * self.rgb_b2_ca(x2_rgb+x1_rgb_f)))

        x3_rgb_f = self.rgb_b3[1]((x3_rgb+x2_rgb_f) + (x3_rgb+x2_rgb_f) * self.rgb_b3_sa((x3_rgb+x2_rgb_f) * self.rgb_b3_ca(x3_rgb+x2_rgb_f)))

        x4_rgb_f = self.rgb_b4[1]((x4_rgb+x3_rgb_f) + (x4_rgb+x3_rgb_f) * self.rgb_b4_sa((x4_rgb+x3_rgb_f) * self.rgb_b4_ca(x4_rgb+x3_rgb_f)))

        y = torch.cat((x1_rgb_f,x2_rgb_f, x3_rgb_f,x4_rgb_f),1)
        y_ca = self.ca[1](y*self.ca[0](y))
        z = y_ca + rgb

        return z
