import torch
import os
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.dataloader_new import get_loader
from utils.utils import clip_gradient
from config import Config
from models.Model_PVTv2_b4 import CL_Model
from CLoss import structure_loss,dice_loss,cross_entropy2d_edge,Fore_Back_CLoss_New

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(42)

cfg = Config()
model = CL_Model().to(cfg.device)
model = nn.DataParallel(model).to(cfg.device)
total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
print('Number of parameter : %.2fM'%(total/1e6))


FBC_Loss1 = Fore_Back_CLoss_New(cfg.temperature1,cfg.temperature2).cuda()
FBC_Loss2 = Fore_Back_CLoss_New(cfg.temperature1,cfg.temperature2).cuda()
FBC_Loss3 = Fore_Back_CLoss_New(cfg.temperature1,cfg.temperature2).cuda()
FBC_Loss4 = Fore_Back_CLoss_New(cfg.temperature1,cfg.temperature2).cuda()
FBC_Loss5 = Fore_Back_CLoss_New(cfg.temperature1,cfg.temperature2).cuda()


train_dataloader = get_loader(cfg.rgb_path, cfg.GT_path, cfg.Edge_path, cfg.batch_size, cfg.train_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
total_step = len(train_dataloader)
params = model.parameters()
optimizer = torch.optim.AdamW(params, cfg.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 学习率调节器，这里采用阶段调节

def train():
    print('Let us start to train the model:')
    ce = []
    cl = []
    lo = []
    edge = []
    epoch_loss = []

    for epoch in range(cfg.num_epochs):
        model.train()
        ep_los = 0
        for i, data in enumerate(train_dataloader, start=1):  # 下标从1开始

            optimizer.zero_grad()  # 每一步都要将之前的梯度进行清零

            images, gts, edges = data
            images = images.to(cfg.device)
            gts = gts.to(cfg.device)
            edges = edges.to(cfg.device)

            s1,s2,s3,s4,s5,e1,e2,e3,e4,e5,f_c1,b_c1,f_c2,b_c2,f_c3,b_c3,f_c4,b_c4,f_c5,b_c5 = model(images,gts)

            loss_ce1 = structure_loss(s1,gts)
            loss_ce2 = structure_loss(s2,gts)
            loss_ce3 = structure_loss(s3,gts)
            loss_ce4 = structure_loss(s4,gts)
            loss_ce5 = structure_loss(s5,gts)

            loss_ce = cfg.af*loss_ce1+cfg.bt*loss_ce2+cfg.ct*loss_ce3+cfg.gm*loss_ce4+cfg.om*loss_ce5

            ce.append(loss_ce.item())

            loss_ed1 = dice_loss(e1, edges) + cross_entropy2d_edge(e1, edges)
            loss_ed2 = dice_loss(e2, edges) + cross_entropy2d_edge(e2, edges)
            loss_ed3 = dice_loss(e3, edges) + cross_entropy2d_edge(e3, edges)
            loss_ed4 = dice_loss(e4, edges) + cross_entropy2d_edge(e4, edges)
            loss_ed5 = dice_loss(e5, edges) + cross_entropy2d_edge(e5, edges)

            loss_ed = cfg.af*loss_ed1+cfg.bt*loss_ed2+cfg.ct*loss_ed3+cfg.gm*loss_ed4+cfg.om*loss_ed5
            edge.append(loss_ed.item())


            loss_cl1 = FBC_Loss1(f_c1,b_c1)
            loss_cl2 = FBC_Loss2(f_c2,b_c2)
            loss_cl3 = FBC_Loss3(f_c3,b_c3)
            loss_cl4 = FBC_Loss4(f_c4,b_c4)
            loss_cl5 = FBC_Loss5(f_c5,b_c5)

            loss_cl = cfg.af*loss_cl1+cfg.bt*loss_cl2+cfg.ct*loss_cl3+cfg.gm*loss_cl4+cfg.om*loss_cl5
            cl.append(loss_cl.item())


            loss = cfg.a * loss_ce + cfg.b * loss_ed + cfg.c * loss_cl
            lo.append(loss.item())
            loss.backward()
            ep_los += loss.item()
            clip_gradient(optimizer, cfg.clip)
            optimizer.step()
            if i % 200 == 0 or i == total_step:
                print(
                    'Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Lossce: {:.4f}, Lossed: {:.4f}, Losscl: {:.4f}, Loss:{:.4f}'.
                    format(epoch, cfg.num_epochs, i, total_step, optimizer.param_groups[0]['lr'], loss_ce.item(),\
                           loss_ed.item(), loss_cl.item(), loss.item()))
        ep_los = ep_los / total_step
        epoch_loss.append(ep_los)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), cfg.save_model_path + '%d' % epoch + 'CL.pth')


if __name__ == '__main__':
    train()