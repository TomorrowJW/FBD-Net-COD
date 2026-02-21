import torch
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 定义配置文件,写入一个配置类中
class Config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 8e-5
    weight_decay = 1e-4
    num_epochs = 150
    a = 1 # 这两个超参数用来调节损失比重
    b = 1
    c = 1
    clip = 0.5

    batch_size = 40
    train_size = 384
    test_size = 384
    num_workers = 8

    #COD_Path
    rgb_path = './data/TrainDataset/Imgs/'  # 训练rgb图片的路径
    GT_path = './data/TrainDataset/GT/'  # 训练标签的路径
    Edge_path = './data/TrainDataset/Edge/'

    test_path = 'G:/0_/3_TOMM_COD_FBD-Net/data/'  # 测试集的路径

    save_model_path = 'G:/0_/3_TOMM_COD_FBD-Net/save_models/'
    save_results_path = 'G:/0_/3_TOMM_COD_FBD-Net/save_results/prediction/'
    save_edge_results_path = 'G:/0_/3_TOMM_COD_FBD-Net/save_results/edges/'


    temperature1 = 1
    temperature2 = 1

    af = 1
    bt = 1
    ct = 1
    gm = 1
    om = 1






