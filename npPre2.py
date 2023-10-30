
from torchvision.utils import save_image
from torch import optim, floor
from torch.utils.data import DataLoader
from OptimUtil import *

from src.datahandler.randFlower import flower256, flowerval
from src.loss.ssimLoss import SSIM

import cv2
import torch
import numpy as np
import torch.nn as nn

from src.model.UformerY import UformerY

import argparse, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# saved_checkpoint = torch.load('/data/tmj/LGBnet/pre/Premodel1.pth')
saved_checkpoint = torch.load('/data/tmj/FakerFlower/test2/noPre2/model89.pth')
# saved_checkpointB = torch.load('/data/tmj/LGBnet/pre/model19.pth')

module = UformerY(img_size=256).cuda()



module.load_state_dict(saved_checkpoint,strict=False)
# moduleB.load_state_dict(saved_checkpointB,strict=True)

# LossS = nn.L1Loss(reduction='mean').cuda()
l1loss = nn.L1Loss(reduction='mean').cuda()

# l1loss = nn.CrossEntropyLoss()
# op = optim.AdamW(module.parameters(),lr = 1e-4,weight_decay=1e-5)#定义优化器
train_data = flower256()
bs = 3
train_loader = DataLoader(train_data,batch_size=1,shuffle=False,drop_last=True)


ssim_loss1 = SSIM().cuda()
ssim_loss2 = SSIM().cuda()

start = 90
end = 120



for epoch in range(start,end):
    module.train()

    for batch_id, x in enumerate(train_loader):

        noiseImg = x['real_noisy2'].cuda()
        noiseImg2 = x['real_noisy1'].cuda()

        parmS = set_S_down(module)
        op1 = optim.Adam(parmS, lr=1e-4, weight_decay=1e-6)  # 定义优化器
        preS = noiseImg[:,1,:,:].unsqueeze(1)
        nowS, nowD = module(noiseImg2)
        p, D = module(noiseImg)
        loss1 = l1loss(nowS,preS) + 0.5* l1loss(nowS,p)
        op1.zero_grad()
        loss1.backward()
        op1.step()

        # parmD_down = set_D_down(module)
        # op2 = optim.Adam(parmD_down, lr=1e-5, weight_decay=1e-6)  # 定义优化器
        # preS, preD= module(noiseImg)
        # nowS, nowD = module(noiseImg2)
        #
        # loss2 = 0.5 * ssim_loss1(preD,nowD)
        #
        # if loss2 > 0:
        #     op2.zero_grad()
        #     loss2.backward(retain_graph=False)
        #     op2.step()


        parm_D = set_D(module)
        op3 = optim.Adam(parm_D, lr=1e-4, weight_decay=1e-6)  # 定义优化器
        nowS, nowD = module(noiseImg2)
        preS, preD = module(noiseImg)

        cc = torch.reshape(nowS,(1,-1))
        max = torch.mode(cc)[0]
        pt = noiseImg2[:,1,:,:].unsqueeze(1)


        ssim_out2 = 1 - ssim_loss2(pt,nowD*(nowS/max)) + 0.5 * ssim_loss2(nowD,preD)
        op3.zero_grad()
        ssim_out2.backward(retain_graph=False)
        op3.step()
        # if i > 0 :
        #     save_image(nowS[0] ,  './img/' + str(i) + '_S.png')
        #     print("saved")
        #     save_image(nowD[0] ,  './img/' + str(i) + '_D.png')
        #     save_image(nowS*(nowD/max) ,  './img/' + str(i) + '_o.png')
        #     save_image( (nowD / max), './img/' + str(i) + '_RS.png')
        # print("epoch:   ",epoch+start, "loss1:",loss1.item()," loss3",ssim_out2.data.item())
        print("epoch:   ",epoch, "loss1:",loss1.item()," loss3",ssim_out2.data.item())
        # print("epoch",epoch,"   loss2：",ssim_out2.data.item())

    module.eval()
    testdata = flowerval()
    for i in range(90):
        x = train_data.__getitem__(i)['real_noisy1'].cuda()
        noise = x[1]
        folder_path = './test2/noPre2/' + str(epoch)
        if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(folder_path)

        save_image(noise, folder_path + '/' + str(i) + '_N.png')
        x = x.unsqueeze(0)
        S, D = module(x)

        save_image(S, folder_path + '/' + str(i) + '_S.png')
        save_image(D, folder_path + '/' + str(i) + '_D.png')

    print('pictureSaved')

    torch.save(module.state_dict(), './test2/noPre2/model' + str(epoch ) + '.pth')
    print('model===saved')




