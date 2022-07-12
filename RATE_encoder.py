import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.utils import make_grid


import torch.nn as nn
import torch.nn.functional as F
import torch



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--code_dim", type=int, default=5, help="latent code")
parser.add_argument("--img_size", type=int, default=36, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_block = nn.Sequential(

            nn.Conv2d(3, 48, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(48, 64, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 160, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(160, 192, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(192, 192, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 192, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),

        )

        self.fc3 = nn.Sequential(nn.Linear(3072, 3072))
        self.fc4 = nn.Sequential(nn.Linear(3072, opt.code_dim))


    def forward(self, img):

        x = self.conv_block(img)
        #print ("x.shape", x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc3(x)
        x = self.fc4(x)

        return x




class transformation_2D(nn.Module):

    def __init__(self):
        super(transformation_2D, self).__init__()

    def stn(self, x, matrix_2D):

        grid = F.affine_grid(matrix_2D, x.size())
        x = F.grid_sample(x, grid, padding_mode = 'border')
        return x

    def forward(self, img, matrix_2D):
        out = self.stn(img, matrix_2D)

        return out

# Loss functions

continuous_loss = torch.nn.MSELoss()

# Loss weights

lambda_affine = 1

# Initialize generator and discriminator

encoder = Encoder()
trans_2D = transformation_2D()


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")


if cuda:
    encoder.cuda()
    trans_2D.cuda()
    continuous_loss.cuda()



transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])


dataset = datasets.ImageFolder('GTSRB-Training_fixed/GTSRB/Training', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=opt.batch_size, 
                                            shuffle=True, num_workers = 16)



# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling


# ----------
#  Training
# ----------


def get_matrix(code_input_raw):


    batch_size = code_input_raw.shape[0]
    code_input = torch.zeros((code_input_raw.shape[0], code_input_raw.shape[1])).cuda()


    r_factor = 12 # 12 rotation
    pq_factor = 0.1
    xy_factor = 0.1  # translation
    #mn_factor = 0.1  # shear

    code_input[:,0] = code_input_raw[:,0] * np.pi/r_factor #theta
    code_input[:,1] = code_input_raw[:,1] * pq_factor + 1 #p
    code_input[:,2] = code_input_raw[:,2] * pq_factor + 1 #q
    code_input[:,3] = code_input_raw[:,3]*xy_factor #x
    code_input[:,4] = code_input_raw[:,4]*xy_factor #y
    #code_input[:,5] = code_input_raw[:,5]*mn_factor #m
    #code_input[:,6] = code_input_raw[:,6]*mn_factor #n


    #rotation matrix
    rot_matrix = torch.zeros(((batch_size, 3,3))).cuda()
    rot_matrix[:,0,0] = torch.cos(code_input[:,0])
    rot_matrix[:,0,1] = -torch.sin(code_input[:,0])
    rot_matrix[:,1,0] = torch.sin(code_input[:,0])
    rot_matrix[:,1,1] = torch.cos(code_input[:,0])
    rot_matrix[:,2,2] = 1

    #scale matrix
    scale_matrix = torch.zeros(((batch_size, 3,3))).cuda()
    scale_matrix[:,0,0] = code_input[:,1]
    scale_matrix[:,1,1] = code_input[:,2]
    scale_matrix[:,2,2] = 1

    #translation matrix
    trans_matrix = torch.zeros(((batch_size, 3,3))).cuda()
    trans_matrix[:,0,0] = 1
    trans_matrix[:,1,1] = 1
    trans_matrix[:,0,2] = code_input[:,3]
    trans_matrix[:,1,2] = code_input[:,4]
    trans_matrix[:,2,2] = 1

    #shear matrix
    
    #shear_matrix = torch.zeros(((batch_size, 3,3))).cuda()
    #shear_matrix[:,0,0] = 1
    #shear_matrix[:,1,1] = 1
    #shear_matrix[:,0,1] = code_input[:,5]
    #shear_matrix[:,1,0] = code_input[:,6]
    #shear_matrix[:,2,2] = 1
    

    #A_matrix = rot_matrix @ scale_matrix @ trans_matrix @ shear_matrix
    A_matrix =   rot_matrix  @scale_matrix @ trans_matrix


    return A_matrix


for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Configure input
        ori_imgs = Variable(imgs.type(FloatTensor))

        code_input_array = np.random.uniform(-1, 1, (batch_size, opt.code_dim))
        code_input_original = Variable(FloatTensor(code_input_array))

        code_input_0 = code_input_original.clone()

        # ---------------------
        #  Train Encoder
        # --------------------
        optimizer_E.zero_grad()

        ori_code = encoder(ori_imgs)

        A_matrix_0 = get_matrix(code_input_0)
        trans_img_0 = trans_2D(ori_imgs, A_matrix_0[:,0:2])
        trans_code_0 = encoder(trans_img_0)

        
        # version 1
        
        # -------------
        predict_affine_0 = trans_code_0.clone()

        ori_M = get_matrix(ori_code)
        trans_M = get_matrix(trans_code_0)
        inv_ori_M = torch.inverse(ori_M)
        pred_M = trans_M @ inv_ori_M
        

        affine_loss_cont = 10* continuous_loss(pred_M[:,0:2], A_matrix_0[:,0:2])
        # -------------
        ''' 
        
        # version 2 
        # -------------
        pred_code = trans_code_0 - ori_code
        affine_loss_cont = continuous_loss(pred_code, code_input_original)

        # -------------
        '''
        
    
        affine_loss = affine_loss_cont

        affine_loss.backward()
        optimizer_E.step()

        batches_done = epoch * len(dataloader) + i
        
        # --------------
        # Log Progress
        # --------------
        if batches_done % 100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader),  affine_loss.item())
            )

        if batches_done  == 6000:

            torch.save(encoder.state_dict(), "encoder_rst_%d.pt" % batches_done)