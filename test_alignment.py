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

from torch.nn.utils import spectral_norm


import torch.nn as nn
import torch.nn.functional as F
import torch



os.makedirs("images/original/", exist_ok=True)
os.makedirs("images/distort/", exist_ok=True)
os.makedirs("images/align_1/", exist_ok=True)
os.makedirs("images/align_all/", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--code_dim", type=int, default=5, help="latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
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
        self.fc4 = nn.Sequential(nn.Linear(3072, 5))


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
PATH = "encoder_rst_6000.pt"
encoder.load_state_dict(torch.load(PATH))
encoder.eval()

trans_2D = transformation_2D()



if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")


if cuda:
    encoder.cuda()
    trans_2D.cuda()



transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])


dataset = datasets.ImageFolder('GTSRB_Final_Test_Images/GTSRB/Final_Test', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=opt.batch_size, 
                                            shuffle=False, num_workers = 16)




FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor




# ----------
#  Training
# ----------

def get_distorted_img(imgs):

    
    batch_size = imgs.shape[0]
    code_input_array = np.random.uniform(-1, 1, (batch_size, opt.code_dim))
    code_input_raw = Variable(FloatTensor(code_input_array))

    code_input = code_input_raw.clone()


    r_factor = 12 # 12 rotation
    pq_factor = 0.1  # scale
    xy_factor = 0.1  # translation
    mn_factor = 0.1  # shear
    

    code_input[:,0] = code_input_raw[:,0] * np.pi/r_factor #theta
    code_input[:,1] = code_input_raw[:,1] * pq_factor + 1 #p
    code_input[:,2] = code_input_raw[:,2] * pq_factor + 1 #q
    code_input[:,3] = code_input_raw[:,3]*xy_factor #x
    code_input[:,4] = code_input_raw[:,4]*xy_factor #y
    #code_input[:,5] = code_input_raw[:,5]*mn_factor #m
    #code_input[:,6] = code_input_raw[:,6]*mn_factor #n

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
    '''
    shear_matrix = torch.zeros(((batch_size, 3,3))).cuda()
    shear_matrix[:,0,0] = 1
    shear_matrix[:,1,1] = 1
    shear_matrix[:,0,1] = code_input[:,5]
    shear_matrix[:,1,0] = code_input[:,6]
    shear_matrix[:,2,2] = 1
    '''

    A_matrix =   rot_matrix  @scale_matrix @ trans_matrix

    distorted_img = trans_2D(imgs, A_matrix[:,0:2])

    return distorted_img


def affine_alignment(code_input_raw):


    # 0:theta, 1:p, 2:q, 3:x, 4:y

    batch_size = code_input_raw.shape[0]


    r_factor = 12 # 12 rotation
    pq_factor = 0.1  # scale
    xy_factor = 0.1  # translation
    mn_factor = 0.1  # shear
    
    code_input = code_input_raw.clone()


    code_input[:,0] = code_input_raw[:,0] * np.pi/r_factor #theta
    code_input[:,1] = code_input_raw[:,1] * pq_factor + 1 #p
    code_input[:,2] = code_input_raw[:,2] * pq_factor + 1 #q
    code_input[:,3] = code_input_raw[:,3]*xy_factor #x
    code_input[:,4] = code_input_raw[:,4]*xy_factor #y
    #code_input[:,5] = code_input_raw[:,5]*mn_factor #m
    #code_input[:,6] = code_input_raw[:,6]*mn_factor #n

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
    '''
    shear_matrix = torch.zeros(((batch_size, 3,3))).cuda()
    shear_matrix[:,0,0] = 1
    shear_matrix[:,1,1] = 1
    shear_matrix[:,0,1] = code_input[:,5]
    shear_matrix[:,1,0] = code_input[:,6]
    shear_matrix[:,2,2] = 1
    '''

    A_matrix =   rot_matrix  @scale_matrix @ trans_matrix

    inverse_affine_refined_ori_full = torch.inverse(A_matrix)

    return inverse_affine_refined_ori_full



def sample_image(oringal_imgs, distorted_img, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""


    # orginal image
    grid_real = make_grid(oringal_imgs.data, nrow =n_row)
    save_image(grid_real, "images/original/%d.png" % batches_done, nrow=n_row, normalize=True)

    # sacled image
    grid_distort = make_grid(distorted_img.data, nrow =n_row)
    save_image(grid_distort, "images/distort/%d.png" % batches_done, nrow=n_row, normalize=True)

    align_latent = encoder(distorted_img)
    align_matrix = affine_alignment(align_latent)
    align_img = trans_2D(distorted_img, align_matrix[:,0:2])


    align_latent_1 = encoder(align_img)
    align_matrix_1 = affine_alignment(align_latent_1)
    align_img_2 = trans_2D(align_img, align_matrix_1[:,0:2])


    align_latent_2 = encoder(align_img_2)
    align_matrix_2 = affine_alignment(align_latent_2)
    align_img_3 = trans_2D(align_img_2, align_matrix_2[:,0:2])


    align_latent_3 = encoder(align_img_3)
    align_matrix_3 = affine_alignment(align_latent_3)
    align_img_4 = trans_2D(align_img_3, align_matrix_3[:,0:2])



    align_matrix_all = align_matrix @ align_matrix_1 @ align_matrix_2 @ align_matrix_3

    align_img_all = trans_2D(distorted_img, align_matrix_all[:,0:2])

    grid_align = make_grid(align_img.data, nrow = n_row)
    save_image(grid_align, "images/align_1/%d.png" % batches_done, nrow=n_row, normalize=True)


    grid_align = make_grid(align_img_all.data, nrow = n_row)
    save_image(grid_align, "images/align_all/%d.png" % batches_done, nrow=n_row, normalize=True)







imgs, labels = next(iter(dataloader))

test_img = imgs

print ("test_img", test_img.shape)

oringal_imgs = Variable(test_img.type(FloatTensor))

distorted_list = []

for i in range(10):
    distorted_img = get_distorted_img(oringal_imgs)
    distorted_list.append(distorted_img)

distorted_img = torch.cat(distorted_list, dim = 0)

print ("distorted_img", distorted_img.shape)

selected_list = []
select_index = 1

for j in range(10):
    selected_list.append(distorted_img[j*128 + select_index])

selected_images = torch.stack(selected_list)

print ("selected_images", selected_images.shape)



sample_image(oringal_imgs[select_index], selected_images, n_row= 10, batches_done=0)


