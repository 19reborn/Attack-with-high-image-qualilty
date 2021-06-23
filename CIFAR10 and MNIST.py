#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pytorch_ssim
# https://github.com/Po-Hsun-Su/pytorch-ssim
from torch.utils.data import DataLoader


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 选择 dataset
batch_size = 10
eps = 0.031
step_size = 2
iteration = 10
K = 0.05


# In[3]:


def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


# In[4]:


transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.CIFAR10("data_CIFAR10",train=False,download=True,transform = transform)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
from model.PreActResNet18 import PreActResNet18
net = PreActResNet18(num_classes=10)
ckpt = filter_state_dict(torch.load("model/AT-AWP_cifar10_l2_preactresnet18.pth", map_location=device))
mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)    
net.load_state_dict(ckpt)
model = nn.Sequential(Normalize(mean=mean, std=std), net)
model.to(device)


# In[5]:


def multi_imshow(imglist, labellist):
    figNum = len(imglist)
    plt.figure(figsize=(5*figNum, 5))
    namelist = []
    for i in range(1, figNum + 1):
        namelist.append('img' + str(i))
        namelist[i - 1] = plt.subplot2grid((1, figNum),(0, i - 1))
        plt.imshow(np.transpose(imglist[i - 1].cpu(),(1,2,0)))
        namelist[i - 1].set_title(labellist[i - 1])
    plt.show()
    # sp = savepath + "img" + str(figNum) + '.png'
    # plt.savefig(sp)


# In[6]:


# l2 Loss
def l2(img1,img2):
    channel = img1.size()[0]
    height = img1.size()[1]
    width = img1.size()[2]
    total_dif = 0.0
    for i in range(0,channel):
        for j in range(0,height):
            for k in range(0, width):
                total_dif += (img1[i][j][k].item()-img2[i][j][k].item()) ** 2
    return total_dif ** 0.5


# In[7]:


# l1 Loss
def l1(img1,img2):
    channel = img1.size()[0]
    height = img1.size()[1]
    width = img1.size()[2]
    total_dif = 0
    for i in range(0, channel):
        for j in range(0, height):
            for k in range(0, width):
                total_dif += abs(img1[i][j][k].item() - img2[i][j][k].item())
    return total_dif


# In[8]:


# l0 Loss 计算差距<=0.01像素点的数目
def l0(img1,img2):
    channel = img1.size()[0]
    height = img1.size()[1]
    width = img1.size()[2]
    total_dif = 0
    for i in range(0, channel):
        for j in range(0, height):
            for k in range(0, width):
                total_dif += 0 if abs(img1[i][j][k].item() - img2[i][j][k].item()) <= 0.01 else 1
    return total_dif


# In[9]:


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, delta, y, images, lam1 , lam2 ):
        mask = torch.ones_like(x).scatter_(1, y.unsqueeze(1), 0.)
        rest = x[mask.bool()].view(x.size(0), x.size(1) - 1)
        xx = torch.tensor(range(x.size(0)))
        f = torch.nn.ReLU()
        loss_attack = 1*f(x[xx, y] - torch.max(rest, 1)[0] + K)
        #print(loss_attack)
        loss_ssim = 0.01*lam1 *(1 - (pytorch_ssim.ssim(images,images+delta)))
        #loss_L_inf = 0.1*lam2 * (torch.max(torch.abs(delta)))
        loss_L_2 = 0.0001*lam2 * torch.dist(images,images+delta ,p=2)
        img = images+delta
        loss_var = 500*(torch.norm(img[:,:,1:,:] - img[:,:,:-1,:])+torch.norm(img[:,:,:,1:] - img[:,:,:,:-1]))/(images.size(0)*images.size(1)*images.size(2)*images.size(3))
        #print(loss_var)
        #loss_var = 100* (torch.norm(torch.Tensor(np.gradient((images+delta).detach().cpu().numpy(),axis=(2,3))))/(images.size(0)*images.size(1)*images.size(2)*images.size(3))).to(device)
        return torch.mean(loss_attack + loss_ssim + loss_L_2)+loss_var
        #return torch.mean(f(x[xx, y] - torch.max(rest, 1)[0] + K) + lam *(eps + torch.max(delta)))


# In[10]:


def PGD_attack_MyLoss(net, images, labels, eps, step_size):
    images = images.to(device)
    labels = labels.to(device)

    delta = torch.zeros(images.size()).to(device)  #变量求导
    lam1 = torch.zeros(1).to(device)
    lam2 = torch.zeros(1).to(device)
    #delta.uniform_(-eps, eps).to(device)
    delta = Variable(delta, requires_grad=True)
    lam1 = Variable(lam1,requires_grad=True)
    lam2 = Variable(lam2,requires_grad=True)
    # print(images.size())
    # print(labels.size())
    # ori_images = images.data
    criterion = My_loss()
    for i in range(iteration):
        # delta.requires_grad = True
        # lam.requires_grad = True
        # print(images.size())
        outputs = net(images + delta)
        net.zero_grad()
        loss = criterion(outputs, delta, labels,images,lam1,lam2).to(device)
        loss.backward()

        # print(delta.grad)
        new_delta = delta - step_size * delta.grad.detach()
        adv_images = torch.clamp(images + new_delta, min=0, max=1)
        delta = adv_images - images
        new_lam1 = lam1 + step_size * lam1.grad.detach()
        if new_lam1 < 0:
            lam1 = torch.zeros(1).to(device)
        else:
            lam1 = new_lam1
            
        new_lam2 = lam2 + step_size *0.3* lam2.grad.detach()
        if new_lam2 < 0:
            lam2 = torch.zeros(1).to(device)
        else:
            lam2 = new_lam2
            
        # delta = torch.clamp(delta, min=-eps, max=eps)
        delta = Variable(delta, requires_grad=True)
        lam1 = Variable(lam1,requires_grad=True)
        lam2 = Variable(lam2,requires_grad=True)
        # for i in range(0, images.size()[0]):
        #     print(torch.max(delta[i]).item())
        #     print(torch.min(delta[i]).item())
    deltas = []
    for i in range(0,images.size()[0]):
        deltas.append(torch.max(torch.abs(delta[i])).item())
    return images + delta, deltas


# In[11]:


def PGD_attack_Conventional(model, image, label, eps, step_size, iters=10):
    image = image.to(device)
    label = label.to(device)
    loss = nn.NLLLoss()
    ori_image = image.data

    for i in range(iters):
        image.requires_grad = True
        output = model(image)
        model.zero_grad()
        cost = loss(output, label).to(device)
        cost.backward()
        adv_image = image + step_size * image.grad.sign()
        delta = torch.clamp(adv_image - ori_image, min=-eps, max=eps)
        image = torch.clamp(ori_image + delta, min=0, max=1).detach_()
    deltas = []
    delta = image-ori_image 
    for i in range(0,images.size()[0]):
        deltas.append(torch.max(torch.abs(delta[i])).item())
    return image ,deltas 


# In[12]:


def fsgm_attack(model, image, label, eps):
    image = image.to(device)
    label = label.to(device)
    loss = nn.NLLLoss()

    image.requires_grad = True
    output = model(image)
    model.zero_grad()
    cost = loss(output, label).to(device)
    cost.backward()

    image = image + eps * image.grad.sign()
    image = torch.clamp(image, min=0, max=1)
    return image


# In[13]:


def cw_l2_attack(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01) :
    import torch.optim as optim
    images = images.to(device)     
    labels = labels.to(device)

    # Define f-function
    def f(x) :

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10
    
    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        #loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss1 = torch.mean(torch.dist(images,a ,p=2))
        #loss1 = torch.mean(torch.max(torch.abs(a-images)))
        loss2 = torch.mean(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                delta = a -images
                deltas=[]
                for i in range(0,images.size()[0]):
                    deltas.append(torch.max(torch.abs(delta[i])).item())                
                return a, deltas
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    delta = attack_images -images
    deltas=[]
    for i in range(0,images.size()[0]):
        deltas.append(torch.max(torch.abs(delta[i])).item())
        
    return attack_images, deltas


# In[ ]:


import sys
sys.path.append('/home/wangyiming/IQA/attack/deepIQA')
from deepIQA.evaluate import eval
# PGD TEST
# N = Novel Approach, C = Conventional Approach
model.eval()
success_N = 0
success_C = 0
success_F = 0
success_cw = 0
SSIM_N_total = 0
SSIM_C_total = 0
SSIM_F_total = 0
SSIM_cw_total = 0
IQA_N_total = 0
IQA_C_total = 0
IQA_F_total = 0
IQA_cw_total = 0
total = 0
pred_suc = 0
attpre_suc = 0
batch_num = 0
for images, labels in test_loader:
    batch_num += 1
    images = images.to(device)
    labels = labels.to(device)
    outputs_N = model(images)
    _, ori_preds = torch.max(outputs_N, 1)   #original outputs
    index = np.arange(labels.size(0))
    # print(labels.size())
    index = index[ori_preds.cpu() == labels.cpu()]
    # print(images.size())
    images = images[index]
    labels = labels[index]
    
    #print(images.size())
    # SSIM_attack
    att_images_N, deltas = PGD_attack_MyLoss(model, images, labels, eps, step_size)
    att_outputs_N = model(att_images_N)
    _, att_preds_N = torch.max(att_outputs_N, 1)
    success_N += (labels != att_preds_N).sum().item()
    # success += (ori_preds != att_preds_N).sum().item()
    # pred_suc += (ori_preds == labels).sum().item()
    # attpre_suc += (att_preds_N == labels).sum().item()

    # PGD_attack
    att_images_C, deltas_conv = PGD_attack_Conventional(model, images, labels, eps = eps,step_size=0.025)
    att_outputs_C = model(att_images_C)
    _, att_preds_C = torch.max(att_outputs_C.data, 1)
    success_C += (labels != att_preds_C).sum().item()

    # FSGM_attack
    att_images_F = fsgm_attack(model, images, labels, eps = eps)
    att_outputs_F = model(att_images_F)
    _, att_preds_F = torch.max(att_outputs_F.data, 1)
    success_F += (labels != att_preds_F).sum().item()

    # CW_attack
    att_images_cw, L_inf_cw = cw_l2_attack(model, images, labels, targeted=False, c=40)
    att_outputs_cw = model(att_images_cw)
    _, att_preds_cw = torch.max(att_outputs_cw.data, 1)
    success_cw += (labels != att_preds_cw).sum().item()

    
    # Display Result
    total += labels.size(0)
    for i in range (0, images.size()[0]):
        SSIM_N = pytorch_ssim.ssim(images[i].unsqueeze(0),att_images_N[i].unsqueeze(0)).item()
        SSIM_C = pytorch_ssim.ssim(images[i].unsqueeze(0),att_images_C[i].unsqueeze(0)).item()
        SSIM_F = pytorch_ssim.ssim(images[i].unsqueeze(0),att_images_F[i].unsqueeze(0)).item()
        SSIM_cw = pytorch_ssim.ssim(images[i].unsqueeze(0),att_images_cw[i].unsqueeze(0)).item()
        
        SSIM_N_total += SSIM_N
        SSIM_C_total += SSIM_C
        SSIM_F_total += SSIM_F
        SSIM_cw_total += SSIM_cw
        
        IQA_N = eval(np.transpose(images[i].detach().cpu().numpy(),(1,2,0))*255,np.transpose(att_images_N[i].detach().cpu().numpy(),(1,2,0))*255)
        IQA_C = eval(np.transpose(images[i].detach().cpu().numpy(),(1,2,0))*255,np.transpose(att_images_C[i].detach().cpu().numpy(),(1,2,0))*255)
        IQA_F = eval(np.transpose(images[i].detach().cpu().numpy(),(1,2,0))*255,np.transpose(att_images_F[i].detach().cpu().numpy(),(1,2,0))*255)
        IQA_cw = eval(np.transpose(images[i].detach().cpu().numpy(),(1,2,0))*255,np.transpose(att_images_cw[i].detach().cpu().numpy(),(1,2,0))*255)
        
        IQA_N_total += IQA_N
        IQA_C_total += IQA_C
        IQA_F_total += IQA_F
        IQA_cw_total += IQA_cw
        
        
        if(batch_num % 10 == 0):
            l0_N = l0(images[i], att_images_N[i])
            l0_C = l0(images[i], att_images_C[i])
            l0_F = l0(images[i], att_images_F[i])
            #l0_cw = l0(images[i], att_images_cw[i])
            l1_N = l1(images[i], att_images_N[i])
            l1_C = l1(images[i], att_images_C[i])
            l1_F = l1(images[i], att_images_F[i])
            #l1_cw = l1(images[i], att_images_cw[i])
            l2_N = l2(images[i], att_images_N[i])
            l2_C = l2(images[i], att_images_C[i])
            l2_F = l2(images[i], att_images_F[i])
            #l2_cw = l2(images[i], att_images_cw[i])
            imglist = []
            imglist.append(torchvision.utils.make_grid(images[i].data, normalize=True))
            imglist.append(torchvision.utils.make_grid(att_images_N[i].data, normalize=True))
            imglist.append(torchvision.utils.make_grid(att_images_C[i].data, normalize=True))
            imglist.append(torchvision.utils.make_grid(att_images_F[i].data, normalize=True))
            #imglist.append(torchvision.utils.make_grid(att_images_cw[i].data, normalize=True))
            labellist = []
            labellist.append("Original: " + str(test_dataset.classes[labels[i]]))
            labellist.append("SSIM_attack: " + str(test_dataset.classes[att_preds_N[i]])
                        + '\n' + "L_0 = " + str(round(l0_N, 3)) + '\n' + "L_1 = " + str(round(l1_N,3)) + ", L_2 = " + str(round(l2_N,3))
                        + '\n' + 'L_inf = ' + str(round(deltas[i],3)) + ', SSIM = ' + str(round(SSIM_N,3)) + ', MOS = ' + str(round(IQA_N,3)))
            labellist.append("PGD_attack: " +str(test_dataset.classes[att_preds_C[i]])
                        + '\n' + "L_0 = " + str(round(l0_C, 3)) + '\n' + "L_1 = " + str(round(l1_C,3)) + ", L_2 = " + str(round(l2_C,3))
                        + '\n' + 'L_inf = ' + str(0.031) + ', SSIM = ' + str(round(SSIM_C, 3))+ ', MOS = ' + str(round(IQA_C,3)))
            labellist.append("FSGM_attack: " +str(test_dataset.classes[att_preds_F[i]])
                        + '\n' + "L_0 = " + str(round(l0_F, 3)) + '\n' + "L_1 = " + str(round(l1_F,3)) + ", L_2 = " + str(round(l2_F,3))
                        + '\n' + 'L_inf = ' + str(0.031) + ', SSIM = ' + str(round(SSIM_F, 3)) + ', MOS = ' + str(round(IQA_F,3)))
            labellist.append("CW_attack: " +str(test_dataset.test_labels[att_preds_cw[i]])
                        + '\n' + "L_0 = " + str(round(l0_cw, 3)) + '\n' + "L_1 = " + str(round(l1_cw,3)) + ", L_2 = " + str(round(l2_cw,3))
                        + '\n' + 'L_inf = ' + str(round(L_inf_cw[i],3)) + ', SSIM = ' + str(round(SSIM_cw, 3)) + ', MOS = ' + str(round(IQA_cw,3)) )
            multi_imshow(imglist, labellist)
    print("Total Image Count:", total,"Success Rate:\n", 
              "SSIM_attack:", success_N / total, "PGD_attack:", success_C / total, "FSGM_attack:", success_F / total,
              "CW_attack:", success_cw / total)
    print("Average SSIM: SSIM_attack:",round(SSIM_N_total/total,3), "PGD_attack:",round(SSIM_C_total/total,3)
         ,"FSGM_attack:",round(SSIM_F_total/total,3),"CW_attack:",round(SSIM_cw_total/total,3))
    print("Average MOS: SSIM_attack:",round(IQA_N_total/total,3), "PGD_attack:",round(IQA_C_total/total,3)
         ,"FSGM_attack:",round(IQA_F_total/total,3),"CW_attack:",round(IQA_cw_total/total,3))
    if(batch_num >=  100):
        break


# In[ ]:





# In[ ]:




