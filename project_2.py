#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:12:09 2019

@author: gongchaoyun
"""


import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import glob
import matplotlib.pyplot as plt
import random
import time

from PIL import Image

def resize3d(img, size):
    return (F.adaptive_avg_pool3d(Variable(img,volatile=True), size)).data

## grab video name list
#filename=glob.glob(r'/students/u6292748/Desktop/dynamic_gestures/data/*.avi')
#depth_name=glob.glob(r'/students/u6292748/Desktop/dynamic_gestures/data/depth/*.avi')
#filename=glob.glob(r'/Users/by/Desktop/dynamic_gestures/data/*.avi')
#depth_name=glob.glob(r'/Users/by/Desktop/dynamic_gestures/data/depth/*.avi')

filename=glob.glob(r'/students/u6329142/Desktop/LISA_HG_Data/dynamic_gestures/data/*.avi')
depth_name=glob.glob(r'/students/u6329142/Desktop/LISA_HG_Data/dynamic_gestures/data/depth/*.avi')
random.seed(7)
np.random.seed(7)
inde=list(np.arange(0,len(depth_name)))
rgbset1,rgbset2,rgbset3,rgbset4,rgbset5,rgbset6,rgbset7,rgbset8=[],[],[],[],[],[],[],[]
depset1,depset2,depset3,depset4,depset5,depset6,depset7,depset8=[],[],[],[],[],[],[],[]
np.random.shuffle(inde)
for i in inde[:180]:
    rgbset1.append(filename[i])
    depset1.append(depth_name[i])
     
for i in inde[180:360]:
    
    rgbset2.append(filename[i])
    depset2.append(depth_name[i])

for i in inde[360:540]:
    rgbset3.append(filename[i])
    depset3.append(depth_name[i])
     
for i in inde[540:720]:
    
    rgbset4.append(filename[i])
    depset4.append(depth_name[i])
    
for i in inde[720:900]:
    rgbset5.append(filename[i])
    depset5.append(depth_name[i])
     
for i in inde[900:1080]:
    
    rgbset6.append(filename[i])
    depset6.append(depth_name[i])

for i in inde[1080:1260]:
    rgbset7.append(filename[i])
    depset7.append(depth_name[i])
     
for i in inde[1260:]:
    
    rgbset8.append(filename[i])
    depset8.append(depth_name[i])
    
train_filename=rgbset2+rgbset3+rgbset4+rgbset1+rgbset8+rgbset6+rgbset7
train_depth=depset2+depset3+depset4+depset1+depset8+depset6+depset7

test_filename=rgbset5
test_depth=depset5

#filename=glob.glob(r'/Users/by/Desktop/dynamic_gestures/data/*.avi')
#depth_name=glob.glob(r'/Users/by/Desktop/dynamic_gestures/data/depth/*.avi')
#filename=glob.glob(r'/Users/gongchaoyun/Desktop/ENGN8536_project/LISA_HG_Data/dynamic_gestures/data/*.avi')
#depth_name=glob.glob(r'/Users/gongchaoyun/Desktop/ENGN8536_project/LISA_HG_Data/dynamic_gestures/data/depth/*.avi')

#check if cuda avialiable
#cuda_avail = torch.cuda.is_available()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

chooseGestures = ['01','02','03','04','06','07','08','13','14','15','16','21','23','27','28','29','30','31','32','70','80']        

def online(p,image):
    mm,rows,cols,frame=image.shape
    if p[0]==1 :  # dropout
        m=np.random.rand(image.shape[1],image.shape[2],image.shape[3])
        m=m<0.3
        image[0,:,:,:]=image[0,:,:,:]*m
        image[1,:,:,:]=image[1,:,:,:]*m
        
    if p[1]==1:    #affine
        # scale
        scale=np.random.randint(0,2)
        translate=np.random.randint(0,2)
        
        rotation=np.random.randint(0,2)
        
        ffx=1
        ffy=1
        tx=0
        ty=0
        theta=0
        if scale==1:
            
            ffx=np.random.randint(10,14)/10
            ffy=ffx
        if translate==1:
            tx=np.random.randint(-4,4)
            ty=np.random.randint(-8,8)
        if rotation==1:
            theta=np.random.randint(-10,10)
        
        for i in range(frame):
            
            # scaley
            scaling0= cv2.resize(image[0,:,:,i],None,fx=ffx, fy=ffy, interpolation = cv2.INTER_CUBIC)
            scaling1= cv2.resize(image[1,:,:,i],None,fx=ffx, fy=ffy, interpolation = cv2.INTER_CUBIC)
            M = np.float32([[1,0,tx],[0,1,ty]])
            translating0 = cv2.warpAffine(scaling0,M,(cols,rows))
            translating1 = cv2.warpAffine(scaling1,M,(cols,rows))
            w = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),theta,1)
            rotation0 = cv2.warpAffine(translating0,w,(cols,rows))
            rotation1 = cv2.warpAffine(translating1,w,(cols,rows))
            r_x=rotation1.shape[0]
            r_y=rotation1.shape[1]
            if i==0:
                
                x_d=r_x-rows
                y_d=r_y-cols
                x_start=np.random.randint(0,max(x_d,1))
                y_start=np.random.randint(0,max(y_d,1))
            
        
            image[0,:,:,i]=rotation0[x_start:x_start+rows, y_start:y_start+cols]
            image[1,:,:,i]=rotation1[x_start:x_start+rows,y_start:y_start+cols]
            
            
    if p[2]==1:
        kernel_size=(5,5)
        sigma = 3
        for i in range(frame):
            image[0,:,:,i]=cv2.GaussianBlur(image[0,:,:,i], kernel_size, sigma)
            image[1,:,:,i]=cv2.GaussianBlur(image[1,:,:,i], kernel_size, sigma)
            
    return image
    
    
## define the dataset
class HandDataset(torch.utils.data.Dataset):
    #initial
    #para:
    #   clip_name: list type for clip file names
    #   clip_label:list type for clip labels
    #   file: glob file 
    #   data_transforms: video/image tensor data transforms method            
    def __init__(self, file,depth_file, data_transforms=1):
        clip_label=[]
        clip_name=[]
        depth_name=[]
        file2=[]
        #self.clip_name=file
        #self.depth_name=depth_file
        for i in file:
            #clip_name.append(i)
            label = i.split('/')[-1].split('_')[1]
            if label in chooseGestures:
                real_label = chooseGestures.index(label)
                clip_name.append(i)
                clip_label.append(real_label)
                file2.append(i)
        for i in depth_file:
            label = i.split('/')[-1].split('_')[1]
            if label in chooseGestures:
                depth_name.append(i)
        self.clip_name=clip_name
        self.depth_name=depth_name
        self.file=file2
        self.clip_label = clip_label
        self.data_transforms = data_transforms
        #self.dataset = dataset
        
    
    def __len__(self):
        return len(self.clip_name)

    #for getitem method, first read the clip file (using cv2) as a np array and its label,
    #then transform it    
    def __getitem__(self, item):
        clip_name = self.clip_name[item]
        depth_name=self.depth_name[item]
        clip_label = self.clip_label[item]
        clip_label=int(clip_label)
        if clip_label==19:
            clip_label=5
        if clip_label==20:
            clip_label=6
        
        clip=np.zeros((2,57,125,32),dtype=np.float32)
        clip2 =np.zeros((2,32,57,125),dtype=np.float32)
        cap1 = cv2.VideoCapture(clip_name) 
        cap2 = cv2.VideoCapture(depth_name) 
        
        i=0

        while (i<32):
                ret, frame = cap1.read()
                _,depth_img=cap2.read()
                if not(frame is None):
                    frame = cv2.resize(frame, (125, 57), interpolation=cv2.INTER_CUBIC)
                    frame=cv2.Sobel(frame, cv2.CV_32F, 1, 0)
                    frame=frame/np.max(frame)
                    depth_img=cv2.resize(depth_img, (125, 57), interpolation=cv2.INTER_CUBIC)
                    depth_img=depth_img/255
                    clip[0,:,:,i]=frame[:,:,0] #clip shape:(H* W* i_th frame)
                    clip[1,:,:,i]=depth_img[:,:,0]
                else:
                    clip[:,:,:,i]=clip[:,:,:,i-1]
                i=i+1        
        #print(clip)
        #print('hhh')
        p_flip=np.random.randint(0,2)
        fr_flip=np.random.randint(0,2)
        
        if self.data_transforms==1:
            
            if fr_flip==1:
                clip[0,:,:,:]=np.flip(clip[0,:,:,:],2)
                clip[1,:,:,:]=np.flip(clip[1,:,:,:],2)
            if p_flip==1:
                clip[0,:,:,:]=np.flip(clip[0,:,:,:],1)
                clip[1,:,:,:]=np.flip(clip[1,:,:,:],1)
            
                
            p=np.random.randint(0,2,3)
            clip=online(p,clip)
        #print(clip)
        #plt.imshow(clip[0,:,:,15])
        clip=torch.from_numpy(clip)
        #clip2=torch.from_numpy(clip2)

        

        return clip, clip_label
    
train_transformations = transforms.Compose(
        [
                
                transforms.ToTensor()]
)
    
testset = HandDataset(   file=test_filename,
                          depth_file=test_depth,data_transforms=0)

trainset = HandDataset(   file=train_filename,
                          depth_file=train_depth,data_transforms=1)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=False)
validloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                          shuffle=False)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
'''

for i, data in enumerate(validloader,0):
    images, labels=data
    #print(images.shape)
    #imshow(torchvision.utils.make_grid(images[0,0,:,:,0]))
    if i == 2:
        break
# show images



for epoch in range(2):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            #print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())

'''         
# =============================================================================
# class Unit(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size):
#         super(Unit,self).__init__()
#  
# 
#         self.conv = nn.Conv3d(in_channels=in_channels,kernel_size=kernel_size,out_channels=out_channels,stride=1,padding=1)
#         self.bn = nn.BatchNorm3d(num_features=out_channels)
#         self.relu = nn.ReLU()
#         self.pool=nn.MaxPool3d(kernel_size=2)
# 
#     def forward(self,input):
#         output = self.conv(input)
#         output = self.bn(output)
#         output = self.relu(output)
#         output = self.pool(output)
# 
#         return output
# =============================================================================

class Unit(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,pool_size):
        super(Unit,self).__init__()
 

        self.conv = nn.Conv3d(in_channels=in_channels,kernel_size=kernel_size,out_channels=out_channels,stride=1)
        self.bn = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.pool=nn.MaxPool3d(kernel_size=pool_size)

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        output = self.pool(output)

        return output    
    
# =============================================================================
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.downsample=nn.MaxPool3d(kernel_size=2)
#         self.high_conv1=Unit(2,4,(7,7,5))
#         self.high_conv2=Unit(4,8,(5,5,5))
#         self.high_conv3=Unit(8,32,(5,5,3))
#         self.high_conv4=Unit(32,64,(3,5,3))
#         
#         self.low_conv1=Unit(2,8,(5,5,5))
#         self.low_conv2=Unit(8,32,(5,5,3))
#         self.low_conv3=Unit(32,64,(3,5,3))
#         
#         self.high_fc1 = nn.Linear(64*2*3*2,512)
#         self.high_fc2 = nn.Linear(512,256)
#         self.high_fc3 = nn.Linear(256,19)
#         
#         self.low_fc1 = nn.Linear(64*2*2*4,512)
#         self.low_fc2 = nn.Linear(512,256)
#         self.low_fc3 = nn.Linear(256,19)
#         
#         self.high_net = nn.Sequential(self.high_conv1,self.high_conv2,self.high_conv3,self.high_conv4)
#         self.low_net = nn.Sequential(self.low_conv1,self.low_conv2,self.low_conv3)
#         
#     def forward(self,Hinput):
#         Linput=self.downsample(Hinput)
#         low_output=self.low_net(Linput)
#         low_output=low_output.view(-1,64*2*2*4)
#         low_output=self.low_fc1(low_output)
#         low_output=self.low_fc2(low_output)
#         low_output=self.low_fc3(low_output)
#         
#         return low_output
# =============================================================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.downsample=nn.MaxPool3d(kernel_size=(2,2,1))
        self.high_conv1=Unit(2,4,(7,7,5),(2,2,2))
        self.high_conv2=Unit(4,8,(5,5,3),(2,2,2))
        self.high_conv3=Unit(8,32,(5,5,3),(1,2,1))
        self.high_conv4=Unit(32,64,(3,5,1),(2,3,1))
        
        self.low_conv1=Unit(2,8,(5,5,5),(2,2,2))
        self.low_conv2=Unit(8,32,(5,5,3),(2,2,2))
        self.low_conv3=Unit(32,64,(3,5,3),(1,4,1))
        
        
        self.fc = nn.Linear(64*64,200)
        self.ld1 = nn.Dropout(p=0.5)
        
        self.high_net = nn.Sequential(self.high_conv1,self.high_conv2,self.high_conv3,self.high_conv4)
        self.low_net = nn.Sequential(self.low_conv1,self.low_conv2,self.low_conv3)
        
        
        
    def forward(self,Hinput):
        Linput=self.downsample(Hinput)
        low_output=self.low_net(Linput)
        low_output=low_output.view(-1,64,2*2*4)
        
        high_output=self.high_net(Hinput)
        high_output=high_output.view(-1,64,2*2*4)
        
        #N = Linput.size()[0]
        N = -1
        out = torch.bmm(low_output, torch.transpose(high_output, 1, 2)) / (2*2*4)  # Bilinear
        #assert out.size() == (N,64,64)
        out = out.view(N,64**2)
        out = torch.sqrt(out + 1e-5)
        out = torch.nn.functional.normalize(out)
        out = self.fc(out)
        #out = self.ld1(out)
        
        #assert out.size() == (N, 200)
        
        output=F.log_softmax(out,dim=1)
        
        return output
    
        
model = Net().to(device)
optimizer = Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

def save_models(epoch):
    torch.save(model.state_dict(), "project_08.model")
    print("Chekcpoint saved")

def valid():
    model.eval()
    valid_acc = 0.0
    valid_loss=0.0
    correct = 0
    total = 0
    for data in validloader:
        # Predict classes using images from the test set
        #print(labels)
        images, labels=data
        images = images.to(device)
        labels = labels.to(device)
        labels=torch.tensor(labels)
        #labels-=1
        outputs = model(images)
        #print(outputs)
        _, prediction = torch.max(outputs.data, 1)
        
        #assert outputs==0, 'shut down'
            
        
        loss = loss_fn(outputs, labels)
        valid_loss += loss.item()
        #_, prediction = torch.max(outputs.data, 1)
        #prediction = prediction.cpu().numpy()
            
        #valid_acc += torch.sum(prediction == labels.data)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    # Compute the average acc and loss over all 10000 test images
    #valid_acc = valid_acc / 1000
    print('total',total)
    valid_loss = valid_loss / len(validloader.dataset)
    valid_acc=correct/len(validloader.dataset)

    return valid_acc,valid_loss

def train(num_epochs):
    best_acc = 0.0
    batch_time = time.time()
    total_time = time.time()
    train_acc_plt=np.zeros(num_epochs)
    train_loss_plt=np.zeros(num_epochs)
    valid_acc_plt=np.zeros(num_epochs)
    valid_loss_plt=np.zeros(num_epochs)
    for epoch in range(num_epochs):
        model.train()
        scheduler.step()
        train_acc = 0.0
        train_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader,0):
            images, labels=data
            images = images.to(device)
            labels = labels.to(device)
            labels=torch.tensor(labels)
            #labels-=1
            #print(images, labels)
            # clear gradients
            optimizer.zero_grad()
            # predict the output of input images
            outputs = model(images)
            # calculate the loss according to labels
            #print(outputs.shape,labels)
            loss = loss_fn(outputs, labels)
            # backward transmit loss
            loss.backward()

            # adjust parameters using Adam
            optimizer.step()

            #train_loss += loss.cpu().data[0] * images.size(0)
            train_loss+=loss.item()
            _, prediction = torch.max(outputs.data, 1)
 
            #train_acc += torch.sum(prediction == labels.data)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        batch_time = time.time() - batch_time
        print('time:{:.2f} '.format(batch_time))
        # Calculate training accuracy and loss among training set
        #train_acc = train_acc / 49000
        train_loss = train_loss / len(trainloader.dataset)
        train_acc=correct/len(trainloader.dataset)
        

        # Valdation
        valid_acc,valid_loss = valid()
        
        # save train/valid loss and acc
        train_acc_plt[epoch]=train_acc
        train_loss_plt[epoch]=train_loss
        valid_acc_plt[epoch]=valid_acc
        valid_loss_plt[epoch]=valid_loss

        # 若测试准确率高于当前最高准确率，则保存模型
        if valid_acc > best_acc:
              torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           './model_checkpoint_12_time_dr.pth')
              best_acc = valid_acc

        print(scheduler.get_lr()[0])
        print("best_acc:",best_acc)
        # 打印度量
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Valid Accuracy: {}, Validloss:{}".\
              format(epoch, train_acc, train_loss,valid_acc,valid_loss))

    total_time = time.time() - total_time
    print('time:{:.2f} '.format(total_time))       
    return train_acc_plt,train_loss_plt,valid_acc_plt,valid_loss_plt,total_time
 
    
if __name__ == "__main__":

    train_acc_plt,train_loss_plt,valid_acc_plt,valid_loss_plt,total_time=train(500)
    f=open("project_15_time_dr.txt",'w')
    f.write(str(total_time))
    f.write('\n')
    f.write(str(train_acc_plt))
    f.write('\n')
    f.write(str(train_loss_plt))
    f.write('\n')
    f.write(str(valid_acc_plt))
    f.write('\n')
    f.write(str(valid_loss_plt))
    f.close()        

