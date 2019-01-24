from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import torch.nn.functional as F
import pretrainedmodels

import xlrd
from skimage import io,transform
import glob
import numpy as np

from tkinter import _flatten
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

path11="E:\陈枫\皮肤镜实验课题所有数据+副本\实验原图+3个压缩"
path12="E:\陈枫\皮肤镜实验课题所有数据+副本\YT2000_3.xls"

'''以下为17年数据路径'''
path1="E:\陈枫\皮肤镜实验课题所有数据+副本\训练原图+切割且压缩"
path1_1="E:\坤展\数据\训练zkz512"
path2="E:\坤展\数据\YT2000.xls"

path3="E:\陈枫\皮肤镜实验课题所有数据+副本\测试原图+切割且压缩"
path3_3="E:\坤展\数据\测试zkz512"
path4="E:\坤展\数据\CS600.xls"

'''以下为16年数据路径'''
path5="E:\\陈枫\皮肤镜实验课题所有数据+副本\\分类+train+test+16年\\train+压缩"
path6="E:\\陈枫\\皮肤镜实验课题所有数据+副本\\分类+train+test+16年\\YT900.xls"
path7="E:\\陈枫\皮肤镜实验课题所有数据+副本\\分类+train+test+16年\\text+压缩"
path8="E:\\陈枫\皮肤镜实验课题所有数据+副本\分类+train+test+16年\\cs379.xls"

path9="E:\\陈枫\皮肤镜实验课题所有数据+副本\\ISIC2018_Task3_Training_Input"
path10="E:\\陈枫\\皮肤镜实验课题所有数据+副本\\YT_2018_task3_train.xls"

w=224
h=224
c=3
#读取图片
def read_img(path1,path2):
    imgs=[] 
    labels=[]
    for im in glob.glob(path1+'/*.jpg'):
            #glob.glob(folder+'/*.jpg')是path1文件夹底下任何一个文件夹里面图片的根目录；eg：D:\\lower_photos\\tulips\\9976515506_d496c5e72c.jpg'
            print('reading the images:%s'%(im))
            img=io.imread(im)                 #读图，img为RGB的3通道数组
            img=transform.resize(img,(w,h))   #img为200*200的3通道数组
#            img = img.permute(2,1,0)
            imgs.append(img)
    count1=0
    for elem1 in imgs:
        count1+=1
    print("imgs 中有%d个元素"%count1)       
    bk = xlrd.open_workbook(path2)
    #shxrange = range(bk.nsheets)
    try:
        sh = bk.sheet_by_name("Sheet1")
    except:
        print("no sheet in %s named Sheet1" % path2)
    #获取行数
    nrows = sh.nrows
    #获取列数
    #ncols = sh.ncols
    #获取各行数据
#    for i in range(0,nrows):              #16年数据用这个
    for i in range(1,nrows):             #17年数据用这个
        row_data = sh.row_values(i)
        labels.append(row_data[1])
    print(labels)
    count2=0
    for elem2 in labels:
        count2+=1
    print("imgs 中有%d个元素"%count1)
    print("labels 中有%d个元素"%count2)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int64)

#对训练集0进行90，80，270的旋转增强
def Augument0(data,label):
    label0=[]
    data0 =[]

    i=0
    for j in label:
#        if j == 1:
        label0.append(j)
        data0.append(data[i])
        i+=1
    rotate90=[]
    rotate180=[]
    rotate270=[]
    label_90=label0
    label_180=label0
    label_270=label0
    for jj in data0:
        rotate90.append(transform.rotate(jj, 90))
        rotate180.append(transform.rotate(jj, 180))
        rotate270.append(transform.rotate(jj, 270))
    data=list(data)
    label=list(label)
    data.extend(rotate90)  
    data.extend(rotate180) 
    data.extend(rotate270)
    label.extend(label_90)
    label.extend(label_180)
    label.extend(label_270)
    return np.asarray(data,np.float32), np.asarray(label,np.int64)    
#data_train,label_train=Augument0(data_train,label_train)


def fliplr_img(data,label):
    label0=[]
    data0 =[]

    i=0
    for j in label:
#        if j == 1:
        label0.append(j)
        data0.append(data[i])
        i+=1
    fli=[]
    for jj in data0:
        fli.append(np.fliplr(jj))
    data=list(data)
    label=list(label)
    data.extend(fli)
    label.extend(label0)
    return np.asarray(data,np.float32), np.asarray(label,np.int64)
#data_train,label_train=fliplr_img(data_train,label_train)


def add_zs(data,label):
    label0=[]
    data0 = data.copy()
    i=0
    for j in label:
    #   if j == 1:
        label0.append(j)
    zs = []
    for jj in data0:        
        # 随机生成500个椒盐
        rows, cols, dims = jj.shape
        for i in range(500):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            jj[x, y, :] = 1
        zs.append(jj)
    data=list(data)
    label=list(label)
    data.extend(zs)
    label.extend(label0)
    return np.asarray(data,np.float32),np.asarray(label,np.int64)

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def one_hot(values):
    n_values = 2
    return np.eye(n_values)[values]

def sp_fn(labels,forecast):
    m=0
    i=0
    k=0
    for a in labels:
        if labels[i] == 0 :
            m=m+1
            if forecast[i] == 0 :
                k=k+1
        i=i+1
    sp=k/m
    return sp

def AAA(a):
    aa=[]
    for i in a:        
        aa.append(i[1])
    return aa

def train_model(model, criterion, optimizer, scheduler,batch_size = 8, num_epochs=25):
    since = time.time()

#    best_model_wts = model.state_dict()
#    best_auc = 0.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
#        for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
#            else:
#                model.train(False)  # Set model to evaluate mode
            
            """每走完一个大循环，将所有训练集数据打乱一次（每次的训练集都被打乱，输入的epoch改变）"""
            #打乱顺序
            num=data_train.shape[0]     #所有图片的数量，小版本数据集为455张图
            arry=np.arange(num)    #把0-454这些数在一个矩阵内按顺序分布
            np.random.shuffle(arry)        #把这454个数在矩阵内随机打乱，即此时arr这个矩阵内的455个数已经打乱
            x_train=data_train[arry]                #data每一个小图片矩阵都按照arr里面的排列顺序进行排列
            y_train=label_train[arry]              #label里面的每一个标签按照arr排列，与data匹配

            
            running_loss = 0.0
            running_corrects = 0
            
#            for start in range(0, num - batch_size + 1, batch_size):
#                slic = arry[start : start + batch_size]            
#                x_train_a, y_train_a = Augument0(x_train[slic], y_train[slic])
##                x_train_a, y_train_a = flim'mplr_img(x_train_a, y_train_a)
##                x_train_a, y_train_a = add_zs(x_train_a, y_train_a)
#                indice = np.arange(len(x_train_a))
#                np.random.shuffle(indice)
#                data = x_train_a[indice]
#                target = y_train_a[indice]
            
            for data, target in minibatches(data_train, label_train, batch_size, shuffle=True):
                
#                target = one_hot(target)
                
                if use_gpu:
                    data, target = torch.from_numpy(data).cuda(), torch.from_numpy(target).cuda()
                    
                else:
                    data, target = torch.from_numpy(data), torch.from_numpy(target)
                
                data, target = Variable(data), Variable(target) #io.imshow(data[0].numpy())
                data = data.permute(0,3,1,2)    #io.imshow(data0[0].permute(1,2,0).numpy())
        #        data1 = data0.permute(0,2,3,1)   #io.imshow(data1[0].numpy())
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(data)
                outputs_softmax = F.softmax(outputs,dim=1)
                
#                loss = criterion(outputs_softmax, target)
                '''把交叉熵拆成了F.softmax与F.nll_loss'''
                loss = F.nll_loss(torch.log(outputs_softmax),target)   

#                loss = -torch.mean(target * torch.log(outputs_softmax), dim=1)
               
                _, preds = torch.max(outputs_softmax.data, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == target.data)
                         
            epoch_loss = float(running_loss) / float(dataset_sizes)
            epoch_acc = float(running_corrects) / float(dataset_sizes)

            print('{} Loss: {:.6f} Acc: {:.5f}'.format(
                phase, epoch_loss, epoch_acc))
            

            # deep copy the model
#            if phase == 'val' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#                best_model_wts = model.state_dict()
        hh1=[]
        jj1=[]
        kk1=[]
        mm1=[]
        # Each epoch has a training and validation phase
#        for phase in ['train', 'val']:
        for phase in ['val']:
            if phase == 'val':
                model.train(False)  # Set model to training mode
#            else:
#                model.train(False)  # Set model to evaluate mode

            loss_val = 0.0
            corrects_val = 0


            
            for x_data, y_target in minibatches(x_val, y_val, batch_size, shuffle=True):
                if use_gpu:
                    x_data, y_target = torch.from_numpy(x_data).cuda(), torch.from_numpy(y_target).cuda()
                else:
                    x_data, y_target = torch.from_numpy(x_data), torch.from_numpy(y_target)
                
                x_data, y_target = Variable(x_data), Variable(y_target) #io.imshow(data[0].numpy())
                x_data = x_data.permute(0,3,1,2)    #io.imshow(data0[0].permute(1,2,0).numpy())
        #        data1 = data0.permute(0,2,3,1)   #io.imshow(data1[0].numpy())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs_val = model(x_data)
                outputs_val_softmax = F.softmax(outputs_val, dim = 1)
                
#                loss_val = criterion(outputs_val_softmax, y_target)
                loss_val = F.nll_loss(torch.log(outputs_val_softmax),y_target)
               
                # statistics
                
                _, preds_val = torch.max(outputs_val_softmax.data, 1)
                
                loss_val += loss_val.item()
                corrects_val += torch.sum(preds_val == y_target.data)
                
#                h1 = torch.tensor(y_target).cpu()
                h1 = np.asarray(y_target.clone().detach().cpu()).tolist()
                
#
#                j1 = torch.tensor(preds_val).cpu()
                j1 = list(_flatten(np.asarray(preds_val.clone().detach().cpu()).tolist()))
#                k1 = h1
#                m1 = torch.tensor(outputs_val_softmax).cpu()
#                outputs_val_softmax.clone().detach().cpu()
                m1 = outputs_val_softmax.clone().detach().cpu().detach().numpy().tolist()
                
                hh1.extend(h1)
                jj1.extend(j1)
                mm1.extend(m1)
        

                
            epoch_val_loss = float(loss_val) / float(x_val_sizes)
            epoch_val_acc = float(corrects_val) / float(x_val_sizes)

            print('{} Loss: {:.6f} Acc: {:.5f}'.format(
                phase, epoch_val_loss, epoch_val_acc))
            
            kk1 = np.asarray(one_hot(hh1))
#            mm1 = np.asarray(AAA(mm1))
            mm1 = np.asarray(mm1)
            
            SE1=metrics.recall_score(hh1,jj1)
            SP1=sp_fn(hh1,jj1) 
        
            AUC=roc_auc_score(kk1,mm1)   
            AP1=average_precision_score(kk1,mm1)       
            
            print(" SE1: %f" % SE1)
            print(" SP1: %f" % SP1)
            print("AUC1: %f" % AUC)
            print(" AP1: %f" % AP1)
            
#            if phase == 'val' and AUC > best_auc:
#                best_auc = AUC
#                best_model_wts = model.state_dict()
                
            print(time.strftime("%b %d %Y %H:%M:%S",time.localtime()))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    

#    model.load_state_dict(best_model_wts)
    return model



if __name__ == '__main__':

    data_train, label_train = read_img(path11,path12)
#    data_train, label_train = Augument0(data_train, label_train)
    dataset_sizes = len(data_train)
    
    x_val , y_val = read_img(path11,path12)
    x_val_sizes = len(x_val)
    
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    
    # get model and replace the original fc layer with your fc layer
    # model_ft = pretrainedmodels.vgg11(num_classes=1000)

#    dim_feats = model_ft.last_conv.in_channels
#    nb_classes = 2
#    model_ft.last_conv = nn.Conv2d(dim_feats, nb_classes, kernel_size=1)


    dim_feats = model_ft.last_linear.in_features # =2048
    nb_classes = 2
    model_ft.last_linear = nn.Linear(dim_feats, nb_classes).cuda()
    
    '''dpn系列使用以下方式 '''
    dim_feats = model_ft.last_linear.in_channels
    nb_classes = 2
    model_ft.last_linear = nn.Conv2d(dim_feats, nb_classes, kernel_size=1, bias=True)
     

    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()
                
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
#    optimizer_ft = optim.RMSprop(model_ft.parameters(),lr=0.001,weight_decay=0.99,momentum=0.9)

    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.2)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           batch_size = 4,
                           num_epochs=155)


#    PATH = 'E:/陈枫/pytorch/moxin_baocun/CF/fbresnet152_08.pth'
##    torch.save(model_ft, './模型保存/123.pth')
#    torch.save(model_ft.state_dict(), PATH)



















