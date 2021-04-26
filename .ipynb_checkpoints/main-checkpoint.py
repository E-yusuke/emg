import torch
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm as tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess as pre
import matplotlib.pyplot as plt
import numpy as np
import model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

Data = None
Labels = None
DATA = {}
LABEL = {}
for name_ in name_list:
    # デスクトップのパス
    path_in = './dataset/{}/'.format(str(name_))
    path_out = './Result/experiment1_FNN_{}/{}/'.format(experiment, str(name_))
    
    try:
        os.makedirs(path_out, exist_ok=True)
    except FileExistsError:
        pass
    data = pre(path_in)
    if experiment == 'IEMG':
        Data, Labels = data.iemgdata()
        input_dim = Data.shape[1]

    elif experiment == 'RMS':
        Data, Labels = data.emgdata()
        input_dim = Data.shape[1]
    DATA[name_] = Data
    LABEL[name_] = Labels
    
    
train_imgs = torchvision.datasets.ImageFolder(
    "./medical_dataset/train/",
    transform=transforms.Compose([
        transforms.ToTensor()]
))

val_imgs = torchvision.datasets.ImageFolder(
    "./medical_dataset/val/",
    transform=transforms.Compose([
        transforms.ToTensor()]
))

test_imgs = torchvision.datasets.ImageFolder(
    "./medical_dataset/test/",
    transform = transforms.Compose([
        transforms.ToTensor()])
)

train_loader = DataLoader(
    train_imgs, batch_size=50, shuffle = True)
val_loader = DataLoader(
    val_imgs, batch_size=50, shuffle=True)
test_loader = DataLoader(
    test_imgs, batch_size=len(test_imgs), shuffle=False)

def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
dataiter = iter(train_loader)
images, labels = dataiter.next()
# 画像の表示
imshow(torchvision.utils.make_grid(images))
# ラベルの表示
print(' '.join('%5s' % labels[labels[j]] for j in range(8)))
print(train_imgs[0][0].view(train_imgs[0][0].shape[0], -1))
print(train_imgs[0][0][0])
print(len(train_loader))
print(len(train_imgs))


imput_num=len(train_imgs)

net = model.FFNNs(input_num).to(device)

writer = SummaryWriter(log_dir='./experiment1')


optimizer=optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss().to(device)

total_batch = len(train_loader)
num_epochs = 10

train_loss_list=[]
val_loss_list = []
train_acc_list=[]
val_acc_list=[]
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(train_loader):
      imgs, labels = data
      imgs = imgs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      out=net(imgs)
      loss = criterion(out, labels)
      loss.backward()
      optimizer.step()
        
      train_loss += loss.detach().cpu().item()
      pred = torch.argmax(out, 1) == labels
      train_acc += pred.detach().cpu().sum()
        
    
      if (i+1) % 100 ==0:
        writer.add_scalar('training loss',
                            train_loss/100,
                            epoch*len(train_loader)+i)
        writer.close()
        with torch.no_grad():
          val_loss=0.0
          val_acc = 0.0
          for j, val_data in enumerate(val_loader):
            imgs,label = val_data
            imgs = imgs.to(device)
            label = label.to(device)
            val_out = net(imgs)
            v_loss = criterion(val_out, label)
            val_loss += v_loss.detach().cpu()
            val_pred = torch.argmax(val_out, 1) == label
            val_acc += val_pred.detach().cpu().sum()
                    
        print("epoch: {}/{}, step: {}/{}, train loss: {:.4f}, val loss: {:.4f}, train acc: {:.2f}, val acc: {:.2f}".format(
            epoch+1, num_epochs, i+1, total_batch, train_loss/100, val_loss/len(val_loader), train_acc/100/50, val_acc/len(val_loader.dataset)
             ))
            
        train_loss_list.append(train_loss/100)
        val_loss_list.append(val_loss/len(val_loader))
        train_acc_list.append(train_acc/100/50)
        val_acc_list.append(val_acc/len(val_loader.dataset))
        train_loss = 0.0
        train_acc = 0.0




plt.figure(figsize = (16, 9))
x_range = list(range(len(train_loss_list)))
plt.plot(x_range, train_loss_list, label="train")
plt.plot(x_range, val_loss_list, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")

plt.figure(figsize = (16, 9))
x_range = range(len(train_loss_list))
plt.plot(x_range, train_acc_list, label="train")
plt.plot(x_range, val_acc_list, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

with torch.no_grad():
    corr_num = 0
    total_num = 0
    for num, data in enumerate(val_loader):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        
        prediction = net(imgs)
        model_label = prediction.argmax(dim=1)
        
        corr  = label[label == model_label].size(0)
        corr_num += corr 
        total_num += label.size(0)
        
print('Accuracy:{:.2f}'.format(corr_num/total_num*100))

with torch.no_grad():
    for num, data in enumerate(test_loader):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        
        prediction = net(imgs)
        
        correct_prediction = torch.argmax(prediction, 1) == label
        
        accuracy = correct_prediction.float().mean()
        print('Accuracy:{:.2f} %'.format(100*accuracy.item()))