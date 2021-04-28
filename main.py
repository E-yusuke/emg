import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess as pre
import matplotlib.pyplot as plt
import numpy as np
import os
import model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
np.random.seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

name = ["nagata", "noda", "shiba", "kirikihira", "asae2"]
name_action = ["nagata_action", "noda_action", "shiba_action", "kirikihira_action", "asae_action"]
name_list = name + name_action
experiment = 'RMS'

Data = None
Labels = None
DATA_list = []
LABEL_list = []
path_out = '../Result/experiment1/'
for name_ in name_list:
    # デスクトップのパス
    path_in = './dataset/{}/'.format(str(name_))

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
    DATA_list.append(Data)
    LABEL_list.append(Labels)
DATA_list = np.array(DATA_list)
LABEL_list = np.array(LABEL_list)
DATA = np.concatenate([DATA_list[0], DATA_list[1], DATA_list[2], DATA_list[3], DATA_list[4],
                      DATA_list[5], DATA_list[6], DATA_list[7], DATA_list[8], DATA_list[9]])
LABEL = np.concatenate([LABEL_list[0], LABEL_list[1], LABEL_list[2], LABEL_list[3], LABEL_list[4],
                        LABEL_list[5], LABEL_list[6], LABEL_list[7], LABEL_list[8], LABEL_list[9]])
np.random.seed(777)
_index = np.arange(len(DATA))
np.random.shuffle(_index)
train_index, validation_index, test_index = np.split(_index, [int(.6*len(DATA)), int(.8*len(DATA))])
emg_data_person = torch.tensor(DATA, dtype=torch.float32)
emg_data_label = torch.tensor(LABEL, dtype=torch.int64)
emg_dataset = torch.utils.data.TensorDataset(emg_data_person, emg_data_label)

train_dataset = Subset(emg_dataset, train_index)
train_loader = DataLoader(train_dataset,
                          batch_size=20,
                          shuffle=True,
                          num_workers=2)

validation_dataset = Subset(emg_dataset, validation_index)
val_loader = DataLoader(validation_dataset,
                        batch_size=20,
                        shuffle=False,
                        num_workers=2)

test_dataset = Subset(emg_dataset, train_index)
test_loader = DataLoader(test_dataset,
                         batch_size=20,
                         shuffle=False,
                         num_workers=2)

input_num = len(train_dataset[0][0])

net = model.FFNNs(input_num).to(device)

writer = SummaryWriter(log_dir=path_out)
# 最適化手法
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss().to(device)

total_batch = len(train_loader)
num_epochs = 800

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(train_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().cpu().item()
        pred = torch.argmax(out, 1) == labels
        train_acc += pred.detach().cpu().sum()

        if (i+1) % total_batch == 0:
            writer.add_scalar('training loss',
                              train_loss/total_batch,
                              epoch*len(train_loader)+i)
            writer.close()
            with torch.no_grad():
                val_loss = 0.0
                val_acc = 0.0
                for j, val_data in enumerate(val_loader):
                    imgs, label = val_data
                    imgs = imgs.to(device)
                    label = label.to(device)
                    val_out = net(imgs)
                    v_loss = criterion(val_out, label)
                    val_loss += v_loss.detach().cpu()
                    val_pred = torch.argmax(val_out, 1) == label
                    val_acc += val_pred.detach().cpu().sum()

            print("epoch: {}/{}, step: {}/{}, train loss: {:.4f}, val loss: {:.4f}, train acc: {:.2f}, val acc: {:.2f}"
                  .format(epoch+1, num_epochs, i+1, total_batch, train_loss/total_batch, val_loss /
                          len(val_loader), train_acc/total_batch/50, val_acc/len(val_loader.dataset)
                          ))

            train_loss_list.append(train_loss/total_batch)
            val_loss_list.append(val_loss/len(val_loader))
            train_acc_list.append(train_acc/total_batch/50)
            val_acc_list.append(val_acc/len(val_loader.dataset))
            train_loss = 0.0
            train_acc = 0.0
# print(train_acc_list, train_loss_list)
plt.figure(figsize=(16, 9))
x_range = list(range(len(train_loss_list)))
plt.plot(x_range, train_loss_list, label="train")
plt.plot(x_range, val_loss_list, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")

plt.figure(figsize=(16, 9))
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

        corr = label[label == model_label].size(0)
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
