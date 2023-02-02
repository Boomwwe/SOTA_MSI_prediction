import torch.nn.functional as F
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
import os
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pickle
import pandas as pd
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Dataset(Dataset):
    def __init__(self, imgs_path, labels, my_transforms):
        self.my_transforms = my_transforms
        self.imgs_path = imgs_path
        self.labels = labels
        self.len = len(self.imgs_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        return my_transforms(img), self.labels[index], self.imgs_path[index]

    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = SwinTransformer(**swin_tiny_cfg)
        self.backbone.load_state_dict(
            torch.load("/data/gbw/TCGA_CRC/backbone.pth"))
        self.fc1 = nn.Linear(768, 2)


    def forward(self, x):
        x = self.backbone.forward_features(x)  # [batch_size, 768]
        x = self.fc1(x)
        return x



def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    y_p = []
    labels = []
    name_list = []
    for batch_idx, (data, target, name) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        soft = F.softmax(output, dim=1)[:, 1]
        soft_array = np.array(soft.detach().cpu())
        for i in range(len(soft_array)):
            y_p.append(soft_array[i])
            labels.append(np.array(target.detach().cpu())[i])
            name_list.append(name[i])
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
    train_acc.append(100. * correct / len(train_loader.dataset))
    train_auc.append(roc_auc_score(labels, y_p))
    # execl = pd.DataFrame(name_list, columns=["Pic_name"])
    # execl["y_p"] = y_p
    # execl["label"] = labels
    # execl.to_csv("/data/gbw/try_process_WSI/Kather_MSI_data_expand/" + "outcome" + str(epoch) + ".csv")
    # torch.save(model.state_dict(), '/data/gbw/try_process_WSI/MSI/model/model_SGD' + str(epoch) + '.pth')
    # torch.save(optimizer.state_dict(), '/data/gbw/try_process_WSI/MSI/model/optimizer_SGD' + str(epoch) + '.pth')
    print(100. * correct / len(train_loader.dataset))

def vali(epoch,cv,seed,save_path):
    model.eval()
    test_loss = 0
    correct = 0
    y_p = []
    labels = []
    name_list = []
    with torch.no_grad():
        for data, target, name in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            soft = F.softmax(output, dim=1)[:, 1]
            soft_array = np.array(soft.detach().cpu())
            for i in range(len(soft_array)):
                y_p.append(soft_array[i])
                labels.append(np.array(target.detach().cpu())[i])
                name_list.append(name[i])
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    execl = pd.DataFrame(name_list, columns=["Pic_name"])
    execl["y_p"] = y_p
    execl["label"] = labels
    execl.to_csv(save_path+"/outcome/"+str(TARGET)+"/" + str(seed) + "vali_outcome" + str(epoch) + str(cv) + ".csv")
    vali_auc.append(roc_auc_score(labels, y_p))
    vali_acc.append(100. * correct / len(test_loader.dataset))


def test(epoch,cv,seed,save_path):
    model.eval()
    test_loss = 0
    correct = 0
    y_p = []
    labels = []
    name_list = []
    with torch.no_grad():
        for data, target, name in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            soft = F.softmax(output, dim=1)[:, 1]
            soft_array = np.array(soft.detach().cpu())
            for i in range(len(soft_array)):
                y_p.append(soft_array[i])
                labels.append(np.array(target.detach().cpu())[i])
                name_list.append(name[i])
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    execl = pd.DataFrame(name_list, columns=["Pic_name"])
    execl["y_p"] = y_p
    execl["label"] = labels
    execl.to_csv(save_path+"/outcome/"+str(TARGET)+"/" + str(seed) + "test_outcome" + str(epoch) + str(cv) + ".csv")
    test_auc.append(roc_auc_score(labels, y_p))
    test_acc.append(100. * correct / len(test_loader.dataset))


def calculate_auc(set, epoch, cv, seed,save_path,cv_path):
    cv_splits  =  pickle.load(open(cv_path, 'rb'))
    sample_execl = pd.read_csv(
        save_path+"/outcome/"+str(TARGET)+"_M/" + str(seed) + str(set)+"_outcome" + str(epoch) + str(cv) + ".csv")
    Patient_name = []
    for i in sample_execl["Pic_name"]:
        Patient_name.append(i[(i.rindex("/")+1):(i.rindex(".")+1)][0:12])
    sample_execl["Patient_name"] = Patient_name
    Patient_list = cv_splits[cv]["test_set"]
    pred_list = []
    label_list = []
    for j in Patient_list:
        df = sample_execl[sample_execl["Patient_name"] == j]
        pred = np.mean(df["y_p"])
        label = np.mean(df["label"])
        pred_list.append(pred)
        label_list.append(label)
    execl = pd.DataFrame(Patient_list, columns=["Patient_name"])
    execl["y_p"] = pred_list
    execl["label"] = label_list
    test_auc = roc_auc_score(label_list, pred_list)
    return test_auc

def get_args():
    parser = argparse.ArgumentParser(
        description='The script to generate the tiles for Whole Slide Image (WSI).')
    parser.add_argument(
        '-cv', '--cv_dir', default="/data/gbw/TCGA/spilt/MSI/4fold_splits.pkl",
        help='cv_spilt Path')
    parser.add_argument('-pp', '--pic_path', default="/data/gbw/TCGA/data/Marcenko_tumor",
                        help='Pic_dir Path')
    parser.add_argument('-lp', '--label_path', default="/data/gbw/MCO/MSI_label.csv",
                        help='label_dir Path')
    parser.add_argument('-lr', '--initial_lr',type = float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('-w', '--weight',type = float, default=0.15)
    parser.add_argument('-t', '--target',default="MSI",help='The label you want to predict')
    parser.add_argument('-sp', '--save_path', default="'/data/gbw/TCGA_CRC",
                        help='Path to directory in which pre-train model and data will be saved.')

    return parser.parse_args()


if __name__ == ('__main__'):
    args = get_args()
    TARGET = args.target
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    random_seed = 12345
    setup_seed(random_seed)
    swin_tiny_cfg = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    swin_tiny = SwinTransformer(**swin_tiny_cfg)
    my_transforms = torchvision.transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 重置图片大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.69261899, 0.51298305, 0.7263637], std=[0.13099993, 0.18332116, 0.14441279])])

    test_transforms = torchvision.transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 重置图片大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.69261899, 0.51298305, 0.7263637], std=[0.13099993, 0.18332116, 0.14441279])])

    for cv in range(4):
        my_net = Net()
        model = my_net.to(device)
        train_pic_list, train_label_list, vali_pic_list, vali_label_list, test_pic_list, test_label_list = cv_pic_list(cv,args.pic_path,args.label_path,arg.cv_dir)
        weights = [args.weight, 1 - args.weight]
        batch_size_train = 128
        batch_size_test = 128
        log_interval = 10

        initial_lr = args.initial_lr
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float())
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        criterion = criterion.to(device)

        train_losses = []
        train_counter = []
        test_losses = []
        vali_losses = []
        vali_auc = []
        train_acc = []
        train_auc = []
        test_acc = []
        test_auc = []
        vali_patient_auc = []
        test_patient_auc = []

        train_dataset = Dataset(train_pic_list, train_label_list, my_transforms)
        vali_dataset = Dataset(vali_pic_list, vali_label_list, my_transforms)
        test_dataset = Dataset(test_pic_list, test_label_list, test_transforms)

        train_loader = DataLoader(train_dataset, 128, shuffle=True)
        vali_loader = DataLoader(vali_dataset, 128, shuffle=True)
        test_loader = DataLoader(test_dataset, 128, shuffle=True)

        for epoch in range(0, 20):

            train(epoch)

            #if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       args.save_path+"/model/"+str(TARGET)+"/"+str(random_seed)+'model' + str(epoch)+str(cv) +'.pth')

            vali(epoch, cv, random_seed,args.save_path)
            auc_vali = calculate_auc("vali", epoch, cv, random_seed,args.save_path,args.cv_dir)
            vali_patient_auc.append(auc_vali)

            test(epoch, cv, random_seed,args.save_path)
            auc = calculate_auc("test", epoch, cv, random_seed,args.save_path,args.cv_dir)
            test_patient_auc.append(auc)


            try:
                np.savetxt(args.save_path + "/outcome/" + str(TARGET) + "/" + str(random_seed) + "vali_losses" + str(cv) + ".txt", vali_losses)
                np.savetxt(args.save_path+"/outcome/"+str(TARGET)+"/"+str(random_seed)+"vali_auc_patient"+ str(cv) + ".txt", vali_patient_auc)
                np.savetxt(args.save_path+"/outcome/"+str(TARGET)+"/"+str(random_seed)+"test_auc_patient"+str(cv)+".txt", patient_auc)
            except:
                print("error!")
