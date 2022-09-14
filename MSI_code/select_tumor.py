
from timm.models.swin_transformer import SwinTransformer
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets,transforms
import os
from PIL import Image

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

swin_tiny_cfg = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
swin_tiny = SwinTransformer(**swin_tiny_cfg)

class Dataset(Dataset):
    def __init__(self, imgs_path, my_transforms):
        self.my_transforms = my_transforms
        self.imgs_path = imgs_path
        self.len = len(self.imgs_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        return my_transforms(img), self.imgs_path[index]

    def __len__(self):
        return self.len

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = SwinTransformer(**swin_tiny_cfg)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # [batch_size, 768]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)# [batch_size, 2]
        return x

def pic_list(path_dir):
    test_patient_list = os.listdir(path_dir)
    all_path = []
    for i in test_patient_list:
        datalist = os.listdir(path_dir + "/" + i)
        for j in datalist:
            all_path.append(path_dir+"/"+str(i)+"/"+str(j))
    return all_path


def test(save_path):
    model.eval()
    with torch.no_grad():
        for data, name in test_loader:
            y_p = []
            name_list = []
            data = data.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            print(pred)
            for i in range(len(pred)):
                y_p.append(pred[i])
                name_list.append(name[i])
            for j in range(len(name_list)):
                save_path=save_path+"/"+name_list[j][(name_list[j].rindex("/")+1):(name_list[j].rindex("/")+13)]
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if y_p[j] == 8:
                    shutil.copy(name_list[j], save_path + "/" + name_list[j][(name_list[j].rindex("/")+1):])
    print("done")

def get_args():
    parser = argparse.ArgumentParser(
        description='The script to select tumor')
    parser.add_argument(
        '-i', '--input_dir', default="/data/gbw/224_orginal_pic",
        help='Input_dir Path')
    parser.add_argument('-o', '--output_dir', default="/data/gbw/224_tumor_pic",
                        help='Output_dir Path')
    parser.add_argument('-mp', '--model_path', default="/data/gbw/TCGA_CRC/model.pth",
                        help='Path to directory in which model is saved.')

    return parser.parse_args()

if __name__ == ('__main__'):
    args = get_args()
    model = Net()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    my_transforms  = torchvision.transforms.Compose(
        [
         torchvision.transforms.Resize((224, 224)),
         torchvision.transforms.ToTensor(),
         transforms.Normalize(mean=[0.705909, 0.5331064, 0.7406105], std=[0.119056195, 0.16060014, 0.12790173])])

    test_pic_list = pic_list(args.input_dir)
    test_dataset = Dataset(test_pic_list, my_transforms)
    test_loader = DataLoader(test_dataset, 400, shuffle=True)
    test(args.output_dir)


