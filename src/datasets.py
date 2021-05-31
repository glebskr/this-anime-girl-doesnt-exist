import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

class LoadDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        self.attr_list = pd.read_csv(csv_file)
        self.attr_list = self.attr_list.astype(int)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.attr_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.attr_list.iloc[idx, 0])+".png")
        image = Image.open(img_name).convert('RGB')
        attrs = self.attr_list.iloc[idx, 1:].values
        attrs = attrs.astype(int)

        hair = torch.FloatTensor(attrs[0:12])
        eye = torch.FloatTensor(attrs[12:])


        if self.transform:
            image = self.transform(image)

        return image, hair, eye

transform_anime = transforms.Compose([
    transforms.ToTensor(),
])

path_data = '../dataset/data/'
A_train_dataset = LoadDataset('./create_data/features.csv', path_data, transform_anime)
train_loader = DataLoader(A_train_dataset, batch_size=64, num_workers=4, shuffle=True, drop_last=True)
