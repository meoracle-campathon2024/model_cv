from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset;
from torchvision.transforms.v2 import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, Resize, Normalize, ColorJitter, GaussianBlur
import pytorch_lightning as pl 
import numpy as np

DATA_LABELS = "../Data/HAM10000/archive/HAM10000_metadata.csv"
DATA_LABELS_TEST = "../Data/ham10000_test.csv"
DATA_LABELS_TRAIN = "../Data/ham10000_train.csv"
DATA_LABELS_TRAIN_BALANCE = "../Data/ham10000_train_balance.csv"
DATA_IMAGES = "../Data/HAM10000/archive/HAM10000_images/"
DATA_IMAGES_SVD = "../Data/HAM10000_SVD/"

class HAM10000_Dataset(Dataset):
    def __init__(self, train_test = None):
        self.label_to_number_label = {'df': 0, 'vasc': 1, 'akiec': 2, 'bcc': 3, 'bkl': 4, 'mel': 5, 'nv': 6}
        self.number_label_to_label = ['df', 'vasc', 'akiec', 'bcc', 'bkl', 'mel', 'nv']
        if train_test == "train":
            self.ham10000 = pd.read_csv(DATA_LABELS_TRAIN)
        elif train_test == "test":
            self.ham10000 = pd.read_csv(DATA_LABELS_TEST)
        else:
            self.ham10000 = pd.read_csv(DATA_LABELS)
        self.transform = Compose([
            RandomAffine(45, (0.1, 0.1), (0.75, 1.25)),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            Resize((299, 299)),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def __len__(self):
        return len(self.ham10000);
    def __getitem__(self, index):
        image = Image.open(f"{DATA_IMAGES_SVD}{self.ham10000.iloc[index]['image_id']}.jpg");
        image = self.transform(image)
        label = self.ham10000.iloc[index]["dx"]
        number_label = self.label_to_number_label[label]
        return image, number_label

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, dataset, have_splited = False):
        super().__init__();
        self.have_splited = have_splited
        self.dataset = dataset;
        self.batch_size = batch_size;
        self.num_workers = num_workers;
    def setup(self, stage):
        if self.have_splited:
            self.train_data, self.validate_data = self.dataset
        else:
            self.train_data, self.validate_data = random_split(self.dataset, [0.9, 0.1])
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True);
    def val_dataloader(self):
        return DataLoader(self.validate_data, batch_size = self.batch_size, num_workers = self.num_workers);
    