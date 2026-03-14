import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CraterDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):

        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = list(self.data.columns)[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        img_name = row.iloc[0]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        labels = row.iloc[1:].values.astype(float)
        label = torch.tensor(labels).argmax().long()

        if self.transform:
            image = self.transform(image)

        return image, label


class DataTransformation:

    def __init__(self, data_dir, image_size=224, batch_size=16):

        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size

    def get_dataloaders(self):

        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

        train_dataset = CraterDataset(
            os.path.join(self.data_dir,"train","_classes.csv"),
            os.path.join(self.data_dir,"train"),
            train_transform
        )

        val_dataset = CraterDataset(
            os.path.join(self.data_dir,"valid","_classes.csv"),
            os.path.join(self.data_dir,"valid"),
            val_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        num_classes = len(train_dataset.classes)

        return train_loader, val_loader, num_classes
