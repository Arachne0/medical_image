import os 
import random
from re import I 
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import warnings 

warnings.filterwarnings("ignore")

class ImageDataset(Dataset):
    
    def __init__(self):

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def get_norm(self , key , values , transform = True):

        self.key = key
        self.images = values[0]
        self.labels = values[1]
        self.transform = transform
        
        self.images_ = []

        if self.key != "train":
            for img in self.images:
                self.images_.append(self.test_transform(img))
        else: # key == "train"
            for img in self.images:
                self.images_.append(self.train_transform(img))
    
        self.images = torch.stack(self.images_)        
        self.labels = torch.tensor(self.labels , dtype = torch.long)
        
        return self.images , self.labels

class collect_images:
    def __init__(self , data_root , label_root):
        self.image_paths = []
        self.images = []
        self.labels = []
        
        image_dir = os.path.join(data_root)
        image_path_list = os.listdir(image_dir)
        self.label_csv =  pd.read_csv(label_root)

        image_path_list = [f for f in image_path_list if f.lower().endswith((".png"))]        

        for fname in image_path_list:
            full_path = os.path.join(data_root , fname)
            img = Image.open(full_path).convert("RGB")
            self.images.append(img)
            base_name = os.path.basename(full_path)
            self.labels.append(self.label_csv["label"][self.label_csv["Image Index"] == base_name].item())
   
    def get_data(self):
        return self.images , self.labels
