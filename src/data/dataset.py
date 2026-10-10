import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class PeopleCountingDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # We load the exact relative path saved during the split step
        image_path = self.data.iloc[index]["image"]
        count = self.data.iloc[index]["count"]
        
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Regression targets must be float32, not integers
        count = torch.tensor(float(count), dtype=torch.float32)
        
        return image, count