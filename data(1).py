import os
from typing import Optional, Union

import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from utils import *

class MyDataSet(Dataset):
    def __init__(self,
                 seeds: np.ndarray,
                 size: int = 16,
                 mode: str = 'train',
                 ):
        self.datas = seeds
        self.size = size
        self.mode = mode

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        key, idx = self.datas[item]
        return key.unsqueeze(0), idx

class MyDataModule(pl.LightningDataModule):
    def __init__(self,
                 pic_size: int = 16,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 **kwargs,
                 ):
        super().__init__()
        self.val_data = None
        self.train_data = None
        self.predict_data = None

        dataset = torch.tensor([])
        for seed in np.random.randint(0, 2**16, (212)):
            torch.manual_seed(seed)
            dataset = torch.cat([dataset, torch.randint(0, 16, (pic_size, pic_size), dtype=torch.float32).unsqueeze(0)], dim=0)

        keys = dataset.clone()
        keys = keys.view(keys.shape[0], -1)
        paras = 1/keys[:, :64].mean(1)/keys[:, 64:128].mean(1), 1/keys[:, 128:192].mean(1)/keys[:, 192:].mean(1)
        idxs = create_pwlcm_paras_metrix(*paras, 8*512*512)        

        dataset = [[i/16, j] for i, j in zip(dataset, idxs)]
        self.trainset = dataset[:160]
        self.valset = dataset[160:192]

        self.predictset = dataset[192:]

        self.size = pic_size
        self.batch_size = batch_size
        self.num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, num_workers]) if num_workers else 0

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_data = MyDataSet(seeds=self.trainset,   
                                        size=self.size,
                                        mode='train')
            self.val_data = MyDataSet(seeds=self.valset, 
                                      size=self.size,
                                      mode='validation')
        if stage == 'predict':
            self.predict_data = MyDataSet(seeds=self.predictset, 
                                          size=self.size,
                                          mode='predict')

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers, 
                          persistent_workers=True,)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers, 
                          persistent_workers=True,)

    def predict_dataloader(self):
        return DataLoader(self.predict_data,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers, 
                          persistent_workers=True,)