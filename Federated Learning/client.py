from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import flwr.cli
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient, start_numpy_client
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from model import HAM10000_Model
from dataset import HAM10000_Dataset, DataModule
import pytorch_lightning as pl

DEVICE = torch.device("cpu") 
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

data = DataModule(batch_size = 16, num_workers = 0, dataset = [HAM10000_Dataset("train"), HAM10000_Dataset("test")], have_splited = True)
Net = HAM10000_Model
trainer = pl.Trainer(accelerator = "gpu", devices = 1, min_epochs = 1, max_epochs = 1);

@dataclass
class FlowerClient(NumPyClient):
    net: pl.LightningModule
    data_module: pl.LightningDataModule
    trainer: pl.Trainer
    
    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.trainer.fit(self.net, self.data_module)
        return self.get_parameters(self.net), 10, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.trainer.validate(self.net, self.data_module)
        loss = 10
        accuracy = self.net.accuracy
        return float(loss), 10, {"accuracy": float(accuracy.compute())}

def client_fn() -> Client:
    net = Net().to(DEVICE)
    return FlowerClient(net, data, trainer).to_client()

start_numpy_client(server_address="127.0.0.1:8080", client = client_fn())
        