import torch;
from torch import nn;
import pytorch_lightning as pl;
from torchvision.models import resnet50;
import numpy as np;
import torchmetrics
from torchmetrics import Accuracy, F1Score

class HAM10000_Model(pl.LightningModule):
    def __init__(self):
        super().__init__();
        
        resnet = resnet50();
        pretrainNet = nn.Sequential(*list(resnet.children())[:0]);
        trainNet = nn.Sequential(*list(resnet.children())[0: -1]);
        
        #### ----- Transfer Learning ----- ###
        pretrainNet.eval();
        for param in pretrainNet.parameters():
            param.requires_grad = False;
        
        self.net = nn.Sequential(
            *list(pretrainNet.children()),
            *list(trainNet.children()),
            
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000, 7),
            nn.Softmax(dim = 1)
        );

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task = "multiclass", num_classes=7)
        self.accuracy = Accuracy(task = "multiclass", num_classes=7)
        self.f1_scores = F1Score(task = "multiclass", num_classes=7)
        self.val_outputs = torch.tensor([])
        self.val_targets = torch.tensor([])
        self.max_acc = 0
        self.sum_acc_ = 0
    
    def forward(self, x):
        return self.net(x);
    def training_step(self, batch, batch_idx):
        x, y = batch;
        scores = self.forward(x);
        train_accuracy = self.train_accuracy(scores, y)
        loss = self.loss_fn(scores, y);
        self.log_dict({'train_loss': loss, 'train_acc': train_accuracy}, on_epoch = True, prog_bar = True);
        return loss;
    def on_validation_epoch_end(self):
        if self.sum_acc_ > self.max_acc:
            torch.save(self.state_dict(), 'resnet50_SVD.pth');
            self.max_acc = self.sum_acc_
        self.sum_acc_ = 0
    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x);
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_scores(scores, y)
        self.val_outputs = torch.concatenate([self.val_outputs, scores.to("cpu")], dim = 0)
        self.val_targets = torch.concatenate([self.val_targets, y.to("cpu")], dim = 0) 
        loss = self.loss_fn(scores, y);
        self.log('val_loss', loss);
        self.log('val_accucary', accuracy)
        self.log('f1_scores', f1_score)
        self.sum_acc_ += accuracy
        return loss;

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-5, weight_decay = 1e-6);