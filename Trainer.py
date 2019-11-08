import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np


class Trainer(object):
    def __init__(self, model):
        self.model = model
        # optimizer: Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        # loss function: BCE with Logits
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, trainset, num_epoches, device="cpu", batch_size=32 , num_workers=16):
        """
            Args:
                trainset: training dataset
                num_epoches: number of epoches to run training
                device: "cpu" or "cuda"
        """
        device = torch.device(device)
        model = self.model.to(device)
        criterion = self.criterion
        optimizer = self.optimizer

        # Todo: convert input data here in order to use Dataloader

        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        num_batches = len(train_loader)

        for e in range(num_epoches):
            loss_avg = 0.

            for data, labels in train_loader:
                # Todo: handle input data, this is just a place holder
                x1, x2 = data

                x1, x2 = x1.to(device), x2.to(device)
                labels = labels.to(device)
                predictions = model(x1, x2)

                loss = criterion(predictions.float, labels)
                loss = loss.to(device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_avg += loss.item()

        # Todo: change path
        torch.save(model.state_dict(), ".")
