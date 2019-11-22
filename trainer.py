import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from model.model import Model

from pathlib import Path
import numpy as np
import os
import psutil
import time

class NewDataset(Dataset):
    def __init__(self):
        self.CSV_PATH = Path("data/csv")
        self.CPD_PATH = Path("data/compounds")
        self.pairs = os.listdir(self.CSV_PATH)
        self.proteins = {}
        self.maxcpd = 1584
        self.maxprt = 953
        with open("data/meta/target_atom2idx.txt") as f:
            atom2idx = {line.strip().split()[0]: int(line.strip().split()[1]) for line in f}
        with open("data/meta/targets_raw.txt") as f:
            for line in f:
                name, seq = line.strip().split()
                self.proteins[name] = [atom2idx[atom] for atom in seq.split(',')]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        with open(self.CSV_PATH / self.pairs[index]) as f:
            prt, cpd, label, _, _, _ = f.readline().strip().split()
            label = int(label)
            label_onehot = torch.zeros(2)
            label_onehot[label] = 1
        with open(self.CPD_PATH / "{0}.csv".format(cpd)) as f:
            cpd_seq = [int(num) for num in ",".join(line.strip() for line in f).split(',')]
        prt_seq = self.proteins[prt]
        cpd_seq = cpd_seq + [0] * (self.maxcpd - len(cpd_seq))
        prt_seq = prt_seq + [0] * (self.maxprt - len(prt_seq))
        return (Tensor(cpd_seq).long(), Tensor(prt_seq).long()), label_onehot

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

        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        num_batches = len(train_loader)

        for e in range(num_epoches):
            loss_avg = 0.

            for data, labels in train_loader:
                # Todo: handle input data, this is just a place holder
                x1, x2 = data

                x1, x2 = x1.to(device), x2.to(device)
                labels = labels.to(device)
                predictions = model((x1, x2))

                loss = criterion(predictions, labels)
                loss = loss.to(device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_avg += loss.item()

            print('epoch:', e, ' train_loss:', float(loss_avg / num_batches))

        # Todo: change path
        torch.save(model.state_dict(), ".")

if __name__ == "__main__":
    ds = NewDataset()
    cpd_length, prt_length = 1584, 953
    model = Model(cpd_length, prt_length, 795, 22)
    trainer = Trainer(model)
    ts = time.time()
    trainer.train(trainset=ds, num_epoches=1, batch_size=4, num_workers=4)
    print(time.time() - ts)