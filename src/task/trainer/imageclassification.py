import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ImageClassificationTrainer:
    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 loss_function: str,
                 optimizer: dict,
                 device
                 ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = getattr(F, loss_function)
        self.optimizer = optimizer
        self.device = device

    def update(self, dataset: Dataset, model: nn.Module, add_item=None):
        '''
        the training method of the task
        :param add_item: the loss item ,such as FedProx
        :param dataset:
        :param model:
        :return:
        '''
        # create the optimizer
        optimizer = getattr(optim, self.optimizer['name'])(model.parameters(), **self.optimizer['args'])
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # start the training procedure
        model.train()
        # set the model to device
        model.to(self.device)

        for epoch in range(self.epochs):
            acc_list, loss_list = [], []
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.loss_function(output, target)
                # it's designed for FedProx
                if add_item is not None:
                    loss += add_item(model)
                loss.backward()
                optimizer.step()

                # calculate the accuracy and loss
                acc_list.append((output.argmax(1) == target).sum().item())
                loss_list.append(loss.item())
            # print the log
            print(f'Epoch {epoch} Accuracy: {sum(acc_list) / len(dataset)} Loss: {sum(loss_list) / len(train_loader)}')

        # return the updated model
        return model

    def eval(self, dataset: Dataset, model: nn.Module):
        '''
        the eval method of the task
        :param dataset:
        :param model:
        :param device:
        :return:
        '''
        model.eval()
        model.to(self.device)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        acc_list, loss_list = [], []
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = self.loss_function(output, target)

            acc_list.append((output.argmax(1) == target).sum().item())
            loss_list.append(loss.item())
        print(f'Accuracy: {sum(acc_list) / len(dataset)} Loss: {sum(loss_list) / len(test_loader)}')
        return sum(acc_list) / len(dataset), sum(loss_list) / len(test_loader)

    def get_loss(self, dataset: Dataset, model: nn.Module):
        '''
        It is an interference used in client side to get the loss of the model
        :param dataset:
        :param model:
        :param device:
        :return:
        '''
        model.eval()
        model.to(self.device)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        loss_list = []
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = self.loss_function(output, target, reduction='none')

            loss_list.extend(loss.tolist())

        return loss_list
