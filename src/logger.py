# import visualdl
import pandas as pd
import numpy as np
from typing import List
import os
import copy


class LoggerFL:
    def __init__(self, save_path):
        self.loss_list = []
        self.acc_list = []
        self.cost_list = []
        self.payment_list = []

        self.save_path = save_path

    def update(self, loss, acc, cost, payment):
        self.loss_list.append(loss)
        self.acc_list.append(acc)
        self.cost_list.append(cost)
        self.payment_list.append(payment)

    def close(self):
        # judge the save_path whether exist, if exists, judge whether the file is empty
        if os.path.exists(self.save_path):
            if os.path.getsize(self.save_path) > 0:
                # delete the file in the dir save_path
                for file in os.listdir(self.save_path):
                    os.remove(os.path.join(self.save_path, file))
        else:
            os.makedirs(self.save_path)

        # transform the loss and acc to the pandas dataframe
        data = {
            'epoch': list(range(1, len(self.loss_list) + 1)),
            'loss': self.loss_list,
            'acc': self.acc_list,
            'cost': self.cost_list,
            'payment': self.payment_list
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_path, 'info.csv'), index=False)
