import numpy as np

from .base import BaseSelector
from .base import List

class MinSelector(BaseSelector):

    def __init__(self, frac):
        super(MinSelector, self).__init__(frac)
        self.bids = []
        self.payment_list = []
        pass

    def feedback(self):
        # 1. get the bid list
        bids = []
        for client in self.client_pools:
            bids.append(client.send_bid())
        self.bids = bids
        #print(self.bids)
        # 2. get the loss list
        self.loss_list = []
        for client in self.client_pools:
            loss = client.get_loss()
            loss = np.mean(loss)
            self.loss_list.append(loss)
        # print(self.loss_list)
        # 3. count the cost performance
        self.cost_performance_list = []
        for i in range(self.N):
            cost_performance = self.loss_list[i] / bids[i]
            self.cost_performance_list.append(cost_performance)
        pass

    def select(self) -> List[int]:
        self.winner = sorted(range(len(self.bids)), key=lambda i: self.bids[i])[:self.K]
        #print(self.min_indices)
        # get the payment
        self.payment_list = self.get_payment()
        print(self.payment_list)
        return self.winner
        pass
    def payment_function(self,x,y):
        # 0
        loss_1 = self.loss_list[self.cp_sort_index[x]]
        cp_2 = self.cost_performance_list[self.cp_sort_index[y]]
        result = round(loss_1 / cp_2, 2)
        return result
        pass

    def get_payment(self):
        self.cp_sort_index = np.argsort(self.bids)
        #print(self.cp_sort_index)
        payment_list = [0]*len(self.cp_sort_index)
        for i in range(len(self.cp_sort_index)-1):
            payment_list[self.cp_sort_index[i]] = self.payment_function(i,i+1)
        payment_list[self.cp_sort_index[-1]] = self.bids[-1]
        payment_list = payment_list[:self.K]
        return payment_list
