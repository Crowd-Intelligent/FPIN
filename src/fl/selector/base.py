from abc import ABC, abstractmethod
from typing import List
import math


class BaseSelector(ABC):
    def __init__(self, frac):
        '''
        :param client_pools: the list of clients
        :param frac: the fraction of clients selected in each round
        '''
        self.clint_selected_cnt = None
        self.winner = None
        self.N = None
        self.K = None
        self.client_pools = None
        self.frac = frac

    def initiate(self, client_pools):
        self.client_pools = client_pools
        self.K = int(self.frac * len(client_pools))
        self.N = len(client_pools)
        self.winner = [i for i in range(self.K)]
        self.clint_selected_cnt = [0] * self.N

    @abstractmethod
    def feedback(self):
        pass

    @abstractmethod
    def select(self) -> List[int]:
        pass

    # TODO: 继承baseselector的selector，都有select方法，在select方法里面，保存好self.winenr ;self.payment_list;self.bid
    def get_stat_data(self):
        bids_len = len(self.bids)
        # 初始化结果列表，所有值初始为0
        result = [0] * bids_len
        if 0 in self.payment_list:
            print("bid", self.bids)
            print("payment_list", self.payment_list)
            return self.payment_list, self.bids
        else:
            # 遍历 self.winner 和 self.payment_list，进行赋值
            for i, winner_index in enumerate(self.winner):
                result[winner_index] = self.payment_list[i]
            #print("bid", self.bids)
            print("payment_list", result)
            return result, self.bids
        pass
    def noise_detect(self):
        pass

def oort_utility(client):
    loss_list = client.get_loss()
    # utility = \sqrt{\sum_{i=1}^{size} loss_i^2 /size}*size, loss_list is a tensor which shape is [size]
    utility = math.sqrt(sum([loss ** 2 for loss in loss_list]) / len(loss_list)) * len(loss_list)

    return utility
