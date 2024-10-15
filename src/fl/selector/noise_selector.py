import math

from .base import BaseSelector
import numpy as np
from .base import oort_utility
from ..aggregator import average_weights
from typing import List
import torch

class NoiseSelector(BaseSelector):

    def __init__(self, frac):
        super().__init__(frac)
        self.bids = []
        self.payment_list = []
        self.utility = []
        self.round_number = 0

    def feedback(self):
        # 1. get the bid list
        bids = []
        for client in self.client_pools:
            bids.append(client.send_bid())
        self.bids = bids
        #print("bid",self.bids)
        # 初始化 self.number_of_selected 长度为 len(self.bids) 且每个元素值为 1
        if not hasattr(self, 'number_of_selected'):  # 确保只初始化一次
            self.number_of_selected = [1] * len(self.bids)
        if not hasattr(self, 'e'):  # 确保只初始化一次
            self.e = [0] * len(self.bids)
        # 2. get the oort loss list
        self.oort_loss_list = []
        for client in self.client_pools:
            self.oort_loss_list.append(oort_utility(client))
        # 3. normalize the oort list
        self.oort_loss_list = np.array(self.oort_loss_list)
        self.oort_loss_list = (self.oort_loss_list-np.min(self.oort_loss_list))/(np.max(self.oort_loss_list)-np.min(self.oort_loss_list))*0.9+0.1
        #print("oort_loss_list", self.oort_loss_list)

    def select(self) -> List[int]:
        self.round_number += 1
        for i in range(len(self.e)):
            self.e[i] = 0.5 * math.sqrt((self.K+1)*math.log(self.round_number)/self.number_of_selected[i])
        self.rho = (self.oort_loss_list + self.e) / self.qtc
        self.utility = self.rho / self.bids
        self.winner = np.argsort(self.utility)[-self.K:].tolist()[::-1]
        for i in range(len(self.winner)):
            self.number_of_selected[self.winner[i]] = self.number_of_selected[self.winner[i]] + 1
        self.payment_list = self.get_payment()
        #print("self.payment_list",self.payment_list)
        return self.winner


    def payment_AUCB(self,x,y):
        loss_1 = self.rho[self.cp_sort_index[x]]
        cp_2 = self.utility[self.cp_sort_index[y]]
        result = round(loss_1 / cp_2, 2)
        return result
        pass

    def get_delta(self, x):
        delta = 0.7 / self.qtc[self.cp_sort_index[x]]
        return delta
    def payment_PDS(self, x):
        delta = self.get_delta(x)
        phi = 0
        bids_PDS = []
        for client in self.client_pools:
            bids_PDS.append(client.send_bid())
        utility = self.utility
        rho = self.rho
        for i in range(len(self.cp_sort_index)):
            if self.utility[self.cp_sort_index[x]] > self.utility[self.cp_sort_index[i]] and self.rho[self.cp_sort_index[x]] < self.rho[self.cp_sort_index[i]]:
                payment = self.payment_AUCB(x, i)
                aim_index = i
                break
            elif i == len(self.cp_sort_index)-1:
                payment_final = self.payment_list_non[self.cp_sort_index[x]]
                return payment_final
        while utility[self.cp_sort_index[aim_index]] < utility[self.cp_sort_index[x]]:
            for i in range(len(bids_PDS)):
                bids_PDS[i] += delta
            utility = rho / bids_PDS
            phi += 1
        payment_final = round(payment - (phi - 1) * (1 - rho[self.cp_sort_index[x]]/rho[self.cp_sort_index[aim_index]]) * delta, 2)
        if payment < payment_final:
            return -1
        else:
            return payment_final

    def get_payment(self):
        self.cp_sort_index = np.argsort(self.utility)[::-1]
        #print(self.cp_sort_index)
        self.payment_list_non = [0] * len(self.cp_sort_index)
        payment_list = [0]*len(self.cp_sort_index)
        for i in range(len(self.cp_sort_index)-1):
            self.payment_list_non[self.cp_sort_index[i]] = self.payment_AUCB(i,i+1)
        self.payment_list_non[self.cp_sort_index[-1]] = self.payment_AUCB(len(self.cp_sort_index)-1,0)
        cp_sort_index = self.cp_sort_index[:self.K]
        for i in range(len(cp_sort_index)):
            payment_list[self.cp_sort_index[i]] = self.payment_PDS(i)
        return payment_list

    def weight_flatten(self, model):
        return torch.cat([param.data.view(-1) for param in model.parameters()])

    def noise_detect(self):
        # TODO:只在第一轮的时候进行detect
        loss_list = []
        model_v = []
        model_list = []
        for client in self.client_pools:
            loss = np.mean(client.get_loss())
            loss_list.append(loss)
            model_v.append(self.weight_flatten(client.model))
            model_list.append(client.model)
        # 获得了长度N的loss的列表
        global_model = self.client_pools[0].model
        global_model = average_weights(global_model,model_list)
        global_model_weight = self.weight_flatten(global_model)

        # 开始计算模型差距
        loss_list = np.array(loss_list)
        model_dis = []
        for weight in model_v:
            model_dis.append(torch.norm(weight-global_model_weight,p=2))
        model_dis = np.array(model_dis)
        #print("model_dis", model_dis)
        #print("loss_list", loss_list)
        qtc = loss_list*model_dis
        qtc_low = qtc
        expected_Qt = np.mean(qtc)
        std_Qt = np.std(qtc)
        for i in range(len(qtc)):
            if qtc_low[i] - expected_Qt > 0.6 * std_Qt:
                qtc_low[i] = 10 * qtc_low[i]
        self.qtc = qtc_low
    '''def check_inequality(self):
        # 计算期望值E(Qt)和标准差σ(Qt)
        expected_Qt = np.mean(self.qtc)
        std_Qt = np.std(self.qtc)
        for i in range(len(self.qtc)):
            if self.qtc[i] - expected_Qt < 0.7 * std_Qt:
                self.qtc[i] = 10
        return self.qtc'''