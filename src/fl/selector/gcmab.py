import math

from .base import BaseSelector
from .base import oort_utility
from itertools import combinations
import numpy as np
import torch


class GCMABSelector(BaseSelector):
    def __init__(self, frac, sigma=10, oracle=True, freeze_graph=False):
        super(GCMABSelector, self).__init__(frac)
        self.graph = None
        self.utilities = None
        self.client_selected_cnt = None

        self.round = 0
        self.freeze_graph = freeze_graph
        self.oracle = oracle
        self.sigma = sigma

    # rewrite the method
    def initiate(self, client_pools):
        self.client_pools = client_pools
        self.K = int(self.frac * len(client_pools))
        self.N = len(client_pools)
        self.winner = [i for i in range(self.K)]
        self.client_selected_cnt = [0] * self.N

        self.graph = np.ones([self.N, self.N]) * 0.5

        # set the diag to 1
        for i in range(self.N):
            self.graph[i][i] = 1

    def feedback(self):
        # the client submit the utility
        self.utilities = []
        for client in self.client_pools:
            self.utilities.append(oort_utility(client))
        pass

    def select(self):
        # update the round
        self.round = self.round + 1

        # if round>=2 and not freeze_graph, update the graph
        if self.round >= 2 and not self.freeze_graph:
            self.update_graph()
            # show the stat data, in order to adjust the sigma
            self.stat_graph()

        # if oracle, select the winner with the max utility
        u_max = -1
        winner_max = None
        if self.oracle:
            for winner in combinations(range(self.N), self.K):
                winner = list(winner)
                utility = self.combinatorial_utility(winner)
                if utility > u_max:
                    u_max = utility
                    winner_max = winner
        else:
            # TODO: implement the graph based selection
            pass
        # save the winner
        self.winner = winner_max

        # update the client_selected_cnt
        for index in self.winner:
            self.client_selected_cnt[index] += 1

        return self.winner

    def combinatorial_utility(self, winner: list):
        '''
        calculate the combinatorial utility of the winner
        :param winner:
        :return:
        '''
        util_min = min(self.utilities)
        util_vec = np.array(
            [self.utilities[index] + math.sqrt(
                (self.K + 1) * self.client_selected_cnt[index] / (math.log(self.round) + 1)
            ) * util_min
             for index in winner]
        )

        util_P = np.stack([util_vec] * self.K, axis=0) + np.stack([util_vec] * self.K, axis=1)
        # del the diag value
        util_P = util_P - np.diag(np.diag(util_P))

        util_P = np.sum(util_P * self.graph[winner][:, winner]) / 2

        return util_P

    def update_graph(self):
        # count the similarity matrix
        for i in range(len(self.winner)):
            for j in range(i + 1, len(self.winner)):
                if self.graph[self.winner[i]][self.winner[j]] != 0.5:continue # 试试效果如何吧
                sim = self.similarity(self.winner[i], self.winner[j])
                self.graph[self.winner[i]][self.winner[j]] = sim
                self.graph[self.winner[j]][self.winner[i]] = sim

    # the similarity between model x and y
    def similarity(self, x, y):
        params_x = self.weight_flatten(x)
        params_y = self.weight_flatten(y)

        sub = (params_x - params_y).view(-1)
        sub = torch.dot(sub, sub)
        return self.e(sub)

    # flatten the model weight
    def weight_flatten(self, x):
        model = self.client_pools[x].last_model
        return torch.cat([param.data.view(-1) for param in model.parameters()])

    # the similarity function
    def e(self, x):
        return 1 - math.exp(-x / self.sigma)

    def get_graph(self):
        return self.graph

    def stat_graph(self):
        data = []
        for i in range(self.N):
            for j in range(self.N):
                if self.graph[i][j] == 1 or self.graph[i][j] == 0.5:
                    continue
                data.append(self.graph[i][j])
        if not data:
            print('The value is so large')
        else:
            print('The min data is {}, the max data is {}'.format(min(data), max(data)))
