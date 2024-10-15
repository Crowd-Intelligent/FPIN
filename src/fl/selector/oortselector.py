from .base import BaseSelector
from .base import oort_utility
import numpy as np


class OortSelector(BaseSelector):
    def __init__(self, frac, explore_rate, explore_rate_min, decay_factor, cut_off_util):
        super(OortSelector, self).__init__(frac)
        self.winner = []
        self.round = 0

        self.explore_rate = explore_rate
        self.explore_rate_min = explore_rate_min
        self.decay_factor = decay_factor
        self.cut_off_util = cut_off_util
        self.bids = []
        self.payment_list = []


    def feedback(self):
        # 1. get the bid list
        bids = []
        for client in self.client_pools:
            bids.append(client.send_bid())
        self.bids = bids
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

        # count the oort utility of each winner
        self.score = {}
        self.utilities = []
        for idx in self.winner:
            u = oort_utility(self.client_pools[idx])
            self.score[idx] = u
            self.utilities.append([idx, u])

    def select(self):

        # update the round
        self.round = self.round + 1

        # update the explore_rate
        self.explore_rate = max(self.explore_rate_min, self.explore_rate * self.decay_factor)

        # sort the utilities ascending
        self.utilities.sort(key=lambda x: x[1], reverse=True)

        # -----------------exploit----------------------
        exploit_len = int(len(self.utilities) * (1 - self.explore_rate))
        self.exploit_len = exploit_len

        # to ensure oort can convergence to the final set
        if exploit_len == self.K:
            return self.winner

        # cut off the util
        cut_off_util = self.utilities[exploit_len][1] * self.cut_off_util

        picked_clients = []
        for idx, u in self.utilities:
            if u < cut_off_util:
                break
            picked_clients.append(idx)

        # sample with probability
        total_utility = max(1e-4, float(sum([self.score[index] for index in picked_clients])))
        picked_clients = list(
            np.random.choice(picked_clients, exploit_len,
                             p=[self.score[index] / total_utility for index in picked_clients], replace=False)
        )

        # -----------------explore----------------------
        exlore_len = self.K - exploit_len
        explore_clients = list(set(range(self.N)) - set(self.winner))
        picked_clients.extend(list(np.random.choice(explore_clients, exlore_len, replace=False)))

        self.winner = picked_clients
        # get the payment
        self.payment_list = self.get_payment()
        return self.winner

    def payment_function(self, x, y):
        for sublist in self.utilities:
            if sublist[0] == self.winner_with_utility[x]:
                loss_1 = sublist[1]
            if sublist[0] == self.winner_with_utility[y]:
                loss_2 = sublist[1]
        bid_2 = self.bids[self.winner_with_utility[y]]
        result = round((loss_1 * bid_2) / loss_2, 2)
        return result
        pass

    def get_payment(self):
        # self.winner = [have utility] + [no utility] =  [5,4,3] +[0,1]
        payment_list = [0] * len(self.bids)
        self.winner_with_utility = self.winner[:self.exploit_len] # [5,4,3]
        self.winner_no_utility = self.winner[self.exploit_len:] # [0,1]
        if len(self.winner_with_utility) == 1:
            for i in range(len(self.winner_no_utility)):
                payment_list[self.winner_no_utility[i]] = self.bids[self.winner_no_utility[i]]
            for i in range(len(self.winner_with_utility)):
                payment_list[self.winner_with_utility[i]] = self.bids[self.winner_with_utility[i]]
        elif len(self.winner_with_utility) == 0:
            for i in range(len(self.winner_no_utility)):
                payment_list[self.winner_no_utility[i]] = self.bids[self.winner_no_utility[i]]
        else:
            for i in range(len(self.winner_no_utility)):
                payment_list[self.winner_no_utility[i]] = self.bids[self.winner_no_utility[i]]
            for i in range(len(self.winner_with_utility)-1):
                payment_list[self.winner_with_utility[i]] = self.payment_function(i, i+1)
            payment_list[self.winner_with_utility[-1]] = self.bids[self.winner_with_utility[-1]]
        return payment_list

