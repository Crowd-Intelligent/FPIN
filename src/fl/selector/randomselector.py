from .base import BaseSelector
import random


class RandomSelector(BaseSelector):
    def __init__(self, frac):
        super(RandomSelector, self).__init__(frac)

    def select(self):
        # return the List[int]
        return random.sample(range(self.N), self.K)  # return the index of selected clients

    def feedback(self):
        pass
