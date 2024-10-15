from .base import BaseSelector


class OracleSelector(BaseSelector):
    def __init__(self, frac, circles: list):
        '''
        :param client_pools:
        :param frac:
        :param circles: the circle of selected clients index in each round
        '''
        super(OracleSelector, self).__init__(frac)

        self.circles = circles
        self.cnt = -1

    def select(self):
        self.cnt += 1
        return self.circles[self.cnt % len(self.circles)]

    def feedback(self):
        pass
