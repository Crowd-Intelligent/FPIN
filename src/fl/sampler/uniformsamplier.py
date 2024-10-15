import random
import pandas as pd


class UniformSampler:
    def __init__(self, dataset, num_clients, groupby=False):
        '''
        :param dataset:
        :param num_clients:
        :param groupby: ensure every client has all labels
        '''
        self.dataset = dataset
        self.num_clients = num_clients
        self.groupby = groupby
        self.num_data = len(dataset)
        self.num_data_per_client = int(self.num_data / self.num_clients)

    def sample(self):
        '''
        sample the idxes
        :return:
        '''
        if not self.groupby:
            # just sample the idxes randomly
            idxes = list(range(self.num_data))
            random.shuffle(idxes)
            user_idxes = []
            for i in range(self.num_clients):
                user_idxes.append(idxes[i * self.num_data_per_client: (i + 1) * self.num_data_per_client])
        else:
            # ensure every client has all labels, so sample the idxes by labels
            if 'targets' not in dir(self.dataset):
                raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
            targets = self.dataset.targets

            df = pd.DataFrame({'idx': range(len(targets)), 'targets': targets})
            labels = df['targets'].unique()

            # group samples by labels
            user_idxes = []
            for i in range(self.num_clients):
                user_idxes.append([])
            for label in labels:
                idxes = df[df['targets'] == label]['idx'].values
                random.shuffle(idxes)
                for i in range(self.num_clients):
                    user_idxes[i].extend(idxes[i * self.num_data_per_client: (i + 1) * self.num_data_per_client])

        return user_idxes
