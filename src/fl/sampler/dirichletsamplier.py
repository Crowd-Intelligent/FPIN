import numpy as np
import pandas as pd


def create_dirichlet_distribution(num_clients, num_classes, beta):
    """
    使用Dirichlet分布生成每个客户端的类别比例。

    参数：
    num_clients (int): 客户端数量
    num_classes (int): 类别数量
    alpha (float): Dirichlet分布的浓度参数

    返回：
    proportions (list of np.array): 每个客户端的类别比例
    """
    # 使用Dirichlet分布生成比例
    proportions = np.random.dirichlet([beta] * num_classes, num_clients)
    return proportions


class DirichletSampler:
    def __init__(self, dataset, num_clients, num_classes, beta):
        """
        使用Dirichlet分布生成每个客户端的类别比例。

        参数：
        num_clients (int): 客户端数量
        num_classes (int): 类别数量
        alpha (float): Dirichlet分布的浓度参数
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.beta = beta

    def sample(self):
        """
        生成每个客户端的类别比例。

        返回：
        proportions (list of np.array): 每个客户端的类别比例
        """
        # judge whether the attribute 'targets' is in the dataset
        if 'targets' not in dir(self.dataset):
            raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
        # get the targets
        targets = np.array(self.dataset.targets)

        # the targets format is like [1,2,3,0,1,...], a tensor
        # create the dataframe, columns is idx and targets
        df = pd.DataFrame({'idx': range(len(targets)), 'targets': targets})

        # 使用Dirichlet分布生成比例
        proportions = create_dirichlet_distribution(self.num_clients, self.num_classes, self.beta)

        user_idxes = [[] for _ in range(self.num_clients)]

        for c in range(self.num_classes):
            class_indices = np.where(targets == c)[0]
            np.random.shuffle(class_indices)

            proportions_c = proportions[:, c]
            proportions_c = (proportions_c / proportions_c.sum()) * len(class_indices)
            proportions_c = proportions_c.astype(int)
            proportions_c[-1] = len(class_indices) - proportions_c[:-1].sum()

            split_class_indices = np.split(class_indices, np.cumsum(proportions_c)[:-1])

            for client_idx, indices in enumerate(split_class_indices):
                user_idxes[client_idx].extend(indices)

        return user_idxes
