

import pandas as pd
import numpy as np
class LabelSkewSampler:
    def __init__(self,dataset,splits:list):
        '''
        :param dataset: the pytorch Dataset
        :param splits: the splits of the dataset, format is like [{'label':[1,2,3],'num':3},...,{'label':[4,5,6],'num':3}]
        '''
        self.dataset = dataset
        self.splits = splits

    def sample(self):
        # judge whether the attribute 'targets' is in the dataset
        if 'targets' not in dir(self.dataset):
            raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
        # get the targets
        targets = self.dataset.targets

        # the targets format is like [1,2,3,0,1,...], a tensor
        # create the dataframe, columns is idx and targets
        df = pd.DataFrame({'idx':range(len(targets)),'targets':targets})

        # sample the idxs
        user_idxes = []
        for split in self.splits:
            labels = split['labels']
            num = split['num']
            split_result = [[] for _ in range(num)]
            # get the idxs of the labels
            for label in labels:
                idxes = df[df['targets']==label]['idx'].values
                # split the idxes
                split_idxes = np.array_split(idxes,num)
                for i in range(num):
                    split_result[i].extend(split_idxes[i])
            user_idxes.extend(split_result)
        return user_idxes
