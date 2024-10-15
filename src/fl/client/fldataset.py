from torch.utils.data import Dataset
import random


class FLDataset(Dataset):
    def __init__(self, dataset, idxs, noisy_rate=0.0):
        '''
        an abstract Dataset class wrapped around Pytorch Dataset class.
        :param dataset:
        :param idxs:
        '''
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.noisy_rate = noisy_rate

        if self.noisy_rate != 0.0:
            self.noise_label()

    def noise_label(self):
        label_set = set()
        for i in range(len(self.idxs)):
            label_set.add(self.dataset[self.idxs[i]][1])
        # get the unique label set
        label_list = list(label_set)

        # sample the index of noisy label
        noisy_idx = random.sample(self.idxs, int(self.noisy_rate * len(self.idxs)))

        # create a noisy label map
        noisy_label_map = {}
        for i in self.idxs:
            noisy_label_map[i] = self.dataset[i][1]
        for i in noisy_idx:
            label = self.dataset[i][1]
            noisy_label_map[i] = random.choice([l for l in label_list if l != label])
        self.noisy_label_map = noisy_label_map

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx = self.idxs[item]
        if self.noisy_rate != 0.0:
            return self.dataset[idx][0], self.noisy_label_map[idx]
        return self.dataset[idx]
