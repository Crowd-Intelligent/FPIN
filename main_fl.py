import copy
from typing import List
import numpy as np

from FL.src.fl.aggregator import average_weights
from FL.src.fl.selector import BaseSelector
from FL.src.fl.selector.randomauctionselector import RandomAuctionSelector
from src.utils import parse_yaml
import src.task as TK
import src.fl.sampler as FLS
from src.fl.sampler import LabelSkewSampler
import src.fl.selector as SLT
import src.fl.algo as FLAO
from src.logger import LoggerFL
from src.fl.client import Client
from src.fl.client import AuctionClient

import argparse

# get the config file path
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='/scientific research/work/FL/config.yaml', help='the path of config file')
args = parser.parse_args()

# load the config file
config = parse_yaml(args.config)
task_config = config['Task']

# get the dataset
train_dataset, test_dataset = getattr(TK.baseloader, task_config['Dataset']['name'])(**task_config['Dataset']['args'])

# get the model
global_model = getattr(TK.models, task_config['Model']['name'])(**task_config['Model']['args'])
global_model.to(task_config['Trainer']['args']['device'])

# get the trainer
global_trainer = getattr(TK.trainer, task_config['Trainer']['name'])(**task_config['Trainer']['args'])

# get the sampler
sampler_config = config['Sampler']
# sampler = LabelSkewSampler(train_dataset, **sampler_config['args'])
sampler = getattr(FLS, sampler_config['name'])(train_dataset, **sampler_config['args'])

user_idxes = sampler.sample()

# get the selector
selector = getattr(SLT, config['Selector']['name'])(**config['Selector']['args'])
# create noisy level list
alpha = config['Noise']['alpha']
sigma = config['Noise']['sigma']
mu = config['Noise']['mu']
lower = config['Noise']['lower']
upper = config['Noise']['upper']
if alpha == 0.0:
    noisy_levels = [0 for _ in range(len(user_idxes))]
else:
    #noisy_levels = np.random.uniform(0, alpha, len(user_idxes))
    #noisy_levels = np.random.binomial(1, alpha, len(user_idxes))
    num_clients = len(user_idxes)  # 客户端数量
    noise_levels = np.random.normal(loc=mu, scale=sigma, size=num_clients)

    # 手动截断，将超出范围的值裁剪到 [lower, upper] 之间
    noisy_levels = np.clip(noise_levels, lower, upper)

print("noisy_levels", noisy_levels)

# crete the logger
logger = LoggerFL(config['save_path'])

#print('the noisy levels mean:', np.mean(noisy_levels))

#print('the noisy levels mean:', np.mean(np.random.uniform(0, 1, 1000)))

# create the client pool
client_pool:List[Client] = []


distribute_first = True
def fl_train(global_model, global_trainer, client_pools, selector: BaseSelector, test_dataset, fl_config, logger=None):
    # TODO: 如果selector是我们自己设计的selector，需要调用noise_detect
    try:
        for client in client_pools:
            client.receive_model(copy.deepcopy(global_model))
            client.train()
        selector.noise_detect()
    except:
        print('pass the noise detection phrase')
        pass

    rounds = fl_config['rounds']
    for round in range(rounds):
        print(f'Round {round} start')
        # store the local models
        local_models = []

        if distribute_first:
            for idx in range(len(client_pools)):
                client = client_pools[idx]
                client.receive_model(copy.deepcopy(global_model))

        # get the feedback, such as utility, from the clients
        selector.feedback()

        # select the clients
        winner = selector.select()
        print(f'Winner: {winner}')

        for idx in winner:
            client = client_pools[idx]
            client.receive_model(copy.deepcopy(global_model))
            model = client.train()
            local_models.append(model)

        # aggregate the local models
        global_model = average_weights(global_model, local_models)

        # evaluate the global model
        acc, loss = global_trainer.eval(
            dataset=test_dataset,
            model=global_model,
        )
        print(f'Round {round} end')

        payment_list, bids = selector.get_stat_data()
        if logger is not None:
            pass
            logger.update(loss, acc, bids, payment_list)


# ------------------------------------------------Fed Avg------------------------------------------------
def federated_average(global_model, global_trainer, user_idxes, selector: BaseSelector, train_dataset, test_dataset,
                      fl_config, logger=None):
    '''
    :param global_model:
    :param global_trainer:
    :param user_idxes:
    :param selector:
    :param train_dataset:
    :param test_dataset:
    :param fl_config:
    :return:
    '''
    # create the client pools
    client_pools: List[AuctionClient] = []
    for i, (user_idx, noisy_level) in enumerate(zip(user_idxes, noisy_levels)):
        client = AuctionClient(i, train_dataset, user_idx, global_trainer, noisy_rate=noisy_level)
        # ensure every client has the model
        client.receive_model(copy.deepcopy(global_model))
        client_pools.append(client)
    # initiate the selector
    selector.initiate(client_pools)

    fl_train(global_model, global_trainer, client_pools, selector, test_dataset, fl_config, logger=logger)

federated_average(global_model, global_trainer, user_idxes, selector, train_dataset, test_dataset, config['FL'],logger=logger)
logger.close()

