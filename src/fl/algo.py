import numpy as np

from .aggregator import average_weights, Yogi
from .client import Client
from typing import List
from .selector import BaseSelector
import copy



distribute_first = True
def fl_train(global_model, global_trainer, client_pools, selector: BaseSelector, test_dataset, fl_config, logger=None):
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

        if logger is not None:
            try:
                graph = selector.get_graph()
            except:
                graph = None
            logger.update(loss, acc, winner, graph)


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
    client_pools: List[Client] = []
    for i, user_idx in enumerate(user_idxes):
        client = Client(i, train_dataset, user_idx, global_trainer)
        # ensure every client has the model
        client.receive_model(copy.deepcopy(global_model))
        client_pools.append(client)
    # initiate the selector
    selector.initiate(client_pools)
    fl_train(global_model, global_trainer, client_pools, selector, test_dataset, fl_config, logger=logger)


# ------------------------------------------------Fed Prox------------------------------------------------
def federated_prox(global_model, global_trainer, user_idxes, selector: BaseSelector, train_dataset, test_dataset,
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
    # create the client pools with prox
    client_pools: List[Client] = []
    for i, user_idx in enumerate(user_idxes):
        client = Client(i, train_dataset, user_idx, global_trainer, mu=fl_config['prox']['mu'])
        # ensure every client has the model
        client.receive_model(copy.deepcopy(global_model))
        client_pools.append(client)

    # initiate the selector
    selector.initiate(client_pools)

    fl_train(global_model, global_trainer, client_pools, selector, test_dataset, fl_config, logger=logger)


# ------------------------------------------------Fed BN------------------------------------------------
# used for fedbn to distribute model
def prepare_model_bn(global_model, local_model):
    copy_model = copy.deepcopy(global_model)
    # the bn layer in the model is the same as the local model
    for (name, layer) in copy_model.named_modules():
        if 'bn' in name:
            # set the bn layer of local_model to the copy_model
            copy_model._modules[name] = local_model._modules[name]
    return copy_model


def fl_train_bn(global_model, global_trainer, client_pools, selector: BaseSelector, test_dataset, fl_config, logger=None):
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
            copy_model = prepare_model_bn(global_model, client.model)
            client.receive_model(copy_model)
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

        if logger is not None:
            try:
                graph = selector.get_graph()
            except:
                graph = None
            logger.update(loss, acc, winner, graph)


def federated_bn(global_model, global_trainer, user_idxes, selector: BaseSelector, train_dataset, test_dataset,
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
    client_pools: List[Client] = []
    for i, user_idx in enumerate(user_idxes):
        client = Client(i, train_dataset, user_idx, global_trainer)
        # ensure every client has the model
        client.receive_model(copy.deepcopy(global_model))
        client_pools.append(client)
    # initiate the selector
    selector.initiate(client_pools)

    fl_train_bn(global_model, global_trainer, client_pools, selector, test_dataset, fl_config, logger=logger)


# ------------------------------------------------Fed Yogi------------------------------------------------
def fl_train_yogi(global_model, global_trainer, client_pools, selector: BaseSelector, test_dataset, fl_config, logger=None):
    # create the yogi optimizer
    yogi = Yogi(**fl_config['yogi'])

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
            copy_model = copy.deepcopy(global_model)
            client.receive_model(copy_model)
            model = client.train()
            local_models.append(model)

        # using the yogi to update the global model
        global_model = yogi.update(global_model, local_models)

        # evaluate the global model
        acc, loss = global_trainer.eval(
            dataset=test_dataset,
            model=global_model,
        )
        print(f'Round {round} end')

        if logger is not None:
            try:
                graph = selector.get_graph()
            except:
                graph = None
            logger.update(loss, acc, winner, graph)


def federated_yogi(global_model, global_trainer, user_idxes, selector: BaseSelector, train_dataset, test_dataset,
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
    client_pools: List[Client] = []
    for i, user_idx in enumerate(user_idxes):
        client = Client(i, train_dataset, user_idx, global_trainer)
        # ensure every client has the model
        client.receive_model(copy.deepcopy(global_model))
        client_pools.append(client)
    # initiate the selector
    selector.initiate(client_pools)

    fl_train_yogi(global_model, global_trainer, client_pools, selector, test_dataset, fl_config, logger=logger)
