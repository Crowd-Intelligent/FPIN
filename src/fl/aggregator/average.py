import torch
import copy


def average_weights(global_model, local_models):
    '''
    average the local models
    :param global_model:
    :param local_models:
    :return:
    '''
    global_weight = {}
    for k in global_model.state_dict().keys():
        global_weight[k] = torch.zeros_like(global_model.state_dict()[k])
    for k in global_weight.keys():
        for model in local_models:
            global_weight[k] += model.state_dict()[k]
        global_weight[k] = torch.div(global_weight[k], len(local_models))
    global_model.load_state_dict(global_weight)
    return global_model
