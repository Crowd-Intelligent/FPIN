import torch


class Yogi:
    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-8, eta=0.01):
        '''
        YOGI optimizer
        :param beta1:
        :param beta2:
        :param epsilon:
        :param eta:
        '''
        self.delta = None
        self.v = None

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

        print('The yogi parameters are beta1: {}, beta2: {}, epsilon: {}, eta: {}'.format(beta1, beta2, epsilon, eta))

    def update(self, global_model, local_models):
        diff_dict = {}
        global_model_dict = {}
        with torch.no_grad():
            # count the diff between local_models and global_models
            for k in global_model.state_dict().keys():
                diff_dict[k] = 0
                for model in local_models:
                    diff_dict[k] += model.state_dict()[k] - global_model.state_dict()[k]
                diff_dict[k] = diff_dict[k] / len(local_models)

            # for k in global_model.state_dict().keys():
            #     global_model_dict[k] = global_model.state_dict()[k] + diff_dict[k]
            # global_model.load_state_dict(global_model_dict)
            # return global_model

            # initiate the delta and v
            if self.delta is None:
                self.delta = {k: torch.zeros_like(global_model.state_dict()[k]) for k in global_model.state_dict().keys()}
                self.v = {k: torch.zeros_like(global_model.state_dict()[k]) for k in global_model.state_dict().keys()}

            # update the delta and v, using the yogi algorithm
            for k in global_model.state_dict().keys():
                self.delta[k] = self.beta1 * self.delta[k] + (1 - self.beta1) * diff_dict[k]
                self.v[k] = self.v[k] - (1 - self.beta2) * torch.sign(self.v[k] - self.delta[k].pow(2)) * self.delta[k].pow(2)

            # update the global model
            for k in global_model.state_dict().keys():
                global_model_dict[k] = global_model.state_dict()[k] + self.eta * self.delta[k] / (torch.sqrt(
                    self.v[k]) + self.epsilon)
            global_model.load_state_dict(global_model_dict)
        return global_model
