class Experiment:
    def __init__(self, model, custom_env, optim_settings, run_settings):
        self.model = model
        self.custom_env = custom_env
        self.optim_settings = optim_settings
        self.run_settings = run_settings

    def train():
        # Instantiate optimizer
        optimizer = self.optim_settings.optimizer(
            params=self.model.parameters(), lr=self.optim_settings.learning_rate)
        for epoch in self.run_settings.num_epochs:
            pass


class Model:
    def parameters(self):
        raise NotImplementedError()

class CustomEnvironment:
    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class OptimizerSettings:
    def __init__(self, optimizer, learning_rate):
        """
        :param optimizer: A class from torch.optim (instantiated later)
        """
        # optimizer should be a class, will be instantiated later
        self.optimizer = optimizer
        self.learning_rate = learning_rate


class RunSettings:
    def __init__(self, num_epochs, batch_size):
        self.num_epochs = num_epochs
        self.batch_size = batch_size