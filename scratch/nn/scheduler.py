"""
This decays the learning rate of each parameter group by gamma
every step_size epochs. Modeled after PyTorch
"""
class lr_scheduler():
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1) -> None:
        self.optim = optimizer # holds optimizer state
        self.step_size = step_size
        self.gamma = gamma # the decay
        self.last_epoch = last_epoch
        self.lr = [self.optim.alpha]

    def state_dict(self):
        return self.optim, self.step_size, self.gamma, self.last_epoch

    def step(self):
        self.last_epoch += 1
        if (self.last_epoch % self.step_size) - self.step_size == -1:
            self.optim.alpha *= self.gamma

    def curr_lr(self): # current learning rate for all param groups
        return self.lr[-1]

    def last_lr(self): # previous learning rate for all param groups
        if len(self.lr) == 1:
            raise Exception("A second learning rate has not yet been established.")
        else:
            return self.lr[-2]
