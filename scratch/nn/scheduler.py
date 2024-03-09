from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


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


class DecayLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False,
                 lr_init=5e-2, lr_final=5e-4, lr_delay_steps=0,
                 lr_delay_mult=1, max_steps=25000):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.lr_delay_steps > 0: # reverse cosine decay
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(self.last_epoch / self.lr_delay_steps, 0, 1)
            )
        else:   # standard exponential decay
            delay_rate = 1.0
        t = np.clip(self.last_epoch / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return [delay_rate * log_lerp for _ in self.base_lrs]
