import numpy as np


# adapted from Plenoxels and JaxNeRF
def create_lr_func(
    step,
    lr_init,
    lr_final,
    lr_delay_steps,
    lr_delay_mult,
    max_steps
):
    if lr_delay_steps > 0: # reverse cosine delay
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp
