from scratch.utils.wizard import WandbWizard
import torch
import torch.nn as nn


def create_hook(n=1, verbose=False):
    """
    This function creates a hook that logs the pre and post statistics of an activation layer every n steps.
    By default, it logs the activations every step (n=1) and is designed for activation layers exclusively. The
    verbose flag controls whether to log statistical measures (mean, std, max, min) of the activation distributions
    (not recommended).
    """

    counter = 0   # counter to keep track of the number of steps

    def hook_fn(module, input, output):
        """
        Three notes:
        1] The module is the layer that we are hooking into - generally we hook into the activation layer
            - The name of the layer is module.__class__.__name__
        2] Input is a tuple; it can contain multiple tensors if the layer accepts multiple inputs
        3] Output is a tensor; if the layer produces multiple outputs, it is a tuple

        For 99% of use cases, input and output contain single tensors
        """
        name = getattr(module, "name", "UnnamedLayer")
        assert name != "UnnamedLayer", "Provide a unique name identifier for each layer"

        nonlocal counter    # nonlocal forces the variable to be accessed from the outer scope
        if counter % n == 0:
            # log distributions
            pre, post = input[0].cpu().detach().numpy(), output.cpu().detach().numpy()
            WandbWizard.log_distributions(
                step=counter,
                **{
                    f"{name}_pre_act": pre,
                    f"{name}_post_act": post
                }
            )

            # log statistical measures if desired
            if verbose:
                WandbWizard.log_scalars(
                    step=counter,
                    **{
                        f"{name}_pre_act_mean": pre.mean(),
                        f"{name}_pre_act_std": pre.std(),
                        f"{name}_pre_act_max": pre.max(),
                        f"{name}_pre_act_min": pre.min(),
                        f"{name}_post_act_mean": post.mean(),
                        f"{name}_post_act_std": post.std(),
                        f"{name}_post_act_max": post.max(),
                        f"{name}_post_act_min": post.min()
                    }
                )

        counter += 1

    return hook_fn


###### ----------------------------- example usage ----------------------------- ######


class SimpleNet(nn.Module):
    def __init__(self, n=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 2)
        ])

        # assign unique name to each ReLU layer and register the hook
        for i, layer in enumerate(self.layers):
            layer.name = f"{layer.__class__.__name__}_{i}"
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(create_hook(n=n))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    n = 5
    model = SimpleNet(n=n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    config = {"lr": 1e-3, "optimizer": "adam"}
    wizard = WandbWizard(project="test", name="test_hooks", config=config)

    for i in range(500):
        optimizer.zero_grad()
        output = model(torch.randn(10, 30))
        loss = output.mean()
        loss.backward()
        if i % n == 0:
            for layer in model.layers:
                if hasattr(layer, "weight"):    # only get gradients for layers with parameters
                    WandbWizard.log_distributions(
                        step=i,
                        **{
                            f"{layer.name}_weight_grad": layer.weight.grad.cpu().detach().numpy(),
                            f"{layer.name}_bias_grad": layer.bias.grad.cpu().detach().numpy()
                        }
                    )
        optimizer.step()
