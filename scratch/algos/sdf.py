import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
from tqdm.auto import tqdm
import scratch.utils.data_2d as data_2d
from fabric.utils.event import EventStorage


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse2psnr = lambda mse: -10.0 * np.log(mse) / np.log(10.0)


def laplace_cdf(x, μ, beta):
    """
    Compute the CDF of the Laplace distribution, where μ is the mean
    and beta is the scaling parameter (intuitively, the variance).
    """
    return torch.cat([
        0.5 * torch.exp((x[x <= 0] - μ) / beta),
        1 - 0.5 * torch.exp(-(x[x > 0] - μ) / beta)
    ])


def volsdf_density(x, beta, alpha=None):
    """
    VolSDF's heuristic for computing the density based on the Laplacian
    CDF of the signed distance values.
    """
    alpha = 1.0 / beta if alpha is None else alpha
    sigma = alpha * laplace_cdf(x, 0.0, beta)
    return sigma


def s_density(s, x):
    """
    NeuS's heuristic for importance sampling. phi is the logistic density
    distribution and its standard deviation is given by 1/s.
    """
    phi = s * torch.exp(-s * x) / (1 + torch.exp(-s * x))**2
    return phi


class SDFTrainer2D(nn.Module):
    def __init__(
        self,
        exp_dir='plots',
        n=1000,
        sampler_default=None,
        meshgrid_gran=101,
        geometric_init=True,
        use_eikonal_loss=True,
        weight_norm=True
    ):
        """
        A neural network that approximates a 2D SDF. The input is a set of 2D points
        of shape (n, 2) and the output is a set of signed distance values of shape (n, 1).
        """
        super().__init__()
        self.exp_dir = exp_dir
        os.makedirs(f'{exp_dir}', exist_ok=True)
        self._generate_data(n, sampler_default, meshgrid_gran)

        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.Softplus(beta=100),
            nn.Linear(64, 64),
            nn.Softplus(beta=100),
            nn.Linear(64, 1)
        ).to(device)

        if geometric_init:  # TODO: apply circular initialization to weight parameters
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer.weight.shape[0]))
                    torch.nn.init.constant_(layer.bias, 0.0)
                    if weight_norm:
                        layer = torch.nn.utils.weight_norm(layer)

        self.use_eikonal_loss = use_eikonal_loss

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)

    def _extract_gt_sdf(self, sampled_pts):
        """
        Extract the ground truth SDF from the given data.
        """
        x = torch.linspace(-4.0, 4.0, self.meshgrid_gran, dtype=torch.float32)
        y = torch.linspace(-4.0, 4.0, self.meshgrid_gran, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        meshgrid_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

        euclidean_dists = torch.cdist(meshgrid_pts, sampled_pts[:, :2]).to(device)
        gt_sdf = euclidean_dists.min(dim=1).values
        assert gt_sdf.shape == torch.Size([self.meshgrid_gran ** 2]), \
            f'gt_sdf expected shape [{self.meshgrid_gran ** 2}], got {gt_sdf.shape}'
        return meshgrid_pts, gt_sdf

    @torch.no_grad()
    def _visualize_sdf(self, iteration=None):
        color = self.gt_sdf.cpu() if iteration is None else self.forward(self.meshgrid_pts).cpu()
        plt.scatter(self.meshgrid_pts[:, 0].cpu(), self.meshgrid_pts[:, 1].cpu(), c=color)
        plt.colorbar()
        plt.contour(    # visualize the level sets
            self.meshgrid_pts[:, 0].cpu().reshape(self.meshgrid_gran, self.meshgrid_gran),
            self.meshgrid_pts[:, 1].cpu().reshape(self.meshgrid_gran, self.meshgrid_gran),
            color.reshape(self.meshgrid_gran, self.meshgrid_gran),
            levels=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            colors=['red','orange','yellow','green','blue','purple','pink','brown','black','gray']
        )
        file = f'{self.exp_dir}/gt_sdf.pdf' if iteration is None else f'{self.exp_dir}/sdf_{iteration}.pdf'
        plt.savefig(f'{file}', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_data(self, n=1000, sampler_default=None, meshgrid_gran=101):
        """
        Generate n samples from a random 2D dataset.
        """
        if sampler_default is None:
            sample_cls = np.random.choice(data_2d._ALL_SAMPLERS)
        else:
            assert sampler_default in data_2d._ALL_SAMPLERS, \
                f'Invalid sampler {sampler_default}'
            sample_cls = sampler_default

        sampler = sample_cls()
        self.meshgrid_gran = meshgrid_gran
        self.sampler_name = sample_cls.__name__
        self.sampled_pts = torch.from_numpy(sampler.sample(n)).float().to(device)[:, :2]
        self.meshgrid_pts, self.gt_sdf = self._extract_gt_sdf(self.sampled_pts)
        self._visualize_sdf()
        data_2d.simple_2d_show(self.sampled_pts.cpu().numpy(), self.sampler_name)

    def forward(self, pts):
        """
        Compute the SDF for the given set of points.
        """
        return self.network(pts)

    def eikonal_loss(self):
        """
        TODO: Compute the Eikonal loss for the sampled set of points.
        """
        pts = torch.cat([self.sampled_pts, self.sampled_pts * 0.99 - 0.01], dim=0).requires_grad_(True)
        sdf = self.forward(pts)
        d_output = torch.ones_like(sdf, requires_grad=False, device=device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        grad_sdf_norm = gradients.norm(dim=1, keepdim=True)
        return ((grad_sdf_norm - 1.0)**2).mean()

    def train(self, n_iters=200000, refresh_rate=10):
        pbar = tqdm(range(n_iters), miniters=refresh_rate, file=sys.stdout)
        with EventStorage() as metric:
            for iteration in pbar:
                self.optimizer.zero_grad()
                sdf = self.forward(self.meshgrid_pts)
                sdf_loss = ((sdf - self.gt_sdf)**2).mean()
                if self.use_eikonal_loss:   # compute Eikonal loss
                    sdf_loss += self.eikonal_loss()

                if iteration % refresh_rate == 0:   # compute PSNR on the SDF
                    psnr = self.eval()
                    metric.put_scalars(sdf_loss=sdf_loss.item(), psnr=psnr)
                    pbar.set_description(
                        f'Iteration {iteration:06d}]'
                        + f' eval_psnr = {psnr:.3f},'
                        + f' sdf_loss = {sdf_loss:.3f}'
                    )

                metric.step()

                if iteration % 1000 == 0:
                    self._visualize_sdf(iteration)

                sdf_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                torch.cuda.empty_cache()

    def eval(self):
        with torch.no_grad():
            sdf = self.forward(self.meshgrid_pts)
            mse_loss = ((sdf - self.gt_sdf)**2).mean()
            return mse2psnr(mse_loss.item())


if __name__ == '__main__':
    model = SDFTrainer2D(sampler_default=data_2d.Line)
    model.train()

    # x = torch.arange(-10, 10.01, 0.01, dtype=torch.float32)

    # for beta in [0.01, 0.1, 0.2, 0.4, 0.8, 1.6, 3.0]:
    #     plt.plot(x, laplace_cdf(x, 0.0, beta), label=f'beta={beta}')

    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('Laplace CDF')
    # plt.title('Laplace CDF for different betas and μ=0.0')
    # plt.savefig('laplace_cdf.pdf', dpi=300, bbox_inches='tight')
    # plt.close()

    # for beta in [0.01, 0.1, 0.2, 0.4, 0.8, 1.6, 3.0]:
    #     plt.plot(x, volsdf_density(x, beta), label=f'beta={beta}')

    # plt.legend()
    # plt.xlabel('signed distance values')
    # plt.ylabel('sigma values')
    # plt.title('volSDF sigmas for different betas and μ=0.0')
    # plt.savefig('volsdf_sigmas.pdf', dpi=300, bbox_inches='tight')
    # plt.close()

    # x = torch.linspace(-2.5, 2.5, 1001, dtype=torch.float32)

    # for s in reversed([1.0, 2.0, 5.0, 10.0, 100.0]):
    #     plt.plot(x, s_density(s, x), label=f's={s}')

    # plt.legend()
    # plt.xlabel('SDF values')
    # plt.ylabel('PDF values')
    # plt.title('Logistic PDF for different 1/s values')
    # plt.savefig('neus_s_density.pdf', dpi=300, bbox_inches='tight')
    # plt.close()
