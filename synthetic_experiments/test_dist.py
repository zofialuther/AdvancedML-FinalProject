# Run RealNVP on a given base/target combination for a given dimension

# %% [markdown]
# # Real NVP

# %%
# Import required packages
import torch
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
# Set up model

def run(base = nf.distributions.DiagGaussian(2), target = nf.distributions.TwoModes(2, 0.1), dim = 2, plot = False, uniform = False, enable_cuda = True):

    # Define flows
    K = 64
    torch.manual_seed(0)

    latent_size = dim
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set target and q0
    target = target
    q0 = base

    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)

    # Move model on GPU if available
    #device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    device = 'cpu'
    nfm = nfm.to(device)
    nfm = nfm.double()

    print(uniform)
    # Initialize ActNorm
    if not uniform:
        z, _ = nfm.sample(num_samples=2 ** 7)
    else:
        z = nfm.q0.sample(num_samples=2 ** 7)
        for flow in nfm.flows:
            z, _ = flow(z)
    z_np = z.to('cpu').data.numpy()
    
    if plot:
        plt.figure(figsize=(15, 15))
        plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (200, 200), range=[[-3, 3], [-3, 3]])
        plt.gca().set_aspect('equal', 'box')
        plt.show()
        

    # %%
    # Plot target distribution
    '''
    grid_size = 200
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    '''
    zz, xx, yy = create_grid(latent_size)
    zz = zz.double().to(device)
    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob_target = torch.exp(log_prob)

    # Plot initial posterior distribution
    log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    if plot:
        print("Initial Posterior Distribution:")
        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        plt.gca().set_aspect('equal', 'box')
        plt.show()
        

    # %%
    # Train model
    max_iter = 2000
    num_samples = 2 * 10
    anneal_iter = 1000
    annealing = True
    show_iter = 100


    loss_hist = np.array([])

    optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-6)
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()
        if annealing:
            loss = nfm.reverse_kld(num_samples, beta=np.min([1., 0.001 + it / anneal_iter]))
        else:
            loss = nfm.reverse_alpha_div(num_samples, dreg=True, alpha=1)
        
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

        # Plot learned posterior
        if (it + 1) % show_iter == 0 and plot:
            log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
            prob = torch.exp(log_prob)
            prob[torch.isnan(prob)] = 0

            plt.figure(figsize=(15, 15))
            plt.pcolormesh(xx, yy, prob.data.numpy())
            plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
            plt.gca().set_aspect('equal', 'box')
            plt.show()

    if plot:
        plt.figure(figsize=(10, 10))
        plt.plot(loss_hist, label='loss')
        plt.legend()
        plt.show()

        # %%
        # Plot learned posterior distribution
        log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        plt.gca().set_aspect('equal', 'box')
        plt.show()


    """
    print("Loss after X iterations:")
    print("20:", loss_hist[19])
    print("200:", loss_hist[199])
    print("2000:", loss_hist[1999])
    print("20000:", loss_hist[19999])
    """
    return loss_hist


def create_grid(dim, grid_size=50, mrange=(-3, 3)):
    """
    Creates a grid in a specified number of dimensions.

    Args:
      dim (int): Number of dimensions.
      grid_size (int): Number of points along each dimension.
      mrange (tuple): The mrange (min, max) for each dimension.

    Returns:
      torch.Tensor: A grid of points in the specified dimension.
    """
    print(dim)

    # Create linspace for each dimension
    linspace = torch.linspace(mrange[0], mrange[1], grid_size)
    
    # Create meshgrid for all dimensions
    linspace_list = [linspace for _ in range(dim)]
    grids = torch.meshgrid(*linspace_list)
    
    # Concatenate and reshape to create the final grid
    grid = torch.cat([g.unsqueeze(-1) for g in grids], dim=-1).view(-1, dim)
    return grid, grids[0], grids[1]