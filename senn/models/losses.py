import torch


def compas_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for the Compas data
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design

    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
   
    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    batch_size = x.size(0)
    num_classes = aggregates.size(1)

    grad_tensor = torch.ones(batch_size, num_classes).to(x.device)
    J_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
                               grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
    # bs x num_features -> bs x num_features x num_classes
    J_yx = J_yx.unsqueeze(-1)

    # J_hx = Identity Matrix; h(x) is identity function
    robustness_loss = J_yx - relevances

    return robustness_loss.norm(p='fro')


def BVAE_loss(x, x_hat, z_mean, z_logvar):
    """ Calculate Beta-VAE loss as in [1]

    Parameters
    ----------
    x : torch.tensor
        input data to the Beta-VAE

    x_hat : torch.tensor
        input data reconstructed by the Beta-VAE

    z_mean : torch.tensor
        mean of the latent distribution of shape
        (batch_size, latent_dim)

    z_logvar : torch.tensor
        diagonal log variance of the latent distribution of shape
        (batch_size, latent_dim)

    Returns
    -------
    loss : torch.tensor
        loss as a rank-0 tensor calculated as:
        reconstruction_loss + beta * KL_divergence_loss

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """
    # recon_loss = F.binary_cross_entropy(x_hat, x.detach(), reduction="mean")
    recon_loss = F.mse_loss(x_hat, x.detach(), reduction="mean")  # replace reference to torch functional
    kl_loss = kl_div(z_mean, z_logvar)
    return recon_loss, kl_loss

def mse_l1_sparsity(x, x_hat, concepts, sparsity_reg):
    """Sum of Mean Squared Error and L1 norm weighted by sparsity regularization parameter

    Parameters
    ----------
    x : torch.tensor
        Input data to the encoder.
    x_hat : torch.tensor
        Reconstructed input by the decoder.
    concepts : torch.Tensor
        Concept (latent code) activations.
    sparsity_reg : float
        Regularizer (xi) for the sparsity term.

    Returns
    -------
    loss : torch.tensor
        Concept loss
    """
    return F.mse_loss(x_hat, x.detach()) + sparsity_reg * torch.abs(concepts).sum()  # replace reference to torch functional


def kl_div(mean, logvar):
    """Computes KL Divergence between a given normal distribution
    and a standard normal distribution

    Parameters
    ----------
    mean : torch.tensor
        mean of the normal distribution of shape (batch_size x latent_dim)

    logvar : torch.tensor
        diagonal log variance of the normal distribution of shape (batch_size x latent_dim)

    Returns
    -------
    loss : torch.tensor
        KL Divergence loss computed in a closed form solution
    """
    batch_loss = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).mean(dim=0)
    loss = batch_loss.sum()
    return loss
