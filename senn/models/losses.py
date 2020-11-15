import tensorflow as tf



def compas_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for the Compas data
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design

    Parameters
    ----------
    x            : torch.tensor #actual, in this case: =input
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor #predicted
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor #the basis for all predictions (combined by an aggregator)
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor #the weights when aggregating concepts
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
   
    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    batch_size = x.size(0)
    num_classes = aggregates.size(1)

    grad_tensor = tf.ones([batch_size, num_classes])  # in torch explicitly converted: .to(x.device)

    #  grad still needs to be replaced when this is converted to a keras class
    J_yx = grad(outputs=aggregates, inputs=x,
                               grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
    #  as 'only_inputs' is True, the function will only return a list of gradients w.r.t the specified inputs.

    # bs x num_features -> bs x num_features x num_classes
    J_yx = J_yx.unsqueeze(-1)

    # J_hx = Identity Matrix; h(x) is identity function
    robustness_loss = J_yx - relevances

    normed = tf.norm(robustness_loss, ord='fro')
    return normed


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
    # recon_loss = F.binary_cross_entropy(x_hat, x.detach(), reduction="mean") #old code
    recon_loss = tf.keras.losses.mean_squared_error(x_hat, x.detach())
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
    mse_loss = tf.keras.losses.mean_squared_error(x_hat, x.detach())
    abs_concepts = tf.abs(concepts).sum()  # does .sum() work properly?
    return mse_loss + sparsity_reg * abs_concepts


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
