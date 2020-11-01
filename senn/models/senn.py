import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt


class SENN(tf.Module):
    def __init__(self, conceptizer, parameterizer, aggregator):
        """Represents a Self Explaining Neural Network (SENN).
        (https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks)

        A SENN model is a neural network made explainable by design. It is made out of several submodules:
            - conceptizer
                Model that encodes raw input into interpretable feature representations of
                that input. These feature representations are called concepts.
            - parameterizer
                Model that computes the parameters theta from given the input. Each concept
                has with it associated one theta, which acts as a ``relevance score'' for that concept.
            - aggregator
                Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
                h_i represents concept i. The aggregator defines the function g, i.e. how each
                concept with its relevance score is combined into a prediction.

        Parameters
        ----------
        conceptizer : Pytorch Module
            Model that encodes raw input into interpretable feature representations of
            that input. These feature representations are called concepts.

        parameterizer : Pytorch Module
            Model that computes the parameters theta from given the input. Each concept
            has with it associated one theta, which acts as a ``relevance score'' for that concept.

        aggregator : Pytorch Module
            Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
            h_i represents concept i. The aggregator defines the function g, i.e. how each
            concept with its relevance score is combined into a prediction.
        """
        super().__init__()
        self.conceptizer = conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):
        """Forward pass of SENN module.
        
        In the forward pass, concepts and their reconstructions are created from the input x.
        The relevance parameters theta are also computed.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        predictions : torch.Tensor
            Predictions generated by model. Of shape (BATCH, *).
            
        explanations : tuple
            Model explanations given by a tuple (concepts, relevances).

            concepts : torch.Tensor
                Interpretable feature representations of input. Of shape (NUM_CONCEPTS, *).

            parameters : torch.Tensor
                Relevance scores associated with concepts. Of shape (NUM_CONCEPTS, *)
        """
        concepts, recon_x = self.conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concepts, relevances)
        explanations = (concepts, relevances)
        return predictions, explanations, recon_x


class DiSENN(tf.keras.Model):
    """Self-Explaining Neural Network with Disentanglement 

    DiSENN is an extension of the Self-Explaining Neural Network proposed by [1]
    
    DiSENN incorporates a constrained variational inference framework on a 
    SENN Concept Encoder to learn disentangled representations of the 
    basis concepts as in [2]. The basis concepts are then independently
    sensitive to single generative factors leading to better interpretability 
    and lesser overlap with other basis concepts. Such a strong constraint 
    better fulfills the "diversity" desiderata for basis concepts
    in a Self-Explaining Neural Network.

    References
    ----------
    [1] Alvarez Melis, et al.
    "Towards Robust Interpretability with Self-Explaining Neural Networks" NIPS 2018
    [2] Irina Higgins, et al. 
    ”β-VAE: Learning basic visual concepts with a constrained variational framework.” ICLR 2017. 
    
    """

    def __init__(self, vae_conceptizer, parameterizer, aggregator):
        """Instantiates the SENDD with a variational conceptizer, parameterizer and aggregator

        Parameters
        ----------
        vae_conceptizer : nn.Module
            A variational inference model that learns a disentangled distribution over
            the prior basis concepts given the input posterior.

        parameterizer : nn.Module
            Model that computes the parameters theta from given the input. Each concept
            has with it associated one theta, which acts as a ``relevance score'' for that concept.

        aggregator : nn.Module
            Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
            h_i represents concept i. The aggregator defines the function g, i.e. how each
            concept with its relevance score is combined into a prediction.
        """
        super().__init__()
        self.vae_conceptizer = vae_conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):
        """Forward pass of a DiSENN model
        
        The forward pass computes a distribution over basis concepts
        and the corresponding relevance scores. The mean concepts 
        and relevance scores are aggregated to generate a prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape [batch_size, ...]
            
        Returns
        -------
        predictions : torch.Tensor
            Predictions generated by the DiSENN model of shape [batch_size, ...]

        explanations : tuple
            Explanation give by the model as a nested tuple of 
            relevance scores and concept distribution as mean and log variance:
            ((concept_mean, concept_log_variance), relevance_score)

            concept_mean : torch.Tensor
                Mean of the disentangled concept distribution of shape
                [batch_size, num_concepts, concept_dim]

            concept_log_varance : torch.Tensor
                Log Variance of the disentangled concept distribution of shape
                [batch_size, num_concepts, concept_dim]

            relevance_score : torch.Tensor
                Relevance scores (for each concept and class) of shape 
                [batch_size, num_concepts, num_classes]
        """
        concept_mean, concept_logvar, x_reconstruct = self.vae_conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concept_mean, relevances)
        explanations = ((concept_mean, concept_logvar), relevances)
        return predictions, explanations, x_reconstruct

    def explain(self, x, contrast_class, num_prototypes=20, traversal_range=0.45, use_cdf=True,
                show=False, save_as=None, gridsize=(1, 6), col_span=3, figure_size=(18, 3)):
        """Explains the DiSENN predictions for input x
        
        DiSENN explanations consists of the Concepts, the corresponding Relevance 
        Scores, and the Prototypes associated with variations in every single Concept.
        The VAE Conceptizer generates the Prototypes by inducing the prior distribution 
        on the latent space given the input posterior. The mean of this latent 
        prior distribution serves as the Basis Concept vector to be shown. 
        The latent distribution so induced is then sampled from. The sampled 
        Concept vector is traversed independently for each dimension and passed 
        through the decoder of the VAE Conceptizer.Each independent traversal 
        of a single dimension produces one prototype. Finally, we end up with an 
        array of changing prototypes for each Concept dimension. The Parameterizer 
        on the other hand produces the corresponding Relevance Scores. 
        The Relevance Scores and Concepts are shown as bar plots side by side.

        Parameters
        ----------
        x : torch.tensor
            input data of shape (channel x width x height)

        contrast_class: int
            index of the class to compare the predicted class against

        num_prototypes : int
            number of prototypes to generate for each concept dimension

        traversal_range : int
            Range of traversal in each concept dimension from -traversal_range to +traversal_range

        show: bool
            whether to show the figure or not

        save_as : string
            file name of the explanation to be saved as, not saved by default

        gridsize : (int, int)
            shape of the figure in terms of rows and columns

        col_span : int
            number of columns for the Prototypes 

        figure_size : (float, float)
            size of the figure
        
        Returns
        -------
            None
        """
        assert len(x.shape) == 3, \
            "input x must be a rank 3 tensor of shape channel x width x height"

        self.eval()
        y_pred, explanations, _ = self.forward(x.unsqueeze(0))
        (x_posterior_mean, x_posterior_logvar), relevances = explanations
        x_posterior_mean = x_posterior_mean.squeeze(-1)
        x_posterior_logvar = x_posterior_logvar.squeeze(-1)

        concepts = x_posterior_mean.cpu().detach().numpy()
        num_concepts = concepts.shape[1]
        concepts_sample = self.vae_conceptizer.sample(x_posterior_mean,
                                                      x_posterior_logvar).detach()
        # generate new concept vector for each prototype
        # by traversing independently in each dimension
        concepts_sample = concepts_sample.repeat(num_prototypes, 1)
        mean = x_posterior_mean.cpu().detach().numpy()
        std = tf.math.exp(x_posterior_logvar.detach() / 2)  # TODO how to convert this?: .cpu().numpy()
        concepts_traversals = [self.traverse(concepts_sample, dim, traversal_range,
                                             num_prototypes, mean[:, dim], std[:, dim], use_cdf)
                               for dim in range(num_concepts)]
        concepts_traversals = tf.concat(concepts_traversals, dim=0)
        prototypes = self.vae_conceptizer.decoder(concepts_traversals)
        prototype_imgs = prototypes.view(-1, x.shape[0], x.shape[1], x.shape[2])

        # nrow is number of original images in a row which must be the number of prototypes
        # prototype_grid_img = make_grid(prototype_imgs, nrow=num_prototypes).cpu().detach().numpy()
        # TODO make tf draw the prototypes

        # prepare to plot
        relevances = relevances.squeeze(0).cpu().detach().numpy()
        predict_class = y_pred.argmax(1).item()
        relevances_pred = relevances[:, predict_class]
        relevances_contrast = relevances[:, contrast_class]
        concepts = concepts.squeeze(0)
        product_pred = concepts * relevances_pred
        product_contrast = concepts * relevances_contrast
        pred_colors = ['g' if r > 0 else 'r' for r in product_pred]
        contrast_colors = ['g' if r > 0 else 'r' for r in product_contrast]

        # plot input image, relevances, concepts, prototypes side by side
        plt.style.use('seaborn-paper')
        fig = plt.figure(figsize=figure_size)
        ax1 = plt.subplot2grid(gridsize, (0, 0))
        ax2 = plt.subplot2grid(gridsize, (0, 1))
        ax3 = plt.subplot2grid(gridsize, (0, 2))
        ax4 = plt.subplot2grid(gridsize, (0, 3), colspan=col_span)

        ax1.imshow(x.cpu().numpy().squeeze(), cmap='gray')
        ax1.set_axis_off()
        ax1.set_title(f'Input Prediction: {y_pred.argmax(1).item()}', fontsize=18)

        ax2.barh(range(num_concepts), product_pred, color=pred_colors)
        ax2.set_xlabel(f"Class:{predict_class} Contribution", fontsize=18)
        ax2.xaxis.set_label_position('top')
        ax2.tick_params(axis='x', which='major', labelsize=12)
        ax2.set_yticks([])

        ax3.barh(range(num_concepts), product_contrast, color=contrast_colors)
        ax3.set_xlabel(f"Class:{contrast_class} Contribution", fontsize=18)
        ax3.xaxis.set_label_position('top')
        ax3.tick_params(axis='x', which='major', labelsize=12)
        ax3.set_yticks([])

        # ax4.imshow(prototype_grid_img.transpose(1, 2, 0))
        # TODO draw prototypes in appropriate format
        ax4.set_title('Prototypes', fontsize=18)
        ax4.set_axis_off()

        fig.tight_layout()

        if save_as is not None: fig.savefig(save_as)
        if show: plt.show()
        plt.close()

    def traverse(self, matrix, dim, traversal_range, steps,
                 mean=None, std=None, use_cdf=True):
        """Linearly traverses through one dimension of a matrix independently
        
        Parameters
        ----------
        matrix: torch.tensor
            matrix whose dimensions will be traversed independently
        dim: int
            dimension of the matrix to be traversed
        traversal_range: int
            maximum value of the traversal range, if use_cdf is true this should be less than 0.5
        steps: int
            number of steps in the traversal range
        mean: float
            mean of the distribution for traversal using cdf
        std: float
            std of the distribution for traversal using cdf
        use_cdf: bool
            whether to use cdf traversal
        """

        if use_cdf:
            assert traversal_range < 0.5, \
                "If CDF is to be used, the traversal range must represent probability range of -0.5 < p < +0.5"
            assert mean is not None and std is not None, \
                "If CDF is to be used, mean and std has to be defined"
            prob_traversal = (1 - 2 * traversal_range) / 2  # from 0.45 to 0.05
            prob_traversal = stats.norm.ppf(prob_traversal, loc=mean, scale=std)[0]  # from 0.05 to -1.645
            traversal = tf.linspace(-1 * prob_traversal, prob_traversal, steps)
            matrix_traversal = matrix.clone()  # to avoid changing the matrix
            matrix_traversal[:, dim] = traversal
        else:
            traversal = tf.linspace(-1 * traversal_range, traversal_range, steps)
            matrix_traversal = matrix.clone()  # to avoid changing the matrix
            matrix_traversal[:, dim] = traversal
        return matrix_traversal
