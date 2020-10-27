from abc import abstractmethod
import tensorflow as tf
import keras
from keras.layers import Dense
import torch


class Conceptizer(tf.module):
    def __init__(self):
        """
        A general Conceptizer meta-class. Children of the Conceptizer class
        should implement encode() and decode() functions.
        """
        super(Conceptizer, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

    def forward(self, x):
        """
        Forward pass of the general conceptizer.

        Computes concepts present in the input.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        encoded : torch.Tensor
            Encoded concepts (batch_size, concept_number, concept_dimension)
        decoded : torch.Tensor
            Reconstructed input (batch_size, *)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded.view_as(x)

    @abstractmethod
    def encode(self, x):
        """
        Abstract encode function to be overridden.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        """
        pass

    @abstractmethod
    def decode(self, encoded):
        """
        Abstract decode function to be overridden.
        Parameters
        ----------
        encoded : torch.Tensor
            Latent representation of the data
        """
        pass


class IdentityConceptizer(Conceptizer):
    def __init__(self, **kwargs):
        """
        Basic Identity Conceptizer that returns the unchanged input features.
        """
        super().__init__()

    def encode(self, x):
        """Encoder of Identity Conceptizer.

        Leaves the input features unchanged  but reshapes them to three dimensions
        and returns them as concepts (use of raw features -> no concept computation)

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, INPUT_FEATURES).

        Returns
        -------
        concepts : torch.Tensor
            Unchanged input features but with extra dimension (BATCH, INPUT_FEATURES, 1).
        """
        return x.unsqueeze(-1)

    def decode(self, z):
        """Decoder of Identity Conceptizer.

        Simulates reconstruction of the original input x by undoing the reshaping of the encoder.
        The 'reconstruction' is identical to the input of the conceptizer.

        Parameters
        ----------
        z : torch.Tensor
            Output of encoder (input x reshaped to three dimensions), size: (BATCH, INPUT_FEATURES, 1).

        Returns
        -------
        reconst : torch.Tensor
            Unchanged input features (identical to x)
        """
        return z.squeeze(-1)


class VaeConceptizer(tf.nn):
    """Variational Auto Encoder to generate basis concepts

    Concepts should be independently sensitive to single generative factors,
    which will lead to better interpretability and fulfill the "diversity" 
    desiderata for basis concepts in a Self-Explaining Neural Network.
    VAE can be used to learn disentangled representations of the basis concepts 
    by emphasizing the discovery of latent factors which are disentangled. 
    """

    def __init__(self, image_size, num_concepts, **kwargs):
        """Initialize Variational Auto Encoder

        Parameters
        ----------
        image_size : int
            size of the width or height of an image, assumes square image
        num_concepts : int
            number of basis concepts to learn in the latent distribution space
        """
        super().__init__()
        self.in_dim = image_size * image_size
        self.z_dim = num_concepts
        self.encoder = VaeEncoder(self.in_dim, self.z_dim)
        self.decoder = VaeDecoder(self.in_dim, self.z_dim)

    def forward(self, x):
        """Forward pass through the encoding, sampling and decoding step

        Parameters
        ----------
        x : torch.tensor 
            input of shape [batch_size x ... ], which will be flattened

        Returns
        -------
        concept_mean : torch.tensor
            mean of the latent distribution induced by the posterior input x
        x_reconstruct : torch.tensor
            reconstruction of the input in the same shape
        """
        concept_mean, concept_logvar = self.encoder(x)
        concept_sample = self.sample(concept_mean, concept_logvar)
        x_reconstruct = self.decoder(concept_sample)
        return (concept_mean.unsqueeze(-1),
                concept_logvar.unsqueeze(-1),
                x_reconstruct.view_as(x))

    def sample(self, mean, logvar):
        """Samples from the latent distribution using reparameterization trick

        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon is drawn from a standard normal distribution
        
        Parameters
        ----------
        mean : torch.tensor
            mean of the latent distribution of shape [batch_size x z_dim]
        log_var : torch.tensor
            diagonal log variance of the latent distribution of shape [batch_size x z_dim]
        
        Returns
        -------
        z : torch.tensor
            sample latent tensor of shape [batch_size x z_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        return z


class VaeEncoder(tf.nn):
    """Encoder of a VAE"""

    def __init__(self, in_dim, z_dim):
        """Instantiate a multilayer perceptron

        Parameters
        ----------
        in_dim: int
            dimension of the input data
        z_dim: int
            latent dimension of the encoder output
        """
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.FC = keras.Sequential(
            [
                keras.layers.Flatten(),
                tf.keras.Input(shape=self.in_dim),  # TODO is this layer necessary?
                Dense(512, activation='linear'),  # (in_dim, 512)
                Dense(512, activation='relu'),
                Dense(256, activation='linear'),  # (512, 256)
                Dense(256, activation='relu'),
                Dense(100, activation='linear'),  # (256, 100)
                Dense(100, activation='relu')
            ]
        )
                                                        # TODO fix those shapes:
        self.mean_layer = Dense(activation='linear'),   # (100, z_dim)
        self.logvar_layer = Dense(activation='linear'), # (100, z_dim)

    def forward(self, x):
        """Forward pass of the encoder
        """
        x = self.FC(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar


class VaeDecoder(tf.nn):
    """Decoder of a VAE"""

    def __init__(self, in_dim, z_dim):
        """Instantiate a multilayer perceptron

        Parameters
        ----------
        in_dim: int
            dimension of the input data
        z_dim: int
            latent dimension of the encoder output
        """
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.FC = keras.Sequential(
            [
                keras.layers.Flatten(),
                tf.keras.Input(shape=z_dim),  # TODO is this layer necessary?
                Dense(100, activation='linear'),  # (z_dim, 100)
                Dense(100, activation='relu'),
                Dense(256, activation='linear'),  # (100, 256)
                Dense(256, activation='relu'),
                Dense(512, activation='linear'),  # (256, 512)
                Dense(512, activation='relu'),
                Dense(in_dim, activation='linear'),  # (512, in_dim)
            ]
        )

    def forward(self, x):
        """Forward pass of a decoder"""
        x_reconstruct = torch.sigmoid(self.FC(x))
        return x_reconstruct


class Flatten(tf.nn):
    def forward(self, x):
        """
        Flattens the inputs to only 3 dimensions, preserving the sizes of the 1st and 2nd.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (dim1, dim2, *).

        Returns
        -------
        flattened : torch.Tensor
            Flattened input (dim1, dim2, dim3)
        """
        return x.view(x.size(0), x.size(1), -1)


def handle_integer_input(input, desired_len):
    """
    Checks if the input is an integer or a list.
    If an integer, it is replicated the number of  desired times
    If a tuple, the tuple is returned as it is

    Parameters
    ----------
    input : int, tuple
        The input can be either a tuple of parameters or a single parameter to be replicated
    desired_len : int
        The length of the desired list

    Returns
    -------
    input : tuple[int]
        a tuple of parameters which has the proper length.
    """
    if type(input) is int:
        return (input,) * desired_len
    elif type(input) is tuple:
        if len(input) != desired_len:
            raise AssertionError("The sizes of the parameters for the CNN conceptizer do not match."
                                 f"Expected '{desired_len}', but got '{len(input)}'")
        else:
            return input
    else:
        raise TypeError(f"Wrong type of the parameters. Expected tuple or int but got '{type(input)}'")


class ScalarMapping(tf.nn):
    def __init__(self, conv_block_size):
        """
        Module that maps each filter of a convolutional block to a scalar value

        Parameters
        ----------
        conv_block_size : tuple (int iterable)
            Specifies the size of the input convolutional block: (NUM_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH)
        """
        super().__init__()
        self.num_filters, self.filter_height, self.filter_width = conv_block_size

        self.layers = nn.ModuleList()
        for _ in range(self.num_filters):
            self.layers.append(nn.Linear(self.filter_height * self.filter_width, 1))

    def forward(self, x):
        """
        Reduces a 3D convolutional block to a 1D vector by mapping each 2D filter to a scalar value.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, CHANNELS, HEIGHT, WIDTH).

        Returns
        -------
        mapped : torch.Tensor
            Reduced input (BATCH, CHANNELS, 1)
        """
        x = x.view(-1, self.num_filters, self.filter_height * self.filter_width)
        mappings = []
        for f, layer in enumerate(self.layers):
            mappings.append(layer(x[:, [f], :]))
        return torch.cat(mappings, dim=1)
