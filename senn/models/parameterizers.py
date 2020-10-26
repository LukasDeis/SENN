import tensorflow as tf
from tensorflow import keras


class LinearParameterizer(tf.nn):
    def __init__(self, num_concepts, num_classes, hidden_sizes=(10, 5, 5, 10), dropout=0.5, **kwargs):
        """Parameterizer for compas dataset.
        
        Solely consists of fully connected modules.

        Parameters
        ----------
        num_concepts : int
            Number of concepts that should be parameterized (for which the relevances should be determined).
        num_classes : int
            Number of classes that should be distinguished by the classifier.
        hidden_sizes : iterable of int
            Indicates the size of each layer in the network. The first element corresponds to
            the number of input features.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.model = keras.Sequential()
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            self.model.add(keras.layers.Dense(h, h_next, activation='linear'))
            self.model.add(keras.layers.Dropout(self.dropout))
            self.model.add(keras.layers.Dense(h, h_next, activation='relu'))
        self.model.pop()

    def forward(self, x):
        """Forward pass of compas parameterizer.

        Computes relevance parameters theta.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        parameters : torch.Tensor
            Relevance scores associated with concepts. Of shape (BATCH, NUM_CONCEPTS, NUM_CLASSES)
        """
        return self.model(x).view(x.size(0), self.num_concepts, self.num_classes)