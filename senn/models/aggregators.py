import tensorflow as tf


class SumAggregator:
    def __init__(self, num_classes, **kwargs):
        """Basic Sum Aggregator that joins the concepts and relevances by summing their products.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, concepts, relevances):
        """Forward pass of Sum Aggregator.

        Aggregates concepts and relevances and returns the predictions for each class.

        Parameters
        ----------
        concepts : torch.Tensor
            Contains the output of the conceptizer with shape (BATCH, NUM_CONCEPTS, DIM_CONCEPT=1).
        relevances : torch.Tensor
            Contains the output of the parameterizer with shape (BATCH, NUM_CONCEPTS, NUM_CLASSES).

        Returns
        -------
        class_predictions : torch.Tensor
            Predictions for each class. Shape - (BATCH, NUM_CLASSES)
            
        """
        permuted = tf.transpose(relevances, perm=[0, 2, 1])  # so that the number of concepts is at the end
        batch_matrix_matrix_product = tf.matmul(permuted, concepts)  # multiply all relevance scores
        #       with their corresponding concepts activation
        aggregated = tf.squeeze(batch_matrix_matrix_product)  # squeeze(-1)  # remove the number of concepts
        return tf.nn.log_softmax(aggregated, dim=1)
