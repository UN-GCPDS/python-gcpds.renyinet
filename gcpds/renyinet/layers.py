import numpy as np
import tensorflow as tf
from tesorflow.keras import layers
import tensorflow_probability as tfp

class GFC(layers.Layer):
    """Gaussian Functional Connectivity Layer for multi-channel time series.

    This layer calculates the functional connectivity along the channels of a time series
    using the Gaussian kernel function defined as:

    ..math::
    k(x,y) = e^{-\frac{||x-y||_2^2}{\sigma^2}}

    The input is expected to be (None, Channels, Time, Features) and will output a
    (None, Features, Channels, Channels) tensor. This can be read as: for each feature,
    a Channels x Channels matrix is calculated, which defines how similar a channel is 
    to another (1 being identical, and 0 being completely different).

    Attributes
    ----------
    gammad: tf.Tensor
        The layer weights

    Methods
    ----------
    build(batch_input_shape)
        The build function which initializes the weights.
    call(X)
        The call function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, batch_input_shape:tuple[int]):
        """The build function.

        This function initializes the weights and creates the layer.

        Parameters
        ----------
        batch_input_shape : tuple[int]
            The expected input shape.
        """
        self.gammad = self.add_weight(name = 'gammad',
                                shape = (),
                                initializer = 'zeros',
                                trainable = True)
        super().build(batch_input_shape)

    def call(self, X : tf.Tensor | np.ndarray) -> tf.Tensor:
        """The call function

        Calculate the functional connectivity for the input tensor.

        Parameters
        ----------
        X : tf.Tensor | np.ndarray
            The input tensor.

        Returns
        -------
        tf.Tensor
            The functional connectivities as calculated by the kernel.
        """
        X = tf.transpose(X, perm  = (0, 3, 1, 2)) #(N, F, C, T)
        R = tf.reduce_sum(tf.math.multiply(X, X), axis = -1, keepdims = True) #(N, F, C, 1)
        D  = R - 2*tf.matmul(X, X, transpose_b = (0, 1, 3, 2)) + tf.transpose(R, perm = (0, 1, 3, 2)) #(N, F, C, C)

        ones = tf.ones_like(D[0,0,...]) #(C, C)
        mask_a = tf.linalg.band_part(ones, 0, -1) #Upper triangular matrix of 0s and 1s (C, C)
        mask_b = tf.linalg.band_part(ones, 0, 0)  #Diagonal matrix of 0s and 1s (C, C)
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool) #Make a bool mask (C, C)
        triu = tf.expand_dims(tf.boolean_mask(D, mask, axis = 2), axis = -1) #(N, F, C*(C-1)/2, 1)
        sigma = tfp.stats.percentile(tf.math.sqrt(triu), 50, axis = 2, keepdims = True) #(N, F, 1, 1)

        A = tf.math.exp(-1/(2*tf.pow(10., self.gammad)*tf.math.square(sigma))*D) #(N, F, C, C)
        A.set_shape(D.shape)
        return A

class MultiscaleGFC(layers.Layer):
    """Gaussian Functional Connectivity Layer for multi-channel time series with multiple scale values.

    This layer calculates the functional connectivity along the channels of a time series
    using the Gaussian kernel function defined as:

    ..math::
    k(x,y) = e^{-\frac{||x-y||_2^2}{\sigma^2}}

    The input is expected to be (None, Channels, Time, Features) and will output a
    (None, Features, Channels, Channels) tensor. This can be read as: for each feature,
    a Channels x Channels matrix is calculated, which defines how similar a channel is 
    to another (1 being identical, and 0 being completely different).

    The main difference between GFC and this layer is that GFC takes the same sigma value for all
    features, while this layer allows for a different sigma for each.

    Attributes
    ----------
    gammad: tf.Tensor
        The layer weights

    Methods
    ----------
    build(batch_input_shape)
        The build function which initializes the weights.
    call(X)
        The call function.
    """
    def __init__(self, initializer: str | tf.keras.initializers.Initializer = 'glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.initializer = initializer

    def build(self, batch_input_shape: tuple[int]):
        """The build function.

        This function initializes the weights and creates the layer.

        Parameters
        ----------
        batch_input_shape : tuple[int]
            The expected input shape.
        """
        self.gammad = self.add_weight(name = 'gammad',
                                shape = (1, batch_input_shape[-1]), #
                                initializer = self.initializer,
                                trainable = True)
        super().build(batch_input_shape)

    def call(self, X: tf.Tensor | np.ndarray) -> tf.Tensor:
        """The call function

        Calculate the functional connectivity for the input tensor.

        Parameters
        ----------
        X : tf.Tensor | np.ndarray
            The input tensor.

        Returns
        -------
        tf.Tensor
            The functional connectivities as calculated by the kernel.
        """
        X = tf.transpose(X, perm  = (0, 3, 1, 2)) #(N, F, C, T)
        R = tf.reduce_sum(tf.math.multiply(X, X), axis = -1, keepdims = True) #(N, F, C, 1)
        D  = R - 2*tf.matmul(X, X, transpose_b = (0, 1, 3, 2)) + tf.transpose(R, perm = (0, 1, 3, 2)) #(N, F, C, C)

        ones = tf.ones_like(D[0,0,...]) #(C, C)
        mask_a = tf.linalg.band_part(ones, 0, -1) #Upper triangular matrix of 0s and 1s (C, C)
        mask_b = tf.linalg.band_part(ones, 0, 0)  #Diagonal matrix of 0s and 1s (C, C)
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool) #Make a bool mask (C, C)
        triu = tf.expand_dims(tf.boolean_mask(D, mask, axis = 2), axis = -1) #(N, F, C*(C-1)/2, 1)
        sigma = tfp.stats.percentile(tf.math.sqrt(triu), 50, axis = 2, keepdims = False) #(N, F, 1, 1)

        #A = (2*tf.pow(10., self.gammad)*tf.math.square(sigma)[...,0])
        A = (2*self.gammad*tf.math.square(sigma)[...,0])
        A = tf.expand_dims(tf.expand_dims(A,-1),-1)
        A = tf.math.exp(-1/A*D) #(N, F, C, C)
        A.set_shape(D.shape)
        return A

class Triu(layers.Layer):
    """An upper triangular layer.

    This layer calculates and flattens the upper triangular of a square matrix.

    The input is expected to have shape (None, Features, M, M) and will output 
    a vector of shape (None, Features, M*(M-1)/2). This can be read as: for each
    feature, the layer will flatten the upper triangular matrix into a vector.

    This layer is useful when working with symetrical matrices which under a normal
    flatten, produce a staggering amount of parameters.

    Attributes
    ----------
    dims: tuple[int]
        The input shape

    Methods
    ----------
    build(batch_input_shape)
        The build function which initializes the weights.
    call(X)
        The call function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, batch_input_shape: tuple[int]):
        """The build function

        This function builds the layer and saves the input's shape

        Parameters
        ----------
        batch_input_shape : tuple[int]
            The expected input shape
        """
        self.dims = batch_input_shape
        super().build(batch_input_shape)

    def call(self, X: tf.Tensor | np.ndarray) -> tf.Tensor:
        """The call function

        Calculates the upper triangular and flattens it.

        Parameters
        ----------
        X : tf.Tensor | np.ndarray
            The input data

        Returns
        -------
        tf.Tensor
            The flattened upper diagonal
        """
        dims = self.dims
        ones = tf.ones(dims[2:]) #(C, C)
        mask_a = tf.linalg.band_part(ones, 0, -1) #Upper triangular matrix of 0s and 1s (C, C)
        mask_b = tf.linalg.band_part(ones, 0, 0)  #Diagonal matrix of 0s and 1s (C, C)
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool) #Make a bool mask (C, C)
        triu = tf.boolean_mask(X, mask, axis = 2) #(N, F, C*(C-1)/2)
        triu.set_shape((dims[0], dims[1], int(dims[2]*(dims[3]-1)/2)))
        return triu