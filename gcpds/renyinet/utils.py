import numpy as np
import tensorflow as tf

class ConstantInitializer(tf.keras.initializers.Initializer):
    """Constant Intitializer

    This is a tensorflow weights intializer which enables the user
    to initialize the weights based on a list of constants.

    Parameters
    ----------
    constants : list
        List of constants
    """
    def __init__(self, constants: list, **kwargs):
        super().__init__(**kwargs)
        self.constants = constants
    
    def __call__(self):
        return tf.constant(
            [self.constants]
        )

def delta_kernel(X:np.ndarray|tf.Tensor,
                 scale:int = 1) -> tf.Tensor:
    """
    This function maps the input vector to a square matrix based on the
    delta kernel function:
    
    ..math::
        K_{ij} = κ(y_i, y_j) = \delta\left|y_i-y_j\right|_1
        
    Parameters
    ----------
    X: np.ndarray | tf.Tensor
        A Binary Array/Tensor with shape (N, 1)
    scale: int, default = 1
        The scale for the delta function. By default, if both elements of
        the array are the same, the function returns 1. This parameter is
        scales this output.
        
    Returns
    ----------
    tf.Tensor
        A Tensor with shape (N, N)
        
    Notes
    ----------
    N in this context is the number of samples/trials.
    
    This version only works for binary arrays/tensors, meaning every element of the
    array has to be a 1 or a 0.
    """
    dims = X.shape
    X_T = tf.reshape(X, (dims[0],1,dims[1]))
    X = tf.reshape(X, (1, dims[0], dims[1]))
    A = scale*tf.reduce_sum(tf.abs(tf.abs(X_T - X)-1), axis=2)/dims[1]
    return A

def custom_softmax(x:tf.Tensor, base:float | int =1) -> tf.Tensor:
    """Custom softmax activation

    This custom version of the softmax allows the user to scale the 
    input values, following the real softmax function:

    ..math::
    out = \frac{e^{\beta x}}{\sum e^{\beta x}}

    This allows the softmax to change behaviour and output a distribution
    closer to an uniform distribution (base = 0) , or a delta distribution (base = inf).

    Parameters
    ----------
    x : tf.Tensor
        Input data
    base : float | int, optional
        scale factor, by default 1

    Returns
    -------
    tf.Tensor
        The softmax output
    
    Notes
    ----------
    More information:
    https://medium.com/mlearning-ai/softmax-temperature-5492e4007f71
    """
    den = tf.reduce_sum((tf.math.exp(base*x)),axis=-1, keepdims=True)
    return tf.math.divide((tf.math.exp(base*x)),den)

def normalize_matrix(X:tf.Tensor) -> tf.Tensor:
    """
    This function Normalizes a matrix following the equation:

    .. math::
        \hat{X}_{ij} = \frac{1}{N}\frac{X_{ij}}{\sqrt{X_{ii}X_{jj}}}

    Parameters
    ----------
    X : tf.Tensor
        A tensor of square matrices

    Returns
    -------
    tf.Tensor
        The normalized matrices
    """
    N = X.shape[-1]
    assert N == X.shape[-2], 'X must be a square matrix'
    diag = tf.expand_dims(tf.linalg.diag_part(X), -1)
    Y = tf.math.divide(X, tf.math.sqrt(tf.linalg.matmul(diag, diag, transpose_b=True)))/N
    return Y


def renyi_entropy(X: tf.Tensor, alpha: int | float) -> tf.Tensor:
    """
    This function calculates Renyi's entropy for matrices defined as:

    .. math::
        H_\alpha(\mathbb{X}) = \frac{1}{1-\alpha}log_2\left(\sum_{\forall i} \lambda_i(\mathbb{X})^\alpha dx \right)
    
    Parameters
    ----------
    X : tf.Tensor
        A tensor of square kernel matrices
    
    alpha: int | float
        alpha value for renyi's entropy
    
    Returns
    -------
    tf.Tensor
        A tensor of entropies
    """
    e, v = tf.linalg.eig(X)
    #e = tf.math.real(e)
    return tf.math.log(tf.reduce_sum(tf.math.real(tf.math.pow(e, alpha)), axis=-1))/(1-alpha)

def joint_renyi_entropy(X, Y, alpha):
    """
    This function calculates Renyi's joint entropy for matrices defined as:

    .. math::
        H_\alpha(\mathbb{X}, \mathbb{Y}) = H_\alpha\left(\frac{\mathbb{X} \circ \mathbb{Y}}{tr(\mathbb{X} \circ \mathbb{Y})}\right)
        
    Parameters
    ----------
    X : tf.Tensor
        A tensor of square kernel matrices
        
    Y : tf.Tensor
        A tensor of square kernel matrices
        
    alpha: int | float
        alpha value for renyi's entropy

    Returns
    -------
    tf.Tensor
        A tensor of entropies
    """
    assert X.shape == Y.shape, f'Both inputs must have same shape'
    Z = tf.multiply(X, Y)
    Z = tf.divide(Z, tf.linalg.trace(Z))
    return renyi_entropy(Z, alpha)

class RenyiMutualInformation(tf.keras.losses.Loss):
    """This is the implementation of Mutual Information based on Renyi's Entropy.

    Mutual information is defined as:

    .. math::
        I(X, Y) = H(X) + H(Y) - H(X,Y)

    and can be understood as the information shared between the random variables
    X and Y. This metric as first proposed by Shannon can also be understood as
    looking at a random variable through the eyes of another.

    Although normally based on Shannon's Entropy, this loss function is based on
    Renyi's entropy defined as:

    .. math::
        H_\alpha(X) = \frac{1}{1-\alpha}log_2\left(\int p(x)^\alpha dx \right)

    This definition has been further modified to work with matrices as follows:

    .. math::
        H_\alpha(\mathbb{X}) = \frac{1}{1-\alpha}log_2\left(\sum_{\forall i} \lambda_i(\mathbb{X})^\alpha dx \right)

    with lambda (\u03BB) being the eigenvalues of the matrix.

    Finally, the joint Renyi entropy is defined as:

    .. math::
        H_\alpha(\mathbb{X}, \mathbb{Y}) = H_\alpha\left(\frac{\mathbb{X} \circ \mathbb{Y}}{tr(\mathbb{X} \circ \mathbb{Y})}\right)

    with A \\circ B being the Hadammard product(element-wise product) between
    A and B.

    Parameters
    ----------
    alpha : float
        The value alpha for Renyi's entropy
    
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def call(self, X, Y):
        """
        Parameters
        ----------
        X : tf.Tensor
            A square kernel matrix
        Y : tf.Tensor
            A square kernel matrix

        Returns
        ----------
        tf.Tensor
            The mutual information between X and Y

        Notes
        ----------
        This Loss function can only be implemented iff A and B are square kernel
        matrices with diagonal = 1/n and trace = 1.
        """
        joint = joint_renyi_entropy(X, Y, self.alpha)
        X_entropy = renyi_entropy(X, self.alpha)
        Y_entropy = renyi_entropy(Y, self.alpha)
        return Y_entropy + X_entropy - joint

class SavePrediction(tf.keras.callbacks.Callback):
    """Custom callback to predict after every epoch
    
    This custom callback runs a prediction after every epoch which serves as a form
    of validation. This is useful when the network has different behaviour depeding
    on the batch_size, as normal validation would require the train and validation
    split to be the same size.
    
    Attributes
    ----------
    history: list
    metrics : list[tf.keras.metrics.Metric]
    X_val : np.ndarray | tf.Tensor
    Y_val : np.ndarray | tf.Tensor
    """
    def __init__(self, metrics, X_val, Y_val):
        super(SavePrediction, self).__init__()
        self.history = []
        self.metrics = metrics
        self.X_val = X_val
        self.Y_val = Y_val

    def on_epoch_end(self, epoch, logs=None):
        Y = self.model.predict(self.X_val, verbose=0)
        l = []
        for f in self.metrics:
            l += [f(Y[0], self.Y_val).numpy()]
        self.history += [l]