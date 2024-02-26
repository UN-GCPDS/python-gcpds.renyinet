import tensorflow as tf
from tesorflow.keras import layers, Model
from tensorflow.keras.constraints import max_norm
from .layers import MultiscaleGFC, Triu
from .utils import ConstantInitializer, normalize_matrix, renyi_entropy

def get_renyinet(nb_classes: int,
          Chans: int,
          Samples: int,
          dropoutRate: float,
          kernLength: int,
          F1: int,
          D: int,
          F2: int,
          norm_rate: int,
          dropoutType: str,
          constants:list) -> tf.keras.Model:
    """This function creates a tensorflow.keras.Model object with the following
    architecture:

    RenyiNet Architecture

    Temporal Block ──> Spatial Block ──> Separable Block ──> Classification Block*
        │                                                      
        │                                                       
        └──> Kernel* ──> Triu ──> Flatten ──> Dense ──> Softmax*
                                          
    Layer* == Output

    Parameters
    ----------
    nb_classes : int
        The number of classes
    Chans : int
        The number of input channels
    Samples : int
        The number of input time points/samples
    dropoutRate : float
        The dropout layers dropout_rate
    kernLength : int
        The temporal convolution kernel length
    F1 : int
        The number of temporal filters at the Temporal Block
    D : int
        The depth multiplier for the Spatial Block Depthwise convolution
    F2 : int
        The number of filters to use at the Seperable Block
    norm_rate : int
        The max_norm normalization norm_rate
    dropoutType : str
        The type of dropout to use
    constants : list
        The list of constants for the MultiscaleGFC Layer

    Returns
    -------
    tf.keras.Model
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = (Chans, Samples, 1))

    ################# Temporal Block ###########################################
    block1       = layers.Conv2D(F1, (1, kernLength), padding = 'same',
                                   name='Conv2D_1',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = layers.BatchNormalization()(block1)
    ################# Kernel ###################################################
    adj_mat      = MultiscaleGFC(name = 'gfc', initializer = ConstantInitializer(constants), trainable=False)(block1)
    triu_kernel  = Triu(name = 'triu')(adj_mat)
    flatten_kernel = layers.Flatten(name = 'flatten_kernel')(triu_kernel)
    flatten_kernel = dropoutType(dropoutRate)(flatten_kernel)
    kernel_weights = layers.Dense(F1,
                                  name = 'kernel_weights', 
                                  use_bias = False,
                                  activation = 'linear')(flatten_kernel)
    #This unit normalization scales the tensor before the softmax, essentially acting like a temperature value.
    kernel_weights = layers.UnitNormalization()(kernel_weights)
    kernel_weights = layers.Activation('softmax')(kernel_weights)
    ################# Spatial Block ############################################
    block1       = layers.DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   name='Depth_wise_Conv2D_1',
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = layers.BatchNormalization()(block1)
    block1       = layers.Activation('elu')(block1)
    block1       = layers.AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    ################# Separable Block ##########################################
    block2       = layers.SeparableConv2D(F2, (1, 16),
                                   name='Separable_Conv2D_1',
                                   use_bias = False, padding = 'same')(block1)
    block2       = layers.BatchNormalization()(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
    ################# Classification Block #####################################    
    flatten      = layers.Flatten(name = 'flatten')(block2)
    dense        = layers.Dense(nb_classes, name = 'output', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = layers.Activation('softmax', name = 'out_activation')(dense)

    return Model(inputs=input1, outputs = [softmax, adj_mat, kernel_weights])

class RenyiNet(Model):
    """A tensorflow model with custom train step
    
    This class allows for the custom training of a model. Since this class
    was developed to use a weighted loss function with Renyi Entropy, it has
    an attribute lam (lambda), to control the weight of the loss function, and
    alpha, to control renyi's entropy.
    
    Attributes
    ----------
    model: tf.keras.Model
        The tensorflow model to train.
    lam: float, default = 0.5
        The lambda value for the loss function.
    alpha: float | int, default = 2
        The renyi entropy alpha value to use.
    
    Methods
    ----------
    call(x)
        Equivalent to model.predict(x)
    train_step(data)
        Custom train function
    test_step(data)
        Custom test function
    
    """
    def __init__(self, model: tf.keras.Model, lam: float = 0.5, alpha: float | int = 2, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        if lam > 1 or lam < 0:
            raise ValueError("Lambda must be a value between 0 and 1 (inclusive).")
        self.lam = lam
        self.alpha = alpha

        #The loss trackers
        self.ce_loss_tracker = tf.keras.metrics.Mean(
            name="ce_loss",
            )
        self.kernel_loss_tracker = tf.keras.metrics.Mean(
            name="kernel_loss",
            )
        self.total_loss_tracker = tf.keras.metrics.Mean(
            name="total_loss"
            )


        #The validation loss trackers
        self.val_ce_loss_tracker = tf.keras.metrics.Mean(
            name="val_ce_loss",
            )
        self.val_kernel_loss_tracker = tf.keras.metrics.Mean(
            name="val_kernel_loss",
            )
        self.val_total_loss_tracker = tf.keras.metrics.Mean(
            name="val_total_loss"
            )

        #Metrics
        self.ca_tracker = tf.keras.metrics.CategoricalAccuracy(name = "CategoricalAcc")
        self.val_ca_tracker = tf.keras.metrics.CategoricalAccuracy(name = "val_CategoricalAcc")

    def call(self,
             x: tf.Tensor) -> tf.Tensor | tuple[tf.Tensor]:
        """Call the model.
        
        Equivalent to model.predict(x)
        
        Parameters
        ----------
        x : tf.Tensor
            The input data

        Returns
        -------
        tf.Tensor | tuple[tf.Tensor]
            The model's output. For single output models this is a tf.Tensor,
            otherwise it is a tuple of tf.Tensors with length equal to the number
            of outputs.
        """

        return self.model(x)

    def train_step(self,
                   data: tf.Tensor) -> dict:
        """Custom train step.
        
        This function defines how the model will train.
        In other words, this is the function that is called when model.fit() is used.

        Parameters
        ----------
        data : tf.Tensor
            The training data.

        Returns
        -------
        dict
            The updated metrics
        """
        try:
            x, y_true = data
            with tf.GradientTape() as tape:
                y_pred, kernel, weights = self.model(x)

                lam = self.lam

                kernel = tf.math.multiply_no_nan(kernel, tf.expand_dims(tf.expand_dims(weights, axis=-1), axis=-1))
                Z = normalize_matrix(kernel)
                Z_joint = tf.math.reduce_prod(Z, axis = 1) # Hadamard product along the features
                trace = tf.expand_dims(tf.expand_dims(tf.linalg.trace(Z_joint),-1),-1) # The trace, we expand dims so we can divide
                Z_joint = tf.math.divide(Z_joint, trace) # Divide the product by it's trace
                joint = renyi_entropy(Z_joint, self.alpha) # Joint entropy of the features
                #kernel_adj = tf.math.reduce_mean(Z, axis = 0)

                ce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)

                #tf.print(Z, joint)
                kernel_loss = tf.reduce_mean(tf.math.reduce_sum(renyi_entropy(Z,self.alpha))-joint)

                total_loss = (1-lam)*ce_loss + lam*kernel_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.ce_loss_tracker.update_state(ce_loss)
            self.kernel_loss_tracker.update_state(kernel_loss)
            self.ca_tracker.update_state(y_true, y_pred)

            losses = {
                    "total_loss": self.total_loss_tracker.result(),
                    "ce_loss": self.ce_loss_tracker.result(),
                    "kernel_loss": self.kernel_loss_tracker.result(),
                    "CategoricalAcc": self.ca_tracker.result()
                }
            return losses
        except Exception as e:
            tf.print(ce_loss)
            tf.print(kernel_loss)
            tf.print(kernel)
            tf.print(weights)

    def test_step(self,
                  data: tf.Tensor) -> dict:
        """
        Custom test step. This function excecuted during evaluation and in
        practice defines how the model will operate over the validation data.
        In other words, this is the function that is called when
        model.evaluate() is used.

        Parameters
        ----------
        data : tf.Tensor
            The training data.

        Returns
        -------
        dict
            The updated metrics
        """
        x, y_true = data

        y_pred, kernel, weights = self.model(x)

        lam = self.lam

        kernel = tf.math.multiply_no_nan(kernel, tf.expand_dims(tf.expand_dims(weights, axis=-1), axis=-1))
        Z = normalize_matrix(kernel)
        Z_joint = tf.math.reduce_prod(Z, axis = 1) # Hadamard product along the features
        trace = tf.expand_dims(tf.expand_dims(tf.linalg.trace(Z_joint),-1),-1) # The trace, we expand dims so we can divide
        Z_joint = tf.math.divide(Z_joint, trace) # Divide the product by it's trace
        joint = renyi_entropy(Z_joint, self.alpha) # Joint entropy of the features
        #kernel_adj = tf.math.reduce_mean(Z, axis = 0)

        ce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)

        #tf.print(Z, joint)
        kernel_loss = tf.reduce_mean(tf.math.reduce_sum(renyi_entropy(Z,self.alpha))-joint)

        total_loss = (1-lam)*ce_loss + lam*kernel_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_ce_loss_tracker.update_state(ce_loss)
        self.val_kernel_loss_tracker.update_state(kernel_loss)
        self.val_ca_tracker.update_state(y_true, y_pred)

        losses = {
                "total_loss": self.val_total_loss_tracker.result(),
                "ce_loss": self.val_ce_loss_tracker.result(),
                "kernel_loss": self.val_kernel_loss_tracker.result(),
                "CategoricalAcc": self.val_ca_tracker.result()
            }

        return losses