import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="Custom")
def compute_mask_layer(mask_value):
    def func(inp):
        return tf.cast(tf.reduce_any(inp != mask_value, axis=-1), tf.float32)
    return func

@tf.keras.utils.register_keras_serializable(package="Custom")
def squeeze_last_axis_func(t):
    return tf.squeeze(t, axis=-1)

@tf.keras.utils.register_keras_serializable(package="Custom")
def mask_attention_scores_func(inputs):
    scores, mask = inputs
    minus_inf = -1e9
    return scores + (1.0 - mask) * minus_inf

@tf.keras.utils.register_keras_serializable(package="Custom")
def apply_attention_func(inputs):
    x, attn = inputs
    return x * tf.expand_dims(attn, axis=-1)

@tf.keras.utils.register_keras_serializable(package="Custom")
class MaskedRepeatVector(tf.keras.layers.Layer):
    def __init__(self, n, **kwargs):
        super(MaskedRepeatVector, self).__init__(**kwargs)
        self.n = n
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # Standard repeat behavior
        outputs = tf.keras.backend.repeat(inputs, self.n)

        # If we have a mask, we need to repeat it as well
        if mask is not None:
            # The repeated mask will be True for all time steps if the original was True
            # This maintains the mask pattern across the repeated dimension
            self._mask = tf.keras.backend.repeat(tf.expand_dims(mask, axis=1), self.n)
        return outputs

    def compute_mask(self, inputs, mask=None):
        # Return the repeated mask
        if mask is None:
            return None
        return self._mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def get_config(self):
        config = {'n': self.n}
        base_config = super(MaskedRepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        # Create trainable weights for attention mechanism
        self.W = self.add_weight(name="attention_weight",
                                 shape=(input_shape[-1], 128),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="attention_bias",
                                 shape=(128,),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="context_vector",
                                 shape=(128, 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, time_steps, features)
        # Calculate attention hidden representation using tanh activation
        uit = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)

        # Calculate attention weights
        ait = tf.keras.backend.dot(uit, self.u)
        ait = tf.keras.backend.squeeze(ait, -1)  # Remove last dimension

        # Apply mask if provided (for padded sequences)
        if mask is not None:
            # Cast the mask to floatX to avoid issues
            mask = tf.keras.backend.cast(mask, tf.keras.backend.floatx())
            # Add a very small negative number to masked positions
            # This is more numerically stable than using -1e10
            ait += -10000.0 * (1.0 - mask)

        # Apply softmax to get normalized weights
        ait = tf.keras.backend.softmax(ait)

        # Reshape for multiplication
        ait = tf.keras.backend.expand_dims(ait, axis=-1)

        # Apply attention weights to input sequence
        weighted_input = inputs * ait

        # Sum over time dimension to get context vector
        output = tf.keras.backend.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        # Output shape is (batch_size, features)
        return (input_shape[0], input_shape[2])

    def get_config(self):
        # For serialization
        config = super(AttentionLayer, self).get_config()
        return config
@tf.keras.utils.register_keras_serializable()
def compute_mask_layer(mask_value):
    def func(inp):
        return tf.cast(tf.reduce_any(inp != mask_value, axis=-1), tf.float32)
    return func

@tf.keras.utils.register_keras_serializable()
def squeeze_last_axis_func(t):
    return tf.squeeze(t, axis=-1)

@tf.keras.utils.register_keras_serializable()
def mask_attention_scores_func(inputs):
    scores, mask = inputs
    minus_inf = -1e9
    return scores + (1.0 - mask) * minus_inf

@tf.keras.utils.register_keras_serializable()
def apply_attention_func(inputs):
    x, attn = inputs
    return x * tf.expand_dims(attn, axis=-1)

