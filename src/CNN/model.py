"""Các lớp và cách xây CNN TensorFlow của nhóm.

Notebook viết trực tiếp Sequential/add. Giữ BatchNorm và cách khởi tạo
của baseline đã chạy; không cần cài thư viện PyTorch.
"""

import math
import tensorflow as tf
from tensorflow.keras import initializers, layers, models


@tf.keras.utils.register_keras_serializable(package="CS231")
class TorchBatchNormalization(layers.Layer):
    """NHWC BatchNorm2d: biased batch variance, unbiased running variance.

    ``momentum`` follows PyTorch's convention: weight of the NEW statistics.
    Accumulate statistics in float32, including when convolution uses AMP.
    """

    def __init__(self, momentum=0.1, epsilon=1e-5, **kwargs):
        kwargs.setdefault("dtype", "float32")
        kwargs["autocast"] = False
        super().__init__(**kwargs)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones")
        self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros")
        self.moving_mean = self.add_weight(
            name="moving_mean", shape=(channels,), initializer="zeros", trainable=False
        )
        self.moving_variance = self.add_weight(
            name="moving_variance", shape=(channels,), initializer="ones", trainable=False
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)

        def train():
            mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
            count = tf.cast(tf.size(x) // tf.shape(x)[-1], tf.float32)
            tf.debugging.assert_greater(count, 1.0, message="BatchNorm needs >1 value/channel")
            running_variance = variance * count / (count - 1.0)
            self.moving_mean.assign((1.0 - self.momentum) * self.moving_mean + self.momentum * mean)
            self.moving_variance.assign(
                (1.0 - self.momentum) * self.moving_variance + self.momentum * running_variance
            )
            return tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, self.epsilon)

        def infer():
            return tf.nn.batch_normalization(
                x, self.moving_mean, self.moving_variance, self.beta, self.gamma, self.epsilon
            )

        if not self.trainable:
            training = False
        if isinstance(training, bool) or training is None:
            result = train() if training else infer()
        else:
            result = tf.cond(tf.cast(training, tf.bool), train, infer)
        return tf.cast(result, inputs.dtype)

    def get_config(self):
        return {**super().get_config(), "momentum": self.momentum, "epsilon": self.epsilon}


def conv_initializer():
    """Khởi tạo Conv từ phân phối chuẩn có phương sai 2/fan_out như baseline."""
    return initializers.VarianceScaling(
        scale=2.0, mode="fan_out", distribution="untruncated_normal"
    )


def build_model(config):
    channels = 1 if config["color"] == "gray" else 3
    model = models.Sequential(name=config["candidate_id"])
    model.add(layers.Input(shape=(224, 224, channels), name="images"))
    model.add(layers.Rescaling(1.0 / 255, dtype="float32", name="rescale"))
    # Augmentation runs on uint8 RGB in data.py, exactly before grayscale/255.
    for index, width in enumerate(config["channels"], 1):
        model.add(
            layers.Conv2D(
                width,
                (3, 3),
                padding="same",
                use_bias=False,
                kernel_initializer=conv_initializer(),
                name=f"conv{index}_1",
            )
        )
        model.add(TorchBatchNormalization(momentum=0.1, epsilon=1e-5, name=f"bn{index}_1"))
        model.add(layers.Activation("relu", name=f"relu{index}_1"))
        model.add(
            layers.Conv2D(
                width,
                (3, 3),
                padding="same",
                use_bias=False,
                kernel_initializer=conv_initializer(),
                name=f"conv{index}_2",
            )
        )
        model.add(TorchBatchNormalization(momentum=0.1, epsilon=1e-5, name=f"bn{index}_2"))
        model.add(layers.Activation("relu", name=f"relu{index}_2"))
        model.add(layers.MaxPooling2D((2, 2), name=f"pool{index}"))
        model.add(layers.SpatialDropout2D(0.1, name=f"drop{index}"))
    model.add(layers.GlobalAveragePooling2D(name="embedding"))
    model.add(
        layers.Dense(
            128,
            activation="relu",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="dense128",
        )
    )
    model.add(layers.Dropout(0.5, name="head_dropout"))
    model.add(
        layers.Dense(
            6,
            activation="softmax",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            dtype="float32",
            name="predictions",
        )
    )
    return model


def cosine_learning_rate(epoch_index, settings):
    """Epoch1 uses lr_max; epoch25 uses cosine at t=24, as in PyTorch."""
    return settings["eta_min"] + 0.5 * (settings["learning_rate"] - settings["eta_min"]) * (
        1.0 + math.cos(math.pi * epoch_index / settings["epochs"])
    )


def compile_model(model, settings):
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        beta_1=settings["betas"][0],
        beta_2=settings["betas"][1],
        epsilon=settings["epsilon"],
        global_clipnorm=settings["gradient_clip_norm"],
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


def inference_model(model):
    """Original PyTorch used AMP for training and float32 for dev/test.

    A Keras mixed policy is saved on each layer, so training=False alone does
    not switch convolution to float32. Clone only for inference and copy the
    exact weights; the training model/optimizer remain intact for checkpoints.
    """
    if all(layer.compute_dtype == "float32" for layer in model.layers):
        return model

    def float32_layer(layer):
        config = layer.get_config()
        config["dtype"] = "float32"
        return layer.__class__.from_config(config)

    result = models.clone_model(model, clone_function=float32_layer)
    result.set_weights(model.get_weights())
    return result
