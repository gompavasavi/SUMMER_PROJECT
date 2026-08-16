# fuzzy_layer.py

import tensorflow as tf

from tensorflow.keras import layers


class Type2Fuzzy(
    layers.Layer
):

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


    def build(
        self,
        input_shape
    ):

        c = int(
            input_shape[-1]
        )

        self.a = self.add_weight(
            name="a",
            shape=(c,),
            initializer=(
                tf.keras.initializers
                .Constant(0.5)
            ),
            trainable=True
        )

        self.P = self.add_weight(
            name="P",
            shape=(c,),
            initializer="ones",
            trainable=True
        )

        self.N = self.add_weight(
            name="N",
            shape=(c,),
            initializer="ones",
            trainable=True
        )

        super().build(
            input_shape
        )


    def _k(
        self,
        s,
        a
    ):

        a = tf.clip_by_value(
            a,
            0.02,
            0.98
        )

        eps = 1e-5

        d1 = (
            a + s - a * s
        )

        d2 = (
            -1.0 + a * s
        )

        d1 = tf.where(
            tf.abs(d1) < eps,
            tf.ones_like(d1) * eps,
            d1
        )

        d2 = tf.where(
            tf.abs(d2) < eps,
            -tf.ones_like(d2) * eps,
            d2
        )

        return 0.5 * (
            1.0 / d1
            +
            (-1.0 + a) / d2
        )


    def call(
        self,
        x
    ):

        sh = (
            [1, 1, 1, -1]
            if len(x.shape) == 4
            else [1, -1]
        )

        a = tf.clip_by_value(
            tf.reshape(
                self.a,
                sh
            ),
            0.02,
            0.98
        )

        P = tf.reshape(
            self.P,
            sh
        )

        N = tf.reshape(
            self.N,
            sh
        )

        return tf.where(
            x > 0,
            P * x * self._k(
                x,
                a
            ),
            N * x * self._k(
                -x,
                a
            )
        )


    def get_config(self):

        return super().get_config()
