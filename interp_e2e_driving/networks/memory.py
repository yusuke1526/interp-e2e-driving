from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts

from interp_e2e_driving.networks.sequential_latent_network import *

tfd = tfp.distributions


@gin.configurable
class MemoryModel(tf.Module):
    """rnn + sampling"""

    def __init__(
        self,
        latent_size,
        action_size,
        gaussian_mixtures,
        name=None,
        z_factor=0.1,
        rew_factor=0.1,
    ):
        super(MemoryModel, self).__init__(name=name)
        self.latent_size = latent_size
        self.action_size = action_size
        self.input_size = latent_size + action_size + 1
        self.gaussian_mixtures = gaussian_mixtures
        self.rnn = tf.keras.layers.LSTM(
            self.input_size, return_sequences=True, return_state=True
        )
        self.mdn = tf.keras.layers.Dense(
            gaussian_mixtures * (3 * latent_size) + 1
        )  # 3 is for log_pi, mu, log_sigma, 1 is for reward pred
        self.z_factor = z_factor
        self.rew_factor = rew_factor

        # p(z_{t+1}|h_t, z_t, a_t)
        # self.prior = self.init_prior_distribution()
        self.prior = tfd.MultivariateNormalDiag

    def get_rnn_output(
        self, input_z, input_action, prev_rew, state_input_h=None, state_input_c=None
    ):
        if (input_action is None) or prev_rew is None:
            input_action = tf.zeros(
                (input_z.shape[0], input_z.shape[1], self.action_size)
            )
            prev_rew = tf.zeros((input_z.shape[0], input_z.shape[1], 1))
        input = [input_z, input_action, prev_rew]

        for i in range(len(input)):
            if len(input[i].shape) == 1:
                input[i] = tf.expand_dims(input[i], axis=0)
            if len(input[i].shape) == 2:
                input[i] = tf.expand_dims(input[i], axis=1)

        input = tf.concat(input, axis=-1)
        if (state_input_h is None) or (state_input_c is None):
            rnn_output, state_h, state_c = self.rnn(input)
        else:
            rnn_output, state_h, state_c = self.rnn(
                input, initial_state=[state_input_h, state_input_c]
            )

        return rnn_output, state_h, state_c

    def pred(
        self,
        input_z,
        input_action=None,
        prev_rew=None,
        step_types=None,
        state_input_h=None,
        state_input_c=None,
        return_state=False,
    ):
        """
        q(z_{t+1}|z_t, a_t, r_t)
        """
        # sequence_length = step_types.shape[1] - 1 TODO: implement transition depending on initial state
        # reset_mask = tf.equal(step_types, ts.StepType.FIRST)
        # reset_mask = tf.expand_dims(reset_mask, axis=-1)

        rnn_output, state_h, state_c = self.get_rnn_output(
            input_z, input_action, prev_rew, state_input_h, state_input_c
        )
        mdn_output = self.mdn(rnn_output)

        z_pred = mdn_output[:, :, :-1]
        z_pred = tf.reshape(z_pred, (-1, 3 * self.gaussian_mixtures))

        log_pi, mu, log_sigma = self.get_mixture_coef(z_pred)

        rew_pred = mdn_output[:, :, -1:]

        if return_state:
            return (log_pi, mu, log_sigma), rew_pred, (state_h, state_c)

        return (log_pi, mu, log_sigma), rew_pred

    def compute_loss(self, y_pred, y_true):
        """
        args:
          y_pred: ((log_pi, mu, log_sigma), rew_pred)
          y_true: array of latent_size z and 1 reward
        """
        z_pred, rew_pred = y_pred
        z_loss = self.z_loss(z_pred, y_true)
        rew_loss = self.rew_loss(rew_pred, y_true)

        return self.z_factor * z_loss + self.rew_factor * rew_loss, z_loss, rew_loss

    def z_loss(self, y_pred, y_true):
        log_pi, mu, log_sigma = y_pred
        z_true = y_true[:, :, : self.latent_size]

        flat_z_true = tf.reshape(z_true, (-1, 1))

        z_loss = log_pi + self.tf_lognormal(flat_z_true, mu, log_sigma)
        z_loss = -tf.reduce_logsumexp(z_loss, axis=1, keepdims=True)

        z_loss = tf.reduce_mean(z_loss)
        # print(z_loss)
        return z_loss

    def rew_loss(self, y_pred, y_true):
        rew_pred = tf.sigmoid(y_pred)
        rew_true = tf.sigmoid(y_true[:, :, -1:])
        rew_loss = tf.keras.backend.binary_crossentropy(rew_true, rew_pred)

        rew_loss = tf.reduce_mean(rew_loss)

        return rew_loss

    def get_mixture_coef(self, z_pred):
        log_pi, mu, log_sigma = tf.split(z_pred, 3, 1)
        log_pi = log_pi - tf.reduce_logsumexp(
            log_pi, axis=1, keepdims=True
        )  # axis 1 is the mixture axis

        return log_pi, mu, log_sigma

    def tf_lognormal(self, z_true, mu, log_sigma):
        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        # print(f'z_true.shape: {z_true.shape}')
        # print(f'mu.shape: {mu.shape}')
        return (
            -0.5 * ((z_true - mu) / tf.math.exp(log_sigma)) ** 2
            - log_sigma
            - logSqrtTwoPI
        )

    def init_prior_distribution(self):
        return None  # TODO

    def sample_z(self, log_pis, mus, log_sigmas):
        """
        inputs: (latent_size, N)
        """
        idx = tfd.Categorical(logits=log_pis).sample()
        mus = self.extract(mus, idx)
        log_sigmas = self.extract(log_sigmas, idx)
        epsilon = tf.keras.backend.random_normal(
            shape=mus.shape[:0], mean=0.0, stddev=1.0
        )
        return mus + tf.math.exp(log_sigmas) * epsilon

    def extract(self, A, IDX):
        # A : 行列
        # IDX : index (dtype=tf.int32, int32でない場合はキャストが必要になることがある)
        _IDX = tf.concat(
            [tf.range(A.shape[0])[:, tf.newaxis], IDX[:, tf.newaxis]], axis=1
        )
        subA = tf.gather_nd(A, _IDX)
        return subA


if __name__ == "__main__":
    batch_size = 16

    # Memory Model
    latent_size = 32
    action_size = 1
    timesteps = 1
    gaussian_mixtures = 5

    memory = MemoryModel(
        latent_size=latent_size,
        action_size=action_size,
        gaussian_mixtures=gaussian_mixtures,
    )
    z = tf.random.normal([batch_size, timesteps, latent_size], 1, 1, tf.float32)
    a = tf.random.normal([batch_size, timesteps, action_size], 1, 1, tf.float32)
    h = tf.zeros(
        [batch_size, latent_size + action_size + 1], tf.float32
    )  # 1 is for reward
    c = tf.zeros([batch_size, latent_size + action_size + 1], tf.float32)
    prev_rew = tf.random.normal([batch_size, timesteps, 1], 1, 1, tf.float32)

    print(h.shape)
    print(z.shape)
    print(a.shape)
    print(memory)
    # input_z, input_action, state_input_h, state_input_c
    y_pred = memory.pred(z, a, prev_rew, h, c)
    (log_pi, mu, log_sigma), rew_pred = y_pred
    print((log_pi.shape, mu.shape, log_sigma.shape), rew_pred.shape)
    y_true = tf.random.normal(
        [batch_size, timesteps, latent_size + 1], 1, 1, tf.float32
    )
    memory.compute_loss(y_pred, y_true)
