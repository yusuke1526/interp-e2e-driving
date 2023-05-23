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
  """rnn + sampling
  """

  def __init__(self, latent_size, action_size, gaussian_mixtures, name=None, z_factor=0.1, rew_factor=0.1):
    super(MemoryModel, self).__init__(name=name)
    self.latent_size = latent_size
    self.action_size = action_size
    self.input_size = latent_size + action_size + 1
    self.gaussian_mixtures = gaussian_mixtures
    self.rnn = tf.keras.layers.LSTM(self.input_size, return_sequences=True, return_state=True)
    self.mdn = tf.keras.layers.Dense(gaussian_mixtures * (3*latent_size) + 1)# 3 is for log_pi, mu, log_sigma, 1 is for reward pred
    self.z_factor = z_factor
    self.rew_factor = rew_factor

    # p(z_{t+1}|h_t, z_t, a_t)
    # self.prior = self.init_prior_distribution()
    self.prior = tfd.MultivariateNormalDiag

  def pred_sequence(self, input_zs, input_actions, prev_rews, state_input_hs=None, state_input_cs=None):
    '''
    args:
      input_zs: (B, T, latent_size),
      input_actions: (B, T, action_size),
      prev_rews: (B, T, 1),
      state_input: (B, T, latent_size+action_size+1)

    return:
      z_preds: (
        log_pis: (B, T, latent_size),
        mus: (B, T, latent_size),
        log_sigmas: (B, T, latent_size)
        ),
        rew_pred: (B, T, 1)
    '''
    batch_size, sequence_length = list(input_zs.shape[:2])
    size = batch_size * sequence_length
    reshape = functools.partial(tf.reshape, shape=(size, -1))
    input_zs = reshape(input_zs)
    input_actions = reshape(input_actions)
    prev_rews = reshape(prev_rews)
    state_input_hs = reshape(state_input_hs) if state_input_hs else None
    state_input_cs = reshape(state_input_cs) if state_input_cs else None

    dists, rew_pred = self.pred(input_zs, input_actions, prev_rews, state_input_hs, state_input_cs)
    reshape = functools.partial(tf.reshape, shape=(batch_size, sequence_length, -1))
    return tuple(map(reshape, dists)), reshape(rew_pred)

  def pred(self, input_z, input_action, prev_rew, state_input_h=None, state_input_c=None):
    input = tf.concat((input_z, input_action, prev_rew), axis=-1)
    if state_input_h is not None and state_input_c is not None:
      rnn_output, state_h, state_c = self.rnn(input, initial_state=[state_input_h, state_input_c])
    else:
      rnn_output, state_h, state_c = self.rnn(input)
    # print(f'rnn_output.shape: {rnn_output.shape}')
    mdn_output = self.mdn(rnn_output)
    # print(f'mdn_output.shape: {mdn_output.shape}')

    z_pred = mdn_output[:, :, :-1]
    z_pred = tf.reshape(z_pred, (-1, 3 * self.gaussian_mixtures))

    log_pi, mu, log_sigma = self.get_mixture_coef(z_pred)    

    rew_pred = mdn_output[:, :, -1]

    return (log_pi, mu, log_sigma), rew_pred
  
  def compute_sequence_loss(self, y_preds, y_trues):
    '''
    args:
      y_preds: (
        log_pis: (B, T, latent_size),
        mus: (B, T, latent_size),
        log_sigmas: (B, T, latent_size)
        ),
        rew_pred: (B, T, 1)
      y_trues: (B, T, latent_size + 1)
    return:
      loss
    '''
    rew_preds = tf.reshape(y_preds[-1], (-1, 1))
    dists = tuple(map(lambda x: tf.reshape(x, (-1, self.latent_size)), y_preds[0]))
    return self.compute_loss(dists, rew_preds)


  def compute_loss(self, y_pred, y_true):
    '''
    args:
      y_pred: ((log_pi, mu, log_sigma), rew_pred)
      y_true: array of latent_size z and 1 reward
    '''
    z_pred, rew_pred = y_pred
    z_loss = self.z_loss(z_pred, y_true)
    rew_loss = self.rew_loss(rew_pred, y_true)

    return self.z_factor * z_loss + self.rew_factor * rew_loss

  def z_loss(self, y_pred, y_true):
    log_pi, mu, log_sigma = y_pred
    z_true = y_true[:, :, :self.latent_size]

    flat_z_true = tf.reshape(z_true,(-1, 1))

    z_loss = log_pi + self.tf_lognormal(flat_z_true, mu, log_sigma)
    z_loss = -tf.reduce_logsumexp(z_loss, axis = 1, keepdims=True)

    z_loss = tf.reduce_mean(z_loss)
    # print(z_loss)
    return z_loss

  def rew_loss(self, y_pred, y_true):
    rew_pred = y_pred
    rew_true = y_true[:, :, -1]
    rew_loss =  tf.keras.metrics.binary_crossentropy(rew_true, rew_pred, from_logits = True)
    
    rew_loss = tf.reduce_mean(rew_loss)

    return rew_loss

  def get_mixture_coef(self, z_pred):
    log_pi, mu, log_sigma = tf.split(z_pred, 3, 1)
    log_pi = log_pi - tf.reduce_logsumexp(log_pi, axis = 1, keepdims = True) # axis 1 is the mixture axis

    return log_pi, mu, log_sigma

  def tf_lognormal(self, z_true, mu, log_sigma):
    logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
    # print(f'z_true.shape: {z_true.shape}')
    # print(f'mu.shape: {mu.shape}')
    return -0.5 * ((z_true - mu) / tf.math.exp(log_sigma)) ** 2 - log_sigma - logSqrtTwoPI
  
  def init_prior_distribution(self):
    return None #TODO
  
  def sample_z(self, mdn_output):
    '''
    input: (B, gaussian_N, 3, latent_size)
    '''

    return self.prior(loc=mean, scale_diag=std).sample()
  
if __name__ == '__main__':
  batch_size = 16

  # Memory Model
  latent_size = 32
  action_size = 1
  timesteps = 1
  gaussian_mixtures = 5

  memory = MemoryModel(latent_size=latent_size, action_size=action_size, gaussian_mixtures=gaussian_mixtures)
  z = tf.random.normal([batch_size, timesteps, latent_size], 1, 1, tf.float32)
  a = tf.random.normal([batch_size, timesteps, action_size], 1, 1, tf.float32)
  h = tf.zeros([batch_size, latent_size + action_size + 1], tf.float32) # 1 is for reward
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
  y_true = tf.random.normal([batch_size, timesteps, latent_size + 1], 1, 1, tf.float32)
  memory.compute_loss(y_pred, y_true)