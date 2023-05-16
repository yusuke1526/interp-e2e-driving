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
class VisionModel(tf.Module):
  """VAE
  """

  def __init__(self, input_names, reconstruct_names, obs_size, latent_size, r_loss_factor=0.1, base_depth=32, decoder_stddev=np.sqrt(0.1, dtype=np.float32), name=None):
    super(VisionModel, self).__init__(name=name)
    self.input_names = input_names
    self.reconstruct_names = reconstruct_names
    self.obs_size = obs_size
    self.base_depth = base_depth
    self.latent_size = latent_size
    self.r_loss_factor= r_loss_factor

    # Create encoders q(f_t|x_t)
    self.encoders = {}
    for name in self.input_names:
      if obs_size == 64:
        self.encoders[name] = Encoder64(base_depth, 8 * base_depth)
      elif obs_size == 128:
        self.encoders[name] = Encoder128(base_depth, 8 * base_depth)
      elif obs_size == 256:
        self.encoders[name] = Encoder256(base_depth, 8 * base_depth)
      else:
        raise NotImplementedError

    # Create decoders q(x_t|z_t)
    self.decoders = {}
    for name in self.reconstruct_names:
      if obs_size == 64:
        self.decoders[name] = Decoder64(base_depth, scale=decoder_stddev)
      elif obs_size == 128:
        self.decoders[name] = Decoder128(base_depth, scale=decoder_stddev)
      elif obs_size == 256:
        self.decoders[name] = Decoder256(base_depth, scale=decoder_stddev)
      else:
        raise NotImplementedError

    self.vae_z_mean = tf.keras.layers.Dense(self.latent_size)
    self.vae_z_log_var = tf.keras.layers.Dense(self.latent_size)

  def __call__(self, images):
    z_mean, z_log_var, z = self.encode(images)
    reconstructions = self.decode(z)

    return z, reconstructions
  
  def get_z_dist(self, images):
    features = {}
    for name, encoder in self.encoders.items():
        features[name] = encoder(images[name])
    feature = tf.concat(list(features.values()), axis=-1)
    z_mean = self.vae_z_mean(feature)
    z_log_var = self.vae_z_log_var(feature)
    return z_mean, z_log_var

  def encode(self, images):
    z_mean, z_log_var = self.get_z_dist(images)
    z = self.sample_z(z_mean, z_log_var)
    return z_mean, z_log_var, z
  
  def decode(self, z):
    reconstructions = {}
    for name, decoder in self.decoders.items():
        reconstructions[name] = decoder(z).mean()
        # print(f'{name}: {reconstructions[name].shape}')
    return reconstructions
  
  def init_prior_distribution(self):
    return None #TODO
  
  def sample_z(self, mu, log_var):
    epsilon = tf.keras.backend.random_normal(shape=mu.shape, mean=0., stddev=1.)
    return mu + tf.math.exp(log_var / 2) * epsilon
  
  def compute_loss(self, images):
    z_mean, z_log_var, z = self.encode(images)
    reconstruction = self.decode(z)
    reconstruction_losses = []
    for name in self.reconstruct_names:
      reconstruction_losses.append(tf.reduce_mean(
          tf.square(images[name] - reconstruction[name]), axis = [1,2,3]
      ))
    reconstruction_loss = tf.stack(reconstruction_losses, axis=-1)
    reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=-1)
    reconstruction_loss *= self.r_loss_factor
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis = 1)
    kl_loss *= -0.5
    total_loss = reconstruction_loss + kl_loss
    return tf.reduce_mean(kl_loss), tf.reduce_mean(reconstruction_loss), tf.reduce_mean(total_loss)

if __name__ == '__main__':
  batch_size = 16

  # Vision Model
  params = {
    'base_depth': 32,
    'input_names': ['rgb'],
    'reconstruct_names': ['rgb', 'mask'],
    'obs_size': 128,
    'latent_size': 32,
  }
  rgb = tf.ones([batch_size, params['obs_size'], params['obs_size'], 3], tf.float32) * 10
  mask = tf.ones([batch_size, params['obs_size'], params['obs_size'], 3], tf.float32) * 100
  images = {'rgb': rgb, 'mask': mask}
  vision = VisionModel(**params)
  z, reconstructions = vision(images)
  # print(z.shape)
  # print(reconstructions['rgb'].shape)
  loss = vision.compute_loss(images)
  # print(loss)
  z_mean, z_log_var, z = vision.encode(images)
  print(z)
