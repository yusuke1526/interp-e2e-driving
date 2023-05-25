from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts

tfd = tfp.distributions


@gin.configurable
class Controller(tf.Module):
  """Controller with 2 layers MLP
  """

  def __init__(self, latent_size, action_size, name=None):
    super(Controller, self).__init__(name=name)
    self.latent_size = latent_size
    self.action_size = action_size

    self.dense1 = tf.keras.layers.Dense(action_size * 2)
    self.dense2 = tf.keras.layers.Dense(action_size)

  def __call__(self, latent):
    h = self.dense1(latent)
    h = self.dense2(h)
    action = tf.math.tanh(h)
    return action
  
  def get_z_dist(self, images):
    features = {}
    for name, encoder in self.encoders.items():
        image_tmp = images[name]
        if image_tmp.dtype != tf.float32:
          images_tmp = tf.image.convert_image_dtype(image_tmp, tf.float32)
        features[name] = encoder(images_tmp)
    # feature = tf.concat(list(features.values()), axis=-1)
    feature = sum(features.values())
    z_mean = self.vae_z_mean(feature)
    z_log_var = self.vae_z_log_var(feature)
    return z_mean, z_log_var
  
  def encode_sequence(self, sequence_dict: dict):
    '''
    args:
      sequence_dict: {name: (B, T, w, h, c)}
    return:
      z_means: (B, T, latent_size)
      z_log_vars: (B, T, latent_size)
      z: (B, T, latent_size)
    '''
    images = self.reshape_sequence_dict(sequence_dict)
    image_sequence_shape = list(sequence_dict[self.input_names[0]].shape)
    latent_shape = image_sequence_shape[:2] + [self.latent_size]
    z_mean, z_log_var, z = self.encode(images)
    return map(lambda x: tf.reshape(x, latent_shape), [z_mean, z_log_var, z])

  def decode_sequence(self, z_sequence):
    '''
    args:
      z_sequence: (B, T, latent_size)
    return:
      sequence_dict: {name: (B, T, w, h, c)}
    '''
    latent_shape = list(z_sequence.shape)
    aligned = tf.reshape(z_sequence, [latent_shape[0]*latent_shape[1], latent_shape[-1]])
    reconstructions = self.decode(aligned)
    image_shape = list(reconstructions[self.reconstruct_names[0]].shape[1:])
    return {k: tf.reshape(v, latent_shape[:2] + image_shape) for k, v in reconstructions.items()}

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
  
  def reshape_sequence_dict(self, sequence_dict):
    sequence_shape = list(sequence_dict[self.input_names[0]].shape)
    image_shape = sequence_shape[2:]
    images = {k: tf.reshape(v, [-1] + image_shape) for k, v in sequence_dict.items()}
    return images

  def compute_sequence_loss(self, sequence_dict):
    '''
    args:
      sequence_dict: {name: (B, T, w, h, c)}
    return:
      kl_loss, reconstruction_loss, total_loss
    '''
    images = self.reshape_sequence_dict(sequence_dict)
    return self.compute_loss(images)
  
  def compute_loss(self, images):
    z_mean, z_log_var, z = self.encode(images)
    reconstruction = self.decode(z)
    reconstruction_losses = {}
    for name in self.reconstruct_names:
      reconstruction_losses[name] = tf.reduce_mean(
          tf.square(tf.image.convert_image_dtype(images[name], tf.float32)
                    - reconstruction[name]), axis = [1,2,3]
      )
    reconstruction_loss = tf.reduce_mean(tf.stack(list(reconstruction_losses.values()), axis=-1), axis=-1)
    reconstruction_loss *= self.r_loss_factor
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis = 1)
    kl_loss *= -0.5
    kl_loss *= self.kl_loss_factor
    total_loss = reconstruction_loss + kl_loss
    return tf.reduce_mean(kl_loss), reconstruction_losses, tf.reduce_mean(total_loss)

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
