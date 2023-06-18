from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import numpy as np
import collections

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts

from interp_e2e_driving.utils import nest_utils, gif_utils
from interp_e2e_driving.networks.sequential_latent_network import *
from interp_e2e_driving.networks.vision import VisionModel
from interp_e2e_driving.networks.memory import MemoryModel

tfd = tfp.distributions

WorldModelLossInfo = collections.namedtuple(
    'WorldModelLossInfo', ('model_loss'))


@gin.configurable
class WorldModel(tf.Module):

  def __init__(self,
               input_names,
               reconstruct_names,
               action_size,
               gaussian_mixtures=5,
               obs_size=64,
               base_depth=32,
               latent_size=64,
               kl_analytic=True,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               optimizer=None,
               train_step_counter=None,
               batch_size=256,
               num_images_per_summary=2,
               name=None):
    """Creates an instance of `WorldModel`.
    Args:
      input_names: the names of the observation inputs (e.g, 'camera', 'lidar').
      reconstruct_names: names of the outputs to reconstruct (e.g, 'mask').
      obs_size: the pixel size of the observation inputs. Here we assume
        the image inputs have same width and height.
      base_depth: base depth of the convolutional layers.
      latent_size: size of the latent state.
      kl_analytic: whether to use analytical KL divergence.
      decoder_stddev: standard deviation of the decoder.
      name: A string representing name of the network.
    """
    super(WorldModel, self).__init__(name=name)
    self.input_names = input_names
    self.reconstruct_names = reconstruct_names
    self.latent_size = latent_size
    self.kl_analytic = kl_analytic
    self.obs_size = obs_size
    self.action_size = action_size
    self.gaussian_mixtures = gaussian_mixtures
    self.optimizer = optimizer
    self.train_step_counter=train_step_counter
    self._model_batch_size = batch_size
    self._num_images_per_summary = num_images_per_summary

    latent_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
    latent_distribution_ctor = MultivariateNormalDiag

    # p(z_1)
    self.latent_first_prior = latent_first_prior_distribution_ctor(latent_size)
    # p(z_{t+1} | z_t, a_t)
    self.memory = MemoryModel(latent_size, action_size, gaussian_mixtures)

    self.vision = VisionModel(input_names, reconstruct_names, obs_size, latent_size)

  def sample_prior(self, batch_size):
    """Sample the prior latent state."""
    latent = self.latent_first_prior(tf.zeros(batch_size)).sample()
    return latent

  def filter(self, image, last_latent, last_action):
    """Apply recursive filter to obtain posterior estimation of latent 
      q(z_{t+1}|z_t,a_t,x_{t+1}).
    """
    feature = self.get_features(image)
    latent = self.latent_posterior(feature, last_latent, last_action).sample()
    return latent

  def first_filter(self, image):
    """Obtain the posterior of the latent at the first timestep q(z_1|x_1)."""
    feature = self.get_features(image)
    latent = self.latent_first_posterior(feature).sample()
    return latent

  def get_features(self, images):
    """Get low dimensional features from images q(f_t|x_t)"""
    features = {}
    for name in self.input_names:
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      features[name] = self.vision.encoders[name](images_tmp)
    features = sum(features.values())
    return features

  def reconstruct(self, latent):
    """Reconstruct the images in reconstruct_names given the latent state."""
    posterior_images = {}
    for name in self.reconstruct_names:
      posterior_images[name] = self.vision.decoders[name](latent).mean()
    posterior_images = tf.concat(list(posterior_images.values()), axis=-2)
    return posterior_images

  def compute_loss(self, images, actions, rewards, step_types, latent_posterior_samples_and_dists=None):
    '''
      images: dict of image: (B, sequence_length+1, h, w, c)
      actions: (B, sequence_length+1, action_size)
      rewards: (B, sequence_length+1, 1) or (B, sequence_length+1)
      step_types: (B, sequence_length+1)
    '''

    next_images = {name: image_sequence[:, 1:] for name, image_sequence in images.items()}
    images = {name: image_sequence[:, :-1] for name, image_sequence in images.items()}

    next_actions = actions[:, 1:]
    actions = actions[:, :-1]

    if len(rewards.shape) == 2:
      rewards = tf.expand_dims(rewards, axis=-1)
    next_rewards = rewards[:, 1:]
    rewards = rewards[:, :-1]

    # Compuate the latents
    next_z_means, next_z_log_vars, next_zs = self.vision.encode_sequence(next_images)
    z_means, z_log_vars, zs = self.vision.encode_sequence(images)

    # Compute the vision loss
    outputs = {}
    kl_divergence, reconstruction_errors, vision_loss = self.vision.compute_sequence_loss(images)
    outputs.update({
      'kl_divergence': kl_divergence,
    })
    for name in self.reconstruct_names:
      outputs.update({
        'reconstruction_error_'+name: tf.reduce_mean(reconstruction_errors[name]),
      })
    loss = vision_loss

    # Compute the memory loss
    z_preds = self.memory.pred(
      input_z=zs,
      input_action=actions,
      prev_rew=rewards,
      step_types=step_types,
    )
    z_true = tf.concat([next_zs, next_rewards], axis=-1)
    memory_loss, z_loss, rew_loss = self.memory.compute_loss(z_preds, z_true)
    outputs.update({
      'memory_loss': memory_loss,
      'z_loss': z_loss,
      'rew_loss': rew_loss
    })
    loss += memory_loss

    # Generate the images #TODO
    posterior_images = {}
    # conditional_prior_images = {}
    posterior_images = self.vision.decode_sequence(zs)
      # conditional_prior_images[name] = self.decoders[name](latent_conditional_prior_samples).mean()

    images = tf.concat([tf.image.convert_image_dtype(images[k], tf.float32)
      for k in list(set(self.input_names+self.reconstruct_names))], axis=-2)
    posterior_images = tf.concat(list(posterior_images.values()), axis=-2)
    # conditional_prior_images = tf.concat(list(conditional_prior_images.values()), axis=-2)

    outputs.update({
      'total_loss': loss,
      'images': images,
      'posterior_images': posterior_images, #TODO p(x|z~q(z|x))からサンプリングした画像の列
      # 'conditional_prior_images': conditional_prior_images, #TODO 条件付き事前分布p(z_1|x_1)からサンプリングして再構成した画像の列
    })
    return loss, outputs

  def train(self, experience, weights=None, train_flag={'vision': True, 'memory': False, 'controller': False}):
    """Train world model except for Controller"""
    with tf.GradientTape() as tape:

      model_loss = self.model_loss(
          experience.observation,
          experience.action,
          experience.reward,
          experience.step_type,
          latent_posterior_samples_and_dists=None,
          weights=weights)
      
    # print(f'self.vision.trainable_variables: {self.vision.trainable_variables}')
    # print(f'self.memory.trainable_variables: {self.memory.trainable_variables}')
      
    # which module trained
    self.vision.trainable = train_flag['vision']
    self.memory.trainable = train_flag['memory']
    # self.controller.trainable = train_flag['controller']

    tf.debugging.check_numerics(model_loss, 'Model loss is inf or nan.')
    # trainable_model_variables = self.trainable_variables
    trainable_model_variables = []
    if train_flag['vision']: trainable_model_variables += self.vision.trainable_variables
    if train_flag['memory']: trainable_model_variables += self.memory.trainable_variables
    assert trainable_model_variables, ('No trainable model variables to '
                                          'optimize.')
    model_grads = tape.gradient(model_loss, trainable_model_variables)
    self.optimizer.apply_gradients(zip(model_grads, trainable_model_variables))

    self.train_step_counter.assign_add(1)

    return model_loss
  
  def model_loss(self,
                 images,
                 actions,
                 rewards,
                 step_types,
                 latent_posterior_samples_and_dists=None,
                 weights=None):
      with tf.name_scope('model_loss'):
        # print(f'images: {images["camera"].shape}')
        # print(f'actions: {actions.shape}')
        model_loss, outputs = self.compute_loss(
            images, actions, rewards, step_types,
            latent_posterior_samples_and_dists=latent_posterior_samples_and_dists)
        for name, output in outputs.items():
          if output.shape.ndims == 0:
            tf.summary.scalar(name, output, step=self.train_step_counter)
          elif output.shape.ndims == 5:
            output = output[:self._num_images_per_summary]
            output = tf.transpose(output, [1,0,2,3,4])
            output = tf.reshape(output, [output.shape[0], output.shape[1]*output.shape[2], output.shape[3], output.shape[4]])
            output = tf.expand_dims(output, axis=0)
            gif_utils.gif_summary(name, output, fps=10,
                         saturate=True, step=self.train_step_counter)
          else:
            raise NotImplementedError

        if weights is not None:
          model_loss *= weights

        model_loss = tf.reduce_mean(input_tensor=model_loss)

        return model_loss
      
  def sample_z_sequence(self, images, actions, rewards, step_types):
    batch_size = step_types.shape[0]
    sequence_length = step_types.shape[1] - 1
    images = {name: image_sequence[:, :sequence_length] for name, image_sequence in images.items()}

    actions = actions[:, :sequence_length]

    if len(rewards.shape) == 2:
      rewards = tf.expand_dims(rewards, axis=-1)
    rewards = rewards[:, :sequence_length]

    _, _, zs = self.vision.encode_sequence(images)
    (log_pi, mu, log_sigma), rew_pred = self.memory.pred(
      input_z=zs,
      input_action=actions,
      prev_rew=rewards,
      step_types=step_types
    )
    z_preds = self.memory.sample_z(log_pi, mu, log_sigma)
    z_preds = tf.reshape(z_preds, (batch_size, sequence_length, self.latent_size))
    
    return tf.concat([z_preds, rew_pred], axis=-1)
  
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
  # print(z)

  # Memory Model
  # latent_size = 32
  # action_size = 1
  # timesteps = 1
  # gaussian_mixtures = 5

  # memory = MemoryModel(latent_size=latent_size, action_size=action_size, gaussian_mixtures=gaussian_mixtures)
  # z = tf.random.normal([batch_size, timesteps, latent_size], 1, 1, tf.float32)
  # a = tf.random.normal([batch_size, timesteps, action_size], 1, 1, tf.float32)
  # h = tf.zeros([batch_size, latent_size + action_size + 1], tf.float32) # 1 is for reward
  # c = tf.zeros([batch_size, latent_size + action_size + 1], tf.float32)
  # prev_rew = tf.random.normal([batch_size, timesteps, 1], 1, 1, tf.float32)

  # print(h.shape)
  # print(z.shape)
  # print(a.shape)
  # print(memory)
  # # input_z, input_action, state_input_h, state_input_c
  # y_pred = memory.pred(z, a, prev_rew, h, c)
  # (log_pi, mu, log_sigma), rew_pred = y_pred
  # print((log_pi.shape, mu.shape, log_sigma.shape), rew_pred.shape)
  # y_true = tf.random.normal([batch_size, timesteps, latent_size + 1], 1, 1, tf.float32)
  # memory.compute_loss(y_pred, y_true)