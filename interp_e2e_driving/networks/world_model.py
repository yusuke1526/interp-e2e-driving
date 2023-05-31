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

  def compute_latents(self, images, actions, step_types, latent_posterior_samples_and_dists=None):
    """Compute the latent states of the sequential latent model."""
    sequence_length = step_types.shape[1] - 1

    # Get posterior and prior samples of latents
    if latent_posterior_samples_and_dists is None:
      latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types)
    latent_posterior_samples, latent_posterior_dists = latent_posterior_samples_and_dists
    latent_prior_samples, _ = self.sample_prior_or_posterior(actions, step_types)  # for visualization

    # Get prior samples of latents conditioned on intial inputs
    first_image = {}
    num_first_image = 3
    for k,v in images.items():
      first_image[k] = v[:, :num_first_image]
    latent_conditional_prior_samples, _ = self.sample_prior_or_posterior(
        actions, step_types, images=first_image)  # for visualization. condition on first image only

    # Reset the initial steps of an episode to first prior latents
    def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
      after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
      prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
      return prior_tensors

    reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
                             tf.equal(step_types[:, 1:], ts.StepType.FIRST)], axis=1)

    latent_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent_size])
    latent_first_prior_dists = self.latent_first_prior(step_types)
    # these distributions start at t=1 and the inputs are from t-1
    latent_after_first_prior_dists = self.latent_prior(
        latent_posterior_samples[:, :sequence_length], actions[:, :sequence_length])
    latent_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent_reset_masks),
        latent_first_prior_dists,
        latent_after_first_prior_dists)

    return (latent_posterior_dists, latent_prior_dists), (latent_posterior_samples,
      latent_prior_samples, latent_conditional_prior_samples)

  def compute_loss(self, images, actions, step_types, latent_posterior_samples_and_dists=None):
    '''
      images: dict of image: (B, sequence_length+1, h, w, c)
      actions: (B, sequence_length+1, 2)
      step_types: (B, sequence_length+1)
    '''
    next_images = {name: image_sequence[:, 1:] for name, image_sequence in images.items()}
    images = {name: image_sequence[:, :-1] for name, image_sequence in images.items()}
    next_actions = actions[:, 1:]
    actions = actions[:, :-1]
    rewards = tf.zeros([actions.shape[0], actions.shape[1], 1], tf.float32) #TODO rewardを引っ張ってこれるようにする
    next_rewards = tf.zeros([actions.shape[0], actions.shape[1], 1], tf.float32)
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
    # z_preds = self.memory.pred_sequence(zs, actions, rewards, None, None)
    # z_true = tf.concat([next_zs, next_rewards], axis=-1)
    # memory_loss = self.memory.compute_sequence_loss(z_preds, z_true)
    # outputs.update({
    #   'memory loss': memory_loss,
    # })
    # loss += memory_loss

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

  def sample_prior_or_posterior(self, actions, step_types=None, images=None):
    """Samples from the prior latents, or latents conditioned on input images."""
    if step_types is None:
      batch_size = tf.shape(actions)[0]
      sequence_length = actions.shape[1]  # should be statically defined
      step_types = tf.fill(
          [batch_size, sequence_length + 1], ts.StepType.MID)
    else:
      sequence_length = step_types.shape[1] - 1
      actions = actions[:, :sequence_length]
    if images is not None:
      features = self.get_features(images)

    # Swap batch and time axes
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])
    if images is not None:
      features = tf.transpose(features, [1, 0, 2])

    # Get latent distributions and samples
    latent_dists = []
    latent_samples = []
    for t in range(sequence_length + 1):
      is_conditional = images is not None and (t < list(images.values())[0].shape[1])
      if t == 0:
        if is_conditional:
          latent_dist = self.latent_first_posterior(features[t])
        else:
          latent_dist = self.latent_first_prior(step_types[t])  # step_types is only used to infer batch_size
        latent_sample = latent_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        reset_mask = tf.expand_dims(reset_mask, 1)
        if is_conditional:
          latent_first_dist = self.latent_first_posterior(features[t])
          latent_dist = self.latent_posterior(features[t], latent_samples[t-1], actions[t-1])
        else:
          latent_first_dist = self.latent_first_prior(step_types[t])
          latent_dist = self.latent_prior(latent_samples[t-1], actions[t-1])
        latent_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent_first_dist, latent_dist)
        latent_sample = latent_dist.sample()

      latent_dists.append(latent_dist)
      latent_samples.append(latent_sample)

    latent_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent_dists)
    latent_samples = tf.stack(latent_samples, axis=1)
    return latent_samples, latent_dists

  def sample_posterior(self, images, actions, step_types, features=None):
    """Sample posterior latents conditioned on input images."""
    sequence_length = step_types.shape[1] - 1
    actions = actions[:, :sequence_length]

    if features is None:
      features = self.get_features(images)

    # Swap batch and time axes
    features = tf.transpose(features, [1, 0, 2])
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])

    # Get latent distributions and samples
    latent_dists = []
    latent_samples = []
    for t in range(sequence_length + 1):
      if t == 0:
        latent_dist = self.latent_first_posterior(features[t])
        latent_sample = latent_dist.sample()
      else:
        reset_mask = tf.equal(step_types[t], ts.StepType.FIRST)
        reset_mask = tf.expand_dims(reset_mask, 1)
        latent_first_dist = self.latent_first_posterior(features[t])
        latent_dist = self.latent_posterior(features[t], latent_samples[t-1], actions[t-1])
        latent_dist = nest_utils.map_distribution_structure(
            functools.partial(tf.where, reset_mask), latent_first_dist, latent_dist)
        latent_sample = latent_dist.sample()

      latent_dists.append(latent_dist)
      latent_samples.append(latent_sample)

    latent_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent_dists)
    latent_samples = tf.stack(latent_samples, axis=1)
    return latent_samples, latent_dists
  
  def train(self, experience, weights=None, train_flag={'vision': True, 'memory': False, 'controller': False}):
    """Train world model except for Controller"""
    
    with tf.GradientTape() as tape:

      model_loss = self.model_loss(
          experience.observation,
          experience.action,
          experience.step_type,
          latent_posterior_samples_and_dists=None,
          weights=weights)
      
    # which module trained
    self.vision.trainable = train_flag['vision']
    self.memory.trainable = train_flag['memory']
    # self.controller.trainable = train_flag['controller']

    tf.debugging.check_numerics(model_loss, 'Model loss is inf or nan.')
    trainable_model_variables = self.trainable_variables
    assert trainable_model_variables, ('No trainable model variables to '
                                          'optimize.')
    model_grads = tape.gradient(model_loss, trainable_model_variables)
    self.optimizer.apply_gradients(zip(model_grads, trainable_model_variables))

    self.train_step_counter.assign_add(1)

    return model_loss
  
  def model_loss(self,
                 images,
                 actions,
                 step_types,
                 latent_posterior_samples_and_dists=None,
                 weights=None):
      with tf.name_scope('model_loss'):
        if self._model_batch_size is not None:
          actions, step_types = tf.nest.map_structure(
              lambda x: x[:self._model_batch_size],
              (actions, step_types))
          images_new = {}
          for k, v in images.items():
            images_new[k] = v[:self._model_batch_size]

        model_loss, outputs = self.compute_loss(
            images_new, actions, step_types,
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