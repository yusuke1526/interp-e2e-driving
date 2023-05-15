from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import time_step as ts

from interp_e2e_driving.utils import nest_utils
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
      
    # self.vae_z_means = {name: tf.keras.layers.Dense(self.latent_size) for name in self.input_names}
    # self.vae_z_log_vars = {name: tf.keras.layers.Dense(self.latent_size) for name in self.input_names}

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
        print(f'{name}: {reconstructions[name].shape}')
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

  def pred(self, input_z, input_action, prev_rew, state_input_h, state_input_c):
    input = tf.concat((input_z, input_action, prev_rew), axis=-1)
    rnn_output, state_h, state_c = self.rnn(input, initial_state=[state_input_h, state_input_c])
    print(f'rnn_output.shape: {rnn_output.shape}')
    mdn_output = self.mdn(rnn_output)
    print(f'mdn_output.shape: {mdn_output.shape}')

    z_pred = mdn_output[:, :, :-1]
    z_pred = tf.reshape(z_pred, (-1, 3 * self.gaussian_mixtures))

    log_pi, mu, log_sigma = self.get_mixture_coef(z_pred)    

    rew_pred = mdn_output[:, :, -1]

    return (log_pi, mu, log_sigma), rew_pred
  
  def compute_loss(self, y_pred, y_true):
    '''
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
    print(z_loss)
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
    print(f'z_true.shape: {z_true.shape}')
    print(f'mu.shape: {mu.shape}')
    return -0.5 * ((z_true - mu) / tf.math.exp(log_sigma)) ** 2 - log_sigma - logSqrtTwoPI
  
  def init_prior_distribution(self):
    return None #TODO
  
  def sample_z(self, mdn_output):
    '''
    input: (B, gaussian_N, 3, latent_size)
    '''

    return self.prior(loc=mean, scale_diag=std).sample()

@gin.configurable
class WorldModel(tf.Module):

  def __init__(self,
               input_names,
               reconstruct_names,
               obs_size=64,
               base_depth=32,
               latent_size=64,
               kl_analytic=True,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
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

    latent_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
    latent_distribution_ctor = MultivariateNormalDiag

    # p(z_1)
    self.latent_first_prior = latent_first_prior_distribution_ctor(latent_size)
    # p(z_{t+1} | z_t, a_t)
    self.memory = MemoryModel(latent_size)

    self.vision = VisionModel()

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
      features[name] = self.encoders[name](images_tmp)
    features = sum(features.values())
    return features

  def reconstruct(self, latent):
    """Reconstruct the images in reconstruct_names given the latent state."""
    posterior_images = {}
    for name in self.reconstruct_names:
      posterior_images[name] = self.decoders[name](latent).mean()
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
    # Compuate the latents
    latent_dists, latent_samples = self.compute_latents(images, actions, step_types, latent_posterior_samples_and_dists)
    latent_posterior_dists, latent_prior_dists = latent_dists
    latent_posterior_samples, latent_prior_samples, latent_conditional_prior_samples = latent_samples

    # Compute the KL divergence part of the ELBO
    outputs = {}
    if self.kl_analytic:
      latent_kl_divergences = tfd.kl_divergence(latent_posterior_dists, latent_prior_dists)
    else:
      latent_kl_divergences = (latent_posterior_dists.log_prob(latent_posterior_samples)
                                - latent_prior_dists.log_prob(latent_posterior_samples))
    latent_kl_divergences = tf.reduce_sum(latent_kl_divergences, axis=1)
    outputs.update({
      'latent_kl_divergence': tf.reduce_mean(latent_kl_divergences),
    })

    elbo = - latent_kl_divergences

    # Compute the reconstruction part of the ELBO
    likelihood_dists = {}
    likelihood_log_probs = {}
    reconstruction_error = {}
    for name in self.reconstruct_names:
      likelihood_dists[name] = self.decoders[name](latent_posterior_samples)
      images_tmp = tf.image.convert_image_dtype(images[name], tf.float32)
      likelihood_log_probs[name] = likelihood_dists[name].log_prob(images_tmp)
      likelihood_log_probs[name] = tf.reduce_sum(likelihood_log_probs[name], axis=1)
      reconstruction_error[name] = tf.reduce_sum(tf.square(images_tmp - likelihood_dists[name].distribution.loc),
                                         axis=list(range(-len(likelihood_dists[name].event_shape), 0)))
      reconstruction_error[name] = tf.reduce_sum(reconstruction_error[name], axis=1)
      outputs.update({
        'log_likelihood_'+name: tf.reduce_mean(likelihood_log_probs[name]),
        'reconstruction_error_'+name: tf.reduce_mean(reconstruction_error[name]),
      })
      elbo += likelihood_log_probs[name]

    # average over the batch dimension
    loss = -tf.reduce_mean(elbo)

    # Generate the images
    posterior_images = {}
    prior_images = {}
    conditional_prior_images = {}
    for name in self.reconstruct_names:
      posterior_images[name] = likelihood_dists[name].mean()
      prior_images[name] = self.decoders[name](latent_prior_samples).mean()
      conditional_prior_images[name] = self.decoders[name](latent_conditional_prior_samples).mean()

    images = tf.concat([tf.image.convert_image_dtype(images[k], tf.float32)
      for k in list(set(self.input_names+self.reconstruct_names))], axis=-2)
    posterior_images = tf.concat(list(posterior_images.values()), axis=-2)
    prior_images = tf.concat(list(prior_images.values()), axis=-2)
    conditional_prior_images = tf.concat(list(conditional_prior_images.values()), axis=-2)

    outputs.update({
      'elbo': tf.reduce_mean(elbo),
      'images': images,
      'posterior_images': posterior_images,
      'prior_images': prior_images,
      'conditional_prior_images': conditional_prior_images,
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