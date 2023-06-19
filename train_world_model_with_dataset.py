# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import functools
import gin
import numpy as np
import os
import tensorflow as tf
import time
from collections import OrderedDict
import glob
from tqdm import tqdm

from tf_agents.utils import common

import gym
import gym_carla

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.td3 import td3_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import q_rnn_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common, example_encoding_dataset

from interp_e2e_driving.agents.ddpg import ddpg_agent
from interp_e2e_driving.agents.world_model import world_model_agent
from interp_e2e_driving.environments import filter_observation_wrapper
from interp_e2e_driving.networks import multi_inputs_actor_rnn_network
from interp_e2e_driving.networks import multi_inputs_critic_rnn_network
from interp_e2e_driving.networks import world_model
from interp_e2e_driving.utils import gif_utils


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('experiment_name', None,
                    'Experiment name used for naming the output directory.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


@gin.configurable
def load_carla_env(
  env_name='carla-v0',
  discount=1.0,
  number_of_vehicles=100,
  number_of_walkers=0,
  display_size=256,
  max_past_step=1,
  dt=0.1,
  discrete=False,
  discrete_acc=[-3.0, 0.0, 3.0],
  discrete_steer=[-0.2, 0.0, 0.2],
  continuous_accel_range=[-3.0, 3.0],
  continuous_steer_range=[-0.3, 0.3],
  ego_vehicle_filter='vehicle.lincoln*',
  port=2000,
  town='Town03',
  task_mode='random',
  max_time_episode=500,
  max_waypt=12,
  obs_range=32,
  lidar_bin=0.5,
  d_behind=12,
  out_lane_thres=2.0,
  desired_speed=8,
  max_ego_spawn_times=200,
  display_route=True,
  pixor_size=64,
  pixor=False,
  obs_channels=None,
  auto_exploration=False,
  action_repeat=1):
  """Loads train and eval environments."""
  env_params = {
    'number_of_vehicles': number_of_vehicles,
    'number_of_walkers': number_of_walkers,
    'display_size': display_size,  # screen size of bird-eye render
    'max_past_step': max_past_step,  # the number of past steps to draw
    'dt': dt,  # time interval between two frames
    'discrete': discrete,  # whether to use discrete control space
    'discrete_acc': discrete_acc,  # discrete value of accelerations
    'discrete_steer': discrete_steer,  # discrete value of steering angles
    'continuous_accel_range': continuous_accel_range,  # continuous acceleration range
    'continuous_steer_range': continuous_steer_range,  # continuous steering angle range
    'ego_vehicle_filter': ego_vehicle_filter,  # filter for defining ego vehicle
    'port': port,  # connection port
    'town': town,  # which town to simulate
    'task_mode': task_mode,  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': max_time_episode,  # maximum timesteps per episode
    'max_waypt': max_waypt,  # maximum number of waypoints
    'obs_range': obs_range,  # observation range (meter)
    'lidar_bin': lidar_bin,  # bin size of lidar sensor (meter)
    'd_behind': d_behind,  # distance behind the ego vehicle (meter)
    'out_lane_thres': out_lane_thres,  # threshold for out of lane
    'desired_speed': desired_speed,  # desired speed (m/s)
    'max_ego_spawn_times': max_ego_spawn_times,  # maximum times to spawn ego vehicle
    'display_route': display_route,  # whether to render the desired route
    'pixor_size': pixor_size,  # size of the pixor labels
    'pixor': pixor,  # whether to output PIXOR observation
    'auto_exploration': auto_exploration,
  }

  gym_spec = gym.spec(env_name)
  gym_env = gym_spec.make(params=env_params)

  if obs_channels:
    gym_env = filter_observation_wrapper.FilterObservationWrapper(gym_env, obs_channels)

  py_env = gym_wrapper.GymWrapper(
    gym_env,
    discount=discount,
    auto_reset=True,
  )

  eval_py_env = py_env

  if action_repeat > 1:
    py_env = wrappers.ActionRepeat(py_env, action_repeat)

  return py_env, eval_py_env


def compute_summaries(valid_dataset,
                      train_step=None,
                      summary_writer=None,
                      model_net=None,
                      num_episodes_to_render=2,
                      num_steps_to_render=100,
                      image_keys=None,
                      fps=10,
                      ):

  error_list_dict = {}
  for experience in valid_dataset:
    images = experience.observation
    actions = experience.action
    rewards = experience.reward
    step_types = experience.step_type
    total_loss, outputs = model_net.compute_loss(images, actions, rewards, step_types)
    for k, v in outputs.items():
      if k in error_list_dict.keys():
        error_list_dict[k].append(v)
      else:
        error_list_dict[k] = [v]

  error_list_dict = {k: np.array(v).mean() for k, v in error_list_dict.items()}

  # Summarize scalars to tensorboard
  if train_step and summary_writer:
    with summary_writer.as_default():
      for name, loss in error_list_dict.items():
        tf.compat.v2.summary.scalar(name='model_loss/'+name, data=loss, step=train_step)

  # generate video from sequence dataset
  valid_dataset = valid_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) # (B, T, w, h, c) -> (T, w, h, c)
  valid_dataset = valid_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)).batch(1) # (T, w, h, c) -> (1, w, h, c)
  iterator = iter(valid_dataset)
  images = [[]]
  reconstruct_images = [[]]
  reconstruct_images_from_latents = [[]]
  episode = 0
  is_first = True
  while episode < num_episodes_to_render:
    experience = next(iterator)

    if (episode < num_episodes_to_render) and experience.is_boundary():
      images.append([])
      reconstruct_images.append([])
      reconstruct_images_from_latents.append([])
      is_first = True

    obs = experience.observation
    images[-1].append(OrderedDict(obs))
    z_mean, z_log_var, z = model_net.vision.encode(obs)
    recon = model_net.vision.decode(z)
    reconstruct_images[-1].append(OrderedDict(recon))

    # reconstruct_images_from_latents
    if is_first:
      state_h, state_c = None, None
      prev_z = z
      is_first = False
    else:
      (log_pi, mu, log_sigma), rew_pred, (state_h, state_c) = model_net.memory.pred(
        input_z=tf.expand_dims(prev_z, axis=0),
        input_action=prev_action,
        prev_rew=prev_rew,
        state_input_h=state_h,
        state_input_c=state_c,
        return_state=True)
      z_pred = model_net.memory.sample_z(log_pi, mu, log_sigma)
      z_pred = tf.expand_dims(z_pred, axis=0)
      recon = model_net.vision.decode(z_pred)
      prev_z = z_pred
    reconstruct_images_from_latents[-1].append(OrderedDict(recon))

    if experience.is_last():
      episode += 1
    prev_rew = tf.expand_dims(tf.expand_dims(experience.reward, axis=0), axis=0)
    prev_action = tf.expand_dims(experience.action, axis=0)
  
  images = concat_images(images, image_keys)
  reconstruct_images = concat_images(reconstruct_images, image_keys)
  reconstruct_images_from_latents = concat_images(reconstruct_images_from_latents, image_keys)

  all_images = tf.concat([images, reconstruct_images, reconstruct_images_from_latents], axis=2)

  # Need to avoid eager here to avoid rasing error
  gif_summary = common.function(gif_utils.gif_summary_v2)

  # Summarize to tensorboard
#   gif_summary('ObservationVideoEvalPolicy', images, 1, fps)
#   gif_summary('ReconstructedVideoEvalPolicy', reconstruct_images, 1, fps)
#   gif_summary('ReconstructedVideoFromLatentsEvalPolicy', reconstruct_images_from_latents, 1, fps)
  gif_summary('ObservationReconstructedVideoEval', all_images, 1, fps)

  return total_loss


def concat_images(images, image_keys):
  # Concat input images of different episodes and generate reconstructed images.
  # Shape of images is [[images in episode as timesteps]].
  if type(images[0][0]) is OrderedDict:
    images = pad_and_concatenate_videos(images, image_keys=image_keys, is_dict=True)
  else:
    images = pad_and_concatenate_videos(images, image_keys=image_keys, is_dict=False)
  images = tf.image.convert_image_dtype([images], tf.uint8, saturate=True)
  images = tf.squeeze(images, axis=2)
  return images


def pad_and_concatenate_videos(videos, image_keys, is_dict=False):
  max_episode_length = max([len(video) for video in videos])
  if is_dict:
    # videos = [[tf.concat(list(dict_obs.values()), axis=2) for dict_obs in video] for video in videos]
    videos = [[tf.concat([dict_obs[key] for key in image_keys], axis=2) for dict_obs in video] for video in videos]
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=2) for frames in zip(*videos)]
  return videos


def get_latent_reconstruction_videos(latents, model_net):
  videos = []
  for latent_eps in latents:
    videos.append([])
    for latent in latent_eps:
      videos[-1].append(model_net.reconstruct(latent)[0])

  max_episode_length = max([len(video) for video in videos])
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=0) for frames in zip(*videos)]
  return videos


@gin.configurable
def train_world_model(
    root_dir,
    experiment_name,  # experiment name
    env_name='carla-v0',
    num_iterations=int(1e7),
    input_names=['camera', 'lidar'],  # names for inputs
    mask_names=['birdeye'],  # names for masks
    latent_size=64,
    dataset_dir='./logs/carla-v0/collect_dataset',
    # Params for train
    train_steps_per_iteration=1,
    batch_size=256,
    model_learning_rate=1e-4,  # learning rate for model training
    sequence_length=4,  # number of timesteps to train model
    train_vision_flag=False,
    train_memory_flag=True,
    train_controller_flag=False,
    # Params for eval
    eval_interval=10000,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    summary_interval=1000,
    summaries_flush_secs=10,
    gpu_allow_growth=True,  # GPU memory growth
    gpu_memory_limit=None,  # GPU memory limit
    action_repeat=1):  # Name of single observation channel, ['camera', 'lidar', 'birdeye']
  # Setup GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpu_allow_growth:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpu_memory_limit:
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(
          gpu,
          [tf.config.experimental.VirtualDeviceConfiguration(
              memory_limit=gpu_memory_limit)])

  # Get train and eval direction
  root_dir = os.path.expanduser(root_dir)
  root_dir = os.path.join(root_dir, env_name, experiment_name)

  # Get summary writers
  summary_writer = tf.summary.create_file_writer(
      root_dir, flush_millis=summaries_flush_secs * 1000)
  summary_writer.set_as_default()

  global_step = tf.compat.v1.train.get_or_create_global_step()

  # Whether to record for summary
  with tf.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    # Create Carla environment
    py_env, eval_py_env = load_carla_env(env_name='carla-v0', obs_channels=input_names+mask_names, action_repeat=action_repeat)

    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    # Specs
    time_step_spec = tf_env.time_step_spec()
    action_spec = tf_env.action_spec()

    tf_env.close()

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=model_learning_rate)

    # Get model network for latent sac
    model_net = world_model.WorldModel(
      input_names=input_names,
      reconstruct_names=input_names+mask_names,
      action_size=action_spec.shape[0],
      gaussian_mixtures=5, # TODO パラメタライズ
      obs_size=py_env.obs_size,
      latent_size=latent_size,
      optimizer=optimizer,
      train_step_counter=global_step,
      batch_size=batch_size,
      )

    # Load dataset from dumped tfrecords with shape [Bxslx...]
    tfrecords = [p for p in list(glob.glob(os.path.join(dataset_dir, 'dataset.tfrecord.*'))) if 'spec' not in p]
    tfrecords = sorted(tfrecords, key=lambda x: int(x.split('.')[-1]))
    dataset = example_encoding_dataset.load_tfrecord_dataset(tfrecords[:int(len(tfrecords)*0.8)], num_parallel_reads=1, add_batch_dim=False, as_trajectories=True) \
        .batch(sequence_length+1, drop_remainder=True).batch(batch_size, drop_remainder=True).repeat(-1).prefetch(3)
    valid_dataset = example_encoding_dataset.load_tfrecord_dataset(tfrecords[int(len(tfrecords)*0.8):], num_parallel_reads=1, add_batch_dim=False, as_trajectories=True) \
        .batch(sequence_length+1, drop_remainder=True).batch(batch_size, drop_remainder=True).prefetch(3)
    
    # prepare checkpoint
    best_loss = tf.Variable(float('inf'))
    checkpoint = tf.train.Checkpoint(model=model_net, optimizer=optimizer, global_step=global_step, best_loss=best_loss)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(root_dir, 'checkpoint'), max_to_keep=5, step_counter=global_step)
    checkpoint_manager.restore_or_initialize()
    # global_step.assign(0)

    total_loss = compute_summaries(
      valid_dataset,
      train_step=global_step,
      summary_writer=summary_writer,
      model_net=model_net,
      image_keys=input_names+mask_names,
    )

    iterator = iter(dataset)

    progress_bar = tqdm(total=num_iterations)

    # Start training
    while global_step.numpy() < num_iterations:
      experience = next(iterator)
      
      model_net.train(experience, train_flag={'vision':train_vision_flag,
                                              'memory': train_memory_flag,
                                              'controller': train_controller_flag})

      # Evaluation
      if global_step.numpy() % eval_interval == 0:
        # Log evaluation metrics
        total_loss = compute_summaries(
          valid_dataset,
          train_step=global_step,
          summary_writer=summary_writer,
          model_net=model_net,
          image_keys=input_names+mask_names,
        )

      # Save checkpoints
      global_step_val = global_step.numpy()
      if global_step_val % train_checkpoint_interval == 0:
        checkpoint_manager.save()
        
      # update progress bar
      progress_bar.update(1)
      progress_bar.set_postfix({'total_loss': total_loss.numpy()})

def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_world_model(FLAGS.root_dir, FLAGS.experiment_name)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  flags.mark_flag_as_required('experiment_name')
  app.run(main)