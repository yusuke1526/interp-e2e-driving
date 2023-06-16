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

import gin
import numpy as np
import os
import tensorflow as tf
import pickle
import glob

import gym
import gym_carla

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common, example_encoding_dataset
from tf_agents.policies import random_tf_policy

from interp_e2e_driving.agents.world_model import world_model_agent
from interp_e2e_driving.environments import filter_observation_wrapper
from interp_e2e_driving.networks import world_model
from interp_e2e_driving.policies import autopilot_policy


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
        auto_exploration=True,
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


@gin.configurable
def collect_dataset(
    root_dir,
    experiment_name,  # experiment name
    env_name='carla-v0',
    input_names=['camera', 'lidar'],  # names for inputs
    mask_names=['birdeye'],  # names for masks
    latent_size=64,
    # Params for collect
    initial_collect_steps=int(1e5),
    replay_buffer_capacity=int(1e5),
    num_iteration=5,
    # Params for train
    batch_size=256,
    model_batch_size=32,  # model training batch size
    sequence_length=4,  # number of timesteps to train model
    model_learning_rate=1e-4,  # learning rate for model training
    gradient_clipping=None,
    # Params for summaries and logging
    num_images_per_summary=1,  # images for each summary
    summarize_grads_and_vars=False,
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

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Create Carla environment
    py_env, eval_py_env = load_carla_env(env_name='carla-v0', obs_channels=input_names+mask_names, action_repeat=action_repeat)

    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    fps = int(np.round(1.0 / (py_env.dt * action_repeat)))

    # Specs
    time_step_spec = tf_env.time_step_spec()
    action_spec = tf_env.action_spec()

    # Make tf agent
    # Get model network for latent sac
    model_net = world_model.WorldModel(
        input_names=input_names,
        reconstruct_names=input_names+mask_names,
        action_size=action_spec.shape[0],
        gaussian_mixtures=5,  # TODO パラメタライズ
        obs_size=py_env.obs_size,
        latent_size=latent_size
    )

    # Build the latent sac agent
    tf_agent = world_model_agent.WorldModelAgent(
        time_step_spec,
        action_spec,
        inner_agent=None,
        model_network=model_net,
        model_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=model_learning_rate),
        model_batch_size=model_batch_size,
        num_images_per_summary=num_images_per_summary,
        sequence_length=sequence_length,
        gradient_clipping=gradient_clipping,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
        py_env=py_env if py_env.gym.auto_exploration else None,
        fps=fps)

    # Get policies
    # initial_collect_policy = autopilot_policy.AutopilotPolicy(
    #     time_step_spec, action_spec, py_env)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
    )

    tfrecords = [p for p in sorted(list(glob.glob(os.path.join(root_dir, 'dataset.tfrecord.*')))) if 'spec' not in p]
    latent_num = int(tfrecords[-1].split('.')[-1])

    for i in range(latent_num+1, latent_num+1 + num_iteration):
        # Get tfrecord observer
        trajectory_spec = tf_agent.collect_data_spec
        dataset_path = os.path.join(root_dir, f'dataset.tfrecord.{i}')
        tfrecord_observer = example_encoding_dataset.TFRecordObserver(dataset_path, trajectory_spec)

        # Collect driver
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=[common.function(tfrecord_observer)],
            num_steps=initial_collect_steps)

        # Optimize the performance by using tf functions
        initial_collect_driver.run = common.function(initial_collect_driver.run)

        # Collect initial replay data.
        logging.info(
            'Initializing replay buffer by collecting experience for %d steps'
            'with a random policy.', initial_collect_steps)
        initial_collect_driver.run()


def main(_):
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    collect_dataset(FLAGS.root_dir, FLAGS.experiment_name)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('experiment_name')
    app.run(main)
