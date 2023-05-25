# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import carla

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step


@gin.configurable
class AutopilotPolicy(tf_policy.TFPolicy):
  def __init__(self, time_step_spec, action_spec, py_env):
      super(AutopilotPolicy, self).__init__(
          time_step_spec,
          action_spec,
          policy_state_spec=(),
      )
      self.action_spec = action_spec
      self.py_env = py_env

  def _action(self, time_step, policy_state, seed, *args, **kwargs):
      # Autopilotから制御信号を取得
      # if self.py_env.gym.ego:
      #   control = self.py_env.gym.ego.get_control()
      #   throttle = control.throttle
      #   steer = control.steer
      #   brake = control.brake
      # else:
      #   throttle = 0.0
      #   steer = 0.0
      #   brake = 0.0
      throttle = 0.0
      steer = 0.0
      brake = 0.0

        
      # Convert throttle and brake to acceleration
      if throttle > 0:
          acc = throttle * 3
      else:
          acc = -brake * 8
      acc = np.clip(acc, self.action_spec.minimum[0], self.action_spec.maximum[0])
      steer = np.clip(steer, self.action_spec.minimum[1], self.action_spec.maximum[1])

      # 制御信号をtf_agentsの形式に変換
      action = tf.constant([
        [acc, steer],
      ], dtype=tf.float32)

      return policy_step.PolicyStep(action=action, state=policy_state, info=())

  def _variables(self):
      return []