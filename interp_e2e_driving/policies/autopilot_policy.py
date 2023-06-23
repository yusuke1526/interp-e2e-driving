# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import carla

from typing import Optional

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
    def __init__(
        self,
        time_step_spec,
        action_spec,
        py_env,
        noise_dist: Optional[list] = None,
    ):
        """
        args:
            noise_dist: [mean, std]
        """
        super(AutopilotPolicy, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec=(),
        )
        self.action_spec = action_spec
        self.py_env = py_env
        self.noise_dist = tfp.distributions.Normal(
                loc=noise_dist[0], scale=noise_dist[1]
            ) if noise_dist else None

    def get_control(self):
        return self.py_env.gym.get_control()

    def _action(self, time_step, policy_state, seed, *args, **kwargs):
        # Autopilotから制御信号を取得
        ego = self.py_env.gym.ego
        # acc, steer = self.get_control()

        # def true_fn():
        #     tf.print('true')
        #     control = ego.get_control()
        #     throttle = control.throttle
        #     steer = control.steer
        #     brake = control.brake
        #     return throttle, steer, brake
        # def false_fn():
        #     tf.print('false')
        #     throttle = 1.0
        #     steer = 1.0
        #     brake = 0.0
        #     return throttle, steer, brake

        # throttle, steer, brake = tf.cond(tf.constant(ego is not None, dtype=tf.bool), true_fn, false_fn)

        if ego:
            control = ego.get_control()
            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            # print(self.py_env.gym._get_reward())
        else:
            throttle = 0.0
            steer = 0.0
            brake = 0.0

        # Convert throttle and brake to acceleration
        if throttle > 0:
            acc = throttle * 3
        else:
            acc = -brake * 8

        if self.noise_dist:
            acc += self.noise_dist.sample(1).numpy()
            steer += self.noise_dist.sample(1).numpy()

        acc = np.clip(
            acc, self.action_spec.minimum[0], self.action_spec.maximum[0]
        )
        steer = np.clip(
            steer, self.action_spec.minimum[1], self.action_spec.maximum[1]
        )

        # 制御信号をtf_agentsの形式に変換
        action = tf.constant(
            [
                [acc, steer],
            ],
            dtype=tf.float32,
        )
        # print(action)
        return policy_step.PolicyStep(
            action=action, state=policy_state, info=()
        )

    def _variables(self):
        return []
