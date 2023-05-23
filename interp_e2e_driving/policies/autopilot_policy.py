# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
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
  def __init__(self, action_spec, ego_vehicle):
      super(AutopilotPolicy, self).__init__()
      self.action_spec = action_spec
      self.ego_vehicle = ego_vehicle

  def _action(self, time_step, policy_state, seed):
      # Autopilotから制御信号を取得
      control = self.ego_vehicle.get_control()

      # 制御信号をtf_agentsの形式に変換
      # action = {
      #     'acceleration': tf.constant([control.throttle]),
      #     'brake': tf.constant([control.brake]),
      #     'steering': tf.constant([control.steer])
      # }

      return control

  def _variables(self):
      return []
  
  def _action(self, time_step, policy_state, seed):
    """This will update the state based on time_step and generate action."""
    distribution_step = self._distribution(time_step, policy_state)
    action = distribution_step.action.sample(seed=seed)

    # Update the last action to policy state
    network_state, latent_state, _ = distribution_step.state
    policy_state = (network_state, latent_state, action)

    return distribution_step._replace(action=action, state=policy_state)

  def _distribution(self, time_step, policy_state):
    network_state, latent_state, last_action = policy_state
    latent_state = tf.where(time_step.is_first(), 
      self._model_network.first_filter(time_step.observation),
      self._model_network.filter(time_step.observation, latent_state, last_action))

    # Update the latent state
    policy_state = (network_state, latent_state, last_action)
    # Actor network outputs nested structure of distributions or actions.
    actions_or_distributions, network_state = self._apply_actor_network(
        time_step, policy_state)
    # Update the network state
    policy_state = (network_state, latent_state, last_action)

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution,
                                          actions_or_distributions)

    # Prepare policy_info.
    if self._collect:
      policy_info = ppo_utils.get_distribution_params(distributions)
    else:
      policy_info = ()

    return policy_step.PolicyStep(distributions, policy_state, policy_info)