import numpy as np

import gym
import pygame
from gym import spaces
from skimage.transform import resize

import tensorflow as tf
from tf_agents.utils import common

from interp_e2e_driving.networks.world_model import WorldModel


def rgb_to_display_surface(rgb, display_size):
    """
    Generate pygame surface given an rgb image uint8 matrix
    :param rgb: rgb image uint8 matrix
    :param display_size: display size
    :return: pygame surface
    """
    surface = pygame.Surface((display_size, display_size)).convert()
    display = resize(rgb, (display_size, display_size))
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    # print(surface)
    # print(display.shape)
    pygame.surfarray.blit_array(surface, display)
    return surface


class WorldModelEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, params):
        # parameters
        self.input_names = params["input_names"]
        self.mask_names = params["mask_names"]
        self.latent_size = params["latent_size"]
        self.max_time_episode = params["max_time_episode"]
        self.obs_range = params["obs_range"]
        self.lidar_bin = params["lidar_bin"]
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.display_size = params["display_size"]

        self.action_space = spaces.Box(
            np.array(
                [
                    params["continuous_accel_range"][0],
                    params["continuous_steer_range"][0],
                ]
            ),
            np.array(
                [
                    params["continuous_accel_range"][1],
                    params["continuous_steer_range"][1],
                ]
            ),
            dtype=np.float32,
        )  # acc, steer
        observation_space_dict = {
            "camera": spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_size, self.obs_size, 3),
                dtype=np.uint8,
            ),
            "lidar": spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_size, self.obs_size, 3),
                dtype=np.uint8,
            ),
            "birdeye": spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_size, self.obs_size, 3),
                dtype=np.uint8,
            ),
            "state": spaces.Box(
                np.array([-2, -1, -5, 0]),
                np.array([2, 1, 30, 1]),
                dtype=np.float32,
            ),
        }
        self.observation_space = spaces.Dict(observation_space_dict)

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        model_net = WorldModel(
            input_names=self.input_names,
            reconstruct_names=self.input_names + self.mask_names,
            action_size=2,  # acc, ster
            gaussian_mixtures=5,  # TODO パラメタライズ
            obs_size=self.obs_size,
            latent_size=self.latent_size,
            optimizer=None,
            train_step_counter=None,
            batch_size=1,
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
        best_loss = tf.Variable(float("inf"))
        model_checkpointer = common.Checkpointer(
            ckpt_dir=params["world_model_ckpt_dir"],
            model=model_net,
            optimizer=optimizer,
            global_step=global_step,
            best_loss=best_loss,
            max_to_keep=5,
        )
        model_checkpointer.initialize_or_restore()

        self.world_model = model_net
        self.reward = None
        self.state_h = None
        self.state_c = None

        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size)
        )

        pygame.display.set_caption("world_model")

    def reset(self):
        # 環境を初期状態にする関数
        # 初期状態をreturnする
        self.latent = self.world_model.sample_prior(1)
        self.reset_step = 0
        return self.latent

    def step(self, action):
        # 行動を受け取り行動後の状態をreturnする
        (
            self.latent,
            self.reward,
            (self.state_h, self.state_c),
        ) = self.world_model.step(
            input_z=tf.expand_dims(self.latent, axis=0),
            input_action=action,
            prev_rew=self.reward,
            state_input_h=self.state_h,
            state_input_c=self.state_c,
        )

        self.reset_step += 1
        self.total_step += 1

        obs = self.latent
        reward = self.reward
        done = self.reset_step > self.max_time_episode
        info = {}

        return obs, reward, done, info

    def render(self):
        images = self.world_model.reconstruct(self.latent).numpy()
        images = np.hsplit(images, 3)
        for i, image in enumerate(images):
            image = np.clip(image * 255, 0, 255)
            image_surface = rgb_to_display_surface(image, self.display_size)
            self.display.blit(image_surface, (self.display_size * i, 0))
        pygame.display.update()
        self.clock.tick(100)

    def close(self):
        pygame.quit()

    def seed(self, seed=None):
        pass


if __name__ == "__main__":
    params = {
        "input_names": ["camera", "lidar"],
        "mask_names": ["birdeye"],
        "latent_size": 1024,
        "max_time_episode": 500,
        "obs_range": 32,
        "lidar_bin": 0.5,
        "continuous_accel_range": [-3.0, 3.0],
        "continuous_steer_range": [-0.3, 0.3],
        "world_model_ckpt_dir": "./logs/carla-v0/auto_memory/checkpoint",
        "display_size": 256,
    }

    env = WorldModelEnv(params)

    obs = env.reset()
    while True:
        pygame.event.pump()
        press = pygame.key.get_pressed()
        acc = 0.0
        steer = 0.0
        if press[pygame.K_w]:
            acc = 3.0
        elif press[pygame.K_s]:
            acc = -3.0
        if press[pygame.K_a]:
            steer = 0.3
        elif press[pygame.K_d]:
            steer = -0.3
        if press[pygame.K_r]:
            env.reset()
        # acc, steer
        action, _states = tf.Variable([[[acc, steer]]]), None
        print(action)
        obs, rewards, dones, info = env.step(action)
        env.render()
        # if dones:
        #     break

    env.close()
