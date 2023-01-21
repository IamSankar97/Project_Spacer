import time
import gym
from gym import spaces
import numpy as np
import os
import sys
import pickle
import random
from PIL import Image

sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs/')
import spacer_gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
from stable_baselines3.common.env_checker import check_env

CHECKPOINT_DIR = 'train/train_Center/A2C_pa5_train_gym2_large2.8'
LOG_DIR = 'logs/log_center/Target_max/A2C_pa5_train_gym2_large2.8'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))

            self.model.save(model_path)
        return True


class Penv(gym.Env):
    def __init__(self, add):
        self.env_id = "blendtorch-spacer-v2"
        self.environments = gym.make(self.env_id, address=add, real_time=False)
        self.up_limit = 60
        self.lw_limit = 40
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.lw_limit, high=self.up_limit, shape=(1,),
                                            dtype=np.float32)
        self.state = [25]
        self.reward = 0
        self.spacer_data_dir = '/home/mohanty/PycharmProjects/Project_Spacer/spacer_data/train/'
        self.spacer_data = os.listdir(self.spacer_data_dir)

    def get_actual_sp(self):
        filename = random.choice(self.spacer_data)
        if filename.endswith('.png'):
            actual_spacer = Image.open(os.path.join(self.spacer_data_dir, filename))
            return np.asarray(actual_spacer)

    def get_reward(self, actual_spacer):
        # ------ This is where Discriminator will be placed.#----
        obs_fake, obs_actual = np.average(self.state[0]), np.average(actual_spacer)
        return [obs_fake] if obs_fake > obs_actual else [obs_actual]

    def reset(self):
        obs_ = self.environments.reset()
        self.state = [np.average(np.asarray(obs_))]
        return self.state

    def step(self, action):
        obs_ = self.environments.step([action])
        self.state, reward, done, info = [np.asarray(obs_[0])], obs_[1], obs_[2], obs_[3]
        spacer_act = self.get_actual_sp()
        # Below is assigned to state, in future it has to be assigned to reward.
        self.state = self.get_reward(spacer_act)
        return self.state, reward, done, info


def main():
    def make_env(address):
        return lambda: Penv(address)

    addresses = [5, 7]
    Py_env = SubprocVecEnv([make_env(address) for address in addresses])

    # obs = Py_env.reset()
    # global i
    # time_total = 0
    # for i in range(3):
    #     start = time.time()
    #     obs_ = Py_env.step(np.array([[1],
    #                                  [1]]))
    #     print(obs_)
    #     time_one_iter = time.time() - start
    #     time_total += time_one_iter
    #     print("time taken_iter{}:".format(i), time_one_iter)
    # print('total time over_ 2 iteration: ', time_total/(i+1))

    print("# Learning")
    #
    model = A2C('MlpPolicy', Py_env, verbose=1, n_steps=1500, learning_rate=0.0001)

    callback = TrainAndLoggingCallback(check_freq=100, save_path=CHECKPOINT_DIR)

    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    total_time_step = 100000
    # Multiprocessed RL Training
    model.learn(total_timesteps=total_time_step, callback=callback, log_interval=1)
    #
    # total_time_multi = time.time() - start_time
    #
    # print(f"Took {total_time_multi:.2f}s for multiprocessed version - {total_time_step / total_time_multi:.2f} FPS")
    #
    # # Evaluate the trained agent
    # env_val = gym.make("env_id", address=num_cpu + 2, real_time=False)
    # for episode in range(10):
    #     obs = env_val.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         action, _ = model.predict(obs)
    #         obs, reward, done, info = Py_env.step(np.array([action]))
    #         time.sleep(0.2)
    #         total_reward += reward
    #     print('Total Reward for episode {} is {}'.format(episode, total_reward[0]))
    #
    # # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # # print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


if __name__ == '__main__':
    main()
