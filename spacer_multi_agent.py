import time

import gym
import numpy as np
import os
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


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id, address=rank, real_time=False)
        env.seed(seed + rank)
        # check_env(env)
        return env

    set_random_seed(seed)
    return _init


env_id = "blendtorch-spacer-v2"
num_cpu = 5 # Number of processes to use


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


def main():

    env = SubprocVecEnv([make_env(env_id, i) for i in range(1, 2)])
    env = VecMonitor(env, LOG_DIR)
    # env = gym.make(env_id, address=0, real_time=False)
    # env = Monitor(env)
    obs = env.reset()
    #
    # for i in range(100):
    #
    #     obs, reward, done, _ = env.step(np.array([1]))
    #     if done:
    #         env.reset()
    #     print(obs, reward, done)

    model = A2C('MlpPolicy', env, verbose=1, n_steps=1500, learning_rate=0.0001)

    callback = TrainAndLoggingCallback(check_freq=100, save_path=CHECKPOINT_DIR)

    start_time = time.time()
    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    total_time_step = 100000
    # Multiprocessed RL Training
    model.learn(total_timesteps=total_time_step, callback=callback, log_interval=1)

    total_time_multi = time.time() - start_time

    print(f"Took {total_time_multi:.2f}s for multiprocessed version - {total_time_step / total_time_multi:.2f} FPS")

    # Evaluate the trained agent
    env_val = gym.make("env_id", address=num_cpu + 2, real_time=False)
    for episode in range(10):
        obs = env_val.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(np.array([action]))
            time.sleep(0.2)
            total_reward += reward
        print('Total Reward for episode {} is {}'.format(episode, total_reward[0]))

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


if __name__ == '__main__':
    main()
