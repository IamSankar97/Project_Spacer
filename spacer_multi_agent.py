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
import torch
import torch.nn as nn
import torch.nn.functional as F
# from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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


discriminator = nn.Sequential(
    # in: 1 x 256 x 256
    nn.Conv2d(1, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 128 x 128

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 64 x 64

    nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(1024),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 1024 x 32 x 32

    nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2048),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 2048 x 16 x 16

    nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4096),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 4096 x 8 x 8

    nn.Conv2d(4096, 1, kernel_size=8, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid())


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def train_discriminator(real_images, fake_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_images = real_images.float()
    fake_images = fake_images.float()

    real_preds = discriminator(real_images)
    real_targets = torch.ones(4, 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images

    # Pass fake images through discriminator
    fake_targets = torch.zeros(4, 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def get_trainable_data(image, assign_device):
    crop_images = [torch.from_numpy(np.array([image[i: i + 256, j: j + 256]])) for i in range(0, 512, 256)
                   for j in range(0, 512, 256)]
    crop_images = torch.stack(crop_images)
    crop_images = crop_images.float()
    return to_device(crop_images, assign_device)


def binary_accuracy(y_pred, y_true):
    y_pred_label = (y_pred > 0.7).long()  # round off the predicted probability to the nearest label (0 or 1)
    correct = torch.eq(y_pred_label, y_true)
    acc = torch.mean(correct.float())
    return acc


def reshape_obs(observation):
    obs = [np.asarray(observation.resize((256, 256)), dtype=np.float64) / 255]
    obs = np.reshape(obs[0], (256, 256, 1))
    return obs


device = get_default_device()
discriminator = to_device(discriminator, device)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))


class Penv(gym.Env):
    def __init__(self, add):
        self.env_id = "blendtorch-spacer-v2"
        self.environments = gym.make(self.env_id, address=add, real_time=False)
        self.state = [25]
        self.reward = 0
        self.spacer_data_dir = 'spacer_data/train/'
        self.spacer_data = os.listdir(self.spacer_data_dir)
        self.image_size = (512, 512)
        self.action_space = spaces.Box(0.1, 0.7, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=255, shape=(256, 256, 1), dtype=np.uint8)

    def get_real_sp(self):
        filename = random.choice(self.spacer_data)
        if filename.endswith('.png'):
            actual_spacer = Image.open(os.path.join(self.spacer_data_dir, filename))
            return actual_spacer

    def reset(self):
        obs_ = self.environments.reset()
        self.state = reshape_obs(obs_)
        return self.state

    def step(self, action):
        #   Take action and collect observations
        obs_ = self.environments.step(action)
        self.state, reward, done, info = reshape_obs(obs_[0]), obs_[1], obs_[2], obs_[3]

        #   Start Reward shaping
        real_spacer = self.get_real_sp()
        real_spacer = np.asarray(real_spacer.resize(self.image_size), dtype=np.float64) / 255
        fake_spacer = np.asarray(obs_[0], dtype=np.float64) / 255
        actual, fake = get_trainable_data(real_spacer, device), get_trainable_data(fake_spacer, device)

        #   Train discriminator
        loss, real_score, fake_score = train_discriminator(actual, fake, opt_d)

        #   Try to fool the discriminator
        preds = discriminator(fake)
        targets = torch.ones(4, 1, device=device)
        reward = binary_accuracy(preds, targets)
        reward = reward.cpu().numpy()
        #   End Reward shaping

        done = True if np.average(self.state[0]) > 0.8 else False

        return self.state, [reward], done, info


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def main():
    def make_env(address):
        return lambda: Penv(address)

    addresses = [5]
    Py_env = SubprocVecEnv([make_env(address) for address in addresses])

    obs = Py_env.reset()
    global i
    time_total = 0
    for i in range(15):
        start = time.time()
        obs_ = Py_env.step(np.array([[0.5]]))
        time_one_iter = time.time() - start
        time_total += time_one_iter
        print("time taken_iter{}:".format(i), time_one_iter)
    print('total time over_ 2 iteration: ', time_total / (i + 1))

    print("# Learning")

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = PPO("CnnPolicy", Py_env, policy_kwargs=policy_kwargs, tensorboard_log=LOG_DIR, learning_rate=0.00001,
                n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9, verbose=1)

    callback = TrainAndLoggingCallback(check_freq=100, save_path=CHECKPOINT_DIR)

    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    total_time_step = 100000
    #   Multi-processed RL Training
    model.learn(total_timesteps=total_time_step, callback=callback, log_interval=1)

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
