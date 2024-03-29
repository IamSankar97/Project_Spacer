import argparse
import math

import gym
from gym import spaces
import numpy as np
import os
import sys
import random
from collections import OrderedDict
from stable_baselines3 import PPO
import pandas as pd

sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs/')
import spacer_gym
import resnet
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DDPG

import datetime
import logging
import torchmetrics
from collections import deque
from utils import get_orth_actions, is_loss_stagnated
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

seed_ = 0

torch.manual_seed(seed_)
random.seed(seed_)
np.random.seed(seed_)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_path = '/home/mohanty/PycharmProjects/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
CHECKPOINT_DIR = log_path + 'train_logs2/spacer{}/PPO_model/'.format(stamp)
LOG_DIR = log_path + 'train_logs2/spacer{}/PPO_log'.format(stamp)
FINAL_MODEL_DIR = log_path + 'train_logs2/spacer{}/PPO_final_model'.format(stamp)
ACTION_LOG_DIR = LOG_DIR + '/action_log'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ACTION_LOG_DIR, exist_ok=True)
train_log = LOG_DIR + 'training_log.csv'
writer = SummaryWriter(ACTION_LOG_DIR)

writer.add_text("log_address", "CHECKPOINT_DIR: " + CHECKPOINT_DIR, global_step=0)
writer.add_text("log_address", "LOG_DIR: " + LOG_DIR, global_step=1)
writer.add_text("log_address", "ACTION_LOG_DIR: " + ACTION_LOG_DIR, global_step=2)
writer.add_text("log_address", "train_log: " + train_log, global_step=3)


def log_dict_to_tensorboard(data_dict, category, step):
    for key, value in data_dict.items():
        writer.add_scalar(category + '/' + key, value, global_step=step)


def get_device(device_index: str):
    """Pick GPU if available, else CPU"""
    return torch.device('cuda:' + device_index if torch.cuda.is_available() else 'cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_attribute_dict(*args):
    # Create a new dictionary and populate it with argument names and their values
    attr_dict = {arg_name: arg_value for arg_name, arg_value in zip([arg_name for arg_name in args], args)}

    return attr_dict


def log_to_file(log_info, log_file):
    with open(log_file, 'a', newline='') as csvfile2:
        writer = csv.DictWriter(csvfile2, fieldnames=list(log_info.keys()))
        if csvfile2.tell() == 0:
            writer.writeheader()

        writer.writerow(log_info)


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, patience=4000, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.patience = patience
        self.best_mean_reward = -float('inf')
        self.steps_since_best_reward = 0

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            gen_model_path = os.path.join(self.save_path, 'PPO_gen_model_{}'.format(self.n_calls))
            self.model.save(gen_model_path)

            self.logger.record('GAN_loss/ep_gen_loss', np.mean(self.training_env.get_attr('generator_loss_mean')))
            self.logger.record('GAN_loss/generator_acc_mean', np.mean(self.training_env.get_attr('generator_acc_mean')))
            self.logger.record('GAN_loss/avg_brightness_mean',
                               np.mean(self.training_env.get_attr('avg_brightness_mean')))

            return True


class CustomImageExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 512, scalar=False):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['0image'].shape[0]
        self.scalar = scalar
        self.cnn = torch.nn.Sequential(*list(resnet.resnet18(n_input_channels, 2).children())[:-2]).extend(
            [torch.nn.Flatten()])

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten_img = self.cnn(torch.as_tensor(observation_space['0image'].sample()[None]).float()).shape[1]
        if self.scalar:
            self.n_scalar = observation_space['scalar'].shape[0]
            self.n_img = len(observation_space.spaces) - self.n_scalar
            self.dropout0 = nn.Dropout(p=0.3)
            self.linear0 = nn.Linear(n_flatten_img, 100)
            self.linear1 = nn.Linear(100 * self.n_img, features_dim * 3)
            self.dropout1 = nn.Dropout(p=0.3)
            self.linear2 = nn.Linear(features_dim * 3, features_dim - self.n_scalar)

        else:
            self.n_img = len(observation_space.spaces)
            # self.dropout0 = nn.Dropout(p=0.2)
            self.linear0 = nn.Linear(n_flatten_img, 100)
            self.linear1 = nn.Linear(100 * self.n_img, features_dim * 3)
            # self.dropout1 = nn.Dropout(p=0.2)
            self.linear2 = nn.Linear(features_dim * 3, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.scalar:
            scalar_obs = observations['scalar']
            del observations['scalar']

        img_features = []
        for _, value in observations.items():
            cnn_output = self.cnn(value)
            # cnn_output = self.dropout0(cnn_output)
            linear0 = torch.relu(self.linear0(cnn_output))
            img_features.append(linear0)
            # cnn_reshape = torch.reshape(cnn_output, (img_obs.shape[0], self.select_crops.size()[0], cnn_output.shape[1]))

        cnn_output = torch.cat(img_features, dim=1)
        x = torch.relu(self.linear1(cnn_output))
        # x = self.dropout1(x)
        out_features = torch.relu(self.linear2(x))
        if self.scalar:
            out_features = torch.cat((x, scalar_obs), dim=-1)

        return out_features


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Penv(gym.Env):
    def __init__(self, img_size, batch_size, episode_length, dat_dir, discriminator, lr_discriminator, weight_decay,
                 device_discriminator, train_discriminator, blender_add, blend_file):
        super(Penv, self).__init__()
        self.disc_train_epoch = 0
        self.target_gen_loss = 0
        self.target_disc_loss = 0
        self.img_size = img_size
        self.disc_ls_epoch = []
        self.disc_ls = None
        self.disc_rl_score = None
        self.disc_fk_score = None
        self.done = False
        self.spacer_data_dir = dat_dir
        self.discriminator = discriminator
        self.disc_device = device_discriminator
        self.train_disc = train_discriminator
        self.dic_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999),
                                              weight_decay=weight_decay)
        self.environments = gym.make("blendtorch-spacer-v2", address=blender_add, blend_file=blend_file,
                                     real_time=False)
        logging.basicConfig(filename=train_log, level=logging.INFO)
        self.environments.reset()
        self.action_paired = {}
        # logging must be after environment generation
        self.action_space = spaces.Box(-1, 1, shape=(6,))
        self.observation_space = spaces.Dict(self._get_obs_space())
        self.actual_dataloader = self.get_image_dataloader()
        # derived by calculating avg value over all the available fake images of shape 64*64
        self.brightness_threshold = np.array([0.85, 0.30])
        self.mean_brightness = torch.mean(to_device(torch.from_numpy(self.brightness_threshold),
                                                    device=self.disc_device))
        # spacer data
        self.state = [32]
        self.spacer_data = os.listdir(self.spacer_data_dir)
        self.episode_length = episode_length
        self.time_step = -1
        self.episodes = 0
        self.epoch = 0
        self.avg_brightness = 0
        self.disc_fake_score = 0
        self.disc_real_score = 0
        self.discriminator_loss = 0
        self.batch_size = batch_size
        self.generator_loss_mean = []
        self.generator_loss = []
        self.generator_acc_mean = []
        self.avg_brightness_mean = []
        self.buffer_act_spacer = deque(maxlen=self.episode_length)
        self.buffer_fake_spacer = deque(maxlen=self.episode_length)
        self.disc_buffer_act_spacer = deque(maxlen=self.episode_length * 10)
        self.disc_buffer_fake_spacer = deque(maxlen=self.episode_length * 10)
        # self.initialize_discriminator()

    def train_discriminator(self, real_images, fake_images, clip=False):
        # Clear discriminator gradients
        self.dic_optimizer.zero_grad()

        # Pass real images through discriminator
        real_images = real_images.float()
        fake_images = fake_images.float()

        real_preds = self.discriminator(real_images)
        real_targets = to_device(torch.tensor([[1, 0] for _ in range(real_preds.size(0))]).float(),
                                 self.disc_device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torchmetrics.functional.accuracy(real_preds, real_targets, task='binary').cpu().numpy().item()

        # Pass fake images through discriminator
        fake_preds = self.discriminator(fake_images)
        fake_targets = to_device(torch.tensor([[0, 1] for _ in range(fake_preds.size(0))]).float(),
                                 self.disc_device)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torchmetrics.functional.accuracy(fake_preds, fake_targets, task='binary').cpu().numpy().item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        if clip:
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1, norm_type=2)
        loss.backward()
        self.dic_optimizer.step()
        # scheduler.step()
        return loss.item(), real_score, fake_score

    def get_attributes(self, attr_names):
        return {attr_name: getattr(self, attr_name) for attr_name in dir(self) if
                not callable(getattr(self, attr_name)) and not attr_name.startswith("__") and attr_name in attr_names}

    def get_fake_data(self, action=None, reset=False):
        if reset:
            obs, info = self.environments.reset(), None
        else:
            obs, _, _, info = self.environments.step(action)
            self.action_paired = info['action_pair']
            info.pop('action_pair')
        fake_spacer = np.array([np.array(obs.crop((i, 0, i + self.img_size[0], self.img_size[0])))
                                for i in range(0, obs.size[0], self.img_size[0])])

        return fake_spacer, info

    def _get_obs_space(self, scalar=False):
        fake_spacer, _ = self.get_fake_data(self.action_space.sample())
        self.no_img_obs = len(fake_spacer)
        obs_space = {}
        for i in range(self.no_img_obs):
            obs_space.update({'{}image'.format(i): spaces.Box(low=0, high=255,
                                                              shape=self.img_size + (1,), dtype=np.uint8)})
        if scalar:
            obs_space.update({'scalar': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)})

        return obs_space

    def get_image_dataloader(self, shuffle=True, device=None):
        # Define the transform to apply to each image
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.RandomHorizontalFlip(),  # Randomly flip images left-right
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # Convert to tensor
        ])

        # Create the dataset
        dataset = ImageFolder(root=self.spacer_data_dir, transform=transform)

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.no_img_obs, shuffle=shuffle)

        # Move the dataloader to the specified device
        if device is not None:
            dataloader = dataloader.to(self.disc_device)

        return dataloader

    def chk_termination(self):
        if self.steps == self.episode_length:
            return True
        else:
            return False

    def get_data(self, action):
        fake_spacer, info = self.get_fake_data(action)
        actual_spacer, _ = next(iter(self.actual_dataloader))
        return actual_spacer, fake_spacer, info

    def match_obs_space(self, fake_spacer):
        if len(fake_spacer) < self.no_img_obs:
            no_obs_duplicate = self.no_img_obs - len(fake_spacer)

            # Choose two random indices to duplicate
            duplicate_indices = np.random.choice(len(fake_spacer), no_obs_duplicate, replace=True)

            # Create a new array of size (12, 250, 250)
            fake_spacer = np.concatenate([fake_spacer, fake_spacer[duplicate_indices]], axis=0)

        if len(fake_spacer) > self.no_img_obs:
            no_obs_delete = len(fake_spacer) - self.no_img_obs
            remove_indices = np.random.choice(len(fake_spacer), no_obs_delete, replace=False)

            #   Create a new array of size (8, 250, 250) by selecting the remaining indices
            fake_spacer = np.delete(fake_spacer, remove_indices, axis=0)

        fake_spacer = np.random.permutation(fake_spacer)
        return fake_spacer

    def initialize_discriminator(self):
        if self.train_disc:
            ortho_actions = pd.DataFrame(get_orth_actions(self.action_space.shape[0]))
            noise_std, theta, dt = 0.12, 0.15, 1e-2

            self.disc_ls_epoch = []
            while True:
                self.disc_fk_score, self.disc_rl_score, self.disc_ls = [], [], []
                for _ in range(self.batch_size):  # episode_length = completing 1 epoch
                    #   Take action and collect observations
                    noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(self.action_space.shape),
                                                         sigma=np.array(noise_std), theta=theta, dt=dt)
                    action = np.array(ortho_actions.sample())
                    noisy_action = np.squeeze(np.clip(action + noise(), self.action_space.low, self.action_space.high))

                    actual_spacer, fake_spacer, info = self.get_data(noisy_action)
                    fake_spacer = self.match_obs_space(fake_spacer)

                    actual_spacer, fake_spacer = actual_spacer, fake_spacer / 255

                    actual_spacer = to_device(actual_spacer, self.disc_device)
                    fake_spacer = to_device(torch.from_numpy(fake_spacer.copy()).unsqueeze(1).float(), self.disc_device)

                    discriminator_loss, disc_real_score, disc_fake_score = self.train_discriminator(actual_spacer,
                                                                                                    fake_spacer)
                    self.disc_fk_score.append(disc_fake_score)
                    self.disc_ls.append(discriminator_loss)
                    self.disc_rl_score.append(disc_real_score)

                print('\033[1mEpoch:', self.epoch, 'disc_ls:', np.mean(self.disc_ls), 'disc_rl_score:',
                      np.mean(self.disc_rl_score), 'disc_fk_score:', np.mean(self.disc_fk_score), '\033[0m', end='\n\n')

                log_dict_to_tensorboard({'disc_ls': np.mean(self.disc_ls), 'disc_rl_score': np.mean(self.disc_rl_score),
                                         'dict_fk_score': np.mean(self.disc_fk_score)}, category='disc_perf',
                                        step=self.epoch)

                self.epoch += 1
                self.disc_ls_epoch.append(np.mean(self.disc_ls))

                if is_loss_stagnated(self.disc_ls_epoch, window_size=10, threshold=1e-3) or \
                        np.mean(self.disc_ls) < self.target_disc_loss:
                    print('stopping disc training as discriminator_loss has stagnated or target ls reached')
                    break

            torch.save(self.discriminator, os.path.join(CHECKPOINT_DIR,
                                                        'Resnet_disc_model_{}_{}.pth'.format('pretrain', self.epoch)))

    def get_state(self, fake_spacer, scalar=False):

        obs_imgs = np.expand_dims(fake_spacer, axis=-1)
        if scalar:
            obs_imgs = OrderedDict(zip(list(self.observation_space.spaces.keys())[:-1], obs_imgs))
            obs_imgs['scalar'] = self.avg_brightness
        else:
            obs_imgs = OrderedDict(zip(list(self.observation_space.spaces.keys()), obs_imgs))
        return obs_imgs

    def reset(self):
        self.time_step += 1
        self.episodes += 1
        self.steps = 0

        fake_spacer, _ = self.get_fake_data(reset=True)
        fake_spacer = self.match_obs_space(fake_spacer)
        self.state = self.get_state(fake_spacer)

        return self.state

    def step(self, action):
        device = self.disc_device
        #   Update and reset attributes
        global break_outer_loop
        self.steps += 1
        self.reward = 0
        self.time_step += 1
        if self.steps == 1:
            self.generator_loss_mean = []
            self.generator_acc_mean = []
            self.avg_brightness_mean = []

        #   Take action and collect observations
        actual_spacer, fake_spacer, info = self.get_data(action)

        self.avg_brightness = np.array(np.mean(fake_spacer, axis=(0, 1, 2)) / 255)

        #   Get state for RL
        fake_spacer = self.match_obs_space(fake_spacer)
        self.state = self.get_state(fake_spacer)

        #   Check if the action has failed
        #   Calculate loss that is mse, l1_loss and cross_entropy
        criterion_mse = nn.MSELoss()
        actual_brightness = to_device(torch.from_numpy(self.avg_brightness), device=device)

        self.mse = criterion_mse(self.mean_brightness, actual_brightness).detach().cpu().numpy().item()

        fake_spacer = fake_spacer / 255
        actual_spacer, fake_spacer = to_device(actual_spacer, device), to_device(torch.from_numpy(fake_spacer.copy())
                                                                                 .unsqueeze(1).float(), device)

        self.l1_loss = torch.mean(torch.abs(fake_spacer - actual_spacer)).detach().cpu().numpy().item()

        #   Getting Generator loss, by trying to fool the discriminator
        self.discriminator.eval()
        with torch.no_grad():
            preds = self.discriminator(fake_spacer)
        #   Fake is termed as real
        targets = to_device(torch.tensor([[1, 0] for _ in range(preds.size(0))]).float(), device)

        self.cross_entropy = F.binary_cross_entropy(preds, targets).detach().cpu().numpy().item()
        self.generator_acc = torchmetrics.functional.accuracy(preds, targets, task='binary').cpu().numpy().item()

        #   Calculate Reward
        self.reward = -self.cross_entropy - (100 * self.l1_loss)

        self.generator_acc_mean.append(self.generator_acc)
        self.generator_loss_mean.append(self.cross_entropy)
        self.avg_brightness_mean.append(self.avg_brightness)
        self.done = self.chk_termination()
        self.disc_buffer_act_spacer.append(actual_spacer)
        self.disc_buffer_fake_spacer.append(fake_spacer)

        if self.done:
            self.generator_loss.append(np.mean(self.generator_loss_mean))
            print('\033[1mgen_acc_mean:', np.mean(self.generator_acc_mean), 'target_gen&disc_ls', self.target_gen_loss,
                  'gen_loss_mean:', np.mean(self.generator_loss_mean), '\033[0m', end='\n\n')

        #   Collect Data's in lists if Discriminator is to be retrained.
        if self.done and self.train_disc:
            self.target_gen_loss = 1 * math.exp(-self.disc_train_epoch / 10)
            #   Discriminator Training
            if (self.done and self.generator_loss[-1] < self.target_gen_loss) \
                    or is_loss_stagnated(self.generator_loss):
                self.disc_train_epoch += 1
                self.target_disc_loss = 1 * math.exp(-self.disc_train_epoch / 10)

                buffer_act_spacer = torch.stack(list(self.disc_buffer_act_spacer))
                buffer_fake_spacer = torch.stack(list(self.disc_buffer_fake_spacer))
                break_outer_loop = False
                while True:
                    self.epoch += 1
                    #   Generate a random permutation of indices along the second dimension
                    # same indices for both actual and real spacer
                    indices0 = torch.randperm(buffer_act_spacer.size(1), device=device)
                    indices1 = torch.randperm(buffer_fake_spacer.size(1), device=device)
                    #   Use the index_select function to shuffle the tensor along the second dimension
                    buffer_act_spacer, buffer_fake_spacer = torch.index_select(buffer_act_spacer, 1, indices0), \
                        torch.index_select(buffer_fake_spacer, 1, indices1)
                    self.disc_fk_score, self.disc_rl_score, self.disc_ls = [], [], []

                    for actual_spacer_n, fake_spacer_n in zip(buffer_act_spacer, buffer_fake_spacer):
                        discriminator_loss, disc_real_score, disc_fake_score = \
                            self.train_discriminator(actual_spacer_n, fake_spacer_n)

                        self.disc_ls.append(discriminator_loss)
                        self.disc_rl_score.append(disc_real_score)
                        self.disc_fk_score.append(disc_fake_score)

                    self.disc_ls_epoch.append(np.mean(self.disc_ls))

                    print('\033[1mEpoch:', self.epoch, 'disc_ls:', np.mean(self.disc_ls), 'disc_rl_score:',
                          np.mean(self.disc_rl_score), 'disc_fk_score:', np.mean(self.disc_fk_score), '\033[0m',
                          end='\n\n')

                    log_dict_to_tensorboard(
                        {'disc_ls': np.mean(self.disc_ls), 'disc_rl_score': np.mean(self.disc_rl_score),
                         'dict_fk_score': np.mean(self.disc_fk_score)}, category='disc_perf',
                        step=self.epoch)

                    if is_loss_stagnated(self.disc_ls_epoch, window_size=self.episode_length * 2,
                                         threshold=1e-6) or np.mean(self.disc_ls) < self.target_disc_loss:
                        print('\033[stopping disc training as discriminator_loss has stagnated'
                              ' or disc_loss{} < target{}'.format(np.mean(self.disc_ls), self.target_disc_loss),
                              '\033[0m')
                        break

                    self.discriminator_loss, self.disc_real_score, self.disc_fake_score = np.mean(self.disc_ls), \
                        np.mean(self.disc_rl_score), np.mean(self.disc_fk_score)

                    torch.save(self.discriminator, os.path.join(CHECKPOINT_DIR, 'Resnet_disc_model_{}_ep{}.pth'.format(
                        self.time_step, self.epoch)))

                # self.target_gen_loss = self.target_gen_loss * math.exp(-self.disc_train_epoch*10)

            #   Logging when discriminator trains
            log_info = self.get_attributes(['time_step', 'episodes', 'steps', 'l1_loss', 'cross_entropy',
                                            'disc_real_score', 'discriminator_loss', 'disc_fake_score', 'done',
                                            'avg_brightness', 'generator_acc', 'mse', 'reward'])
        else:  # If discriminator is not trained
            log_info = self.get_attributes(['time_step', 'episodes', 'steps', 'l1_loss', 'cross_entropy', 'done',
                                            'avg_brightness', 'generator_acc', 'mse', 'reward'])

        gen_parameters = self.get_attributes(['episodes', 'cross_entropy', 'mse', 'l1_loss', 'avg_brightness',
                                              'reward', 'generator_acc'])

        # self.action_paired.update({'failed_action': failed_action})
        log_dict_to_tensorboard(self.action_paired, category='action', step=self.time_step)
        log_dict_to_tensorboard(gen_parameters, category='gen_param', step=self.time_step)
        log_to_file(log_info, train_log)
        return self.state, self.reward, self.done, info


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/mohanty/PycharmProjects/Data/data_128/train/good/',
                        help='Cropped spacer images for discriminator training')
    parser.add_argument('--img_size', nargs='*', type=int, default=[128, 128], help='img_size of disc training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for PPO training')
    parser.add_argument('--batches_in_episode', type=int, default=44, help='Episode length = batch_size * '
                                                                            'batches_in_episode')
    parser.add_argument('--lr_discriminator', type=float, default=0.00001, help='Learning rate for discriminator')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for discriminator optimizer')
    parser.add_argument('--device_discriminator', type=int, default=0, help='Discriminator device')
    parser.add_argument('--retrain_disc', type=bool, default=False, help='Discriminator device')

    parser.add_argument('--lr_generator', type=float, default=0.0001, help='Learning rate for generator PPO')
    parser.add_argument('--total_steps', type=int, default=100000, help='Total steps for PPO to be trained')
    parser.add_argument('--device_generator', type=int, default=1, help='Generator device')
    parser.add_argument('--blender_add', type=int, default=10, help='Blendtorch launcher address')
    parser.add_argument('--blend_file', type=str, default='spacer1_normal_22.6_exp_no_mesh_6action.blend',
                        help='blend_file aligned with code in spacer.blend.py')
    return parser.parse_args()


def main(data_dir, img_size, batch_size, batches_in_episode, lr_discriminator, weight_decay,
         device_discriminator, retrain_disc, lr_generator, total_steps, device_generator, blender_add, blend_file):
    batch_size = batch_size
    # * 34  # Roll_out Buffer Size/ How many steps in an episode*50
    episode_length = batch_size * batches_in_episode

    device_0 = get_device('{}'.format(device_discriminator))
    device_1 = get_device('{}'.format(device_generator))
    disc_file = '/home/mohanty/PycharmProjects/train_logs_disc/spacer2023-04-14 ' \
                '02:14:14/PPO_model/Resnet_disc_model_pretrain_4.pth'
    discriminator = torch.load(disc_file)
    discriminator = to_device(discriminator, device_0)
    writer.add_text("hyper_parameters", "disc_model: " + str(discriminator.model_name), global_step=5)
    writer.add_text("hyper_parameters", "disc_model: " + disc_file, global_step=5)
    print("batch_size:", batch_size, 'episode_length:', episode_length)

    py_env = Monitor(Penv(img_size=img_size, batch_size=batch_size, episode_length=episode_length, dat_dir=data_dir,
                          discriminator=discriminator, lr_discriminator=lr_discriminator, weight_decay=weight_decay,
                          device_discriminator=device_0, train_discriminator=retrain_disc, blender_add=blender_add,
                          blend_file=blend_file))
    # obs = Py_env.reset()
    # sample_obs = Py_env.observation_space.sample()
    # env_checker.check_env(Py_env,  warn=True)
    policy_kwargs = dict(net_arch=dict(pi=[100, 50], qf=[100, 50]),
                         features_extractor_class=CustomImageExtractor)

    logging_callback = TrainAndLoggingCallback(check_freq=episode_length, save_path=CHECKPOINT_DIR)

    # Separate evaluation env
    # Stop training if there is no improvement after more than 3 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=2, verbose=1)
    # Evaluate the call back every 3000 steps
    # eval_callback = EvalCallback(py_env, eval_freq=2, callback_after_eval=stop_train_callback, verbose=1)

    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

    n_actions = py_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG('MultiInputPolicy', py_env, tensorboard_log=LOG_DIR, verbose=1, action_noise=action_noise, learning_rate=lr_generator,
                 batch_size=batch_size, learning_starts=episode_length, buffer_size=5000, train_freq=(1, 'episode'),
                 gamma=.95, policy_kwargs=policy_kwargs, seed=seed_, device=device_1, gradient_steps=10)

    model.set_logger(new_logger)

    #   Multi-processed RL Training
    model.learn(total_timesteps=total_steps, callback=logging_callback, log_interval=1, tb_log_name="first_run",
                reset_num_timesteps=False)
    model.save(FINAL_MODEL_DIR + '{}'.format(total_steps))
    torch.save(discriminator, FINAL_MODEL_DIR + '{}'.format(total_steps) + 'resnet.pth')
    writer.close()


if __name__ == '__main__':
    args = parse_arguments()
    args_dict = vars(args)
    writer.add_text("hyper_parameters", "Arguments0: " + str(args_dict), global_step=4)

    main(args.data_dir, tuple(args.img_size), args.batch_size, args.batches_in_episode, args.lr_discriminator,
         args.weight_decay, args.device_discriminator, args.retrain_disc, args.lr_generator, args.total_steps,
         args.device_generator, args.blender_add, args.blend_file)
