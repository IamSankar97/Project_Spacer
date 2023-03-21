import gym
from gym import spaces
import numpy as np
import os
import sys
import random
from PIL import Image, ImageFilter
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
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
import datetime
import logging
import torchmetrics
from collections import deque
from utils import get_circular_corps, get_orth_actions
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import cv2

seed_ = 0

torch.manual_seed(seed_)
random.seed(seed_)
np.random.seed(seed_)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_path = '/home/mohanty/PycharmProjects/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
CHECKPOINT_DIR = log_path + 'train_logs/spacer{}/PPO_model/'.format(stamp)
LOG_DIR = log_path + 'train_logs/spacer{}/PPO_log'.format(stamp)
FINAL_MODEL_DIR = log_path + 'train_logs/spacer{}/PPO_final_model'.format(stamp)
FINAL_R_BUFFER_DIR = log_path + 'train_logs/spacer{}/PPO_BUFFER_model'.format(stamp)
ACTION_LOG_DIR = LOG_DIR + '/action_log'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ACTION_LOG_DIR, exist_ok=True)
train_log = LOG_DIR + 'training_log.csv'
writer = SummaryWriter(ACTION_LOG_DIR)


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


def get_trainable_data(image, crop_size, assign_device):
    crop_images = [torch.from_numpy(np.array([image[i: i + crop_size, j: j + crop_size]])) for i in
                   range(0, int(crop_size * 2), crop_size)
                   for j in range(0, int(crop_size * 2), crop_size)]
    crop_images = torch.stack(crop_images)
    crop_images = crop_images.float()
    return to_device(crop_images, assign_device)


def binary_accuracy(y_pred, y_true):
    y_pred_label = (y_pred > 0.7).long()  # round off the predicted probability to the nearest label (0 or 1)
    correct = torch.eq(y_pred_label, y_true)
    acc = torch.mean(correct.float())
    return acc


def normalize_loss(loss):
    return 1 - (loss / (-torch.log(torch.tensor(1e-1))))


def augument(image):
    image = image.filter(ImageFilter.GaussianBlur(0.5))
    random_number = random.randint(0, 3)
    if random_number == 0:
        return image
    elif random_number == 1:
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif random_number == 2:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(Image.FLIP_LEFT_RIGHT)


def reshape_image(image, img_size=(256, 256)):
    if image.size == img_size:
        image = np.asarray(image, dtype=np.float32) / 255
    else:
        image = np.asarray(image.resize(img_size), dtype=np.float32) / 255
    return image


device_1 = get_device('1')
device_0 = device_1  # get_device('0')
discriminator = resnet.resnet10(1, 2)

print(discriminator)

opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))

discriminator = to_device(discriminator, device_0)


def train_discriminator(real_images, fake_images, opt_d, clip=False):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_images = real_images.float()
    fake_images = fake_images.float()

    real_preds = discriminator(real_images)
    real_targets = to_device(torch.tensor([[1, 0] for _ in range(real_preds.size(0))]).float(), device_0)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torchmetrics.functional.accuracy(real_preds, real_targets, task='binary').cpu().numpy().item()

    # Pass fake images through discriminator
    fake_preds = discriminator(fake_images)
    fake_targets = to_device(torch.tensor([[0, 1] for _ in range(fake_preds.size(0))]).float(), device_0)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torchmetrics.functional.accuracy(fake_preds, fake_targets, task='binary').cpu().numpy().item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    if clip:
        nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1, norm_type=2)
    loss.backward()
    opt_d.step()
    # scheduler.step()
    return loss.item(), real_score, fake_score


def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))


def binary_cross_entropy_loss(predictions, labels):
    """
    Computes binary cross entropy loss given predictions and one-hot encoded labels.
    """
    predictions = torch.clamp(predictions, min=1e-8, max=1 - 1e-8)
    return -torch.sum(labels * torch.log(predictions) + (1 - labels) * torch.log(1 - predictions))


def log_to_file(log_info, log_file):
    with open(log_file, 'a', newline='') as csvfile2:
        writer = csv.DictWriter(csvfile2, fieldnames=list(log_info.keys()))
        if csvfile2.tell() == 0:
            writer.writeheader()

        writer.writerow(log_info)


def find_range(number, dictionary):
    for key, value in dictionary.items():
        if value[0] <= number <= value[1]:
            return key
    return None


def _center_crop_to_square(img: Image):
    width, height = img.size
    size = min(width, height)
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return img.crop((left, top, right, bottom))


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return 0.00001

    return func


def get_fake_sp(obs):
    obs = np.asarray(augument(obs), dtype=np.uint8)
    return obs


def center_ring(actual_spacer):
    circles = cv2.HoughCircles(actual_spacer, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=100, minRadius=0,
                               maxRadius=0)
    height, width = actual_spacer.shape
    center_x = width // 2
    center_y = height // 2
    x, y, r = np.uint16(np.around(circles[0][0]))
    delta_x = center_x - x
    delta_y = center_y - y
    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    centered_ring = cv2.warpAffine(actual_spacer, M, (width, height))
    return centered_ring


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
            gen_model_path = os.path.join(self.save_path, 'PPO_gen_model_{}'.format(self.n_calls))
            disc_model_path = os.path.join(self.save_path, 'Resnet_disc_model_{}.pth'.format(self.n_calls))
            torch.save(discriminator, disc_model_path)
            self.model.save(gen_model_path)

            torch.save(discriminator, disc_model_path)
            self.model.save(gen_model_path)

            self.logger.record('GAN_loss/ep_disc_loss', self.training_env.get_attr('discriminator_loss')[0])
            self.logger.record('GAN_loss/ep_gen_loss', np.mean(self.training_env.get_attr('generator_loss_mean')))
            self.logger.record('GAN_loss/generator_acc_mean', np.mean(self.training_env.get_attr('generator_acc_mean')))
            self.logger.record('GAN_loss/ep_disc_rl_score', self.training_env.get_attr('disc_real_score')[0])
            self.logger.record('GAN_loss/ep_disc_fk_score', self.training_env.get_attr('disc_fake_score')[0])
            self.logger.record('GAN_loss/done_cond_mean', np.mean(self.training_env.get_attr('done_cond_mean')))
            self.logger.record('rollout/done_cond', self.training_env.get_attr('done_cond')[0])
            self.logger.record('rollout/ep_end_reward', self.training_env.get_attr('reward')[0])


class CustomImageExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['0image'].shape[0]
        self.cnn = torch.nn.Sequential(*list(resnet.resnet10(n_input_channels, 2).children())[:-1]).extend(
            [torch.nn.Flatten()])

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten_img = self.cnn(torch.as_tensor(observation_space['0image'].sample()[None]).float()).shape[1]
        self.n_scalar = observation_space['scalar'].shape[0]
        self.n_img = len(observation_space.spaces) - self.n_scalar
        self.dropout0 = nn.Dropout(p=0.3)
        self.linear0 = nn.Linear(n_flatten_img, 100)
        self.linear1 = nn.Linear(100 * self.n_img, features_dim * 3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(features_dim * 3, features_dim-self.n_scalar)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        scalar_obs = observations['scalar']
        del observations['scalar']

        img_features = []
        for _, value in observations.items():
            cnn_output = self.cnn(value)
            cnn_drop = self.dropout0(cnn_output)
            linear0 = torch.relu(self.linear0(cnn_drop))
            img_features.append(linear0)
            # cnn_reshape = torch.reshape(cnn_output, (img_obs.shape[0], self.select_crops.size()[0], cnn_output.shape[1]))

        cnn_output = torch.cat(img_features, dim=1)
        # flat = torch.unsqueeze(torch.flatten(cnn_output, start_dim=0), dim=0)

        x = torch.relu(self.linear1(cnn_output))
        x = self.dropout1(x)
        x = torch.relu(self.linear2(x))
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
    def __init__(self, batch_size, episode_length):
        super(Penv, self).__init__()
        self.epoch = 0
        self.action_paired = {}
        self.spacer_data_dir = '/home/mohanty/PycharmProjects/Data/spacer_data/train_64*64*32/good/'
        self.environments = gym.make("blendtorch-spacer-v2", address=1, real_time=False)
        self.environments.reset()
        # logging must be after environment generation
        logging.basicConfig(filename=train_log, level=logging.INFO)
        self.action_space = spaces.Box(-1, 1, shape=(11,))
        self.observation_space = spaces.Dict(self._get_obs_space())
        self.actual_dataloader = self.get_image_dataloader()
        self.initialize_discriminator(no_of_stps=500)
        self.done = 0
        self.l1_ = 0
        self.crose_entropy = 0
        self.done_cond = 0
        self.done_cond_mean = []
        self.disc_fake_score = 0
        self.disc_real_score = 0
        self.discriminator_loss = 0
        self.done_threshold = np.array([0.85, 0.30]) # derived by calculating max avg value over all the available actual
        # spacer data
        self.state = [32]
        self.reward = 0
        self.spacer_data = os.listdir(self.spacer_data_dir)
        self.episodes = 0
        self.episode_length = episode_length
        self.steps = 0
        self.done_cond_list = []
        self.generator_acc_mean = []
        self.done_cond_mean = []
        self.time_step = -1
        self.batch_size = batch_size
        self.generator_loss_mean = deque(maxlen=self.episode_length)
        self.buffer_act_spacer = deque(maxlen=self.episode_length)  # * len(self.observation_space.spaces))
        self.buffer_fake_spacer = deque(maxlen=self.episode_length)

    def get_fake_data(self, action=None, reset=False):
        if reset:
            obs, info = self.environments.reset(), None
        else:
            obs, _, _, info = self.environments.step(action)
            self.action_paired = info['action_pair']
            info.pop('action_pair')
        fake_spacer = np.array([np.array(obs.crop((i, 0, i + 64, 64))) for i in range(0, obs.size[0], 64)])

        return fake_spacer, info

    def _get_obs_space(self):
        fake_spacer, _ = self.get_fake_data(self.action_space.sample())
        self.no_img_obs = len(fake_spacer)
        obs_space = {}
        for i in range(self.no_img_obs):
            obs_space.update({'{}image'.format(i): spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)})
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
            dataloader = dataloader.to(device)

        return dataloader

    def chk_termination(self):
        # max(self.done_threshold) > self.done_cond > min(self.done_threshold) and
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

    def initialize_discriminator(self, no_of_stps=10, device=device_0):
        global actual_spacer
        dicc_fk_score, disc_rl_score, disc_ls = [], [], []
        ortho_actions = get_orth_actions(self.action_space.shape[0])

        orth_action = pd.DataFrame(ortho_actions)

        noise_std, theta, dt = 0.12, 0.15, 1e-2

        for i in range(no_of_stps):

            #   Take action and collect observations
            noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(self.action_space.shape),
                                                 sigma=np.array(noise_std), theta=theta, dt=dt)
            action = np.array(orth_action.sample())
            noisy_action = np.squeeze(np.clip(action + noise(), self.action_space.low, self.action_space.high))

            actual_spacer, fake_spacer, info = self.get_data(noisy_action)
            fake_spacer = self.match_obs_space(fake_spacer)

            #   MAKING SOME IMAGES TO COMPLETE BLACK IMAGES
            num_zero_images = int(fake_spacer.shape[0] * np.random.random())

            mask = np.random.rand(fake_spacer.shape[0]) < (num_zero_images / fake_spacer.shape[0])
            fake_spacer[mask] = 0

            actual_spacer, fake_spacer = actual_spacer, fake_spacer / 255

            actual_spacer = to_device(actual_spacer, device)
            fake_spacer = to_device(torch.from_numpy(fake_spacer.copy()).unsqueeze(1).float(), device)

            self.discriminator_loss, self.disc_real_score, self.disc_fake_score = train_discriminator(actual_spacer,
                                                                                                      fake_spacer,
                                                                                                      opt_d)

            dicc_fk_score.append(self.disc_fake_score)
            disc_ls.append(self.discriminator_loss)
            disc_rl_score.append(self.disc_real_score)

            print('pre_train_step:', i, 'discrim_loss:', np.mean(disc_ls), 'disc_rl_score:',
                  np.mean(disc_rl_score), 'disc_fk_score:', np.mean(dicc_fk_score), end='\n\n')

            log_dict_to_tensorboard({'disc_ls': np.mean(disc_ls), 'disc_rl_score': np.mean(disc_rl_score),
                                     'dict_fk_score': np.mean(dicc_fk_score)}, category='disc_perf', step=self.epoch)

            if np.mean(disc_ls) < 0.1:
                print('stopping pretraining as disc_fk_score {} > 0.95 and discriminator_loss {}'.format(
                    np.mean(dicc_fk_score), np.mean(disc_ls)))
                break
            self.epoch += 1
        torch.save(discriminator, os.path.join(CHECKPOINT_DIR, 'Resnet_disc_model_{}.pth'.format('pretrain')))

    def get_state(self, fake_spacer):

        obs_imgs = np.expand_dims(fake_spacer, axis=-1)
        obs_imgs = OrderedDict(zip(list(self.observation_space.spaces.keys())[:-1], obs_imgs))
        obs_imgs['scalar'] = self.done_cond
        return obs_imgs

    def reset(self):
        self.time_step += 1
        self.episodes += 1
        self.steps = 0

        fake_spacer, _ = self.get_fake_data(reset=True)
        fake_spacer = self.match_obs_space(fake_spacer)
        self.state = self.get_state(fake_spacer)

        return self.state

    def step(self, action, device=device_0):
        global done_cond_lst, disc_ls, disc_fk_sc, disc_rl_sc, discriminator_loss
        self.steps += 1
        self.reward = 0
        self.time_step += 1
        failed_action = 0
        if self.steps == 1:
            self.generator_loss_mean = []
            self.generator_acc_mean = []
            self.done_cond_mean = []

        #   Take action and collect observations
        actual_spacer, fake_spacer, info = self.get_data(action)
        self.done_cond = np.array(np.mean(fake_spacer, axis=(0, 1, 2)) / 255)
        fake_spacer = self.match_obs_space(fake_spacer)
        self.state = self.get_state(fake_spacer)
        don_cond_pixel = max(self.done_threshold) > self.done_cond > min(self.done_threshold)
        if np.all(fake_spacer == 0):
            failed_action = 1
            print('failed:', self.action_paired)
        criterion_mse = nn.MSELoss()
        criterion_kl = nn.KLDivLoss()
        x, y = torch.mean(to_device(torch.from_numpy(self.done_threshold), device=device)),\
            to_device(torch.from_numpy(self.done_cond), device=device)
        self.mse = criterion_mse(x, y).detach().cpu().numpy().item()
        self.kl = criterion_kl(x, y).detach().cpu().numpy().item()

        actual_spacer, fake_spacer = actual_spacer , fake_spacer / 255
        actual_spacer = to_device(actual_spacer, device)
        fake_spacer = to_device(torch.from_numpy(fake_spacer.copy()).unsqueeze(1).float(), device)

        self.buffer_act_spacer.append(actual_spacer)
        self.buffer_fake_spacer.append(fake_spacer)

        #   Getting Generator loss, Try to fool the discriminator
        discriminator.eval()
        with torch.no_grad():
            preds = discriminator(fake_spacer)
        # fake is told as real
        targets = to_device(torch.tensor([[1, 0] for _ in range(preds.size(0))]).float(), device)

        self.crose_entropy = F.binary_cross_entropy(preds, targets).detach().cpu().numpy().item()
        self.reward = -self.crose_entropy - self.mse - (0.1*self.kl)
        generator_acc = torchmetrics.functional.accuracy(preds, targets, task='binary').cpu().numpy().item()
        self.generator_acc_mean.append(generator_acc)
        self.generator_loss_mean.append(self.crose_entropy)
        self.done_cond_mean.append(self.done_cond)
        self.l1_ = l1_loss(fake_spacer, actual_spacer).detach().cpu().numpy().item()
        self.done = self.chk_termination()

        if self.done and np.mean(self.generator_loss_mean) < 0.1:  # == 0 and self.steps != 1:
            disc_ls, disc_rl_score, dicc_fk_score = [], [], []
            buffer_act_spacer = torch.stack(list(self.buffer_act_spacer))
            buffer_fake_spacer = torch.stack(list(self.buffer_fake_spacer))
            # Generate a random permutation of indices along the second dimension
            indices0, indices1 = torch.randperm(buffer_act_spacer.size(1), device=device_1), \
                torch.randperm(buffer_fake_spacer.size(1), device=device_1)

            # Use the index_select function to shuffle the tensor along the second dimension
            buffer_act_spacer, buffer_fake_spacer = torch.index_select(buffer_act_spacer, 1, indices0), \
                torch.index_select(buffer_fake_spacer, 1, indices1)
            for actual_spacer_n, fake_spacer_n in zip(buffer_act_spacer, buffer_fake_spacer):
                discriminator_loss, disc_real_score, disc_fake_score = train_discriminator(actual_spacer_n,
                                                                                       fake_spacer_n, opt_d, clip=True)
                self.epoch += 1
                disc_ls.append(discriminator_loss)
                disc_rl_score.append(disc_real_score)
                dicc_fk_score.append(disc_fake_score)
                print('episode:', self.episodes, 'epoch', self.epoch, 'discrim_loss:', np.mean(disc_ls), 'disc_rl_score:',
                      np.mean(disc_rl_score), 'disc_fk_score:', np.mean(dicc_fk_score), end='\n\n')
                self.disc_fake_score = np.mean(dicc_fk_score)

                log_dict_to_tensorboard({'disc_ls': np.mean(disc_ls), 'disc_rl_score': np.mean(disc_rl_score),
                                         'dict_fk_score': np.mean(dicc_fk_score)}, category='disc_perf',
                                        step=self.epoch)

                if np.mean(disc_ls) < 0.5:
                    print('stopping disc training as disc_fk_score {} < 0.5 or discriminator_loss {} < 0.5'.format(
                        np.mean(dicc_fk_score), np.mean(disc_ls)))
                    break

            self.discriminator_loss, self.disc_real_score, self.disc_fake_score = np.mean(disc_ls),\
                np.mean(disc_rl_score), np.mean(dicc_fk_score)

        print('generator_acc_mean:', np.mean(self.generator_acc_mean))
        log_info = {"timestep": self.time_step, "episode": self.episodes, "steps": self.steps,
                    "l1_loss": self.l1_, "entp_loss": self.crose_entropy, "disc_loss": self.discriminator_loss,
                    "real_score": self.disc_real_score, "fake_score": self.disc_fake_score,
                    "episode_terminate": self.done, 'don_cond_pixel': don_cond_pixel,
                    'generator_acc': generator_acc, 'done': self.done_cond, 'mse': self.mse,
                    "gen_loss": -self.reward, "reward": self.reward}

        self.action_paired.update({'failed_action': failed_action})
        gen_prameters = {"timestep": self.episodes, "entp_loss": self.crose_entropy,
                         'mse': self.mse, 'kl': self.kl, "l1_loss": self.l1_, 'done': self.done_cond,
                         "reward": self.reward, "real_score": self.disc_real_score, "fake_score": self.disc_fake_score,
                         'generator_acc': generator_acc}
        log_dict_to_tensorboard(self.action_paired, category='action', step=self.time_step)
        log_dict_to_tensorboard(gen_prameters, category='gen_param', step=self.time_step)
        log_to_file(log_info, train_log)
        return self.state, self.reward, self.done, info


def main():
    batch_size = 2
    episode_length = batch_size #* 34  # Roll_out Buffer Size/ How many steps in an episode*50
    print("batch_size:", batch_size, 'episode_length:', episode_length)
    discr_train_freq = episode_length
    Py_env = Penv(batch_size=batch_size, episode_length=episode_length)
    # obs = Py_env.reset()
    # sample_obs = Py_env.observation_space.sample()
    # env_checker.check_env(Py_env,  warn=True)
    policy_kwargs = dict(net_arch=dict(pi=[100, 64, 32], vf=[100, 64, 32]),
                         features_extractor_class=CustomImageExtractor)

    callback = TrainAndLoggingCallback(check_freq=episode_length, save_path=CHECKPOINT_DIR)

    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

    model = PPO('MultiInputPolicy', Py_env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001,
                batch_size=batch_size, n_steps=episode_length, n_epochs=10, clip_range=.1, gamma=.95, gae_lambda=.9,
                policy_kwargs=policy_kwargs,
                seed=seed_, device=device_1)
    # model = stable_baselines3.PPO.load("train_logs/pre_trained_5k/DDPG_model/PPO_gen_model_5000.zip")
    # model.set_env(Py_env)
    # model.policy_kwargs['tensorboard_log'] = LOG_DIR
    # model.policy_kwargs['device'] = device_0
    model.set_logger(new_logger)

    #   Multi-processed RL Training
    model.learn(total_timesteps=50000, callback=callback, log_interval=1, tb_log_name="first_run",
                reset_num_timesteps=False)
    model.save(FINAL_MODEL_DIR + '30k')
    torch.save(discriminator, FINAL_MODEL_DIR + '30k' + 'resnet.pth')

    # ----------------------------Below Code to be Updated-------------------------#
    # Evaluate the trained agent
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

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


if __name__ == '__main__':
    main()
