import time
import gym
import stable_baselines3
from gym import spaces
import numpy as np
import os
import sys
import random
from PIL import Image

sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs/')
import spacer_gym
import resnet
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
from stable_baselines3.common.env_checker import check_env
import datetime
import logging
import torchmetrics

seed_ = 0
torch.manual_seed(seed_)
random.seed(seed_)
np.random.seed(seed_)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
CHECKPOINT_DIR = 'train_logs/spacer{}/DDPG_model'.format(stamp)
LOG_DIR = 'train_logs/spacer{}/DDPG_log'.format(stamp)
FINAL_MODEL_DIR = 'train_logs/spacer{}/DDPG_final_model'.format(stamp)
FINAL_R_BUFFER_DIR = 'train_logs/spacer{}/DDPG_BUFFER_model'.format(stamp)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.makedirs(FINAL_R_BUFFER_DIR, exist_ok=True)

log_file = LOG_DIR + 'discriminator_training.csv'
log_file2 = LOG_DIR + 'correct_actions.csv'


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

    def on_step(self):
        self.logger.record('GAN_loss/discriminator_loss', self.training_env.get_attr('discriminator_loss')[0])
        self.logger.record('GAN_loss/generator_loss', self.training_env.get_attr('generator_loss')[0])
        self.logger.record('GAN_loss/disc_real_score', self.training_env.get_attr('disc_real_score')[0])
        self.logger.record('GAN_loss/disc_fake_score', self.training_env.get_attr('disc_fake_score')[0])
        self.logger.record('rollout/done_cond', self.training_env.get_attr('done_cond')[0])
        self.logger.record('rollout/reward', self.training_env.get_attr('reward')[0])
        return True


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
    return 2 - (loss / (-torch.log(torch.tensor(1e-4))))


def augument(image):
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


def reshape_obs(observation, img_size=(256, 256)):
    obs = [np.asarray(observation.resize(img_size), dtype=np.float64) / 255]
    obs = np.reshape(obs[0], img_size + tuple([1]))
    return obs


device_0 = get_device('0')
device_1 = get_device('1')
device_0 = device_1
discriminator = resnet.resnet18(1, 2)
discriminator = to_device(discriminator, device_0)


def lr_schedule(step):
    return max(0.00006, 0.001 * np.power(0.1, step / 500))


weight_decay = 0.00002
opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lr_schedule)


def train_discriminator(real_images, fake_images, opt_d):
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
    loss.backward()
    opt_d.step()
    # scheduler.step()
    return loss.item(), real_score, fake_score


def binary_cross_entropy_loss(predictions, labels):
    """
    Computes binary cross entropy loss given predictions and one-hot encoded labels.
    """
    predictions = torch.clamp(predictions, min=1e-8, max=1 - 1e-8)
    return -torch.sum(labels * torch.log(predictions) + (1 - labels) * torch.log(1 - predictions))


def log_to_file(log_info, log_file2):
    with open(log_file2, 'a', newline='') as csvfile2:
        writer = csv.DictWriter(csvfile2, fieldnames=list(log_info.keys()))
        if csvfile2.tell() == 0:
            writer.writeheader()

        writer.writerow(log_info)


def find_range(number, dictionary):
    for key, value in dictionary.items():
        if value[0] <= number <= value[1]:
            return key
    return None


class Penv(gym.Env):
    def __init__(self):
        self.done_cond = None
        self.disc_fake_score = None
        self.disc_real_score = None
        self.discriminator_loss = None
        self.generator_loss = None
        self.done_threshold = [0.1282, 0.1618]  # derived by calculating max avg value over all the available actual
        # spacer data
        self.env_id = "blendtorch-spacer-v2"
        self.environments = gym.make(self.env_id, address=5, real_time=False)
        self.state = [25]
        self.reward = 0
        self.spacer_data_dir = 'spacer_data/train/'
        self.spacer_data = os.listdir(self.spacer_data_dir)
        self.image_size = (int(448), int(448))
        self.action_space = spaces.Box(0, 1, shape=(25,))
        self.observation_space = spaces.Box(low=0, high=255, shape=(256, 256, 1), dtype=np.uint8)
        self.episodes = 0
        self.steps = 0
        logging.basicConfig(filename=log_file, level=logging.INFO)
        logging.basicConfig(filename=log_file2, level=logging.INFO)
        self.done_cond_list = []

    def get_real_sp(self):
        if not self.spacer_data:
            self.spacer_data = os.listdir(self.spacer_data_dir)
        filename = random.choice(self.spacer_data)
        if filename.endswith('.png'):
            actual_spacer = Image.open(os.path.join(self.spacer_data_dir, filename))
            self.spacer_data.remove(filename)
            return actual_spacer

    def chk_termination(self):
        if max(self.done_threshold) > self.done_cond > min(self.done_threshold) and self.steps < 500:
            return False
        else:
            return True

    def reset(self):
        self.episodes += 1
        self.steps = 0
        obs_ = self.environments.reset()
        self.state = reshape_obs(obs_, (256, 256))
        done_cond = np.average(self.state)
        print('reset_done_cond', done_cond)
        return self.state

    def step(self, action):
        self.reward = 0
        self.steps += 1
        #   Take action and collect observations
        obs_ = self.environments.step(action)
        self.state, _, done, info = reshape_obs(obs_[0], (256, 256)), obs_[1], obs_[2], obs_[3]

        #   Start Reward shaping
        real_spacer = self.get_real_sp()
        real_spacer = np.asarray(real_spacer.resize(self.image_size), dtype=np.float64) / 255
        fake_spacer = augument(obs_[0])
        fake_spacer = np.asarray(fake_spacer, dtype=np.float64) / 255
        actual, fake = get_trainable_data(real_spacer, int(self.image_size[0] / 2), device_0), \
            get_trainable_data(fake_spacer, int(self.image_size[0] / 2), device_0)

        #   Train discriminator
        self.discriminator_loss, self.disc_real_score, self.disc_fake_score = train_discriminator(actual, fake, opt_d)

        #   Try to fool the discriminator
        discriminator.eval()
        preds = discriminator(fake)
        targets = to_device(torch.tensor([[1, 0] for _ in range(preds.size(0))]).float(),
                            device_0)  # fake is told as real
        self.generator_loss = F.binary_cross_entropy(preds, targets).detach().cpu().numpy().item()

        # generator_loss = 1 if generator_loss > 1 else generator_loss

        self.reward = normalize_loss(self.generator_loss).item()
        self.done_cond = np.average(self.state)
        self.done_cond_list.append(self.done_cond)

        done = self.chk_termination()
        # Log actions
        if not done:
            self.reward += 0.5

            # log_info = {"actions": action, "img_avg": self.done_cond}
            # log_to_file(log_info, log_file2)

        # print("episode: {}, steps: {}, avg_image: {:.2f}, done:{}, generator_loss: {:.2f}, disc_loss: {:.2f}, "
        #       "reward: {:.2f}".format(self.episodes, self.steps, self.done_cond, done, self.generator_loss,
        #                               self.discriminator_loss, reward))

        return self.state, self.reward, done, info

    def get_attr(self, att_name):
        if att_name == 'discriminator_loss':
            return self.discriminator_loss
        elif att_name == 'generator_loss':
            return self.generator_loss
        elif att_name == 'disc_real_score':
            return self.disc_real_score
        elif att_name == 'disc_fake_score':
            return self.disc_fake_score
        elif att_name == 'done_cond':
            return self.done_cond
        elif att_name == 'reward':
            return self.reward


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

        self.cnn = torch.nn.Sequential(*list(resnet.resnet18(n_input_channels, 2).children())[:-1]).extend(
            [torch.nn.Flatten()])

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def main():
    Py_env = Penv()
    no_of_actions = Py_env.action_space.shape[0]

    # obs = Py_env.reset()
    # global i
    # time_total = 0
    # for i in range(5):
    #     start = time.time()
    #     action_dummy = Py_env.action_space.sample()
    #     obs_ = Py_env.step(np.array(action_dummy))
    #     time_one_iter = time.time() - start
    #     time_total += time_one_iter
    #     print("time taken_iter{}:".format(i), time_one_iter)
    # print('total time over_ 2 iteration: ', time_total / (i + 1))

    # print("# Learning")

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128))

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(no_of_actions), sigma=0.1 * np.ones(no_of_actions))

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func

    # linear_schedule(0.00006)
    model = DDPG("CnnPolicy", Py_env, policy_kwargs=policy_kwargs, tensorboard_log=LOG_DIR, learning_starts=500,
                 learning_rate=0.0001, batch_size=10, gamma=.95, verbose=1, device=device_0,
                 buffer_size=1500, train_freq=10, action_noise=action_noise, seed=seed_)

    callback = TrainAndLoggingCallback(check_freq=250, save_path=CHECKPOINT_DIR)

    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    #   Multi-processed RL Training
    model.learn(total_timesteps=5000, callback=callback, log_interval=1, tb_log_name="first_run",
                reset_num_timesteps=False)
    msg = input("Press Enter to continue...")
    if msg == 'train':
        model.learn(total_timesteps=2500, callback=callback, log_interval=1, tb_log_name="second_run",
                    reset_num_timesteps=False)
    msg = input("Press Enter to continue...")
    if msg == 'train':
        model.learn(total_timesteps=2500, callback=callback, log_interval=1, tb_log_name="third_run",
                    reset_num_timesteps=False)
    if msg == 'end':
        model.save(FINAL_MODEL_DIR)
        model.save_replay_buffer(FINAL_R_BUFFER_DIR)

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
