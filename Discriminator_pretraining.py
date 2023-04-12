import datetime
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import resnet
from utils import is_loss_stagnated

seed_ = 0

torch.manual_seed(seed_)
random.seed(seed_)
np.random.seed(seed_)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_path = '/home/mohanty/PycharmProjects/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
CHECKPOINT_DIR = log_path + 'train_logs_disc/spacer{}/PPO_model/'.format(stamp)
LOG_DIR = log_path + 'train_logs_disc/spacer{}/PPO_log'.format(stamp)
FINAL_MODEL_DIR = log_path + 'train_logs_disc/spacer{}/PPO_final_model'.format(stamp)
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


def train_discriminator(discriminator, dic_optimizer, disc_device, real_images, fake_images, clip=False):
    # Clear discriminator gradients
    dic_optimizer.zero_grad()

    # Pass real images through discriminator
    real_images = real_images.float()
    fake_images = fake_images.float()

    real_preds = discriminator(real_images)
    real_targets = to_device(torch.tensor([[1, 0] for _ in range(real_preds.size(0))]).float(),
                             disc_device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torchmetrics.functional.accuracy(real_preds, real_targets, task='binary').cpu().numpy().item()

    # Pass fake images through discriminator
    fake_preds = discriminator(fake_images)
    fake_targets = to_device(torch.tensor([[0, 1] for _ in range(fake_preds.size(0))]).float(),
                             disc_device)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torchmetrics.functional.accuracy(fake_preds, fake_targets, task='binary').cpu().numpy().item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    if clip:
        nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1, norm_type=2)
    loss.backward()
    dic_optimizer.step()
    return loss.item(), real_score, fake_score


def get_image_dataloader(spacer_data_dir=None, shuffle=True):
    # Define the transform to apply to each image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(),  # Randomly flip images left-right
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Create the dataset
    dataset = ImageFolder(root=spacer_data_dir, transform=transform)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=shuffle, )

    return dataloader


# Create the train dataset
dir = '/home/mohanty/PycharmProjects/Data/spacer_data/synthetic_data2/Discriminator_pre_train'

# Create the train dataloader
train_dataloader = get_image_dataloader(dir, shuffle=True)


def main():
    discriminator = resnet.resnet10(1, 2)
    device = get_device('1')
    discriminator = to_device(discriminator, device)
    disc_device = get_device('1')
    dic_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    disc_ls_epoch = []
    for epoch in range(100):
        disc_fk_score, disc_rl_score, disc_ls = [], [], []
        batch = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            actual_spacer = inputs[labels == 0]

            # Separate inputs and labels for class B (label 1)
            fake_spacer = inputs[labels == 1]

            actual_spacer, fake_spacer = actual_spacer, fake_spacer

            # actual_spacer = to_device(actual_spacer, disc_device)
            # fake_spacer = to_device(torch.from_numpy(fake_spacer.copy()).unsqueeze(1).float(), disc_device)

            discriminator_loss, disc_real_score, disc_fake_score = train_discriminator(discriminator, dic_optimizer,
                                                                                       disc_device, actual_spacer,
                                                                                       fake_spacer, clip=True)
            disc_fk_score.append(disc_fake_score)
            disc_ls.append(discriminator_loss)
            disc_rl_score.append(disc_real_score)
            batch += 1

            print('\033[1mEpoch:', epoch, 'batch', batch, 'disc_ls:', np.mean(disc_ls), 'disc_rl_score:',
                  np.mean(disc_rl_score), 'disc_fk_score:', np.mean(disc_fk_score), '\033[0m', end='\n\n')

        log_dict_to_tensorboard({'disc_ls': np.mean(disc_ls), 'disc_rl_score': np.mean(disc_rl_score),
                                 'dict_fk_score': np.mean(disc_fk_score)}, category='disc_perf',
                                step=epoch)

        disc_ls_epoch.append(np.mean(disc_ls))

        if is_loss_stagnated(disc_ls_epoch, window_size=10, threshold=1e-3):
            print('stopping disc training as discriminator_loss has stagnated or target ls reached')
            torch.save(discriminator, os.path.join(CHECKPOINT_DIR,
                                                   'Resnet_disc_model_{}_{}.pth'.format('pretrain', epoch)))
            break
        if epoch != 0 and epoch % 25 == 0:
            torch.save(discriminator, os.path.join(CHECKPOINT_DIR,
                                                   'Resnet_disc_model_{}_{}.pth'.format('pretrain', epoch)))


if __name__ == '__main__':
    main()
