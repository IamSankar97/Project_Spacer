from PIL import Image
from utils import slid_window
import os
import numpy as np

spacer_dir = '/home/mohanty/PycharmProjects/Data/spacer_data/'
defect_dir = '/home/mohanty/PycharmProjects/Data/data_128/val/defect/'
good_dir = '/home/mohanty/PycharmProjects/Data/data_128/val/good/'

defect_im_dir = defect_dir+'image/'
defect_mask_dir = defect_dir+'mask/'

good_im_dir = good_dir+'image/'
good_mask_dir = good_dir+'mask/'

os.makedirs(defect_im_dir, exist_ok=True)
os.makedirs(defect_mask_dir, exist_ok=True)

os.makedirs(good_im_dir, exist_ok=True)
os.makedirs(good_mask_dir, exist_ok=True)

train_img_dir = spacer_dir + 'train/'
train_mask_dir = spacer_dir + 'train_masks/'

val_img_dir= spacer_dir + 'val/'
val_mask_dir = spacer_dir+ 'val_masks/'


train_imgs = os.listdir(train_img_dir)
total_def_imgs, total_mask_imgs, total_good_imgs, total_good_mask_imgs = [], [], [], []
count = 0
for im in train_imgs:
    image = Image.open(train_img_dir+im).convert('L')
    image_mask = Image.open(train_mask_dir+im).convert('L')

    defect_imgs, defect_mask_imgs, good_imgs, good_mask_imgs = slid_window(np.asarray(image), np.asarray(image_mask),win_size=(128, 128), pecentage_defect=0.1)

    imc = 0
    for def_img, def_mask in zip(defect_imgs, defect_mask_imgs):
        imc += 1
        def_img, def_mask = Image.fromarray(def_img).convert('L'), Image.fromarray(def_mask).convert('L')
        def_img.save(defect_im_dir+'{}_'.format(imc)+im)
        def_mask.save(defect_mask_dir+'{}_'.format(imc)+im)

    num_images = len(good_imgs)  # get the number of images in the list
    # indices = np.random.choice(num_images, size=len(num_images), replace=False)  # choose 100 random indices without replacement
    good_img_selected = np.array(good_imgs)#[indices]
    good_mask_img_selected = np.array(good_mask_imgs)#[indices]

    for good_img, good_mask in zip(good_img_selected, good_mask_img_selected):
        imc += 1
        good_img, good_mask = Image.fromarray(good_img).convert('L'), Image.fromarray(good_mask).convert('L')
        good_img.save(good_im_dir+'{}_'.format(imc)+im)
        good_mask.save(good_mask_dir+'{}_'.format(imc)+im)

    #
    # total_def_imgs.extend(defect_imgs)
    # total_mask_imgs.extend(total_mask_imgs)
    # total_good_imgs.extend(good_imgs)
    # total_good_mask_imgs.extend(good_mask_imgs)
    print(count := count + 1, end= ' ')
