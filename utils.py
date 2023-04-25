import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import itertools
import random
from skimage.util import view_as_windows


def get_theta(r1, r2, theta0, distance):
    """

    :param r1: micrometer
    :param r2: micrometer
    :param theta0: degree
    :param distance: micrometer
    :return: radian
    """
    m = (r1 ** 2 + (r2 ** 2) - (distance ** 2)) / (2 * r1 * r2)
    if m > 1:
        m = 1
    elif m < -1:
        m = -1
    theta1 = (np.arccos(m)) + np.radians(theta0)
    return theta1


def closest_number_(df: pd.DataFrame, value: float, column: str):
    df.sort_values(by=column)
    df.reset_index(inplace=True)
    low = df.index[0]
    high = df.index[-1]
    while low <= high:
        mid = math.floor((low + high) / 2)
        if df.loc[mid, column] < value:
            low = mid + 1
        elif df.loc[mid, column] > value:
            high = mid - 1
        else:
            return df[column][mid], df['index'][mid]

    # If target is not found, return closest number
    if abs(df[column][low] - value) < abs(df[column][high] - value):
        return df[column][low], df['index'][mid]
    else:
        return df[column][high], df['index'][mid]


def closest_number_np(df: pd.DataFrame, value: float, column: str):
    df.reset_index(inplace=True)

    index = np.searchsorted(df[column], value, side='left')
    if index == len(df):
        return df[column].iat[-1], df['index'].iat[-1]
    elif index == 0:
        return df[column].iat[0], df['index'].iat[0]
    else:
        left_diff = abs(df[column].iat[index - 1] - value)
        right_diff = abs(df[column].iat[index] - value)
        if left_diff < right_diff:
            return df[column].iat[index - 1], df['index'].iat[index - 1]
        else:
            return df[column].iat[index], df['index'].iat[index]


def cart2pol(x, y):
    """

    :param x: micrometers
    :param y: micrometers
    :return: rho in micrometer phi in radian
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """
    :param rho: Radius in micrometer
    :param phi: degree in radians
    :return: x, y in micrometer
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def get_spacer_center_radius(img: np.array):
    assert img.shape[0] == img.shape[1]
    # Perform Canny edge detection on the image
    edges = cv2.Canny(img, 100, 150)

    # plt.imshow(edges, cmap='gray')
    # plt.show()

    # Get the coordinates of the outer and inner edges
    # Find the contours of the outer and inner edges
    outer_contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner_contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding rectangles of the outer and inner contours
    outer_rect = cv2.boundingRect(outer_contours[0])
    inner_rect = cv2.boundingRect(inner_contours[0])

    # Calculate the outer and inner radius
    outer_radius = int(max(outer_rect[2], outer_rect[3]) / 2)
    inner_radius = int(min(inner_rect[2], inner_rect[3]) / 2)
    mid_radius = (outer_radius + inner_radius) / 2

    # print(outer_radius, inner_radius, mid_radius)
    # mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # cv2.drawContours(mask, inner_contours, -1, 255, -1)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    outer_center = (outer_rect[0] + outer_radius, outer_rect[1] + outer_radius)
    inner_center = (inner_rect[0] + inner_radius, inner_rect[1] + inner_radius)

    # print("Outer Center:", outer_center)
    # print("Inner Center:", inner_center)

    return outer_center, mid_radius, outer_radius, inner_radius


def get_circular_corps(img, center: tuple = (1024, 1024), radius: int = 850, window_size: tuple = (250, 250),
                       step_angle=10, address=None):
    center_croped_images = []
    for angle in range(0, 360, step_angle):
        angle_rad = math.radians(angle)
        x_p = int(center[0] + radius * math.cos(angle_rad))
        y_p = int(center[1] + radius * math.sin(angle_rad))
        x = x_p - window_size[0] // 2
        y = y_p - window_size[1] // 2
        window = img[y:y + window_size[1], x:x + window_size[0]]
        if address:
            im = Image.fromarray(window, mode='L')
            im.save(address + '{}.png'.format(angle))
        center_croped_images.append(window)
    return np.array(center_croped_images)


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


def center_crop_to_square(img):
    width, height = img.size
    size = min(width, height)
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return img.crop((left, top, right, bottom))


def get_slid_window(image, win_size=(64, 64), step_size=32):
    result_windows = []

    # Iterate over image with sliding window
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # Crop out window
            window = image[y:y + win_size[1], x:x + win_size[0]]
            result_windows.append(window)
    return result_windows


def get_no_defect_crops(image, win_size=(64, 64), step_size=32, thresh_old=1, count=30):
    # Generate windows using view_as_windows
    windows = view_as_windows(image, win_size, step=step_size)
    # Flatten windows and count number of pixels below threshold
    counts = np.sum(windows <= thresh_old, axis=(2, 3))
    # Find indices where number of pixels below threshold is less than count
    y, x = np.where(counts < count)
    # Extract windows with desired indices
    result_windows = windows[y, x]
    # Reshape windows to 2D and return
    return result_windows.reshape(len(result_windows), *win_size)


def get_orth_actions(no_of_actions, action_range=(-1, 1)):
    var_ranges = [action_range] * no_of_actions

    # generate all the possible combinations
    combinations = list(itertools.product(*[range(r[0], r[1] + 1) for r in var_ranges]))

    # filter out the non-orthogonal combinations
    orthogonal_combinations = []
    for c1 in combinations:
        is_orthogonal = True
        for c2 in orthogonal_combinations:
            if sum([c1[i] * c2[i] for i in range(len(c1))]) != 0:
                is_orthogonal = False
                break
        if is_orthogonal:
            orthogonal_combinations.append(c1)

    # print the orthogonal combinations
    return orthogonal_combinations


def reshape_image(image, img_size=(256, 256)):
    if image.size == img_size:
        image = np.asarray(image, dtype=np.float32) / 255
    else:
        image = np.asarray(image.resize(img_size), dtype=np.float32) / 255
    return image


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


def is_loss_stagnated(loss_list, window_size=100, threshold=1e-4):
    """
    Check if the loss is stagnant or increasing by taking the last `window_size` entries of the `loss_list`.
    Returns True if the standard deviation of the last `window_size` entries is below `threshold`,
    indicating that the loss is stagnant, or if the current loss value is greater than or equal to the minimum
    loss value observed over the last `window_size` iterations, indicating that the loss is increasing.
    """

    if len(loss_list) < window_size:
        return False
    last_losses = loss_list[-window_size:]
    std_dev = np.std(last_losses)
    # min_loss = min(last_losses)
    if std_dev < threshold:  # or loss_list[-1] >= min_loss
        return True
    else:
        return False
