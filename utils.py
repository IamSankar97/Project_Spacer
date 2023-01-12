import numpy as np
import pandas as pd
import math


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
    theta1 = np.arccos(m) + np.radians(theta0)
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
