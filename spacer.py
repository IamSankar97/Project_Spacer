from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from utils import cart2pol, pol2cart, closest_number_, get_theta


class Spacer:
    def __init__(self, surface: np.ndarray, grid_spacing: float, outer_radius=16, thickness=3.222):
        self.surface = surface
        self.grid_spacing = grid_spacing
        self.point_coo = pd.DataFrame()
        self.vertices = None
        self.spacer_coo = pd.DataFrame()
        self.outer_r = outer_radius * 1e-3
        self.inner_r = self.outer_r - (thickness * 1e-3)

    def get_point_co(self):

        end_l, end_r = int(-self.surface.shape[0] / 2), int(self.surface.shape[0] / 2)

        X = np.array([i * self.grid_spacing for i in range(end_l, end_r)])
        Y = X
        Xmesh, Ymesh = np.meshgrid(X, Y)

        point_coords = np.array(
            [np.array([x, y, z]) for x_, y_, z_ in zip(Xmesh, Ymesh, self.surface)
             for x, y, z in zip(x_, y_, z_)])

        df_point_coords = pd.DataFrame(point_coords, dtype=np.float64)
        df_point_coords.rename(columns={0: 'X', 1: 'Y', 2: 'Z'}, inplace=True)
        df_point_coords.sort_values(by=['X', 'Y'], inplace=True)
        self.point_coo = df_point_coords

    def get_spacer_point_co(self):
        """

        Parameters
        ----------
        r0: Spacer Inner radius in mm
        r1: Spacer Outer radius in mm

        Returns
        -------
        results vertices, by modifying point coordinates such that
        """

        # r0, r1 = r0 * 1e-3, r1 * 1e-3

        end_l, end_r = int(-self.surface.shape[0] / 2), int(self.surface.shape[0] / 2)

        X = np.array([i * self.grid_spacing for i in range(end_l, end_r)])
        Y = X
        Xmesh, Ymesh = np.meshgrid(X, Y)

        x = 5
        y = 10
        max_value = x if x > y else y
        print(max_value)  # prints 10

        def update_z(a, b, c):
            r = np.sqrt(a ** 2 + b ** 2)
            if self.outer_r > r > self.inner_r:
                return a,b,c
            else:
                return a,b,1

        point_coords = np.array(
            [update_z(x, y, z) for x_, y_, z_ in zip(Xmesh, Ymesh, self.surface)
             for x, y, z in zip(x_, y_, z_)])

        df_point_coords = pd.DataFrame(point_coords, dtype=np.float64)  # Saves 10 mili seconds
        df_point_coords.rename(columns={0: 'X', 1: 'Y', 2: 'Z'}, inplace=True)
        df_point_coords.sort_values(by=['X', 'Y'], inplace=True)
        self.point_coo = df_point_coords

    def generate_defect(self, r0, r1, theta0, alpha=0.0,
                        beta=0.0, scratch_length: float = 0.0, return_theta: bool = False):
        """
        Note:
            It takes starting point of defect (r0, theta0), end point of defect r1 (calculates theta2) and defect length to prescribe
            the defect location

        :param r0: starting radius of defect in mm
        :param r1: end radius of defect in mm
        :param theta0: starting angle of defect in degree
        :param alpha:
        :param beta:
        :param scratch_length: length of the scratch in mm
        :return: Generates defect in the spacer
        """
        if self.point_coo.empty:
            self.get_point_co()
        # Convert to meter
        r0, r1, scratch_length = r0 * 1e-3, r1 * 1e-3, scratch_length * 1e-3
        if scratch_length < abs(r0 - r1):
            scratch_length = abs(r0 - r1)
            warnings.warn("defect length is smaller than asked radius boundary, considered distance = r2-r1")

        h_up, h_total = self.grid_spacing / np.tan(np.radians(alpha)), self.grid_spacing / np.tan(np.radians(beta))
        h_defect = h_total - h_up

        #   Calculating start and end coordinate of defect
        theta1 = get_theta(r0, r1, theta0, scratch_length)
        x0, y0 = pol2cart(r0, np.radians(theta0))
        x1, y1 = pol2cart(r1, theta1)

        rise = abs(y1 - y0)
        run = abs(x1 - x0)

        if rise > scratch_length:
            run = rise
        slope_scratch = rise / run

        # This check is to choose whether to discritize along X direction or Y direction
        if rise > run:

            df = self.point_coo.copy()
            df['Y_grid'] = df['Y'].div(self.grid_spacing)
            df['Y_grid'] = df['Y_grid'].round()
            df['Y_grid'] = df['Y_grid'].astype(int)
            no_of_grids_y = int(np.round_(rise / self.grid_spacing))
            for i in range(no_of_grids_y):
                y_sc_co = y0 + (i * self.grid_spacing)  # Scratch coordinate y
                x_sc_co = ((y_sc_co - y0) / slope_scratch) + x0  # Scratch coordinate x
                y_sc_co_grid = np.round_(y_sc_co / self.grid_spacing)  # Scratch coordinate y's grid point
                df_ = df[df.loc[:, "Y_grid"] == y_sc_co_grid]  # Filtering out x points where ygrid id y's grid point
                try:
                    x_sc_co_actual, index = closest_number_(df_, x_sc_co,
                                                            "X")  # Filtering out x points where ygrid id y's grid point
                    self.point_coo["Z"][index] = -h_defect
                except:
                    pass

        else:
            df = self.point_coo.copy()
            df['X_grid'] = df['X'].div(self.grid_spacing)
            df['X_grid'] = df['X_grid'].round()
            df['X_grid'] = df['X_grid'].astype(int)
            no_of_grids_x = int(np.round_(run / self.grid_spacing))
            for i in range(no_of_grids_x):
                x_sc_co = x0 + (i * self.grid_spacing)  # Scratch coordinate x
                y_sc_co = (abs(x_sc_co - x0) * slope_scratch) + y0  # Scratch coordinate y
                x_sc_co_grid = np.round_(x_sc_co / self.grid_spacing)  # Scratch coordinate y's grid point
                df_ = df[df.loc[:, "X_grid"] == x_sc_co_grid]  # Filtering out x points where ygrid id y's grid point
                try:
                    x_sc_co_actual, index = closest_number_(df_, y_sc_co,
                                                            "Y")  # Filtering out x points where ygrid id y's grid point
                    self.point_coo["Z"][index] = -h_defect
                except:
                    pass
        if return_theta:
            return np.degrees(theta1)

    def randomize_defect(self, r0, r1, theta0, alpha=0.0,
                         beta=0.0, scratch_length: float = 0.0, defect_type: int = 0):
        """
        0: cluster linear defect
        1. cluster cross_defect
        2. cluster linear and cross_defect
        Returns
        -------
        Topology with defect
        """
        if defect_type == 0:
            number_of_defects = np.random.randint(1, 4)
            for i in range(number_of_defects):
                if i == 0:
                    self.generate_defect(r0, r1, theta0, alpha, beta, scratch_length)
                else:
                    angle_difference = np.random.randint(4, 10)
                    self.generate_defect(r0, r1, theta0 + angle_difference, alpha, beta, scratch_length)

        elif defect_type == 1:
            number_of_defects = np.random.randint(1, 4)
            split_region = np.random.randint(3, 5)
            r2 = r1
            r1, scratch_length1 = r0 + ((r2 - r0) * (split_region * 0.1)), scratch_length * split_region * 0.1
            scratch_length2 = scratch_length
            for i in range(number_of_defects):
                if i == 0:
                    theta1 = self.generate_defect(r0, r1, theta0, alpha, beta,
                                                  scratch_length1, return_theta=True)
                else:
                    angle_difference = np.random.randint(4, 10)
                    theta1 = self.generate_defect(r0, r1, theta0 + angle_difference, alpha, beta,
                                                  scratch_length1, return_theta=True)
                self.generate_defect(r1, r2, theta1, alpha, beta,
                                     scratch_length2)
