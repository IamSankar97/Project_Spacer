from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from utils import cart2pol, pol2cart, closest_number_, get_theta


class Spacer:
    def __init__(self, surface: np.ndarray, grid_spacing: float):
        self.surface = surface
        self.grid_spacing = grid_spacing
        self.point_coo = pd.DataFrame()
        self.vertices = None

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

        self.point_coo = df_point_coords

    def generate_defect(self, r0, r1, theta0, alpha=0.0,
                        beta=0.0, scratch_length: float = 0.0):
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
        if rise > run:

            df = self.point_coo.copy()
            df['Y_grid'] = df['Y'].div(self.grid_spacing)
            df['Y_grid'] = df['Y_grid'].round()
            df['Y_grid'] = df['Y_grid'].astype(int)
            no_of_grids_y = int(np.round_(rise / self.grid_spacing))
            for i in range(no_of_grids_y):
                y_sc_co = y0 + (i * self.grid_spacing)  # Scratch coordinate y
                x_sc_co = (y_sc_co - y0) / slope_scratch + x0  # Scratch coordinate x
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

    def get_vertices(self):
        if self.point_coo.empty:
            self.get_point_co()
        #   Read and sort the vertices coordinates (sort by x and y)
        self.vertices = sorted([(float(r[0]), float(r[1]), float(r[2])) for r in self.point_coo], key=itemgetter(0, 1))