import numpy as np
import pandas as pd
import warnings
from utils import pol2cart, closest_number_, get_theta


class Spacer:
    def __init__(self, surface: np.ndarray, grid_spacing: float, outer_radius=16, thickness=3.222):
        self.h_defect = None
        self.surface = surface
        self.grid_spacing = grid_spacing
        self.point_coo = pd.DataFrame()
        self.vertices = None
        self.spacer_coo = pd.DataFrame()
        self.outer_r = outer_radius * 1e-3
        self.inner_r = self.outer_r - (thickness * 1e-3)

    def get_point_co(self):
        """
        Notes:
            Convert a 2D mesh to a list of point coordinates[X,Y,Z]
        Returns
        -------
            Point Coordinates [X, Y, Z]
        """

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
        df_point_coords['X_grid'] = df_point_coords['X'].div(self.grid_spacing).round().astype(int)
        df_point_coords['Y_grid'] = df_point_coords['Y'].div(self.grid_spacing).round().astype(int)

        self.point_coo = df_point_coords

    def get_spacer_point_co(self):
        """
        Notes:
            Convert a 2D mesh to a list of point coordinates[X,Y,Z] while replacing Z to 1 where it doesnt fall within
            Outer dia and inner dia of Spacer. This Z information is latter used while creating polymesh to avoid
            creating meshes outside the boundary
        Returns
        -------
            Point Coordinates [X, Y, Z]
        """

        end_l, end_r = int(-self.surface.shape[0] / 2), int(self.surface.shape[0] / 2)

        X = np.array([i * self.grid_spacing for i in range(end_l, end_r)])
        Y = X
        Xmesh, Ymesh = np.meshgrid(X, Y)

        def update_z(x, y, h):
            """
            Parameters
            ----------
            x : Position of grid at x
            y : Position of grid at y
            h : Height at that grid point

            Returns
            -------
                if grid_point falls between spacer Outer_radius and Inner_radius retains height else updates height to 1
                This 1 will later be used in blender script to avoid these vertices getting created
            """
            r = np.sqrt(x ** 2 + y ** 2)
            if self.outer_r > r > self.inner_r:
                return x, y, h
            else:
                return x, y, 1

        point_coords = np.array(
            [update_z(x, y, z) for x_, y_, z_ in zip(Xmesh, Ymesh, self.surface)
             for x, y, z in zip(x_, y_, z_)])

        df_point_coords = pd.DataFrame(point_coords, dtype=np.float64)  # Saves 10 mili seconds
        df_point_coords.rename(columns={0: 'X', 1: 'Y', 2: 'Z'}, inplace=True)
        df_point_coords.sort_values(by=['X', 'Y'], inplace=True)
        df_point_coords['X_grid'] = df_point_coords['X'].div(self.grid_spacing).round().astype(int)
        df_point_coords['Y_grid'] = df_point_coords['Y'].div(self.grid_spacing).round().astype(int)
        self.point_coo = df_point_coords

    def update_defect_height(self, row):
        """

        Parameters
        ----------
        row: point coordinate where defect has to be created

        Returns
        -------
        point coordinate by updating itz z value t0 defect
        """
        global independent, dependent_actual
        # df = self.point_coo
        # df['X_grid'] = df['X'].div(self.grid_spacing).round().astype(int)
        # df['Y_grid'] = df['Y'].div(self.grid_spacing).round().astype(int)

        index_defect = []
        for i in range(row['no_of_grids']):
            independent = row[row['discretize'] + '0'] + (i * self.grid_spacing)
            if row['discretize'] == 'X':
                # Calculating Y knowing Independent/parent grid X
                dependent = (abs(independent - row[row['discretize'] + '0']) * row['slope']) + row[
                    row['find_coo'] + '0']
            else:
                # Calculating X knowing Independent/parent grid Y
                dependent = (abs(independent - row[row['discretize'] + '0']) / row['slope']) + row[
                    row['find_coo'] + '0']
            independent_grid = np.round_(independent / self.grid_spacing)
            df_ = self.point_coo[self.point_coo.loc[:, row['discretize'] + '_grid'] == independent_grid]
            try:
                dependent_actual, index = closest_number_(df_, dependent, row['find_coo'])
                if self.point_coo.at[index, "Z"] != 1:
                    self.point_coo.at[index, "Z"] -= self.h_defect
                    index_defect.append(index)
            except:
                pass
        # 1st return X 2nd Return Y
        if row['discretize'] == 'X':
            return independent, dependent_actual
        else:
            return dependent_actual, independent

    def generate_defect(self, r0, r1, theta0_set: np.ndarray = np.array(np.NaN), origin_cord: np.ndarray = np.array(np.NaN),
                        alpha=0.0, beta=0.0, scratch_length: float = 0.0, return_def_end_co: bool = False):
        """
        Note:
            It takes starting point of defect (r0, theta0), end point of defect r1 (calculates theta2) and defect length to prescribe
            the defect location

        :param r0: starting radius of defect in mm
        :param r1: end radius of defect in mm
        :param theta0_set: starting angles of defect in degree
        :param alpha: defect geometry
        :param beta: defect geometry
        :param origin_cord: origin of the defect in cartesian form (Not to be mentioned if origin is known in polar coo)
        :param return_def_end_co: Returns the end defect coordinate in cartesian form
        :param scratch_length: length of the scratch in mm
        :return: Generates defect in the spacer

        """
        if self.point_coo.empty:
            self.get_spacer_point_co()

        # Convert to meter
        r0, r1, scratch_length = r0 * 1e-3, r1 * 1e-3, scratch_length * 1e-3
        if scratch_length < abs(r0 - r1):
            scratch_length = abs(r0 - r1)
            warnings.warn("defect length is smaller than asked radius boundary, considered distance = r2-r1")

        #   Defect grove height and depth from mean surface
        h_up, h_total = self.grid_spacing / np.tan(np.radians(alpha)), \
                        self.grid_spacing / np.tan(np.radians(beta))
        self.h_defect = h_total - h_up

        #   Calculating start and end coordinate of defect
        if not np.isnan(theta0_set).any():
            defect_geometry = pd.DataFrame([[pol2cart(r0, np.radians(theta0)),
                                             pol2cart(r1, get_theta(r0, r1, theta0, scratch_length)),
                                             get_theta(r0, r1, theta0, scratch_length)]
                                            for theta0 in theta0_set])
        else:
            defect_geometry = \
                pd.DataFrame([[(xo, yo),
                               pol2cart(r1, get_theta(np.sqrt(xo**2 + yo**2), r1, np.arctan2(yo, xo), scratch_length)),
                               get_theta(np.sqrt(xo**2 + yo**2), r1, np.arctan2(yo, xo), scratch_length)]
                              for xo, yo in origin_cord])

        def get_slope(row):
            rise, run = abs(row['Y1'] - row['Y0']), abs(row['X1'] - row['X0'])
            if rise > scratch_length:
                run = rise
            slope = rise / run
            if rise > run:
                no_of_grids = int(np.round_(rise / self.grid_spacing) + 1)
                discrete, find_coo = 'Y', 'X'
            else:
                no_of_grids = int(np.round_(run / self.grid_spacing) + 1)
                discrete, find_coo = 'X', 'Y'

            return slope, rise, run, no_of_grids, discrete, find_coo

        defect_geometry[['X0', 'Y0']] = pd.DataFrame(defect_geometry[0].tolist(), index=defect_geometry.index)
        defect_geometry[['X1', 'Y1']] = pd.DataFrame(defect_geometry[1].tolist(), index=defect_geometry.index)
        defect_geometry.rename(columns={2: 'theta1'}, inplace=True)
        defect_geometry.drop(defect_geometry.columns[0:2], axis=1, inplace=True)
        defect_geometry['result'] = defect_geometry[['X0', 'Y0', 'X1', 'Y1']].apply(get_slope, axis=1)
        defect_geometry[['slope', 'rise', 'run', 'no_of_grids', 'discretize', 'find_coo']] \
            = pd.DataFrame(defect_geometry['result'].tolist(), index=defect_geometry.index)

        defect_geometry.drop(['result'], axis=1, inplace=True)

        df = self.point_coo.copy()
        df['X_grid'] = df['X'].div(self.grid_spacing).round().astype(int)
        df['Y_grid'] = df['Y'].div(self.grid_spacing).round().astype(int)

        defect_end_co = []
        for index, row in defect_geometry.iterrows():
            defect_end_co.append(self.update_defect_height(row))

        if return_def_end_co:
            return np.array(defect_end_co)

    def randomize_defect(self, r0, r1, theta0, alpha=0.0,
                         beta=0.0, scratch_length: float = 0.0, defect_type: int = 0):
        """
        0: cluster straight_line defect
        1. cluster not straight line defect
        2. cluster straight_line and not straight line
        Returns
        -------
        Topology with defect
        """
        number_of_defects = np.random.choice(np.arange(3, 10, 2), size=np.random.randint(1, 3), replace=False)
        theta_set = number_of_defects + theta0
        if defect_type == 0:
            self.generate_defect(r0=r0, r1=r1, theta0_set=theta_set, alpha=alpha,
                                 beta=beta, scratch_length=scratch_length)
        elif defect_type == 1:
            split_region = np.random.uniform(0.1, 0.7)
            r2 = r1
            r1, scratch_length1 = r0 + ((r2 - r0) * split_region), scratch_length * split_region
            scratch_length2 = scratch_length

            origin_co_set = self.generate_defect(r0=r0, r1=r1, theta0_set=theta_set,
                                                 alpha=alpha, beta=beta, scratch_length=scratch_length1,
                                                 return_def_end_co=True)

            self.generate_defect(r0=r1, r1=r2, alpha=alpha, beta=beta, scratch_length=scratch_length2,
                                 origin_cord=origin_co_set)