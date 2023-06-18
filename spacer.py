import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from utils import pol2cart, get_theta
import multiprocessing as mp


class Spacer:
    def __init__(self, shared_matrix, grid_spacing: float, outer_radius=16 * 1e-3, thickness=3.222 * 1e-3, produce_defect=True):
        '''
        surface: Point_coordinates in a numpy array
        grid_spacing:
        outer_radius: outer radius of the spacer # originally 16
        thickness: thickness of spacer # originally 3.222
        '''

        self.h_defect = None
        self.grid_spacing = grid_spacing
        self.grid_radius = outer_radius + self.grid_spacing
        self.grid_dim = int(self.grid_radius / self.grid_spacing) * 2
        self.surface = np.frombuffer(shared_matrix.get_obj(), dtype=np.float32, count=self.grid_dim**2)
        self.surface = self.surface.reshape((self.grid_dim, self.grid_dim))
        # self.surface = np.zeros([self.grid_dim, self.grid_dim])
        self.point_coo = None
        self.vertices = None
        self.spacer_coo = pd.DataFrame()
        self.outer_r = outer_radius + self.grid_spacing
        self.inner_r = self.outer_r - thickness - self.grid_spacing
        # self.produce_defect = produce_defect
        self.get_spacer()

    def get_spacer(self):
        end_l, end_r = int(-self.surface.shape[0] / 2), int(self.surface.shape[0] / 2)
        X = np.linspace(end_l * self.grid_spacing, end_r * self.grid_spacing, num=self.surface.shape[0])
        Y = X
        self.Xmesh, self.Ymesh = np.meshgrid(X, Y)

        x_center, y_center = 0, 0

        # Calculate the distance of each point in the mesh grid from the center of the circle
        distance = np.sqrt((self.Xmesh - x_center) ** 2 + (self.Ymesh - y_center) ** 2)

        # Create a boolean mask for the circular annulus
        mask = (distance > self.outer_r) | (distance < self.inner_r)

        # Update the values of surface outside the circular annulus
        # self.surface[mask] = 1
        self.spacer_mask = mask

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
        point_coords = np.column_stack((self.Xmesh.ravel(), self.Ymesh.ravel(), self.surface.ravel()))

        self.point_coo = point_coords[np.lexsort((point_coords[:, 1], point_coords[:, 0]))]

    def generate_defect(self, theta0, alpha=40, beta=70):

        r0 = np.round(np.random.uniform(self.inner_r, self.outer_r), 2)
        r1 = np.round(np.random.uniform(self.inner_r, self.outer_r), 2)
        scratch_length = np.round(np.random.uniform(0.1, 10), 2)

        # Convert to meter
        r0, r1, scratch_length = r0, r1, scratch_length * 1e-3
        if scratch_length < abs(r0 - r1):
            r1 = r0 + scratch_length
            warnings.warn(
                "defect length is smaller than asked radius boundary, r1 is adjusted to meet the scratch_length")

        #   Defect grove height and depth from mean surface
        h_up, h_total = self.grid_spacing / np.tan(np.radians(alpha)), \
                        self.grid_spacing / np.tan(np.radians(beta))
        self.h_defect = h_total - h_up

        #   Calculating start and end coordinate of defects in terms of grid points
        x0, y0 = pol2cart(r0, np.radians(theta0))
        x1, y1 = pol2cart(r1, get_theta(r0, r1, theta0, scratch_length))

        # convert coordinates in meters to grid points
        def_co = np.array([x0, y0, x1, y1]) / self.grid_spacing
        def_co += self.surface.shape[0] * 0.5
        def_co = def_co.astype(int)
        x0, y0, x1, y1 = def_co

        # Calculate the distances and angles between start and end points
        dx, dy = x1 - x0, y1 - y0
        distance = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)
        sin_angle, cos_angle = np.sin(angle), np.cos(angle)

        # Calculate the number of grid points along the line
        num_points = int(distance / self.grid_spacing)
        x_points, y_points = np.full(num_points, x0), np.full(num_points, y0)

        indices = np.arange(num_points)
        x_coords = x_points + (indices * self.grid_spacing * cos_angle)
        y_coords = y_points + (indices * self.grid_spacing * sin_angle)

        x_coords, y_coords = np.array(np.round(x_coords).astype(int)), np.array(np.round(y_coords).astype(int))
        mask = (x_coords < self.surface.shape[0]) & (y_coords < self.surface.shape[1])

        # create a copy of the surface array and update only the values that need to be updated
        # surface = np.zeros_like(self.surface)
        self.surface[x_coords[mask], y_coords[mask]] = self.h_defect

    def update_matrix(self, value):
        index = random.randint(0, self.surface.size - 1)
        row = index // self.grid_dim
        col = index % self.grid_dim
        self.surface[row][col] = value

    def generate_defects_parallel(self, thetas, num_processes):
        thetas_split = np.array_split(thetas, num_processes)
        pool = mp.Pool(num_processes)
        results = pool.map(self._generate_defects_worker, thetas_split)
        pool.close()
        pool.join()
        for result in results:
            print(np.min(self.surface))
            plt.imshow(result, cmap='gray')
            plt.plot()
            self.surface += result

    def _generate_defects_worker(self, thetas):
        result = np.zeros_like(self.surface)
        for theta in thetas:
            defect = self.generate_defect(theta)
            result = defect
        return result