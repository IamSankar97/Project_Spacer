# # Use below only during debugging
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

# *******************The code name is 6 actions but 5
# action is used with the black musgrave texture with white back ground****

import multiprocessing
import sys
import os
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
import warnings

sys.path.append(os.getcwd())
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs')
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/')

import bpy
import time
import bmesh
import multiprocessing as mp
import cv2
from PIL import Image
import colorsys
import numpy as np
import pickle
import pandas as pd
import random
from blendtorch import btb
from spacer import Spacer
import datetime
from utils import augument, pol2cart, get_theta, get_no_defect_crops

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# else:
# angle = np.arctan2(spacer_class.Ymesh, spacer_class.Xmesh)
# angle = (angle + 2 * np.pi) % (2 * np.pi)
# selected_coordinates = np.logical_and.reduce([np.abs(spacer_class.distance - r0) <= spacer_class.grid_spacing * 2,
#                                               (angle >= theta0) & (angle <= theta1)])
# spacer_class.surface[selected_coordinates] = h_defect
# mask = np.logical_and(spacer_class.surface != 0, spacer_class.surface != 1)
# noise = np.random.uniform(low=-1e-7, high=1e-6, size=spacer_class.surface.shape)
# spacer_class.surface = np.where(mask, spacer_class.surface + noise, spacer_class.surface)

def generate_defect(surface, grid_spacing, Xmesh, Ymesh, r0, r1, scratch_length, theta0, pair=1, space=0, alpha=40):
    '''

    Parameters
    ----------
    surface
    grid_spacing
    r0
    r1
    scratch_length
    theta0
    alpha
    width: width of scratch interms of coordinates. if width = 2, defect width = 2*grid spacing

    Returns
    -------

    '''
    beta = random.randint(40, 80)
    # beta = 70
    # Convert to meter

    r0, r1, scratch_length = r0 * 1e-3, r1 * 1e-3, scratch_length * 1e-3
    if scratch_length < abs(r0 - r1):
        r1 = r0 + scratch_length
        warnings.warn(
            "defect length is smaller than asked radius boundary, r1 is adjusted to meet the scratch_length")

    #   Defect grove height and depth from mean surface
    h_up, h_total = grid_spacing / np.tan(np.radians(alpha)), \
                    grid_spacing / np.tan(np.radians(beta))

    h_defect = h_total - h_up
    if h_defect >= 0:
        h_defect = -0.00001

    #   Calculating start and end coordinate of defects in terms of grid points
    theta1 = get_theta(r0, r1, theta0, scratch_length)
    x0, y0 = pol2cart(r0, np.radians(theta0))
    x1, y1 = pol2cart(r1, theta1)

    # convert coordinates in meters to grid points
    def_co = np.array([x0, y0, x1, y1]) / grid_spacing
    def_co += surface.shape[0] * 0.5
    def_co = def_co.astype(int)
    x0, y0, x1, y1 = def_co

    # Calculate the distances and angles between start and end points
    dx, dy = x1 - x0, y1 - y0
    distance = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    sin_angle, cos_angle = np.sin(angle), np.cos(angle)

    # Calculate the number of grid points along the line
    num_points = int(distance / grid_spacing)
    x_points, y_points = np.full(num_points, x0), np.full(num_points, y0)

    indices = np.arange(num_points)
    x_coords = x_points + (indices * grid_spacing * cos_angle)
    y_coords = y_points + (indices * grid_spacing * sin_angle)

    x_coords, y_coords = np.array(np.round(x_coords).astype(int)), np.array(np.round(y_coords).astype(int))

    x_coords_left, y_coords_left = np.add(x_coords, 1), y_coords
    x_coords_right, y_coords_right = x_coords, np.add(y_coords, 1)
    if pair != 1 and space != 0:
        # x_co, y_co = x_coords, y_coords
        offset = np.arange(pair)
        offset_space = np.linspace(0, space, len(offset))
        offset = offset + offset_space
        x_coords = np.repeat(x_coords, pair) + np.tile(offset, len(x_coords))
        y_coords = np.repeat(y_coords, pair) #+ np.tile(offset, len(y_coords))
        x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)

    mask = (x_coords < surface.shape[0]) & (y_coords < surface.shape[1])
    # mask_left = (x_coords_left < surface.shape[0]) & (y_coords_left < surface.shape[1])
    # mask_right = (x_coords_right < surface.shape[0]) & (y_coords_right < surface.shape[1])
    noise = np.random.uniform(low=-1e-7, high=1e-6, size=x_coords[mask].shape)

    # Add the noise to the height of the bump
    h_bump = h_defect + noise
    # hup_l = h_up + noise
    # hup_r = h_up - noise
    surface[x_coords[mask], y_coords[mask]] = h_bump
    # surface[x_coords[mask_left], y_coords[mask_left]] = hup_l
    # surface[x_coords[mask_right], y_coords[mask_right]] = hup_r


class SpacerEnv(btb.env.BaseEnv):
    def __init__(self, agent):
        super().__init__(agent)
        self.def_spacer = None
        self.spacer = None
        self.action_inverted = None
        self.action_pair = None
        self.grid_spacing = 0.00001  # 1e-6
        self.camera = bpy.data.objects["Camera"]
        self.camera.location = (0.0, 0.0, 0.15)
        self.light0 = bpy.data.objects["L0_top"]
        self.light0.location = (0.0, 0.0, 0.15)
        self.episodes, self.step, self.total_step = 4, 0, 0
        self.img_addr = 'spacer_gym/spacer_env_render/image_spacer.png'
        # Note, ensure that physics run at same speed.
        self.fps = bpy.context.scene.render.fps
        # self.texture_nodes = bpy.data.materials.get("spacer").node_tree.nodes
        self.avg_pixel = 30
        self.np_random = None
        self.state = 40
        self.vertices = None
        self.topology_dir = '/home/mohanty/PycharmProjects/Data/pkl_6/'
        self.g_time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.topologies = os.listdir(self.topology_dir)
        self.x_128_64 = [12, 13, 14, 15, 16, 17, 18, 9, 10, 20, 21, 3, 27, 3, 27, 2, 28, 2, 28, 2, 28, 2, 28, 2, 28, 2,
                         28, 2, 28, 3, 27, 3, 27, 9, 10, 20, 21, 12, 13, 14, 15, 16, 17, 18]

        self.y_128_64 = [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 9, 9, 10, 10, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
                         18, 18, 20, 20, 21, 21, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28]
        #   (1)
        self.action_Material = {'roughness': [0, 0.5]}
        #   (1)
        self.action_mix = {'Factor': [0.0, 0.3]}  # {'Factor': [0.0, 0.4]} old noise material3

        # self.action_light_cmn = {"value": [0.8, 1]}
        #   (3)
        self.action_light = {'energy0': [0.02, 0.18], 'ro_x': [-0.42, 0.42], 'ro_y': [-0.42, 0.42]}
        # Total = 5

        self.reset_action = {'specular': 0.2, 'ior': 2.3, 'roughness': 0.1, 'factor': 0.05, "value": 0.8,
                             'energy0': 0.01, 'ro_x': 0, 'ro_y': 0}
        self.action_bound = {**self.action_Material, **self.action_mix, **self.action_light}
        self.action_keys = list(self.action_bound.keys())
        self.outer_radius = 16 * 1e-3
        self.thickness = 3.222 * 1e-3
        self.get_set = 0
        self.grid_radius = self.outer_radius + self.grid_spacing
        self.grid_dim = int(self.grid_radius / self.grid_spacing) * 2
        self.material = bpy.data.materials.get("spacer")
        self.def_material = bpy.data.materials.get("defect")
        self.generate_spacer_assign_mat()
        self.texture_nodes = bpy.data.materials.get("spacer").node_tree.nodes
        self.df_stats = pd.DataFrame(columns=['defect_width', 'r0s', 'r1s', 'sls', 'thetas', 'pairs', 'spaces'])
        self.df_stat = []

        # self.update_scene()

    def update_scene(self):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def update_vertices(self, chunk):
        for vertice_old, vertice_new in chunk:
            if vertice_old.co.z != 1:
                vertice_old.co.z = vertice_new[2]

    # def generate_polygon(self, spacer_mesh_name: str = 'spacer_ring', smoothing: bool = 0):
    #     """
    #     Parameters
    #     ----------
    #     spacer_mesh_name
    #     def_mesh_name
    #     smoothing
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     #   ********* Assuming we have a rectangular grid *************
    #     x_diff = np.diff(self.vertices[:, 0])
    #     x_change_idx = np.nonzero(x_diff)[0][0] + 1
    #     self.xSize = x_change_idx
    #     # self.xSize = next(i for i in range(len(self.vertices))
    #     #                   if self.vertices[i][0] != self.vertices[i + 1][0]) + 1  # Find the first change in X
    #     self.Size = len(self.vertices) // self.xSize
    #
    #     i = np.arange(1, len(self.vertices) - self.xSize)
    #     poly_set = np.array([
    #         self.vertices[i, 2],
    #         self.vertices[i - 1, 2],
    #         self.vertices[i - 1 + self.xSize, 2],
    #         self.vertices[i + self.xSize, 2]
    #     ])
    #
    #     mask = (i % self.xSize != 0) & (self.vertices[i, 2] != 1) & np.all(poly_set != 1, axis=0)
    #     xSize_arr = np.full(len(i), self.xSize)  # Create an array of the same length as i with the value of self.xSize
    #     # polygons = np.column_stack((i[mask], i - 1[mask], i - 1 + xSize_arr[mask], i + xSize_arr[mask]))
    #     polygons = np.column_stack((
    #         i[mask],
    #         i[mask] - 1,
    #         i[mask] - 1 + self.xSize,
    #         i[mask] + self.xSize
    #     )).tolist()
    #
    #     # #   Generate the polygons (four vertices linked in a face)
    #     # polygons = []
    #     # for i in range(1, len(self.vertices) - self.xSize):
    #     #     poly_set = np.array([self.vertices[i][2], self.vertices[i - 1][2], self.vertices[i - 1 + self.xSize][2],
    #     #                          self.vertices[i + self.xSize][2]])
    #         # collecting polygons
    #         # if i % self.xSize != 0 and self.vertices[i][2] != 1 and np.all(poly_set != 1):
    #         #     polygons.append((i, i - 1, i - 1 + self.xSize, i + self.xSize))
    #
    #     mesh = bpy.data.meshes.new(spacer_mesh_name)  # Create the mesh (inner data)
    #     obj = bpy.data.objects.new(spacer_mesh_name, mesh)  # Create an object
    #     obj.data.from_pydata(self.vertices, [], polygons)  # Associate vertices and polygons
    #
    #     if smoothing:
    #         for p in obj.data.polygons:  # Set smooth shading (if needed)
    #             p.use_smooth = True
    #
    #     bpy.context.scene.collection.objects.link(obj)  # Link the object to the scene
    #     self.spacer = bpy.data.objects[spacer_mesh_name]
    #     self.spacer.location = (0.0, 0.0, 0.0)
    #
    #
    #
    #     # Assign the material to the object
    #     self.spacer.active_material = self.material
    #     bpy.context.view_layer.objects.active = self.spacer
    #     self.spacer.select_set(True)

    def get_defect_mask(self, def_mesh_name: str = 'spacer_defect', spacer_mesh_name: str = 'spacer_ring'):

        x_diff = np.diff(self.vertices[:, 0])
        x_change_idx = np.nonzero(x_diff)[0][0] + 1
        self.xSize = x_change_idx
        # self.xSize = next(i for i in range(len(self.vertices))
        #                   if self.vertices[i][0] != self.vertices[i + 1][0]) + 1  # Find the first change in X
        self.Size = len(self.vertices) // self.xSize

        # create an array of indices for each polygon
        indices = np.arange(1, len(self.vertices) - self.xSize)
        # create an array of vertices for each polygon
        vertices = np.array([indices, indices - 1, indices - 1 + self.xSize, indices + self.xSize]).T
        # create a boolean mask for the defect polygons
        mask_defect = (indices % self.xSize != 0) & (self.vertices[indices, 2] != 1) & (self.vertices[indices, 2] != 0)
        # create a boolean mask for the polygons with defects
        poly_set = self.vertices[vertices[:, :], 2]

        mask_defect = mask_defect & np.all(poly_set != 1, axis=1) & np.any((-1 <= poly_set)
                                                                           & (poly_set != 0), axis=1)

        mask_spacer = (indices % self.xSize != 0) & (self.vertices[indices, 2] != 1)
        mask_spacer = mask_spacer & np.all(poly_set != 1, axis=1)
        mask_spacer = mask_spacer & ~mask_defect

        # filter the indices using the mask
        polygons_def = vertices[mask_defect, :].tolist()
        polygons_spacer = vertices[mask_spacer, :].tolist()

        collection = bpy.context.scene.collection

        # Unlink the object from the collection
        if self.def_spacer:
            collection.objects.unlink(self.def_spacer)
            # Remove the object from the scene
            bpy.data.objects.remove(self.def_spacer, do_unlink=True)

        # Create defect mask object
        mesh = bpy.data.meshes.new(def_mesh_name)  # Create the mesh (inner data)
        obj = bpy.data.objects.new(def_mesh_name, mesh)  # Create an object
        obj.data.from_pydata(self.vertices, [], polygons_def)

        bpy.context.scene.collection.objects.link(obj)  # Link the object to the scene
        self.def_spacer = bpy.data.objects[def_mesh_name]
        self.def_spacer.location = (0, 0.0, 0.0)
        self.def_spacer.active_material = self.def_material

        # Unlink the object from the collection
        if self.spacer:
            collection.objects.unlink(self.spacer)
            # Remove the object from the scene
            bpy.data.objects.remove(self.spacer, do_unlink=True)

        # Create defect mask object
        mesh = bpy.data.meshes.new(spacer_mesh_name)  # Create the mesh (inner data)
        obj = bpy.data.objects.new(spacer_mesh_name, mesh)  # Create an object
        obj.data.from_pydata(self.vertices, [], polygons_spacer)

        bpy.context.scene.collection.objects.link(obj)  # Link the object to the scene
        self.spacer = bpy.data.objects[spacer_mesh_name]
        self.spacer.location = (0, 0.0, 0.0)
        self.spacer.active_material = self.material

    def update_mesh_back_ground(self):
        """
        :return: updates mesh in blender
        """
        z_coordinates = np.array(self.vertices).flatten()

        # Update the z-coordinates directly in the mesh data
        self.spacer.data.vertices.foreach_set("co", z_coordinates)

        # Update the mesh in Blender
        self.spacer.data.update()  # Updates the mesh in scene by refreshing the screen

        start_time = time.time()
        self.get_defect_mask()
        print('get_defect_mask', time.time() - start_time)

    def get_sample_surface(self, with_defect=True, return_statistics=False, paired_defect=False):
        shared_matrix = multiprocessing.Array('i', self.grid_dim * self.grid_dim)
        self.spacer_s = Spacer(shared_matrix, self.grid_spacing, self.outer_radius, self.thickness)
        if with_defect:
            # no of defects
            N = 5
            subinterval = int(360 / N)
            thetas = np.array([i * subinterval for i in range(N)])
            noise = np.random.uniform(-45, 45, N)
            thetas = thetas + noise
            # Clip the angles to ensure they are between 0 and 360 degrees
            thetas = np.clip(thetas, 0, 360)

            r0s = np.round(np.random.uniform(12.5, 16, N), 2)
            r1s = np.round(np.random.uniform(12.5, 16, N), 2)
            sls = np.round(np.random.uniform(0.1, 10, N), 2)
            print('Generate_defect')
            if paired_defect:
                pairs = np.random.randint(1, 3, size=N)
                spaces = np.random.choice([0, 8, 16, 20, 24], size=N)
                # generate_defect(surface=self.spacer_s.surface, grid_spacing=self.spacer_s.grid_spacing, r0=r0s[0],
                #                 r1=r1s[0], scratch_length=sls[0], theta0=thetas[0], width=widths[0], space=spaces[0])

                processes = [multiprocessing.Process(target=generate_defect,
                                                     args=(self.spacer_s.surface, self.spacer_s.grid_spacing,
                                                           self.spacer_s.Xmesh, self.spacer_s.Ymesh, r0, r1, sl, theta,
                                                           pair, space))
                             for r0, r1, sl, theta, pair, space in zip(r0s, r1s, sls, thetas, pairs, spaces)]
            else:
                pairs, spaces = 1, 0
                processes = [multiprocessing.Process(target=generate_defect,
                                                     args=(self.spacer_s.surface, self.spacer_s.grid_spacing,
                                                           self.spacer_s.Xmesh, self.spacer_s.Ymesh, r0, r1, sl,
                                                           theta))
                             for r0, r1, sl, theta in zip(r0s, r1s, sls, thetas)]

            for p in processes:
                p.start()

            # wait for all the processes to finish
            for p in processes:
                p.join()

        self.spacer_s.surface[self.spacer_s.spacer_mask] = 1
        if return_statistics:
            return {'defect_width': self.grid_spacing, 'r0s': r0s, 'r1s': r1s, 'sls': sls, 'thetas': thetas,
                    'pairs': pairs, 'spaces': spaces}

    def inverse_normalization(self, action_range):
        # inverse actions from -1- 1 to respective range
        action_inverse_normalized = {}
        for key, range_ in action_range.items():
            value = (((self.action_pair[key] + 1) * 0.5) * (max(range_) - min(range_)) + min(range_))
            action_inverse_normalized[key] = value

        return action_inverse_normalized

    def generate_spacer_assign_mat(self):
        self.dfct_statics = self.get_sample_surface(with_defect=True, return_statistics=True)
        self.spacer_s.get_spacer_point_co()
        self.vertices = self.spacer_s.point_coo
        # self.generate_polygon()
        self.get_defect_mask()
        self.update_scene()
        # bpy.ops.wm.save_as_mainfile(filepath='/home/mohanty/Desktop/with_def.blend')

    def _env_prepare_step(self, actions: np.ndarray):
        self.step += 1
        self.total_step += 1
        self.take_action(actions)
        dfct_statics = self.get_sample_surface(with_defect=True, return_statistics=True)
        new_row = pd.DataFrame(dfct_statics, columns=self.df_stats.columns)
        self.df_stat.append(new_row)
        self.spacer_s.get_spacer_point_co()
        self.vertices = self.spacer_s.point_coo
        self.get_defect_mask()
        # self.update_mesh_back_ground()

    def _env_reset(self):
        # global dummy_actions
        self.episodes += 1
        self.total_step += 1
        self.step = 0

        # Generate random Gaussian noise
        noise = np.random.normal(scale=0.1, size=len(self.reset_action))

        # Add noise to the action
        action = np.array(list(self.reset_action.values())) + noise

        # Clip the action to the range bounds
        clipped_action = {key: np.clip(action[i], self.action_bound[key][0],
                                       self.action_bound[key][1]) for i, key in enumerate(self.action_bound.keys())}
        self.reset_action = clipped_action
        self.update_mat(0.5, 2.3, self.reset_action['roughness'])
        self.update_Mix(self.reset_action['Factor'])
        self.update_lights(self.reset_action['energy0'], self.reset_action['ro_x'], self.reset_action['ro_y'])

        # spacer = self.reset_sample_surface(with_defect=False)
        file_path_full = '/home/mohanty/PycharmProjects/Data/spacer_data/synthetic_data2/temp{}/'.format(
            self.g_time_stamp)
        os.makedirs(file_path_full, exist_ok=True)
        try:
            df_stats = pd.concat(self.df_stat, ignore_index=True)
            df_stats.to_csv(file_path_full + 'defect_statistics.csv', index=False)
        except:
            pass
        return self._env_post_step()

    def _env_post_step(self):
        self.update_scene()

        # Setup default image rendering
        global r_
        cam = btb.Camera()
        off = btb.OffScreenRenderer(camera=cam, mode='rgb')
        file_path_full = '/home/mohanty/PycharmProjects/Data/spacer_data/synthetic_data2/temp{}/full/{}'.format(
            self.g_time_stamp, self.episodes)
        img_file_path = file_path_full + '/image'
        mask_file_path = file_path_full + '/mask'
        os.makedirs(img_file_path, exist_ok=True)
        os.makedirs(mask_file_path, exist_ok=True)

        self.def_spacer.hide_render = False
        self.spacer.hide_render = False
        pil_img = off.render(img_file_path + '/image{}_{}.png'.format(self.episodes, self.total_step))
        # pil_img.save(img_file_path + '/image{}_{}.png'.format(self.episodes, self.total_step))

        self.spacer.hide_render = True
        self.def_spacer.hide_render = False

        mat = self.def_spacer.data.materials[0]
        if not mat.use_nodes:
            mat.use_nodes = True
        mat_nodes = mat.node_tree.nodes['Principled BSDF']
        mat_nodes.inputs['Emission'].default_value = (1, 1, 1, 1)

        mask_img = off.render(mask_file_path + '/mask{}_{}.png'.
                              format(self.episodes, self.total_step))
        mat_nodes.inputs['Emission'].default_value = (0, 0, 0, 1)
        # ret, binary_img = cv2.threshold(np.array(mask_img), 1.0, 255, cv2.THRESH_BINARY)
        # RM_BG_img = cv2.bitwise_and(np.array(mask_img), np.array(mask_img), mask=binary_img)
        # RM_BG_img[RM_BG_img != 0] = 255
        #
        # kernel = np.ones((3, 3), np.uint8)
        # img_dilation = cv2.dilate(RM_BG_img, kernel, iterations=1)
        # mask_img = Image.fromarray(img_dilation, mode='L')
        # mask_img.save(mask_file_path + '/mask{}_{}.png'.format(self.episodes, self.total_step))

        # pil_img.show()
        gray_img = np.array(pil_img, dtype=np.uint8)
        # ret, binary_img = cv2.threshold(gray_img, 1.0, 255, cv2.THRESH_BINARY)
        # RM_BG_img = cv2.bitwise_and(gray_img, gray_img, mask=binary_img)
        # blur_img = cv2.GaussianBlur(RM_BG_img, (3, 3), 0)
        # obs = Image.fromarray(RM_BG_img)
        # obs.show()
        # obs = np.asarray(RM_BG_img, dtype=np.uint8)

        # self.state = get_circular_corps(obs, step_angle=20, radius=850, address=self.file_path_croped + '/{}_{}_'
        # .format(self.episodes, self.step))
        windows = view_as_windows(gray_img, (128, 128), step=64)
        result_windows = windows[self.y_128_64, self.x_128_64]
        self.state = Image.fromarray(np.hstack(result_windows))

        if self.total_step % 60 == 0:
            file_path_croped = '/home/mohanty/PycharmProjects/Data/spacer_data/synthetic_data2/temp{}/croped/{}'.format(
                self.g_time_stamp, self.total_step)
            os.makedirs(file_path_croped, exist_ok=True)
            self.state.save(file_path_croped + '/{}_{}_.png'.format(self.episodes, self.step))
        done, r_ = False, 0
        # bpy.ops.wm.save_as_mainfile(filepath='/home/mohanty/Desktop/with_def{}.blend'.format(self.step))
        return dict(obs=self.state, reward=r_, done=done, action_pair=self.action_inverted)

    def take_action(self, actions):
        self.action_pair = dict(zip(self.action_keys, actions))
        action_inverse_mat = self.inverse_normalization(self.action_Material)
        # self.update_mat(action_inverse_mat['specular'], action_inverse_mat['ior'], action_inverse_mat['roughness'])
        self.update_mat(0.5, 2.3, action_inverse_mat['roughness'])

        action_inverse_mix = self.inverse_normalization(self.action_mix)
        self.update_Mix(action_inverse_mix['Factor'])

        # actions_inverse_cmn_ligt = self.inverse_normalization(self.action_light_cmn)
        actions_inverse_specific_ligt = self.inverse_normalization(self.action_light)
        self.update_lights(actions_inverse_specific_ligt['energy0'], actions_inverse_specific_ligt['ro_x'],
                           actions_inverse_specific_ligt['ro_y'])

        self.action_inverted = {**action_inverse_mat, **action_inverse_mix, **actions_inverse_specific_ligt}

        self.action_inverted = {key: round(value, 2) for key, value in self.action_inverted.items()}

        self.update_mapping()
        # self.update_spacer_orientation()

    def update_mat(self, specular, ior, roughness):
        mat = self.spacer.data.materials[0]
        if not mat.use_nodes:
            mat.use_nodes = True
        mat_nodes = mat.node_tree.nodes['Principled BSDF']
        mat_nodes.inputs['Specular'].default_value = specular
        mat_nodes.inputs['IOR'].default_value = ior
        mat_nodes.inputs['Roughness'].default_value = roughness

    def update_noise_texture(self, Scale, n_roughness, Distortion):
        texture_node = self.texture_nodes['Noise Texture']
        texture_node.inputs['Scale'].default_value = Scale
        texture_node.inputs['Roughness'].default_value = n_roughness
        texture_node.inputs['Distortion'].default_value = Distortion

    def update_mapping(self):
        texture_node = self.texture_nodes['Mapping']
        randomize = np.random.random_sample(3, )
        texture_node.inputs['Rotation'].default_value = randomize

    def update_Mix(self, factor):
        texture_node = self.texture_nodes['Mix (Legacy)']
        texture_node.inputs['Fac'].default_value = factor

    def update_lights(self, energy0, ro_x, ro_y):
        #   set color
        # existing_rgb = self.light0.color[:3]
        # h, s, v = colorsys.rgb_to_hsv(*existing_rgb)
        # hsv_color = (h, s, value)
        # rgb_color = colorsys.hsv_to_rgb(*hsv_color)
        # dummy_alpha = tuple([1.0])
        # self.light0.color = rgb_color + dummy_alpha
        #   set energy
        self.light0.data.energy = energy0 # 0.5
        # Exclude for 6 actions
        # self.light0.data.spread = Spread
        # #   set light angle
        self.light0.rotation_euler.x = ro_x
        self.light0.rotation_euler.y = ro_y

    def update_clr_ramp(self, Pos_black, Pos_white):
        color_ramp_node = self.texture_nodes['ColorRamp']
        color_ramp = color_ramp_node.color_ramp
        color_ramp.elements[0].position = Pos_black
        color_ramp.elements[1].position = Pos_white

    # def update_spacer_orientation(self):
    # self.spacer.location.x = np.random.uniform(-0.0003, 0.0003)
    # self.spacer.location.y = np.random.uniform(-0.0003, 0.0003)
    # self.spacer.location.z = np.random.uniform(-0.0003, 0.0003)
    # self.spacer.rotation_euler.z = np.random.uniform(0, 3.14)


def main():
    args, remainder = btb.parse_blendtorch_args()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render-every", default=None, type=int)
    parser.add_argument("--real-time", dest="realtime", action="store_true")
    parser.add_argument("--no-real-time", dest="realtime", action="store_false")
    envargs = parser.parse_args(remainder)

    agent = btb.env.RemoteControlledAgent(
        args.btsockets["GYM"], real_time=envargs.realtime
    )
    env = SpacerEnv(agent)
    # env.attach_default_renderer(every_nth=envargs.render_every)
    env.run(frame_range=(1, 10000), use_animation=False)


if __name__ == "__main__":
    main()
