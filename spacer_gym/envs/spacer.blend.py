# # Use below during debugging
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

import bpy
import bmesh
import time
import multiprocessing
import sys
import os
from skimage.util import view_as_windows
import warnings
from PIL import Image
import numpy as np
import pandas as pd
import random
from blendtorch import btb
from spacer import Spacer
import datetime
from utils import pol2cart, get_theta

sys.path.append(os.getcwd())
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs')
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generate_defect(surface, grid_spacing, r0, r1, scratch_length, theta0, pair=1, space=0):
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
    beta = random.choice([70, 80])
    alpha = random.choice([35, 40])

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
    x0, y0 = pol2cart(r0, np.radians(theta0))
    x1, y1 = pol2cart(r1, get_theta(r0, r1, theta0, scratch_length))

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

    if pair != 1 and space != 0:
        offset = np.arange(pair)
        offset_space = np.linspace(0, space, len(offset))
        offset = offset + offset_space
        x_coords = np.repeat(x_coords, pair) + np.tile(offset, len(x_coords))
        y_coords = np.repeat(y_coords, pair)
        x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)

    mask = (x_coords < surface.shape[0]) & (y_coords < surface.shape[1])
    noise = np.random.uniform(low=-1e-7, high=1e-6, size=x_coords[mask].shape)

    # Add the noise to the height of the bump
    h_bump = h_defect + noise
    surface[x_coords[mask], y_coords[mask]] = h_bump


class SpacerEnv(btb.env.BaseEnv):
    def __init__(self, agent):
        super().__init__(agent)
        self.training = True    # If training true defect data and it's mask image is not generated.
        # Sapcer Dimesnions
        self.outer_radius = 16 * 1e-3
        self.thickness = 3.222 * 1e-3
        self.grid_spacing = 0.00001  # 1e-6
        self.grid_radius = self.outer_radius + self.grid_spacing
        self.grid_dim = int(self.grid_radius / self.grid_spacing) * 2

        self.camera = bpy.data.objects["Camera"]
        self.camera.location = (0.0, 0.0, 0.15)
        self.light0 = bpy.data.objects["L0_top"]
        self.light0.location = (0.0, 0.0, 0.15)

        self.episodes, self.step, self.total_step = -2, 0, 0

        self.topology_dir = '/home/mohanty/PycharmProjects/Data/pkl_6/'
        self.g_time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.topologies = os.listdir(self.topology_dir)
        self.x_128_64 = [12, 13, 14, 15, 16, 17, 18, 9, 10, 20, 21, 3, 27, 3, 27, 2, 28, 2, 28, 2, 28, 2, 28, 2, 28, 2,
                         28, 2, 28, 3, 27, 3, 27, 9, 10, 20, 21, 12, 13, 14, 15, 16, 17, 18]

        self.y_128_64 = [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 9, 9, 10, 10, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
                         18, 18, 20, 20, 21, 21, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28]

        # Action Space
        #   (3)
        self.action_Material = {'roughness': [0, 0.5], 'specular': [0, 0.5], 'ior': [2, 2.5]}
        #   (1)
        self.action_mix = {'Factor': [0.0, 0.3]}
        #   (3)
        self.action_light = {'energy0': [0.02, 0.18], 'ro_x': [-0.45, 0.45], 'ro_y': [-0.45, 0.45]}
        # Total = 5
        self.reset_action = {'specular': 0.2, 'ior': 2.3, 'roughness': 0.1, 'factor': 0.05, "value": 0.8,
                             'energy0': 0.01, 'ro_x': 0, 'ro_y': 0}
        self.action_bound = {**self.action_Material, **self.action_mix, **self.action_light}
        self.action_keys = list(self.action_bound.keys())

        self.state = None
        self.vertices = None
        self.def_spacer = not self.training
        self.action_inverted = None
        self.action_pair = None

        self.generate_spacer_assign_mat()
        self.texture_nodes = bpy.data.materials.get("spacer").node_tree.nodes
        self.df_stats = pd.DataFrame(columns=['r0s', 'r1s', 'sls', 'thetas', 'pairs', 'spaces'])
        self.df_stat = []


    def update_scene(self):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def update_vertices(self, chunk):
        for vertice_old, vertice_new in chunk:
            if vertice_old.co.z != 1:
                vertice_old.co.z = vertice_new[2]

    def generate_polygon(self, spacer_mesh_name: str = 'spacer_ring', smoothing: bool = 0):
        """
        Parameters
        ----------
        spacer_mesh_name
        def_mesh_name
        smoothing

        Returns
        -------
        """

        #   ********* Assuming we have a rectangular grid *************
        x_diff = np.diff(self.vertices[:, 0])
        x_change_idx = np.nonzero(x_diff)[0][0] + 1
        self.xSize = x_change_idx
        self.Size = len(self.vertices) // self.xSize

        #   Generate the polygons (four vertices linked in a face)
        polygons = []
        for i in range(1, len(self.vertices) - self.xSize):
            poly_set = np.array([self.vertices[i][2], self.vertices[i - 1][2], self.vertices[i - 1 + self.xSize][2],
                                 self.vertices[i + self.xSize][2]])
            # collecting spacer and defect polygons
            if i % self.xSize != 0 and self.vertices[i][2] != 1 and np.all(poly_set != 1):
                polygons.append((i, i - 1, i - 1 + self.xSize, i + self.xSize))

        mesh = bpy.data.meshes.new(spacer_mesh_name)  # Create the mesh (inner data)
        obj = bpy.data.objects.new(spacer_mesh_name, mesh)  # Create an object
        obj.data.from_pydata(self.vertices, [], polygons)  # Associate vertices and polygons

        if smoothing:
            for p in obj.data.polygons:  # Set smooth shading (if needed)
                p.use_smooth = True

        bpy.context.scene.collection.objects.link(obj)  # Link the object to the scene
        self.spacer = bpy.data.objects[spacer_mesh_name]
        self.spacer.location = (0.0, 0.0, 0.0)

        self.material = bpy.data.materials.get("spacer")

        # Assign the material to the object
        self.spacer.active_material = self.material
        bpy.context.view_layer.objects.active = self.spacer
        self.spacer.select_set(True)

    def get_defect_mask(self, def_mesh_name: str = 'spacer_defect'):

        # create an array of indices for each polygon
        indices = np.arange(1, len(self.vertices) - self.xSize)
        # create an array of vertices for each polygon
        vertices = np.array([indices, indices - 1, indices - 1 + self.xSize, indices + self.xSize]).T
        # create a boolean mask for the defect polygons
        mask = (indices % self.xSize != 0) & (self.vertices[indices, 2] != 1) & (self.vertices[indices, 2] != 0)
        # create a boolean mask for the polygons with defects
        poly_set = self.vertices[vertices[:, :], 2]

        mask = mask & np.all(poly_set != 1, axis=1) & np.any((-1 <= poly_set) & (poly_set != 0), axis=1)
        # filter the indices using the mask
        polygons_def = vertices[mask, :].tolist()

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
            self.def_spacer.location = (0.0, 0.0, 0.0)
            self.def_spacer.active_material = self.material

    def update_mesh_back_ground(self):
        """
        :return: updates mesh in blender
        """
        z_coordinates = np.array(self.vertices).flatten()

        # Update the z-coordinates directly in the mesh data
        self.spacer.data.vertices.foreach_set("co", z_coordinates)

        # Update the mesh in Blender
        self.spacer.data.update()  # Updates the mesh in scene by refreshing the screen

        if not self.training:
            self.get_defect_mask()

    def get_sample_surface(self, with_defect=True, return_statistics=False, No_of_Defect=5):
        shared_matrix = multiprocessing.Array('i', self.grid_dim * self.grid_dim)
        self.spacer_s = Spacer(shared_matrix, self.grid_spacing, self.outer_radius, self.thickness)
        if with_defect:
            # no of defects
            subinterval = int(360 / No_of_Defect)
            thetas = np.array([i * subinterval for i in range(No_of_Defect)])
            r0s = np.round(np.random.uniform(12.5, 16, No_of_Defect), 2)
            r1s = np.round(np.random.uniform(12.5, 16, No_of_Defect), 2)
            sls = np.round(np.random.uniform(0.1, 10, No_of_Defect), 2)
            pairs = np.random.randint(1, 5, size=No_of_Defect)
            spaces = np.random.choice([0, 8, 16, 20, 24], size=No_of_Defect)

            # Below code is to be used during debugging as multiprocessing cant be debugged
            # generate_defect(surface=self.spacer_s.surface, grid_spacing=self.spacer_s.grid_spacing, r0=r0s[0],
            #                 r1=r1s[0], scratch_length=sls[0], theta0=thetas[0], width=widths[0], space=spaces[0])

            processes = [multiprocessing.Process(target=generate_defect,
                                                 args=(self.spacer_s.surface, self.spacer_s.grid_spacing, r0,
                                                       r1, sl, theta, pair, space))
                         for r0, r1, sl, theta, pair, space in zip(r0s, r1s, sls, thetas, pairs, spaces)]

            for p in processes:
                p.start()

            # wait for all the processes to finish
            for p in processes:
                p.join()
        self.spacer_s.surface[self.spacer_s.spacer_mask] = 1
        if return_statistics:
            return {'r0s': r0s, 'r1s': r1s, 'sls': sls, 'thetas': thetas, 'pairs': pairs, 'spaces': spaces}

    def inverse_normalization(self, action_range):
        # inverse actions from -1- 1 to respective range
        action_inverse_normalized = {}
        for key, range_ in action_range.items():
            value = (((self.action_pair[key] + 1) * 0.5) * (max(range_) - min(range_)) + min(range_))
            action_inverse_normalized[key] = value

        return action_inverse_normalized

    def generate_spacer_assign_mat(self):
        if self.training:
            self.spacer = bpy.data.objects['spacer_ring_train.64']

        else:
            self.dfct_statics = self.get_sample_surface(with_defect=not self.training,
                                                        return_statistics=not self.training)
            self.spacer_s.get_spacer_point_co()
            self.vertices = self.spacer_s.point_coo
            self.generate_polygon()
            self.get_defect_mask()
        self.update_scene()

    def _env_prepare_step(self, actions: np.ndarray):
        self.step += 1
        self.total_step += 1
        self.take_action(actions)

        if not self.training:
            dfct_statics = self.get_sample_surface(with_defect=not self.training, return_statistics=not self.training)

            new_row = pd.DataFrame(dfct_statics, columns=self.df_stats.columns)
            self.df_stat.append(new_row)
            self.spacer_s.get_spacer_point_co()
            self.vertices = self.spacer_s.point_coo
            self.update_mesh_back_ground()

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

        file_path_full = '/home/mohanty/PycharmProjects/Data/spacer_data/synthetic_data2/temp{}/'.format(
            self.g_time_stamp)
        os.makedirs(file_path_full, exist_ok=True)

        if not self.training:
            df_stats = pd.concat(self.df_stat, ignore_index=True)
            df_stats.to_csv(file_path_full + 'defect_statistics.csv', index=False)

        return self._env_post_step()

    def _env_post_step(self, save_blend_file=False):
        self.update_scene()
        # Setup default image rendering
        global r_
        cam = btb.Camera()
        off = btb.OffScreenRenderer(camera=cam, mode='rgb')
        file_path_full = '/home/mohanty/PycharmProjects/Data/spacer_data/synthetic_data2/temp{}/full/{}'.format(
            self.g_time_stamp, self.episodes)
        img_file_path = file_path_full + '/image'
        os.makedirs(img_file_path, exist_ok=True)

        self.spacer.hide_render = False
        pil_img = off.render(img_file_path + '/image{}_{}.png'.format(self.episodes, self.total_step))

        if not self.training:
            mask_file_path = file_path_full + '/mask'
            os.makedirs(mask_file_path, exist_ok=True)
            self.spacer.hide_render = True
            self.def_spacer.hide_render = False
            off.render(mask_file_path + '/mask{}_{}.png'.format(self.episodes, self.total_step))

        gray_img = np.array(pil_img, dtype=np.uint8)

        windows = view_as_windows(gray_img, (128, 128), step=64)
        result_windows = windows[self.y_128_64, self.x_128_64]
        self.state = Image.fromarray(np.hstack(result_windows))

        if self.total_step % 60 == 0:
            file_path_croped = '/home/mohanty/PycharmProjects/Data/spacer_data/synthetic_data2/temp{}/croped/{}'.format(
                self.g_time_stamp, self.total_step)
            os.makedirs(file_path_croped, exist_ok=True)
            self.state.save(file_path_croped + '/{}_{}_.png'.format(self.episodes, self.step))
        done, r_ = False, 0
        if save_blend_file:
            bpy.ops.wm.save_as_mainfile(filepath='/home/mohanty/Desktop/with_def{}.blend'.format(self.step))
        return dict(obs=self.state, reward=r_, done=done, action_pair=self.action_inverted)

    def take_action(self, actions):
        self.action_pair = dict(zip(self.action_keys, actions))
        action_inverse_mat = self.inverse_normalization(self.action_Material)
        self.update_mat(action_inverse_mat['specular'], action_inverse_mat['ior'], action_inverse_mat['roughness'])

        action_inverse_mix = self.inverse_normalization(self.action_mix)
        self.update_Mix(action_inverse_mix['Factor'])

        actions_inverse_specific_ligt = self.inverse_normalization(self.action_light)
        self.update_lights(actions_inverse_specific_ligt['energy0'], actions_inverse_specific_ligt['ro_x'],
                           actions_inverse_specific_ligt['ro_y'])

        self.action_inverted = {**action_inverse_mat, **action_inverse_mix, **actions_inverse_specific_ligt}

        self.action_inverted = {key: round(value, 2) for key, value in self.action_inverted.items()}

        self.update_mapping()

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
        self.light0.data.energy = energy0
        self.light0.rotation_euler.x = ro_x
        self.light0.rotation_euler.y = ro_y

    def update_clr_ramp(self, Pos_black, Pos_white):
        color_ramp_node = self.texture_nodes['ColorRamp']
        color_ramp = color_ramp_node.color_ramp
        color_ramp.elements[0].position = Pos_black
        color_ramp.elements[1].position = Pos_white


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
