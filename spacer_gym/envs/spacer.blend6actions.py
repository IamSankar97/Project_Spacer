# # Use below only during debugging
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

import sys
import os

from skimage.util import view_as_windows

sys.path.append(os.getcwd())
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs')
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/')

import bpy
import time
import bmesh
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
from img_processing import remove_back_ground

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class SpacerEnv(btb.env.BaseEnv):
    def __init__(self, agent):
        super().__init__(agent)
        self.action_inverted = None
        self.action_pair = None
        self.spacer = bpy.data.objects["spacer_ring"]
        self.camera = bpy.data.objects["Camera"]
        self.light0 = bpy.data.objects["L0_top"]
        self.episodes, self.step, self.total_step = -2, 0, 0
        # self.light1 = bpy.data.objects["L1_0"]
        # self.light1 = bpy.data.objects["L2_240"]
        # Note, ensure that physics run at same speed.
        self.fps = bpy.context.scene.render.fps
        self.texture_nodes = bpy.data.materials.get("spacer").node_tree.nodes
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
        #   (3)
        self.action_Material = {'specular': [0.3, 0.7], 'ior': [2, 2.6], 'roughness': [0, 0.5]}
        #   (1)
        self.action_mix = {'Factor': [0.0, 0.3]} #{'Factor': [0.0, 0.4]} old noise material3
        #   (1)
        self.action_light_cmn = {"value": [0.8, 1]}
        #   (1)
        self.action_light = {'energy0': [0.001, 0.015]}
        # Total = 6

        self.reset_action = {'specular': 0.5, 'ior': 2.3, 'roughness': 0.2, 'factor': 0.1, "value": 0.8,
                             'energy0': 0.005, 'ro_z': 0}
        self.action_bound = {**self.action_Material, **self.action_mix, **self.action_light_cmn, **self.action_light}
        self.action_keys = list(self.action_bound.keys())
        self.update_scene()
        time.sleep(5)

    def update_scene(self):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def update_vertices(self, chunk):
        for vertice_old, vertice_new in chunk:
            if vertice_old.co.z != 1:
                vertice_old.co.z = vertice_new[2]

    def update_mesh_back_ground(self):
        """
        :return: updates mesh in blender
        """
        z_coordinates = np.array(self.vertices).flatten()

        # Update the z-coordinates directly in the mesh data
        self.spacer.data.vertices.foreach_set("co", z_coordinates)

        # Update the mesh in Blender
        self.spacer.data.update() # Updates the mesh in scene by refreshing the screen

    def get_sample_surface(self, with_defect=True):
        filename = random.choice(self.topologies)
        if filename.endswith('.pkl'):
            with open(os.path.join(self.topology_dir, filename), 'rb') as f:
                my_realisation = pickle.load(f) * 1e-6
                grid_spacing, sample_surface = my_realisation[0][0], np.array(my_realisation[1:, :])

                spacer = Spacer(sample_surface, grid_spacing)
                if with_defect:
                    ro, r1, theta0, defect_length, defect_type = \
                        np.round(np.random.uniform(12.5, 15), 2), \
                            np.round(np.random.uniform(13.8, 16), 2), \
                            np.random.randint(1, 85), \
                            np.round(np.random.uniform(0.1, 10), 2), \
                            random.choice([0, 1])
                    spacer.randomize_defect(ro, r1, theta0, 70, 40, defect_length, defect_type)
                # spacer.randomize_defect(ro, r1, np.random.randint(92, 179), 70, 40, defect_length, 0)
                # spacer.randomize_defect(ro, r1, np.random.randint(180, 240), 70, 40, defect_length, 0)
                # spacer.randomize_defect(ro, r1, np.random.randint(240, 360), 70, 40, defect_length, 0)
                f.close()
                return spacer

    def reset_sample_surface(self, with_defect=True):
        filename = 'points_6_dx22.182_0.pkl'  # points_5_dx18.485_0.pkl'
        if filename.endswith('.pkl'):
            with open(os.path.join(self.topology_dir, filename), 'rb') as f:
                my_realisation = pickle.load(f) * 1e-6
                grid_spacing, sample_surface = my_realisation[0][0], np.array(my_realisation[1:, :])

                spacer = Spacer(sample_surface, grid_spacing)
                if with_defect:
                    ro, r1, theta0, defect_length, defect_type = 12.5, 13.8, 2, 3, 0
                    spacer.randomize_defect(ro, r1, 70, 40, theta0, defect_length, defect_type)
                f.close()
                return spacer

    def inverse_normalization(self, action_range):
        # inverse actions from -1- 1 to respective range
        action_inverse_normalized = {}
        for key, range_ in action_range.items():
            value = (((self.action_pair[key] + 1) * 0.5) * (max(range_) - min(range_)) + min(range_))
            action_inverse_normalized[key] = value

        return action_inverse_normalized

    def _env_prepare_step(self, actions: np.ndarray):
        self.take_action(actions)
        # spacer = self.get_sample_surface(with_defect=False)
        # self.update_mesh_back_ground(np.array(spacer.point_coo[['X', 'Y', 'Z']]))

    def _env_reset(self):
        # global dummy_actions
        self.episodes += 1
        self.step = 0

        # Generate random Gaussian noise
        noise = np.random.normal(scale=0.1, size=len(self.reset_action))

        # Add noise to the action
        action = np.array(list(self.reset_action.values())) + noise

        # Clip the action to the range bounds
        clipped_action = {key: np.clip(action[i], self.action_bound[key][0],
                                       self.action_bound[key][1]) for i, key in enumerate(self.action_bound.keys())}
        self.reset_action = clipped_action
        self.update_mat(self.reset_action['specular'], self.reset_action['ior'], self.reset_action['roughness'])
        self.update_Mix(self.reset_action['Factor'])
        self.update_lights(self.reset_action['value'], self.reset_action['energy0'])

        # spacer = self.reset_sample_surface(with_defect=False)
        # self.update_mesh_back_ground(np.array(spacer.point_coo[['X', 'Y', 'Z']]))
        return self._env_post_step()

    def _env_post_step(self):
        self.step += 1
        self.total_step += 1
        self.update_scene()

        # Setup default image rendering
        global r_
        cam = btb.Camera()
        off = btb.OffScreenRenderer(camera=cam, mode='rgb')
        file_path_full = '/home/mohanty/PycharmProjects/Data/spacer_data/synthetic_data2/temp{}/full/{}'.format(
            self.g_time_stamp, self.episodes)
        os.makedirs(file_path_full, exist_ok=True)
        file_path = file_path_full + '/image{}_{}.png'.format(self.episodes, self.step)

        pil_img = off.render(file_path)
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
                self.g_time_stamp, self.episodes)
            os.makedirs(file_path_croped, exist_ok=True)
            self.state.save(file_path_croped + '/{}_{}_.png'.format(self.episodes, self.step))
        done, r_ = False, 0

        return dict(obs=self.state, reward=r_, done=done, action_pair=self.action_inverted)

    def take_action(self, actions):
        self.action_pair = dict(zip(self.action_keys, actions))
        action_inverse_mat = self.inverse_normalization(self.action_Material)
        self.update_mat(action_inverse_mat['specular'], action_inverse_mat['ior'], action_inverse_mat['roughness'])

        action_inverse_mix = self.inverse_normalization(self.action_mix)
        self.update_Mix(action_inverse_mix['Factor'])

        actions_inverse_cmn_ligt = self.inverse_normalization(self.action_light_cmn)
        actions_inverse_specifice_ligt = self.inverse_normalization(self.action_light)
        self.update_lights(actions_inverse_cmn_ligt['value'], actions_inverse_specifice_ligt['energy0'])

        self.action_inverted = {**action_inverse_mat, **action_inverse_mix, **actions_inverse_cmn_ligt,
                                **actions_inverse_specifice_ligt}

        self.action_inverted = {key: round(value, 2) for key, value in self.action_inverted.items()}

        self.update_mapping()
        self.update_spacer_orientation()

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
        randomize = np.random.random_sample(3,)
        texture_node.inputs['Rotation'].default_value = randomize

    def update_Mix(self, factor):
        texture_node = self.texture_nodes['Mix (Legacy)']
        texture_node.inputs['Fac'].default_value = factor

    def update_lights(self, value, energy0):
        #   set color
        existing_rgb = self.light0.color[:3]
        h, s, v = colorsys.rgb_to_hsv(*existing_rgb)
        hsv_color = (h, s, value)
        rgb_color = colorsys.hsv_to_rgb(*hsv_color)
        dummy_alpha = tuple([1.0])
        self.light0.color = rgb_color + dummy_alpha
        #   set energy
        self.light0.data.energy = energy0
        # Exclude for 6 actions
        # self.light0.data.spread = Spread
        # #   set light angle
        # self.light0.rotation_euler.x = ro_x
        # self.light0.rotation_euler.y = ro_y

    def update_clr_ramp(self, Pos_black, Pos_white):
        color_ramp_node = self.texture_nodes['ColorRamp']
        color_ramp = color_ramp_node.color_ramp
        color_ramp.elements[0].position = Pos_black
        color_ramp.elements[1].position = Pos_white

    def update_spacer_orientation(self):
        # self.spacer.location.x = np.random.uniform(-0.0003, 0.0003)
        # self.spacer.location.y = np.random.uniform(-0.0003, 0.0003)
        # self.spacer.location.z = np.random.uniform(-0.0003, 0.0003)
        self.spacer.rotation_euler.z = np.random.uniform(0, 3.14)


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
