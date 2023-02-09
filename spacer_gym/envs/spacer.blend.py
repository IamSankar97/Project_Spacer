# Use below only during debugging
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

import math
import sys
import os
import bpy
import bmesh
import colorsys
import numpy as np
import pickle
import pandas as pd
import random
from blendtorch import btb
from spacer import Spacer

sys.path.append(os.getcwd())
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs')


class SpacerEnv(btb.env.BaseEnv):
    def __init__(self, agent):
        super().__init__(agent)
        self.action_pair = None
        self.spacer = bpy.data.objects["spacer_ring"]
        self.camera = bpy.data.objects["Camera"]
        self.light0 = bpy.data.objects["L0_top"]
        self.light1 = bpy.data.objects["L1_0"]
        self.light2 = bpy.data.objects["L2_120"]
        self.img_addr = 'spacer_gym/spacer_env_render/image_spacer.png'
        # Note, ensure that physics run at same speed.
        self.fps = bpy.context.scene.render.fps
        self.avg_pixel = 30
        self.np_random = None
        self.state = 40
        self.vertices = None
        self.topology_dir = 'topology/pkl_5'
        self.topologies = os.listdir(self.topology_dir)
        #   (3)
        self.action_Material = {'specular': [0.5, 1], 'ior': [1.5, 2.5], 'b_clr_value': [0.2, 1]}  # 'b_clr_hue': [0.2, 1], 'b_clr_satur': [0.2, 1],
        #   (4)
        self.action_light_cmn = {"area_size": [0.036, 0.15], "hue": [0, 1], "saturation": [0, 1], "value": [0, 1]}
        #   (4, 7, 7)
        self.action_light = {"Z0": [0.05, 0.2], "x_r0": [-35, 35], 'y_r0': [-35, 35], 'energy0': [0.01, 0.1],
                             #   light1
                             "X1": [0.04, 0.1], "Y1": [0.01, 0.1], "Z1": [0.05, 0.15],
                             "x_r1": [0, 35], 'y_r1': [20, 40], 'z_r1': [50, 150], 'energy1': [0.01, 0.1],
                             #   light2
                             "X2": [-0.04, -0.1], "Y2": [0.04, 0.1], "Z2": [0.05, 0.15],
                             "x_r2": [-15, 15], 'y_r2': [-20, -45], 'z_r2': [-50, -75], 'energy2': [0.01, 0.1]}
        # Total = 25
        self.action_keys = list(self.action_Material.keys()) + list(self.action_light_cmn.keys()) + \
                           list(self.action_light.keys())

        self.initial_action = pd.read_csv('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs/'
                                          'initial_actions2.csv', header=None)
        # bpy.ops.object.select_all(action='DESELECT')
        #
        # for area in bpy.context.screen.areas:
        #     if area.type == 'VIEW_3D':
        #         # Found an active 3D View
        #         space = area.spaces[0]
        #         break
        # else:
        #     # No active 3D View found
        #     print("No active 3D View found.")
        #     space = None
        #
        # if space:
        #     # Set the shading mode to 'RENDERED'
        #     space.shading.type = 'RENDERED'

    def update_scene(self):
        print()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def update_mesh_back_ground(self, new_realisation: np.ndarray):
        """

        :param new_realisation: a numpy array with new mesh deta
        :param existing_mesh: existing mesh where the new_realisation data to be updated
        :return: updates mesh in blender
        """
        # Get a BMesh representation
        bm = bmesh.new()  # create an empty BMesh
        bm.from_mesh(self.spacer.data)  # fill it in from a Mesh
        self.vertices = list(new_realisation)
        for vertice_old, vertice_new in zip(bm.verts, self.vertices):
            if vertice_old.co.z != 1:
                vertice_old.co.z = vertice_new[2]

        bm.to_mesh(self.spacer.data)  # Finish up, write the bmesh back to the mesh
        bm.free()
        self.update_scene()  # Updates the mesh in scene by refreshing the screen

    def get_sample_surface(self):
        filename = random.choice(self.topologies)
        if filename.endswith('.pkl'):
            with open(os.path.join(self.topology_dir, filename), 'rb') as f:
                my_realisation = pickle.load(f) * 1e-6
                grid_spacing, sample_surface = my_realisation[0][0], np.array(my_realisation[1:, :])

                spacer = Spacer(sample_surface, grid_spacing)
                ro, r1, theta0, defect_length, defect_type = \
                    np.round(np.random.uniform(12.5, 13.2), 2), \
                        np.round(np.random.uniform(13.8, 16), 2), \
                        np.random.randint(1, 85), \
                        np.round(np.random.uniform(0.1, 10), 2), \
                        random.choice([0, 1])
                spacer.randomize_defect(ro, r1, theta0, 70, 40, defect_length, defect_type)
                f.close()
                return spacer, ro, r1, theta0, defect_length, defect_type

    def inverse_normalization(self, action_range):
        action_inverse_normalized = {}
        for key, range_ in action_range.items():
            value = (self.action_pair[key] * (max(range_) - min(range_)) + min(range_))
            action_inverse_normalized[key] = value

        return action_inverse_normalized

    def _env_prepare_step(self, actions: np.ndarray):
        self.take_action(actions)
        spacer, ro, r1, theta0, defect_length, defect_type = \
            self.get_sample_surface()
        self.update_mesh_back_ground(np.array(spacer.point_coo[['X', 'Y', 'Z']]))

    def _env_reset(self):
        dummy_actions =np.array(self.initial_action.sample())[0]
        self.take_action(dummy_actions)

        return self._env_post_step()

    def _env_post_step(self):

        # Setup default image rendering
        global r_
        cam = btb.Camera()
        off = btb.OffScreenRenderer(camera=cam, mode='rgb')
        self.state = off.render(image_size=(448, 448))

        done, r_ = False, 1

        return dict(obs=self.state, reward=r_, done=done)

    def take_action(self, actions):
        self.action_pair = dict(zip(self.action_keys, actions))

        self.update_mat()
        self.update_lights()

    def update_mat(self):
        actions = self.inverse_normalization(self.action_Material)
        mat = self.spacer.data.materials[0]
        if not mat.use_nodes:
            mat.use_nodes = True
        mat_nodes = mat.node_tree.nodes['Principled BSDF']
        mat_nodes.inputs['Specular'].default_value = actions['specular']
        mat_nodes.inputs['IOR'].default_value = actions['ior']
        base_clr_rgb = mat_nodes.inputs['Base Color'].default_value[:3]
        h, s, v = colorsys.rgb_to_hsv(*base_clr_rgb)
        v_new = actions['b_clr_value']
        r, g, b = colorsys.hsv_to_rgb(h, s, v_new)
        mat_nodes.inputs['Base Color'].default_value = (r, g, b, 1)

    def update_lights(self):
        cmn_actions = self.inverse_normalization(self.action_light_cmn)
        specific_actions = self.inverse_normalization(self.action_light)

        #   set size
        self.light0.data.size, self.light1.data.size, self.light2.data.size = cmn_actions["area_size"], cmn_actions[
            "area_size"], cmn_actions["area_size"]

        #   set color
        hsv_color = (cmn_actions['hue'], cmn_actions['saturation'], cmn_actions['value'])
        rgb_color = colorsys.hsv_to_rgb(*hsv_color)
        dummy_alpha = tuple([1.0])
        self.light0.color, self.light1.color, self.light2.color = rgb_color + dummy_alpha, rgb_color + dummy_alpha, \
                                                                  rgb_color + dummy_alpha
        #   set energy
        self.light0.data.energy, self.light1.data.energy, self.light2.data.energy = specific_actions['energy0'], \
            specific_actions['energy1'], specific_actions['energy2']

        #   set locations
        self.light0.location.z = specific_actions["Z0"]
        radians0 = (specific_actions["x_r0"], specific_actions["y_r0"], self.light0.rotation_euler[2])
        self.light0.rotation_euler = [math.radians(r) for r in radians0]

        self.light1.location = (specific_actions["X1"], specific_actions["Y1"], specific_actions["Z1"])
        radians1 = (specific_actions["x_r1"], specific_actions["y_r1"], specific_actions["z_r1"])
        self.light1.rotation_euler = [math.radians(r) for r in radians1]

        self.light2.location = (specific_actions["X2"], specific_actions["Y2"], specific_actions["Z2"])
        radians2 = (specific_actions["x_r2"], specific_actions["y_r2"], specific_actions["z_r2"])
        self.light2.rotation_euler = [math.radians(r) for r in radians2]

    # def run(self, frame_range=None, use_animation=True):
    #     super().run(frame_range, use_animation)


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