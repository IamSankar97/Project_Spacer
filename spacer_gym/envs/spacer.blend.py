# Use below only during debugging
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

import sys
import os
import bpy
import bmesh
import numpy as np
import pickle
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
        self.spacer.mat.use_nodes = True
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
        self.action_Material = {'specular': [0.5, 1], 'ior': [1.5, 2.5], 'b_clr_hue': [0.2, 1],
                                'b_clr_satur': [0.2, 1], 'b_clr_value': [0.2, 1]}
        self.action_light_cmn = {"area_size": [36, 150], "hue": [0, 1], "saturation": [0, 1], "value": [0, 1]}
        #   Total 27 keys
        self.action_light = {"Z0": [50, 200], "x_r0": [-35, 35], 'y_r0': [-35, 35], 'energy0': [10, 100],
                             #   light1
                             "X1": [40, 100], "Y1": [40, 100], "Z1": [50, 150],
                             "x_r1": [0, 35], 'y_r1': [20, 40], 'z_r1': [50, 150], 'energy1': [10, 100],
                             #   light2
                             "X2": [-40, -100], "Y2": [40, 100], "Z2": [50, 150],
                             "x_r2": [-15, 15], 'y_r2': [-20, -45], 'z_r2': [-50, -75], 'energy2': [10, 100]
                             }

        self.action_keys = list(self.action_Material.keys()) + list(self.action_light_cmn.keys()) + \
                           list(self.action_light.keys())

    def update_scene(self):
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
                        np.round(np.random.uniform(2, 10), 2), \
                        random.choice([0, 1])
                spacer.randomize_defect(ro, r1, theta0, 70, 40, defect_length, defect_type)
                f.close()
                return spacer, ro, r1, theta0, defect_length, defect_type

    def inverse_normalization(self, action_range):
        action_inverse_normalized = {}
        for key, range in action_range:
            value = (self.action_pair[key] * (range.max() - range.min())) + range.min()
            action_inverse_normalized[key] = value

        return action_inverse_normalized

    def _env_prepare_step(self, actions: np.ndarray):
        self.take_action(actions)
        spacer, ro, r1, theta0, defect_length, defect_type = \
            self.get_sample_surface()
        self.update_mesh_back_ground(np.array(spacer.point_coo[['X', 'Y', 'Z']]))

    def _env_reset(self):
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
        action_material = self.inverse_normalization(self.action_Material)
        action_light_cmn = self.inverse_normalization(self.action_light_cmn)
        action_light = self.inverse_normalization(self.action_light)

        self.update_mat(action_material)
        self.update_lights(action_light_cmn, action_light_specific)

    def update_mat(self, actions):
        mat_nodes = self.spacer.node_tree.nodes['Principled BSDF']
        mat_nodes.inputs['Specular'].default_value = actions['specular']
        mat_nodes.inputs['IOR'].default_value = actions['ior']
        mat_nodes.inputs['Base Color'].default_value = (actions['b_clr_hue'], actions['b_clr_satur'],
                                                        actions['b_clr_value'], 1)

    def update_lights(self, cmn_actions, specific_actions):

        #   set size
        self.light0.size, self.light1.size, self.light2.size = cmn_actions["area_size"]

        #   set color
        light0_rgb = self.light0.color
        hsv_color = light0_rgb.hsv
        hsv_color[0], hsv_color[1], hsv_color[2] = cmn_actions['hue'], cmn_actions['saturation'], cmn_actions['value']
        light0_rgb = hsv_color.rgb
        self.light0.color, self.light1.color, self.light2.color = light0_rgb

        #   set energy
        self.light0.data.energy, self.light1.data.energy, self.light2.data.energy = specific_actions['energy0'], \
            specific_actions['energy1'], specific_actions['energy2']

        #   set locations
        self.light0.location.z = specific_actions["Z0"]
        self.light0.rotation_euler = (specific_actions["x_r"], specific_actions["y_r"], self.light0.rotation_euler[2])

        self.light1.location = (specific_actions["X1"], specific_actions["Y1"], specific_actions["Z1"])
        self.light1.rotation_euler = (specific_actions["x_r1"], specific_actions["y_r1"], specific_actions["z_r1"])

        self.light2.location = (specific_actions["X2"], specific_actions["Y2"], specific_actions["Z2"])
        self.light2.rotation_euler = (specific_actions["x_r2"], specific_actions["y_r2"], specific_actions["z_r2"])

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


main()
