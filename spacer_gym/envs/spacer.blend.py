import sys
import os

# Use below only during debugging
import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

sys.path.append(os.getcwd())
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs')

import bpy
import bmesh
import numpy as np
import pickle
from PIL import Image
from gym import logger, spaces
import random
from blendtorch import btb
from spacer import Spacer


def get_sample_surface(folder):
    while True:
        count = 0
        for filename in os.listdir(folder):
            if filename.endswith('.pkl'):
                count += 1
                with open(os.path.join(folder, filename), 'rb') as f:
                    my_realisation = pickle.load(f) * 1e-6
                    grid_spacing = my_realisation[0][0]
                    sample_surface = np.array(my_realisation[1:, :])

                    spacer = Spacer(sample_surface, grid_spacing)
                    ro, r1, theta0, defect_length, defect_type = \
                        np.round(np.random.uniform(12.5, 13.2), 2), \
                        np.round(np.random.uniform(13.8, 16), 2), \
                        np.random.randint(1, 85), \
                        np.round(np.random.uniform(2, 10), 2), \
                        count % 2
                    spacer.randomize_defect(ro, r1, theta0, 70, 40, defect_length, defect_type)

                    yield spacer, ro, r1, theta0, defect_length, defect_type


class SpacerEnv(btb.env.BaseEnv):
    def __init__(self, agent):
        super().__init__(agent)

        self.mat = None
        self.mesh_name = None
        self.vertices = None
        self.unit = None
        self.objs = None

        self.spacer = bpy.data.objects["spacer_ring"]
        self.camera = bpy.data.objects["Camera"]
        self.light = bpy.data.objects["Light"]
        self.img_addr = 'spacer_gym/spacer_env_render/image_spacer.png'
        # Note, ensure that physics run at same speed.
        self.fps = bpy.context.scene.render.fps
        self.avg_pixel = 30
        self.episodes = 110
        self.episode_length = self.episodes
        self.reward_threshold = 1000
        self.np_random = None
        self.state = 40
        self.steps_beyond_done = None

    def update_scene(self):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def update_mesh_back_ground(self, new_realisation: np.ndarray, existing_mesh):
        """

        :param new_realisation: a numpy array with new mesh deta
        :param existing_mesh: existing mesh where the new_realisation data to be updated
        :return: updates mesh in blender
        """
        # Get a BMesh representation
        bm = bmesh.new()  # create an empty BMesh
        bm.from_mesh(existing_mesh.data)  # fill it in from a Mesh
        self.vertices = list(new_realisation)
        for vertice_old, vertice_new in zip(bm.verts, self.vertices):
            vertice_old.co.z = vertice_new[2]

        bm.to_mesh(existing_mesh.data)  # Finish up, write the bmesh back to the mesh
        bm.free()
        self.update_scene()  # Updates the mesh in scene by refreshing the screen

    def render_(self, resolution: tuple = (2448, 2048), save: bool = True,
                engine: str = 'CYCLES'):

        if engine == "CYCLES":
            # Set the rendering engine to Cycles
            bpy.context.scene.render.engine = engine
            bpy.context.scene.render.filepath = self.img_addr

            bpy.ops.render.render(write_still=save)
            image = Image.open(self.img_addr)
            return image

        elif engine == 'EEVEE':
            # bpy.context.scene.render.engine = 'BLENDER_EEVEE'

            cam = btb.Camera()
            off = btb.OffScreenRenderer(camera=cam, mode='rgb')
            off.set_render_style(shading='RENDERED', overlays=False)
            # env.render()
            image = off.render()

            return image

    def _env_prepare_step(self, action: np.ndarray):
        print('prepare_step-----')
        self._action(action)
        self.episode_length -= 1

    def _env_reset(self):
        print("reset-----")
        self.steps_beyond_done = None
        self.episode_length = self.episodes
        self.light.data.energy = random.uniform(95 * 1e-3, 105 * 1e-3)
        return self._env_post_step()

    def _env_post_step(self):

        # Setup default image rendering

        global r_
        cam = btb.Camera()
        off = btb.OffScreenRenderer(camera=cam, mode='rgb')
        # off.set_render_style(shading='RENDERED', overlays=False)
        image = off.render()
        self.state = np.average(image[:, :, 0:3])
        # global r_
        # result_image = self.render_(save=True, engine='EEVEE')
        # # result_image.show("image")
        # result_image = np.array(result_image)
        # self.state = np.average(result_image)

        # Check if shower is done

        if 24 >self.state > 31:
            done = True     # Env resets
        else:
            done = False    # Env Continues
        # done = bool(24 > abs(self.state) > 31)  # or self.episode_length <= 0)
        # done = False
        if not done:
            r_ = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            r_ = 1.0

        print("post_step-----",self.state, image[:, :, 0:3].shape, done, r_)
        return dict(obs=self.state, reward=r_, done=done)

    def _action(self, action):
        if action == 0:
            delta = self.normalize_(abs(self.state - self.avg_pixel))
        else:
            delta = -self.normalize_(abs(self.state - self.avg_pixel))

        self.light.data.energy += delta

    def normalize_(self, state_v):
        state_v = ((0.85 * state_v) / 5.0)
        return state_v

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
