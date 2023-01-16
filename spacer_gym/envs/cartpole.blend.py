import bpy
import numpy as np
from gym import logger, spaces
import random
from blendtorch import btb


# Use below only during debugging
# import pydevd_pycharm
#
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


class SpacerEnv(btb.env.BaseEnv):
    def __init__(self, agent):
        super().__init__(agent)
        self.slab = bpy.data.objects["Steel_slab"]
        self.camera = bpy.data.objects["Camera"]
        self.light = bpy.data.objects["Light"]
        # Note, ensure that physics run at same speed.
        self.fps = bpy.context.scene.render.fps
        self.avg_pixel = 180.0
        self.episodes = 110
        self.episode_length = self.episodes

        self.reward_threshold = 1000
        self.np_random = None
        self.state = 180
        self.steps_beyond_done = None

    def _env_prepare_step(self, action: np.ndarray):
        self._action(action)
        self.episode_length -= 1

    def _env_reset(self):
        self.steps_beyond_done = None
        self.episode_length = self.episodes
        # 15.72114 = 180.0002
        # 15.1 = 175.24
        # 16.45 = 185
        #below limit is set 15.72114+0.005 15.72114-0.005
        # Enviromment should be reset to the mean of avg pixel requirement that is 180
        # Above is how Cartpole problem is designed
        # self.light.data.energy = random.uniform(15.72614,15.71614)
        self.light.data.energy = random.uniform(15.65, 15.75)
        # self.light.data.energy = 16.5
        return self._env_post_step()

    def _env_post_step(self):
        # Setup default image rendering
        cam = btb.Camera()
        off = btb.OffScreenRenderer(camera=cam, mode='rgb')
        off.set_render_style(shading='RENDERED', overlays=False)
        image = off.render()

        self.state = np.average(image)

        # Check if shower is done
        done = bool(abs(self.state) > 185 or abs(self.state) < 175)# or self.episode_length <= 0)
        if not done:
            r_ = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            r_ = 1.0
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     r_ = 0.0

        return dict(obs=np.array([self.state], dtype=np.float32), reward=r_, done=done)

    def _action(self, action):

        diff_ = abs(self.state - self.avg_pixel)

        diff_ = self.normalize_(diff_)

        if action == 0.0:
            delta = diff_
        else:
            delta = -diff_
        self.light.data.energy += delta

    def normalize_(self, state_v):
        state_v = ((0.85*state_v) / 5.0)
        return state_v

    def run(self, frame_range=None, use_animation=True):
        super().run(frame_range, use_animation)


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
    env.attach_default_renderer(every_nth=envargs.render_every)
    env.run(frame_range=(1, 10000), use_animation=True)


main()
