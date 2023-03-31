from pathlib import Path
from blendtorch import btt
import random
import numpy as np
from gym.utils import seeding
import sys
import os
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/spacer_gym/envs/')


class SpacerEnv(btt.env.OpenAIRemoteEnv):
    def __init__(self, address, render_every=1, real_time=False):
        super().__init__(version="0.0.1")
        self.np_random = None
        self.launch(
            scene=Path(__file__).parent / "spacer1_normal_22.6_exp_no_mesh_new_materail2.blend",
            script=Path(__file__).parent / "spacer.blend.py",
            real_time=real_time,
            render_every=render_every,
            address=address,
        )
        self.seed()

    def seed(self, seed=1):
        np.random.seed(seed)
        random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

