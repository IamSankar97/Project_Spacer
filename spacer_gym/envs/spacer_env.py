from pathlib import Path
from gym import spaces
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
            scene=Path(__file__).parent / "spacer.blend",
            script=Path(__file__).parent / "spacer.blend.py",
            real_time=real_time,
            render_every=render_every,
            address=address,
        )

        self.up_limit = 60
        self.lw_limit = 40
        self.action_space = spaces.Box(0.1, 0.7, shape=(1,))

        self.observation_space = spaces.Box(low=self.lw_limit, high=self.up_limit, shape=(1,),
                                            dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

