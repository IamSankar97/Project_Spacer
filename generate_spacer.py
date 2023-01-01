# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

import sys
import os
sys.path.append(os.getcwd())
from spacer import Spacer
import numpy as np
from blender import Blender
import pickle


def get_sample_surface(folder):
    while True:
        for filename in os.listdir(folder):
            if filename.endswith('.pkl'):
                with open(os.path.join(folder, filename), 'rb') as f:
                    my_realisation = pickle.load(f) * 1e-6
                    grid_spacing = my_realisation[0][0]
                    surface = np.array(my_realisation[1:, :])
                    yield surface, grid_spacing


def main(address):
    count = 0
    for sample_surface, grid_spacing in get_sample_surface(address):
        count += 1
        spacer = Spacer(sample_surface, grid_spacing)
        spacer.get_point_co()
        spacer.generate_defect(13, 16, 0.5, 70, 40, 1)
        if count == 1:
            blend = Blender(np.array(spacer.point_coo))
            blend.set_scene_linear_unit('METERS')
            blend.get_vertices()

            blend.generate_polygon('my_mesh')
            objects = blend.get_objs()
            spacer_surf = objects['my_mesh']
            cylinder = objects['Cylinder']
            blend.bool_intersect(spacer_surf, cylinder)
            blend.update_scene()
            break
        else:
            break


if __name__ == "__main__":
    path = 'topology/pkl_50/'
    main(path)
