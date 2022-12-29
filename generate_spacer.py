# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

import sys
import os
sys.path.append(os.getcwd())
from spacer import Spacer
# from generate_spacer_in_blender import get_vertices
import numpy as np
from blender import Blender


def main(address):
    my_realisation = np.genfromtxt(address, delimiter=',') * 1e-6
    grid_spacing = my_realisation[0][0]
    surface = np.array(my_realisation[1:, :])

    spacer = Spacer(surface, grid_spacing)
    # spacer.get_point_co()
    spacer.generate_defect(13, 16, 0.5, 70, 40, 1)
    blend = Blender(np.array(spacer.point_coo))
    blend.set_scene_linear_unit('METERS')
    blend.get_vertices()

    csv_file = address.split('/')[-1].split('.')
    name = csv_file[0] + '.' + csv_file[1]

    blend.generate_polygon(name)
    objects = blend.get_objs()
    spacer_surf = objects[name]
    cylinder = objects['Cylinder']
    blend.bool_intersect(spacer_surf, cylinder)
    blend.update_scene()


if __name__ == "__main__":
    path = 'topology/points_X5_dx18.485.csv'
    main(path)