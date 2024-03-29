# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

'''
    The code is to be used for generating spacer in blender
'''


import sys
import os
import random
import pickle
sys.path.append(os.getcwd())
sys.path.append('/home/mohanty/PycharmProjects/Scribed_PPO_1light_2_22.6/')
from spacer import Spacer

import numpy as np
from blender import Blender
import time

np.random.seed(110)


def get_sample_surface(addr):
    filename = random.choice(os.listdir(addr))
    if filename.endswith('.pkl'):
        with open(os.path.join(addr, filename), 'rb') as f:
            my_realisation = pickle.load(f) * 1e-6
            grid_spacing, sample_surface = my_realisation[0][0], np.array(my_realisation[1:, :])

            spacer = Spacer(sample_surface, grid_spacing)
            ro, r1, theta0, defect_length, defect_type = \
                np.round(np.random.uniform(12.5, 13.2), 2), \
                    np.round(np.random.uniform(13.8, 16), 2), \
                    np.random.randint(1, 85), \
                    np.round(np.random.uniform(2, 10), 2), \
                    random.choice([0, 1])
            spacer.randomize_defect(ro, r1, theta0, 70, 40, defect_length, 0)
            f.close()
            yield spacer, ro, r1, theta0, defect_length, defect_type


def main(addr_topology: str, addr_img_save: str):
    count = 0
    global spacer_surf, blend, start
    start = time.time()
    for spacer, ro, r1, theta0, defect_length, defect_type in get_sample_surface(addr_topology):
        count += 1
        if count == 1:
            point_coordinates = spacer.point_coo[['X', 'Y', 'Z']]
            blend = Blender(np.array(point_coordinates))
            objects = blend.get_objs()
            blend.set_scene_linear_unit('METERS')
            blend.generate_polygon('spacer_ring')

            time_poly = time.time() - start
            spacer_surf = objects['spacer_ring']
            blend.assign_material(spacer_surf)

        else:
            blend.update_mesh_back_ground(np.array(spacer.point_coo[['X', 'Y', 'Z']]), spacer_surf)
            time_poly = time.time() - start

        start = time.time()
        blend.update_mat({'Roughness': 0, 'Metallic': 1})
        result_image = blend.render(file_path=addr_img_save, save=True, engine='CYCLES')
        time_render = time.time() - start
        print("time elapsed_ {}_st generation for poly_mesh: {} and for rendering: {}:"
              .format(count, time_poly, time_render))
        result_image.save('spacer_render/image_{cout}_{r0}r0_{r1}r1_{th}th0_{ln}ln_{ty}ty.png'
                          .format(cout=count, r0=ro, r1=r1, th=theta0, ln=defect_length, ty=defect_type))
        start = time.time()
        break


if __name__ == "__main__":
    path1 = '/home/mohanty/PycharmProjects/Data/pkl_6'
    path2 = 'blend_torch_img/image.png'
    main(addr_topology=path1, addr_img_save=path2)
