# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

import sys
import os

sys.path.append(os.getcwd())
sys.path.append('/home/mohanty/PycharmProjects/Project_Spacer/blendtorch')
from spacer import Spacer
import numpy as np
from blender import Blender
import pickle
import time


def get_sample_surface(folder):
    while True:
        for filename in os.listdir(folder):
            if filename.endswith('.pkl'):
                with open(os.path.join(folder, filename), 'rb') as f:
                    my_realisation = pickle.load(f) * 1e-6
                    grid_spacing = my_realisation[0][0]
                    surface = np.array(my_realisation[1:, :])
                    yield surface, grid_spacing


def main(addr_topology: str, addr_img_save: str):
    start = time.time()
    count = 0
    for sample_surface, grid_spacing in get_sample_surface(addr_topology):
        count += 1
        spacer = Spacer(sample_surface, grid_spacing)
        spacer.get_point_co()
        ro, r1, theta0, defect_length, defect_type = \
            np.round(np.random.uniform(12.5, 13.2), 2), \
            np.round(np.random.uniform(13.8, 16), 2), \
            np.random.randint(1, 72), \
            np.round(np.random.uniform(2, 10), 2), \
            count%2
        spacer.randomize_defect(ro, r1, theta0, 70, 40, 4, defect_type)
        if count == 1:
            blend = Blender(np.array(spacer.point_coo))
            objects = blend.get_objs()
            blend.set_scene_linear_unit('METERS')
            blend.generate_polygon('my_mesh')
            spacer_surf = objects['my_mesh']
            cylinder = objects['Cylinder']
            blend.bool_intersect(spacer_surf, cylinder)
            blend.assign_material(spacer_surf)
            blend.update_mat({'Roughness': 0, 'Metallic': 1})

            result_image = blend.render(file_path=addr_img_save, save=True, engine='CYCLES')
            end = time.time()
            print("Total time elapsed_ {}_st generation:".format(count), end - start)
            result_image.save('spacer_render/image_{cout}_{r0}r0_{r1}r1_{th}_th0.png'
                              .format(cout=count, r0=ro, r1=r1, th=theta0))

        else:
            start = time.time()
            blend.update_mesh_back_ground(np.array(spacer.point_coo), spacer_surf)
            #   blend.assign_material(spacer_surf)
            blend.update_mat({'Roughness': 0, 'Metallic': 1})
            result_image = blend.render(file_path=addr_img_save, save=True, engine='CYCLES')
            end = time.time()
            print("Total time elapsed_{}_st generation:".format(count), end - start)
            # result_image.show()
            result_image.save('spacer_render/image_{cout}_{r0}r0_{r1}r1_{th}th0.png'
                              .format(cout=count, r0=ro, r1=r1, th=theta0))


if __name__ == "__main__":
    path1 = 'topology/pkl_5/'
    path2 = '/home/mohanty/PycharmProjects/Project_Spacer/blendtorch_image/' \
            '18dx_rendered-code_3_cycles.png'
    main(path1, path2)
