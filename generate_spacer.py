from spacer import Spacer
# from generate_spacer_in_blender import get_vertices
import numpy as np


def main(address):
    my_realisation = np.genfromtxt(address, delimiter=',') * 1e-6
    grid_spacing = my_realisation[0][0]
    surface = np.array(my_realisation[1:, :])

    spacer = Spacer(surface, grid_spacing)
    # spacer.get_point_co()
    spacer.generate_defect(13, 16, 0.5, 70, 40, 1)
    print(spacer.point_coo)


if __name__ == "__main__":
    path = 'topology/points_X5_dx18.485.csv'
    main(path)