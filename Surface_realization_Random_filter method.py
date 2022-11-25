import slippy.surface as S
import numpy as np
import pandas as pd

np.random.seed(0)


def generate_random_filter_surface(filepath: str, length: float,  reduce_resolution_by_factor=1, grid_spacing=3.697,
                                   no_of_grid_points=100):

    surf_height = pd.read_csv(filepath, delimiter=',')
    surf_height.fillna(surf_height.mean(), inplace=True)  # Filling few empty cells with mean
    slippy_s = S.assurface(surf_height)  # Creating Slippy surface object
    slippy_s.grid_spacing = grid_spacing  # Data from original height1.csv -> "XY calibration"
    slippy_s.get_psd()  # Calculating PSD
    slippy_s.get_acf()  # Calculating acf

    # Generating Random spacer surface
    random_surf = S.RandomFilterSurface(target_acf=slippy_s.acf, grid_spacing=slippy_s.grid_spacing)
    random_surf.linear_transform(filter_shape=(35, 35), gtol=1e-5, symmetric=True, method='BFGS', max_it=1000)

    no_of_grid_points = int(np.round(length / grid_spacing))
    my_realisation = random_surf.discretise([no_of_grid_points, no_of_grid_points], periodic=True, create_new=True)

    if reduce_resolution_by_factor != 1:
        my_realisation.resample(random_surf.grid_spacing *
                                reduce_resolution_by_factor)  # Scaling to increase grid spacing

    print("Grid_Spacing: ", my_realisation.grid_spacing)
    print("No_of_Ctrl_points: ", my_realisation.profile.shape[0])

    return my_realisation


def main():

    # A spacer ring separates the magnetic area inside of a hard disk. This research mainly
    # focuses on the titanium alloy material for defect detection. Its appearance size is 32.6mm in Outer diameter 25mm
    # in Inner diameter and 4 mm in surface width and 1.7 mm in thickness.

    path = '/home/mohanty/PycharmProjects/Digital-twin-for-hard-disk-spacer-ring-defect-detection' \
           '/Spacer_Inspection/ｓ１/ｈｅｉｇｈｔ１_corrected.csv'
    grid_spacing = 3.697
    spacer_outer_dia = (32.6 + 1.4) * 1000  # mm to micrometer *1000, adding buffer of 1.4 mm
    reduce_resolution_by_factor = 10

    my_realisation = generate_random_filter_surface(path, spacer_outer_dia, reduce_resolution_by_factor, grid_spacing)
    # Method to add grid spacing information
    realisation = pd.DataFrame(my_realisation.profile)
    realisation.loc[-1] = 0
    realisation.index = realisation.index + 1  # shifting index
    realisation.sort_index(inplace=True)
    realisation[0][0] = my_realisation.grid_spacing

    np.savetxt('points_X{multi}_dx{dx}.csv'.format(multi=reduce_resolution_by_factor,
                                                   dx=np.round(my_realisation.grid_spacing, 3)),
               np.array(realisation), delimiter=',')


if __name__ == "__main__":
    main()
