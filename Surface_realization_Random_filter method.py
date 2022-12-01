import slippy.surface as S
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)


def fit_random_filter_surface(slippy_s: S, grid_spacing: float):
    r""" Fits filter to generat random surface


        Parameters
        ----------
        slippy_s: slippy.surface
            A slippy surface; to be used to caculate the trage autocorrelation function of the surface independent of grid size
        grid_spacing: float, optional (None)
            The distance between surface points, must be set before the filter coefficients can be found
     """

    # grid_spacing

    slippy_s.get_acf()  # Calculating acf independent of grid spacing

    # Generating Random spacer surface
    lin_trans_surface = S.RandomFilterSurface(target_acf=slippy_s.acf, grid_spacing=grid_spacing)
    lin_trans_surface.linear_transform(filter_shape=(35, 35), gtol=1e-5, symmetric=True, method='BFGS', max_it=1000)

    return lin_trans_surface


def generate_surface(lin_trans_surface, length: float, grid_spacing: float, grid_details: bool = False):
    no_of_grid_points = int(np.round(length / grid_spacing))
    my_realisation = lin_trans_surface.discretise([no_of_grid_points, no_of_grid_points], periodic=True,
                                                  create_new=True)

    if grid_details:
        print("Grid_Spacing: ", my_realisation.grid_spacing)
        print("No_of_grid_points: ", my_realisation.profile.shape[0])

    return my_realisation


def compare_acf(lin_trans_surface, my_realisation, address: str):
    target = lin_trans_surface.target_acf_array
    my_realisation.get_acf()
    actual = np.array(my_realisation.acf)
    n, m = actual.shape
    tn, tm = target.shape
    actual_comparible = actual[n // 2:n // 2 + tn, m // 2:m // 2 + tm]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(actual_comparible);
    axes[0].set_title('Actual')
    axes[1].imshow(target);
    axes[1].set_title('Target')
    axes[2].plot(actual_comparible[0, :], label='actual')
    axes[2].plot(target[0, :], label='target')
    axes[2].legend();
    axes[2].set_title('Comparison along y direction')
    axes[3].plot(actual_comparible[:, 0], label='actual')
    axes[3].plot(target[:, 0], label='target')
    axes[3].legend();
    axes[3].set_title('Comparison along x direction')
    plt.savefig(address)


def get_properties(slippy_surface):
    rou2 = slippy_surface.roughness(['sa', 'sq', 'ssk', 'sku'], no_flattening=True)
    nam = ['Mean abs height: ', 'Root mean square height: ', 'Skew: ', 'Kurtosis: ']
    for n, r in zip(nam, rou2):
        print(n, r)


def main(address: str, reduce_resolution_by: int = 1, compare: bool = False):
    # A spacer ring separates the magnetic area inside of a hard disk. This research mainly
    # focuses on the titanium alloy material for defect detection. Its appearance size is 32.6mm in Outer diameter 25mm
    # in Inner diameter and 4 mm in surface width and 1.7 mm in thickness.

    path = '/home/mohanty/PycharmProjects/Digital-twin-for-hard-disk-spacer-ring-defect-detection' \
           '/Spacer_Inspection/ｓ１/ｈｅｉｇｈｔ１_corrected.csv'
    grid_spacing = 3.697  # Original grid_spacing
    spacer_outer_dia = (32.6 + grid_spacing * 4) * 1000  # mm to micrometer *1000, adding buffer of 1.4 mm

    surf_height = pd.read_csv(path, delimiter=',')
    surf_height.fillna(surf_height.mean(), inplace=True)  # Filling few empty cells with mean
    slippy_s = S.assurface(surf_height)  # Creating Slippy surface object
    slippy_s.grid_spacing = grid_spacing  # Data from original height1.csv -> "XY calibration"

    desired_grid_spacing = grid_spacing * reduce_resolution_by
    lin_trans_surface_original = fit_random_filter_surface(slippy_s, grid_spacing)
    lin_trans_surface_realised = fit_random_filter_surface(slippy_s, desired_grid_spacing)

    my_realisation = generate_surface(lin_trans_surface_realised, spacer_outer_dia, desired_grid_spacing, True)

    if compare:
        compare_acf(lin_trans_surface_original, my_realisation, 'my_plot_X1_dx3.69to{}.png'
                    .format(np.round(desired_grid_spacing), 2))

        print("\nProperties: Target\n")
        get_properties(slippy_s)

        print("\nProperties: actual\n")
        get_properties(my_realisation)

    # Method to add grid spacing information
    realisation = pd.DataFrame(my_realisation.profile)
    realisation.loc[-1] = 0
    realisation.index = realisation.index + 1  # shifting index
    realisation.sort_index(inplace=True)
    realisation[0][0] = my_realisation.grid_spacing

    np.savetxt(address+"points_X{multi}_dx{dx}.csv".format(multi=reduce_resolution_by,
                                                   dx=np.round(my_realisation.grid_spacing, 3)),
               np.array(realisation), delimiter=',')


if __name__ == "__main__":
    main("/home/mohanty/PycharmProjects/Blender_Debug/topology/", 2)
