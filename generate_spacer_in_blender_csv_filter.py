import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)

import bmesh
import bpy
from operator import itemgetter
import time
import os as S # Slippy to be replaced here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)


def fit_random_filter_surface(slippy_s: S, desired_grid_spacing: float):

    """

    :param slippy_s: slippy.surface
            A slippy surface; to be used to caculate the trage autocorrelation function of the surface independent of grid size
    :param desired_grid_spacing: The distance between surface points, must be set before the filter coefficients can be found
    :return: Linear trans surface capable of generating surface multiple times
    """

    # grid_spacing

    slippy_s.get_acf()  # Calculating acf independent of grid spacing

    # Generating Random spacer surface
    lin_trans_surface = S.RandomFilterSurface(target_acf=slippy_s.acf, grid_spacing=desired_grid_spacing)
    lin_trans_surface.linear_transform(filter_shape=(35, 35), gtol=1e-5, symmetric=True, method='BFGS', max_it=1000)

    return lin_trans_surface


def generate_surface(lin_trans_surface, length: float, grid_details: bool = False):
    """

    :param lin_trans_surface: Slippy surface with the topology profile and grid spacing in microns
    :param length: Dimension of desired topology in microns
    :param grid_details: To print grid spacing and No of grid points in an axis
    :return:
    """
    # Below step is to come up with no of grid points required for the desired length
    no_of_grid_points = int(np.round(length / lin_trans_surface.grid_spacing))
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


def random_surf(address: str, grid_spacing: float, reduce_resolution_by: int = 1, compare: bool = False,
                save_topology: bool = False, save_address: str = './'):
    """

    :param address: File path to the topology generated from microscope or stylus instrument, unit to be in microns
    :param grid_spacing: resolution of the topology
    :param reduce_resolution_by: factor to relax the resolution
    :param compare: compare original topology with generated topology
    :param save_topology: save the realised topology as csv file
    :param save_address: Address to save the csv file
    :return: A slippy surface object with generated topology and its grid spacing in microns
    """
    # A spacer ring separates the magnetic area inside of a hard disk. This research mainly
    # focuses on the titanium alloy material for defect detection. Its appearance size is 32.6mm in Outer diameter 25mm
    # in Inner diameter and 4 mm in surface width and 1.7 mm in thickness.

    surf_height = pd.read_csv(address, delimiter=',')
    surf_height.fillna(surf_height.mean(), inplace=True)  # Filling few empty cells with mean
    slippy_s = S.assurface(surf_height)  # Creating Slippy surface object
    slippy_s.grid_spacing = grid_spacing  # Data from original height1.csv -> "XY calibration"

    desired_grid_spacing = grid_spacing * reduce_resolution_by
    lin_trans_surface = fit_random_filter_surface(slippy_s, desired_grid_spacing)

    # if compare:
    #     compare_acf(lin_trans_surface_original, my_realisation, 'my_plot_X1_dx3.69to{}.png'
    #                 .format(np.round(desired_grid_spacing), 2))
    #
    #     print("\nProperties: Target\n")
    #     get_properties(slippy_s)
    #
    #     print("\nProperties: actual\n")
    #     get_properties(my_realisation)
    #
    # if save_topology:
    #     # Method to add grid spacing information
    #     realisation = pd.DataFrame(my_realisation.profile)
    #     realisation.loc[-1] = 0
    #     realisation.index = realisation.index + 1  # shifting index
    #     realisation.sort_index(inplace=True)
    #     realisation[0][0] = my_realisation.grid_spacing
    #
    #     np.savetxt(save_address+"points_X{multi}_dx{dx}.csv".format(multi=reduce_resolution_by,
    #                                                    dx=np.round(my_realisation.grid_spacing, 3)),
    #                np.array(realisation), delimiter=',')

    return lin_trans_surface


def select_obj(objName, additive=False):
    if not additive:
        bpy.ops.object.select_all(action='DESELECT')
    bpy.context.scene.objects[objName].select_set(True)


def bool_intersect(obj1, obj2):
    bool_one = obj1.modifiers.new(type="BOOLEAN", name="bool 1")
    bool_one.object = obj2
    bool_one.operation = 'INTERSECT'
    obj2.hide_set(True)


def generate_topology(list_vertices):
    X_unique = np.sort(np.unique(list_vertices[:, 0], axis=None))
    Y_unique = np.sort(np.unique(list_vertices[:, 1], axis=None))

    # Assuming Equally placed grid
    dx = X_unique[3:4] - X_unique[2:3]

    # Adding 1 to include add X_unique and Y_unique as grid points in later stage
    topology = np.zeros([len(Y_unique) + 1, len(X_unique) + 1])
    topology[:1, 1:] = np.unique(X_unique.T)
    topology[1:, 0] = Y_unique.T

    for i in range(len(list_vertices[:, 0])):
        x, y, z = np.where(X_unique == list_vertices[i:i + 1, 0])[0][0], \
                  np.where(Y_unique == list_vertices[i:i + 1, 1])[0][0], \
                  list_vertices[i:i + 1, 2]
        topology[y + 1:y + 2, x + 1:x + 2] = z

    return topology, dx


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def radial_2d(surface_height_topology, PixelWidth):
    z = surface_height_topology

    [m, n] = z.shape

    # converting to same pixelwidth
    minimum_width = np.minimum(m, n)

    if minimum_width == m:
        z = z[:, :m]
        n = m
    else:
        z = z[:n, :]
        m = n

    # Make dimension even
    if np.mod(n, 2):
        z = z[:-1, :]
        n = n - 1

    if np.mod(m, 2):
        z = z[:, :-1]
        m = m - 1

    Ln = n * PixelWidth  # width of image
    Lm = m * PixelWidth  # length of image

    a = PixelWidth  # Lattice spacing in meter

    # =========================================================================
    win1 = (1 - ((np.array(range(0, n)) - ((n - 1) / 2)) / ((n + 1) / 2)) ** 2).reshape((n, 1))
    win2 = (1 - ((np.array(range(0, m)) - ((m - 1) / 2)) / ((m + 1) / 2)) ** 2)
    win = np.multiply(win1, win2)

    recInt = np.trapz(np.trapz((np.square(np.ones((n, 1)) * np.ones(m))), np.array(range(0, m)), 2),
                      np.array(range(0, n)), 1)  # integral of squared rectangular window
    winInt = np.trapz(np.trapz((win ** 2), np.array(range(0, m)), 2),
                      np.array(range(0, n)), 1)  # integral of square of selected window function
    U = (winInt / recInt)

    z_win = z * win

    # =========================================================================
    # Calculate 2D PSD
    Hm = np.fft.fftshift(np.fft.fft2(z_win, (m, n)))
    Cq = np.squeeze((1 / U) * (a ** 2 / ((n * m) * ((2 * np.pi) ** 2)) * ((abs((Hm))) ** 2)))
    Cq[int(n / 2):int(n / 2 + 1), int(m / 2):int(m / 2 + 1)] = 0

    # =========================================================================
    # corresponding wavevectors to Cq values after fftshift has been applied

    qx_1 = np.zeros(m)
    qx_1 = [((2 * np.pi / m) * k) for k in range(len(qx_1))]
    qx_2 = np.fft.fftshift(qx_1)
    qx_3 = np.unwrap(qx_2 - 2 * np.pi)
    qx = qx_3 / a

    qy_1 = np.zeros(n)
    qy_1 = [((2 * np.pi / n) * k) for k in range(len(qy_1))]
    qy_2 = np.fft.fftshift(qy_1)
    qy_3 = np.unwrap(qy_2 - 2 * np.pi)
    qy = qy_3 / a

    # =========================================================================
    # Radial Averaging
    qxx, qyy = np.meshgrid(qx, qy)
    rho, pi = cart2pol(qxx, qyy)
    rho = np.floor(rho)
    J = 500  # resolution in q space (increase if you want)
    qrmin = np.log10(np.sqrt((((2 * np.pi) / Lm) ** 2 + ((2 * np.pi) / Ln) ** 2)))
    qrmax = np.log10(np.sqrt(qx[-1:] ** 2 + qy[-1:] ** 2))  # Nyquistnpi
    q = np.floor(10 ** np.linspace(qrmin, qrmax, J))

    # =========================================================================
    # Averaging Cq values
    C_AVE = np.zeros(len(q))
    ind = []
    Cq_flat = Cq.flatten()

    for j in range(len(q) - 1):
        mn = np.flatnonzero((rho > q[j]) & (rho <= q[j + 1]))
        ind.append(mn)
        C_AVE[j] = np.nanmean(Cq_flat.take(indices=ind[j]))

    ind = ~np.isnan(C_AVE)
    C = C_AVE.take(indices=np.flatnonzero(ind))
    q = q.take(indices=np.flatnonzero(ind))

    return {'C': C, "q": q, "Cq": Cq, "qx": qx, "qy": qy, "z_win": z_win}


def get_vertices(Xmesh: np.ndarray, Ymesh: np.ndarray, surface: np.ndarray):

    point_coords = np.array([np.array([a1, b1, c1]) for a, b, c in zip(Xmesh, Ymesh, surface) for a1, b1, c1 in
                             zip(a, b, c)])

    #   Read and sort the vertices coordinates (sort by x and y)
    vertices = sorted([(float(r[0]), float(r[1]), float(r[2])) for r in point_coords], key=itemgetter(0, 1))
    return vertices


def main(address: str, reduce_resolution: int= 1, smoothing: bool = False, perform_boolean: bool = False):
    # spacer_outer_dia = (32.6 + grid_spacing * 2) * 1000  # mm to micrometer *1000, adding buffer of 1.4 mm
    #
    # lin_trans_surface_realised = random_surf(address, grid_spacing=grid_spacing,
    #                                              reduce_resolution_by=reduce_resolution)
    #

    #
    # my_realisation = generate_surface(lin_trans_surface_realised, spacer_outer_dia, True)

    my_realisation = np.genfromtxt(address, delimiter=',') *1e-6
    grid_spacing = my_realisation[0][0]
    my_realisation = np.array(my_realisation[1:, :])*500

    X = np.array([i * grid_spacing for i in range(my_realisation.shape[0])])
    Y = X
    x_mesh, y_mesh = np.meshgrid(X, Y)

    vertices = get_vertices(x_mesh, y_mesh, my_realisation)

    #   ********* Assuming we have a rectangular grid *************
    xSize = next(
        i for i in range(len(vertices)) if vertices[i][0] != vertices[i + 1][0]) + 1  # Find the first change in X
    ySize = len(vertices) // xSize

    #   Generate the polygons (four vertices linked in a face)
    polygons = [(i, i - 1, i - 1 + xSize, i + xSize) for i in range(1, len(vertices) - xSize) if i % xSize != 0]

    csv_file = address.split('/')[-1].split('.')
    name = csv_file[0] + '.' + csv_file[1]
    mesh = bpy.data.meshes.new(name)  # Create the mesh (inner data)
    obj = bpy.data.objects.new(name, mesh)  # Create an object

    obj.data.from_pydata(vertices, [], polygons)  # Associate vertices and polygons

    if smoothing:
        for p in obj.data.polygons:  # Set smooth shading (if needed)
            p.use_smooth = True

    bpy.context.scene.collection.objects.link(obj)  # Link the object to the scene

    spacer_surf = bpy.data.objects[name]
    cylinder = bpy.data.objects['Cylinder']
    cylinder.location.y += spacer_surf.dimensions.y / 2
    cylinder.location.x += spacer_surf.dimensions.x / 2

    bool_intersect(spacer_surf, cylinder)

    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    # Get a BMesh representation

    for i in range(0, 1):
        bm = bmesh.new()  # create an empty BMesh
        bm.from_mesh(spacer_surf.data)  # fill it in from a Mesh

        start_time = time.time()

        my_realisation = np.genfromtxt('topology/points_X100_dx369.7_{}.csv'.format(i), delimiter=',')*1e-6
        my_realisation = np.array(my_realisation[1:, :])
        vertices_new = get_vertices(x_mesh, y_mesh, my_realisation)
        count = 0

        for vertice_old, vertice_new in zip(bm.verts, vertices_new):
            count += 1

            vertice_old.co.z = vertice_new[2]   # if vertice_old.co.x == vertice_new[0] and
                                                # vertice_old.co.y == vertice_new[1]:

            # if count == 5000:
            #     break

        bm.to_mesh(spacer_surf.data)    # Finish up, write the bmesh back to the mesh
        bm.free()
        # bool_intersect(spacer_surf, cylinder)

    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    path = '/home/mohanty/PycharmProjects/Blender_Debug/topology/points_X100_dx369.7.csv'
    main(path, perform_boolean=True)
