import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)
import bpy
from mathutils import Vector
import numpy as np


def select_obj(objName, additive=False):
    if not additive:
        bpy.ops.object.select_all(action='DESELECT')
    bpy.context.scene.objects[objName].select_set(True)


def get_ctrl_points(height_matrix: np.ndarray, grid_spacing: float):
    # Centring
    x_y = int(height_matrix.shape[0] / 2)

    ctrl_points = []

    for i in range(-x_y, x_y):
        one_spline_ctrl_points = []
        for j in range(-x_y, x_y):
            x, y = (i * grid_spacing) * 1e-6, (j * grid_spacing) * 1e-6  # To convert it to Meter from micrometer
            one_spline_ctrl_points.append(Vector(
                (x, y, height_matrix[i][j] * 1e-6, 1))  # To convert it to Meter from micrometer
            )
        ctrl_points.append(one_spline_ctrl_points)

    return ctrl_points


def generate_nurbs_surface(ctrl_points: list):
    scene = bpy.context.scene

    surface_data = bpy.data.curves.new('wook', 'SURFACE')
    surface_data.dimensions = '3D'

    for one_spline in ctrl_points:
        spline = surface_data.splines.new(type='NURBS')
        spline.points.add(len(one_spline) - 1)  # already has a default zero vector
        for p, new_co in zip(spline.points, one_spline):
            p.co = new_co

    surface_object = bpy.data.objects.new('NURBS_OBJ', surface_data)
    scene.collection.objects.link(surface_object)

    splines = surface_object.data.splines
    for s in splines:
        for p in s.points:
            p.select = True

    bpy.context.view_layer.objects.active = surface_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.curve.make_segment()
    bpy.ops.object.editmode_toggle()


def main():
    my_realisation = np.genfromtxt(
        "/home/mohanty/PycharmProjects/Digital-twin-for-hard-disk-spacer-ring-defect-detection/points_X50_dx184.85.csv",
        delimiter=',')
    dx = my_realisation[0][0]
    my_realisation = np.array(my_realisation[1:, :])

    ctrl_points = get_ctrl_points(my_realisation, dx)

    generate_nurbs_surface(ctrl_points)

    select_obj('NURBS_OBJ')
    bpy.ops.object.convert(target='MESH')

    spacer_surf = bpy.data.objects['NURBS_OBJ']
    cylinder = bpy.data.objects['Cylinder']

    bool_one = spacer_surf.modifiers.new(type="BOOLEAN", name="bool 1")
    bool_one.object = cylinder
    bool_one.operation = 'INTERSECT'
    cylinder.hide_set(True)


if __name__ == "__main__":
    main()
