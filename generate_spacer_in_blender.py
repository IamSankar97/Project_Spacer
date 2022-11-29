# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)
import bpy
from mathutils import Vector
import numpy as np


def select_obj(objName, additive=False):
    if not additive:
        bpy.ops.object.select_all(action='DESELECT')
    bpy.context.scene.objects[objName].select_set(True)


def get_ctrl_points(height_matrix: np.ndarray, grid_spacing: float, center=True):
    # Centring
    if center:
        end_right = int(height_matrix.shape[0] / 2)
        end_left = -int(height_matrix.shape[0] / 2)
    else:
        end_right = height_matrix.shape[0]
        end_left = 0

    ctrl_points = []

    for i in range(end_left, end_right):
        one_spline_ctrl_points = []
        for j in range(end_left, end_right):
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


def bool_intersect(obj1, obj2):
    bool_one = obj1.modifiers.new(type="BOOLEAN", name="bool 1")
    bool_one.object = obj2
    bool_one.operation = 'INTERSECT'
    obj2.hide_set(True)


def convert_mesh(obj: str):
    select_obj(obj)
    bpy.ops.object.convert(target='MESH')
    bpy.ops.object.select_all(action='DESELECT')


def main():
    my_realisation = np.genfromtxt(
        "control_points/points_X50_dx184.85.csv",
        delimiter=',')
    dx = my_realisation[0][0]
    my_realisation = np.array(my_realisation[1:, :])

    ctrl_points = get_ctrl_points(my_realisation, dx)

    generate_nurbs_surface(ctrl_points)

    spacer_surf = bpy.data.objects['NURBS_OBJ']
    cylinder = bpy.data.objects['Cylinder']

    convert_mesh('NURBS_OBJ')

    bool_intersect(spacer_surf, cylinder)


if __name__ == "__main__":
    main()
