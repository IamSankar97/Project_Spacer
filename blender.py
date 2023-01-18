from operator import itemgetter
import bpy
import bmesh
import numpy as np
import sys
sys.path.append('/home/mohanty/.local/lib/python3.10/site-packages/')
from blendtorch import btb
from PIL import Image


class Blender:
    def __init__(self, point_coo: np.ndarray):
        self.mat = None
        self.mesh_name = None
        self.vertices = list(point_coo)
        self.unit = None
        self.objs = None

    def get_objs(self):
        self.objs = bpy.data.objects
        return self.objs

    def set_scene_linear_unit(self, desired: str = 'METERS'):
        self.unit = desired
        bpy.context.scene.unit_settings.length_unit = self.unit

    def update_scene(self):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def generate_polygon(self, mesh_name: str = 'my_mesh', smoothing: bool = 0):
        """
        :param vertices: sorted list of vertices
        :param mesh_name: name of the generate mesh(optional)
        :param smoothing: to smooth the mesh
        :return: generates polygon mesh in blender
        """
        self.mesh_name = mesh_name

        #   ********* Assuming we have a rectangular grid *************
        xSize = next(i for i in range(len(self.vertices))
                     if self.vertices[i][0] != self.vertices[i + 1][0]) + 1  # Find the first change in X
        ySize = len(self.vertices) // xSize

        #   Generate the polygons (four vertices linked in a face)

        polygons = []
        for i in range(1, len(self.vertices) - xSize):
            if i % xSize != 0 and self.vertices[i][2] != 1 and \
                    self.vertices[i - 1][2] != 1 and self.vertices[i - 1 + xSize][2] != 1 and \
                    self.vertices[i + xSize][2] != 1:
                polygons.append((i, i - 1, i - 1 + xSize, i + xSize))

        mesh = bpy.data.meshes.new(mesh_name)  # Create the mesh (inner data)
        obj = bpy.data.objects.new(mesh_name, mesh)  # Create an object
        obj.data.from_pydata(self.vertices, [], polygons)  # Associate vertices and polygons

        if smoothing:
            for p in obj.data.polygons:  # Set smooth shading (if needed)
                p.use_smooth = True

        bpy.context.scene.collection.objects.link(obj)  # Link the object to the scene
        self.update_scene()

    def bool_intersect(self, obj1, obj2):
        bool_one = obj1.modifiers.new(type="BOOLEAN", name="bool 1")
        bool_one.object = obj2
        bool_one.operation = 'INTERSECT'
        obj2.hide_set(True)
        self.update_scene()

    def update_mesh_back_ground(self, new_realisation: np.ndarray, existing_mesh):
        """

        :param new_realisation: a numpy array with new mesh deta
        :param existing_mesh: existing mesh where the new_realisation data to be updated
        :return: updates mesh in blender
        """
        # Get a BMesh representation
        bm = bmesh.new()  # create an empty BMesh
        bm.from_mesh(existing_mesh.data)  # fill it in from a Mesh
        self.vertices = list(new_realisation)
        for vertice_old, vertice_new in zip(bm.verts, self.vertices):
            vertice_old.co.z = vertice_new[2]  # if vertice_old.co.x == vertice_new[0] and
            # vertice_old.co.y == vertice_new[1]:

        bm.to_mesh(existing_mesh.data)  # Finish up, write the bmesh back to the mesh
        bm.free()
        self.update_scene()  # Updates the mesh in scene by refreshing the screen

    def assign_material(self, obj: bpy.data.objects, mat_name: str = 'Material'):
        self.mat = bpy.data.materials.new(name=mat_name)
        obj.data.materials.append(self.mat)
        self.mat.use_nodes = True

    def update_mat(self, mat_property: dict):
        self.mat.use_nodes = True
        mat_nodes = self.mat.node_tree.nodes
        for key, value in mat_property.items():
            mat_nodes['Principled BSDF'].inputs[key].default_value = value

    def render(self, file_path: str, resolution: tuple = (2448, 2048), save: bool = True,
               engine: str = 'CYCLES'):
        """

        Parameters
        ----------
        file_path: File where render image to be saved
        resolution: Pixel resolution
        save: To save the image
        engine: rendering engine CYCLES

        Returns
        -------

        """

        if engine == "CYCLES":
            # Set the rendering engine to Cycles
            bpy.context.scene.render.engine = engine
            bpy.context.scene.render.filepath = file_path

            bpy.ops.render.render(write_still=save)
            image = Image.open(file_path)
            return image

        elif engine == 'EEVEE':
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'

            cam = btb.Camera()
            off = btb.OffScreenRenderer(camera=cam, mode='rgb')
            off.set_render_style(shading='RENDERED', overlays=False)
            image = np.array(off.render())

            image_ = Image.fromarray(image.astype(np.uint8))

            # Save the image to a file
            image_.save(file_path, format='PNG')

            return image_
