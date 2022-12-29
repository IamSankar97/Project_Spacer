from operator import itemgetter
import bpy


class Blender:
    def __init__(self, point_coo):
        self.mesh_name = None
        self.point_coo = point_coo
        self.vertices = None
        self.unit = None
        self.objs = bpy.data.objects

    def set_scene_linear_unit(self, desired: str):
        self.unit = desired
        bpy.context.scene.unit_settings.length_unit = self.unit

    def get_vertices(self):
        #   Read and sort the vertices coordinates (sort by x and y)
        self.vertices = sorted([(float(r[0]), float(r[1]), float(r[2])) for r in self.point_coo],
                               key=itemgetter(0, 1))

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
        polygons = [(i, i - 1, i - 1 + xSize, i + xSize) for i in range(1, len(self.vertices) - xSize) if
                    i % xSize != 0]

        mesh = bpy.data.meshes.new(mesh_name)  # Create the mesh (inner data)
        obj = bpy.data.objects.new(mesh_name, mesh)  # Create an object
        obj.data.from_pydata(self.vertices, [], polygons)  # Associate vertices and polygons

        if smoothing:
            for p in obj.data.polygons:  # Set smooth shading (if needed)
                p.use_smooth = True

        bpy.context.scene.collection.objects.link(obj)  # Link the object to the scene

    def bool_intersect(self, obj1, obj2):
        bool_one = obj1.modifiers.new(type="BOOLEAN", name="bool 1")
        bool_one.object = obj2
        bool_one.operation = 'INTERSECT'
        obj2.hide_set(True)

    def update_scene(self):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)