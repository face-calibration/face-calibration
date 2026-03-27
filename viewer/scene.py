import numpy as np
from enum import Enum

from . import camera

class WireframeMode(Enum):
    SHADED = 1
    WIREFRAME = 2
    WIREFRAME_ON_SHADED = 3

class ShadingMode(Enum):
    VERTEX = 1
    TEXTURE = 2
    DEFAULT = 3

class NormalsMode(Enum):
    FLAT = 1
    SMOOTH = 2

class LightingMode(Enum):
    DIRECT = 1
    SH = 2
    NONE = 3

class Scene:
    next_wireframe_mode = {WireframeMode.SHADED: WireframeMode.WIREFRAME,
                           WireframeMode.WIREFRAME: WireframeMode.WIREFRAME_ON_SHADED,
                           WireframeMode.WIREFRAME_ON_SHADED: WireframeMode.SHADED}
    next_shading_mode = {ShadingMode.VERTEX: ShadingMode.TEXTURE,
                         ShadingMode.TEXTURE: ShadingMode.DEFAULT,
                         ShadingMode.DEFAULT: ShadingMode.VERTEX}
    next_normals_mode = {NormalsMode.FLAT: NormalsMode.SMOOTH,
                         NormalsMode.SMOOTH: NormalsMode.FLAT}
    next_lighting_mode = {LightingMode.DIRECT: LightingMode.SH,
                          LightingMode.SH : LightingMode.NONE,
                          LightingMode.NONE: LightingMode.DIRECT}

    def __init__(self):
        self.meshes = []
        self.lines = []
        self.camera = camera.TrackBall()
        self.amb = np.asarray([0.1, 0.1, 0.1]).astype(np.float32)
        self.direct = np.asarray([0.8, 0.8, 0.8]).astype(np.float32)
        self.spec_ang = 5
        self.spec_mag = 0.1
        self.sh_coeffs = np.asarray([[.38, .43, .45], [.29, .36, .41], [.04, .03, .01],
            [-.10, -.10, -.09], [-.06, -.06, -.04], [.01, -.01, -.05],
            [-.09, -.13, -.15], [-.06, -.05, -.04], [.02, .00, -.05]])
        self.wireframe_mode = WireframeMode.SHADED
        self.shading_mode = ShadingMode.TEXTURE
        self.normals_mode = NormalsMode.FLAT
        self.lighting_mode = LightingMode.DIRECT
        self.background_color = np.asarray([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        self.two_sided_lighting = False

    def add_mesh(self, mesh):
        '''Add a RenderObject triangulated mesh to the current list of meshes'''
        self.meshes.append(mesh)

    def add_lines(self, lines):
        '''Add a RenderObjectLines set of lines to the current list of lines'''
        self.lines.append(lines)

    def set_camera(self, camera):
        '''Set a new Camera() object'''
        self.camera = camera

    def set_wireframe_mode(self, wireframe_mode):
        self.wireframe_mode = wireframe_mode

    def toggle_wireframe_mode(self):
        self.wireframe_mode = self.next_wireframe_mode[self.wireframe_mode]

    def set_normals_mode(self, normals_mode):
        self.normals_mode = normals_mode

    def toggle_normals_mode(self):
        self.normals_mode = self.next_normals_mode[self.normals_mode]

    def set_shading_mode(self, shading_mode):
        self.shading_mode = shading_mode

    def toggle_shading_mode(self):
        self.shading_mode = self.next_shading_mode[self.shading_mode]

    def set_lighting_mode(self, lighting_mode):
        self.lighting_mode = lighting_mode

    def toggle_lighting_mode(self):
        self.lighting_mode = self.next_lighting_mode[self.lighting_mode]

    def set_two_sided_lighting(self, two_sided_lighting):
        self.two_sided_lighting = two_sided_lighting

    def toggle_two_sided_lighting(self):
        self.two_sided_lighting = not self.two_sided_lighting

    def set_background_color(self, background_color):
        '''Set background color for glClear()'''
        if len(background_color) == 3:
            background_color.append(1.0)
        self.background_color = np.asarray(background_color).astype(np.float32)

    def set_amb(self, amb):
        '''Set ambient lighting color'''
        self.amb = np.asarray(amb).astype(np.float32)

    def set_direct(self, direct):
        '''Set direct (diffuse) lighting color'''
        self.direct = np.asarray(direct).astype(np.float32)

    def frame_camera(self):
        '''Rotates camera so it is facing the center of the object(s)'''
        all_verts = np.concatenate([obj.v for obj in self.meshes])
        center = np.mean(all_verts, axis=0)
        self.camera.set_look_at(center, self.camera.t)

    def copy(self):
        copy = Scene()
        for obj in self.meshes:
            copy.add_mesh(obj.copy())
        for line in self.lines:
            copy.add_lines(line.copy())
        copy.camera = self.camera.copy()
        copy.amb = self.amb.copy()
        copy.direct = self.direct.copy()
        copy.normals_mode = self.normals_mode
        copy.wireframe_mode = self.wireframe_mode
        copy.shading_mode = self.shading_mode
        copy.background_color = self.background_color

        return copy
