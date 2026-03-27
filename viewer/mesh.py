from ctypes import *
import numpy as np
from OpenGL import GL, GLU
import time

class RenderObject:
    '''
    Triangular meshes
    '''
    def __init__(self, v, f, vn=None, fn=None, t=None, dynamic=False, uv=None):
        '''
        Initializes the RenderObject.
        v: vertex positions (num_verts, 3)
        f: faces array (num_faces, 3)
        vn: vertex normals (num_verts, 3). can either use vertex normals or face normals
        fn: face normals (num_faces, 3). can either use vertex normals or face normals
        t: vertex colors (num_verts, 3)
        dynamic: will the object's vertex positions ever change
        uv: vertex UV coordinates, per-face (num_faces, 3, 2)
        '''
        self.v = v.copy().astype(np.float32)
        self.f = f.copy().astype(np.int32)
        self.vn = None if vn is None else vn.copy().astype(np.float32)
        self.fn = None if fn is None else fn.copy().astype(np.float32)
        self.s = np.ones(self.v.shape[0]).astype(np.float32)
        self.uv = None if uv is None else uv.copy().astype(np.float32)
        # default grey vertex colors
        self.t = np.ones(v.shape).astype(np.float32) * 0.5 if t is None else t.copy().astype(np.float32)

        self.dynamic = dynamic
        self.draw = GL.GL_DYNAMIC_DRAW if dynamic else GL.GL_STATIC_DRAW
        self.wireframe_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Compute the color id texture (num_faces, 3, 3)
        t_id = np.zeros((self.f.shape[0], 3, 3)).astype(np.float32)
        t_id = t_id.reshape((-1, 3))
        for i in range(t_id.shape[0] // 3):
            b = i % 256
            g = int(int(i) / 256) % 256
            r = int(int(int(i) / 256) / 256) % 256
            for t in range(3):
                t_id[3*i+t, 0] = r / 255.0
                t_id[3*i+t, 1] = g / 255.0
                t_id[3*i+t, 2] = b / 255.0
        self.t_id = t_id.reshape((-1, 3, 3))

        # Compute barycentric texture (num_faces, 3, 3)
        t_bary = np.zeros(self.t_id.shape).astype(np.float32)
        t_bary = t_bary.reshape((-1, 3, 3))
        for i in range(t_bary.shape[0]):
            t_bary[i, 0, 0] = 1
            t_bary[i, 1, 1] = 1
            t_bary[i, 2, 2] = 1
        self.t_bary = t_bary

        self.initialized = False
        self.texture_map = None
        self.normal_map = None
        self.ao_map = None
        self.visible = True

    def set_texture_map(self, texture_map):
        '''texture_map: TextureObject'''
        self.texture_map = texture_map

    def set_normal_map(self, normal_map):
        '''normal_map: TextureObject'''
        self.normal_map = normal_map

    def set_ao_map(self, ao_map):
        '''ao_map: TextureObject'''
        self.ao_map = ao_map

    def set_wireframe_color(self, wireframe_color):
        '''wireframe_color: (3,)'''
        self.wireframe_color = np.asarray(wireframe_color, dtype=np.float32)

    def is_initialized(self):
        return self.initialized

    def set_visibility(self, visibility):
        self.visible = visibility

    def initialize_mesh(self):
        null = c_void_p(0)
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Vertex
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        vertices = self.v[self.f.reshape(-1)].reshape(-1)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vertices) * 4, (c_float * len(vertices))(*vertices), self.draw)

        # Normal
        self.nbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.nbo)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        if self.vn is not None:
            normals = self.vn[self.f.reshape(-1)].reshape(-1)
        elif self.fn is not None:
            normals = np.repeat(self.fn[:, np.newaxis], 3, axis=1).reshape(-1)
        else:
            raise ValueError('No normals exist for mesh')
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(normals) * 4, (c_float * len(normals))(*normals), self.draw)

        # Vertex color
        self.cbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        textures = self.t[self.f.reshape(-1)].reshape(-1)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(textures) * 4, (c_float * len(textures))(*textures), self.draw)

        # Shadow flag
        self.sbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.sbo)
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(3, 1, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        shadows = self.s[self.f.reshape(-1)].reshape(-1)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(shadows) * 4, (c_float * len(shadows))(*shadows), self.draw)

        # UV
        if self.uv is not None:
            self.uvbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.uvbo)
            GL.glEnableVertexAttribArray(4)
            GL.glVertexAttribPointer(4, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
            # uv already exists in per-face VBO format
            uv = self.uv.reshape(-1)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, len(uv) * 4, (c_float * len(uv))(*uv), self.draw)

        self.wireframe = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.wireframe)
        self.wireframe_indices = np.asarray([[i, i+1, i+1, i+2, i+2, i] for i in range(0, self.f.size - 1, 3)]).reshape(-1).astype(np.int32)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, len(self.wireframe_indices) * 4, (c_int * len(self.wireframe_indices))(*self.wireframe_indices), self.draw)

        GL.glBindVertexArray(0)
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(1)
        GL.glDisableVertexAttribArray(2)
        GL.glDisableVertexAttribArray(3)
        GL.glDisableVertexAttribArray(4)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        self.initialized = True

    def reload_mesh(self, v=None, vn=None, fn=None, t=None):
        '''
        v: new vertex positions (num_verts, 3)
        vn: new vertex normals (num_verts, 3). either use vn or fn
        fn: new face normals (num_faces, 3). either use vn or fn
        t: new vertex colors (num_verts, 3)
        '''
        if v is not None:
            self.v = v
            vertices = v[self.f.reshape(-1)].reshape(-1).astype(np.float32)
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vertices) * 4, vertices, self.draw)
        if vn is not None:
            self.fn = None
            self.vn = vn
            normals = vn[self.f.reshape(-1)].reshape(-1).astype(np.float32)
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.nbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(normals) * 4, normals, self.draw)
        if fn is not None:
            self.vn = None
            self.fn = fn
            normals = np.repeat(fn[:, np.newaxis], 3, axis=1).reshape(-1).astype(np.float32)
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.nbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(normals) * 4, normals, self.draw)
        if t is not None:
            self.t = t
            textures = t[self.f.reshape(-1)].reshape(-1).astype(np.float32)
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(textures) * 4, textures, self.draw)

    def color_texture(self, use_id=True, use_bary=False):
        if not self.initialized:
            self.initialize_mesh()
        if use_id:
            if use_bary:
                textures = self.t_bary
            else:
                textures = self.t_id
        else:
            textures = self.t[self.f.reshape(-1)].reshape(-1).astype(np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(textures) * 4, (c_float * len(textures))(*textures), self.draw)

    def render(self, program, draw_wireframe=False, render_depth=False):
        if not self.initialized:
            self.initialize_mesh()
        if not self.visible:
            return

        if self.texture_map is not None:
            self.texture_map.activate(0)
            self.normal_map.activate(1)
        if self.ao_map is not None:
            self.ao_map.activate(2)
        if draw_wireframe:
            ambIdx = GL.glGetUniformLocation(program, 'amb')
            biasIdx = GL.glGetUniformLocation(program, 'bias')
            if render_depth:
                amb = self.wireframe_color
                GL.glUniform3fv(ambIdx, 1, (c_float * len(amb))(*amb))
                GL.glUniform1f(biasIdx, np.float32(0.0))
                GL.glBindVertexArray(self.vao)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.f) * 3)
            amb = self.wireframe_color
            GL.glUniform3fv(ambIdx, 1, (c_float * len(amb))(*amb))
            GL.glUniform1f(biasIdx, np.float32(-0.00005))
            GL.glBindVertexArray(self.vao)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.wireframe)
            GL.glDrawElements(GL.GL_LINES, len(self.wireframe_indices), GL.GL_UNSIGNED_INT, c_void_p(0))
        else:
            GL.glBindVertexArray(self.vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.f) * 3)
        GL.glBindVertexArray(0)

    def copy(self):
        '''
        note: copy is not initialized
        '''
        copy = RenderObject(self.v, self.f, self.vn, self.fn, self.t, self.dynamic, self.uv)
        if self.texture_map is not None:
            copy.set_texture_map(self.texture_map.copy())
        if self.normal_map is not None:
            copy.set_normal_map(self.normal_map.copy())
        if self.ao_map is not None:
            copy.set_ao_map(self.ao_map.copy())
        copy.set_wireframe_color(self.wireframe_color.copy())
        copy.set_visibility(self.visible)
        return copy


class RenderObjectLines:
    '''
    Sets of lines (edges) only
    '''
    def __init__(self, v, e, c=None, dynamic=False, draw_endpoints=True):
        '''
        Initializes the RenderObject.
        v: vertex positions (num_verts, 3)
        e: edges array int(num_edges, 2)
        c: edge colors (num_edges, 3)
        dynamic: will the object's vertex positions ever change
        draw_endpoints: draw circles on endpoints
        '''
        self.v = v.copy().astype(np.float32)
        self.e = e.copy().astype(np.int32)
        # default dark grey edge colors
        self.c = np.ones(v.shape).astype(np.float32) * 0.1 if c is None else c.copy().astype(np.float32)

        self.dynamic = dynamic
        self.draw = GL.GL_DYNAMIC_DRAW if dynamic else GL.GL_STATIC_DRAW

        self.initialized = False
        self.visible = True
        self.draw_endpoints = draw_endpoints

    def is_initialized(self):
        return self.initialized

    def set_visibility(self, visibility):
        self.visible = visibility

    def set_draw_endpoints(self, draw_endpoints):
        self.draw_endpoints = draw_endpoints

    def initialize_mesh(self):
        null = c_void_p(0)
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Vertex
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        vertices = self.v[self.e.reshape(-1)].reshape(-1)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vertices) * 4, vertices, self.draw)

        # Edge color
        self.cbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, null)
        colors = np.repeat(self.c, 2, axis=0).reshape(-1)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(colors) * 4, colors, self.draw)

        GL.glBindVertexArray(0)
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        self.initialized = True

    def reload_mesh(self, v=None, c=None):
        '''
        v: new vertex positions (num_verts, 3)
        c: new edge colors (num_edges, 3)
        '''
        if v is not None:
            self.v = v
            vertices = v[self.e.reshape(-1)].reshape(-1).astype(np.float32)
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vertices) * 4, (c_float * len(vertices))(*vertices), self.draw)
        if c is not None:
            self.c = c
            colors = np.repeat(c, 2, axis=0).reshape(-1)
            if self.initialized:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), self.draw)

    def render(self, program):
        if not self.initialized:
            self.initialize_mesh()
        if not self.visible:
            return

        biasIdx = GL.glGetUniformLocation(program, 'bias')
        GL.glUniform1f(biasIdx, np.float32(-0.00005))

        GL.glBindVertexArray(self.vao)

        GL.glLineWidth(3)
        GL.glDrawArrays(GL.GL_LINES, 0, len(self.e) * 2)
        GL.glLineWidth(1)

        if self.draw_endpoints:
            # Change endpoint colors to black and yellow
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
            colors = np.stack([np.zeros((self.c.shape[0], 3), dtype=np.float32),
                               np.repeat(np.array([1, 1, 0], dtype=np.float32)[np.newaxis], self.c.shape[0], axis=0)],
                               axis=1).reshape(-1)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), self.draw)

            GL.glPointSize(3)
            GL.glDrawArrays(GL.GL_POINTS, 0, len(self.e) * 2)
            GL.glPointSize(1)

            # Revert endpoint colors to original
            colors = np.repeat(self.c, 2, axis=0).reshape(-1)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, len(colors) * 4, (c_float * len(colors))(*colors), self.draw)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        GL.glBindVertexArray(0)

    def copy(self):
        '''
        note: copy is not initialized
        '''
        copy = RenderObjectLines(self.v, self.e, self.c, self.dynamic)
        copy.set_visibility(self.visible)
        return copy

class InstanceObject:

    def __init__(self, render_object):
        self.render_object = render_object
        self.s = None
        self.uv = render_object.uv
        self.visible = True

    def set_object(self, render_object):
        self.render_object = render_object

    def set_orientation(self, t, s=None):
        self.t = t
        if s is not None:
            self.s = s

    def set_visibility(self, visibility):
        self.visible = visibility

    def initialized(self):
        return self.render_object.initialized()

    def initialize_mesh(self):
        self.render_object.initialize_mesh()

    def render(self, mvp, mvp_id, draw_wireframe=False, program=-1, render_depth=False):
        if not self.visible:
            return
        if not isinstance(mvp, np.ndarray):
            mvp = np.asarray(mvp)
        mvp = mvp.reshape((4, 4)).T
        t = np.eye(4)
        t[:3, 3] = self.t
        if self.s is not None:
            t[:3, :3] *= self.s
        new_mat = list(mvp.dot(t).T.reshape(-1).astype(np.float32))
        GL.glUniformMatrix4fv(mvp_id, 1, GL.GL_FALSE, (c_float * len(new_mat))(*new_mat))
        self.render_object.render(mvp, mvp_id, draw_wireframe=draw_wireframe, program=program, render_depth=render_depth)
        mvp = list(mvp.T.reshape(-1).astype(np.float32))
        GL.glUniformMatrix4fv(mvp_id, 1, GL.GL_FALSE, (c_float * len(mvp))(*mvp))

    @property
    def texture_map(self):
        return self.render_object.texture_map

    @property
    def normal_map(self):
        return self.render_object.normal_map

    @property
    def ao_map(self):
        return self.render_object.ao_map

class TextureObject:

    def __init__(self, image):
        image = image.astype(np.uint8)
        image = image[..., :3].copy()
        self.initialized = False
        self.image = image

    def activate(self, index):
        if not self.initialized:
            self.texture = GL.glGenTextures(1)
            height, width = self.image.shape[:2]
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GLU.gluBuild2DMipmaps(GL.GL_TEXTURE_2D, GL.GL_RGB, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.image)
            self.initialized = True

        index_map = {0: GL.GL_TEXTURE0, 1: GL.GL_TEXTURE1, 2: GL.GL_TEXTURE2}
        GL.glActiveTexture(index_map[index])
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

    def copy(self):
        '''
        note: copy is not initialized
        '''
        copy = TextureObject(self.image)
        return copy
