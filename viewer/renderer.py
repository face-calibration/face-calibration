from ctypes import *
import numpy as np
from OpenGL import GL
from PySide2 import QtGui

from . import shaders
from .scene import WireframeMode, ShadingMode, LightingMode

class Renderer:

    def __init__(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)

    def load_program(self, scene, use_texture=False, texture_idx=0, normal_idx=1, draw_wireframe=False,
                     use_id=False, use_ao=False, ao_idx=2, draw_lines=False):
        '''
        texture_idx, normal_idx, and ao_idx are their locations when bound as GL_TEXTURE{idx} in mesh.py
        '''
        if use_texture:
            if scene.lighting_mode == LightingMode.DIRECT or scene.lighting_mode == LightingMode.NONE:
                use_program = shaders.program_tex_ao if use_ao else shaders.program_tex
            elif scene.lighting_mode == LightingMode.SH:
                use_program = shaders.program_sh_tex_ao if use_ao else shaders.program_sh_tex
        else:
            if scene.lighting_mode == LightingMode.DIRECT or scene.lighting_mode == LightingMode.NONE:
                use_program = shaders.program
            elif scene.lighting_mode == LightingMode.SH:
                use_program = shaders.program_sh
        if draw_wireframe:
            use_program = shaders.program_wireframe
        if use_id:
            use_program = shaders.id_program
        if draw_lines:
            use_program = shaders.program_lines

        camera = scene.camera
        id_mvp = GL.glGetUniformLocation(use_program, 'MVP')

        # Find locations of constants (OpenGL "uniforms") from compiled shader
        if scene.lighting_mode == LightingMode.DIRECT or scene.lighting_mode == LightingMode.NONE:
            id_amb = GL.glGetUniformLocation(use_program, 'amb')
            id_direct = GL.glGetUniformLocation(use_program, 'direct')
            id_direct_dir = GL.glGetUniformLocation(use_program, 'directDir')
            id_cam_t = GL.glGetUniformLocation(use_program, 'camT')
            id_spec_ang = GL.glGetUniformLocation(use_program, 'specAng')
            id_spec_mag = GL.glGetUniformLocation(use_program, 'specMag')
            id_two_sided = GL.glGetUniformLocation(use_program, 'twoSided')
        elif scene.lighting_mode == LightingMode.SH:
            id_b_coeffs = [GL.glGetUniformLocation(use_program, 'b'+str(i)) for i in range(9)]

        if use_texture:
            id_sampler = GL.glGetUniformLocation(use_program, 'sampler')
            id_normal_map = GL.glGetUniformLocation(use_program, 'normalMap')
            if use_ao:
                id_ao_map = GL.glGetUniformLocation(use_program, 'aoMap')

        GL.glUseProgram(use_program)

        # OpenGL by default expects "column-major" inputs, so transpose MVP
        # aka { rx.1 rx.2 rx.3 0 yx.1 yx.2 yx.3 0 zx.1 zx.2 zx.3 0 tx ty tz 1 }
        mvp = list(camera.MVP().astype(np.float32).T.reshape(-1))
        GL.glUniformMatrix4fv(id_mvp, 1, GL.GL_FALSE, (c_float * len(mvp))(*mvp))

        # Load constants (OpenGL "uniforms") into memory
        if scene.lighting_mode == LightingMode.DIRECT or scene.lighting_mode == LightingMode.NONE:
            cam_t = list(scene.camera.t.astype(np.float32))
            direct_dir = list(scene.camera.look_dir.astype(np.float32))
            two_sided = float(scene.two_sided_lighting)
            if scene.lighting_mode == LightingMode.DIRECT:
                amb = list(scene.amb.astype(np.float32))
                direct = list(scene.direct.astype(np.float32))
                spec_ang, spec_mag = scene.spec_ang, scene.spec_mag
            elif scene.lighting_mode == LightingMode.NONE:
                amb = [1., 1., 1.]
                direct = [0., 0., 0.]
                spec_ang, spec_mag = 1.0, 0.0

            GL.glUniform3fv(id_amb, 1, (c_float * len(amb))(*amb))
            GL.glUniform3fv(id_direct, 1, (c_float * len(direct))(*direct))
            GL.glUniform3fv(id_direct_dir, 1, (c_float * len(direct_dir))(*direct_dir))
            GL.glUniform3fv(id_cam_t, 1, (c_float * len(cam_t))(*cam_t))
            GL.glUniform1f(id_spec_ang, spec_ang)
            GL.glUniform1f(id_spec_mag, spec_mag)
            GL.glUniform1f(id_two_sided, two_sided)
        elif scene.lighting_mode == LightingMode.SH:
            for i in range(len(id_b_coeffs)):
                b_coeff = list(scene.sh_coeffs[i].astype(np.float32))
                GL.glUniform3fv(id_b_coeffs[i], 1, (c_float * len(b_coeff))(*b_coeff))

        if use_texture:
            GL.glUniform1i(id_sampler, texture_idx)
            GL.glUniform1i(id_normal_map, normal_idx)
            if use_ao:
                GL.glUniform1i(id_ao_map, ao_idx)

        return use_program

    def render_scene(self, scene, clear=True):
        if clear:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        rendered_list = []

        if scene.wireframe_mode != WireframeMode.WIREFRAME:
            # If "texture" shading, try to render all objects with the textured shader
            if scene.shading_mode == ShadingMode.TEXTURE:
                # Texture with AO
                obj_list = [obj for obj in scene.meshes if obj.texture_map is not None and obj.ao_map is not None]
                rendered_list += obj_list
                if obj_list:
                    program = self.load_program(scene, use_texture=True, use_ao=True)
                for obj in obj_list:
                    obj.render(program)

                # Texture without AO
                obj_list = [obj for obj in scene.meshes if obj.texture_map is not None and obj.ao_map is None]
                rendered_list += obj_list
                if obj_list:
                    program = self.load_program(scene, use_texture=True)
                for obj in obj_list:
                    obj.render(program)

            # Any objects not rendered with the textured shader get vertex colors
            obj_list = [obj for obj in scene.meshes if obj not in rendered_list]
            if obj_list:
                program = self.load_program(scene)
            for obj in obj_list:
                obj.render(program)

        if scene.wireframe_mode != WireframeMode.SHADED:
            # Wireframe rendering. Can render on top of shaded.
            program = self.load_program(scene, draw_wireframe=True)
            for obj in scene.meshes:
                obj.render(program, draw_wireframe=True)

        # Render "lines" objects on top afterwards
        if scene.lines:
            program = self.load_program(scene, draw_lines=True)
            for obj in scene.lines:
                obj.render(program)

        GL.glUseProgram(0)

    def render_id(self, scene, idx):
        # TODO: factor this into render_scene
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        obj = scene.meshes[idx]
        obj.colorTexture(use_id=True)
        program = self.load_program(scene, use_id=True)
        obj.render(program)
        obj.colorTexture(use_id=False)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
