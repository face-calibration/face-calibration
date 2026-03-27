from ctypes import *
from PySide2 import QtWidgets, QtCore, QtGui
from OpenGL import GL
import matplotlib.pyplot as plt
import numpy as np

from . import renderer
from . import shaders

class Viewer(QtWidgets.QOpenGLWidget):
    '''
    This is a Qt widget which displays a single Scene. This widget can be
    embedded in a larger Qt application.
    This class handles Qt widget and QtOpenGL callbacks and passes the information
    to be processed to the Scene, Camera, or Renderer
    '''

    def __init__(self, scene, parent=None):
        '''
        initialize this viewer
        scene: Scene class
        parent: MultiViewerWidget class, from gui.py (optional)
        '''
        super().__init__(parent)

        # strong focus = keypresses captured if window is clicked or tabbed to
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.scene = scene
        self.parent_widget = parent
        if self.parent_widget:
            self.parent_widget.add_viewer(self)

    # override
    def initializeGL(self):
        print('Initializing GL')
        self.context().functions().glClearColor(*self.scene.background_color)
        shaders.initialize()
        self.renderer = renderer.Renderer()

    # override
    def resizeGL(self, width, height):
        self.width = width
        self.height = height
        GL.glViewport(0, 0, width, height)
        self.scene.camera.set_canvas_dim(width, height)
        self.scene.camera.update_proj()

    # override
    def minimumSizeHint(self):
        return QtCore.QSize(600, 400)

    # override
    def paintGL(self):
        self.renderer.render_scene(self.scene)

    def get_triangle_index(self, point, idx):
        self.makeCurrent()
        self.renderer.render_id(self.scene, idx)
        self.doneCurrent()
        image = self.get_image()
        tri_id = image[point[1], point[0]]
        tri_id = tri_id[0] * 2**16 + tri_id[1] * 2**8 + tri_id[2]
        # print('tri_id: '+str(tri_id))
        # plt.imshow(image)
        # plt.plot(point[0],point[1],'.',markersize=10)
        # plt.show()
        self.update()
        return tri_id

    # override
    def mousePressEvent(self, event):
        if self.parent_widget:
            self.parent_widget.on_viewer_mouse_press_(event)
        else:
            self.on_mouse_press_(event)

    # override
    def mouseMoveEvent(self, event):
        if self.parent_widget:
            self.parent_widget.on_viewer_mouse_move_(event)
        else:
            self.on_mouse_move_(event)

    # override
    def keyPressEvent(self, event):
        if self.parent_widget:
            # allow parent widget to handle key presses captured over this viewer
            # maybe there is a better way to do this
            self.parent_widget.keyPressEvent(event)
        else:
            self.on_key_press_(event)

    def on_mouse_press_(self, event):
        point = np.asarray([event.x(), event.y()])
        # self.get_triangle_index(point)
        self.scene.camera.down(point)

    def on_mouse_move_(self, event):
        point = np.asarray([event.x(), event.y()])
        if event.buttons() & QtCore.Qt.LeftButton:
            self.scene.camera.tumble(point)
        elif event.buttons() & QtCore.Qt.RightButton:
            self.scene.camera.zoom(point)
        elif event.buttons() & QtCore.Qt.MidButton:
            self.scene.camera.pan(point)
        self.update()

    def on_key_press_(self, event):
        if event.key() == QtCore.Qt.Key_W:
            self.scene.toggle_wireframe_mode()
            self.update()
        if event.key() == QtCore.Qt.Key_S:
            self.scene.toggle_normals_mode()
            self.update()
        if event.key() == QtCore.Qt.Key_T:
            self.scene.toggle_shading_mode()
            self.update()
        if event.key() == QtCore.Qt.Key_U:
            self.scene.toggle_lighting_mode()
            self.update()
        if event.key() == QtCore.Qt.Key_F:
            self.scene.frame_camera()
            self.update()
        if event.key() == QtCore.Qt.Key_N:
            self.scene.toggle_two_sided_lighting()
            self.update()

    def get_image(self, width=None, height=None):
        '''
        Returns a np.uint8 BGR image array of the current scene.
        Note: I believe Qt wants us to bind framebuffers and draw always within the
              makeCurrent() context. So call this function within that context
        '''
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        prev_width, prev_height = self.width, self.height
        self.resizeGL(width, height)

        # allocate framebuffer, bind it, and draw to it
        framebuffer = QtGui.QOpenGLFramebufferObject(self.width, self.height, QtGui.QOpenGLFramebufferObject.Depth)
        framebuffer.bind()
        self.renderer.render_scene(self.scene)
        framebuffer.release()

        image = framebuffer.toImage()
        image = np.array(image.constBits(), dtype=np.uint8).reshape((self.height, self.width, 4))

        self.resizeGL(prev_width, prev_height)
        # convert to BGR
        return np.ascontiguousarray(image)
