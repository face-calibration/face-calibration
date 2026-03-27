from PySide2 import QtCore, QtWidgets
import numpy as np
from moviepy.editor import ImageSequenceClip
import cv2
from pathlib import Path
from tqdm import tqdm
import sys
import shutil
import os
import time

sys.path.append('..')

from . import viewer
from . import scene
from util.colormaps import get_specials, mesh_variant, sample_colormap
from util.mesh_util import vertex_normals, face_normals
from util.constants import MH_FACE_PARTS

this_dir = Path(__file__).parent

def save_video(file_name, images, fps=60):
    images = [cv2.cvtColor(i.astype(np.uint8), cv2.COLOR_BGRA2RGBA) for i in images]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(images)):
        cv2.putText(images[i], str(i + 1), (10, images[i].shape[0] - 10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(file_name)


class MultiViewerWidget(QtWidgets.QWidget):
    '''
    This class exists to rebroadcast events captured by one viewer to all the viewers
    so their renderings always match
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.viewers = []
        self.key_handlers = {}

    def add_viewer(self, viewer):
        self.viewers.append(viewer)

    def add_key_press_handler(self, key, handler):
        self.key_handlers[key] = handler

    def on_viewer_mouse_press_(self, event):
        for viewer in self.viewers:
            viewer.on_mouse_press_(event)

    def on_viewer_mouse_move_(self, event):
        for viewer in self.viewers:
            viewer.on_mouse_move_(event)

    def on_key_press_(self, event):
        # all viewers handle keypress (not sending a new event)
        for viewer in self.viewers:
            viewer.on_key_press_(event)
        if event.key() in self.key_handlers:
            # handle keypress at GUI level as well
            self.key_handlers[event.key()](event)

    # override
    def keyPressEvent(self, event):
        self.on_key_press_(event)


class MultiFieldDialogOutput():
    pass

class MultiFieldDialog(QtWidgets.QDialog):
    '''
    Dialog pop-up for inputting multiple values at a time
    title: window title

    construct this class, add fields, call get_fields(), access members of returned object
    like argparse!
    '''
    def __init__(self, *args, title=None, **kwargs):
        super().__init__(*args, **kwargs)
        if title:
            self.setWindowTitle(title)

        self.fields = []
        self.types = []
        self.defaults = []

    def add_field(self, name, type=str, default=None):
        '''
        name: name of field displayed in the dialog
        type: the string in the field will be cast to this type for get_field()
        default: default value pre-filled
        '''
        self.fields.append(name)
        self.types.append(type)
        self.defaults.append(default)

    def build_gui(self):
        self.line_edits = []
        layout = QtWidgets.QVBoxLayout()
        for field_name, default in zip(self.fields, self.defaults):
            line_edit = QtWidgets.QLineEdit()
            if default:
                line_edit.setText(str(default))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(field_name))
            row.addStretch()
            row.addWidget(line_edit)
            self.line_edits.append(line_edit)
            layout.addLayout(row)

        button_row = QtWidgets.QHBoxLayout()
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_row.addWidget(button_box)
        layout.addLayout(button_row)

        self.setLayout(layout)

    def get_fields(self):
        self.build_gui()
        result = self.exec()

        if result:
            ret = MultiFieldDialogOutput()
            for field, line_edit, type in zip(self.fields, self.line_edits, self.types):
                setattr(ret, field.replace(' ', '_'), type(line_edit.text()))
            return ret


class StaticGUI:
    default_settings = {
        'ambient': [0.15, 0.15, 0.15],
        'direct': [1.0, 1.0, 1.0],
        'background': [0.2, 0.2, 0.2],
        'img_width': 'curr', # if an integer, set to that number. Otherwise is current viewer size
        'img_height': 'curr', # if an integer, set to that number. Otherwise is current viewer size
        'fps': 59.94
    }

    def __init__(self, objs, colors=None, lines=None, settings=None, title='Viewer'):
        """
        Initialize a GUI for static meshes. Starts with one viewer.
        objs: RenderObject or list of RenderObjects for the scene
        colors: array(num_verts, 3) or list of arrays for mesh(es) vertex colors
        lines: RenderObjectLines or list of RenderObjectLines for the scene
        """
        if not isinstance(objs, list):
            objs = [objs]
        if lines is None: lines = []
        if not isinstance(lines, list):
            lines = [lines]
        self.title = title
        self.viewers = []

        if colors is not None:
            if not isinstance(colors, list):
                colors = [colors]
            self.colors = colors
            for obj, col in zip(objs, colors):
                obj.reload_mesh(t=col)
        else:
            self.colors = [None for _ in range(len(objs))]

        # default settings
        if settings:
            self.settings = settings
        else:
            self.settings = self.default_settings

        # build default scene
        curr_scene = scene.Scene()
        curr_scene.set_amb([0.15, 0.15, 0.15])
        curr_scene.set_direct([1.0, 1.0, 1.0])
        curr_scene.set_background_color([0.2, 0.2, 0.2]) # slate gray
        # curr_scene.set_background_color([1.0, 1.0, 1.0]) # white
        for obj in objs:
            curr_scene.add_mesh(obj)
        for line in lines:
            curr_scene.add_lines(line)
        # default camera is centered in front of all the objects and pointing towards -z
        all_verts = np.concatenate([obj.v for obj in objs])
        center = np.mean(all_verts, axis=0)
        max_mag = np.max(np.linalg.norm(all_verts - center, axis=-1))
        curr_scene.camera.set_look_at(center, center + [0, 0, max_mag * 2])
        self.first_scene = curr_scene

        self.build_gui()
        self.main_widget.setWindowTitle(self.title)

        self.refresh_meshes_normals()
        self.first_scene.frame_camera()

        self.update_viewers()

    def build_gui(self):
        '''no inheritance for build_gui, build everything for this class here'''
        self.main_widget = MultiViewerWidget()
        self.main_grid_layout = QtWidgets.QGridLayout(self.main_widget)

        view = viewer.Viewer(self.first_scene, self.main_widget)
        self.main_grid_layout.addWidget(view, 0, 0)
        self.viewers.append(view)

        button = QtWidgets.QPushButton('Save Image')
        self.main_grid_layout.addWidget(button, 1, 0)
        button.clicked.connect(self.on_save_image_)

        # GUI handles normals recalculation. Is it elegant? no, not at all.
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_S, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_C, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_V, self.on_key_press_)

    def add_viewer(self):
        new_scene = self.first_scene.copy()
        view = viewer.Viewer(new_scene, self.main_widget)
        self.main_grid_layout.addWidget(view, 0, len(self.viewers))
        self.viewers.append(view)

    def update_viewers(self):
        for view in self.viewers:
            view.update()

    def set_camera_pose(self, pose):
        for view in self.viewers:
            view.scene.camera.set_pose(pose)

    def get_camera_pose(self):
        return self.viewers[0].scene.camera.pose

    def set_camera_intrinsics(self, fx, fy, cx, cy, w, h):
        for view in self.viewers:
            view.scene.camera.set_intrinsics_from_opencv(fx, fy, cx, cy, w, h)
            view.scene.camera.set_canvas_dim(w, h)
            view.scene.camera.update_proj()

    def on_save_image_(self):
        dialog = MultiFieldDialog(title='Options')
        width = self.viewers[0].width if self.settings['img_width'] == 'curr' else self.settings['img_width']
        height = self.viewers[0].width if self.settings['img_height'] == 'curr' else self.settings['img_height']
        dialog.add_field('width', type=int, default=width)
        dialog.add_field('height', type=int, default=height)

        fields = dialog.get_fields()
        if fields is None:
            return

        # I think makeCurrent() and doneCurrent() have to be done outside of the
        # get_image() function for some reason
        self.viewers[0].makeCurrent()
        image = self.viewers[0].get_image(fields.width, fields.height)
        self.viewers[0].doneCurrent()

        file_name = QtWidgets.QFileDialog.getSaveFileName(None, 'Save image as...')[0]
        if file_name:
            cv2.imwrite(file_name, image)
            print('Saved', file_name)

    def on_key_press_(self, event):
        if event.key() == QtCore.Qt.Key_S:
            # (un)smooth: send new normals to meshes
            # the viewers already handled flipping their secene's 'smooth' attribute
            self.refresh_meshes_normals()
        if event.key() == QtCore.Qt.Key_C:
            # Save camera pose
            pose = self.get_camera_pose()
            file_name = QtWidgets.QFileDialog.getSaveFileName(None, 'Save camera pose')[0]
            if file_name:
                np.save(file_name, pose)
        if event.key() == QtCore.Qt.Key_V:
            # Load camera pose
            file_name = QtWidgets.QFileDialog.getOpenFileName(None, 'Load camera pose')[0]
            if file_name:
                self.set_camera_pose(np.load(file_name))

    def resize(self, width, height):
        self.main_widget.resize(width, height)

    def show(self):
        self.main_widget.show()

    def refresh_meshes_normals(self):
        # only update the normals of each mesh. Used when toggling between smoothed
        # and flat shading. When updating vertices, normals are also recalculated there
        for view in self.viewers:
            for obj in view.scene.meshes:
                vn, fn = None, None
                if view.scene.normals_mode is scene.NormalsMode.FLAT:
                    vn = vertex_normals(obj.v, obj.f)
                else:
                    fn = face_normals(obj.v, obj.f)
                obj.reload_mesh(vn=vn, fn=fn)


class MeshDiffGUI(StaticGUI):

    def __init__(self, obj, pos1, pos2, settings=None, title='Compare'):
        """
        Initialize a GUI for comparison between two versions of a mesh.
        obj: RenderObject for the scene
        pos1, pos2: list of vertex positions, length == obj verts length
        """
        self.pos1 = np.asarray(pos1).astype(np.float32)
        self.pos2 = np.asarray(pos2).astype(np.float32)
        self.blend = 0.0

        # calls build_gui(), so call after new class members
        StaticGUI.__init__(self, obj, settings=settings, title=title)

        # overwrite self.colors
        diff = np.linalg.norm(self.pos1 - self.pos2, axis=-1)
        if np.sum(diff) == 0: diff += 1e-8
        diff = (diff - np.min(diff)) / np.max(diff)
        cmap = mesh_variant(get_specials()['viridis'])
        self.colors = sample_colormap(cmap, diff)
        self.set_mesh(self.blend)

        self.first_scene.frame_camera()


    def build_gui(self):
        '''no inheritance for build_gui, build everything for this class here'''
        self.main_widget = MultiViewerWidget()
        self.main_grid_layout = QtWidgets.QGridLayout(self.main_widget)

        view = viewer.Viewer(self.first_scene, self.main_widget)
        self.main_grid_layout.addWidget(view, 0, 0)
        self.viewers.append(view)

        self.blend_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.main_grid_layout.addWidget(self.blend_slider, 1, 0)
        self.blend_slider.valueChanged.connect(self.on_blend_slider_change_)

        button = QtWidgets.QPushButton('Save Image')
        self.main_grid_layout.addWidget(button, 2, 0)
        button.clicked.connect(self.on_save_image_)

        self.main_widget.add_key_press_handler(QtCore.Qt.Key_S, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_C, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_V, self.on_key_press_)

    def on_blend_slider_change_(self):
        value = self.blend_slider.value()
        self.blend = float(value) / 100.0
        self.set_mesh(self.blend)

    def add_viewer(self):
        raise NotImplementedError

    def set_mesh(self, blend):
        # for every viewer...
        for view in self.viewers:
            # only compute first obj for comparison
            obj = view.scene.meshes[0]
            # linear blend between two sets of vertices
            v = (1.0 - blend) * self.pos1 + blend * self.pos2
            vn, fn = None, None
            if view.scene.normals_mode is scene.NormalsMode.FLAT:
                vn = vertex_normals(obj.v, obj.f)
            else:
                fn = face_normals(obj.v, obj.f)
            t = self.colors
            obj.reload_mesh(v=v, vn=vn, fn=fn, t=t)
        self.update_viewers()


class AnimationGUI(StaticGUI):

    def __init__(self, objs, positions, colors=None, lines=None, lines_positions=None, lines_colors=None, settings=None, title='Viewer'):
        """
        Initialize a GUI for dynamic meshes with per-frame vertex animation.
        Starts with one viewer, add more with add_viewer()
        objs: RenderObject or list of RenderObjects for the primary scene
        positions: Array of animated vertex positions or list of arrays for each mesh part
        colors: Array of animated vertex colors or list of arrays for each mesh part (optional)
        lines: RenderObjectLines or list of RenderObjectLiness for the primary scene (optional)
        lines_positions: Array of animated vertex positions or list of arrays for each lines object (required if 'lines' specified)
        lines_colors: Array of animated line colors or list of arrays for each lines object (optional)
        """
        if not isinstance(positions, list):
            positions = [positions]
        # each entry in self.positions is a list of position arrays for each mesh part in one viewer
        self.positions = [positions]
        # arrange lines_positions same as positions if lines are present
        if lines:
            if not isinstance(lines_positions, list):
                lines_positions = [lines_positions]
            self.lines_positions = [lines_positions]
        else:
            self.lines_positions = None

        self.frame = 0
        self.images = []
        self.playing = False
        self.recording = False
        self.batch_out = None # used for rendering without GUI input

        # calls build_gui(), so call after new class members
        StaticGUI.__init__(self, objs, lines=lines, settings=settings, title=title)

        if colors is not None:
            if not isinstance(colors, list):
                colors = [colors]
            self.colors = [colors]
        else:
            self.colors = [[None for _ in range(len(positions))]]
        if self.lines_positions is not None:
            if lines_colors is not None:
                if not isinstance(lines_colors, list):
                    lines_colors = [lines_colors]
                self.lines_colors = [lines_colors]
            else:
                self.lines_colors = [[None for _ in range(len(lines_positions))]]
        else:
            self.lines_colors = None

        self.set_meshes_from_frame(0)
        self.first_scene.frame_camera()

    def build_gui(self):
        '''no inheritance for build_gui, build everything for this class here'''
        self.main_widget = MultiViewerWidget()
        self.main_grid_layout = QtWidgets.QGridLayout(self.main_widget)

        view = viewer.Viewer(self.first_scene, self.main_widget)
        self.main_grid_layout.addWidget(view, 0, 0 * 4, 1, 4)
        self.viewers.append(view)

        button = QtWidgets.QPushButton('Pause/Play')
        self.main_grid_layout.addWidget(button, 1, 0, 1, 1)
        button.clicked.connect(self.on_pause_play_)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, self.anim_len() - 1)
        self.main_grid_layout.addWidget(self.frame_slider, 1, 1, 1, 2)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_change_)

        self.text = QtWidgets.QLabel('1/' + str(self.anim_len()))
        self.main_grid_layout.addWidget(self.text, 1, 3, 1, 1)

        button = QtWidgets.QPushButton('Save Image')
        self.main_grid_layout.addWidget(button, 2, 1, 1, 1)
        button.clicked.connect(self.on_save_image_)

        button = QtWidgets.QPushButton('Save Video')
        self.main_grid_layout.addWidget(button, 2, 2, 1, 1)
        button.clicked.connect(self.on_save_video_)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer_tick_)

        self.main_grid_layout.setColumnStretch(0 * 4 + 0, 1) # pause/play
        self.main_grid_layout.setColumnStretch(0 * 4 + 1, 5) # time slider + save img
        self.main_grid_layout.setColumnStretch(0 * 4 + 2, 5) # time slider + save video
        self.main_grid_layout.setColumnStretch(0 * 4 + 3, 1) # frame count

        self.main_widget.add_key_press_handler(QtCore.Qt.Key_J, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_K, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_L, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_S, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_C, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_V, self.on_key_press_)

    def add_viewer(self, positions, colors=None, lines_positions=None, lines_colors=None):
        new_scene = self.first_scene.copy()
        view = viewer.Viewer(new_scene, self.main_widget)
        self.main_grid_layout.addWidget(view, 0, len(self.viewers) * 4, 1, 4)

        self.main_grid_layout.setColumnStretch(len(self.viewers) * 4 + 0, 1) # pause/play
        self.main_grid_layout.setColumnStretch(len(self.viewers) * 4 + 1, 5) # time slider + save img
        self.main_grid_layout.setColumnStretch(len(self.viewers) * 4 + 2, 5) # time slider + save video
        self.main_grid_layout.setColumnStretch(len(self.viewers) * 4 + 3, 1) # frame count

        self.viewers.append(view)

        if not isinstance(positions, list):
            positions = [positions]
        self.positions.append(positions)
        if colors is not None:
            if not isinstance(colors, list):
                colors = [colors]
            self.colors.append(colors)
        else:
            self.colors.append([None for _ in range(len(positions))])

        if lines_positions is not None:
            if not isinstance(lines_positions, list):
                lines_positions = [lines_positions]
            self.lines_positions.append(lines_positions)
            if lines_colors is not None:
                if not isinstance(lines_colors, list):
                    lines_colors = [lines_colors]
                self.lines_colors.append(lines_colors)
            else:
                self.lines_colors.append([None for _ in range(len(lines_positions))])

    def anim_len(self):
        return len(self.positions[0][0])

    def on_key_press_(self, event):
        if event.key() == QtCore.Qt.Key_J:
            # back one frame
            if not self.playing and not self.recording:
                self.frame_slider.setValue(self.frame - 1)
        if event.key() == QtCore.Qt.Key_L:
            # forward one frame
            if not self.playing and not self.recording:
                self.frame_slider.setValue(self.frame + 1)
        if event.key() == QtCore.Qt.Key_K:
            # pause/play
            self.on_pause_play_()
        if event.key() == QtCore.Qt.Key_S:
            # (un)smooth: send new normals to meshes
            self.refresh_meshes_normals()
        if event.key() == QtCore.Qt.Key_C:
            # Save camera pose
            pose = self.get_camera_pose()
            file_name = QtWidgets.QFileDialog.getSaveFileName(None, 'Save camera pose')[0]
            if file_name:
                np.save(file_name, pose)
        if event.key() == QtCore.Qt.Key_V:
            # Load camera pose
            file_name = QtWidgets.QFileDialog.getOpenFileName(None, 'Load camera pose')[0]
            if file_name:
                self.set_camera_pose(np.load(file_name))

    def on_pause_play_(self):
        if self.playing:
            self.playing = False
            self.timer.stop()
        else:
            self.playing = True
            self.timer.setInterval(33.33)
            self.timer.start()

    def on_frame_slider_change_(self):
        value = self.frame_slider.value()
        self.frame = value
        self.text.setText(str(value + 1)+ '/' + str(self.anim_len()))
        self.set_meshes_from_frame(value)

    def setup_batch_render(self, batch_out):
        self.batch_out = batch_out
        # working on the asusmption that this timer doesn't start until exec_() is called
        QtCore.QTimer.singleShot(1000, self.on_save_video_)

    def on_save_video_(self):
        '''
        If batch_out is passed, running without UI input so skip all of those steps
        '''
        width = self.viewers[0].width if self.settings['img_width'] == 'curr' else self.settings['img_width']
        height = self.viewers[0].width if self.settings['img_height'] == 'curr' else self.settings['img_height']

        if self.batch_out:
            fields = lambda: None
            fields.first_frame = 1
            fields.last_frame = self.anim_len()
            fields.width = width
            fields.height = height
            fields.fps = self.settings['fps']

            file_name = self.batch_out
        else:
            dialog = MultiFieldDialog(title='Options')
            dialog.add_field('width', type=int, default=width)
            dialog.add_field('height', type=int, default=height)
            dialog.add_field('first frame', type=int, default=1)
            dialog.add_field('last frame', type=int, default=self.anim_len())
            dialog.add_field('fps', type=float, default=self.settings['fps'])

            fields = dialog.get_fields()
            if fields is None:
                return

            file_name = QtWidgets.QFileDialog.getSaveFileName(None, 'Save video', '', '')[0]

        if file_name:
            if file_name[-4:] == '.mp4':
                file_name = file_name[:-4]
            print('rendering images...')
            for v in range(len(self.viewers)):
                images = []
                self.viewers[v].makeCurrent()

                for i in tqdm(range(fields.first_frame - 1, fields.last_frame)):
                    self.frame_slider.setValue(i)
                    images.append(self.viewers[v].get_image(fields.width, fields.height))

                self.viewers[v].doneCurrent()

                save_video(f'{file_name}_{v}.mp4', images, fields.fps)

            if shutil.which('ffmpeg') and len(self.viewers) > 1:
                # Horizontally stack input videos
                inputs = ''.join([f'-i {file_name}_{v}.mp4 ' for v in range(len(self.viewers))])
                cmd = f'ffmpeg {inputs} -filter_complex hstack=inputs={len(self.viewers)} ' + \
                      f'-c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p {file_name}_combined.mp4'
                os.system(cmd)

        if self.batch_out:
            QtCore.QCoreApplication.instance().quit()

    def on_timer_tick_(self):
        next_frame = self.frame + 2
        if next_frame < self.anim_len():
            self.frame_slider.setValue(next_frame)
        else:
            self.frame_slider.setValue(0)

    def set_meshes_from_frame(self, frame):
        # for every viewer...
        for view, position, color in zip(self.viewers, self.positions, self.colors):
            # for every mesh part in the scene...
            for obj, pos, col in zip(view.scene.meshes, position, color):
                v = pos[frame]
                vn, fn = None, None
                if view.scene.normals_mode is scene.NormalsMode.FLAT:
                    vn = vertex_normals(v, obj.f)
                else:
                    fn = face_normals(v, obj.f)
                t = None if col is None else col[frame]
                obj.reload_mesh(v=v, vn=vn, fn=fn, t=t)
        # now do the same for lines objects, if they exist
        if self.lines_positions is not None:
            for view, position, color in zip(self.viewers, self.lines_positions, self.lines_colors):
                # for every mesh part in the scene...
                for obj, pos, col in zip(view.scene.lines, position, color):
                    v = pos[frame]
                    c = None if col is None else col[frame]
                    obj.reload_mesh(v=v, c=c)
        self.update_viewers()


class RigGUI(AnimationGUI):
    def __init__(self, objs, model, rig_param_names, rig_param_ranges,
                 rig_params=None, part_names=MH_FACE_PARTS, settings=None, title='Viewer'):
        '''
        objs: RenderObject or list of RenderObjects for the primary scene
        model: rig with eval() func
        rig_param_names: list of rig parameter names
        rig_param_ranges: tuple of (rig_params_min, rig_params_max)
        rig_params: list of rig parameters for every frame
        part_names: keys to index into rig output
        '''
        self.rig_param_names = rig_param_names
        self.curr_rig_params = np.zeros(len(rig_param_names), dtype=np.float32)
        self.rig_sliders = []
        self.models = [model]
        self.rig_params_min = rig_param_ranges[0]
        self.rig_params_max = rig_param_ranges[1]
        if rig_params is None:
            rig_params = np.zeros((1, len(rig_param_names)), dtype=np.float32)
        self.rig_params = rig_params
        self.used_v_parts = part_names

        # calls build_gui(), so call after new class members
        AnimationGUI.__init__(self, objs, None, settings=settings, title=title)

    def build_gui(self):
        '''no inheritance for build_gui, build everything for this class here'''

        # main window with viewers and slider
        self.main_widget = MultiViewerWidget()
        self.main_grid_layout = QtWidgets.QGridLayout(self.main_widget)

        view = viewer.Viewer(self.first_scene, self.main_widget)
        self.main_grid_layout.addWidget(view, 0, 0 * 4, 1, 4)
        self.main_grid_layout.setColumnStretch(0 * 4 + 1, 1)
        self.main_grid_layout.setColumnStretch(0 * 4 + 2, 1)
        self.viewers.append(view)

        button = QtWidgets.QPushButton('Pause/Play')
        self.main_grid_layout.addWidget(button, 1, 0, 1, 1)
        button.clicked.connect(self.on_pause_play_)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, self.anim_len() - 1)
        self.main_grid_layout.addWidget(self.frame_slider, 1, 1, 1, 2)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_change_)

        self.ml_blend_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ml_blend_slider.setRange(0, 100)
        self.ml_blend_slider.setValue(100)
        self.main_grid_layout.addWidget(self.ml_blend_slider, 3, 1, 1, 2)
        self.ml_blend_slider.valueChanged.connect(self.on_ml_blend_slider_change_)

        ml_blend_slider_label = QtWidgets.QLabel('Blend ML Offsets')
        self.main_grid_layout.addWidget(ml_blend_slider_label, 3, 0, 1, 1)

        self.text = QtWidgets.QLabel('1/' + str(self.anim_len()))
        self.main_grid_layout.addWidget(self.text, 1, 3, 1, 1)

        button = QtWidgets.QPushButton('Save Image')
        self.main_grid_layout.addWidget(button, 2, 1, 1, 1)
        button.clicked.connect(self.on_save_image_)

        button = QtWidgets.QPushButton('Save Video')
        self.main_grid_layout.addWidget(button, 2, 2, 1, 1)
        button.clicked.connect(self.on_save_video_)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer_tick_)

        # blendshape sliders (secondary window)
        self.sliders_widget = QtWidgets.QWidget()

        sliders_reset_button = QtWidgets.QPushButton('Reset controls')
        sliders_reset_button.clicked.connect(self.reset_controls_)

        sliders_scroll_area = QtWidgets.QScrollArea(self.sliders_widget)
        sliders_scroll_area.setWidgetResizable(True)

        sliders_internal_widget = QtWidgets.QWidget(sliders_scroll_area)
        sliders_scroll_area.setWidget(sliders_internal_widget)

        sliders_widget_layout = QtWidgets.QVBoxLayout(self.sliders_widget)
        sliders_widget_layout.addWidget(sliders_reset_button)
        sliders_widget_layout.addWidget(sliders_scroll_area)

        sliders_grid_layout = QtWidgets.QGridLayout(sliders_internal_widget)
        sliders_grid_layout.setColumnStretch(1, 1)
        sliders_grid_layout.setColumnStretch(2, 1)

        with open(this_dir / 'style' / 'slider_style.css') as f:
            sliders_style = f.read()

        for i, param_name in enumerate(self.rig_param_names):
            text = QtWidgets.QLabel(param_name)
            sliders_grid_layout.addWidget(text, i, 0)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            # QSlider can only have integer values
            slider.setRange(0, 100)
            if self.rig_params_min[i] < 0 and self.rig_params_max[i] > 0:
                sliders_grid_layout.addWidget(slider, i, 1, 1, 2)
            elif self.rig_params_min[i] < 0 and self.rig_params_max[i] == 0:
                sliders_grid_layout.addWidget(slider, i, 1, 1, 1)
            elif self.rig_params_min[i] == 0 and self.rig_params_max[i] > 0:
                sliders_grid_layout.addWidget(slider, i, 2, 1, 1)
            else:
                raise ValueError('uh oh, why are we here')
            self.rig_sliders.append(slider)
            slider.valueChanged.connect(self.create_rig_slider_func(i))

            slider.setStyleSheet(sliders_style)

        self.main_widget.add_key_press_handler(QtCore.Qt.Key_J, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_K, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_L, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_S, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_C, self.on_key_press_)
        self.main_widget.add_key_press_handler(QtCore.Qt.Key_V, self.on_key_press_)

    def create_rig_slider_func(self, idx):
        def on_rig_slider_change_():
            value = float(self.rig_sliders[idx].value()) / 100.
            value = value * (self.rig_params_max[idx] - self.rig_params_min[idx]) + self.rig_params_min[idx]
            self.curr_rig_params[idx] = value
            self.set_meshes_from_rig_sliders()
        return on_rig_slider_change_

    def add_viewer(self, model):
        new_scene = self.first_scene.copy()
        view = viewer.Viewer(new_scene, self.main_widget)
        self.main_grid_layout.addWidget(view, 0, len(self.viewers) * 4, 1, 4)
        self.main_grid_layout.setColumnStretch(len(self.viewers) * 4 + 1, 1)
        self.main_grid_layout.setColumnStretch(len(self.viewers) * 4 + 2, 1)
        self.viewers.append(view)

        self.models.append(model)

    def anim_len(self):
        return self.rig_params.shape[0]

    def show(self):
        self.main_widget.show()
        self.sliders_widget.show()

    def on_ml_blend_slider_change_(self):
        value = float(self.ml_blend_slider.value()) / 100.
        for model in self.models:
            model.deltas_scale = value
        self.set_meshes_from_frame(self.frame)

    def reset_controls_(self):
        for i, slider in enumerate(self.rig_sliders):
            # Only do a rig evaluation once all sliders are finalized
            # slider.blockSignals(True)
            slider.setValue(0)
            # slider.blockSignals(False)
        # self.set_meshes_from_rig_sliders()

    def set_meshes_from_frame(self, frame):
        self.curr_rig_params = self.rig_params[frame]
        self.set_meshes_from_rig_sliders()

        for i, slider in enumerate(self.rig_sliders):
            # Don't want to re-calculate mesh when just setting slider values from frame
            slider.blockSignals(True)
            # normalize to [0, 1]
            value = (self.curr_rig_params[i] - self.rig_params_min[i]) / (self.rig_params_max[i] - self.rig_params_min[i])
            slider.setValue(value * 100)
            slider.blockSignals(False)

    def set_meshes_from_rig_sliders(self):
        # for each viewer...
        for view, model in zip(self.viewers, self.models):
            res = model.eval(self.curr_rig_params)
            v_parts = [res[part] for part in self.used_v_parts]
            # for each mesh part in the scene...
            for obj, v in zip(view.scene.meshes, v_parts):
                vn, fn = None, None
                if view.scene.normals_mode is scene.NormalsMode.FLAT:
                    vn = vertex_normals(v, obj.f)
                else:
                    fn = face_normals(v, obj.f)
                obj.reload_mesh(v=v, vn=vn, fn=fn)
        self.update_viewers()