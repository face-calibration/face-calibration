import numpy as np
import sys
import trimesh

sys.path.append('..')

from util.mesh_util import normalize

def look_at_matrix(at, eye, up=[0, 1, 0]):
    '''
    Return a 4x4 homogeneous matrix transform (inverse of view matrix)
    representing the camera at position 'eye' pointing
    towards the point 'at', with up vector 'up'
    'at' and 'eye' are in world space
    '''
    up = np.asarray(up).astype(np.float32)
    z = normalize(eye - at)
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)

    return np.array([[x[0], y[0], z[0], eye[0]],
                     [x[1], y[1], z[1], eye[1]],
                     [x[2], y[2], z[2], eye[2]],
                     [0, 0, 0, 1]]).astype(np.float32)


class Camera:

    def __init__(self):
        # self.pose aka cam2world
        self.pose = np.eye(4)
        # note that the intrinsics we store are not the same as OpenCV (HZ)
        # because we factor in the width and height of the canvas for resizing
        self.intrinsics = {'fx': 0, 'fy': 0, 'cx': 0, 'cy': 0}
        self.near_val = 0.01
        self.far_val = 2000.0
        self.set_canvas_dim(500, 500)
        self.set_intrinsics_from_fov(45)
        self.update_proj()

    def set_canvas_dim(self, width, height):
        self.width = width
        self.height = height
        self.a = height / width

    def set_pose(self, pose):
        self.pose = pose

    def set_intrinsics_from_fov(self, fov):
        '''
        set simple camera intrinsics from a field of view scalar
        '''
        f = 1. / np.tan(fov / 2. * np.pi / 180.)
        self.intrinsics['fx'] = f
        self.intrinsics['fy'] = f
        self.intrinsics['cx'] = 0
        self.intrinsics['cy'] = 0

    def set_intrinsics_from_opencv(self, fx, fy, cx, cy, width, height):
        '''
        http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        width and height should come from the OpenCV camera's image size,
        not the current GL canvas size. This means the rendering only matches
        a calibrated camera when the GL canvas size is the width and height
        you give here (but won't distort if you change the viewport size)
        '''
        self.intrinsics['fx'] = 2 * fx / width
        # so when our fy is divided by self.a you get 2 * fy / height
        self.intrinsics['fy'] = 2 * fy / width
        # depending on the conventions of the calibrated camera, one or
        # both of these may be negated. This also corresponds with flipping
        # the x and y axes of the camera view matrix
        self.intrinsics['cx'] = (width - 2 * cx) / width
        self.intrinsics['cy'] = -(height - 2 * cy) / height


    def update_proj(self):
        '''
        projection matrix and NDC matrix combined
        call this after you update self.a or self.intrinsics or clip planes
        '''
        A = -(self.far_val + self.near_val) / (self.far_val - self.near_val)
        B = -2. * self.far_val * self.near_val / (self.far_val - self.near_val)
        self.proj = np.array([[self.intrinsics['fx'], 0, self.intrinsics['cx'], 0],
                              [0, self.intrinsics['fy'] / self.a, self.intrinsics['cy'], 0],
                              [0, 0, A, B],
                              [0, 0, -1, 0]]).astype(np.float32)

    def MVP(self):
        # invert pose to get view matrix
        view = np.eye(4)
        view[:3, :3] = self.R.T
        view[:3, 3] = -self.R.T.dot(self.t)

        return self.proj @ view

    @property
    def look_dir(self):
        '''vector representing the direction the camera is facing'''
        return -self.pose[:3, 2]

    @property
    def t(self):
        # note: the world-space xform of the camera itself; aka cam2world
        return self.pose[:3, 3]

    @property
    def R(self):
        # note: the world-space xform of the camera itself; aka cam2world
        return self.pose[:3, :3]

    def copy(self):
        copy = Camera()
        copy.pose = self.pose.copy()
        copy.intrinsics = self.intrinsics.copy()
        copy.near_val, copy.far_val = self.near_val, self.far_val
        copy.width, copy.height, copy.a = self.width, self.height, self.a
        copy.proj = self.proj.copy()
        return copy


class TrackBall(Camera):
    # constants for reasonable ui movements given mouse pixel measurements
    TUMBLE_SPEED_SCALE = 0.01
    ZOOM_SPEED_SCALE = 0.01
    PAN_SPEED_SCALE = 0.001

    def __init__(self):
        Camera.__init__(self)
        self.target = np.asarray([0, 0, 0]).astype(np.float32)
        self.pose = np.eye(4)
        self.pose[:3, 3] = np.asarray([0, 0, 75])
        self.prev_pose, self.prev_target = None, None
        self.up = [0, 1, 0]
        # determines how far in global space to pan/zoom based on mouse movement.
        # becomes smaller for smaller-scaled scenes, also when zooming in.
        self.ui_interact_scale = 1.
        self.prev_ui_interact_scale = 1.

    def set_ui_interact_scale(self, ui_interact_scale):
        self.ui_interact_scale = ui_interact_scale

    def set_look_at(self, at=None, eye=None, up=None):
        '''
        Set new camera view matrix where camera is at 'eye' looking at 'at'
        eye: (3,)
        at: (3,)
        '''
        if at is None:
            at = self.target
        if eye is None:
            eye = self.pose[:3, 3]
        if up is None:
            up = self.up
        self.pose = look_at_matrix(at, eye, up)
        self.target = at
        self.ui_interact_scale = np.linalg.norm(at - eye)

    def set_pose(self, pose):
        self.pose = pose
        # approximate new target location by self.ui_interact_scale units in front of camera
        self.target = self.t + self.look_dir * self.ui_interact_scale

    def down(self, point):
        '''
        A button has been pressed and will be dragged from this point.
        '''
        self.down_point = np.asarray(point).astype(np.float32)
        self.prev_pose = self.pose.copy()
        self.prev_target = self.target.copy()
        self.prev_ui_interact_scale = self.ui_interact_scale

    def pan(self, point):
        '''
        Translate self.target and camera t based on point's relative
        position to self.down_point
        '''
        prev_t = self.prev_pose[:3, 3]
        prev_target = self.prev_target.copy()
        prev_rot = self.prev_pose[:3, :3]
        point = np.asarray(point).astype(np.float32)
        dx, dy = point - self.down_point

        delta_cam = np.array([-dx, dy, 0])
        delta_cam *= self.PAN_SPEED_SCALE * self.prev_ui_interact_scale
        delta_world = (prev_rot @ delta_cam)

        self.pose[:3, 3] = prev_t + delta_world
        self.target = prev_target + delta_world

    def tumble(self, point):
        '''
        Rotate around self.target based on point's relative
        position to self.down_point
        Uses look-at transform to point towards self.target
        '''
        at = self.target.copy()
        prev_eye = self.prev_pose[:3, 3]

        point = np.asarray(point).astype(np.float32)
        dx, dy = point - self.down_point
        # tumbling speed is invariant to global scale
        xang = -dx * self.TUMBLE_SPEED_SCALE
        yang = -dy * self.TUMBLE_SPEED_SCALE

        # calculate new camera rotation by rotating old position around target
        prev_eye_centered = prev_eye - at
        xRot = trimesh.transformations.rotation_matrix(xang, [0, 1, 0])
        yRot = trimesh.transformations.rotation_matrix(yang, self.prev_pose[:3, 0])
        new_eye_centered = yRot.dot(xRot.dot(np.append(prev_eye_centered, [1])))
        self.pose = look_at_matrix(at, new_eye_centered[:3] + at)

    def zoom(self, point):
        '''
        Change camera's local z coordinate based on point's
        relative position to self.down_point
        '''
        prev_t = self.prev_pose[:3, 3].copy()
        point = np.asarray(point, dtype=np.float32)
        # only care about mouse dragging up or down
        _, dy = point - self.down_point
        # rotate local z displacement into world space
        dz_cam = dy * self.ZOOM_SPEED_SCALE * self.prev_ui_interact_scale
        delta_t = self.R @ np.array([0, 0, dz_cam], dtype=np.float32)
        self.pose[:3, 3] = prev_t + delta_t

        self.ui_interact_scale = np.linalg.norm(self.target - self.t)

    def copy(self):
        copy = TrackBall()

        copy.pose = self.pose.copy()
        copy.intrinsics = self.intrinsics.copy()
        copy.near_val, copy.far_val = self.near_val, self.far_val
        copy.width, copy.height, copy.a = self.width, self.height, self.a
        copy.proj = self.proj.copy()
        copy.ui_interact_scale = self.ui_interact_scale
        copy.prev_ui_interact_scale = self.prev_ui_interact_scale

        copy.target = self.target.copy()
        if self.prev_pose is not None:
            copy.prev_pose = self.prev_pose.copy()
        if self.prev_target is not None:
            copy.prev_target = self.prev_target.copy()
        return copy
