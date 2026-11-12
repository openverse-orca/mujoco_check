import argparse
import asyncio
import math
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import numpy as np
from imgui_bundle import imgui
from OpenGL import GL

from .inspection import (
    ModelIndex,
    Selection,
    apply_highlight,
    body_focus,
    joint_focus,
    pick_geom,
)
from .inspector_ui import APP_NAME, InspectorUI
from .model_injection import inject_ground, joint_layout
from .model_sources import (
    cleanup_orca_fetch,
    fetch_orca_xml,
    poll_orca_states,
    stream_orca_states,
)
from .viewport_layout import framebuffer_view_rect, grid_view_rects, view_index_at


ZOOM_SPEED = 0.18
PAN_SPEED = 6.0
NEAR_CLIP_RATIO = 0.3
KEYBOARD_MOVE_SPEED = 300.0
CLOSE_ZOOM_RANGE = 0.1
MIN_CLOSE_ZOOM_FACTOR = 0.1
CLICK_DISTANCE = 4.0
MIN_DISTANCE_SCALE = 1e-4
MAX_DISTANCE_SCALE = 1e3
MIN_WINDOW_WIDTH = 640
MIN_WINDOW_HEIGHT = 480
ENVIRONMENT_BACKGROUNDS = {
    "blue": (0.56, 0.74, 0.9, 1.0),
    "black": (0.0, 0.0, 0.0, 1.0),
}


def window_resize_edges(width, height, x, y, margin=6, corner_margin=24):
    if x < corner_margin and y < corner_margin:
        return "left", "top"
    if x >= width - corner_margin and y < corner_margin:
        return "right", "top"
    if x < corner_margin and y >= height - corner_margin:
        return "left", "bottom"
    if x >= width - corner_margin and y >= height - corner_margin:
        return "right", "bottom"
    edges = []
    if x < margin:
        edges.append("left")
    elif x >= width - margin:
        edges.append("right")
    if y < margin:
        edges.append("top")
    elif y >= height - margin:
        edges.append("bottom")
    return tuple(edges)


def resized_window_geometry(x, y, width, height, edges, dx, dy):
    new_width = width
    new_height = height
    if "left" in edges:
        new_width = max(MIN_WINDOW_WIDTH, width - dx)
        x += width - new_width
    elif "right" in edges:
        new_width = max(MIN_WINDOW_WIDTH, width + dx)
    if "top" in edges:
        new_height = max(MIN_WINDOW_HEIGHT, height - dy)
        y += height - new_height
    elif "bottom" in edges:
        new_height = max(MIN_WINDOW_HEIGHT, height + dy)
    return round(x), round(y), round(new_width), round(new_height)


def _valid_extent(extent):
    extent = float(extent)
    return extent if math.isfinite(extent) and extent > 0.0 else 1.0


def camera_limits(extent):
    extent = _valid_extent(extent)
    minimum = max(extent * MIN_DISTANCE_SCALE, np.finfo(float).tiny)
    maximum = max(extent * MAX_DISTANCE_SCALE, minimum * 10.0)
    return minimum, maximum


def reset_camera(cam, center, extent, fovy, aspect):
    extent = _valid_extent(extent)
    center = np.asarray(center, dtype=float)
    cam.lookat[:] = center if np.all(np.isfinite(center)) else np.zeros(3)
    cam.azimuth = 45.0
    cam.elevation = -30.0

    vertical_half_angle = math.radians(fovy) * 0.5
    horizontal_half_angle = math.atan(
        math.tan(vertical_half_angle) * max(float(aspect), 1e-6)
    )
    half_angle = max(min(vertical_half_angle, horizontal_half_angle), 1e-6)
    distance = 1.2 * extent / math.sin(half_angle)
    minimum, maximum = camera_limits(extent)
    cam.distance = float(np.clip(distance, minimum, maximum))


def focus_camera(cam, center, radius, fovy, aspect, max_distance=None):
    radius = _valid_extent(radius)
    center = np.asarray(center, dtype=float)
    if np.all(np.isfinite(center)):
        cam.lookat[:] = center
    vertical_half_angle = math.radians(fovy) * 0.5
    horizontal_half_angle = math.atan(
        math.tan(vertical_half_angle) * max(float(aspect), 1e-6)
    )
    half_angle = max(min(vertical_half_angle, horizontal_half_angle), 1e-6)
    minimum, maximum = camera_limits(radius)
    distance = 1.2 * radius / math.sin(half_angle)
    if max_distance is not None:
        distance = min(distance, float(max_distance))
    cam.distance = float(np.clip(distance, minimum, maximum))


def close_step_scale(distance, extent):
    extent = _valid_extent(extent)
    return float(
        np.clip(
            distance / (extent * CLOSE_ZOOM_RANGE),
            MIN_CLOSE_ZOOM_FACTOR,
            1.0,
        )
    )


def zoom_camera(cam, extent, scroll_y, speed=ZOOM_SPEED):
    extent = _valid_extent(extent)
    minimum, maximum = camera_limits(extent)
    current = float(np.clip(cam.distance, minimum, maximum))
    close_factor = close_step_scale(current, extent)
    log_distance = math.log(current) - float(speed) * close_factor * float(scroll_y)
    distance = math.exp(
        float(np.clip(log_distance, math.log(minimum), math.log(maximum)))
    )
    cam.distance = float(np.clip(distance, minimum, maximum))


def is_click(start, end, threshold=CLICK_DISTANCE):
    return math.hypot(end[0] - start[0], end[1] - start[1]) < threshold


def orbit_camera(cam, dx, dy):
    cam.azimuth = (cam.azimuth - 0.25 * dx) % 360.0
    cam.elevation = float(np.clip(cam.elevation + 0.25 * dy, -89.0, 89.0))


def pan_camera(cam, dx, dy, viewport_height, fovy, extent=None, speed=PAN_SPEED):
    if viewport_height <= 0:
        return

    azimuth = math.radians(cam.azimuth)
    elevation = math.radians(cam.elevation)
    right = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    up = np.array(
        [
            -math.sin(elevation) * math.cos(azimuth),
            -math.sin(elevation) * math.sin(azimuth),
            math.cos(elevation),
        ]
    )
    units_per_pixel = (
        2.0 * cam.distance * math.tan(math.radians(fovy) * 0.5) / viewport_height
    )
    units_per_pixel *= float(speed)
    if extent is not None:
        units_per_pixel *= close_step_scale(cam.distance, extent)
    cam.lookat[:] += (right * dx + up * dy) * units_per_pixel


def keyboard_move_camera(
    cam, keys, frame_time, viewport_height, fovy, extent=None, speed=PAN_SPEED
):
    if not keys or frame_time <= 0.0 or viewport_height <= 0:
        return

    azimuth = math.radians(cam.azimuth)
    right = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    forward = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    movement = np.zeros(3)
    if glfw.KEY_W in keys:
        movement += forward
    if glfw.KEY_S in keys:
        movement -= forward
    if glfw.KEY_D in keys:
        movement += right
    if glfw.KEY_A in keys:
        movement -= right
    if glfw.KEY_E in keys:
        movement[2] += 1.0
    if glfw.KEY_Q in keys:
        movement[2] -= 1.0

    length = float(np.linalg.norm(movement))
    if length == 0.0:
        return
    units_per_second = (
        2.0
        * cam.distance
        * math.tan(math.radians(fovy) * 0.5)
        * KEYBOARD_MOVE_SPEED
        / viewport_height
    )
    units_per_second *= float(speed)
    if extent is not None:
        units_per_second *= close_step_scale(cam.distance, extent)
    cam.lookat[:] += movement / length * units_per_second * frame_time


def interaction_drag(cam, drag, viewport_height, fovy, reference_point):
    if viewport_height <= 0:
        return np.zeros(3)
    azimuth = math.radians(cam.azimuth)
    elevation = math.radians(cam.elevation)
    up = np.array(
        [
            -math.sin(elevation) * math.cos(azimuth),
            -math.sin(elevation) * math.sin(azimuth),
            math.cos(elevation),
        ]
    )
    forward = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ]
    )
    right = np.cross(forward, up)
    camera_position = np.asarray(cam.lookat) - forward * cam.distance
    depth = max(float(np.dot(np.asarray(reference_point) - camera_position, forward)), 1e-6)
    units_per_pixel = 2.0 * depth * math.tan(math.radians(fovy) * 0.5) / viewport_height
    dx, dy = drag
    return (right * dx - up * dy) * units_per_pixel


def spring_damper_force(point, target, velocity, strength, mass, extent):
    stiffness = max(float(strength), 0.0)
    if stiffness == 0.0:
        return np.zeros(3)
    mass = max(float(mass), 1e-6)
    damping = 2.0 * math.sqrt(stiffness)
    acceleration = stiffness * (np.asarray(target) - np.asarray(point))
    acceleration -= damping * np.asarray(velocity)
    force = mass * acceleration
    magnitude = float(np.linalg.norm(force))
    maximum = mass * max(stiffness * _valid_extent(extent), 1.0)
    if magnitude > maximum:
        force *= maximum / magnitude
    return force


def draw_interaction_force(scene, point, force, extent):
    magnitude = float(np.linalg.norm(force))
    if magnitude <= 1e-12 or scene.ngeom >= scene.maxgeom:
        return
    start = np.asarray(point, dtype=float)
    length = min(extent * (0.1 + 0.08 * magnitude), extent)
    end = start + np.asarray(force) / magnitude * length
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.array([1.0, 0.08, 0.04, 0.95], dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        extent * 0.008,
        start,
        end,
    )
    geom.emission = 0.25
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def draw_interaction_target(scene, point, target, extent):
    point = np.asarray(point, dtype=float)
    target = np.asarray(target, dtype=float)
    color = np.array([1.0, 0.72, 0.08, 0.9], dtype=np.float32)
    if np.linalg.norm(target - point) > 1e-9 and scene.ngeom < scene.maxgeom:
        line = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            line,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            color,
        )
        mujoco.mjv_connector(
            line,
            mujoco.mjtGeom.mjGEOM_LINE,
            extent * 0.003,
            point,
            target,
        )
        line.emission = 0.6
        line.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
    if scene.ngeom < scene.maxgeom:
        marker = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            marker,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([extent * 0.015, 0.0, 0.0]),
            target,
            np.eye(3).ravel(),
            color,
        )
        marker.emission = 0.8
        marker.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def draw_environment_background(color):
    GL.glPushAttrib(
        GL.GL_ENABLE_BIT
        | GL.GL_DEPTH_BUFFER_BIT
        | GL.GL_COLOR_BUFFER_BIT
        | GL.GL_TEXTURE_BIT
        | GL.GL_CURRENT_BIT
        | GL.GL_TRANSFORM_BIT
    )
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glPushMatrix()
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glPushMatrix()
    try:
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_FOG)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_TEXTURE_CUBE_MAP)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendEquation(GL.GL_FUNC_ADD)
        GL.glBlendFunc(GL.GL_ONE_MINUS_DST_ALPHA, GL.GL_ONE)
        GL.glColor4f(*color)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex2f(-1.0, -1.0)
        GL.glVertex2f(1.0, -1.0)
        GL.glVertex2f(1.0, 1.0)
        GL.glVertex2f(-1.0, 1.0)
        GL.glEnd()
    finally:
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glPopAttrib()


@dataclass
class ViewState:
    id: int
    camera: object
    scene: object = None
    rect: object = None
    framebuffer_rect: tuple | None = None


class Viewer:
    def __init__(self, xml_path=None, orca_address=None):
        self.startup_orca_address = orca_address
        self.xml_path = (
            Path(xml_path).expanduser().resolve() if xml_path is not None else None
        )
        if self.xml_path is not None and not self.xml_path.is_file():
            raise FileNotFoundError(f"XML 文件不存在：{self.xml_path}")

        self.model = (
            mujoco.MjModel.from_xml_path(str(self.xml_path))
            if self.xml_path is not None
            else mujoco.MjModel.from_xml_string("<mujoco/>")
        )
        self.xml_text = (
            self.xml_path.read_text(encoding="utf-8")
            if self.xml_path is not None
            else ""
        )
        self.effective_xml_text = self.xml_text
        self.ground_injected = False
        self.base_joint_layout = joint_layout(self.model)
        self.orca_resource_dir = None
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.center = self.model.stat.center.copy()
        self.extent = _valid_extent(self.model.stat.extent)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewports = [ViewState(0, camera)]
        self.active_view_id = 0
        self.next_view_id = 1
        self.mouse_view_id = None
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 1
        self.context = None
        self.window = None
        self.model_index = ModelIndex(self.model)
        self.selection = Selection()
        self.zoom_speed = ZOOM_SPEED
        self.pan_speed = PAN_SPEED
        self.near_clip_ratio = NEAR_CLIP_RATIO
        self.environment_preset = "blue"
        self.inspector = None

        self.running = False
        self.interaction_enabled = False
        self.force_strength = 20.0
        self.force_body_id = -1
        self.force_vector = np.zeros(3)
        self.force_local_point = np.zeros(3)
        self.force_start = np.zeros(3)
        self.force_target = np.zeros(3)
        self.force_point = np.zeros(3)
        self.accumulator = 0.0
        self.last_time = time.perf_counter()
        self.buttons = set()
        self.movement_keys = set()
        self.last_cursor = None
        self.left_press_origin = None
        self.left_dragging = False
        self.resizing_window = False
        self.resize_edges = ()
        self.resize_origin = None
        self.pending_pick = None
        self.orca_results = queue.SimpleQueue()
        self.orca_request_id = 0
        self.orca_loading = False
        self.orca_sync_results = queue.Queue(maxsize=1)
        self.orca_sync_request_id = 0
        self.orca_sync_stop = None
        self.orca_sync_enabled = False
        self.orca_sync_connected = False
        self.orca_push_results = queue.SimpleQueue()
        self.orca_push_request_id = 0
        self.orca_push_busy = False
        self.orca_push_connected = False
        self.orca_push_stop = None
        self.orca_push_states = None
        self.orca_push_address = ""
        self.shutting_down = threading.Event()

    @property
    def model_name(self):
        return self.xml_path.name if self.xml_path is not None else "未加载模型"

    def _view_by_id(self, view_id):
        return next((view for view in self.viewports if view.id == view_id), None)

    @property
    def active_view(self):
        view = self._view_by_id(self.active_view_id)
        if view is None:
            view = self.viewports[0]
            self.active_view_id = view.id
        return view

    @property
    def camera(self):
        return self.active_view.camera

    @camera.setter
    def camera(self, camera):
        self.active_view.camera = camera

    @property
    def scene(self):
        return self.active_view.scene

    @scene.setter
    def scene(self, scene):
        self.active_view.scene = scene

    def add_view(self):
        source = self.active_view
        camera = mujoco.MjvCamera()
        camera.type = source.camera.type
        camera.azimuth = source.camera.azimuth
        camera.elevation = source.camera.elevation
        camera.distance = source.camera.distance
        camera.lookat[:] = source.camera.lookat
        scene = (
            mujoco.MjvScene(self.model, maxgeom=10000)
            if source.scene is not None
            else None
        )
        view = ViewState(self.next_view_id, camera, scene)
        self.next_view_id += 1
        self.viewports.append(view)
        self.active_view_id = view.id
        return view.id

    def close_view(self, view_id):
        if len(self.viewports) == 1:
            return
        index = next(
            (index for index, view in enumerate(self.viewports) if view.id == view_id),
            None,
        )
        if index is None:
            return
        was_active = view_id == self.active_view_id
        del self.viewports[index]
        if was_active:
            self.active_view_id = self.viewports[min(index, len(self.viewports) - 1)].id
        if self.mouse_view_id == view_id:
            self.mouse_view_id = None
            self.buttons.clear()
            self.last_cursor = None
            self.left_press_origin = None
            self.left_dragging = False
            self._clear_external_force()
        if self.pending_pick is not None and self.pending_pick[0] == view_id:
            self.pending_pick = None

    def _update_view_layout(self, window_size=None, framebuffer_size=None):
        window_size = window_size or glfw.get_window_size(self.window)
        framebuffer_size = framebuffer_size or glfw.get_framebuffer_size(self.window)
        content_rect = getattr(self.inspector, "content_rect", None)
        if content_rect is None:
            content_rect = (0.0, 0.0, float(window_size[0]), float(window_size[1]))
        rects = grid_view_rects(len(self.viewports), *content_rect)
        if not rects:
            for view in self.viewports:
                view.rect = None
                view.framebuffer_rect = None
            return
        for view, rect in zip(self.viewports, rects):
            view.rect = rect
            view.framebuffer_rect = framebuffer_view_rect(
                rect, window_size, framebuffer_size
            )

    def _view_at_cursor(self, cursor):
        self._update_view_layout()
        index = view_index_at([view.rect for view in self.viewports], cursor)
        return self.viewports[index] if index is not None else None

    def _set_source_status(self, text, error=False):
        if self.inspector is not None:
            self.inspector.set_source_status(text, error)

    def _compile_candidate(
        self,
        path,
        base_xml_text,
        ground_enabled,
        expected_layout=None,
        from_path=False,
    ):
        effective_xml_text = inject_ground(base_xml_text) if ground_enabled else base_xml_text
        if from_path:
            model = mujoco.MjModel.from_xml_path(str(path))
        else:
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".xml",
                    prefix=".mtool_",
                    dir=path.parent,
                    delete=False,
                ) as stream:
                    stream.write(effective_xml_text)
                    temporary_path = Path(stream.name)
                model = mujoco.MjModel.from_xml_path(str(temporary_path))
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        if expected_layout is not None and joint_layout(model) != expected_layout:
            raise ValueError("注入后 Joint 布局发生变化，已取消加载")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        model_index = ModelIndex(model)
        scenes = (
            [mujoco.MjvScene(model, maxgeom=10000) for _view in self.viewports]
            if self.scene is not None
            else [None] * len(self.viewports)
        )
        context = None
        try:
            if self.context is not None:
                context = mujoco.MjrContext(
                    model, mujoco.mjtFontScale.mjFONTSCALE_150
                )
        except Exception:
            if context is not None:
                context.free()
            raise
        return model, data, model_index, scenes, context, effective_xml_text

    def _commit_model(
        self, path, base_xml_text, ground_enabled, candidate
    ):
        model, data, model_index, scenes, context, effective_xml_text = candidate
        self.stop_orca_sync()
        self.stop_orca_push()
        self._clear_external_force()
        self.interaction_enabled = False
        old_context = self.context
        self.xml_path = path
        self.xml_text = base_xml_text
        self.effective_xml_text = effective_xml_text
        self.ground_injected = ground_enabled
        self.base_joint_layout = joint_layout(model)
        self.model = model
        self.data = data
        self.model_index = model_index
        for view, scene in zip(self.viewports, scenes):
            view.scene = scene
        self.context = context
        self.center = model.stat.center.copy()
        self.extent = _valid_extent(model.stat.extent)
        self.selection.body_id = -1
        self.selection.geom_id = -1
        self.selection.joint_id = -1
        self.running = False
        self.accumulator = 0.0
        self.last_time = time.perf_counter()
        self._on_focus(None, False)
        if self.inspector is not None:
            self.inspector.mode = "view"
            self.inspector.orca_sync_enabled = False
            self.inspector.orca_push_busy = False
            self.inspector.local_running = False
            self.inspector.ground_injected = ground_enabled
            self.inspector.rebind_model(
                model, data, model_index, self.model_name, self.effective_xml_text
            )
        self._reset_all_cameras()
        self._update_title()
        if old_context is not None:
            old_context.free()

    def _replace_orca_resource_dir(self, resource_dir):
        old_resource_dir = self.orca_resource_dir
        self.orca_resource_dir = resource_dir
        if old_resource_dir is not None and old_resource_dir != resource_dir:
            cleanup_orca_fetch(old_resource_dir)

    def load_xml(self, xml_path, resource_dir=None):
        path = Path(xml_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"XML 文件不存在：{path}")

        xml_text = path.read_text(encoding="utf-8")
        candidate = self._compile_candidate(path, xml_text, False, from_path=True)
        self._commit_model(path, xml_text, False, candidate)
        self._replace_orca_resource_dir(resource_dir)

    def set_ground_enabled(self, enabled):
        enabled = bool(enabled)
        if self.xml_path is None or enabled == self.ground_injected:
            return
        try:
            candidate = self._compile_candidate(
                self.xml_path,
                self.xml_text,
                enabled,
                expected_layout=self.base_joint_layout,
            )
        except (OSError, ValueError, RuntimeError) as error:
            self._set_source_status(f"地面注入失败：{error}", True)
            return
        self._commit_model(self.xml_path, self.xml_text, enabled, candidate)
        self._set_source_status(
            "已注入本地地面，仅影响本地模型和仿真"
            if enabled
            else "已移除本地地面"
        )

    def load_local_xml(self, xml_path):
        self.orca_request_id += 1
        self.orca_loading = False
        if self.inspector is not None:
            self.inspector.orca_busy = False
        try:
            self.load_xml(xml_path)
        except (OSError, ValueError, RuntimeError) as error:
            self._set_source_status(f"打开失败：{error}", True)
        else:
            self._set_source_status(f"已打开：{self.model_name}")

    def start_orca_fetch(self, address):
        address = address.strip()
        if not address or self.orca_loading:
            return
        self.stop_orca_sync()
        self.orca_request_id += 1
        request_id = self.orca_request_id
        self.orca_loading = True
        self.inspector.orca_busy = True
        self._set_source_status(f"正在连接 OrcaLab：{address}")
        threading.Thread(
            target=self._orca_worker,
            args=(request_id, address),
            daemon=True,
        ).start()

    def _orca_worker(self, request_id, address):
        cache_dir = None
        try:
            xml_path, cache_dir = fetch_orca_xml(address)
            if self.shutting_down.is_set():
                cleanup_orca_fetch(cache_dir)
                return
            self.orca_results.put((request_id, xml_path, cache_dir, None))
        except Exception as error:
            self.orca_results.put((request_id, None, cache_dir, str(error)))

    def _process_orca_results(self):
        while True:
            try:
                request_id, xml_path, cache_dir, error = self.orca_results.get_nowait()
            except queue.Empty:
                break
            if request_id != self.orca_request_id:
                if cache_dir is not None:
                    cleanup_orca_fetch(cache_dir)
                continue

            self.orca_loading = False
            self.inspector.orca_busy = False
            if error is not None:
                if cache_dir is not None:
                    cleanup_orca_fetch(cache_dir)
                self._set_source_status(f"OrcaLab 拉取失败：{error}", True)
                continue
            try:
                self.load_xml(xml_path, resource_dir=cache_dir)
            except (OSError, ValueError, RuntimeError) as load_error:
                cleanup_orca_fetch(cache_dir)
                self._set_source_status(f"模型加载失败：{load_error}", True)
            else:
                self._set_source_status(f"已从 OrcaLab 加载：{self.model_name}")

    def _clear_external_force(self):
        if self.force_body_id >= 0:
            self.data.xfrc_applied[self.force_body_id] = 0.0
        self.force_body_id = -1
        self.force_vector = np.zeros(3)
        self.force_local_point = np.zeros(3)
        self.force_start = np.zeros(3)
        self.force_target = np.zeros(3)
        self.force_point = np.zeros(3)

    def _update_interaction_force(self):
        body_id = self.force_body_id
        if body_id < 0:
            return
        rotation = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        point = self.data.xpos[body_id] + rotation @ self.force_local_point
        body_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            body_velocity,
            0,
        )
        point_velocity = body_velocity[3:] + np.cross(
            body_velocity[:3], point - self.data.xipos[body_id]
        )
        force = spring_damper_force(
            point,
            self.force_target,
            point_velocity,
            self.force_strength,
            self.model.body_mass[body_id],
            self.extent,
        )
        self.force_point = point
        self.force_vector = force
        self.data.xfrc_applied[body_id, :3] = force
        self.data.xfrc_applied[body_id, 3:] = np.cross(
            point - self.data.xipos[body_id], force
        )

    def set_mode(self, mode, address):
        if mode == "view" and self.interaction_enabled:
            self.reset_interaction()
        if mode != "interaction":
            self.stop_orca_push()
        if mode == "sync":
            self.interaction_enabled = False
            self.running = False
            self.accumulator = 0.0
            self._clear_external_force()
            self.start_orca_sync(address)
            return

        self.stop_orca_sync()
        self._clear_external_force()
        self.interaction_enabled = mode == "interaction"
        self.running = self.interaction_enabled
        self.accumulator = 0.0
        if self.inspector is not None:
            self.inspector.mode = "interaction" if self.interaction_enabled else "view"
            self.inspector.orca_sync_enabled = False
            self.inspector.local_running = self.running
        self._set_source_status(
            "本地主仿真（推送）"
            if self.interaction_enabled
            else "模型查看模式"
        )
        if self.interaction_enabled:
            self.push_local_state(address)

    def set_orca_sync(self, enabled, address):
        self.set_mode("sync" if enabled else "view", address)

    def toggle_interaction_running(self):
        if self.interaction_enabled:
            self.running = not self.running
            self.accumulator = 0.0

    def step_interaction(self):
        if self.interaction_enabled and not self.running:
            self._update_interaction_force()
            mujoco.mj_step(self.model, self.data)

    def reset_interaction(self):
        if not self.interaction_enabled:
            return
        self._clear_external_force()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.accumulator = 0.0

    def set_joint_qpos(self, joint_id, values):
        if not self.interaction_enabled or not 0 <= joint_id < self.model.njnt:
            return False
        joint_type = int(self.model.jnt_type[joint_id])
        sizes = {
            int(mujoco.mjtJoint.mjJNT_FREE): 7,
            int(mujoco.mjtJoint.mjJNT_BALL): 4,
            int(mujoco.mjtJoint.mjJNT_HINGE): 1,
            int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        }
        size = sizes.get(joint_type)
        values = np.asarray(values, dtype=float)
        if size is None or values.shape != (size,) or not np.all(np.isfinite(values)):
            self._set_source_status("关节状态值无效", True)
            return False
        if joint_type in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
            quaternion = values[-4:]
            norm = float(np.linalg.norm(quaternion))
            if norm <= np.finfo(float).eps:
                self._set_source_status("关节四元数不能为零", True)
                return False
            values = values.copy()
            values[-4:] = quaternion / norm

        self.running = False
        self.accumulator = 0.0
        address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[address : address + size] = values
        velocity_address = int(self.model.jnt_dofadr[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            velocity_size = 6
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            velocity_size = 3
        else:
            velocity_size = 1
        self.data.qvel[velocity_address : velocity_address + velocity_size] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return True

    def stop_orca_push(self):
        self.orca_push_request_id += 1
        self.orca_push_busy = False
        self.orca_push_connected = False
        self.orca_push_address = ""
        if self.orca_push_stop is not None:
            self.orca_push_stop.set()
            self.orca_push_stop = None
        state_queue = self.orca_push_states
        self.orca_push_states = None
        if state_queue is not None:
            while True:
                try:
                    state_queue.get_nowait()
                except queue.Empty:
                    break
        if self.inspector is not None:
            self.inspector.orca_push_busy = False

    def _queue_orca_push_state(self):
        state_queue = self.orca_push_states
        if state_queue is None:
            return
        result = (
            self.orca_push_request_id,
            self.data.qpos.copy(),
            float(self.data.time),
        )
        while True:
            try:
                state_queue.put_nowait(result)
                return
            except queue.Full:
                try:
                    state_queue.get_nowait()
                except queue.Empty:
                    pass

    def push_local_state(self, address):
        address = address.strip()
        if not self.interaction_enabled:
            return
        if not address:
            self._set_source_status("OrcaLab 地址不能为空", True)
            return
        if self.xml_path is None:
            self._set_source_status("请先加载与 OrcaLab 对应的 XML", True)
            return
        if self.orca_push_stop is not None:
            if address == self.orca_push_address:
                self._queue_orca_push_state()
                return
            self.stop_orca_push()

        self.orca_push_request_id += 1
        request_id = self.orca_push_request_id
        stop_event = threading.Event()
        state_queue = queue.Queue(maxsize=1)
        self.orca_push_stop = stop_event
        self.orca_push_states = state_queue
        self.orca_push_address = address
        self.orca_push_busy = True
        self.orca_push_connected = False
        if self.inspector is not None:
            self.inspector.orca_push_busy = True
        self._queue_orca_push_state()
        self._set_source_status(f"正在连接 OrcaLab 实时同步：{address}")
        threading.Thread(
            target=self._orca_push_worker,
            args=(request_id, address, state_queue, stop_event),
            daemon=True,
        ).start()

    def _orca_push_worker(self, request_id, address, state_queue, stop_event):
        try:
            asyncio.run(
                stream_orca_states(
                    address,
                    state_queue,
                    stop_event,
                    lambda pushed_id: self.orca_push_results.put((pushed_id, None)),
                )
            )
            error = None
        except Exception as exception:
            error = str(exception)
        if (
            error is not None
            and not stop_event.is_set()
            and not self.shutting_down.is_set()
        ):
            self.orca_push_results.put((request_id, error))

    def _process_orca_push_results(self):
        while True:
            try:
                request_id, error = self.orca_push_results.get_nowait()
            except queue.Empty:
                break
            if request_id != self.orca_push_request_id:
                continue
            self.orca_push_busy = False
            if self.inspector is not None:
                self.inspector.orca_push_busy = False
            if error is not None:
                self.orca_push_connected = False
                self.orca_push_address = ""
                self.orca_push_states = None
                if self.orca_push_stop is not None:
                    self.orca_push_stop.set()
                    self.orca_push_stop = None
                self._set_source_status(f"OrcaLab 实时同步失败：{error}", True)
            elif not self.orca_push_connected:
                self.orca_push_connected = True
                self._set_source_status("本地主仿真（正在推送到 OrcaLab）")

    def start_orca_sync(self, address):
        address = address.strip()
        if not address or self.xml_path is None or self.orca_loading:
            if self.inspector is not None:
                self.inspector.orca_sync_enabled = False
                self.inspector.mode = "view"
                self.inspector.local_running = False
            if not address:
                self._set_source_status("OrcaLab 地址不能为空", True)
            elif self.xml_path is None:
                self._set_source_status("请先加载与 OrcaLab 对应的 XML", True)
            else:
                self._set_source_status("请等待 OrcaLab XML 加载完成", True)
            return

        self.stop_orca_sync()
        self.interaction_enabled = False
        self._clear_external_force()
        self.running = False
        self.accumulator = 0.0
        self.orca_sync_request_id += 1
        request_id = self.orca_sync_request_id
        stop_event = threading.Event()
        self.orca_sync_stop = stop_event
        self.orca_sync_enabled = True
        if self.inspector is not None:
            self.inspector.orca_sync_enabled = True
            self.inspector.mode = "sync"
            self.inspector.local_running = False
        self._set_source_status(f"正在连接 OrcaLab 主仿真（拉取）：{address}")
        threading.Thread(
            target=self._orca_sync_worker,
            args=(request_id, address, stop_event),
            daemon=True,
        ).start()

    def stop_orca_sync(self, notify=False):
        was_enabled = self.orca_sync_enabled
        self.orca_sync_request_id += 1
        self.orca_sync_enabled = False
        self.orca_sync_connected = False
        if self.orca_sync_stop is not None:
            self.orca_sync_stop.set()
            self.orca_sync_stop = None
        if self.inspector is not None:
            self.inspector.orca_sync_enabled = False
            if getattr(self.inspector, "mode", None) == "sync":
                self.inspector.mode = "view"
        if notify and was_enabled:
            self._set_source_status("已停止拉取 OrcaLab 状态，当前为模型查看模式")

    def _queue_orca_sync_result(self, result):
        while True:
            try:
                self.orca_sync_results.put_nowait(result)
                return
            except queue.Full:
                try:
                    self.orca_sync_results.get_nowait()
                except queue.Empty:
                    pass

    def _orca_sync_worker(self, request_id, address, stop_event):
        try:
            asyncio.run(
                poll_orca_states(
                    address,
                    lambda state: self._queue_orca_sync_result(
                        (request_id, state, None)
                    ),
                    stop_event,
                )
            )
        except Exception as error:
            if not stop_event.is_set() and not self.shutting_down.is_set():
                self._queue_orca_sync_result((request_id, None, str(error)))

    def _process_orca_sync_results(self):
        while True:
            try:
                request_id, state, error = self.orca_sync_results.get_nowait()
            except queue.Empty:
                break
            if request_id != self.orca_sync_request_id or not self.orca_sync_enabled:
                continue
            if error is not None:
                self.stop_orca_sync()
                self._set_source_status(f"OrcaLab 同步失败：{error}", True)
                continue

            try:
                qpos, qvel, qacc = (
                    np.asarray(values, dtype=float) for values in state
                )
                if (
                    qpos.shape != (self.model.nq,)
                    or qvel.shape != (self.model.nv,)
                    or qacc.shape != (self.model.nv,)
                ):
                    raise ValueError("远端模型与当前 XML 的状态维度不一致")
                if not all(np.all(np.isfinite(values)) for values in (qpos, qvel, qacc)):
                    raise ValueError("远端仿真状态包含无效数值")
                self.data.qpos[:] = qpos
                self.data.qvel[:] = qvel
                self.data.qacc[:] = qacc
                mujoco.mj_forward(self.model, self.data)
            except (TypeError, ValueError, RuntimeError) as state_error:
                self.stop_orca_sync()
                self._set_source_status(f"OrcaLab 同步失败：{state_error}", True)
                continue

            if not self.orca_sync_connected:
                self.orca_sync_connected = True
                self._set_source_status("OrcaLab 主仿真（正在拉取状态）")

    def _aspect_ratio(self, view=None):
        view = view or self.active_view
        if view.rect is not None and view.rect.height > 0:
            return view.rect.width / view.rect.height
        if self.window is None:
            return 1.0
        width, height = glfw.get_framebuffer_size(self.window)
        return width / height if height > 0 and width > 0 else 1.0

    def _reset_camera(self, view=None):
        view = view or self.active_view
        reset_camera(
            view.camera,
            self.center,
            self.extent,
            self.model.vis.global_.fovy,
            self._aspect_ratio(view),
        )

    def _reset_all_cameras(self):
        for view in self.viewports:
            self._reset_camera(view)

    def _update_title(self):
        if self.window is None:
            return
        glfw.set_window_title(self.window, APP_NAME)

    def _on_mouse_button(self, window, button, action, mods):
        self.inspector.backend.mouse_button_callback(window, button, action, mods)
        if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            return
        if (
            button == glfw.MOUSE_BUTTON_LEFT
            and action == glfw.RELEASE
            and self.resizing_window
        ):
            self.resizing_window = False
            self.resize_edges = ()
            self.resize_origin = None
            self.last_cursor = None
            return
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT and not glfw.get_window_attrib(
                window, glfw.MAXIMIZED
            ):
                width, height = glfw.get_window_size(window)
                cursor = glfw.get_cursor_pos(window)
                edges = window_resize_edges(width, height, *cursor)
                if edges:
                    window_x, window_y = glfw.get_window_pos(window)
                    self.resizing_window = True
                    self.resize_edges = edges
                    self.resize_origin = (
                        window_x,
                        window_y,
                        width,
                        height,
                        window_x + cursor[0],
                        window_y + cursor[1],
                    )
                    self.last_cursor = cursor
                    return
            if imgui.get_io().want_capture_mouse:
                return
            cursor = glfw.get_cursor_pos(window)
            view = self._view_at_cursor(cursor)
            if view is None:
                return
            self.active_view_id = view.id
            self.mouse_view_id = view.id
            self.buttons.add(button)
            self.last_cursor = cursor
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_press_origin = self.last_cursor
                self.left_dragging = False
                if (
                    self.interaction_enabled
                    and mods & glfw.MOD_CONTROL
                    and view.scene is not None
                ):
                    picked = pick_geom(
                        self.model,
                        self.data,
                        self.option,
                        view.scene,
                        (cursor[0] - view.rect.x, cursor[1] - view.rect.y),
                        (view.rect.width, view.rect.height),
                        view.framebuffer_rect[2:],
                    )
                    if picked.body_id > 0 and picked.point is not None:
                        point = np.asarray(picked.point, dtype=float)
                        rotation = np.asarray(self.data.xmat[picked.body_id]).reshape(3, 3)
                        self.force_body_id = picked.body_id
                        self.force_local_point = (
                            rotation.T @ (point - self.data.xpos[picked.body_id])
                        )
                        self.force_start = point.copy()
                        self.force_target = point.copy()
                        self.force_point = point.copy()
                        self.selection = picked
                        self.inspector.reveal_body(picked.body_id)
                        self._update_interaction_force()
        elif action == glfw.RELEASE:
            was_active = button in self.buttons
            self.buttons.discard(button)
            cursor = glfw.get_cursor_pos(window)
            if (
                button == glfw.MOUSE_BUTTON_LEFT
                and was_active
                and not self.left_dragging
                and self.left_press_origin is not None
                and is_click(self.left_press_origin, cursor)
                and not imgui.get_io().want_capture_mouse
            ):
                self.pending_pick = (self.mouse_view_id, cursor)
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._clear_external_force()
                self.left_press_origin = None
                self.left_dragging = False
            self.last_cursor = cursor if self.buttons else None
            if not self.buttons:
                self.mouse_view_id = None

    def _on_cursor_pos(self, window, x, y):
        self.inspector.backend.mouse_callback(window, x, y)
        if self.resizing_window:
            if self.last_cursor is not None:
                origin = self.resize_origin
                window_x, window_y = glfw.get_window_pos(window)
                dx = window_x + x - origin[4]
                dy = window_y + y - origin[5]
                new_x, new_y, new_width, new_height = resized_window_geometry(
                    *origin[:4],
                    self.resize_edges,
                    dx,
                    dy,
                )
                if (new_x, new_y) != (window_x, window_y):
                    glfw.set_window_pos(window, new_x, new_y)
                glfw.set_window_size(window, new_width, new_height)
            self.last_cursor = (x, y)
            return
        if not self.buttons:
            if not glfw.get_window_attrib(window, glfw.MAXIMIZED):
                width, height = glfw.get_window_size(window)
                edges = window_resize_edges(width, height, x, y)
                if edges:
                    horizontal = "left" in edges or "right" in edges
                    vertical = "top" in edges or "bottom" in edges
                    if horizontal and vertical:
                        north_west = ("left" in edges) == ("top" in edges)
                        cursor = (
                            imgui.MouseCursor_.resize_nwse
                            if north_west
                            else imgui.MouseCursor_.resize_nesw
                        )
                    elif horizontal:
                        cursor = imgui.MouseCursor_.resize_ew
                    else:
                        cursor = imgui.MouseCursor_.resize_ns
                    imgui.set_mouse_cursor(cursor)
            self.last_cursor = None
            return
        if self.last_cursor is None:
            self.last_cursor = (x, y)
            return
        view = self._view_by_id(self.mouse_view_id)
        if view is None or view.rect is None:
            self.buttons.clear()
            self._clear_external_force()
            return

        dx = x - self.last_cursor[0]
        dy = y - self.last_cursor[1]
        self.last_cursor = (x, y)
        if glfw.MOUSE_BUTTON_LEFT in self.buttons:
            if not self.left_dragging and not is_click(self.left_press_origin, (x, y)):
                self.left_dragging = True
            if self.left_dragging and self.force_body_id > 0:
                self.force_target = self.force_start + interaction_drag(
                    view.camera,
                    (x - self.left_press_origin[0], y - self.left_press_origin[1]),
                    view.rect.height,
                    self.model.vis.global_.fovy,
                    self.force_start,
                )
                self._update_interaction_force()
            elif self.left_dragging:
                orbit_camera(view.camera, dx, dy)
        else:
            pan_camera(
                view.camera,
                dx,
                dy,
                view.rect.height,
                self.model.vis.global_.fovy,
                self.extent,
                self.pan_speed,
            )

    def _on_scroll(self, window, x_offset, y_offset):
        self.inspector.backend.scroll_callback(window, x_offset, y_offset)
        if imgui.get_io().want_capture_mouse:
            return
        view = self._view_at_cursor(glfw.get_cursor_pos(window))
        if view is not None:
            self.active_view_id = view.id
            zoom_camera(view.camera, self.extent, y_offset, self.zoom_speed)

    def _on_focus(self, _window, focused):
        if not focused:
            self._clear_external_force()
            self.buttons.clear()
            self.movement_keys.clear()
            self.last_cursor = None
            self.left_press_origin = None
            self.left_dragging = False
            self.resizing_window = False
            self.resize_edges = ()
            self.resize_origin = None
            self.pending_pick = None
            self.mouse_view_id = None

    def _on_key(self, window, key, scancode, action, mods):
        self.inspector.backend.keyboard_callback(window, key, scancode, action, mods)
        movement_keys = {
            glfw.KEY_W,
            glfw.KEY_A,
            glfw.KEY_S,
            glfw.KEY_D,
            glfw.KEY_Q,
            glfw.KEY_E,
        }
        if action == glfw.RELEASE and key in movement_keys:
            self.movement_keys.discard(key)
            return
        if imgui.get_io().want_capture_keyboard:
            self.movement_keys.clear()
            return
        if key in movement_keys:
            if action == glfw.PRESS:
                self.movement_keys.add(key)
            return
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            self.inspector.request_exit()
        elif key == glfw.KEY_SPACE and self.interaction_enabled:
            self.toggle_interaction_running()
            self._update_title()
        elif key == glfw.KEY_PERIOD and self.interaction_enabled:
            self.step_interaction()
        elif key == glfw.KEY_R and self.interaction_enabled:
            self.reset_interaction()
        elif key == glfw.KEY_HOME:
            self._reset_camera()

    def _on_char(self, window, codepoint):
        self.inspector.backend.char_callback(window, codepoint)

    def _on_resize(self, window, width, height):
        self.inspector.backend.resize_callback(window, width, height)

    def _step_simulation(self):
        now = time.perf_counter()
        frame_time = min(now - self.last_time, 0.1)
        self.last_time = now
        if self.inspector is not None and imgui.get_io().want_capture_keyboard:
            self.movement_keys.clear()
        if self.movement_keys:
            view = self.active_view
            height = (
                view.rect.height
                if view.rect is not None
                else glfw.get_window_size(self.window)[1]
            )
            keyboard_move_camera(
                view.camera,
                self.movement_keys,
                frame_time,
                height,
                self.model.vis.global_.fovy,
                self.extent,
                self.pan_speed,
            )
        if self.interaction_enabled:
            self._queue_orca_push_state()
        if (
            not self.running
            or not self.interaction_enabled
            or self.orca_sync_enabled
        ):
            return

        self.accumulator += frame_time
        timestep = self.model.opt.timestep
        steps = 0
        while self.accumulator >= timestep and steps < 20:
            self._update_interaction_force()
            mujoco.mj_step(self.model, self.data)
            self.accumulator -= timestep
            steps += 1
        if steps == 20:
            self.accumulator = 0.0
        self._queue_orca_push_state()

    def _update_view_scene(self, view):
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            view.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            view.scene,
        )
        near = max(
            view.camera.distance * self.near_clip_ratio, np.finfo(float).tiny
        )
        far = max(view.camera.distance + self.extent * 10.0, near * 1000.0)
        for camera in view.scene.camera:
            if camera.frustum_near > 0.0:
                near_scale = near / camera.frustum_near
                camera.frustum_top *= near_scale
                camera.frustum_bottom *= near_scale
            camera.frustum_near = near
            camera.frustum_far = far

    def _render(self):
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return

        window_size = glfw.get_window_size(self.window)
        self.inspector.begin_frame()
        self.inspector.local_running = self.running
        self.inspector.force_strength = self.force_strength
        self.inspector.view_ids = [view.id for view in self.viewports]
        self.inspector.active_view_id = self.active_view_id
        previous_zoom_speed = self.zoom_speed
        (
            self.selection,
            self.zoom_speed,
            self.pan_speed,
            self.environment_preset,
            self.near_clip_ratio,
        ) = self.inspector.draw(
            self.selection,
            self.zoom_speed,
            self.pan_speed,
            self.environment_preset,
            self.near_clip_ratio,
        )
        self.force_strength = self.inspector.force_strength
        self._update_view_layout(window_size, (width, height))
        if self.zoom_speed != previous_zoom_speed:
            self._update_title()
        focus_body = self.inspector.consume_focus_request()
        if focus_body is not None:
            view = self.active_view
            max_distance = None
            if self.selection.joint_id >= 0:
                center, radius = joint_focus(
                    self.model, self.data, self.selection.joint_id
                )
                max_distance = view.camera.distance
            else:
                center, radius = body_focus(
                    self.model, self.data, self.model_index, focus_body
                )
            focus_camera(
                view.camera,
                center,
                radius,
                self.model.vis.global_.fovy,
                self._aspect_ratio(view),
                max_distance=max_distance,
            )
        for view in self.viewports:
            if (
                view.scene is not None
                and view.rect is not None
                and view.rect.width > 0
                and view.rect.height > 0
            ):
                self._update_view_scene(view)

        if self.pending_pick is not None:
            view_id, cursor = self.pending_pick
            view = self._view_by_id(view_id)
            self.pending_pick = None
            if (
                view is not None
                and view.scene is not None
                and view.rect is not None
                and view.framebuffer_rect is not None
            ):
                self.selection = pick_geom(
                    self.model,
                    self.data,
                    self.option,
                    view.scene,
                    (cursor[0] - view.rect.x, cursor[1] - view.rect.y),
                    (view.rect.width, view.rect.height),
                    view.framebuffer_rect[2:],
                )
                self.active_view_id = view.id
                if self.selection.valid:
                    self.inspector.reveal_body(self.selection.body_id)

        background = ENVIRONMENT_BACKGROUNDS[self.environment_preset]
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(*background)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for view in self.viewports:
            if (
                view.scene is None
                or view.framebuffer_rect is None
                or view.framebuffer_rect[2] <= 0
                or view.framebuffer_rect[3] <= 0
            ):
                continue
            apply_highlight(view.scene, self.selection, self.model_index)
            if self.force_body_id > 0:
                visual_extent = min(
                    self.extent, max(view.camera.distance * 0.35, 1e-6)
                )
                draw_interaction_force(
                    view.scene, self.force_point, self.force_vector, visual_extent
                )
                draw_interaction_target(
                    view.scene, self.force_point, self.force_target, visual_extent
                )
            left, bottom, view_width, view_height = view.framebuffer_rect
            GL.glEnable(GL.GL_SCISSOR_TEST)
            GL.glScissor(left, bottom, view_width, view_height)
            GL.glViewport(left, bottom, view_width, view_height)
            mujoco.mjr_render(
                mujoco.MjrRect(left, bottom, view_width, view_height),
                view.scene,
                self.context,
            )
            if self.environment_preset == "blue":
                draw_environment_background(background)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        active = self.active_view
        if active.scene is not None and active.rect is not None:
            self.inspector.draw_object_label(
                self.selection,
                active.scene,
                active.framebuffer_rect[2:],
                window_size=(active.rect.width, active.rect.height),
                window_offset=(active.rect.x, active.rect.y),
            )
        self.inspector.end_frame()
        glfw.swap_buffers(self.window)

    def run(self):
        initialized = glfw.init()
        if not initialized:
            raise RuntimeError("无法初始化 GLFW，请检查图形显示环境")
        try:
            glfw.window_hint(glfw.DECORATED, False)
            self.window = glfw.create_window(
                1280, 800, APP_NAME, None, None
            )
            if not self.window:
                raise RuntimeError("无法创建 OpenGL 窗口")

            glfw.make_context_current(self.window)
            glfw.focus_window(self.window)
            glfw.swap_interval(1)
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.context = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_150
            )
            self.inspector = InspectorUI(
                self.window,
                self.model,
                self.data,
                self.model_index,
                self.option,
            )
            self.inspector.on_open_xml = self.load_local_xml
            self.inspector.on_orca_pull = self.start_orca_fetch
            self.inspector.on_orca_sync = self.set_orca_sync
            self.inspector.on_mode_change = self.set_mode
            self.inspector.on_joint_set = self.set_joint_qpos
            self.inspector.on_toggle_running = self.toggle_interaction_running
            self.inspector.on_single_step = self.step_interaction
            self.inspector.on_reset_simulation = self.reset_interaction
            self.inspector.on_push_state = self.push_local_state
            self.inspector.on_ground_change = self.set_ground_enabled
            self.inspector.on_add_view = self.add_view
            self.inspector.on_close_view = self.close_view
            self.inspector.on_exit = lambda: glfw.set_window_should_close(self.window, True)
            self.inspector.rebind_model(
                self.model,
                self.data,
                self.model_index,
                self.model_name,
                self.xml_text,
            )
            if self.xml_path is not None:
                self.inspector.set_source_status(f"已打开：{self.model_name}")
            if self.startup_orca_address is not None:
                self.inspector.orca_address = self.startup_orca_address
                self.start_orca_fetch(self.startup_orca_address)

            glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
            glfw.set_scroll_callback(self.window, self._on_scroll)
            glfw.set_window_focus_callback(self.window, self._on_focus)
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_char_callback(self.window, self._on_char)
            glfw.set_window_size_callback(self.window, self._on_resize)

            self._reset_camera()
            self._update_title()
            self.last_time = time.perf_counter()
            print(
                f"左键单击选择，左键拖动旋转，右键平移，"
                f"滚轮缩放 {self.zoom_speed * 100:.1f}%/格"
            )
            print("WASD 前后左右移动，Q/E 下降/上升")
            print("Space 播放/暂停，. 单步，R 重置，Home 全景，Esc 打开退出确认")

            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self._process_orca_results()
                self._process_orca_sync_results()
                self._process_orca_push_results()
                self._step_simulation()
                width, height = glfw.get_framebuffer_size(self.window)
                if width <= 0 or height <= 0:
                    glfw.wait_events()
                    continue
                self._render()
        finally:
            self.shutting_down.set()
            self.stop_orca_sync()
            self._clear_external_force()
            self.orca_request_id += 1
            self.stop_orca_push()
            self._process_orca_results()
            if self.inspector is not None:
                self.inspector.shutdown()
                self.inspector = None
            if self.context is not None:
                self.context.free()
                self.context = None
            if self.window is not None:
                glfw.destroy_window(self.window)
                self.window = None
            self._replace_orca_resource_dir(None)
            glfw.terminate()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="带对象拾取、层级和分组检查功能的 MuJoCo XML 查看器"
    )
    parser.add_argument("xml", nargs="?", help="要查看的 MuJoCo XML 文件")
    parser.add_argument(
        "--orca",
        nargs="?",
        const="localhost:50051",
        metavar="地址",
        help="启动后直接连接 OrcaLab 并拉取 XML（默认 localhost:50051）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        Viewer(args.xml, args.orca).run()
    except (FileNotFoundError, ValueError, RuntimeError, glfw.GLFWError) as error:
        raise SystemExit(f"错误：{error}") from error


if __name__ == "__main__":
    main()
