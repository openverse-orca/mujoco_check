import argparse
import math
import queue
import threading
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np
from imgui_bundle import imgui

from inspection import ModelIndex, Selection, apply_highlight, body_focus, pick_geom
from inspector_ui import InspectorUI
from model_sources import cleanup_orca_fetch, fetch_orca_xml


ZOOM_SPEED = 0.04
PAN_SPEED = 1.0
KEYBOARD_MOVE_SPEED = 300.0
CLOSE_ZOOM_RANGE = 0.1
MIN_CLOSE_ZOOM_FACTOR = 0.1
CLICK_DISTANCE = 4.0
MIN_DISTANCE_SCALE = 1e-4
MAX_DISTANCE_SCALE = 1e3


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


def focus_camera(cam, center, radius, fovy, aspect):
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
    cam.distance = float(np.clip(1.2 * radius / math.sin(half_angle), minimum, maximum))


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
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.center = self.model.stat.center.copy()
        self.extent = _valid_extent(self.model.stat.extent)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.option = mujoco.MjvOption()
        self.scene = None
        self.context = None
        self.window = None
        self.model_index = ModelIndex(self.model)
        self.selection = Selection()
        self.zoom_speed = ZOOM_SPEED
        self.pan_speed = PAN_SPEED
        self.inspector = None

        self.running = False
        self.accumulator = 0.0
        self.last_time = time.perf_counter()
        self.buttons = set()
        self.movement_keys = set()
        self.last_cursor = None
        self.left_press_origin = None
        self.left_dragging = False
        self.pending_pick = None
        self.orca_results = queue.SimpleQueue()
        self.orca_request_id = 0
        self.orca_loading = False
        self.shutting_down = threading.Event()

    @property
    def model_name(self):
        return self.xml_path.name if self.xml_path is not None else "未加载模型"

    def _set_source_status(self, text, error=False):
        if self.inspector is not None:
            self.inspector.set_source_status(text, error)

    def load_xml(self, xml_path):
        path = Path(xml_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"XML 文件不存在：{path}")

        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        model_index = ModelIndex(model)
        scene = mujoco.MjvScene(model, maxgeom=10000) if self.scene is not None else None
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

        old_context = self.context
        self.xml_path = path
        self.model = model
        self.data = data
        self.model_index = model_index
        self.scene = scene
        self.context = context
        self.center = model.stat.center.copy()
        self.extent = _valid_extent(model.stat.extent)
        self.selection.body_id = -1
        self.selection.geom_id = -1
        self.running = False
        self.accumulator = 0.0
        self.last_time = time.perf_counter()
        self._on_focus(None, False)
        if self.inspector is not None:
            self.inspector.rebind_model(model, data, model_index)
        self._reset_camera()
        self._update_title()
        if old_context is not None:
            old_context.free()

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
                self._set_source_status(f"OrcaLab 拉取失败：{error}", True)
                continue
            try:
                self.load_xml(xml_path)
            except (OSError, ValueError, RuntimeError) as load_error:
                self._set_source_status(f"模型加载失败：{load_error}", True)
            else:
                self._set_source_status(f"已从 OrcaLab 加载：{self.model_name}")
            finally:
                cleanup_orca_fetch(cache_dir)

    def _aspect_ratio(self):
        if self.window is None:
            return 1.0
        width, height = glfw.get_framebuffer_size(self.window)
        return width / height if height > 0 and width > 0 else 1.0

    def _reset_camera(self):
        reset_camera(
            self.camera,
            self.center,
            self.extent,
            self.model.vis.global_.fovy,
            self._aspect_ratio(),
        )

    def _update_title(self):
        if self.window is None:
            return
        state = "运行" if self.running else "暂停"
        glfw.set_window_title(
            self.window,
            f"{self.model_name} | {state} | 滚轮缩放 {self.zoom_speed * 100:.1f}%/格",
        )

    def _on_mouse_button(self, window, button, action, mods):
        self.inspector.backend.mouse_button_callback(window, button, action, mods)
        if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            return
        if action == glfw.PRESS:
            if imgui.get_io().want_capture_mouse:
                return
            self.buttons.add(button)
            self.last_cursor = glfw.get_cursor_pos(window)
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_press_origin = self.last_cursor
                self.left_dragging = False
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
                self.pending_pick = cursor
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_press_origin = None
                self.left_dragging = False
            self.last_cursor = cursor if self.buttons else None

    def _on_cursor_pos(self, window, x, y):
        self.inspector.backend.mouse_callback(window, x, y)
        if not self.buttons:
            self.last_cursor = None
            return
        if self.last_cursor is None:
            self.last_cursor = (x, y)
            return

        dx = x - self.last_cursor[0]
        dy = y - self.last_cursor[1]
        self.last_cursor = (x, y)
        if glfw.MOUSE_BUTTON_LEFT in self.buttons:
            if not self.left_dragging and not is_click(self.left_press_origin, (x, y)):
                self.left_dragging = True
            if self.left_dragging:
                orbit_camera(self.camera, dx, dy)
        else:
            _width, height = glfw.get_window_size(self.window)
            pan_camera(
                self.camera,
                dx,
                dy,
                height,
                self.model.vis.global_.fovy,
                self.extent,
                self.pan_speed,
            )

    def _on_scroll(self, window, x_offset, y_offset):
        self.inspector.backend.scroll_callback(window, x_offset, y_offset)
        if not imgui.get_io().want_capture_mouse:
            zoom_camera(self.camera, self.extent, y_offset, self.zoom_speed)

    def _on_focus(self, _window, focused):
        if not focused:
            self.buttons.clear()
            self.movement_keys.clear()
            self.last_cursor = None
            self.left_press_origin = None
            self.left_dragging = False
            self.pending_pick = None

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
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.running = not self.running
            self.accumulator = 0.0
            self._update_title()
        elif key == glfw.KEY_PERIOD and not self.running:
            mujoco.mj_step(self.model, self.data)
        elif key == glfw.KEY_R:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.accumulator = 0.0
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
            _width, height = glfw.get_window_size(self.window)
            keyboard_move_camera(
                self.camera,
                self.movement_keys,
                frame_time,
                height,
                self.model.vis.global_.fovy,
                self.extent,
                self.pan_speed,
            )
        if not self.running:
            return

        self.accumulator += frame_time
        timestep = self.model.opt.timestep
        steps = 0
        while self.accumulator >= timestep and steps < 20:
            mujoco.mj_step(self.model, self.data)
            self.accumulator -= timestep
            steps += 1
        if steps == 20:
            self.accumulator = 0.0

    def _render(self):
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return

        self.inspector.begin_frame()
        previous_zoom_speed = self.zoom_speed
        self.selection, self.zoom_speed, self.pan_speed = self.inspector.draw(
            self.selection, self.zoom_speed, self.pan_speed
        )
        if self.zoom_speed != previous_zoom_speed:
            self._update_title()
        focus_body = self.inspector.consume_focus_request()
        if focus_body is not None:
            center, radius = body_focus(
                self.model, self.data, self.model_index, focus_body
            )
            focus_camera(
                self.camera,
                center,
                radius,
                self.model.vis.global_.fovy,
                self._aspect_ratio(),
            )
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        near = max(self.camera.distance * 0.001, np.finfo(float).tiny)
        far = max(self.camera.distance + self.extent * 10.0, near * 1000.0)
        for camera in self.scene.camera:
            if camera.frustum_near > 0.0:
                near_scale = near / camera.frustum_near
                camera.frustum_top *= near_scale
                camera.frustum_bottom *= near_scale
            camera.frustum_near = near
            camera.frustum_far = far

        if self.pending_pick is not None:
            self.selection = pick_geom(
                self.model,
                self.data,
                self.option,
                self.scene,
                self.pending_pick,
                glfw.get_window_size(self.window),
                (width, height),
            )
            self.pending_pick = None
            if self.selection.valid:
                self.inspector.reveal_body(self.selection.body_id)
        apply_highlight(self.scene, self.selection, self.model_index)

        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        self.inspector.draw_object_label(self.selection, self.scene, (width, height))
        self.inspector.end_frame()
        glfw.swap_buffers(self.window)

    def run(self):
        initialized = glfw.init()
        if not initialized:
            raise RuntimeError("无法初始化 GLFW，请检查图形显示环境")
        try:
            self.window = glfw.create_window(
                1280, 800, self.model_name, None, None
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
            print("Space 播放/暂停，. 单步，R 重置，Home 全景，Esc 退出")

            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self._process_orca_results()
                self._step_simulation()
                width, height = glfw.get_framebuffer_size(self.window)
                if width <= 0 or height <= 0:
                    glfw.wait_events()
                    continue
                self._render()
        finally:
            self.shutting_down.set()
            self.orca_request_id += 1
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
