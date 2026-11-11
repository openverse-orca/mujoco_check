import math
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import glfw
import mujoco
import numpy as np

from inspection import project_to_window
from main import (
    Viewer,
    MIN_CLOSE_ZOOM_FACTOR,
    ENVIRONMENT_BACKGROUNDS,
    ZOOM_SPEED,
    camera_limits,
    draw_interaction_force,
    draw_interaction_target,
    focus_camera,
    interaction_drag,
    is_click,
    keyboard_move_camera,
    orbit_camera,
    pan_camera,
    parse_args,
    reset_camera,
    resized_window_geometry,
    spring_damper_force,
    window_resize_edges,
    zoom_camera,
)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = SimpleNamespace(
            lookat=np.zeros(3),
            distance=1.0,
            azimuth=0.0,
            elevation=0.0,
        )

    def test_one_scroll_notch_uses_default_zoom_speed(self):
        self.camera.distance = 2.0
        zoom_camera(self.camera, extent=1.0, scroll_y=1.0)
        self.assertAlmostEqual(
            self.camera.distance, 2.0 * math.exp(-ZOOM_SPEED), places=14
        )

    def test_user_can_adjust_zoom_speed(self):
        self.camera.distance = 2.0
        zoom_camera(self.camera, extent=1.0, scroll_y=1.0, speed=0.01)
        self.assertAlmostEqual(self.camera.distance, 2.0 * math.exp(-0.01), places=14)

    def test_close_zoom_automatically_uses_smaller_steps(self):
        self.camera.distance = 0.001
        zoom_camera(self.camera, extent=1.0, scroll_y=1.0)
        self.assertAlmostEqual(
            self.camera.distance,
            0.001 * math.exp(-ZOOM_SPEED * MIN_CLOSE_ZOOM_FACTOR),
            places=14,
        )

    def test_zoom_and_initial_fit_are_scale_invariant(self):
        normalized_distances = []
        for extent in (1e-6, 1.0, 1e6):
            with self.subTest(extent=extent):
                reset_camera(self.camera, [0.0, 0.0, 0.0], extent, 45.0, 1.6)
                normalized_distances.append(self.camera.distance / extent)
                before = self.camera.distance
                zoom_camera(self.camera, extent, 1.0)
                self.assertAlmostEqual(
                    self.camera.distance / before, math.exp(-ZOOM_SPEED), places=14
                )
        self.assertAlmostEqual(normalized_distances[0], normalized_distances[1])
        self.assertAlmostEqual(normalized_distances[1], normalized_distances[2])

    def test_zoom_is_clamped_at_both_limits(self):
        minimum, maximum = camera_limits(1e-6)
        self.camera.distance = minimum
        zoom_camera(self.camera, 1e-6, 1e6)
        self.assertEqual(self.camera.distance, minimum)

        self.camera.distance = maximum
        zoom_camera(self.camera, 1e-6, -1e6)
        self.assertEqual(self.camera.distance, maximum)
        self.assertTrue(math.isfinite(self.camera.distance))

    def test_initial_fit_uses_center_and_accounts_for_aspect(self):
        center = np.array([0.001, -0.002, 0.003])
        reset_camera(self.camera, center, 0.01, 45.0, 2.0)
        wide_distance = self.camera.distance
        np.testing.assert_array_equal(self.camera.lookat, center)
        self.assertEqual(self.camera.azimuth, 45.0)
        self.assertEqual(self.camera.elevation, -30.0)

        reset_camera(self.camera, center, 0.01, 45.0, 0.5)
        self.assertGreater(self.camera.distance, wide_distance)
        np.testing.assert_array_equal(center, [0.001, -0.002, 0.003])

    def test_invalid_scene_statistics_fall_back_safely(self):
        reset_camera(self.camera, [math.nan, 0.0, 0.0], 0.0, 45.0, 1.0)
        np.testing.assert_array_equal(self.camera.lookat, np.zeros(3))
        self.assertTrue(math.isfinite(self.camera.distance))
        self.assertGreater(self.camera.distance, 0.0)

    def test_focus_centers_body_without_changing_view_direction(self):
        self.camera.azimuth = 33.0
        self.camera.elevation = -22.0
        self.camera.distance = 10.0
        focus_camera(self.camera, [0.1, 0.2, 0.3], 0.01, 45.0, 1.6)
        np.testing.assert_array_equal(self.camera.lookat, [0.1, 0.2, 0.3])
        self.assertEqual(self.camera.azimuth, 33.0)
        self.assertEqual(self.camera.elevation, -22.0)
        self.assertLess(self.camera.distance, 10.0)

    def test_focus_can_avoid_moving_camera_farther_away(self):
        self.camera.distance = 0.25
        focus_camera(
            self.camera,
            [0.1, 0.2, 0.3],
            100.0,
            45.0,
            1.6,
            max_distance=self.camera.distance,
        )
        self.assertEqual(self.camera.distance, 0.25)

    def test_click_threshold_separates_click_from_drag(self):
        self.assertTrue(is_click((10, 10), (12, 12)))
        self.assertFalse(is_click((10, 10), (14, 10)))
        self.assertFalse(is_click((10, 10), (15, 15)))

    def test_orbit_wraps_azimuth_and_limits_elevation(self):
        orbit_camera(self.camera, dx=400.0, dy=1000.0)
        self.assertEqual(self.camera.azimuth, 260.0)
        self.assertEqual(self.camera.elevation, 89.0)
        orbit_camera(self.camera, dx=0.0, dy=-2000.0)
        self.assertEqual(self.camera.elevation, -89.0)

    def test_pan_scales_with_distance_and_inverse_viewport_height(self):
        self.camera.distance = 1.0
        pan_camera(self.camera, 10.0, -5.0, 800.0, 45.0, speed=1.0)
        reference = self.camera.lookat.copy()
        units_per_pixel = math.tan(math.radians(45.0) * 0.5) / 400.0
        np.testing.assert_allclose(
            reference, [0.0, 10.0 * units_per_pixel, -5.0 * units_per_pixel]
        )

        self.camera.lookat[:] = 0.0
        self.camera.distance = 2.0
        pan_camera(self.camera, 10.0, -5.0, 800.0, 45.0, speed=1.0)
        np.testing.assert_allclose(self.camera.lookat, reference * 2.0)
        self.assertEqual(self.camera.distance, 2.0)

        self.camera.lookat[:] = 0.0
        pan_camera(self.camera, 10.0, -5.0, 400.0, 45.0, speed=1.0)
        np.testing.assert_allclose(self.camera.lookat, reference * 4.0)

    def test_pan_speed_adjusts_and_slows_down_close_to_scene(self):
        self.camera.distance = 0.05
        pan_camera(self.camera, 10.0, 0.0, 800.0, 45.0, speed=1.0)
        reference = self.camera.lookat.copy()

        self.camera.lookat[:] = 0.0
        pan_camera(self.camera, 10.0, 0.0, 800.0, 45.0, extent=1.0, speed=0.5)
        np.testing.assert_allclose(self.camera.lookat, reference * 0.25)

    def test_keyboard_moves_horizontally_and_vertically_without_zoom(self):
        self.camera.distance = 2.0
        keyboard_move_camera(
            self.camera, {glfw.KEY_W}, 0.5, 800.0, 45.0
        )
        self.assertGreater(self.camera.lookat[0], 0.0)
        self.assertEqual(self.camera.lookat[1], 0.0)
        self.assertEqual(self.camera.lookat[2], 0.0)
        self.assertEqual(self.camera.distance, 2.0)

        self.camera.lookat[:] = 0.0
        keyboard_move_camera(self.camera, {glfw.KEY_E}, 0.5, 800.0, 45.0)
        self.assertGreater(self.camera.lookat[2], 0.0)
        self.camera.lookat[:] = 0.0
        keyboard_move_camera(self.camera, {glfw.KEY_Q}, 0.5, 800.0, 45.0)
        self.assertLess(self.camera.lookat[2], 0.0)

        self.camera.lookat[:] = 0.0
        keyboard_move_camera(self.camera, {glfw.KEY_W}, 1.0, 800.0, 45.0)
        reference_length = np.linalg.norm(self.camera.lookat)
        self.camera.lookat[:] = 0.0
        keyboard_move_camera(
            self.camera, {glfw.KEY_W, glfw.KEY_D}, 1.0, 800.0, 45.0
        )
        self.assertAlmostEqual(np.linalg.norm(self.camera.lookat), reference_length)

    def test_vertical_pan_matches_render_camera_up_without_changing_depth(self):
        model = mujoco.MjModel.from_xml_path("small_scene.xml")
        data = mujoco.MjData(model)
        scene = mujoco.MjvScene(model, maxgeom=100)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 1.0
        camera.azimuth = 45.0
        camera.elevation = -30.0
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )

        pan_camera(camera, 0.0, 100.0, 800.0, 45.0)
        movement = camera.lookat.copy()
        self.assertEqual(camera.distance, 1.0)
        self.assertAlmostEqual(float(np.dot(movement, scene.camera[0].forward)), 0.0)
        np.testing.assert_allclose(
            movement / np.linalg.norm(movement), scene.camera[0].up, atol=1e-12
        )


class InteractionForceTests(unittest.TestCase):
    def setUp(self):
        self.camera = SimpleNamespace(
            lookat=np.zeros(3), distance=1.0, azimuth=0.0, elevation=0.0
        )

    def test_drag_maps_to_camera_right_and_up(self):
        right = interaction_drag(self.camera, (10.0, 0.0), 800.0, 45.0, np.zeros(3))
        up = interaction_drag(self.camera, (0.0, -10.0), 800.0, 45.0, np.zeros(3))
        expected_length = 10.0 * math.tan(math.radians(45.0) * 0.5) / 400.0

        np.testing.assert_allclose(right, [0.0, -expected_length, 0.0])
        np.testing.assert_allclose(up, [0.0, 0.0, expected_length])

    def test_drag_target_projects_back_to_mouse_position(self):
        viewer = Viewer("small_scene.xml")
        viewer.camera.azimuth = 45.0
        viewer.camera.elevation = -30.0
        scene = mujoco.MjvScene(viewer.model, maxgeom=100)
        mujoco.mjv_updateScene(
            viewer.model,
            viewer.data,
            viewer.option,
            None,
            viewer.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        scene.camera[0].pos = (
            np.asarray(scene.camera[0].pos) + np.asarray(scene.camera[1].pos)
        ) * 0.5
        scene.camera[0].frustum_width = scene.camera[0].frustum_top * 640.0 / 480.0
        start = viewer.data.xipos[1].copy()
        cursor = project_to_window(start, scene, (640, 480), (640, 480))

        target = start + interaction_drag(
            viewer.camera,
            (100.0, -50.0),
            480.0,
            viewer.model.vis.global_.fovy,
            start,
        )
        projected = project_to_window(target, scene, (640, 480), (640, 480))

        np.testing.assert_allclose(
            projected, (cursor[0] + 100.0, cursor[1] - 50.0), atol=1e-5
        )

    def test_zero_height_produces_no_drag_and_depth_scales_linearly(self):
        zero = interaction_drag(self.camera, (10.0, 10.0), 0.0, 45.0, np.zeros(3))
        drag = interaction_drag(self.camera, (10.0, 0.0), 800.0, 45.0, np.zeros(3))
        doubled = interaction_drag(
            self.camera, (10.0, 0.0), 800.0, 45.0, np.array([1.0, 0.0, 0.0])
        )

        np.testing.assert_array_equal(zero, np.zeros(3))
        np.testing.assert_allclose(doubled, drag * 2.0)

    def test_spring_force_has_damping_and_magnitude_limit(self):
        force = spring_damper_force(
            np.zeros(3), np.array([0.1, 0.0, 0.0]), np.zeros(3), 20.0, 1.0, 1.0
        )
        damping = spring_damper_force(
            np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), 20.0, 1.0, 1.0
        )
        limited = spring_damper_force(
            np.zeros(3), np.array([100.0, 0.0, 0.0]), np.zeros(3), 20.0, 1.0, 1.0
        )

        np.testing.assert_allclose(force, [2.0, 0.0, 0.0])
        self.assertLess(damping[0], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(limited)), 20.0)

    def test_force_arrow_follows_grasp_point_and_force_direction(self):
        model = mujoco.MjModel.from_xml_path("small_scene.xml")
        scene = mujoco.MjvScene(model, maxgeom=100)
        original_geom_count = model.ngeom
        point = np.array([0.2, 0.1, 0.3])

        draw_interaction_force(scene, point, np.array([10.0, 0.0, 0.0]), 2.0)

        self.assertEqual(scene.ngeom, 1)
        self.assertEqual(model.ngeom, original_geom_count)
        arrow = scene.geoms[0]
        self.assertEqual(arrow.type, int(mujoco.mjtGeom.mjGEOM_ARROW))
        self.assertEqual(arrow.category, int(mujoco.mjtCatBit.mjCAT_DECOR))
        np.testing.assert_allclose(arrow.rgba, [1.0, 0.08, 0.04, 0.95])
        np.testing.assert_allclose(arrow.pos, point, atol=1e-6)
        self.assertAlmostEqual(float(arrow.size[2]), 1.8)
        np.testing.assert_allclose(arrow.mat[:, 2], [1.0, 0.0, 0.0], atol=1e-6)

    def test_target_marker_and_line_are_decorative(self):
        model = mujoco.MjModel.from_xml_path("small_scene.xml")
        scene = mujoco.MjvScene(model, maxgeom=100)

        draw_interaction_target(scene, np.zeros(3), np.array([1.0, 0.0, 0.0]), 2.0)

        self.assertEqual(scene.ngeom, 2)
        self.assertEqual(scene.geoms[0].type, int(mujoco.mjtGeom.mjGEOM_LINE))
        self.assertEqual(scene.geoms[1].type, int(mujoco.mjtGeom.mjGEOM_SPHERE))
        np.testing.assert_allclose(scene.geoms[1].pos, [1.0, 0.0, 0.0])

    def test_zero_force_does_not_add_scene_geometry(self):
        model = mujoco.MjModel.from_xml_path("small_scene.xml")
        scene = mujoco.MjvScene(model, maxgeom=100)

        draw_interaction_force(scene, np.zeros(3), np.zeros(3), 2.0)

        self.assertEqual(scene.ngeom, 0)


class WindowFrameTests(unittest.TestCase):
    def test_resize_edges_include_corners(self):
        self.assertEqual(window_resize_edges(1280, 800, 0, 799), ("left", "bottom"))
        self.assertEqual(window_resize_edges(1280, 800, 1279, 0), ("right", "top"))
        self.assertEqual(window_resize_edges(1280, 800, 1260, 780), ("right", "bottom"))
        self.assertEqual(window_resize_edges(1280, 800, 1260, 700), ())
        self.assertEqual(window_resize_edges(1280, 800, 1279, 700), ("right",))
        self.assertEqual(window_resize_edges(1280, 800, 640, 400), ())

    def test_resize_geometry_preserves_minimum_and_opposite_edges(self):
        self.assertEqual(
            resized_window_geometry(100, 200, 1280, 800, ("right", "bottom"), 40, 50),
            (100, 200, 1320, 850),
        )
        self.assertEqual(
            resized_window_geometry(100, 200, 1280, 800, ("left", "top"), 30, 20),
            (130, 220, 1250, 780),
        )
        self.assertEqual(
            resized_window_geometry(100, 200, 640, 480, ("left", "top"), 50, 50),
            (100, 200, 640, 480),
        )

    def test_resize_geometry_returns_glfw_compatible_integers(self):
        geometry = resized_window_geometry(
            100, 200, 1280, 800, ("left", "bottom"), 10.25, 20.75
        )
        self.assertTrue(all(isinstance(value, int) for value in geometry))
        self.assertEqual(geometry, (110, 200, 1270, 821))

    def test_resize_uses_drag_origin_when_window_size_updates_are_delayed(self):
        viewer = Viewer(str(Path(__file__).with_name("small_scene.xml")))
        viewer.window = object()
        viewer.inspector = SimpleNamespace(
            backend=SimpleNamespace(mouse_callback=lambda *_args: None)
        )
        viewer.resizing_window = True
        viewer.resize_edges = ("right", "bottom")
        viewer.resize_origin = (100, 100, 800, 600, 880, 680)
        viewer.last_cursor = (780, 580)

        with (
            patch("main.glfw.get_window_pos", return_value=(100, 100)),
            patch("main.glfw.get_window_size", return_value=(800, 600)),
            patch("main.glfw.set_window_size") as set_window_size,
        ):
            for total_x, total_y in ((50, 40), (100, 80), (150, 120)):
                viewer._on_cursor_pos(
                    viewer.window, 780 + total_x, 580 + total_y
                )

        self.assertEqual(
            set_window_size.call_args_list,
            [
                call(viewer.window, 850, 640),
                call(viewer.window, 900, 680),
                call(viewer.window, 950, 720),
            ],
        )

    def test_escape_requests_confirmation_without_closing_window(self):
        viewer = Viewer(str(Path(__file__).with_name("small_scene.xml")))
        exit_requests = []
        viewer.inspector = SimpleNamespace(
            backend=SimpleNamespace(keyboard_callback=lambda *_args: None),
            request_exit=lambda: exit_requests.append(True),
        )

        with (
            patch(
                "main.imgui.get_io",
                return_value=SimpleNamespace(want_capture_keyboard=False),
            ),
            patch("main.glfw.set_window_should_close") as set_should_close,
        ):
            viewer._on_key(object(), glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)

        self.assertEqual(exit_requests, [True])
        set_should_close.assert_not_called()


class ModelLoadingTests(unittest.TestCase):
    def test_cli_can_connect_to_orca_on_startup(self):
        args = parse_args(["--orca"])
        self.assertIsNone(args.xml)
        self.assertEqual(args.orca, "localhost:50051")
        args = parse_args(["--orca", "127.0.0.1:60051"])
        self.assertEqual(args.orca, "127.0.0.1:60051")
        args = parse_args(["model.xml"])
        self.assertEqual(args.xml, "model.xml")
        self.assertIsNone(args.orca)

    def test_empty_startup_uses_in_memory_model(self):
        viewer = Viewer()
        self.assertIsNone(viewer.xml_path)
        self.assertEqual(viewer.model_name, "未加载模型")
        self.assertEqual(viewer.model.nbody, 1)
        self.assertEqual(viewer.model.ngeom, 0)
        self.assertEqual(viewer.zoom_speed, 0.18)
        self.assertEqual(viewer.pan_speed, 6.0)
        self.assertEqual(viewer.near_clip_ratio, 0.3)
        self.assertEqual(viewer.environment_preset, "blue")
        np.testing.assert_array_equal(viewer.option.geomgroup, np.ones(6))
        self.assertEqual(viewer.xml_text, "")

    def test_xml_reload_replaces_model_and_resets_interaction(self):
        viewer = Viewer()
        viewer.running = True
        viewer.buttons.add(0)
        viewer.selection.body_id = 0
        viewer.selection.joint_id = 0
        xml_path = Path(__file__).with_name("small_scene.xml")
        viewer.load_xml(xml_path)
        self.assertEqual(viewer.xml_path, xml_path.resolve())
        self.assertEqual(viewer.model_name, "small_scene.xml")
        self.assertEqual(viewer.xml_text, xml_path.read_text(encoding="utf-8"))
        self.assertGreater(viewer.model.ngeom, 0)
        self.assertFalse(viewer.running)
        self.assertFalse(viewer.selection.valid)
        self.assertEqual(viewer.selection.joint_id, -1)
        self.assertFalse(viewer.buttons)

    def test_environment_backgrounds(self):
        self.assertEqual(ENVIRONMENT_BACKGROUNDS["blue"], (0.56, 0.74, 0.9, 1.0))
        self.assertEqual(ENVIRONMENT_BACKGROUNDS["black"], (0.0, 0.0, 0.0, 1.0))

    def test_invalid_xml_preserves_current_model(self):
        xml_path = Path(__file__).with_name("small_scene.xml")
        viewer = Viewer(xml_path)
        old_model = viewer.model
        old_data = viewer.data
        old_index = viewer.model_index
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "invalid.xml"
            invalid_path.write_text("<mujoco><invalid></mujoco>", encoding="utf-8")
            with self.assertRaises(ValueError):
                viewer.load_xml(invalid_path)
        self.assertIs(viewer.model, old_model)
        self.assertIs(viewer.data, old_data)
        self.assertIs(viewer.model_index, old_index)
        self.assertEqual(viewer.xml_path, xml_path.resolve())


class LocalInteractionTests(unittest.TestCase):
    def setUp(self):
        self.xml_path = Path(__file__).with_name("small_scene.xml")
        self.viewer = Viewer(self.xml_path)
        self.statuses = []
        self.viewer.inspector = SimpleNamespace(
            mode="view",
            local_running=False,
            orca_sync_enabled=False,
            orca_push_busy=False,
            set_source_status=lambda text, error=False: self.statuses.append(
                (text, error)
            ),
        )

    def test_modes_are_mutually_exclusive(self):
        with patch("main.threading.Thread") as thread_class:
            self.viewer.set_mode("interaction", "localhost:50051")
            self.assertTrue(self.viewer.interaction_enabled)
            self.assertTrue(self.viewer.running)
            self.assertFalse(self.viewer.orca_sync_enabled)
            self.assertEqual(self.viewer.inspector.mode, "interaction")
            thread_class.return_value.start.assert_called_once_with()

            self.viewer.set_mode("sync", "localhost:50051")
            self.assertFalse(self.viewer.interaction_enabled)
            self.assertFalse(self.viewer.running)
            self.assertTrue(self.viewer.orca_sync_enabled)
            self.assertEqual(self.viewer.inspector.mode, "sync")
            self.assertEqual(thread_class.return_value.start.call_count, 2)

            stop_event = self.viewer.orca_sync_stop
            self.viewer.set_mode("view", "localhost:50051")
            self.assertFalse(self.viewer.interaction_enabled)
            self.assertFalse(self.viewer.running)
            self.assertFalse(self.viewer.orca_sync_enabled)
            self.assertEqual(self.viewer.inspector.mode, "view")
            self.assertTrue(stop_event.is_set())

    def test_view_does_not_step_and_interaction_controls_local_simulation(self):
        self.viewer.inspector = None
        original_qvel = self.viewer.data.qvel.copy()
        self.viewer.running = True
        self.viewer.last_time -= 1.0
        self.viewer._step_simulation()
        np.testing.assert_array_equal(self.viewer.data.qvel, original_qvel)

        with patch("main.threading.Thread"):
            self.viewer.set_mode("interaction", "localhost:50051")
        self.viewer.toggle_interaction_running()
        self.assertFalse(self.viewer.running)
        self.viewer.step_interaction()
        self.assertGreater(self.viewer.data.time, 0.0)

        self.viewer.reset_interaction()
        self.assertEqual(self.viewer.data.time, 0.0)
        np.testing.assert_array_equal(self.viewer.data.qpos, self.viewer.model.qpos0)

    def test_only_ctrl_left_press_starts_body_force(self):
        self.viewer.interaction_enabled = True
        self.viewer.scene = object()
        self.viewer.inspector.backend = SimpleNamespace(
            mouse_button_callback=lambda *_args: None
        )
        self.viewer.inspector.reveal_body = lambda _body_id: None
        window = object()
        with (
            patch("main.imgui.get_io", return_value=SimpleNamespace(want_capture_mouse=False)),
            patch("main.glfw.get_window_attrib", return_value=0),
            patch("main.glfw.get_window_size", return_value=(800, 600)),
            patch("main.glfw.get_framebuffer_size", return_value=(800, 600)),
            patch("main.glfw.get_cursor_pos", return_value=(400.0, 300.0)),
            patch(
                "main.pick_geom",
                return_value=SimpleNamespace(
                    body_id=1, point=tuple(self.viewer.data.xpos[1])
                ),
            ) as pick,
        ):
            self.viewer._on_mouse_button(window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
            pick.assert_not_called()
            self.assertEqual(self.viewer.force_body_id, -1)

            self.viewer._on_mouse_button(window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
            self.viewer._on_mouse_button(
                window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, glfw.MOD_CONTROL
            )
            pick.assert_called_once()
            self.assertEqual(self.viewer.force_body_id, 1)

    def test_spring_drag_applies_force_and_torque_at_grasp_point(self):
        body_id = 1
        self.viewer.force_body_id = body_id
        self.viewer.force_local_point = np.array([0.05, 0.0, 0.0])
        self.viewer.force_start = (
            self.viewer.data.xpos[body_id] + self.viewer.force_local_point
        )
        self.viewer.force_target = self.viewer.force_start + np.array([0.1, 0.0, 0.0])

        self.viewer._update_interaction_force()

        np.testing.assert_allclose(self.viewer.force_point, self.viewer.force_start)
        np.testing.assert_allclose(
            self.viewer.data.xfrc_applied[body_id, :3], self.viewer.force_vector
        )
        expected_torque = np.cross(
            self.viewer.force_point - self.viewer.data.xipos[body_id],
            self.viewer.force_vector,
        )
        np.testing.assert_allclose(
            self.viewer.data.xfrc_applied[body_id, 3:], expected_torque
        )
        self.assertGreater(np.linalg.norm(self.viewer.force_vector), 0.0)

    def test_light_body_spring_drag_is_stable_and_converges(self):
        body_id = 1
        self.viewer.model.opt.gravity[:] = 0.0
        self.viewer.force_body_id = body_id
        self.viewer.force_local_point = np.zeros(3)
        self.viewer.force_start = self.viewer.data.xpos[body_id].copy()
        self.viewer.force_target = self.viewer.force_start + np.array([0.002, 0.0, 0.0])

        for _step in range(2000):
            self.viewer._update_interaction_force()
            mujoco.mj_step(self.viewer.model, self.viewer.data)

        self.assertTrue(np.all(np.isfinite(self.viewer.data.qpos)))
        self.assertLess(
            np.linalg.norm(self.viewer.data.xpos[body_id] - self.viewer.force_target),
            1e-5,
        )
        self.assertLess(np.linalg.norm(self.viewer.data.qvel), 1e-3)

    def test_external_force_is_cleared(self):
        self.viewer.force_body_id = 1
        self.viewer.force_vector = np.array([1.0, 2.0, 3.0])
        self.viewer.data.xfrc_applied[1, :3] = self.viewer.force_vector

        self.viewer._clear_external_force()

        np.testing.assert_array_equal(self.viewer.data.xfrc_applied[1], np.zeros(6))
        self.assertEqual(self.viewer.force_body_id, -1)
        np.testing.assert_array_equal(self.viewer.force_vector, np.zeros(3))


class JointInteractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        xml_path = Path(self.temporary_directory.name) / "joints.xml"
        xml_path.write_text(
            """
            <mujoco>
              <worldbody>
                <body name="hinge_body" pos="0 0 1">
                  <joint name="hinge" type="hinge" axis="0 0 1"/>
                  <geom type="sphere" size="0.1"/>
                </body>
                <body name="slide_body" pos="1 0 1">
                  <joint name="slide" type="slide" axis="1 0 0"/>
                  <geom type="sphere" size="0.1"/>
                </body>
                <body name="ball_body" pos="2 0 1">
                  <joint name="ball" type="ball"/>
                  <geom type="sphere" size="0.1"/>
                </body>
                <body name="free_body" pos="3 0 1">
                  <freejoint name="free"/>
                  <geom type="sphere" size="0.1"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            encoding="utf-8",
        )
        self.viewer = Viewer(xml_path)
        self.viewer.interaction_enabled = True
        self.viewer.running = True
        self.statuses = []
        self.viewer.inspector = SimpleNamespace(
            set_source_status=lambda text, error=False: self.statuses.append(
                (text, error)
            )
        )

    def joint_id(self, name):
        return mujoco.mj_name2id(
            self.viewer.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )

    def test_all_joint_types_set_qpos_normalize_quaternions_and_clear_velocity(self):
        self.viewer.data.qvel[:] = 0.5
        cases = (
            ("hinge", [0.4], [0.4]),
            ("slide", [0.2], [0.2]),
            ("ball", [2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            (
                "free",
                [1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            ),
        )
        for name, values, expected in cases:
            with self.subTest(joint=name):
                joint_id = self.joint_id(name)
                address = int(self.viewer.model.jnt_qposadr[joint_id])
                self.assertTrue(self.viewer.set_joint_qpos(joint_id, values))
                np.testing.assert_allclose(
                    self.viewer.data.qpos[address : address + len(expected)], expected
                )
                self.assertFalse(self.viewer.running)
        np.testing.assert_array_equal(self.viewer.data.qvel, np.zeros(self.viewer.model.nv))

    def test_invalid_joint_values_are_rejected_without_changing_state(self):
        original_qpos = self.viewer.data.qpos.copy()
        self.assertFalse(self.viewer.set_joint_qpos(self.joint_id("hinge"), [1.0, 2.0]))
        self.assertFalse(self.viewer.set_joint_qpos(self.joint_id("slide"), [math.nan]))
        self.assertFalse(
            self.viewer.set_joint_qpos(self.joint_id("ball"), [0.0, 0.0, 0.0, 0.0])
        )
        np.testing.assert_array_equal(self.viewer.data.qpos, original_qpos)
        self.assertTrue(all(error for _text, error in self.statuses))

    def test_joint_setting_requires_interaction_mode(self):
        self.viewer.interaction_enabled = False
        self.assertFalse(self.viewer.set_joint_qpos(self.joint_id("hinge"), [0.4]))


class OrcaPushTests(unittest.TestCase):
    def setUp(self):
        self.xml_path = Path(__file__).with_name("small_scene.xml")
        self.viewer = Viewer(self.xml_path)
        self.viewer.interaction_enabled = True
        self.statuses = []
        self.viewer.inspector = SimpleNamespace(
            orca_push_busy=False,
            set_source_status=lambda text, error=False: self.statuses.append(
                (text, error)
            ),
        )

    def test_push_starts_one_stream_and_keeps_only_latest_snapshot(self):
        self.viewer.data.qpos[2] = 0.02
        self.viewer.data.time = 1.25
        with patch("main.threading.Thread") as thread_class:
            self.viewer.push_local_state(" localhost:50051 ")
            self.viewer.data.qpos[2] = 0.04
            self.viewer.data.time = 2.5
            self.viewer.push_local_state("localhost:50051")

            thread_class.assert_called_once()
            arguments = thread_class.call_args.kwargs["args"]
            self.assertEqual(arguments[0], self.viewer.orca_push_request_id)
            self.assertEqual(arguments[1], "localhost:50051")
            self.assertIs(arguments[2], self.viewer.orca_push_states)
            self.assertIs(arguments[3], self.viewer.orca_push_stop)
            thread_class.return_value.start.assert_called_once_with()

        request_id, qpos, time = self.viewer.orca_push_states.get_nowait()
        self.assertEqual(request_id, self.viewer.orca_push_request_id)
        self.assertEqual(qpos[2], 0.04)
        self.assertEqual(time, 2.5)
        self.assertTrue(self.viewer.orca_push_busy)
        self.assertTrue(self.viewer.inspector.orca_push_busy)

    def test_simulation_loop_automatically_queues_current_state(self):
        with patch("main.threading.Thread"):
            self.viewer.push_local_state("localhost:50051")
        self.viewer.orca_push_states.get_nowait()
        self.viewer.data.qpos[2] = 0.03
        self.viewer.data.time = 3.5
        self.viewer.running = False
        self.viewer.inspector = None

        self.viewer._step_simulation()

        _request_id, qpos, time = self.viewer.orca_push_states.get_nowait()
        self.assertEqual(qpos[2], 0.03)
        self.assertEqual(time, 3.5)

    def test_worker_publishes_success(self):
        state_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        request_id = self.viewer.orca_push_request_id
        qpos = self.viewer.data.qpos.copy()
        state_queue.put((request_id, qpos, 1.25))

        async def fake_stream(_address, received_queue, received_stop, on_pushed):
            pushed_id, _qpos, _time = received_queue.get_nowait()
            on_pushed(pushed_id)
            received_stop.set()

        with patch("main.stream_orca_states", new_callable=AsyncMock) as stream:
            stream.side_effect = fake_stream
            self.viewer._orca_push_worker(
                request_id, "localhost:50051", state_queue, stop_event
            )

        stream.assert_awaited_once()
        self.assertEqual(self.viewer.orca_push_results.get_nowait(), (request_id, None))

    def test_success_failure_and_stale_results(self):
        request_id = self.viewer.orca_push_request_id
        self.viewer.orca_push_busy = True
        self.viewer.inspector.orca_push_busy = True
        self.viewer.orca_push_results.put((request_id - 1, None))
        self.viewer._process_orca_push_results()
        self.assertTrue(self.viewer.orca_push_busy)
        self.assertEqual(self.statuses, [])

        self.viewer.orca_push_results.put((request_id, None))
        self.viewer._process_orca_push_results()
        self.assertFalse(self.viewer.orca_push_busy)
        self.assertTrue(self.viewer.orca_push_connected)
        self.assertFalse(self.viewer.inspector.orca_push_busy)
        self.assertEqual(
            self.statuses[-1], ("本地主仿真（正在推送到 OrcaLab）", False)
        )

        stop_event = threading.Event()
        self.viewer.orca_push_stop = stop_event
        self.viewer.orca_push_states = queue.Queue(maxsize=1)
        self.viewer.orca_push_results.put((request_id, "write failed"))
        self.viewer._process_orca_push_results()
        self.assertFalse(self.viewer.orca_push_busy)
        self.assertFalse(self.viewer.orca_push_connected)
        self.assertTrue(stop_event.is_set())
        self.assertIsNone(self.viewer.orca_push_stop)
        self.assertIsNone(self.viewer.orca_push_states)
        self.assertEqual(
            self.statuses[-1], ("OrcaLab 实时同步失败：write failed", True)
        )

    def test_push_validation_does_not_start_worker(self):
        with patch("main.threading.Thread") as thread_class:
            self.viewer.push_local_state("  ")
            self.viewer.interaction_enabled = False
            self.viewer.push_local_state("localhost:50051")
        thread_class.assert_not_called()
        self.assertFalse(self.viewer.orca_push_busy)
        self.assertEqual(self.statuses, [("OrcaLab 地址不能为空", True)])

    def test_model_reload_invalidates_and_stops_pending_push(self):
        self.viewer.inspector = None
        self.viewer.orca_push_busy = True
        self.viewer.orca_push_stop = threading.Event()
        self.viewer.orca_push_states = queue.Queue(maxsize=1)
        stop_event = self.viewer.orca_push_stop
        request_id = self.viewer.orca_push_request_id

        self.viewer.load_xml(self.xml_path)
        self.viewer.orca_push_results.put((request_id, None))
        self.viewer._process_orca_push_results()

        self.assertGreater(self.viewer.orca_push_request_id, request_id)
        self.assertTrue(stop_event.is_set())
        self.assertIsNone(self.viewer.orca_push_stop)
        self.assertIsNone(self.viewer.orca_push_states)
        self.assertFalse(self.viewer.orca_push_busy)
        self.assertFalse(self.viewer.interaction_enabled)


class OrcaSyncTests(unittest.TestCase):
    def setUp(self):
        self.xml_path = Path(__file__).with_name("small_scene.xml")

    def test_start_and_stop_control_worker_lifecycle(self):
        viewer = Viewer(self.xml_path)
        viewer.inspector = SimpleNamespace(
            orca_sync_enabled=False,
            set_source_status=lambda _text, _error=False: None,
        )
        viewer.running = True
        with patch("main.threading.Thread") as thread_class:
            viewer.start_orca_sync(" localhost:50051 ")
            stop_event = viewer.orca_sync_stop

            self.assertTrue(viewer.orca_sync_enabled)
            self.assertTrue(viewer.inspector.orca_sync_enabled)
            self.assertFalse(viewer.running)
            self.assertIsNotNone(stop_event)
            self.assertEqual(
                thread_class.call_args.kwargs["args"][1], "localhost:50051"
            )
            thread_class.return_value.start.assert_called_once_with()

        viewer.stop_orca_sync()
        self.assertFalse(viewer.orca_sync_enabled)
        self.assertFalse(viewer.inspector.orca_sync_enabled)
        self.assertTrue(stop_event.is_set())
        self.assertIsNone(viewer.orca_sync_stop)

    def test_remote_snapshot_updates_data_and_forward_kinematics(self):
        viewer = Viewer(self.xml_path)
        statuses = []
        viewer.inspector = SimpleNamespace(
            orca_sync_enabled=True,
            set_source_status=lambda text, error=False: statuses.append(
                (text, error)
            ),
        )
        viewer.orca_sync_enabled = True
        qpos = viewer.data.qpos.copy()
        qpos[2] += 0.004
        qvel = np.linspace(0.1, 0.6, viewer.model.nv)
        qacc = np.zeros(viewer.model.nv)
        viewer.orca_sync_results.put_nowait(
            (viewer.orca_sync_request_id, (qpos, qvel, qacc), None)
        )

        viewer._process_orca_sync_results()

        np.testing.assert_array_equal(viewer.data.qpos, qpos)
        np.testing.assert_array_equal(viewer.data.qvel, qvel)
        np.testing.assert_array_equal(viewer.data.xpos[1], qpos[:3])
        self.assertTrue(viewer.orca_sync_connected)
        self.assertEqual(statuses[-1], ("OrcaLab 主仿真（正在拉取状态）", False))

    def test_stale_snapshot_is_ignored(self):
        viewer = Viewer(self.xml_path)
        original_qpos = viewer.data.qpos.copy()
        changed_qpos = original_qpos.copy()
        changed_qpos[2] += 0.01
        viewer.orca_sync_enabled = True
        viewer.orca_sync_results.put_nowait(
            (
                viewer.orca_sync_request_id - 1,
                (
                    changed_qpos,
                    np.zeros(viewer.model.nv),
                    np.zeros(viewer.model.nv),
                ),
                None,
            )
        )

        viewer._process_orca_sync_results()

        np.testing.assert_array_equal(viewer.data.qpos, original_qpos)
        self.assertFalse(viewer.orca_sync_connected)

    def test_mismatched_remote_model_stops_sync_without_changing_data(self):
        viewer = Viewer(self.xml_path)
        statuses = []
        viewer.inspector = SimpleNamespace(
            orca_sync_enabled=True,
            set_source_status=lambda text, error=False: statuses.append(
                (text, error)
            ),
        )
        viewer.orca_sync_enabled = True
        original_qpos = viewer.data.qpos.copy()
        viewer.orca_sync_results.put_nowait(
            (
                viewer.orca_sync_request_id,
                (
                    np.zeros(viewer.model.nq + 1),
                    np.zeros(viewer.model.nv),
                    np.zeros(viewer.model.nv),
                ),
                None,
            )
        )

        viewer._process_orca_sync_results()

        np.testing.assert_array_equal(viewer.data.qpos, original_qpos)
        self.assertFalse(viewer.orca_sync_enabled)
        self.assertFalse(viewer.inspector.orca_sync_enabled)
        self.assertTrue(statuses[-1][1])
        self.assertIn("状态维度不一致", statuses[-1][0])

    def test_sync_mode_never_steps_local_simulation(self):
        viewer = Viewer(self.xml_path)
        viewer.running = True
        viewer.orca_sync_enabled = True
        viewer.last_time -= 1.0
        original_qvel = viewer.data.qvel.copy()

        viewer._step_simulation()

        np.testing.assert_array_equal(viewer.data.qvel, original_qvel)


if __name__ == "__main__":
    unittest.main()
