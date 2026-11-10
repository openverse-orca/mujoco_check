import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import glfw
import mujoco
import numpy as np

from main import (
    Viewer,
    MIN_CLOSE_ZOOM_FACTOR,
    ENVIRONMENT_BACKGROUNDS,
    ZOOM_SPEED,
    camera_limits,
    focus_camera,
    is_click,
    keyboard_move_camera,
    orbit_camera,
    pan_camera,
    parse_args,
    reset_camera,
    resized_window_geometry,
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


class WindowFrameTests(unittest.TestCase):
    def test_resize_edges_include_corners(self):
        self.assertEqual(window_resize_edges(1280, 800, 0, 799), ("left", "bottom"))
        self.assertEqual(window_resize_edges(1280, 800, 1279, 0), ("right", "top"))
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
        self.assertEqual(viewer.near_clip_ratio, 0.001)
        self.assertEqual(viewer.environment_preset, "blue")
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
        self.assertEqual(statuses[-1], ("正在同步 OrcaLab 仿真", False))

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
