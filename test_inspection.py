import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from inspection import (
    ModelIndex,
    Selection,
    apply_highlight,
    body_focus,
    pick_geom,
    project_to_window,
    relative_cursor_position,
)


class ModelIndexTests(unittest.TestCase):
    def setUp(self):
        xml_path = Path(__file__).with_name("small_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.index = ModelIndex(self.model)

    def test_body_tree_paths_and_owned_objects(self):
        self.assertEqual(self.index.children[0], (1,))
        self.assertEqual(self.index.children[1], ())
        self.assertEqual(self.index.ancestors(1), (0, 1))
        self.assertEqual(self.index.body_path(1), "world / free_cube")
        self.assertEqual(self.index.body_geoms(0), (0, 1, 2))
        self.assertEqual(self.index.body_geoms(1), (3,))
        self.assertEqual(self.index.body_joints(0), ())
        self.assertEqual(self.index.body_joints(1), (0,))

    def test_group_counts_include_empty_groups(self):
        self.assertEqual(self.index.group_counts, (1, 1, 2, 0, 0, 0))

    def test_nested_and_unnamed_bodies_have_stable_labels(self):
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body><geom size="0.1"/><body name="child"/></body>
              </worldbody>
            </mujoco>
            """
        )
        index = ModelIndex(model)
        self.assertEqual(index.body_name(1), "Body #1")
        self.assertEqual(index.geom_name(0), "Geom #0")
        self.assertEqual(index.children[0], (1,))
        self.assertEqual(index.children[1], (2,))
        self.assertEqual(index.body_path(2), "world / Body #1 / child")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        xml_path = Path(__file__).with_name("small_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.scene = mujoco.MjvScene(self.model, maxgeom=100)
        self.index = ModelIndex(self.model)
        self.cube_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_visual"
        )
        self.ground_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        self.world_geom_ids = self.index.body_geoms(0)
        mujoco.mj_forward(self.model, self.data)

    def _update_scene(self):
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

    def _scene_geom(self, geom_id):
        for index in range(self.scene.ngeom):
            geom = self.scene.geoms[index]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == geom_id:
                return geom
        raise AssertionError(f"Geom {geom_id} 不在 Scene 中")

    def test_body_focus_uses_current_geom_position_and_bounds(self):
        center, radius = body_focus(self.model, self.data, self.index, 1)
        np.testing.assert_array_equal(center, self.data.geom_xpos[self.cube_id])
        self.assertAlmostEqual(radius, float(self.model.geom_rbound[self.cube_id]))

    def test_projection_tracks_object_in_logical_window_coordinates(self):
        camera = SimpleNamespace(
            pos=np.array([0.0, 0.0, -2.0]),
            forward=np.array([0.0, 0.0, 1.0]),
            up=np.array([0.0, 1.0, 0.0]),
            frustum_near=1.0,
            frustum_top=0.5,
            frustum_width=0.5,
            orthographic=0,
        )
        scene = SimpleNamespace(camera=[camera])
        center = project_to_window([0, 0, 0], scene, (800, 600), (400, 300))
        np.testing.assert_allclose(center, (200, 150))
        right_edge = project_to_window([-1, 0, 0], scene, (800, 600), (400, 300))
        np.testing.assert_allclose(right_edge, (400, 150))
        self.assertIsNone(
            project_to_window([0, 0, -3], scene, (800, 600), (400, 300))
        )

    def test_relative_cursor_position_flips_y(self):
        self.assertEqual(relative_cursor_position(20, 30, 100, 200), (0.2, 0.85))
        self.assertIsNone(relative_cursor_position(20, 30, 0, 200))

    @patch("inspection.mujoco.mjv_select")
    def test_pick_uses_framebuffer_aspect_and_accepts_world_body(self, select):
        def select_world(_model, _data, _option, aspect, relx, rely, *_args):
            self.assertEqual(aspect, 2.0)
            self.assertEqual((relx, rely), (0.25, 0.75))
            _args[2][0] = 0
            return 0

        select.side_effect = select_world
        result = pick_geom(
            self.model,
            self.data,
            self.option,
            self.scene,
            (25, 50),
            (100, 200),
            (400, 200),
        )
        self.assertEqual(result, Selection(body_id=0, geom_id=0))

    @patch("inspection.mujoco.mjv_select")
    def test_invalid_size_and_blank_click_clear_selection(self, select):
        result = pick_geom(
            self.model,
            self.data,
            self.option,
            self.scene,
            (25, 50),
            (100, 0),
            (400, 200),
        )
        self.assertEqual(result, Selection())
        select.assert_not_called()

        select.return_value = -1
        result = pick_geom(
            self.model,
            self.data,
            self.option,
            self.scene,
            (25, 50),
            (100, 200),
            (400, 200),
        )
        self.assertEqual(result, Selection())

    def test_geom_highlight_preserves_alpha_and_model_colors(self):
        self._update_scene()
        original_alpha = self._scene_geom(self.cube_id).rgba[3]
        original_ground = self._scene_geom(self.ground_id).rgba.copy()
        original_model = self.model.geom_rgba.copy()

        apply_highlight(self.scene, Selection(1, self.cube_id), self.index)

        np.testing.assert_allclose(
            self._scene_geom(self.cube_id).rgba, (1.0, 0.75, 0.1, original_alpha)
        )
        self.assertAlmostEqual(self._scene_geom(self.cube_id).emission, 0.65)
        np.testing.assert_array_equal(
            self._scene_geom(self.ground_id).rgba, original_ground
        )
        np.testing.assert_array_equal(self.model.geom_rgba, original_model)

    def test_body_highlight_only_affects_owned_visible_geoms(self):
        self._update_scene()
        original_cube = self._scene_geom(self.cube_id).rgba.copy()

        apply_highlight(self.scene, Selection(0), self.index)

        for geom_id in self.world_geom_ids:
            self.assertAlmostEqual(self._scene_geom(geom_id).emission, 0.65)
        np.testing.assert_array_equal(self._scene_geom(self.cube_id).rgba, original_cube)
        self.assertEqual(self._scene_geom(self.cube_id).emission, 0.0)

    def test_hidden_group_is_absent_from_scene_and_highlight(self):
        self.option.geomgroup[1] = 0
        self._update_scene()
        visible_ids = {
            int(self.scene.geoms[index].objid)
            for index in range(self.scene.ngeom)
            if self.scene.geoms[index].objtype == mujoco.mjtObj.mjOBJ_GEOM
        }
        self.assertNotIn(self.cube_id, visible_ids)
        apply_highlight(self.scene, Selection(1, self.cube_id), self.index)
        self.assertTrue(
            all(self.scene.geoms[index].emission == 0.0 for index in range(self.scene.ngeom))
        )


if __name__ == "__main__":
    unittest.main()
