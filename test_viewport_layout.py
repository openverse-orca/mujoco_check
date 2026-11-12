import unittest

from viewport_layout import (
    ViewRect,
    framebuffer_view_rect,
    grid_view_rects,
    view_index_at,
)


class GridViewRectTests(unittest.TestCase):
    def test_layout_sizes_order_and_no_overlap(self):
        cases = (
            (1, 1, 1),
            (2, 2, 1),
            (3, 2, 2),
            (4, 2, 2),
            (5, 3, 2),
            (9, 3, 3),
        )
        for count, columns, rows in cases:
            with self.subTest(count=count):
                rects = grid_view_rects(count, 10, 20, 90, 60)
                self.assertEqual(len(rects), count)
                cell_width = 90 / columns
                cell_height = 60 / rows
                for index, rect in enumerate(rects):
                    row, column = divmod(index, columns)
                    self.assertEqual(
                        rect,
                        ViewRect(
                            10 + column * cell_width,
                            20 + row * cell_height,
                            cell_width,
                            cell_height,
                        ),
                    )
                for index, first in enumerate(rects):
                    for second in rects[index + 1 :]:
                        overlap_width = min(
                            first.x + first.width, second.x + second.width
                        ) - max(first.x, second.x)
                        overlap_height = min(
                            first.y + first.height, second.y + second.height
                        ) - max(first.y, second.y)
                        self.assertFalse(overlap_width > 0 and overlap_height > 0)

    def test_uneven_layout_uses_shared_integer_edges(self):
        rects = grid_view_rects(5, 7, 11, 92, 61)
        for rect in rects:
            for edge in (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height):
                self.assertEqual(edge, round(edge))
        self.assertEqual(rects[0].x + rects[0].width, rects[1].x)
        self.assertEqual(rects[1].x + rects[1].width, rects[2].x)
        self.assertEqual(rects[0].y + rects[0].height, rects[3].y)
        self.assertEqual(rects[1].y + rects[1].height, rects[4].y)

    def test_zero_or_invalid_layout_area_is_empty(self):
        self.assertEqual(grid_view_rects(0, 0, 0, 100, 100), [])
        self.assertEqual(grid_view_rects(-1, 0, 0, 100, 100), [])
        self.assertEqual(grid_view_rects(2, 0, 0, 0, 100), [])
        self.assertEqual(grid_view_rects(2, 0, 0, 100, 0), [])
        self.assertEqual(grid_view_rects(2, 0, 0, -100, 100), [])


class ViewRectTests(unittest.TestCase):
    def test_contains_uses_half_open_positive_rectangles(self):
        rect = ViewRect(10, 20, 30, 40)
        self.assertTrue(all(isinstance(value, float) for value in vars(rect).values()))
        self.assertTrue(rect.contains((10, 20)))
        self.assertTrue(rect.contains((39.999, 59.999)))
        self.assertFalse(rect.contains((40, 20)))
        self.assertFalse(rect.contains((10, 60)))
        self.assertFalse(rect.contains((9.999, 20)))
        self.assertFalse(rect.contains((10, 19.999)))
        self.assertFalse(ViewRect(10, 20, 0, 40).contains((10, 20)))
        self.assertFalse(ViewRect(10, 20, 30, -40).contains((10, 20)))

    def test_view_index_at_uses_layout_order_and_returns_none_outside(self):
        rects = grid_view_rects(2, 10, 20, 90, 60)
        self.assertEqual(view_index_at(rects, (10, 20)), 0)
        self.assertEqual(view_index_at(rects, (55, 20)), 1)
        self.assertEqual(view_index_at(rects, (100, 20)), None)
        self.assertIsNone(view_index_at(rects, (9, 20)))


class FramebufferViewRectTests(unittest.TestCase):
    def test_hidpi_scaling_and_y_flip(self):
        result = framebuffer_view_rect(
            ViewRect(10, 20, 30, 40),
            window_size=(100, 200),
            framebuffer_size=(300, 500),
        )
        self.assertEqual(result, (30, 350, 90, 100))
        self.assertTrue(all(isinstance(value, int) for value in result))

    def test_shared_edges_round_consistently(self):
        first = framebuffer_view_rect(
            ViewRect(10.1, 20, 20.2, 30), (100, 100), (250, 300)
        )
        second = framebuffer_view_rect(
            ViewRect(30.3, 20, 20, 30), (100, 100), (250, 300)
        )
        self.assertEqual(first[0] + first[2], second[0])

    def test_zero_sizes_are_safe(self):
        rect = ViewRect(10, 20, 30, 40)
        self.assertEqual(
            framebuffer_view_rect(ViewRect(10, 20, 0, 40), (100, 200), (300, 500)),
            (0, 0, 0, 0),
        )
        self.assertEqual(
            framebuffer_view_rect(rect, (0, 200), (300, 500)), (0, 0, 0, 0)
        )
        self.assertEqual(
            framebuffer_view_rect(rect, (100, 200), (300, 0)), (0, 0, 0, 0)
        )


if __name__ == "__main__":
    unittest.main()
