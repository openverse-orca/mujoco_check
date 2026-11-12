import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewRect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))

    def contains(self, point):
        if self.width <= 0.0 or self.height <= 0.0:
            return False
        point_x, point_y = point
        return (
            self.x <= point_x < self.x + self.width
            and self.y <= point_y < self.y + self.height
        )


def grid_view_rects(count, x, y, width, height):
    if not isinstance(count, int) or count <= 0:
        return []
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return []
    if width <= 0.0 or height <= 0.0:
        return []

    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    x_edges = [round(x + width * column / columns) for column in range(columns + 1)]
    y_edges = [round(y + height * row / rows) for row in range(rows + 1)]

    rects = []
    for row in range(rows):
        for column in range(columns):
            if len(rects) == count:
                return rects
            left = x_edges[column]
            right = x_edges[column + 1]
            top = y_edges[row]
            bottom = y_edges[row + 1]
            rects.append(ViewRect(left, top, right - left, bottom - top))
    return rects


def framebuffer_view_rect(rect, window_size, framebuffer_size):
    window_width, window_height = window_size
    framebuffer_width, framebuffer_height = framebuffer_size
    values = (
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        window_width,
        window_height,
        framebuffer_width,
        framebuffer_height,
    )
    if not all(math.isfinite(value) for value in values):
        return 0, 0, 0, 0
    if (
        rect.width <= 0.0
        or rect.height <= 0.0
        or window_width <= 0.0
        or window_height <= 0.0
        or framebuffer_width <= 0.0
        or framebuffer_height <= 0.0
    ):
        return 0, 0, 0, 0

    scale_x = framebuffer_width / window_width
    scale_y = framebuffer_height / window_height
    left = round(rect.x * scale_x)
    right = round((rect.x + rect.width) * scale_x)
    bottom = round(framebuffer_height - (rect.y + rect.height) * scale_y)
    top = round(framebuffer_height - rect.y * scale_y)
    return left, bottom, right - left, top - bottom


def view_index_at(rects, point):
    for index, rect in enumerate(rects):
        if rect is not None and rect.contains(point):
            return index
    return None
