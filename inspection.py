from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Selection:
    body_id: int = -1
    geom_id: int = -1
    joint_id: int = -1
    point: tuple | None = None

    @property
    def valid(self):
        return self.body_id >= 0


class ModelIndex:
    def __init__(self, model):
        self.model = model
        self.children = tuple(
            tuple(
                child_id
                for child_id in np.flatnonzero(model.body_parentid == body_id).tolist()
                if child_id != body_id
            )
            for body_id in range(model.nbody)
        )
        counts = np.bincount(model.geom_group, minlength=6)
        self.group_counts = tuple(int(count) for count in counts[:6])

    def name(self, obj_type, obj_id, prefix):
        name = mujoco.mj_id2name(self.model, obj_type, obj_id)
        return name if name else f"{prefix} #{obj_id}"

    def body_name(self, body_id):
        return self.name(mujoco.mjtObj.mjOBJ_BODY, body_id, "Body")

    def geom_name(self, geom_id):
        return self.name(mujoco.mjtObj.mjOBJ_GEOM, geom_id, "Geom")

    def joint_name(self, joint_id):
        return self.name(mujoco.mjtObj.mjOBJ_JOINT, joint_id, "Joint")

    def body_path(self, body_id):
        return " / ".join(self.body_name(part) for part in self.ancestors(body_id))

    def ancestors(self, body_id):
        lineage = []
        while body_id >= 0:
            lineage.append(body_id)
            parent_id = int(self.model.body_parentid[body_id])
            if parent_id == body_id:
                break
            body_id = parent_id
        return tuple(reversed(lineage))

    def body_geoms(self, body_id):
        address = int(self.model.body_geomadr[body_id])
        count = int(self.model.body_geomnum[body_id])
        if address < 0 or count == 0:
            return ()
        return tuple(range(address, address + count))

    def body_joints(self, body_id):
        address = int(self.model.body_jntadr[body_id])
        count = int(self.model.body_jntnum[body_id])
        if address < 0 or count == 0:
            return ()
        return tuple(range(address, address + count))


def body_focus(model, data, model_index, body_id):
    geom_ids = model_index.body_geoms(body_id)
    if not geom_ids:
        return data.xpos[body_id].copy(), max(float(model.stat.extent) * 0.02, 1e-6)

    centers = [data.geom_xpos[geom_id] for geom_id in geom_ids]
    center = np.mean(centers, axis=0)
    radius = max(
        np.linalg.norm(data.geom_xpos[geom_id] - center)
        + float(model.geom_rbound[geom_id])
        for geom_id in geom_ids
    )
    return center, max(float(radius), 1e-6)


def joint_focus(model, data, joint_id):
    center = data.xanchor[joint_id].copy()
    extent = float(model.stat.extent)
    visual_radius = float(model.vis.scale.jointwidth) * 8.0
    radius = max(min(visual_radius, extent * 0.05), 1e-6)
    return center, radius


def project_to_window(point, scene, framebuffer_size, window_size):
    framebuffer_width, framebuffer_height = framebuffer_size
    window_width, window_height = window_size
    if min(framebuffer_width, framebuffer_height, window_width, window_height) <= 0:
        return None

    camera = scene.camera[0]
    forward = np.asarray(camera.forward, dtype=float)
    up = np.asarray(camera.up, dtype=float)
    right = np.cross(forward, up)
    relative = np.asarray(point, dtype=float) - np.asarray(camera.pos, dtype=float)
    depth = float(np.dot(relative, forward))
    near = float(camera.frustum_near)
    if depth <= 0.0 or near <= 0.0:
        return None

    if camera.orthographic:
        half_height = float(camera.frustum_top)
        half_width = float(camera.frustum_width)
    else:
        if depth < near:
            return None
        half_height = float(camera.frustum_top) * depth / near
        half_width = float(camera.frustum_width) * depth / near
    if half_height <= 0.0 or half_width <= 0.0:
        return None

    ndc_x = float(np.dot(relative, right)) / half_width
    ndc_y = float(np.dot(relative, up)) / half_height
    if not (-1.0 <= ndc_x <= 1.0 and -1.0 <= ndc_y <= 1.0):
        return None
    return (
        (ndc_x + 1.0) * 0.5 * window_width,
        (1.0 - ndc_y) * 0.5 * window_height,
    )


def relative_cursor_position(x, y, window_width, window_height):
    if window_width <= 0 or window_height <= 0:
        return None
    return float(x) / window_width, 1.0 - float(y) / window_height


def pick_geom(
    model,
    data,
    option,
    scene,
    cursor,
    window_size,
    framebuffer_size,
):
    window_width, window_height = window_size
    framebuffer_width, framebuffer_height = framebuffer_size
    relative = relative_cursor_position(cursor[0], cursor[1], window_width, window_height)
    if (
        relative is None
        or framebuffer_width <= 0
        or framebuffer_height <= 0
    ):
        return Selection()

    selected_point = np.zeros(3)
    geom_id = np.full(1, -1, dtype=np.int32)
    flex_id = np.full(1, -1, dtype=np.int32)
    skin_id = np.full(1, -1, dtype=np.int32)
    body_id = mujoco.mjv_select(
        model,
        data,
        option,
        framebuffer_width / framebuffer_height,
        relative[0],
        relative[1],
        scene,
        selected_point,
        geom_id,
        flex_id,
        skin_id,
    )
    if body_id < 0 or geom_id[0] < 0:
        return Selection()
    return Selection(
        int(body_id), int(geom_id[0]), point=tuple(float(value) for value in selected_point)
    )


def apply_highlight(scene, selection, model_index):
    if not selection.valid:
        return
    if selection.joint_id >= 0:
        for index in range(scene.ngeom):
            geom = scene.geoms[index]
            if (
                geom.objtype == mujoco.mjtObj.mjOBJ_JOINT
                and int(geom.objid) == selection.joint_id
            ):
                alpha = geom.rgba[3]
                geom.rgba[:] = (1.0, 0.75, 0.1, alpha)
                geom.emission = 0.65
        return
    descendant_targets = set()
    if selection.geom_id >= 0:
        targets = {selection.geom_id}
    else:
        targets = set(model_index.body_geoms(selection.body_id))
        pending = list(model_index.children[selection.body_id])
        while pending:
            body_id = pending.pop()
            descendant_targets.update(model_index.body_geoms(body_id))
            pending.extend(model_index.children[body_id])
    if not targets and not descendant_targets:
        return

    highlight_color = np.array([1.0, 0.75, 0.1])
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        geom_id = int(geom.objid)
        if geom_id in targets:
            alpha = geom.rgba[3]
            geom.rgba[:] = (1.0, 0.75, 0.1, alpha)
            geom.emission = 0.65
        elif geom_id in descendant_targets:
            geom.rgba[:3] = 0.65 * geom.rgba[:3] + 0.35 * highlight_color
            geom.emission = max(float(geom.emission), 0.18)
