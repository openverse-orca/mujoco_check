from pathlib import Path

import glfw
import mujoco
from imgui_bundle import imgui, portable_file_dialogs
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from inspection import body_focus, project_to_window


FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def _vector(values):
    return "(" + ", ".join(f"{float(value):.5g}" for value in values) + ")"


def _joint_type(joint_type):
    return mujoco.mjtJoint(int(joint_type)).name.removeprefix("mjJNT_").lower()


def _geom_type(geom_type):
    return mujoco.mjtGeom(int(geom_type)).name.removeprefix("mjGEOM_").lower()


def _begin_properties(table_id):
    flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_saved_settings
    if not imgui.begin_table(table_id, 2, flags):
        return False
    imgui.table_setup_column("属性", imgui.TableColumnFlags_.width_fixed, 130.0)
    imgui.table_setup_column("值", imgui.TableColumnFlags_.width_stretch)
    return True


def _property_row(label, value):
    imgui.table_next_row()
    imgui.table_set_column_index(0)
    imgui.text_disabled(label)
    imgui.table_set_column_index(1)
    imgui.text_wrapped(str(value))


class InspectorUI:
    panel_width = 560
    tree_width = 420

    def __init__(self, window, model, data, model_index, option):
        self.window = window
        self.model = model
        self.data = data
        self.index = model_index
        self.option = option
        self.backend = None
        self.label_font = None
        self.left_panel_width = float(self.tree_width)
        self.right_panel_width = float(self.panel_width)
        self.open_bodies = set()
        self.scroll_body = None
        self.focus_body = None
        self.file_dialog = None
        self.on_open_xml = None
        self.on_orca_pull = None
        self.orca_address = "localhost:50051"
        self.orca_busy = False
        self.source_status = "请选择本地 XML 或连接 OrcaLab"
        self.source_error = False

        imgui.create_context()
        try:
            io = imgui.get_io()
            for font_path in FONT_CANDIDATES:
                if Path(font_path).is_file():
                    io.font_default = io.fonts.add_font_from_file_ttf(font_path, 36.0)
                    self.label_font = io.fonts.add_font_from_file_ttf(font_path, 20.0)
                    break
            self.backend = GlfwRenderer(window, attach_callbacks=False)
        except Exception:
            imgui.destroy_context()
            raise

    def begin_frame(self):
        self.backend.process_inputs()
        imgui.new_frame()

    def end_frame(self):
        imgui.render()
        self.backend.render(imgui.get_draw_data())

    def reveal_body(self, body_id):
        if body_id < 0:
            return
        self.open_bodies.update(self.index.ancestors(body_id))
        self.scroll_body = body_id

    def consume_focus_request(self):
        body_id = self.focus_body
        self.focus_body = None
        return body_id

    def rebind_model(self, model, data, model_index):
        self.model = model
        self.data = data
        self.index = model_index
        self.open_bodies.clear()
        self.scroll_body = None
        self.focus_body = None

    def set_source_status(self, text, error=False):
        self.source_status = text
        self.source_error = error

    def _poll_file_dialog(self):
        if self.file_dialog is None or not self.file_dialog.ready(0):
            return
        paths = self.file_dialog.result()
        self.file_dialog = None
        if paths and self.on_open_xml is not None:
            self.on_open_xml(paths[0])

    def _draw_model_sources(self):
        if not imgui.collapsing_header(
            "模型来源", flags=imgui.TreeNodeFlags_.default_open
        ):
            return
        imgui.begin_disabled(self.file_dialog is not None)
        if imgui.button("打开 XML"):
            self.file_dialog = portable_file_dialogs.open_file(
                "打开 MuJoCo XML",
                str(Path.home()),
                ["XML 文件", "*.xml"],
            )
        imgui.end_disabled()

        button_width = imgui.calc_text_size("拉取 XML").x + 24.0
        spacing = imgui.get_style().item_spacing.x
        address_width = imgui.get_content_region_avail().x - button_width - spacing
        imgui.set_next_item_width(max(address_width, 80.0))
        _changed, self.orca_address = imgui.input_text(
            "##OrcaLab 地址", self.orca_address
        )
        imgui.same_line()
        imgui.begin_disabled(self.orca_busy)
        if imgui.button("拉取 XML") and self.on_orca_pull is not None:
            self.on_orca_pull(self.orca_address)
        imgui.end_disabled()
        if self.source_error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.35, 0.3, 1.0))
            imgui.text_wrapped(self.source_status)
            imgui.pop_style_color()
        else:
            imgui.text_wrapped(self.source_status)
        imgui.spacing()

    def _draw_splitter(self, name, x, height, flags):
        splitter_flags = (
            flags
            | imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse
        )
        imgui.set_next_window_pos(imgui.ImVec2(x - 5, 0), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(10, height), imgui.Cond_.always)
        imgui.push_style_var(imgui.StyleVar_.window_min_size, imgui.ImVec2(1, 1))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))
        imgui.begin(name, flags=splitter_flags)
        try:
            imgui.invisible_button(f"{name}_handle", imgui.get_content_region_avail())
            hovered = imgui.is_item_hovered()
            active = imgui.is_item_active()
            if hovered or active:
                imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
            return imgui.get_io().mouse_delta.x if active else 0.0
        finally:
            imgui.end()
            imgui.pop_style_var(2)

    def draw(self, selection, zoom_speed, pan_speed):
        self._poll_file_dialog()
        window_width, window_height = glfw.get_window_size(self.window)
        panel_width = min(self.right_panel_width, window_width)
        tree_width = min(self.left_panel_width, window_width - panel_width)
        flags = (
            imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_saved_settings
        )

        if tree_width > 0:
            imgui.set_next_window_pos(imgui.ImVec2(0, 0), imgui.Cond_.always)
            imgui.set_next_window_size(
                imgui.ImVec2(tree_width, window_height), imgui.Cond_.always
            )
            imgui.begin("Body 层级", flags=flags)
            try:
                self._draw_model_sources()
                self._draw_body_tree(selection)
            finally:
                imgui.end()

        imgui.set_next_window_pos(
            imgui.ImVec2(window_width - panel_width, 0), imgui.Cond_.always
        )
        imgui.set_next_window_size(
            imgui.ImVec2(panel_width, window_height), imgui.Cond_.always
        )
        imgui.begin("检查面板", flags=flags)
        try:
            if imgui.begin_tab_bar("inspector_tabs"):
                try:
                    object_tab, _open = imgui.begin_tab_item("对象详情")
                    if object_tab:
                        imgui.begin_child("object_details", imgui.ImVec2(0, 0))
                        try:
                            if selection.valid:
                                self._draw_details(selection)
                            else:
                                imgui.text_wrapped("在视口或左侧 Body 树中选择对象")
                        finally:
                            imgui.end_child()
                        imgui.end_tab_item()

                    display_tab, _open = imgui.begin_tab_item("显示设置")
                    if display_tab:
                        self._draw_groups()
                        self._draw_visual_flags()
                        imgui.separator_text("交互速度")
                        imgui.text("滚轮速度 (%/格)")
                        imgui.set_next_item_width(-1)
                        changed, zoom_percent = imgui.slider_float(
                            "##滚轮速度", zoom_speed * 100.0, 0.2, 30.0, "%.1f"
                        )
                        if changed:
                            zoom_speed = zoom_percent / 100.0
                        imgui.text("平移速度 (倍)")
                        imgui.set_next_item_width(-1)
                        changed, pan_speed = imgui.slider_float(
                            "##平移速度", pan_speed, 0.1, 10.0, "%.1f"
                        )
                        imgui.text_disabled("接近场景时自动精细缩放和平移")
                        imgui.end_tab_item()
                finally:
                    imgui.end_tab_bar()
        finally:
            imgui.end()

        if tree_width > 0:
            left_delta = self._draw_splitter("左侧分隔条", tree_width, window_height, flags)
            max_left = max(0.0, window_width - panel_width - 120.0)
            min_left = min(220.0, max_left)
            self.left_panel_width = min(
                max_left, max(min_left, self.left_panel_width + left_delta)
            )
        right_delta = self._draw_splitter(
            "右侧分隔条", window_width - panel_width - 5.0, window_height, flags
        )
        max_right = max(0.0, window_width - tree_width - 120.0)
        min_right = min(300.0, max_right)
        self.right_panel_width = min(
            max_right, max(min_right, self.right_panel_width - right_delta)
        )
        self.open_bodies.clear()
        self.scroll_body = None
        return selection, zoom_speed, pan_speed

    def draw_object_label(self, selection, scene, framebuffer_size):
        if not selection.valid:
            return
        if selection.geom_id >= 0:
            point = self.data.geom_xpos[selection.geom_id]
            text = (
                f"{self.index.geom_name(selection.geom_id)} · "
                f"{self.index.body_name(selection.body_id)}"
            )
            group = int(self.model.geom_group[selection.geom_id])
            if 0 <= group < 6 and not self.option.geomgroup[group]:
                text = f"[已隐藏] {text}"
        else:
            point, _radius = body_focus(
                self.model, self.data, self.index, selection.body_id
            )
            text = self.index.body_name(selection.body_id)

        position = project_to_window(
            point, scene, framebuffer_size, glfw.get_window_size(self.window)
        )
        if position is None:
            return
        imgui.set_next_window_pos(
            imgui.ImVec2(position[0], position[1] - 16.0),
            imgui.Cond_.always,
            imgui.ImVec2(0.5, 1.0),
        )
        imgui.set_next_window_bg_alpha(0.78)
        label_flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_mouse_inputs
            | imgui.WindowFlags_.no_nav
        )
        if self.label_font is not None:
            imgui.push_font(self.label_font)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(10, 6))
        try:
            imgui.begin("选中对象##object_label", flags=label_flags)
            try:
                imgui.text(text)
            finally:
                imgui.end()
        finally:
            imgui.pop_style_var()
            if self.label_font is not None:
                imgui.pop_font()

    def _draw_details(self, selection):
        model = self.model
        data = self.data
        body_id = selection.body_id
        default_open = imgui.TreeNodeFlags_.default_open

        if imgui.collapsing_header("Body", flags=default_open):
            if _begin_properties("body_properties"):
                _property_row("ID", body_id)
                _property_row("名称", self.index.body_name(body_id))
                parent_id = int(model.body_parentid[body_id])
                parent_name = (
                    self.index.body_name(parent_id)
                    if body_id > 0 and parent_id >= 0
                    else "无"
                )
                _property_row("父 Body", parent_name)
                _property_row("本地 pos", _vector(model.body_pos[body_id]))
                _property_row("本地 quat", _vector(model.body_quat[body_id]))
                _property_row("当前 xpos", _vector(data.xpos[body_id]))
                _property_row("当前 xquat", _vector(data.xquat[body_id]))
                _property_row("Geom 数", len(self.index.body_geoms(body_id)))
                _property_row("Joint 数", len(self.index.body_joints(body_id)))
                imgui.end_table()

        if selection.geom_id >= 0 and imgui.collapsing_header("Geom", flags=default_open):
            geom_id = selection.geom_id
            if _begin_properties("geom_properties"):
                _property_row("ID", geom_id)
                _property_row("名称", self.index.geom_name(geom_id))
                _property_row("所属 Body", self.index.body_name(body_id))
                _property_row("类型", _geom_type(model.geom_type[geom_id]))
                _property_row("Group", int(model.geom_group[geom_id]))
                _property_row("size", _vector(model.geom_size[geom_id]))
                _property_row("pos", _vector(model.geom_pos[geom_id]))
                _property_row("quat", _vector(model.geom_quat[geom_id]))
                _property_row("rgba", _vector(model.geom_rgba[geom_id]))
                imgui.end_table()

        if imgui.collapsing_header("Joints", flags=default_open):
            joints = self.index.body_joints(body_id)
            if not joints:
                imgui.text_disabled("当前 Body 无 Joint")
            for joint_id in joints:
                joint_flags = default_open if len(joints) == 1 else 0
                title = f"[{joint_id}] {self.index.joint_name(joint_id)}"
                if imgui.collapsing_header(title, flags=joint_flags):
                    if _begin_properties(f"joint_properties_{joint_id}"):
                        _property_row("类型", _joint_type(model.jnt_type[joint_id]))
                        _property_row("pos", _vector(model.jnt_pos[joint_id]))
                        _property_row("axis", _vector(model.jnt_axis[joint_id]))
                        _property_row("qposadr", int(model.jnt_qposadr[joint_id]))
                        _property_row("dofadr", int(model.jnt_dofadr[joint_id]))
                        if model.jnt_limited[joint_id]:
                            _property_row("range", _vector(model.jnt_range[joint_id]))
                        imgui.end_table()

    def _draw_visual_flags(self):
        imgui.separator_text("诊断显示")
        items = [
            ("关节", mujoco.mjtVisFlag.mjVIS_JOINT),
            ("质心", mujoco.mjtVisFlag.mjVIS_COM),
            ("接触点", mujoco.mjtVisFlag.mjVIS_CONTACTPOINT),
            ("接触力", mujoco.mjtVisFlag.mjVIS_CONTACTFORCE),
            ("透明显示", mujoco.mjtVisFlag.mjVIS_TRANSPARENT),
        ]
        table_flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
        )
        if imgui.begin_table("visual_flags", 2, table_flags):
            for index, (label, flag) in enumerate(items):
                if index % 2 == 0:
                    imgui.table_next_row()
                imgui.table_set_column_index(index % 2)
                flag_index = int(flag)
                changed, enabled = imgui.checkbox(
                    label, bool(self.option.flags[flag_index])
                )
                if changed:
                    self.option.flags[flag_index] = int(enabled)
            imgui.end_table()

        frames = [
            ("不显示", mujoco.mjtFrame.mjFRAME_NONE),
            ("Body 坐标系", mujoco.mjtFrame.mjFRAME_BODY),
            ("Geom 坐标系", mujoco.mjtFrame.mjFRAME_GEOM),
            ("Site 坐标系", mujoco.mjtFrame.mjFRAME_SITE),
            ("Camera 坐标系", mujoco.mjtFrame.mjFRAME_CAMERA),
            ("Light 坐标系", mujoco.mjtFrame.mjFRAME_LIGHT),
            ("接触坐标系", mujoco.mjtFrame.mjFRAME_CONTACT),
            ("World 坐标系", mujoco.mjtFrame.mjFRAME_WORLD),
        ]
        current_label = next(
            label for label, frame in frames if int(frame) == int(self.option.frame)
        )
        imgui.text("坐标系")
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##坐标系", current_label):
            for label, frame in frames:
                _selected, clicked = imgui.selectable(
                    label, int(self.option.frame) == int(frame)
                )
                if clicked:
                    self.option.frame = frame
            imgui.end_combo()

        imgui.text("关节粗细")
        imgui.set_next_item_width(-1)
        changed, joint_width = imgui.slider_float(
            "##关节粗细", float(self.model.vis.scale.jointwidth), 0.01, 1.0, "%.2f"
        )
        if changed:
            self.model.vis.scale.jointwidth = joint_width
        imgui.text("坐标轴粗细")
        imgui.set_next_item_width(-1)
        changed, frame_width = imgui.slider_float(
            "##坐标轴粗细", float(self.model.vis.scale.framewidth), 0.01, 1.0, "%.2f"
        )
        if changed:
            self.model.vis.scale.framewidth = frame_width

    def _draw_body_tree(self, selection):
        imgui.begin_child(
            "body_tree",
            imgui.ImVec2(0, 0),
            imgui.ChildFlags_.borders,
            imgui.WindowFlags_.horizontal_scrollbar,
        )
        try:
            self._draw_body_node(0, selection)
        finally:
            imgui.end_child()

    def _draw_body_node(self, body_id, selection):
        children = self.index.children[body_id]
        geom_ids = self.index.body_geoms(body_id)
        joint_ids = self.index.body_joints(body_id)
        flags = imgui.TreeNodeFlags_.span_avail_width
        if body_id == 0:
            flags |= imgui.TreeNodeFlags_.default_open
        if not children and not geom_ids and not joint_ids:
            flags |= imgui.TreeNodeFlags_.leaf
        if selection.body_id == body_id:
            flags |= imgui.TreeNodeFlags_.selected
        if body_id in self.open_bodies:
            imgui.set_next_item_open(True, imgui.Cond_.always)
        label = (
            f"{self.index.body_name(body_id)}  "
            f"G{len(geom_ids)} J{len(joint_ids)}##body{body_id}"
        )
        imgui.push_style_color(imgui.Col_.text, (0.45, 0.75, 1.0, 1.0))
        try:
            opened = imgui.tree_node_ex(label, flags)
        finally:
            imgui.pop_style_color()
        if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
            selection.body_id = body_id
            selection.geom_id = -1
            self.focus_body = body_id
        if self.scroll_body == body_id:
            imgui.set_scroll_here_y(0.5)
        if opened:
            try:
                for geom_id in geom_ids:
                    imgui.push_style_color(
                        imgui.Col_.text, (0.45, 0.9, 0.55, 1.0)
                    )
                    try:
                        _selected, clicked = imgui.selectable(
                            f"G  {self.index.geom_name(geom_id)}##outline_geom{geom_id}",
                            selection.geom_id == geom_id,
                        )
                    finally:
                        imgui.pop_style_color()
                    if clicked:
                        selection.body_id = body_id
                        selection.geom_id = geom_id
                for joint_id in joint_ids:
                    imgui.push_style_color(
                        imgui.Col_.text, (1.0, 0.75, 0.3, 1.0)
                    )
                    try:
                        _selected, clicked = imgui.selectable(
                            f"J  {self.index.joint_name(joint_id)}##outline_joint{joint_id}",
                            False,
                        )
                    finally:
                        imgui.pop_style_color()
                    if clicked:
                        selection.body_id = body_id
                        selection.geom_id = -1
                for child_id in children:
                    self._draw_body_node(child_id, selection)
            finally:
                imgui.tree_pop()

    def _draw_groups(self):
        imgui.separator_text("Geom Groups")
        table_flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
        )
        if imgui.begin_table("geom_groups", 2, table_flags):
            for group in range(6):
                if group % 2 == 0:
                    imgui.table_next_row()
                imgui.table_set_column_index(group % 2)
                imgui.begin_disabled(self.index.group_counts[group] == 0)
                try:
                    changed, visible = imgui.checkbox(
                        f"Group {group} ({self.index.group_counts[group]})",
                        bool(self.option.geomgroup[group]),
                    )
                    if changed:
                        self.option.geomgroup[group] = int(visible)
                finally:
                    imgui.end_disabled()
            imgui.end_table()
        if imgui.button("全部显示"):
            for group in range(6):
                self.option.geomgroup[group] = 1
        imgui.same_line()
        if imgui.button("全部隐藏"):
            for group in range(6):
                self.option.geomgroup[group] = 0

    def shutdown(self):
        if self.file_dialog is not None:
            self.file_dialog.kill()
            self.file_dialog = None
        if self.backend is not None:
            self.backend.shutdown()
            self.backend = None
        if imgui.get_current_context():
            imgui.destroy_context()
