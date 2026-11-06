# MuJoCo 模型检查工具：核心检查闭环

## Context

现有查看器已经解决小场景精细缩放、窗口焦点和投影视场角问题。下一步目标不是继续增加单纯观看功能，而是形成用于排查模型的基本工作流：**点击对象 → 高亮定位 → 查看 Body/Geom/Joint 详情 → 通过 Body 层级和 Geom Group 控制显示**。

首版只实现上述检查闭环。自动质量/惯性检查、关节诊断、接触穿插诊断、XML 编辑和搜索均暂不实现。

## 一、界面与交互

### 右侧检查面板

使用 `imgui-bundle` 在现有 GLFW 窗口内绘制约 320px 宽的右侧面板，采用紧凑单列布局，不引入 Qt，不改变现有 MuJoCo 视口和相机宽高比。面板包含：

1. **选择摘要**
   - 当前 Body 路径和 Geom 名称/ID。
   - 无选择时显示“点击模型查看详情”。
   - 选中 Geom 所在 Group 被隐藏时显示隐藏状态，但保留详情。

2. **对象详情**
   - Body：ID、名称、父 Body、本地 pos/quat、当前 xpos/xquat、Geom/Joint 数量。
   - Geom：ID、名称、所属 Body、类型、Group、size、pos/quat、rgba。
   - Joint：列出当前 Body 的全部 Joint，显示 ID、名称、类型、pos/axis、qposadr/dofadr，有限位时显示 range。
   - 未命名对象使用 `Body #id`、`Geom #id`、`Joint #id`，不显示空白标签。

3. **Body 层级树**
   - 以 world 为唯一根，按 `body_parentid` 展示父子关系。
   - 节点附带 Geom/Joint 数量，父子和兄弟层级清晰。
   - 点击 Body 节点后选中该 Body 并高亮其全部可见 Geom。
   - 从 3D 点选后自动展开祖先并滚动到对应 Body。

4. **Geom Groups**
   - 固定显示 Group 0–5、每组 Geom 数量和复选框。
   - 提供全部显示/全部隐藏；空 Group 保留但禁用。
   - 只控制 `MjvOption.geomgroup`，不修改 XML、模型物理或其他对象类别。

### 鼠标与键盘规则

- 左键单击且位移小于 4 个逻辑像素：拾取 Geom；点击空白清除选择。
- 左键位移超过阈值：沿用现有旋转，不触发拾取。
- 右键拖动：沿用现有平移。
- 滚轮：保持当前约 4%/格的镜头推近/拉远，不修改已验证的相机算法。
- 鼠标或键盘被面板捕获时，不操作相机、不拾取、不触发仿真快捷键。
- 场景拖动进入面板后仍完成当前手势；释放或失焦时无条件清理手势状态。

## 二、模块设计

### `main.py`

保留现有相机函数、仿真和 GLFW 生命周期，新增：

- 唯一的 `Selection(body_id, geom_id)` 状态；Body-only 选择时 `geom_id=-1`，无选择时两者均为 `-1`。
- 左键 click/drag 手势状态和待处理拾取请求。
- 在回调中手动转发 ImGui 事件，并根据 capture 状态分流场景输入。
- 组装模型索引、检查面板、拾取和高亮。
- 在 OpenGL 上下文仍有效时先关闭 UI，再释放 MuJoCo/GLFW。

### `inspection.py`

不依赖 GLFW/ImGui，集中实现可测试的 MuJoCo 规则：

- `ModelIndex`：加载后一次性缓存 Body 子树、名称回退、Body 路径、Group 计数和祖先链。
- 通过 `body_geomadr/body_geomnum`、`body_jntadr/body_jntnum` 获取 Body 的 Geom 和 Joint。
- 将窗口坐标转换为 `mjv_select` 参数：`relx=x/window_width`、`rely=1-y/window_height`，aspect 使用实际 framebuffer 宽高比。
- 调用 `mjv_select` 并映射结果；`bodyid=0` 是合法 world 命中，只有 `-1` 表示未命中。
- 每帧在 `mjv_updateScene` 后通过 `objtype + objid` 匹配并修改 `scene.geoms` 的 RGB/emission，保留 alpha，不写回 `model.geom_rgba`。

### `inspector_ui.py`

隔离 `imgui-bundle`：

- 创建和销毁 ImGui context 与 GLFW/OpenGL 后端。
- 后端使用 `attach_callbacks=False`，避免覆盖现有 GLFW 回调。
- 绘制选择摘要、详情、Body 树和 Group 控件。
- 处理面板锚定、窗口 resize、树展开/滚动请求和资源清理。

## 三、帧处理顺序

每帧保持以下顺序，避免运动对象拾取使用旧 Scene：

1. `glfw.poll_events()`，转发 UI 输入并更新场景手势；拾取仅记录请求。
2. 执行现有仿真步进。
3. 创建 ImGui frame 并构建面板，应用树选择和 Group 修改。
4. 使用同一个 `self.option` 调用 `mjv_updateScene()`。
5. 保留现有 near/far 与视锥边界同步修正，确保视场角不变。
6. 基于当前 Scene、Camera 和 Option 处理待拾取请求。
7. 对选中的 Geom 或 Body 应用高亮。
8. `mjr_render()`，随后绘制 ImGui，最后交换缓冲区。

隐藏 Group 同时从 Scene 和拾取中排除；选中对象被隐藏后保留 Selection 和详情，重新显示时恢复高亮。

## 四、实施步骤

1. 在 `pyproject.toml` 增加 `imgui-bundle` 并更新 `uv.lock`，保留现有 MuJoCo 和相机代码。
2. 实现 `inspection.py` 的模型索引、选择映射、Group 统计及高亮，先完成无窗口测试。
3. 接入 ImGui 面板和输入分流，先验证 Group 显隐及 UI/场景输入互不影响。
4. 加入 click/drag 状态机、`mjv_select`、3D 高亮和 Body/Geom/Joint 详情。
5. 加入 Body 树双向联动、隐藏选择状态、world/未命名/多 Joint 等边界处理。
6. 给 `small_scene.xml` 的现有 Geom/Joint 补充名称和 Group 0/1/2 标记，仅修改元数据，不改变模型物理，便于手工验收。

## 五、关键文件

- `main.py`：扩展现有 Viewer，保留相机和渲染主干。
- `inspection.py`：新增模型索引、拾取、选择和高亮逻辑。
- `inspector_ui.py`：新增 ImGui 检查面板和后端生命周期。
- `test_viewer.py`：保留七项相机测试，补充可纯函数化的手势测试。
- `test_inspection.py`：新增模型索引、拾取映射、Group 和高亮测试。
- `small_scene.xml`：补充演示用名称和 Group 标记。
- `pyproject.toml` / `uv.lock`：增加并锁定 UI 依赖。

## 六、验证

### 自动测试

- 原七项相机测试全部通过，缩放比例和固定视场角不回归。
- Body 树以 world 为根，嵌套 Body、world Geom、未命名对象和空 Group 处理正确。
- Geom 选择同步所属 Body，Body 选择清空 Geom，点击空白清除选择。
- 高亮只修改目标 Scene Geom，保留 alpha；Body-only 高亮不波及其他 Body。
- 隐藏 Group 不可被拾取，恢复显示后高亮恢复。
- 坐标转换覆盖 Y 翻转、高 DPI 和非法尺寸。
- 小位移点击与拖动旋转严格区分，释放和失焦不会留下卡住状态。

### 实际窗口验收

- 点选 free cube、world 上的 ground/sphere，详情、Joint 和 Body 路径正确，高亮位置准确。
- 左键拖动只旋转，右键平移和滚轮精细缩放保持现有体验。
- Group 复选框立即控制对应 Geom，隐藏对象不能再次被点中。
- Body 树选择与 3D 点选双向联动。
- 面板内点击、滚动和键盘操作不影响场景；窗口 resize、最小化恢复正常。
- 播放、暂停、单步和重置后选择持续跟踪同一对象。
- 退出及初始化失败路径无 UI/OpenGL 资源清理错误。
