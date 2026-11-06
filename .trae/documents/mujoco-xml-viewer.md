# MuJoCo XML 精细缩放查看器

## Context

当前 MuJoCo 默认查看器在观察小尺度模型时，滚轮缩放步进过大，容易一次拉近到极限，无法仔细检查小物体。项目现有 `main.py` 只是占位入口，已有 uv、Python 3.12 虚拟环境，但尚未安装 MuJoCo、GLFW 和 NumPy。本次将实现一个轻量、可直接传入 XML 的 3D 查看器，重点提供尺度自适应取景和稳定、细粒度的缩放。

## 实施方案

### 1. 配置运行依赖

修改 `pyproject.toml`，在现有项目配置中加入 `mujoco`、`glfw`、`numpy`，通过 `uv sync` 安装到已有 `.venv`，不引入 Qt 或其他 GUI 框架。

### 2. 实现单文件查看器

改造 `main.py`，采用 MuJoCo 原生 Python 渲染 API 和 GLFW 窗口：

- 命令行接收 XML 路径，使用 `MjModel.from_xml_path()` 加载，确保 XML 内相对资源路径正常解析。
- 创建 `MjData`、`MjvCamera`、`MjvScene`、`MjrContext`，处理 framebuffer 尺寸变化和 OpenGL 资源清理。
- 默认暂停并执行 `mj_forward`，便于直接观察静态 XML；播放时按 `model.opt.timestep` 仿真。
- 使用 `model.stat.center` 和 `model.stat.extent` 自动设置初始观察中心、方位角、仰角及距离，同时考虑窗口宽高比和相机视场角。
- 对无效或零尺度提供安全回退，初始距离及后续缩放都使用与场景尺度成比例的上下限，不写死米制最小距离。

### 3. 自定义相机交互

将相机数学计算写成 `main.py` 内的独立小函数，便于无窗口测试：

- 滚轮采用 `distance *= exp(-0.04 * scroll_y)`，每格约缩放 4%，替代默认的大幅缩放；支持触控板的连续滚动值，并在每帧限幅。
- 左键拖动旋转，仰角限制在 ±89°；右键拖动平移，平移速度随当前距离和视口高度变化。
- 根据当前相机距离动态调整渲染近、远裁剪面，避免近距离观察小物体时模型被裁掉。
- 快捷键：`Space` 播放/暂停，`.` 单步，`R` 重置仿真，`Home` 恢复全景视角，`Esc/Q` 退出。
- 窗口标题显示 XML 文件名、播放状态和滚轮缩放比例，不增加额外控制面板。

### 4. 提供小场景验证样例和相机测试

- 新建 `small_scene.xml`：包含毫米级几何体及简单地面，用于稳定复现和验证小物体观察效果。
- 新建 `test_viewer.py`：使用标准库 `unittest` 测试滚轮每格比例、极小/极大场景尺度、距离上下限、初始取景、旋转约束和平移尺度；测试不创建 GLFW 窗口。

## 关键文件

- `main.py`：替换占位代码，实现 XML 加载、渲染、仿真和相机交互。
- `pyproject.toml`：仅增加运行依赖。
- `small_scene.xml`：小尺度演示模型。
- `test_viewer.py`：无 GUI 相机逻辑测试。

## 验证

1. 执行 `uv sync` 安装依赖。
2. 执行 `uv run python -B -m unittest -v test_viewer.py`，确认相机计算测试通过。
3. 执行 `uv run python main.py small_scene.xml`，实际打开窗口检查：
   - 初始场景完整居中，小物体可见。
   - 连续滚轮缩放平滑，每格约 4%，不会突然放大到极限。
   - 极近观察时物体不被近裁剪面截断，拉远和拉近边界有效。
   - 左键旋转、右键平移、窗口缩放均正常。
   - 播放、暂停、单步、重置、恢复视角和退出正常。
4. 检查不存在或损坏的 XML 会输出清晰错误并返回非零退出码。
