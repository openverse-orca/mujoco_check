# GUI 本地 XML 与 OrcaLab 模型加载

## Context

当前查看器必须通过命令行位置参数指定 XML，且 `Viewer` 初始化后模型、索引、Scene、MjrContext 和检查面板都绑定同一模型。用户需要保留这种启动方式，同时支持无参数启动后在 GUI 中打开本地 XML，或通过已安装的 `orca-gym` 从 OrcaLab 拉取当前 XML。

已核实环境为 `orca-gym 26.10.1`。真实接口是 `grpc.aio.insecure_channel` → `GrpcServiceStub` → `OrcaGymLocal.load_model_xml()`，返回本地 XML 绝对路径并下载 mesh/hfield，不需要初始化或控制 OrcaLab 仿真。`imgui-bundle` 已包含 `portable_file_dialogs`，无需新增依赖。

## 一、使用方式

保留：

```bash
uv run python main.py path/to/model.xml
```

新增：

```bash
uv run python main.py
```

无参数时使用 `MjModel.from_xml_string("<mujoco/>")` 创建纯内存空模型，不生成伪造的磁盘 XML。窗口标题显示“未加载模型”，左侧提供模型来源入口，右侧继续使用现有详情/显示设置布局。

左侧 Body 树上方增加紧凑的“模型来源”折叠区：

- “打开 XML”按钮：弹出原生文件选择框，只允许选择 `.xml`；取消不改变当前模型。
- OrcaLab 地址输入框：默认 `localhost:50051`。
- “拉取 XML”按钮：拉取期间禁用重复操作并显示进度；成功、连接失败、XML 无效等状态在同一区域显示。
- Body 树占据剩余空间，右侧对象详情和显示设置不增加来源控件。

## 二、OrcaLab 拉取

新增 `model_sources.py` 隔离外部接口：

- 按本机真实 API 创建 `grpc.aio` channel、`GrpcServiceStub` 和 `OrcaGymLocal`，只调用 `load_model_xml()`，不调用 `init_simulation()`。
- 每次请求使用独立临时资产目录，避免 `~/.orcagym/tmp` 中同名缓存导致旧 XML/旧 mesh 被复用；模型加载结束后清理由本次请求创建的目录，不删除共享缓存。
- 在守护工作线程的独立 asyncio 事件循环中执行连接和下载，`finally` 关闭 channel，避免阻塞 GLFW/ImGui。
- 结果通过线程安全队列交回 GUI 线程。每次请求带递增序号，退出或已有更新请求时丢弃迟到结果；工作线程不触碰 MuJoCo、GLFW 或 OpenGL 对象。
- 为网络请求设置有限超时，连接失败只显示错误，不影响已经打开的模型。

## 三、事务式模型切换

在 `main.py` 增加统一的 `Viewer.load_xml(path)`，本地文件与 OrcaLab 结果都走这一入口：

1. 校验路径并创建候选 `MjModel/MjData`，执行 `mj_forward`，创建新 `ModelIndex`。
2. 在当前 OpenGL 上下文中创建候选 `MjvScene` 和 `MjrContext`。
3. 全部成功后才交换模型引用；重绑 `InspectorUI.model/data/index`，保留 GLFW 窗口、ImGui、字体和回调。
4. 清空选择、待拾取、鼠标手势、Body 展开/滚动/聚焦状态，暂停仿真并清零计时累积，按新模型统计重置相机和窗口标题。
5. 释放旧 `MjrContext`，候选创建失败时释放半成品并保留旧模型、旧 Scene/context，确保当前场景仍可操作。

`xml_path=None` 时空模型使用零中心和现有安全 extent 回退；空 Scene 正常渲染，不启用仿真运行。OrcaLab 临时目录只在对应模型加载结束后清理。

## 四、关键文件

- `main.py`：可选 CLI 参数、空模型启动、统一加载、GL 资源替换、后台结果调度和退出清理。
- `inspector_ui.py`：左侧来源区、非阻塞文件对话框、地址/进度/错误展示、模型切换后的引用与状态重置。
- `model_sources.py`：新增 OrcaLab 异步拉取、独立临时目录和线程结果通信。
- `test_model_sources.py`：新增 RPC 流程、channel 关闭、超时/异常、临时目录及迟到结果测试。
- `test_viewer.py` / `test_inspection.py`：保留现有相机、投影、选择测试，补充可选 CLI、空模型和模型切换状态测试。
- `pyproject.toml` / `uv.lock`：已有 `orca-gym` 和对话框能力，不修改依赖。

## 五、验证

自动测试：

- 有参 CLI 仍加载同一 XML；无参创建空模型且标题、相机、空 Body 树正常。
- 文件对话框取消、非法 XML 和 OrcaLab 失败均保留旧模型及 GL 资源。
- 连续切换本地/OrcaLab 模型后索引、详情、Group、拾取和高亮引用一致，旧 context 仅释放一次。
- mock gRPC 验证真实调用链、地址、超时、channel 关闭、临时目录清理和异常传播。
- 原 23 项测试全部通过，旋转、平移、滚轮和面板布局不回归。

实际窗口验收：

- `uv run python main.py` 无参启动，分别通过文件选择和 OrcaLab 拉取 `small_scene.xml` 或 OrcaLab 当前场景。
- OrcaLab 先正常启动场景，再用默认或实际地址拉取；含 mesh 的模型完整显示，重复拉取能反映同名 XML 的更新。
- 下载期间窗口、相机和面板保持响应；失败提示清楚，旧模型仍可操作。
- `uv run python main.py small_scene.xml` 保持原行为；连续切换和退出无 OpenGL/线程清理错误。
