# MuJoCo 模型检查器

一个用于查看 MuJoCo XML 模型、检查 Body / Joint / Geom，并与 OrcaLab 同步显示状态的桌面工具。

## 安装

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

在项目目录执行：

```bash
uv tool install --python 3.12 .
```

安装完成后，可以在任意目录直接运行 `mtool`，不需要进入项目目录，也不需要使用 `uv run`。

更新已安装的工具：

```bash
uv tool install --python 3.12 --force .
```

## 启动

启动后手动打开 XML：

```bash
mtool
```

启动时直接打开本地模型：

```bash
mtool /path/to/model.xml
```

启动时连接 OrcaLab 并拉取模型：

```bash
mtool --orca
```

OrcaLab 不在默认地址 `localhost:50051` 时：

```bash
mtool --orca 主机地址:50051
```

也可以直接在项目目录通过 uv 运行：

```bash
uv run mtool /path/to/model.xml
```

## 多视图

在右侧“显示设置”中点击“+ 添加视图”即可增加视图，点击对应视图的“关闭”即可移除。所有视图共享同一模型和仿真状态，各自拥有独立相机，并自动等分中间显示区域；至少保留一个视图。点击或滚动某个视图后，该视图成为键盘移动、Home 和对象聚焦的活动视图。

## 运行模式

- **模型查看**：只查看本地模型，不同步仿真状态。
- **OrcaLab 主仿真（拉取）**：从 OrcaLab 实时拉取状态，在本地显示。
- **本地主仿真（推送）**：在本地运行和交互，并将显示状态实时推送到 OrcaLab。

本地主仿真模式下，可在“交互控制”中调整 Joint，或按住 `Ctrl + 鼠标左键` 拖动 Body 施加弹簧阻尼力。

## 常用操作

| 操作 | 功能 |
| --- | --- |
| 鼠标左键单击 | 选择对象 |
| 鼠标左键拖动 | 旋转相机 |
| 鼠标右键拖动 | 平移相机 |
| 鼠标滚轮 | 缩放 |
| WASD | 前后左右移动相机 |
| Q / E | 下降 / 上升 |
| Space | 本地仿真播放 / 暂停 |
| `.` | 本地仿真单步 |
| R | 重置本地仿真 |
| Home | 恢复全景视角 |
| Ctrl + 鼠标左键拖动 | 对 Body 施加外力 |
| Esc | 打开退出确认窗口 |
