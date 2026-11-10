# Debug Session: window-resize-delta
- **Status**: [OPEN]
- **progress_status**: collecting
- **Issue**: 无边框窗口右下角拖动很远，但窗口尺寸只增加少量；预期窗口右下边缘跟随鼠标总位移。
- **Evidence Source**: HTTP Debug Server（待启动）
- **Evidence Directory**: /tmp/mujoco-check-window-resize-delta
- **Log File**: /tmp/mujoco-check-window-resize-delta/trae-debug-log-window-resize-delta.ndjson

## Reproduction Steps
1. 创建真实的隐藏 GLFW 无边框窗口。
2. 在右下角命中区域按下鼠标，记录窗口几何和鼠标屏幕坐标。
3. 分多次向右下移动，每次调用真实窗口回调并读取新几何。
4. 比较累计鼠标位移与窗口尺寸增量。

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | 使用窗口局部坐标逐帧增量，`set_window_size` 后的坐标回调使累计位移丢失 | High | Low | 鼠标累计屏幕位移明显大于窗口尺寸增量，局部 delta 总和偏小 |
| B | 回调收到的是 framebuffer 像素与逻辑坐标混用，缩放比例导致尺寸增量缩小 | Medium | Low | 局部位移与屏幕位移存在固定比例 |
| C | 最小尺寸限制提前截断，导致无法继续放大 | Low | Low | 请求尺寸大于最小值但最终尺寸被钳制 |
| D | 右下角热点或鼠标捕获状态中途丢失，后续事件不再进入 resizing 分支 | Medium | Low | 拖动过程中 resizing 提前清空或 edges 变为空 |

## Log Evidence
Pending.

## Verification Conclusion
Pending.

## Final Evidence Gate
- **Evidence Sources**: /tmp/mujoco-check-window-resize-delta/trae-debug-log-window-resize-delta.ndjson
- **progress_status**: collecting
- **Hypothesis Conclusions**: A/B/C/D 均待验证
