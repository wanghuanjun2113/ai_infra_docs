# 昇腾 NPU 底层软件栈与流调度机制

> 面向推理/训练框架（vLLM、torch_npu 等）开发者的 CANN 底层知识整理。
> 涵盖：软件栈分层、算子、运行时、流调度与仲裁、多流并发工程实践、错误码分层诊断。
> 环境参考：Atlas 310P3 / CANN 9.1 / torch_npu 2.10；概念同样适用于 910 系列。

---

## 目录

1. [软件栈总览](#1-软件栈总览)
2. [驱动层（HDK）](#2-驱动层hdk)
3. [CANN 层](#3-cann-层)
4. [torch_npu 层](#4-torch_npu-层)
5. [算子：从概念到一次调用的完整旅程](#5-算子从概念到一次调用的完整旅程)
6. [运行时核心对象：上下文、流、事件、内存](#6-运行时核心对象上下文流事件内存)
7. [流的调度：三道门与仲裁器](#7-流的调度三道门与仲裁器)
8. [多流并发的工程实践](#8-多流并发的工程实践)
9. [Python 层的流控制](#9-python-层的流控制)
10. [流与事件的生命周期](#10-流与事件的生命周期)
11. [错误码分层与调试方法论](#11-错误码分层与调试方法论)
12. [术语表](#12-术语表)

---

## 1. 软件栈总览

```
┌─────────────────────────────────────────────────┐
│  应用层：vLLM / 训练框架                          │
├─────────────────────────────────────────────────┤
│  框架适配层：PyTorch + torch_npu                 │
│    （把 torch 算子翻译成昇腾调用）                 │
├─────────────────────────────────────────────────┤
│  CANN（≈ NVIDIA 的 CUDA）                        │
│    - 算子库：aclnnXxx 接口 + 预编译算子实现        │
│    - 运行时（libruntime）：上下文/流/事件/任务下发  │
│    - 编译工具链：Ascend C                         │
├─────────────────────────────────────────────────┤
│  驱动层（HDK，内核态）                            │
│    - 设备管理 / 显存分配 / 任务下发 / HDC 通信     │
├─────────────────────────────────────────────────┤
│  硬件：AI Core（矩阵/向量单元）、显存、PCIe        │
└─────────────────────────────────────────────────┘
```

与 NVIDIA 生态的对照记忆：

| 昇腾 | NVIDIA | 说明 |
|---|---|---|
| CANN | CUDA | 计算架构与软件栈 |
| torch_npu | torch CUDA 后端 | 框架适配 |
| HDK 驱动 | NVIDIA driver | 内核态驱动 |
| npu-smi | nvidia-smi | 设备管理工具 |
| Ascend C | CUDA C++ | 算子开发语言 |
| aclnn 接口 | cuDNN/cuBLAS API | 算子调用接口 |

**层次认知的意义**：上层报错常常只是"最先撞墙的人"。拿到故障先问"这是哪层的错"，再往下追——这是全文反复出现的调试纪律。

---

## 2. 驱动层（HDK）

驱动是内核态模块，用户态一切操作最终落到它暴露的**设备文件**上。以容器化部署为例，必须挂载四类设备文件，各自对应驱动的四大职责：

| 设备文件 | 职责 | 缺失时的现象 |
|---|---|---|
| `/dev/davinciN` | 计算设备本体（N = 芯片号） | 该卡不可见 |
| `/dev/davinci_manager` | 卡管理（拓扑/健康） | torch_npu 初始化失败 |
| `/dev/devmm_svm` | **显存分配**（SVM = 共享虚拟内存） | `libascend_hal.so` 加载失败 |
| `/dev/hisi_hdc` | **HDC 通信通道** | 同上 |

### HDC（Host Device Communication）

宿主机 CPU 与 NPU 卡之间的通信通道，属于驱动层机制。任务描述符下发、事件完成信号、内存操作都经它传递。**HDC 断连 = CPU 与该卡彻底失联**，进程内所有后续调用报 `hdc disconnect`，通常意味着驱动/链路级故障（硬件或宿主-设备通信问题），而非应用层 bug。

### dmesg：驱动层的黑匣子

驱动级错误（用户态看不到）记录在宿主机内核日志中。排查 NPU 故障**必须上宿主机查 dmesg**，并与故障时刻做秒级时间戳对齐。典型条目：

```
[ascend] [tsdrv] [ERROR] hdc connect down, devid(3) fid(0) tsid(0)
```

### npu-smi

驱动包附带的管理工具：设备健康、AICore 利用率、显存占用。**AICore 利用率是分层诊断的关键指标**——详见[第 11 节](#11-错误码分层与调试方法论)。

---

## 3. CANN 层

CANN（Compute Architecture for Neural Networks）包含三大部分：

### 3.1 算子库

- 调用接口统一为 **`aclnn` 前缀 + 算子名 + 版本**（如 `aclnnNonzeroV2`），声明于 `libascendcl.so`。
- 预编译实现在 `opp/built-in/op_impl/` 下（`LD_LIBRARY_PATH` 中可见）。
- 调用 `aclnnXxx` 只是"下单"——参数打包入队后**立即返回**，真正执行在设备上异步完成（见[第 5 节](#5-算子从概念到一次调用的完整旅程)）。

### 3.2 运行时（libruntime.so）

管理执行事务的常驻层，接口统一带 **`rt` 前缀**（runtime 缩写）：

| 接口 | 职责 |
|---|---|
| Context 相关 | 进程与 NPU 的"会话"，所有资源挂在它下面 |
| `rtStreamCreate` 等 | 流（任务队列）管理 |
| `rtEventXXX` | 事件（同步点）管理 |
| `rtStreamSynchronize` | 等待流排空（崩溃日志常客） |
| 任务下发 | 把算子调用组装成设备任务，经驱动送卡 |

一句话定位：**runtime = 加速器的"随身管家"**——代码只管下单，它负责排队（流）、盯进度（事件）、维持会话（上下文）、联系后勤（驱动+显存）。类比：runtime 之于 NPU，如同操作系统之于 CPU。

### 3.3 编译工具链：Ascend C

自定义算子开发语言（≈ CUDA C++）。开发者用 Ascend C 编写 kernel（`.cpp`），经 CMake + CANN 工具链编译为 `.so`，在 PyTorch 侧通过 `torch.ops.load_library` 注册。运行时随框架自动加载。

### 3.4 tiling

算子执行前把大张量切块以适配硬件的工作，在 host 侧计算。tiling 结果决定 block 数（→ 占多少核，见[第 7 节](#7-流的调度三道门与仲裁器)），是 host 侧开销与设备并发的交汇点。

---

## 4. torch_npu 层

PyTorch 算子（`torch.matmul`）经 torch_npu 分发到对应 aclnn 接口。这一层的头文件（`op_api_common.h`）里有个重要的分支机制——**算子下发路径**：

```cpp
#define EXEC_NPU_CMD(aclnn_api, ...)                              \
    do {                                                          \
        static const auto task_queue_enable =                      \
            c10_npu::option::OptionsManager::GetTaskQueueEnable(); \
        if (task_queue_enable == 2) {                             \
            EXEC_NPU_CMD_V2(aclnn_api, __VA_ARGS__);              \
        } else {                                                  \
            EXEC_NPU_CMD_V1(aclnn_api, __VA_ARGS__);              \
        }                                                         \
    } while (false)
```

环境变量 `TASK_QUEUE_ENABLE`（1=V1，2=V2）在这里生效——它改变**任务下发的打包/下发方式**（host 开销优化），不改变算子数学语义。这解释了为什么排查数值/调度问题时可以先排除它。

---

## 5. 算子：从概念到一次调用的完整旅程

**算子（operator）是 NPU 上的基本计算单元**，类似 GPU 的 kernel：矩阵乘、逐元素加、nonzero 等各是一个算子。

### 一次算子调用的完整旅程

以模型里的一次 RMSNorm 为例：

```
Python:  y = rms_norm(x, w)
  → torch_npu 分发到 aclnnRmsNorm（参数打包入队，立即返回）
  → CANN 运行时算好 tiling，把任务描述符写入"当前流"
  → 驱动经 HDC 把描述符推到设备
  → AI Core 执行（此刻 npu-smi 的 AICore% 才会上涨）
  → 完成信号经事件回传
  → 之后某处 event.synchronize() / 流同步等到它，结果可用
```

**关键认知：算子调用是异步的**。Python 里一行算子返回 ≠ 算完。因此：

- "报错的调用"和"真正出错的执行"可能隔很远；
- 读结果前必须同步（流 FIFO 或事件）；
- 单看 Python 栈定位问题会被误导，需要 AICore 利用率 + 栈 + trace 三方交叉。

### 自定义算子（Ascend C）示例骨架

```cpp
// kernel：每 block 处理若干行，块数决定核占用
class MyOp310 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t tokens) {
        xg.SetGlobalBuffer((__gm__ half*)x);
        yg.SetGlobalBuffer((__gm__ half*)y);
        T = tokens;
        pipe.InitBuffer(buf, D * sizeof(half));   // 在本核私有 UB 里划缓冲
    }
    __aicore__ inline void Process() {
        uint32_t row = GetBlockIdx();
        if (row >= T) return;
        for (;;) {
            auto h = buf.Get<half>();
            DataCopy(h, xg[row * D], D);          // DMA 搬入（MTE2）
            Fence<HardEvent::MTE2_V>(pipe);       // 流水线栅栏：搬完才能算
            /* ... 向量计算 ... */
            Fence<HardEvent::V_MTE3>(pipe);       // 算完才能搬出
            DataCopy(yg[row * D], h, D);          // DMA 搬出（MTE3）
            uint32_t next = row + GetBlockNum();
            if (next >= T) break;
            row = next;
        }
    }
};

// binding：发射块数 = 核占用，从"当前流"拾取执行流
void my_op(const at::Tensor& x, at::Tensor& y) {
    uint32_t t = x.size(0), blocks = std::min(t, 8u);
    auto stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("MyOp310");
    cmd.SetCustomHandler([x = x.detach(), y = y.detach(), t, blocks, stream]() -> int {
        launch_my_op(blocks, stream, (uint8_t*)x.data_ptr(), (uint8_t*)y.data_ptr(), t);
        return 0;
    });
    cmd.Run();
}
```

三个要点：

1. **`blocks` 即核占用**：`min(T, 8)` 表示最多占 8 个核——小算子故意留地盘给并发（见[第 7.4 节](#74-block-与核的映射)）。
2. **`getCurrentNPUStream()`**：C++ 从不决定用哪条流，被动拾取 Python 层设置的线程局部状态（见[第 9 节](#9-python-层的流控制)）。
3. **`Fence<HardEvent>`**：管理**单核内部**异步流水线（MTE2 搬入 / V 向量 / MTE3 搬出）的顺序，与其它算子无关——片上缓冲的同步纪律全部在单算子、单核内部。

---

## 6. 运行时核心对象：上下文、流、事件、内存

| 对象 | 本质 | 类比 |
|---|---|---|
| Context | 进程与 NPU 的会话，资源容器 | 数据库连接 |
| Stream | **FIFO 任务队列**：流内严格按序，流间无隐含顺序 | 后厨的排菜单 |
| Event | 同步工具：流内插里程碑，可查询/等待 | 传菜铃 |
| Device Memory | 设备显存，驱动分配（`halMemAlloc`） | — |
| Pinned Memory | 锁页主机内存，异步 H2D/D2H 的前提 | — |

### 流的两条铁律

1. **流内 FIFO**：任务 B 必须等前面的任务 A 完全结束——哪怕 B 很小、A 很大、核闲着。同一流上的内存复用因此是"免费的同步"。
2. **流间无隐含顺序**：谁先完成取决于资源空闲情况（机会主义）。跨流先后**必须显式声明**：

```
流A：... → 写数据 → event.record() → ...
流B：... → event.wait() → 读数据 → ...
```

漏掉事件交接的后果：**不报错、不出异常，只是静默读半新数据**——多流 bug 里最阴险的一类。

### 锁页内存（pinned）

异步 H2D/D2H 拷贝要求 host 内存锁页。普通分页内存的拷贝会被拆散、走 staging 缓冲、甚至阻塞流。框架优化中常见的"pinned buffer 暂存"正是解决此问题。

---

## 7. 流的调度：三道门与仲裁器

### 7.1 任务派发的三道门

硬件调度器判定一个任务可否派发，依次检查：

```
门1 顺序门：是否为本流队列头？（FIFO，不可越过）
门2 依赖门：它等待的事件都完成了吗？
门3 资源门：空闲核/输出缓冲装得下它的 block 方案吗？
```

三道门全过才进入仲裁。这解释了一个常见困惑：**同一流里小任务排在大任务后必须等，另一条流的小任务却能插进去跑**——门 3 按各任务自己的资源足迹独立判定，不是整卡锁。

### 7.2 仲裁策略

多流同时就绪时，典型设计为**等优先级轮询（round-robin）+ 资源适配**，常加老化（aging）防饿死。部分运行时允许建流时指定优先级（通常只是提示）。

**可移植的正确态度：跨流相对调度顺序是"未定义"的**。任何依赖"流 A 赢过流 B"的代码都是错的——仲裁是实现细节。

### 7.3 无抢占

kernel 一旦派发到 AI Core，**跑到结束为止**，新就绪的高优先级任务不能踢掉它（与 CPU 的时间片抢占最大的差异）。后果：长 kernel 制造调度"空洞期"，对尾延迟（P99）影响远大于平均吞吐。

### 7.4 block 与核的映射

- 算子发射时带 **block（任务实例）数**，大致一个 block 占一个核跑到结束——"用 10 个核"的真正含义是"发射了 10 个 block"。
- 常规算子的 block 数由 tiling 引擎按形状自动计算，策略通常是**占满所有核**；自定义算子（Ascend C）由开发者显式指定（如 `min(T, 8)`）。
- **两个算子能否并行**：需要 (a) 在不同流上，(b) 无数据依赖，(c) 各自 block 数 ≤ 空闲核数。同流则 FIFO 强制串行，与核空闲与否无关。

### 7.5 片上存储：为什么并行算子不打架

达芬奇架构中 **UB、L1、L0、寄存器堆是每核私有**的物理 SRAM。并行 = 占用**不同的核**（一个核同一时刻只跑一个 block，不共享、不分时），因此：

- 算子 A 在核 3 用它的 UB，算子 B 在核 8 用另一块物理无关的 UB——**不存在共享，不存在冲突**；
- 核间真正的汇合点是 **L2 与 DDR**：并行算子在此竞争带宽——后果是**变慢，不会算错**（硬件保证一致性）；
- `Fence<HardEvent>` 管理的是单核内 DMA/向量流水线的顺序，与邻核无关。

对比：NVIDIA 的 SM 允许两个 kernel 的 block 同时驻留同一 SM，靠硬件按 block 划分寄存器堆/共享内存保证隔离。路线不同，结论相同：**算子作者无需为"隔壁算子"操心片上缓冲，只需管好核内流水线纪律 + 全局内存的跨流同步**。

---

## 8. 多流并发的工程实践

### 8.1 为什么要通信/计算分流

计算（AI Core）与通信（SDMA/网络引擎）、拷贝（DMA）是**互不占用预算的物理引擎**。同一条流时 FIFO 强制串行——通信期间 AI Core 晒太阳；分流后调度器各自派发，并行自动发生。

```
单流：[计算 8ms][AllReduce 2ms][计算 8ms][AllReduce 2ms]      总 20ms

双流（分桶重叠）：
计算流： [桶1 4ms][桶2 4ms][桶1' 4ms][桶2' 4ms]
通信流：           [AR桶1 2ms]  [AR桶2 2ms]                   总 ~16ms
```

### 8.2 分流的代价

1. **事件交接开销**：每次跨流交接 = record + wait，host 侧有实打实的延迟（亚毫秒~毫秒级）。
2. **生命周期拉长**：异步执行要求 tensor 活到通信真正完成——跨线程生命周期问题（后台回调析构要拿 GIL 等）由此产生。
3. **图捕获困难**：ACL graph 捕获要求工作在捕获流上顺序执行，跨流事件等待在图内难以表达。

### 8.3 决策表

| 场景 | 通信特征 | 选择 | 理由 |
|---|---|---|---|
| prefill / 大 batch | 通信大（几十 MB），计算长 | 双流重叠 | 通信藏得进计算；事件开销占比小 |
| decode（小 token、延迟敏感） | 通信小、碎、频繁 | 单流（"当前流集合通信"） | 事件开销 ≥ 重叠收益；且可整图捕获 |
| H2D/D2H 拷贝 | 与计算无依赖 | 双流 | DMA 独立引擎，重叠近零代价（需 pinned 内存） |

> 实例参考：vLLM-ascend 在 310P decode（MTP3、FDO 图模式）下采用"当前流 AllGather + 整图捕获"，放弃双流重叠换取顺序确定性与图重放开销的削减——是"分流收益 < 同步开销 + 图不可捕获"时的正确取舍。

### 8.4 黄金法则

**流内靠 FIFO，流间靠事件，仲裁当不存在，内存复用要么同流要么显式同步。**
凡是"碰巧能跑"的跨流时序依赖，都是换一版驱动就可能翻车的定时炸弹。

---

## 9. Python 层的流控制

流的**选择**在 Python 层，但机制是**线程局部的"当前流"状态**，而非逐调用传参：

```python
s = torch.npu.Stream()          # 建流
with torch.npu.stream(s):       # 块内"当前流" = s
    y = x @ w                   # 提交到 s
# 离开块后恢复之前的流
```

- **不设置 = 默认流**。99% 的代码从不显式碰流。
- C++ 侧所有算子通过 `c10_npu::getCurrentNPUStream()` **被动拾取**该状态——Python 的 `with` 与 C++ 的 `getCurrentNPUStream` 是同一状态的两端。
- **线程局部**：每个线程有独立的"当前流"。多流并发的代码形态 = 多线程 + 各自的 with 块。新线程默认捡默认流，因此框架的子线程开头总有 set_device / set_stream 对齐样板。
- **离开 with ≠ 执行完成**：块内算子只是提交。跨流读结果必须先 `event.record(s)` + 等待。

**谁在做选择**：框架（如 vllm-ascend 在 Python 里建通信流/拷贝流并用 with 切换），用户和启动脚本完全不感知。集合通信库传统上自带专用通信流；"当前流 AllGather"类优化即是在 C++ 绑定层改为拾取当前流——一行选择改变整个调度策略。

---

## 10. 流与事件的生命周期

**流没有"闲置回收"**——显式生命周期的运行时对象，创建后一直存在直到销毁或进程退出，与有无任务无关。

```
创建：torch.npu.Stream()（或框架启动时）
存活：无论忙闲，占一个队列结构 + 硬件槽位（KB 级，可忽略）
销毁：三种途径
  1. Python 包装无引用 → GC（时机不确定）
  2. 显式销毁（框架基本不用）
  3. 进程退出，上下文销毁
```

要点：

- **销毁是延迟的**：队列有未完成任务时，等排空再真正释放。
- **未完成任务钉住资源**：流上在跑的 kernel 会让它读写的 tensor 显存无法回收——这才是流相关资源问题的主战场（不是"流本身"）。
- **创建数量有限度**：硬件队列槽位有限，无节制建流会耗尽池子。
- **框架实践**：启动时建好、进程存活期间永远复用（vLLM 三流即进程级单例）。长服务里动态建/销毁流是反模式。
- 事件、图同理：**"闲置"从来不是释放的触发条件，"失去引用"才是**。

---

## 11. 错误码分层与调试方法论

### 11.1 常见错误码的层次归属

| 报错 | 层 | 含义 |
|---|---|---|
| `aicore exception 507015` | 硬件/算子执行 | kernel 把 AI Core 跑挂——设备执行层实锤故障 |
| `aclnnXxx failed`（如 361001） | 算子接口 | 算子下单/执行失败——**可能是替罪羊** |
| `halMemAlloc failed, drvRetCode=42` | 驱动 | 驱动分不出显存——比算子底层 |
| `hdc connect down`（dmesg） | 驱动/链路 | CPU↔卡通信断——最底层，基本宣判硬件/链路问题 |
| `EE9999` / `ERR00100` | CANN 通用 | "内部错误"总称，本身不指明层 |
| `rtStreamSynchronize/rtEventSynchronize failed` | 运行时 | 同步等不到回应——通常因底层故障（如 HDC 断） |

### 11.2 分层诊断流程

```
1. 定层：报错来自哪一层？（算子 / 运行时 / 驱动 / 链路）
2. 查 dmesg（宿主机！容器内看不到）：驱动层有无对齐时间戳的错误
3. 看 AICore 利用率（npu-smi）：
   - 0% + host 忙     → host 侧瓶颈（调度/同步空转），设备无辜
   - 高 + 吞吐低      → 设备在算但算得慢（算子/带宽）
4. 抓 Python 栈（py-spy，多进程/多 rank 全抓）：
   - 卡在 event/流同步 → 依赖链问题，继续找谁没完成
   - 注意：同步点的栈 ≠ 根因位置（异步性）
5. 需要时抓 profiler trace（按流分泳道，看重叠/空洞/等待）
```

### 11.3 跨层塌方的判定案例

某推理服务崩溃的完整链条：

```
16K 边界请求在跑
  → draft 路径调 nonzero 算子 → 报错 361001        （算子层）
  → 紧接着驱动 2MB 显存分配失败                     （驱动层）
  → dmesg: hdc connect down（时间戳秒级吻合）        （链路层）
  → 全部 worker 死亡，服务下线
```

判定逻辑：**报错顺序逐级向下塌**。若只是算子写错（tiling 错），通常表现为算子返回错误但设备与驱动安然无恙；连驱动分配和 HDC 都倒，说明塌到硬件/链路层——上层报错只是最先撞墙的人。反之，把算子层报错当根因，就会去修一个无辜的算子。

### 11.4 实用工具箱

| 工具/方法 | 用途 | 备注 |
|---|---|---|
| `npu-smi` | 健康/利用率/显存 | AICore% 是分层关键指标 |
| 宿主 `dmesg -T` | 驱动层黑匣子 | 必须**上宿主机**，秒级对齐时间戳 |
| `py-spy dump/record` | Python 栈/火焰图 | 多进程全抓；`--locals` 可看调度器内部状态 |
| Profiler（MindStudio Insight 等） | 按流分泳道的 trace | 看 kernel 重叠/等待/空洞 |
| 编译缓存 `computation_graph.py` | 查哪些自定义 op 进了编译图 | 排查图/ eager 双路径不一致 |

---

## 12. 术语表

| 术语 | 全称/出处 | 一句话解释 |
|---|---|---|
| CANN | Compute Architecture for Neural Networks | 昇腾计算软件栈（≈CUDA） |
| ACL | Ascend Computing Language | CANN 的 C API 族 |
| aclnnXxx | ACL Neural Network 接口 | 算子调用接口（≈cuBLAS/cuDNN API） |
| HDK | 驱动包 | 内核态驱动 + npu-smi（≈NVIDIA driver） |
| HDC | Host Device Communication | 宿主与卡之间的驱动层通信通道 |
| AI Core | 达芬奇架构计算单元 | 含 Cube（矩阵）/Vector（向量）单元，每核私有 UB/L1 |
| UB / L1 | Unified Buffer / L1 | 每核私有片上 SRAM，核间不共享 |
| MTE2 / MTE3 / V | 搬入/搬出 DMA / 向量单元 | 单核内异步流水线，用 Fence 协调 |
| tiling | — | 算子切块方案，决定 block 数（核占用） |
| block | 任务实例 | 调度最小单元，≈一个 block 占一个核到跑完 |
| stream | 流 | FIFO 任务队列；流内严格有序，流间无隐含顺序 |
| event | 事件 | 流内里程碑，跨流同步的唯一正确工具 |
| current stream | 当前流 | 线程局部状态，Python `with torch.npu.stream(s)` 设置，C++ `getCurrentNPUStream()` 拾取 |
| Ascend C | — | 自定义算子开发语言（≈CUDA C++） |
| TQ | Task Queue（`TASK_QUEUE_ENABLE`） | torch_npu 算子下发路径开关（V1/V2），改下发方式不改数学 |
| pinned memory | 锁页内存 | 异步 H2D/D2H 的前提，否则拷贝被拆散/阻塞 |
| NPUGraph / ACL graph | — | 设备侧图捕获与重放；捕获要求单流顺序 |
| dmesg | — | 内核日志，驱动层故障的取证地 |

---

## 附：一页速查

```
栈：    应用 → torch_npu → CANN(算子库+runtime) → 驱动(HDC) → 硬件
调用：  Python 下单 → aclnn 入队(异步!) → runtime 组任务 → 驱动过 HDC → AI Core 执行
流：    流内 FIFO；流间靠 event；仲裁未定义，别依赖
调度：  三道门(顺序/依赖/资源) → 仲裁(RR+资源适配) → 无抢占
并行：  block 数=核占用；UB/L1 每核私有不冲突；L2/DDR 是共享点(慢不错)
分流：  大通信→双流重叠；小碎通信→单流进图；拷贝永远独立流(pinned)
生命周期： 闲置不释放，失去引用才回收；未完成任务钉住内存
排障：  先定层 → 宿主 dmesg → AICore% → 多进程栈 → trace；上层报错≠根因
```
