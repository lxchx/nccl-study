# NCCL 对外抽象与使用指南

## NCCL 是什么

NCCL (NVIDIA Collective Communications Library) — **GPU 集体通信原语库**。

核心定位：用户**只需要声明 GPU（计算节点）之间要做什么样的集体通信操作**（AllReduce、Broadcast 等），NCCL 会根据 GPU 的拓扑连接关系自动规划出效率最高的计算与通信方式。

> 集体通信操作可以理解为需要多个GPU同时参与的运算&传输，比如多个GPU的数据的求和这样的计算<br><br>NCCL 在初始化时扫描 GPU 集群的拓扑，构建由多种链路组成的通信图（机内：NVLink、PCIe P2P、Shared Memory；机间：InfiniBand、TCP/IP），然后基于这张图自动决策走哪条链路、用什么算法（ring/tree/混合）、分多少并发通道。

用户不需要知道GPU的物理连接拓扑，也不需要在计算时评估使用哪些路径能最大化带宽，只需要调用封装好的通信原语函数，NCCL 会帮你搞定。

---

## NCCL 的位置

```
┌─────────────────────────────────────┐
│         PyTorch Distributed         │     ← 用户代码: model = DDP(model)
│    (DDP / FSDP / DeepSpeed)         │
│         TensorFlow / JAX            │
└──────────┬──────────────────────────┘
           │ 调用 ncclAllReduce(ncclSum)
┌──────────▼──────────────────────────┐
│              NCCL Library           │     ← "NCCL 层"（自动路由优化）
│   Graph & Routing (拓扑选择)        │
│   Scheduler (调度流水线)            │
└──────────┬──────────────────────────┘
           │ 选择传输后端
┌──────────▼──────────────────────────┐
│         Transport Backends          │     ← 实际硬件操作
│   NVLink / PCIe P2P / InfiniBand    │
│   Shared Memory / TCP/IP Socket     │
└─────────────────────────────────────┘
```

---

## API 核心抽象与概念

### Communicator & Rank

`通信域`: 集合通信操作涉及多个GPU的计算&传输，所以NCCL概念上会将参与计算的GPU组叫做**通信域(Communicator [group])**。

`rank`: 在通信域下的单个GPU计算设备称为 **rank**，序号从 0 开始不间隔，4 张 GPU 就是 rank 0、1、2、3。

`通信器`: 每个rank维护一个通信器(Communicator)(`ncclComm_t`)，存放通信时这个rank需要知道的上下文，在rank调用任何集体通信原语时都需要传入。

`多通信域`: 上层框架实际使用 NCCL 时，其所需的计算并不总是涉及所有GPU的计算（比如两组DP数据并行的GPU之间不需要任何通讯），而集合通信操作总是会调动到通信域内的所有GPU，因此NCCL支持从初始的、包含所有GPU的大通信域split出只包含部分rank的通信域用于部分GPU参与的集合通信计算。在小的通信域里也依然会给其中的GPU编上rank0~n的编号，所以同一个gpu在不同的通信域里可以是编号不同的rank。rank是通信域内的概念不跨通信域。

> **举个例子**：假设一个包含 8 张 GPU 的大通信域。其中 `0~3` 和 `4~7` 之间做数据并行（DP），而在每个 DP 组内，两两之间再做张量并行（TP）——即 `0&1`、`2&3` 为一组，`4&5`、`6&7` 为一组。<br><br>在这种情况下，系统实际上会为这些逻辑关系分别创建通信域用于计算：
	- **DP 组**：`0~3`, `4~7`
	- **TP 组**：`0&1`, `2&3`, `4&5`, `6&7`

> 如果对 `ncclComm_t` 通信器上下文具体信息感兴趣:
	* `ncclComm_t` 在初始化阶段 (`ncclCommInitRank` 调用时) 被配置好并固定，后续基本只读（例外是懒加载、错误标志 `abortFlag` 以及内部计数器）
	* rank 信息: 本 rank 的序号、rank 数量、cudaDev 指针 (`rank` / `nRanks` / `cudaDev`)
	* peer rank info: 如 pci bus id、cpu 亲和性、GPU 架构等 (`peerInfo`)
	* 拓扑信息: NCCL 初始化阶段探测出来的全量连接图，节点包括 GPU、CPU、NIC、PCIe switch 类型，边包括 NVLink、PCIe、IB、QPI 类型等等，存有带宽字段 (`topo`)
	* [TOAGENT:等完全理解机制之后再回来补充 graphs[NCCL_NUM_ALGORITHMS]、connectSend / connectRecv、ncclNet / netContext]

### 通信原语 — NCCL 向上提供的计算操作

NCCL 对外暴露 10 个集体通信原语。上层训推框架会把自己定义的运算最终翻译为这些集体通信原语的排列组合，下发给 NCCL 执行。

每个通信原语接受特定或者多个 rank 的数据，在计算后输出给特定或者多个 rank。当输入或者输出目标为特定 rank 时，NCCL 概念上会称这个 rank 为 root。

数据的类型是**一维数组**，在 NCCL 内概念上叫 **元素数组(element array)** (`ncclDataType_t`)——类型可以是 float32、uint64 等特定数字类型（当然数组内类型需要统一）

> 是的，一维数组。即使上层框架待算数据是一个多维张量，在 NCCL 层也都要先 flatten 为一维，计算完再变换回去。

一些原语名字里带有 `Reduce`，意味着这个原语输入多个数据然后以某种方式聚合(Reduce)成一个数据。具体聚合方式是用户配置的，可以从内置的 Sum、Prod(求积)、Max、Min、Avg 还有一种比较特殊的 PreMulSum 操作中选一种，在调用时传入对应的操作符即可。其他操作很好理解，PreMulSum 就是先乘一个常量再求和（当然因为交换性所以与先求和再乘是等效的），这个常量 `k` 是用户可配置的。所以和其他操作符不同的是，PreMulSum 操作符需要先调用一个函数传入常量 `k` 构造(new)出操作符实例，然后再传入这个实例；其他操作符只需要直接使用 NCCL 预先声明好的全局 const 变量实例。

10 个通信原语的大致解释如下，数据流示例以 4 GPU 为例，如果涉及 root 则以 rank 0 为 root，如果涉及 Reduce 则操作符为 sum，`-` 表示此处没有输入或者输出，实际一般是 NULL。

| 操作 | 简单解释 | 数据流示例 |
|---|---|---|
| **`AllReduce`** | 聚合所有 rank 的输入，输出聚合结果给所有 rank | `len=1`: [1],[2],[3],[4] → [10],[10],[10],[10]<br>`len=2`: [1,2],[3,4],[5,6],[7,8] → [16,20],[16,20],[16,20],[16,20] (同位置数字聚合) |
| **`Reduce`** | `AllReduce`，但只输出给 root | [1],[2],[3],[4] → [10], -, -, - |
| **`AllGather`** | 收集所有 rank 的输入，然后 flatten 成 list 输出给所有 rank | [1,2],[3,4],[5,6],[7,8] → [1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8] |
| **`Gather`** | `AllGather`，但只输出到 root | [1],[2],[3],[4] → [1,2,3,4], -, -, - |
| **`Scatter`** | root 的数据拆块发给所有 rank，是 `Gather` 的反向操作 | `len=4`: [1,2,3,4],-,-,- → [1],[2],[3],[4]<br>`len=8`: [1,2,3,4,5,6,7,8],-,-,- → [1,2],[3,4],[5,6],[7,8] |
| **`ReduceScatter`** | 先聚合再拆分，每个 rank 只拿属于自己的那块。与 `Reduce and-then Scatter` 语义等效，但更省带宽 | [1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16] → [28], [32], [36], [40] |
| **`Broadcast`** | 将 root 的输入给到所有 rank 的输出 | [3,4],-,-,- → [3,4],[3,4],[3,4],[3,4] |
| **`AlltoAll`** | **(矩阵转置)**<br>每个 rank 的数据视分成 nranks 块，如果将数据按照 rank 顺序排成矩阵，可以视为 rank i 的数据原来存放第 i 行的数据，操作后变成存第 i 列的数据（以块为单位） | `块大小=1`: [0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15] → [0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15]<br>`块大小=2`: [0,1, 2,3, 4,5, 6,7],[8,9, 10,11, 12,13, 14,15],[16,17, 18,19, 20,21, 22,23],[24,25, 26,27, 28,29, 30,31] → [0,1, 8,9, 16,17, 24,25], [2,3, 10,11, 18,19, 26,27], [4,5, 12,13, 20,21, 28,29], [6,7, 14,15, 22,23, 30,31] |
| **`Send/Recv`** | 一对一点对点通信 | `以 rank 1 发、rank 3 收为例`: -,[1],-,- → -,-,-,[1] |

#### sendbuff / recvbuff -- 通信原语的输入和输出源

实现上，每次调用通信原语时，涉及到的 rank 需要提供 **GPU 内存**指针作为数据源和目的地：
- **sendbuff** — 输入给原语的数据
- **recvbuff** — 原语计算结束后的输出地
- 一些操作中某些 rank 的 sendbuff/recvbuff 不使用，传 NULL 即可
- **In-place 支持**：原语支持 In-place 操作，即当 `sendbuff == recvbuff` 时原地操作，可以节省显存

### CUDA stream

NCCL的通信操作都是异步的，通信操作的函数返回只表示这个任务已经入队到CUDA工作队列，也就是**CUDA stream**。使用哪个stream是在调用原语时传入的。

## 简单代码示例

实践上，上层框架使用NCCL时习惯一个代码执行实体只处理一个rank的通信计算，这个执行实体可以是线程也可以是进程，所以就有多进程模式和单进程多线程模式，不过在NCCL接口上也并没有不同

> 初始化接口稍微有一点不同，单进程可以用`ncclCommInitAll`而多进程用`ncclCommInitRank(rank)`，虽然`ncclCommInitAll`其实等价于遍历rank并发执行ncclCommInitRank。<br><br>pytorch默认是多进程模式，tensorflow一般是单进程多线程。

一般来说，NCCL的使用流程和涉及函数如下：

1. **初始化**：创建 CUDA stream + communicator
   - `cudaStreamCreate`: CUDA stream
   - `ncclCommInitAll`(单进程)/`ncclCommInitRank`(多进程): communicator，rank间的拓扑探测也会在这个阶段完成
2. **准备数据**：分配并填充 sendbuff/recvbuff
   - `cudaMalloc`: GPU 内存分配
   - `cudaMemcpy`: 从 CPU 内存拷贝数据到 GPU
3. **调通信操作**
   - `ncclAllReduce`、`ncclBroadcast` 等
4. **销毁**
   - `cudaFree`: GPU 内存释放
   - `cudaStreamDestroy`: stream 清理
   - `ncclCommDestroy`: communicator 清理

下面是一个完整的按照这个流程编写的单进程模式的NCCL使用demo

### 完整C语言demo

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "cuda_runtime.h"
#include "nccl.h"

// 执行CUDA相关cmd后检查结果，有问题日志报错并退出
#define CUDACHECK(cmd) do { cudaError_t err = cmd; \
    if (err != cudaSuccess) { printf("CUDA FAILED: %s:%d '%s'
", \
        __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)

// 执行NCCL相关cmd后检查结果，有问题日志报错并退出
#define NCCLCHECK(cmd) do { ncclResult_t res = cmd; \
    if (res != ncclSuccess) { printf("NCCL FAILED: %s:%d '%s'
", \
        __FILE__, __LINE__, ncclGetErrorString(res)); exit(1); } } while(0)

#define N_DEV      4
#define BUF_SIZE   (32*1024)

// communicator
ncclComm_t comms[N_DEV];

// sendbuff / recvbuff（GPU 内存）
int *sendbuff[N_DEV], *recvbuff[N_DEV];

// CUDA stream
cudaStream_t streams[N_DEV];

// rank通信函数，一个函数线程负责一个rank
void *rank_thread(void *arg);

int main(int argc, char *argv[])
{
    printf("=== NCCL Demo: AllReduce with %d GPUs ===

", N_DEV);

    // Step 1: 创建 CUDA stream 和 communicator
    for (int i = 0; i < N_DEV; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }
    printf("[main] ncclCommInitAll for %d devices...
", N_DEV);
    // devlist = NULL 意味着使用所有可用 GPU
    // 如果不想全用需要传入使用的GPU序号数组
    NCCLCHECK(ncclCommInitAll(comms, N_DEV, NULL));
    printf("[main] initialized

");

    // Step 2: 创建和开始rank线程
    pthread_t threads[N_DEV];
    int ranks[N_DEV];
    for (int i = 0; i < N_DEV; i++) {
        ranks[i] = i;
        pthread_create(&threads[i], NULL, rank_thread, &ranks[i]);
    }
    for (int i = 0; i < N_DEV; i++)
        pthread_join(threads[i], NULL);

    printf("
=== Cleanup ===
");

    // Step 3: 销毁
    for (int i = 0; i < N_DEV; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < N_DEV; i++)
        NCCLCHECK(ncclCommDestroy(comms[i]));

    printf("=== Done ===
");
    return 0;
}

void *rank_thread(void *arg)
{
    int rank = *(int *)arg;
    CUDACHECK(cudaSetDevice(rank));  // cuda层面绑定GPU

    // 分配 sendbuff / recvbuff
    CUDACHECK(cudaMalloc((void **)&sendbuff[rank], BUF_SIZE * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&recvbuff[rank], BUF_SIZE * sizeof(int)));

    // Zero-fill buffer, then set first element to rank
    CUDACHECK(cudaMemset(sendbuff[rank], 0, BUF_SIZE * sizeof(int)));
    int val = rank;
    CUDACHECK(cudaMemcpy(sendbuff[rank], &val, sizeof(int), cudaMemcpyHostToDevice));
    printf("[rank %d] sendbuf[0] = %d (rest are 0)
", rank, val);

    // AllReduce（Sum）
    NCCLCHECK(ncclAllReduce(
        sendbuff[rank], recvbuff[rank], BUF_SIZE,
        ncclInt32, ncclSum, comms[rank], streams[rank]));

    // 等待通信完成
    CUDACHECK(cudaStreamSynchronize(streams[rank]));

    int result;
    CUDACHECK(cudaMemcpy(&result, recvbuff[rank], sizeof(int), cudaMemcpyDeviceToHost));
    printf("[rank %d] recvbuf[0] = %d (expected: 6)
", rank, result);

    return NULL;
}
```

**编译运行：**
```bash
nvcc -o nccl_demo nccl_demo.c -lnccl -lpthread
./nccl_demo
```

---

### 多进程模式的 Communicator 初始化

多进程下Communicator 初始化会稍微复杂一点，首先需要在每个进程都调用ncclCommInitRank，单进程只需要整体调用一次ncclCommInitAll，其次是NCCL需要生成一个标识全通信域的unique_id给到每个rank的Communicator(防止和系统里运行的其他NCCL大系统撞车)，单进程下ncclCommInitAll会在内部自动生成并分发给comms，多进程下NCCL不负责分发，需要使用NCCL之外的通讯方式如TCP或者MPI等。

> pytorch使用tcp socket分发unique_id

```c
int myRank = ...; // 用户代码决定当前进程的 rank
cudaSetDevice(myRank);                             // ← 关键：先绑定 GPU（假设 rank == device）

ncclUniqueId id;
if (myRank == 0) ncclGetUniqueId(&id);             // rank0 生成唯一 ID
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);  // 使用非NCCL手段，广播给所有 rank（如 MPI/TCP）

ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);       // Communicator 初始化
```

### Group API：批量提交 collective，提高计算效率

串行调用通信原语往往不能充分利用当前环境的全部带宽，造成浪费，因此NCCL提供了Group API可以批量提交通信运算。用 `GroupStart/GroupEnd` 包裹后：**多个原语被打包提交给 NCCL 调度器**，NCCL 会想办法并行推进，减少 CPU 等待时间、提升带宽利用率：

```c
// 每个 rank 线程/进程：
ncclGroupStart();

ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
ncclBroadcast(data, data, count2, ncclFloat, root, comm, stream);

ncclGroupEnd();  // 打包交给stream

cudaStreamSynchronize(stream);  // 阻塞等待计算完成
```

### ncclCommSplit: 通信域拆分

很多时候上层框架使用GPU时并不总是希望所有GPU在同一个大组里通信，而是划分成n个独立的通信域（比如两组DP数据并行的GPU之间不需要任何通讯），所以NCCL提供了ncclCommSplit方法满足这类需求。

> 实践上如果使用这个方法，会在rank初始化后就会立即开始分组后续不再改动，不需要担心基于通信域可分诞生出太多花活<br><br>为什么不直接开两个独立的NCCL实体？因为独立的NCCL之间无法感知到对方存在，很可能会在独立计算的过程中因为挤占同一条物理通信链路而导致效率下降。

**NCCL 对应机制：`ncclCommSplit(oldcomm, color, key, &newcomm)`**

它的作用类似 MPI 的 `MPI_Comm_split`。通过 `color` 参数把大 communicator 切成多个独立的小 communicator：
- `color=0` 的 rank（比如 rank 0, 1）组成 DP Group A
- `color=1` 的 rank（比如 rank 2, 3）组成 DP Group B

切出来的新 communicator 内部有自己独立的 ring/tree 拓扑。Group A 做 AllReduce 时，不会干扰 Group B 跑自己的通信任务，带宽利用率翻倍。

**实际框架中的做法：** PyTorch DDP 通过 `split_group` API 实现多后端共享；Megatron-LM 用 `ncclCommSplit` 将 TP/DP/PP 分组隔离开。

### 计算与通信重叠：专用 Stream + Event 依赖链

Demo 里用了 `cudaStreamSynchronize(stream)` 阻塞 CPU 等 NCCL 结束。实际训练中（如 DDP backward），如果等 AllReduce 完再算下一个 bucket，GPU 会大量空闲。

**NCCL + CUDA 对应机制：创建专用的 internal_stream_ + Event 依赖链**

框架通常不会直接用用户的主计算流跑 NCCL，而是创建一个专用 stream：
1. **提交通信前**：在用户的计算流上 `cudaEventRecord(start_event, compute_stream)`，然后让 `internal_stream_` 等待这个 event (`cudaStreamWaitEvent(internal_stream_, start_event)`)。这保证了输入数据算好了 NCCL 才开始传。
2. **NCCL 运行时**：用户在 `compute_stream` 上继续算下一个 bucket（比如做 backward），NCCL 在后台的 `internal_stream_` 跑 AllReduce。**两者在 GPU 硬件层面是并行的**。
3. **通信结束后**：NCCL 完成时在 `internal_stream_` 上 `cudaEventRecord(end_event, internal_stream_)`。当用户下一步计算需要用到 NCCL 的结果时，再让 `compute_stream` 等 `end_event`。

整个过程 CPU 不阻塞（不调用 `cudaStreamSynchronize`），全靠 Event 在 GPU 调度器里排好了先后顺序，实现了完美的流水线掩盖（Pipeline Hiding）。

**实际框架中的做法：** PyTorch DDP 的 Reducer 将梯度分成多个 bucket，利用 internal_stream_ 异步 AllReduce；TensorFlow/XLA 通过 `ExecuteOnStream` + `PollUntilDone` 实现类似效果。

