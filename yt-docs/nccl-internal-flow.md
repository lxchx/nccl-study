# NCCL 内部关键流程

```mermaid
flowchart TD
    %% Initialization Phase
    Start([NCCL 初始化]) --> Scan["硬件扫描<br/>(PCIe / NVML)"]
    Scan --> Topo["构建拓扑图<br/>(ncclTopoSystem)"]
    Topo --> Search["路径搜索<br/>(search.cc)"]
    Search --> Routing["生成路由表<br/>(ncclChannel)"]

    %% Runtime Phase
    Call([Collective API 调用]) --> Enqueue["任务入队<br/>(enqueue.cc)"]
    Enqueue --> Select{"算法决策<br/>(Size & Type)"}
    
    Select -- Ring --> RingOp[Ring 通信逻辑]
    Select -- Tree --> TreeOp[Tree 通信逻辑]
    
    RingOp & TreeOp --> Lazy{是否首次调用?}
    Lazy -- Yes --> InitChannel["懒加载<br/>Channel 资源"]
    Lazy -- No --> Launch[发射 GPU Kernel]
    InitChannel --> Launch
    
    Launch --> Execute["数据分片与并发执行<br/>(device/prims_ll.h)"]
    Execute --> End([通信完成])

NCCL 在 `ncclCommInitRank` 时不会立刻开始通信，而是先跑一遍完整的系统扫描与路径规划：

1. **硬件扫描**：通过 `/sys/bus/pci/` 和 NVML 读取当前机器/节点的所有 GPU、CPU socket、NIC、PCIe switch 信息。
2. **构建全量拓扑图 (`ncclTopoSystem`)**：将所有硬件抽象为 Node，物理连线抽象为 Link（带带宽字段），形成一张邻接表。如果跨节点，还会包含 IB/RoCE switch。
3. **最短路搜索与带宽预留 (`paths.cc` + `search.cc`)**：
   - NCCL 首先用 BFS 计算任意两个节点间的所有可达路径，并标记路径类型（NVLink、PCIe、System Bus 等）和基础带宽。
   - 接着通过**递归暴力搜索**尝试不同的 GPU 排列组合来构建 Ring 和 Tree。在搜索过程中，它会模拟真实通信时的链路占用：**每选一条路径，就把对应链路的可用带宽减去（Bandwidth Reservation）**，防止多条路径挤占同一条物理链路导致超售。
   - 最终通过 `ncclTopoCompareGraphs` 选出最优解：第一优先级是**总聚合带宽最高**（`通道数 × 单通道瓶颈带宽`），如果带宽打平，则选**跳数最少**（延迟最低）的方案。

> **深度解析：Ring/Tree 的最优路径到底是怎么算的？**
>
> - **Ring 的顺序怎么定？** NCCL 会遍历所有 GPU 的排列组合。它会优先尝试走 NVLink 直连的邻居，因为带宽高。如果必须跨 PCIe，它会通过评分函数 (`ncclTopoSearchNextGpuSort`) 给每个候选 GPU 打分：`(链路带宽, 跳数)`，然后按分数排序逐个尝试。最终选出的 Ring 顺序是能让**所有边带宽之和最大**且**木桶效应最弱**的那条环。
> - **Tree 的父子关系怎么定？** Tree 的构建逻辑类似，但更复杂。因为它需要同时建立 `up`（向上汇聚）和 `down`（向下广播）两条路径。对于 Balanced Tree，NCCL 还会把前两个 GPU 到 NIC 的带宽除以 2（因为要双向通信）。最终选出的 Tree 是能让根节点到叶子节点的**总聚合带宽最大**的结构。
> - **"延迟最低 vs 带宽最高"是什么意思？** NCCL 并不是找两条独立的路，而是用一套优先级系统做决策：
>   1. **首要目标是带宽（Throughput）**：NCCL 会尽可能开多个 Channel（并发环/树）。最终方案的优劣取决于 `Channel数量 × 单通道瓶颈带宽`。这里的"带宽最高"指的是**聚合吞吐能力最大**。
>   2. **次要目标是延迟（Latency）**：只有在两个方案的总带宽完全打平的情况下，NCCL 才会去比较跳数 (`nHops`)，选出物理路径更短的那个。
> - **"木桶原理"怎么理解？** 在通信域里，一条环的传输速度取决于**最慢的那条边**。比如一个 Ring 里其他链路都是 NVLink (25GB/s)，但有一条边只能走 PCIe (16GB/s)，那这个 Channel 的有效带宽就是 16GB/s。NCCL 的搜索算法会极力避免这种"短板"，尽量让所有边都跑在高速链路上。

4. **生成路由表与性能模型**：
   - 将全局路径拆解为每个 rank 实际使用的邻居表（写入 `ncclChannel` 的 `ring.prev/next` 和 `tree.up/down[]`）。
   - 结合链路带宽估算每种算法在不同数据量下的耗时，生成调优参数表 (`bandwidths[]`, `latencies[]`)。初始化完成后，这些路由信息基本固定（只读）。

## 2. 通信原语的调用与调度（运行时阶段）

当用户调用 `ncclAllReduce` 等原语时，NCCL 内部经历以下流水线：

1. **入队与算法选择 (`enqueue.cc`)**：
   - API 调用并不直接执行硬件操作，而是将任务打包为 `ncclTaskColl` 排入用户指定的 CUDA stream。
   - NCCL 读取初始化阶段生成的调优参数表，根据当前通信的 **数据量 (`nBytes`)** 和 **原语类型**，动态决定使用 Ring 还是 Tree（以及具体的 Protocol）。小数据通常走 Ring，大数据走 Tree。
2. **懒初始化通道**：如果某个算法（如 NVLS 或特定 Ring channel）是首次被触发，NCCL 会按需分配该算法的 `channel` 资源，而不是在启动时全部预占显存和上下文。
3. **发射 GPU Kernel**：
   - NCCL 从 `channel.ring/tree` 邻居表中读取上游/下游 rank 信息。
   - 将数据分片（chunking）映射到不同 channel 上并发执行。
   - 通过 CUDA launch kernel 启动 GPU 侧的通信计算混合操作（如 PreMulSum）。跨节点通信则由 CPU Proxy 线程在后台接管 IB/RoCE 收发，与 GPU 计算异步重叠。
