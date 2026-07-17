# NCCL 最短路径搜索与带宽预留算法详解

## 核心问题：NCCL 要解决什么？

在多卡训练中，NCCL 需要在 GPU 之间建立高效的通信拓扑。它面临两个关键决策：

1. **路径选择**：从 GPU A 到 GPU B 走哪条物理链路？（NVLink / PCIe Switch / CPU Bridge）
2. **带宽分配**：多个 Channel 同时工作时，如何避免共享链路的带宽冲突？

---

## 一、最短路径搜索算法 (BFS)

### 1.1 拓扑结构回顾

NCCL 将硬件抽象为一张图：
```
[GPU] ←→ [PCIe Switch] ←→ [CPU Bridge] ←→ [NIC]
          ↑___________________↓
```

每个节点都有属性：
- **类型**：`LINK_LOC`(本地), `LINK_NVL`(NVLink), `LINK_PCI`(PCIe)
- **带宽**：实际可用吞吐量（GB/s）
- **跳数**：经过的中间节点数量

### 1.2 BFS 搜索流程

```cpp
// 伪代码：ncclTopoSetPaths() 核心逻辑
for each GPU in system:
    // 初始化：所有路径标记为不可达 (PATH_DIS)
    for each target in all_nodes:
        gpu.paths[target].type = PATH_DIS
        gpu.paths[target].bw = 0
    
    // BFS 队列初始化为 [GPU]
    queue = [GPU]
    
    while queue not empty:
        current_node = queue.pop()
        
        for each link from current_node:
            next_node = link.remNode
            
            // 计算新路径的带宽（取瓶颈）
            new_bw = min(current_path.bw, link.bw)
            
            // 路径类型升级规则：
            // LOC < PIX < PXB < PHB < SYS < DIS
            // 越小的类型表示越"近"、越快
            new_type = max(current_path.type, link.type)
            
            // 更新条件（三者优先级）：
            // 1. 新路径类型更小（更快链路）
            // 2. 类型相同但带宽更大
            // 3. 类型和带宽都相同但跳数更少
            if new_type < existing_type OR
               (new_type == existing_type AND new_bw > existing_bw) OR
               (new_type == existing_type AND new_bw == existing_bw AND new_hops < existing_hops):
                
                update_path(next_node, new_type, new_bw, path + link)
                queue.push(next_node)
```

### 1.3 路径类型优先级表

| 类型 | 含义 | 速度等级 |
|------|------|----------|
| `PATH_LOC` (0) | 本地（同一 GPU） | ~5000 GB/s (显存) |
| `PATH_NVL` (1) | NVLink 直连 | ~450-600 GB/s |
| `PATH_PIX` (2) | PCIe Switch 内部 | ~64 GB/s |
| `PATH_PXB` (3) | PCIe Switch 间桥接 | ~64 GB/s |
| `PATH_PHB` (4) | 经过 CPU Host Bridge | ~48 GB/s |
| `PATH_SYS` (5) | 系统总线（最慢） | ~40 GB/s |
| `PATH_NET` (6) | 网络通信 | 取决于网卡 |
| `PATH_DIS` (7) | 不可达 | 0 |

**关键规则**：NCCL 认为 "LOC → SYS" 的路径比 "PIX → PHB" 更快，即使后者跳数更少。

---

## 二、带宽预留算法 (Backtracking Search)

### 2.1 问题场景

假设有 4 张 GPU：
```
GPU0 ──── NVLink ──── GPU1
  │                       │
PCIe                    PCIe
  │                       │
CPU Bridge ───────────────┘
```

NCCL 需要为每个 Channel 分配带宽：
- **Channel 0**: GPU0 → GPU1 (NVLink)
- **Channel 1**: GPU0 → CPU → GPU2 (PCIe)
- **Channel 2**: GPU1 → CPU → GPU3 (PCIe)

**冲突点**：CPU Bridge 的总带宽有限，不能同时满足所有 Channel。

### 2.2 带宽预留机制

```cpp
// followPath() 核心逻辑
for each link in path:
    // 计算需要的带宽
    required_bw = channel_bandwidth
    
    // 检查链路剩余容量
    if link.available_bw < required_bw:
        return false // 路径不可用
    
    // 预留带宽（扣减可用容量）
    link.available_bw -= required_bw
    
    // 特殊处理：反向路径也需要预留
    if path_type == BALANCED_TREE:
        reverse_link.available_bw -= required_bw / 2

// 如果所有链路都成功预留，返回 true
```

### 2.3 搜索策略 (ncclTopoSearchRec)

这是一个 **回溯搜索算法**，结合贪心策略：

```python
def search_channels(system, graph):
    """
    为当前 Channel 找到最佳 GPU 排列
    
    参数:
        system: 硬件拓扑
        graph: 通信图模式 (Ring/Tree/SplitTree)
    
    返回:
        最优的 GPU 连接顺序和带宽分配
    """
    
    # 步骤 1: 从当前节点开始 BFS 探索
    for next_gpu in get_candidates(system.current_gpu):
        
        # 步骤 2: 尝试预留带宽
        if can_reserve_bandwidth(next_gpu, required_bw):
            
            # 步骤 3: 标记为已使用
            mark_used(next_gpu)
            
            # 步骤 4: 递归搜索下一个节点
            result = search_channels(system, graph)
            
            if result.success:
                return result
            
            # 步骤 5: 回溯（释放带宽）
            release_bandwidth(next_gpu, required_bw)
            unmark_used(next_gpu)
    
    # 所有候选都失败，返回空结果
    return Failure()
```

### 2.4 剪枝优化 (ncclTopoSearchNextGpuSort)

为了避免指数级爆炸，NCCL 使用 **评分排序**：

```cpp
struct GpuScore {
    int g;              // GPU 索引
    int intraNhops;     // 内网跳数
    int interNhops;     // 外网跳数
    int intraBw;        // 内网带宽
    int interBw;        // 外网带宽
};

// 排序规则（优先级从高到低）：
// 1. 外网带宽越大越好 (interBw DESC)
// 2. 外网跳数越少越好 (interNhops ASC)
// 3. 内网带宽越大越好 (intraBw DESC)
// 4. 内网跳数越少越好 (intraNhops ASC)

qsort(all_candidates, GpuScore::compare);
```

**效果**：先尝试最有可能成功的路径，找到解就立即返回（贪心）。

---

## 三、完整案例演示

### 3.1 RTX 5090 测试机拓扑

```xml
<system>
    <!-- GPU0 (busId: 27) - NUMA 0 -->
    <gpu rank="0" busid="27">
        <link type="LOC" bw="5000.0"/>
        <link type="PHB" bw="48.0" target="GPU1"/>
        <link type="SYS" bw="40.0" target="GPU2..7"/>
    </gpu>
    
    <!-- GPU1 (busId: 38) - NUMA 0 -->
    <gpu rank="1" busid="38">
        <link type="PHB" bw="48.0" target="GPU0"/>
        <link type="SYS" bw="40.0" target="GPU2..7"/>
    </gpu>
</system>
```

### 3.2 Channel 0 搜索过程

**目标**：为 Ring Pattern 找到 GPU0 → ? → ... → GPU7 → GPU0 的路径

1. **从 GPU0 开始**：`candidates = [GPU1(PHB), GPU2..7(SYS)]`
2. **评分排序**：`[GPU1(48GB/s), GPU2(40GB/s), ..., GPU7(40GB/s)]`
3. **尝试 GPU1**：预留 48GB/s → ✅ 成功
4. **从 GPU1 继续**：`candidates = [GPU0(used), GPU2..7(SYS)]`
5. **尝试 GPU2**：预留 40GB/s → ✅ 成功 (PHB→SYS 无冲突)
6. ... 重复直到所有 8 张卡都连接到

**最终路径**：`GPU0(PHB,48) → GPU1(SYS,40) → GPU2(SYS,40) → ... → GPU7(PHB,48)`

### 3.3 Channel 1 搜索过程

由于 PHB 带宽已被 Channel 0 占用，Channel 1 必须使用 SYS 路径：
- `GPU0(SYS,40) → GPU2(SYS,40) → ...` (避开已占用的 PHB)

---

## 四、算法复杂度分析

| 维度 | 复杂度 | 说明 |
|------|--------|------|
| BFS 路径搜索 | O(V+E) | 每个节点探索一次，每条边检查一次 |
| 带宽预留搜索 | O(N! × C) 最坏情况 | N=GPU数, C=候选分支数，实际中剪枝大幅优化 |
| 总时间 | ~1秒 (典型配置) | NCCL_SEARCH_GLOBAL_TIMEOUT 限制 |

---

## 五、关键环境变量

| 变量 | 作用 |
|------|------|
| `NCCL_GRAPH_FILE` | 跳过搜索，直接使用预计算的 XML 拓扑 |
| `NCCL_P2P_LEVEL` | 控制 P2P 通信允许的最大距离 (LOC~SYS) |
| `NCCL_IGNORE_DISABLED_P2P` | 忽略 NVML 报告的 P2P 禁用状态 |

---

## 六、总结

NCCL 的路径搜索本质上是 **带约束的图遍历**：

1. **BFS 找到所有可能的物理路径**（考虑硬件拓扑）
2. **评分排序优先尝试高速链路**（NVLink > PCIe > SYS）
3. **带宽预留避免资源冲突**（同一通道不能超卖）
4. **回溯搜索保证全局最优解**（局部失败时探索其他分支）

这就是为什么 NCCL 初始化需要时间——它在做复杂的组合优化。