# NCCL 拓扑搜索算法伪代码

围绕 ncclTopoCompute → ncclTopoSearchRec → ncclTopoSearchTryGpu → ncclTopoSearchRecGpu → ncclTopoFollowPath
五个核心函数，展示主体意图和调用关系。省略了 trace、错误处理、NVLS/CollNet 等特殊分支。

---

## ncclTopoCompute(ncclTopoSystem* system, ncclTopoGraph* graph) — 搜索的总调度

```
ncclTopoCompute(system, graph):
    tmpGraph = copy(graph)                          // 工作副本，搜索过程中修改它

    // 1. 选初始带宽：从 speedArray 里挑一个不超过硬件上限的
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex]

    // 2. 多轮尝试，逐步放宽约束条件
    pass = 1
search:
    // 每次 search 给一个时间预算（实际是递归深度上限）
    time = sameChannels ? 256 : 16384
    tmpGraph.nChannels = 0

    ncclTopoSearchRec(system, &tmpGraph, graph, &time)       // graph 充当 saveGraph

    // 3. 满意就结束
    if time == -1:          goto done               // 达到 maxChannels，完美解
    if graph.nChannels * graph.bwInter >= totalBw:  goto done  // 带宽够用了

    // 4. pass 1：还没找到满意解，逐个放宽条件重试（goto search）
    if pass == 1:
        if sameChannels == 1:
            sameChannels = 0;  goto search         // 放弃复用，允许不同排列

        if typeIntra < maxIntra:                     // 放宽链路类型（PATH_PHB→PATH_SYS→...）
            typeIntra += 1;  goto search

        if typeInter < maxInter:                     // 放宽节点间链路类型
            typeInter += 1;  goto search

        if crossNic == 0:                           // 允许跨 NIC
            crossNic = 2;  goto search

        // 以上条件都放宽过了还不行，降带宽重试
        if speedIndex < nspeeds - 1:
            speedIndex++
            tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex]
            goto search

        // 所有方案都试过了，reset speedIndex，进入 done
        speedIndex = 0

done:
    // 5. pass 1 找到了一个基础解（在 graph/saveGraph 里），进入 pass 2
    if pass == 1:
        ncclTopoDupChannels(graph, ccMin, ngpus)    // channel 数翻倍，带宽减半
        // 例：1 channel × 45 → 2 channel × 22.5（总带宽不变）
        memcpy(tmpGraph, graph)                     // 从 DupChannels 后的 graph 出发

        // 关键：speedIndex 归零，基于 DupChannels 后的（更低的）带宽重新定位
        speedIndex = 0
        while speedArray[speedIndex] > graph.bwInter:
            speedIndex++                            // 找到第一个 ≤ DupChannels 后带宽的速度
        // 例：graph.bwInter=22.5 → speedIndex 停在 20（speedArray 里第一个 ≤ 22.5 的值）
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex]
        tmpGraph.minChannels = graph.nChannels      // 要求至少保持已有的 channel 数
        pass = 2

    // 6. pass 2：从 DupChannels 后的带宽往上跳一档，看能否找到更优解
    if pass == 2:
        if time != 0 and speedIndex > 0:
            if pattern == RING:
                speedIndex--                        // 例：20 → 24（speedArray 上一档）
                tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex]
                goto search                         // 用 2 channel × 24 = 48 重搜，比原来的 1 × 45 = 45 更优
            elif pattern == NVLS and bwInter == graph.bwInter and bwInter < bwIntra * 2:
                tmpGraph.bwInter = speedArray[--speedIndex]  // 升节点间带宽
                goto search
            elif bwIntra == graph.bwIntra and bwIntra < bwInter * 2:
                tmpGraph.bwIntra = speedArray[--speedIndex]  // 升节点内带宽（树/collnet）
                goto search

    return graph   // 最终结果在 saveGraph（即传入的 graph）里
```

**核心意图**：从最严格的条件开始（高带宽、窄链路类型、复用 channel），
找不到就逐步放宽（降带宽、放宽链路类型、允许不同排列），直到找到满足需求的方案。

---

## ncclTopoSearchRec(ncclTopoSystem* system, ncclTopoGraph* graph, ncclTopoGraph* saveGraph, int* time) — 搜索一个新 channel 的起点

```
ncclTopoSearchRec(system, graph, saveGraph, time):
    backToNet, backToFirstRank = ncclTopoSearchParams(system, graph.pattern, &backToNet, &backToFirstRank)

    if system.inter:                               // 跨节点
        ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time)
        return

    // 纯节点内

    if pattern == NVLS:
        ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, graph.nChannels)
        return

    if graph.nChannels == 0:
        // 第一个 channel：用 PCI 顺序试 GPU 0 作为起点
        ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0)

    else:
        // 后续 channel：先尝试 Replay（复用上一个 channel 的 GPU 排列）
        g = ncclTopoReplayGetGpu(system, graph, -1, &g)    // 从 intra[] 取下一个 GPU rank
        ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g)

    // 如果不复用 channel，暴力遍历所有 GPU 作为起点
    if sameChannels == 0 or graph.nChannels == 0:
        for g = 0 .. system.nodes[GPU].count - 1:
            ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g)

    return
```

**核心意图**：为新的 channel 选一个 GPU 作为环的起点。
- 第一个 channel：试 PCI 顺序 + 所有 GPU
- 后续 channel：先 Replay（快），再暴力遍历（保底）

---

## ncclTopoSearchTryGpu(ncclTopoSystem* system, ncclTopoGraph* graph, ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank, int forcedOrder, int* time, int type, int index, int g) — 尝试选一个 GPU 并递归

```
ncclTopoSearchTryGpu(system, graph, saveGraph, step, backToNet, backToFirstRank, forcedOrder, time, type, index, g):
    // 1. 从当前位置走到目标 GPU g，扣带宽
    ncclTopoFollowPath(system, graph, type, index, GPU, g, 1.0, &gpu)
    //    成功 → gpu 非 NULL（带宽够）
    //    失败 → gpu = NULL（带宽不够或路径不通）

    if gpu != NULL:
        gpu.used ^= flag                             // 标记 GPU 已用
        ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time)
        gpu.used ^= flag                             // 取消标记

        ncclTopoFollowPath(system, graph, type, index, GPU, g, -1.0, &gpu)   // 回加带宽（回退）

    return
```

**核心意图**：ncclTopoFollowPath 扣带宽 → 递归 ncclTopoSearchRecGpu → ncclTopoFollowPath 还带宽。
是一个"借-用-还"的模式：先借带宽走到目标 GPU，递归搜索后续 GPU，最后还回带宽。

---

## ncclTopoSearchRecGpu(ncclTopoSystem* system, ncclTopoGraph* graph, ncclTopoGraph* saveGraph, ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int* time) — 递归构建环（核心递归函数）

```
ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time):
    if time <= 0:  return                            // 带宽预算耗尽，停
    time--

    ngpus = system.nodes[GPU].count

    // ─── 基线条件：环闭合 ───
    if step == ngpus:
        graph.nChannels++
        copy = 0
        ncclTopoCompareGraphs(system, graph, saveGraph, &copy)  // 比 saveGraph 更好吗？
        if copy:
            memcpy(saveGraph, graph)                // 更新最优解
            if graph.nChannels == graph.maxChannels:
                time = -1                           // 完美解，通知上层停止

        if graph.nChannels < graph.maxChannels:
            ncclTopoSearchRec(system, graph, saveGraph, time)   // ★递归：搜索下一个 channel

        graph.nChannels--                           // 回退，让上层尝试其他排列
        return

    // ─── 递归步骤：选下一个 GPU ───
    graph.intra[nChannels * ngpus + step] = gpu.rank    // 记录当前 GPU 在环中的位置
    g = gpu - system.nodes[GPU].nodes                    // 当前 GPU 的索引

    if step == backToNet:
        // 走网络分支（跨节点时回到 NIC）
        ncclTopoSelectNets(system, graph.typeInter, g, nets, &netCount)
        for n in nets:
            ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1.0, &net)
            if net != NULL:
                graph.inter[nChannels * 2 + 1] = net.id
                ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank, forcedOrder, time)
                ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1.0, &net)  // 还带宽

    elif step < ngpus - 1:
        // 选下一个 GPU 候选
        if forcedOrder == FORCED_ORDER_PCI:
            next[0] = step + 1;  count = 1           // PCI 顺序
        elif forcedOrder == FORCED_ORDER_REPLAY:
            ncclTopoReplayGetGpu(system, graph, step, next);  count = 1  // 复用上 channel 排列
        else:
            ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, sortNet)  // 按带宽排序的所有可达 GPU

        for i = 0 .. count - 1:
            ncclTopoSearchTryGpu(system, graph, saveGraph, step + 1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i])

    elif step == backToFirstRank:
        // 闭环：从最后一个 GPU 走回第一个 GPU
        ncclTopoFollowPath(system, graph, GPU, g, GPU, firstGpuIndex, 1.0, &node)
        if node != NULL:
            ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step + 1, backToNet, backToFirstRank, forcedOrder, time)  // 进入 step==ngpus 分支
        ncclTopoFollowPath(system, graph, GPU, g, GPU, firstGpuIndex, -1.0, &node)  // 还带宽

    return
```

**核心意图**：
- 递归构建环：step 0→1→2→...→ngpus-1，每步选一个 GPU
- step==ngpus 时环闭合：比较是否更优 → 保存 → 递归 ncclTopoSearchRec 搜索下一个 channel
- 这是一个**双重递归**：
  - 外层递归（ncclTopoSearchTryGpu→ncclTopoSearchRecGpu→ncclTopoSearchTryGpu→...）：构建一个完整的环
  - 内层递归（step==ngpus 时 ncclTopoSearchRec→ncclTopoSearchTryGpu→ncclTopoSearchRecGpu→...）：搜索下一个 channel

---

## ncclTopoFollowPath(ncclTopoSystem* system, ncclTopoGraph* graph, int type1, int index1, int type2, int index2, float mult, ncclTopoNode** node) — 沿拓扑路径走，扣/还带宽

```
ncclTopoFollowPath(system, graph, type1, index1, type2, index2, mult, node):
    // type1==-1 表示起点（环的第一个 GPU），没有"上一跳"
    if type1 == -1:
        *node = system.nodes[type2].nodes + index2   // 直接返回终点
        return SUCCESS

    node1 = system.nodes[type1].nodes + index1
    path = node1.paths[type2] + index2               // 预计算的路径
    revPath = node2.paths[type1] + index1            // 反向路径（树模式用）

    if path == NULL:  return ERROR                  // 路径不存在

    *node = NULL                                     // 默认失败

    if path.type >= PATH_DIS:  return SUCCESS       // 路径断开，返回 NULL
    if mult == 1 and path.type > type:  return SUCCESS  // 链路类型降级，拒绝
    if mult == 1 and 树模式 and revPath.type > type:  return SUCCESS  // 同上，树模式特判

    bw = (intra ? graph.bwIntra : graph.bwInter) * mult   // 要扣（mult>0）或还（mult<0）的带宽

    // 沿路径逐跳扣/还带宽
    step = 0
    followPath(path, node1, path.count, bw, &step)   // 逐跳扣，带宽不够则提前停在 step
    if step < path.count:  goto rewind              // 带宽不够，回退

    // 成功：到达终点
    graph.nHops += mult * path.count
    *node = system.nodes[type2].nodes + index2
    return SUCCESS

rewind:
    // 带宽不够：把已扣的还回去
    followPath(path, node1, step, -bw, &step)        // 逐跳还
    return SUCCESS   // *node 保持 NULL，表示没走通
```

**核心意图**：
- 沿预计算路径逐跳扣带宽（mult>0）或还带宽（mult<0）
- 带宽不够就回退已扣的部分，返回 NULL
- type1==-1 是特例：环的第一个 GPU 没有"来源"，直接返回终点
- mult 正负实现扣/还的对称性：ncclTopoSearchTryGpu 先 +1.0 扣，递归回来后 -1.0 还

---

## 调用关系总览

```
ncclTopoCompute(system, graph)
  └─ ncclTopoSearchRec(system, graph, saveGraph, time)                          ← 为新 channel 选起点
       └─ ncclTopoSearchTryGpu(system, graph, saveGraph, step, ..., g, time)    ← 借带宽走到 g
            ├─ ncclTopoFollowPath(system, graph, type, index, GPU, g, +1.0, &gpu)  ← 扣带宽
            └─ ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, ..., time)  ← 递归建环
                 ├─ step < ngpus:
                 │    └─ ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, ..., nextGpu, time)  ← 递归选下一个 GPU
                 │         ├─ ncclTopoFollowPath(system, graph, GPU, g, GPU, nextGpu, +1.0, &gpu)
                 │         └─ ncclTopoSearchRecGpu(system, graph, saveGraph, nextGpu, step+1, ..., time)
                 │              └─ ...
                 │
                 └─ step == ngpus (环闭合):
                      ├─ ncclTopoCompareGraphs(system, graph, saveGraph, &copy)  ← 比 saveGraph 更优？
                      ├─ memcpy(saveGraph, graph)     ← 更新最优解
                      └─ ncclTopoSearchRec(system, graph, saveGraph, time)       ← ★递归：搜索下一个 channel
                           └─ ncclTopoSearchTryGpu(system, graph, saveGraph, 0, ..., g, time)
                                └─ ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, 1, ..., time)
                                     └─ ...
            └─ ncclTopoFollowPath(system, graph, type, index, GPU, g, -1.0, &gpu)  ← 还带宽
```

**双重递归**：
1. **建环递归**：ncclTopoSearchTryGpu → ncclTopoSearchRecGpu → ncclTopoSearchTryGpu → ... → step==ngpus（环完成）
2. **channel 递归**：step==ngpus → ncclTopoSearchRec → ncclTopoSearchTryGpu → ncclTopoSearchRecGpu → ...（下一个 channel）

**带宽管理**：每次 ncclTopoFollowPath(+1.0) 扣，对应一次 ncclTopoFollowPath(-1.0) 还。
带宽不够时 ncclTopoFollowPath 内部自动回退，返回 NULL，上层换一个 GPU 试。
