#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'fs';

const g = JSON.parse(readFileSync('.understand-anything/knowledge-graph.json', 'utf8'));

const nodes = [];

// ============================================================
// DOMAINS (6)
// ============================================================
nodes.push(
  { id: 'domain:collective-operations', type: 'domain', name: 'Collective 集合通信操作',
    summary: 'NCCL 的核心业务 — GPU collective 通信原语：all-reduce、all-gather、reduce-scatter、broadcast、reduce。每个操作必须跨任意数量的 GPU 以最优性能运行。',
    tags: ['核心','gpu-通信','collectives'], complexity: 'complex',
    domainMeta: { entities: ['ncclComm_t (通信器)','ncclOp_t (操作类型)','ncclRedOp_t (归约算子)'], businessRules: ['所有 rank 必须参与相同的 collective 操作','操作在所有 rank 完成前阻塞','支持任意数据类型，每 kernel 最多 4GB'], crossDomainInteractions: ['使用设备执行层运行 GPU 内核','依赖传输后端进行数据传输','依赖拓扑路由层进行算法选择'] } },
  { id: 'domain:device-execution', type: 'domain', name: '设备端执行 (GPU Kernel)',
    summary: 'collective 算法的 GPU 端实现。包含 ring/tree 算法内核、primitives（LL/LL128/Simple）以及 LSA barrier 机制，直接在 GPU 运行以实现最大带宽。',
    tags: ['cuda','gpu-内核','性能关键'], complexity: 'complex',
    domainMeta: { entities: ['Ring 算法内核','Tree 算法内核','LSA barrier','对称内存操作'], businessRules: ['内核必须 warp-cooperative 且正确同步','内存合并对齐对 NVLink 带宽至关重要','内核选择取决于数据大小和 rank 数量'], crossDomainInteractions: ['由 Collective 操作层编排'] } },
  { id: 'domain:transport-backends', type: 'domain', name: '传输后端',
    summary: '硬件通信通道实现。每个后端（P2P/NVLink、SHM、InfiniBand、TCP/IP）实现 ncclTransport 接口，针对其硬件优化。NCCL 自动选择最快的可用传输。',
    tags: ['硬件','网络','nvlink','infiniband'], complexity: 'complex',
    domainMeta: { entities: ['ncclTransport (接口)','ncclNet_t (网络插件)','连接句柄 (QP, FD)'], businessRules: ['选择优先级：P2P > SHM > IB > Socket','每个传输必须支持连接/发送/接收/清理','IB 需要 MR 注册以支持 RDMA'], crossDomainInteractions: ['被 Collective 操作层通过通道使用','拓扑路由层决定哪些传输可用'] } },
  { id: 'domain:topology-routing', type: 'domain', name: '拓扑与路由',
    summary: 'GPU 互连拓扑发现和最优路由计算。映射 NVLink、PCIe 层级和 CPU 互连构建图模型，搜索带宽最大化的 ring/tree 路径。',
    tags: ['图算法','拓扑','性能调优'], complexity: 'complex',
    domainMeta: { entities: ['ncclTopo_t (拓扑图)','Ring 路径','Tree 结构'], businessRules: ['任何 collective 前必须完成拓扑发现','每个通道 Ring 路径恰好访问每个 GPU 一次','算法选择取决于拓扑和消息大小'], crossDomainInteractions: ['为 Collective 操作层提供路由数据','通知传输后端可用连接'] } },
  { id: 'domain:initialization-lifecycle', type: 'domain', name: '初始化与生命周期',
    summary: 'NCCL communicator 的 Bootstrap 和生命周期管理。涵盖进程发现、CUDA 上下文设置、通道创建、communicator 初始化和清理流程。',
    tags: ['bootstrap','生命周期'], complexity: 'moderate',
    domainMeta: { entities: ['ncclComm_t','ncclGroup','代理线程','Bootstrap socket'], businessRules: ['所有 rank 必须以匹配的 rank/count 调用 ncclCommInitRank','GroupStart/End 划定集体操作边界','代理线程异步处理通信'], crossDomainInteractions: ['触发拓扑路由层发现','设置传输后端','为 Collective 操作创建通道'] } },
  { id: 'domain:plugin-ecosystem', type: 'domain', name: '插件生态系统',
    summary: 'NCCL 的可扩展性框架。支持可插拔的网络后端、性能分析器（inspector）、算法调优器、GIN 支持和 RMA。',
    tags: ['可扩展性','插件-API'], complexity: 'moderate',
    domainMeta: { entities: ['ncclNetPlugin_t (v5)','ncclTunerPlugin_t','ncclProfilerPlugin_t'], businessRules: ['插件通过 LD_PRELOAD 或 dlopen 加载','每种插件类型有版本化的 API 契约','自定义 net 插件可替换内置传输'], crossDomainInteractions: ['Net 插件扩展传输后端','Profiler 插件监控 Collective 操作'] } },
);

// ============================================================
// FLOWS (12)
// ============================================================
const flowMeta = [
  { id: 'flow:communicator-initialization', name: 'Communicator 初始化流程', summary: '完整 ncclCommInitRank 流程：Bootstrap → 设备检测 → 拓扑发现 → 传输选择 → 通道创建。', domain: 'initialization-lifecycle' },
  { id: 'flow:group-operation-lifecycle', name: '组操作流程', summary: 'ncclGroupStart → [collectives] → ncclGroupEnd 序列，批量并发执行多个 collective 操作。', domain: 'initialization-lifecycle' },
  { id: 'flow:all-reduce-execution', name: 'All-Reduce 执行流程', summary: '入队工作 → 选择算法（ring/tree）→ GPU 内核启动 → 代理线程协调 → 完成通知。', domain: 'collective-operations' },
  { id: 'flow:all-gather-execution', name: 'All-Gather 执行流程', summary: '使用 ring 或 tree 算法从所有 GPU 收集数据段到每个 GPU。', domain: 'collective-operations' },
  { id: 'flow:reduce-scatter-execution', name: 'Reduce-Scatter 执行流程', summary: '跨所有 rank 归约数据然后散射结果段。大模型训练的关键操作。', domain: 'collective-operations' },
  { id: 'flow:topology-discovery', name: 'GPU 拓扑发现流程', summary: 'NVML GPU 发现 → PCIe 层级映射 → NVLink 链路检测 → 拓扑图构建。', domain: 'topology-routing' },
  { id: 'flow:routing-computation', name: 'Ring/Tree 路由计算', summary: '最优 ring 路径图搜索 → tree 构建 → 基于带宽评估的算法选择。', domain: 'topology-routing' },
  { id: 'flow:transport-connection-setup', name: '传输连接建立流程', summary: '交换对端信息 → 选择传输（P2P/SHM/IB/Sock）→ 建立连接 → 注册内存区域。', domain: 'transport-backends' },
  { id: 'flow:plugin-loading', name: '插件加载与注册', summary: 'LD_PRELOAD/dlopen 发现 → 版本协商 → API 注册 → 接入 NCCL 生命周期。', domain: 'plugin-ecosystem' },
  { id: 'flow:kernel-launch-pipeline', name: '内核启动流水线', summary: '算法选择 → 参数准备 → CUDA 内核启动 → LSA barrier 同步 → 设备端执行。', domain: 'device-execution' },
  { id: 'flow:proxy-thread-loop', name: '代理线程循环', summary: '后台轮询所有传输完成状态 → 处理 IB/SHM 数据传输 → 向调度器报告完成。', domain: 'transport-backends' },
  { id: 'flow:python-binding-call', name: 'Python 绑定调用路径', summary: 'Python API → Cython 绑定 → nccl4py 通信器 → C NCCL API → 原生执行。', domain: 'plugin-ecosystem' },
];

flowMeta.forEach(f => {
  nodes.push({ id: f.id, type: 'flow', name: f.name, summary: f.summary, tags: ['流程'], complexity: 'moderate', domainMeta: { entryPoint: '', entryType: 'api' } });
});

// ============================================================
// STEPS (key implementation steps)
// ============================================================
const stepDefs = [
  // Communicator Initialization
  { id: 'step:communicator-initialization:bootstrap', name: 'Rank 间 Bootstrap', summary: '通过 bootstrap 层建立所有 rank 间的初始 TCP socket 连接。', flow: 'flow:communicator-initialization', filePath: 'src/bootstrap.cc' },
  { id: 'step:communicator-initialization:device-detection', name: '设备检测', summary: '查询当前 rank 的 CUDA 设备，获取 GPU 数量和 capabilities。', flow: 'flow:communicator-initialization', filePath: 'src/init.cc' },
  { id: 'step:communicator-initialization:topo-discovery', name: '拓扑发现', summary: '用 NVML 发现 GPU 互连拓扑：NVLink、PCIe 交换机、CPU socket。', flow: 'flow:communicator-initialization', filePath: 'src/graph/topo.cc' },
  { id: 'step:communicator-initialization:routing-calculation', name: 'Ring/Tree 路径计算', summary: '基于拓扑为所有通道计算最优 ring/tree 路由。', flow: 'flow:communicator-initialization', filePath: 'src/graph/connect.cc' },
  { id: 'step:communicator-initialization:channel-setup', name: '通道创建与传输设置', summary: '创建 N 个通道，选择传输方式，建立连接。', flow: 'flow:communicator-initialization', filePath: 'src/channel.cc' },
  { id: 'step:communicator-initialization:proxy-launch', name: '代理线程启动', summary: '生成代理线程处理持续传输轮询和数据移动。', flow: 'flow:communicator-initialization', filePath: 'src/proxy.cc' },
  
  // All-Reduce Execution
  { id: 'step:all-reduce-execution:enqueue', name: '入队 Collective 工作', summary: '验证参数，分配工作结构，选择算法（ring/tree）。', flow: 'flow:all-reduce-execution', filePath: 'src/enqueue.cc' },
  { id: 'step:all-reduce-execution:param-calc', name: '计算内核参数', summary: '为选定算法计算 chunk 大小、线程数、内存布局。', flow: 'flow:all-reduce-execution', filePath: 'src/enqueue.cc' },
  { id: 'step:all-reduce-execution:kernel-launch', name: '启动 GPU 内核', summary: '编译内核参数，在活跃通道上启动 CUDA 内核（ring/tree all-reduce）。', flow: 'flow:all-reduce-execution', filePath: 'src/device/all_reduce.h' },
  { id: 'step:all-reduce-execution:proxy-sync', name: '代理线程同步', summary: '轮询 IB/SHM 完成状态，处理异步数据传输。', flow: 'flow:all-reduce-execution', filePath: 'src/proxy.cc' },
  { id: 'step:all-reduce-execution:completion', name: '完成通知', summary: '所有通道报告完成 → CUDA stream 同步 → 返回结果。', flow: 'flow:all-reduce-execution', filePath: 'src/collectives.cc' },
  
  // Topology Discovery
  { id: 'step:topology-discovery:nvml-query', name: 'NVML GPU 查询', summary: '通过 NVML 查询 GPU PCI bus ID、NVLink 状态和链路速度。', flow: 'flow:topology-discovery', filePath: 'src/graph/topo.cc' },
  { id: 'step:topology-discovery:pcie-map', name: 'PCIe 层级映射', summary: '遍历 sysfs 构建 PCIe 交换机拓扑，通过 NUMA 映射 GPU 到 CPU socket。', flow: 'flow:topology-discovery', filePath: 'src/graph/topo.cc' },
  { id: 'step:topology-discovery:graph-build', name: '拓扑图构建', summary: '从 NVML + sysfs 数据构建带带宽标注的 ncclTopo_t 图。', flow: 'flow:topology-discovery', filePath: 'src/graph/topo.cc' },
  
  // Routing Computation
  { id: 'step:routing-computation:search', name: 'Ring 路径搜索', summary: '带宽最大化搜索找到最优 ring 路径。', flow: 'flow:routing-computation', filePath: 'src/graph/search.cc' },
  { id: 'step:routing-computation:tree-build', name: 'Tree 构建', summary: '为 broadcast/reduce 在 GPU 拓扑上构建 tree 结构。', flow: 'flow:routing-computation', filePath: 'src/graph/connect.cc' },
  { id: 'step:routing-computation:algo-select', name: '算法选择', summary: '基于数据大小、带宽和 GPU 数量选择 ring 或 tree。', flow: 'flow:routing-computation', filePath: 'src/graph/tuning.cc' },
  
  // Transport Connection Setup
  { id: 'step:transport-connection-setup:peer-exchange', name: '交换对端信息', summary: '在所有 rank 间共享 GPU bus ID、CUDA 设备、缓冲区地址。', flow: 'flow:transport-connection-setup', filePath: 'src/transport.cc' },
  { id: 'step:transport-connection-setup:transport-select', name: '传输选择', summary: '为每对 GPU 选最快传输：P2P → SHM → IB → Socket。', flow: 'flow:transport-connection-setup', filePath: 'src/transport.cc' },
  { id: 'step:transport-connection-setup:connect', name: '建立连接', summary: '创建传输特定连接：QP for IB、FD for socket、CUDA IPC for P2P。', flow: 'flow:transport-connection-setup', filePath: 'src/transport/net_ib/connect.cc' },
  
  // Plugin Loading
  { id: 'step:plugin-loading:dlopen', name: '插件发现与 dlopen', summary: '搜索 LD_PRELOAD、NCCL_NET_PLUGIN 环境变量和标准路径找 .so 文件。', flow: 'flow:plugin-loading', filePath: 'src/misc/param.cc' },
  { id: 'step:plugin-loading:register', name: 'API 注册', summary: '验证版本后注册插件入口点到 NCCL 内部表，替换或增强内置传输。', flow: 'flow:plugin-loading', filePath: 'src/plugin/net/net_plugin.cc' },
  
  // Kernel Launch Pipeline
  { id: 'step:kernel-launch-pipeline:algo-select', name: '算法选择', summary: '选 ring/tree 和 primitive（LL/LL128/Simple）基于数据大小和 dtype。', flow: 'flow:kernel-launch-pipeline', filePath: 'src/enqueue.cc' },
  { id: 'step:kernel-launch-pipeline:cuda-launch', name: 'CUDA 内核启动', summary: '在多个 stream 上启动 CUDA 内核，每个通道一个 stream。', flow: 'flow:kernel-launch-pipeline', filePath: 'src/device/all_reduce.h' },
  
  // Proxy Thread Loop
  { id: 'step:proxy-thread-loop:poll', name: '轮询传输完成状态', summary: '循环检查 IB、SHM、socket 的 pending 数据传输是否完成。', flow: 'flow:proxy-thread-loop', filePath: 'src/proxy.cc' },
  { id: 'step:proxy-thread-loop:handle-transfer', name: '处理数据传输', summary: '执行实际数据移动：IB post send/recv、SHM memcpy、socket read/write。', flow: 'flow:proxy-thread-loop', filePath: 'src/proxy.cc' },
];

stepDefs.forEach(s => {
  nodes.push({ id: s.id, type: 'step', name: s.name, summary: s.summary, tags: ['步骤'], complexity: 'moderate', filePath: s.filePath });
});

// ============================================================
// EDGES
// ============================================================
const edges = [];

// Domain → Flow (contains_flow)
flowMeta.forEach(f => {
  edges.push({ source: 'domain:' + f.domain, target: f.id, type: 'contains_flow', direction: 'forward', weight: 1.0 });
});

// Flow → Step (flow_step with order weights)
const stepsByFlow = {};
stepDefs.forEach(s => {
  if (!stepsByFlow[s.flow]) stepsByFlow[s.flow] = [];
  stepsByFlow[s.flow].push(s);
});

Object.entries(stepsByFlow).forEach(([flowId, steps]) => {
  steps.forEach((s, i) => {
    const weight = Math.round(((i + 1) / steps.length) * 10) / 10;
    edges.push({ source: flowId, target: s.id, type: 'flow_step', direction: 'forward', weight: weight });
  });
});

// Cross-domain interactions
edges.push(
  { source: 'domain:initialization-lifecycle', target: 'domain:topology-routing', type: 'cross_domain', direction: 'forward', description: '初始化触发拓扑发现', weight: 0.8 },
  { source: 'domain:initialization-lifecycle', target: 'domain:transport-backends', type: 'cross_domain', direction: 'forward', description: '初始化设置传输后端', weight: 0.8 },
  { source: 'domain:topology-routing', target: 'domain:collective-operations', type: 'cross_domain', direction: 'forward', description: '