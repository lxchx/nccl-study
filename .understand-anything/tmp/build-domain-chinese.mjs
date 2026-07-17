#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'fs';

const g = JSON.parse(readFileSync('.understand-anything/knowledge-graph.json', 'utf8'));

const domainGraph = {
  version: '1.0.0',
  project: {
    ...g.project,
    description: 'GPU 间通信原语库 — 业务域知识图谱'
  },
  nodes: [],
  edges: [],
  layers: [],
  tour: []
};

// ============================================================
// DOMAINS (6) - Chinese
// ============================================================
const domains = [
  { id: 'domain:collective-operations', type: 'domain', name: 'Collective 集合通信操作',
    summary: 'NCCL 的核心业务 — 实现 GPU collective 通信原语：all-reduce、all-gather、reduce-scatter、broadcast 和 reduce。每个操作必须跨任意数量的 GPU 以最优性能运行。',
    tags: ['核心','gpu-通信','collectives'], complexity: 'complex',
    domainMeta: { entities: ['ncclComm_t (通信器)','ncclOp_t (操作类型)','ncclRedOp_t (归约算子)','ncclDataType_t (数据类型)'], businessRules: ['通信器内的所有 rank 必须参与相同的 collective 操作','操作在所有 rank 完成前是阻塞的','必须支持任意数据类型，每个 kernel 最多 4GB'], crossDomainInteractions: ['使用设备执行层运行 GPU 内核','依赖传输后端进行实际数据传输','依赖拓扑路由层进行算法选择'] } },
  { id: 'domain:device-execution', type: 'domain', name: '设备端执行 (GPU Kernel)',
    summary: 'collective 算法的 GPU 端实现。包含 ring 和 tree 算法内核、primitives（LL/LL128/Simple/Oneshot）以及 LSA barrier 机制，直接在 GPU 上运行以实现最大带宽。',
    tags: ['cuda','gpu-内核','性能关键'], complexity: 'complex',
    domainMeta: { entities: ['Ring 算法内核','Tree 算法内核','LSA barrier','对称内存操作'], businessRules: ['内核必须是 warp-cooperative 且正确同步','内存合并对齐对 NVLink 带宽至关重要','内核选择取决于数据大小、数据类型和 rank 数量'], crossDomainInteractions: ['由 Collective 操作层编排'] } },
  { id: 'domain:transport-backends', type: 'domain', name: '传输后端',
    summary: '硬件通信通道实现。每个后端（P2P/NVLink、Shared Memory、InfiniBand、TCP/IP）实现 ncclTransport 接口，但针对其特定硬件进行了优化。',
    tags: ['硬件','网络','nvlink','infiniband'], complexity: 'complex',
    domainMeta: { entities: ['ncclTransport (接口)','ncclNet_t (网络插件)','ncclPeerInfo_t','连接句柄 (QP, FD)'], businessRules: ['传输选择优先级：P2P > SHM > IB > Socket','每个传输必须支持连接建立、发送、接收和清理','IB 需要 MR 注册以支持 RDMA 操作'], crossDomainInteractions: ['通过通道被 Collective 操作层使用','拓扑路由层决定哪些传输可用'] } },
  { id: 'domain:topology-routing', type: 'domain', name: '拓扑与路由',
    summary: 'GPU 互连拓扑发现和最优路由路径计算。映射 NVLink、PCIe 层级和 CPU 互连构建图模型，然后搜索带宽最大化的 ring/tree 路径。',
    tags: ['图算法','拓扑','性能调优'], complexity: 'complex',
    domainMeta: { entities: ['ncclTopo_t (拓扑图)','ncclTopoNode_t (GPU/CPU/Switch 节点)','Ring 路径','Tree 结构'], businessRules: ['在任何 collective 操作前必须完成拓扑发现','每个通道的 Ring 路径必须恰好访问每个 GPU 一次','算法选择取决于拓扑结构和消息大小'], crossDomainInteractions: ['为 Collective 操作层提供路由数据','通知传输后端哪些连接可用'] } },
  { id: 'domain:initialization-lifecycle', type: 'domain', name: '初始化与生命周期',
    summary: 'NCCL communicator 的 Bootstrap 和生命周期管理。涵盖进程发现、CUDA 上下文设置、通道创建、communicator 初始化和清理流程。',
    tags: ['bootstrap','生命周期','cuda-上下文'], complexity: 'moderate',
    domainMeta: { entities: ['ncclComm_t','ncclGroup (组操作)','代理线程','Bootstrap socket'], businessRules: ['所有 rank 必须在任何 collective 前以匹配的 rank/count 调用 ncclCommInitRank','GroupStart/End 划定集体操作边界','代理线程异步处理正在进行的通信'], crossDomainInteractions: ['触发拓扑路由层发现','设置传输后端','为 Collective 操作创建通道'] } },
  { id: 'domain:plugin-ecosystem', type: 'domain', name: '插件生态系统',
    summary: 'NCCL 的可扩展性框架。支持可插拔的网络后端、性能分析器（inspector）、算法调优器、GIN 支持和 RMA。',
    tags: ['可扩展性','插件-API','性能分析'], complexity: 'moderate',
    domainMeta: { entities: ['ncclNetPlugin_t (v5)','ncclTunerPlugin_t','ncclProfilerPlugin_t','ncclGinPlugin_t'], businessRules: ['插件通过 LD_PRELOAD 或 dlopen 在初始化时加载','每种插件类型有版本化的 API 契约','自定义 net 插件可用时可替换内置传输'], crossDomainInteractions: ['Net 插件扩展传输后端','Profiler 插件监控 Collective 操作'] } },
];

// ============================================================
// FLOWS (12) - Chinese
// ============================================================
const flows = [
  { id: 'flow:communicator-initialization', type: 'flow', name: 'Communicator 初始化流程',
    summary: '完整的 ncclCommInitRank 流程：Bootstrap → 设备检测 → 拓扑发现 → 传输选择 → 通道创建 → 准备就绪。',
    tags: ['生命周期','bootstrap'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclCommInitRank(rootComm, nRanks, rank)', entryType: 'api' } },
  { id: 'flow:group-operation-lifecycle', type: 'flow', name: '组操作流程',
    summary: 'ncclGroupStart → [collectives] → ncclGroupEnd 序列，将多个 collective 操作批处理为并发执行。',
    tags: ['生命周期','批处理'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclGroupStart() / ncclGroupEnd()', entryType: 'api' } },
  { id: 'flow:all-reduce-execution', type: 'flow', name: 'All-Reduce 执行流程',
    summary: 'ncclAllReduce 流程：入队工作 → 选择算法（ring/tree）→ 启动 GPU 内核 → 代理线程协调 → 完成通知。',
    tags: ['collective','allreduce'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm)', entryType: 'api' } },
  { id: 'flow:all-gather-execution', type: 'flow', name: 'All-Gather 执行流程',
    summary: 'ncclAllGather 流程：使用 ring 或 tree 算法从所有 GPU 收集数据段到每个 GPU。',
    tags: ['collective','allgather'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclAllGather(sendbuff, recvbuff, count, datatype, comm)', entryType: 'api' } },
  { id: 'flow:reduce-scatter-execution', type: 'flow', name: 'Reduce-Scatter 执行流程',
    summary: 'ncclReduceScatter 流程：跨所有 rank 归约数据然后散射结果段。大模型训练的关键操作。',
    tags: ['collective','reducescatter'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm)', entryType: 'api' } },
  { id: 'flow:topology-discovery', type: 'flow', name: 'GPU 拓扑发现流程',
    summary: 'NVML 驱动的 GPU 发现 → PCIe 层级映射 → NVLink 链路检测 → 拓扑图构建。',
    tags: ['拓扑','nvml'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclTopoInit(comm)', entryType: 'internal' } },
  { id: 'flow:routing-computation', type: 'flow', name: 'Ring/Tree 路由计算流程',
    summary: '最优 ring 路径的图搜索 → tree 构建 → 基于带宽评估的算法选择（ring vs tree）。',
    tags: ['路由','图算法'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclTopoSetChannels(comm)', entryType: 'internal' } },
  { id: 'flow:transport-connection-setup', type: 'flow', name: '传输连接建立流程',
    summary: '交换对端信息 → 选择传输（P2P/SHM/IB/Sock）→ 建立连接 → 注册内存区域 → 准备数据传输。',
    tags: ['传输','连接'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclTransportConnect(comm, channels)', entryType: 'internal' } },
  { id: 'flow:plugin-loading', type: 'flow', name: '插件加载与注册流程',
    summary: 'LD_PRELOAD/dlopen 插件发现 → 版本协商 → API 注册 → 接入 NCCL 生命周期。',
    tags: ['插件','ld-preload'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclNetPluginLoad() / ncclTunerPluginLoad()', entryType: 'internal' } },
  { id: 'flow:kernel-launch-pipeline', type: 'flow', name: '内核启动流水线',
    summary: '算法选择 → 内核参数准备 → CUDA 内核启动 → LSA barrier 同步 → 设备端 collective 执行。',
    tags: ['cuda','内核启动'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclLaunch(kernelArgs)', entryType: 'internal' } },
  { id: 'flow:proxy-thread-loop', type: 'flow', name: '代理线程循环',
    summary: '后台代理线程轮询所有活跃传输的完成状态 → 处理 IB/SHM 数据传输 → 向调度器报告完成。',
    tags: ['代理','异步'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclProxyThread(comm)', entryType: 'internal' } },
  { id: 'flow:python-binding-call', type: 'flow', name: 'Python 绑定调用路径',
    summary: 'Python API 调用 → Cython 绑定 → nccl4py communicator 包装器 → C NCCL API → 原生执行。',
    tags: ['python','cython','nccl4py'], complexity: 'simple',
    domainMeta: { entryPoint: 'comm.allreduce(tensor) (Python)', entryType: 'api' } },
];

// ============================================================
// STEPS - Chinese
// ============================================================
const steps = [
  // Communicator Initialization
  { id: 'step:communicator-initialization:bootstrap', type: 'step', name: 'Rank 间 Bootstrap', summary: '通过 bootstrap 层建立所有 rank 之间的初始 TCP socket 连接。根 rank 创建服务端，其他连接过来。', tags: ['bootstrap'], complexity: 'moderate', filePath: 'src/bootstrap.cc' },
  { id: 'step:communicator-initialization:device-detection', type: 'step', name: '设备检测', summary: '查询当前 rank 的 CUDA 设备，获取 GPU 数量、设备能力和 CUDA compute 版本。', tags: ['cuda'], complexity: 'simple', filePath: 'src/init.cc' },
  { id: 'step:communicator-initialization:topo-discovery', type: 'step', name: '拓扑发现', summary: '使用 NVML 发现 GPU 互连拓扑：NVLink 链路、PCIe 交换机、CPU socket。', tags: ['nvml'], complexity: 'complex', filePath: 'src/graph/topo.cc' },
  { id: 'step:communicator-initialization:routing-calculation', type: 'step', name: 'Ring/Tree 路径计算', summary: '基于发现的拓扑，为所有通道计算最优 ring 和 tree 路由路径。', tags: ['路由'], complexity: 'complex', filePath: 'src/graph/connect.cc' },
  { id: 'step:communicator-initialization:channel-setup', type: 'step', name: '通道创建与传输设置', summary: '创建 N 个通道，为每个选择传输方式，建立连接。', tags: ['通道'], complexity: 'moderate', filePath: 'src/channel.cc' },
  { id: 'step:communicator-initialization:proxy-launch', type: 'step', name: '代理线程启动', summary: '生成代理线程，处理持续的传输轮询和数据移动。', tags: ['代理'], complexity: 'simple', filePath: 'src/proxy.cc' },

  // All-Reduce Execution
  { id: 'step:all-reduce-execution:enqueue', type: 'step', name: '入队 Collective 工作', summary: '验证参数，分配内部工作结构，基于数据大小和拓扑选择算法（ring/tree）。', tags: ['调度器'], complexity: 'moderate', filePath: 'src/enqueue.cc' },
  { id: 'step:all-reduce-execution:param-calc', type: 'step', name: '计算内核参数', summary: '为选定算法计算 chunk 大小、线程数和内存布局。', tags: ['调度器'], complexity: 'moderate', filePath: 'src/enqueue.cc' },
  { id: 'step:all-reduce-execution:kernel-launch', type: 'step', name: '启动 GPU 内核', summary: '编译内核参数，在每个活跃通道上启动 CUDA 内核。内核实现 ring/tree all-reduce。', tags: ['cuda','内核'], complexity: 'complex', filePath: 'src/device/all_reduce.h' },
  { id: 'step:all-reduce-execution:proxy-sync', type: 'step', name: '代理线程同步', summary: '代理线程轮询传输的 IB/SHM 完成状态，处理 collective 的异步数据传输。', tags: ['代理'], complexity: 'moderate', filePath: 'src/proxy.cc' },
  { id: 'step:all-reduce-execution:completion', type: 'step', name: '完成通知', summary: '所有通道报告完成 → CUDA stream 同步 → ncclResult_t 返回给调用者。', tags: ['生命周期'], complexity: 'simple', filePath: 'src/collectives.cc' },

  // Topology Discovery
  { id: 'step:topology-discovery:nvml-query', type: 'step', name: 'NVML GPU 查询', summary: '通过 NVML 查询 GPU PCI bus ID、NVLink 状态和链路速度。', tags: ['nvml'], complexity: 'moderate', filePath: 'src/graph/topo.cc' },
  { id: 'step:topology-discovery:pcie-map', type: 'step', name: 'PCIe 层级映射', summary: '遍历 sysfs 构建 PCIe 交换机拓扑：根端口、交换机、端点。通过 NUMA 将 GPU 映射到 CPU socket。', tags: ['sysfs','pci'], complexity: 'complex', filePath: 'src/graph/topo.cc' },
  { id: 'step:topology-discovery:graph-build', type: 'step', name: '拓扑图构建', summary: '从 NVML + sysfs 数据构建带带宽标注的 ncclTopo_t 图。', tags: ['图'], complexity: 'complex', filePath: 'src/graph/topo.cc' },

  // Routing Computation
  { id: 'step:routing-computation:search', type: 'step', name: 'Ring 路径搜索', summary: '在拓扑图中使用带宽最大化搜索找到最优 ring 路径。', tags: ['图算法'], complexity: 'complex', filePath: 'src/graph/search.cc' },
  { id: 'step:routing-computation:tree-build', type: 'step', name: 'Tree 构建', summary: '为 broadcast/reduce 操作在 GPU 拓扑上构建 tree 结构。', tags: ['图算法'], complexity: 'moderate', filePath: 'src/graph/connect.cc' },
  { id: 'step:routing-computation:algo-select', type: 'step', name: '算法选择', summary: '基于数据大小、拓扑带宽和 GPU 数量选择 ring 或 tree 算法。', tags: ['调优'], complexity: 'moderate', filePath: 'src/graph/tuning.cc' },

  // Transport Connection Setup
  { id: 'step:transport-connection-setup:peer-exchange', type: 'step', name: '交换对端信息', summary: '在所有 rank 之间共享 GPU bus ID、CUDA 设备和缓冲区地址。', tags: ['bootstrap'], complexity: 'simple', filePath: 'src/transport.cc' },
  { id: 'step:transport-connection-setup:transport-select', type: 'step', name: '传输选择', summary: '为每对 GPU 选择最快的可用传输：P2P → SHM → IB → Socket。', tags: ['传输'], complexity: 'moderate', filePath: 'src/transport.cc' },
  { id: 'step:transport-connection