#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'fs';

const g = JSON.parse(readFileSync('.understand-anything/knowledge-graph.json', 'utf8'));

// ============================================================
// DOMAINS (6)
// ============================================================
const nodes = [
  { id: 'domain:collective-operations', type: 'domain', name: 'Collective Operations',
    summary: 'Core of NCCL — GPU collective communication primitives: all-reduce, all-gather, reduce-scatter, broadcast, reduce across any number of GPUs with optimal performance.',
    tags: ['core','gpu-communication','collectives'], complexity: 'complex',
    domainMeta: { entities: ['ncclComm_t','ncclOp_t','ncclRedOp_t','ncclDataType_t'], businessRules: ['All ranks must participate in same collective','Blocking until completion across all ranks','Handles arbitrary data types up to 4GB per kernel'], crossDomainInteractions: ['Uses Device Execution for GPU kernels','Relies on Transport Backends for data movement','Depends on Topology & Routing for algorithm selection'] } },
  { id: 'domain:device-execution', type: 'domain', name: 'Device Execution (GPU Kernels)',
    summary: 'GPU-side collective algorithms: ring/tree kernel implementations, primitives (LL/LL128/Simple/Oneshot), and LSA barrier mechanisms for maximum bandwidth.',
    tags: ['cuda','gpu-kernel','performance-critical'], complexity: 'complex',
    domainMeta: { entities: ['Ring algorithm kernel','Tree algorithm kernel','LSA barrier','Symmetric memory ops'], businessRules: ['Kernels must be warp-cooperative with proper synchronization','Memory coalescing critical for NVLink bandwidth','Kernel selection depends on data size, dtype, and rank count'], crossDomainInteractions: ['Orchestrated by Collective Operations domain'] } },
  { id: 'domain:transport-backends', type: 'domain', name: 'Transport Backends',
    summary: 'Hardware communication backends: P2P/NVLink, Shared Memory, InfiniBand, TCP/IP. Each implements ncclTransport interface optimized for its hardware.',
    tags: ['hardware','network','nvlink','infiniband'], complexity: 'complex',
    domainMeta: { entities: ['ncclTransport','ncclNet_t','ncclPeerInfo_t','Connection handles (QP,FD)'], businessRules: ['Selection priority: P2P > SHM > IB > Socket','Each transport must support connect/send/recv/cleanup','IB requires MR registration for RDMA'], crossDomainInteractions: ['Used by Collective Operations via channels','Topology determines viable transports'] } },
  { id: 'domain:topology-routing', type: 'domain', name: 'Topology & Routing',
    summary: 'GPU interconnect topology discovery and optimal routing computation. Maps NVLink, PCIe hierarchy, CPU interconnects to find bandwidth-maximized ring/tree paths.',
    tags: ['graph-algorithms','topology','performance-tuning'], complexity: 'complex',
    domainMeta: { entities: ['ncclTopo_t','ncclTopoNode_t','Ring path','Tree structure'], businessRules: ['Topology discovered before any collective','Ring paths visit every GPU exactly once per channel','Algorithm selection depends on topology and message size'], crossDomainInteractions: ['Feeds routing data to Collective Operations','Informs Transport Backends about viable connections'] } },
  { id: 'domain:initialization-lifecycle', type: 'domain', name: 'Initialization & Lifecycle',
    summary: 'Bootstrap and lifecycle management: process discovery, CUDA context setup, channel creation, communicator init/cleanup.',
    tags: ['bootstrap','lifecycle','cuda-context'], complexity: 'moderate',
    domainMeta: { entities: ['ncclComm_t','ncclGroup','Proxy thread','Bootstrap socket'], businessRules: ['All ranks must call ncclCommInitRank with matching rank/count','GroupStart/End delimits collective boundaries','Proxy thread handles async communication'], crossDomainInteractions: ['Triggers Topology & Routing discovery','Sets up Transport Backends','Creates channels for Collective Operations'] } },
  { id: 'domain:plugin-ecosystem', type: 'domain', name: 'Plugin Ecosystem',
    summary: 'Extensibility framework for custom network backends, profilers (inspector), tuners, GIN support, and RMA via pluggable modules.',
    tags: ['extensibility','plugin-api','profiling'], complexity: 'moderate',
    domainMeta: { entities: ['ncclNetPlugin_t','ncclTunerPlugin_t','ncclProfilerPlugin_t','ncclGinPlugin_t'], businessRules: ['Plugins loaded via LD_PRELOAD or dlopen at init','Each plugin type has versioned API contract','Custom net plugins replace built-in transports'], crossDomainInteractions: ['Net plugins extend Transport Backends','Profiler plugins instrument Collective Operations'] } },

  // ============================================================
  // FLOWS (12)
  // ============================================================
  { id: 'flow:communicator-initialization', type: 'flow', name: 'Communicator Initialization',
    summary: 'Full ncclCommInitRank flow: bootstrap → device detection → topology discovery → transport selection → channel creation → ready.',
    tags: ['lifecycle','bootstrap'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclCommInitRank(rootComm, nRanks, rank)', entryType: 'api' } },
  { id: 'flow:group-operation-lifecycle', type: 'flow', name: 'Group Operation Lifecycle',
    summary: 'ncclGroupStart → [collectives] → ncclGroupEnd sequence that batches multiple collectives for concurrent execution.',
    tags: ['lifecycle','batching'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclGroupStart() / ncclGroupEnd()', entryType: 'api' } },
  { id: 'flow:all-reduce-execution', type: 'flow', name: 'All-Reduce Execution',
    summary: 'Enqueue → algorithm selection → GPU kernel launch → proxy thread coordination → completion notification.',
    tags: ['collective','allreduce'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm)', entryType: 'api' } },
  { id: 'flow:all-gather-execution', type: 'flow', name: 'All-Gather Execution',
    summary: 'Gather data segments from all GPUs to every GPU using ring or tree algorithm.',
    tags: ['collective','allgather'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclAllGather(sendbuff, recvbuff, count, datatype, comm)', entryType: 'api' } },
  { id: 'flow:reduce-scatter-execution', type: 'flow', name: 'Reduce-Scatter Execution',
    summary: 'Reduce data across all ranks then scatter result segments. Critical for large-model training.',
    tags: ['collective','reducescatter'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm)', entryType: 'api' } },
  { id: 'flow:topology-discovery', type: 'flow', name: 'GPU Topology Discovery',
    summary: 'NVML-based GPU discovery → PCIe hierarchy mapping → NVLink link detection → topology graph construction.',
    tags: ['topology','nvml'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclTopoInit(comm)', entryType: 'internal' } },
  { id: 'flow:routing-computation', type: 'flow', name: 'Ring/Tree Routing Computation',
    summary: 'Graph search for optimal ring paths → tree construction → algorithm selection based on bandwidth estimation.',
    tags: ['routing','graph-algorithm'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclTopoSetChannels(comm)', entryType: 'internal' } },
  { id: 'flow:transport-connection-setup', type: 'flow', name: 'Transport Connection Setup',
    summary: 'Exchange peer info → select transport (P2P/SHM/IB/Sock) → establish connection → register memory regions.',
    tags: ['transport','connection'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclTransportConnect(comm, channels)', entryType: 'internal' } },
  { id: 'flow:plugin-loading', type: 'flow', name: 'Plugin Loading & Registration',
    summary: 'LD_PRELOAD/dlopen plugin discovery → version negotiation → API registration → hook into NCCL lifecycle.',
    tags: ['plugin','ld-preload'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclNetPluginLoad() / ncclTunerPluginLoad()', entryType: 'internal' } },
  { id: 'flow:kernel-launch-pipeline', type: 'flow', name: 'Kernel Launch Pipeline',
    summary: 'Algorithm selection → kernel parameter preparation → CUDA kernel launch → LSA barrier sync → device-side execution.',
    tags: ['cuda','kernel-launch'], complexity: 'complex',
    domainMeta: { entryPoint: 'ncclLaunch(kernelArgs)', entryType: 'internal' } },
  { id: 'flow:proxy-thread-loop', type: 'flow', name: 'Proxy Thread Loop',
    summary: 'Background proxy thread polls all active transports for completions → handles IB/SHM data transfer → reports to scheduler.',
    tags: ['proxy','async'], complexity: 'moderate',
    domainMeta: { entryPoint: 'ncclProxyThread(comm)', entryType: 'internal' } },
  { id: 'flow:python-binding-call', type: 'flow', name: 'Python Binding Call Path',
    summary: 'Python API call → Cython binding → nccl4py communicator wrapper → C NCCL API → native execution.',
    tags: ['python','cython','nccl4py'], complexity: 'simple',
    domainMeta: { entryPoint: 'comm.allreduce(tensor) (Python)', entryType: 'api' } },

  // ============================================================
  // STEPS
  // ============================================================
  // Communicator Initialization
  { id: 'step:communicator-initialization:bootstrap', type: 'step', name: 'Bootstrap Between Ranks', summary: 'Establish initial TCP socket connection between all ranks via bootstrap layer.', tags: ['bootstrap'], complexity: 'moderate', filePath: 'src/bootstrap.cc' },
  { id: 'step:communicator-initialization:device-detection', type: 'step', name: 'Device Detection', summary: 'Query CUDA devices on this rank, get GPU count and capabilities.', tags: ['cuda'], complexity: 'simple', filePath: 'src/init.cc' },
  { id: 'step:communicator-initialization:topo-discovery', type: 'step', name: 'Topology Discovery', summary: 'Use NVML to discover GPU interconnect topology: NVLink links, PCIe switches, CPU sockets.', tags: ['nvml'], complexity: 'complex', filePath: 'src/graph/topo.cc' },
  { id: 'step:communicator-initialization:routing-calculation', type: 'step', name: 'Ring/Tree Path Calculation', summary: 'Compute optimal ring and tree routing paths for all channels based on topology.', tags: ['routing'], complexity: 'complex', filePath: 'src/graph/connect.cc' },
  { id: 'step:communicator-initialization:channel-setup', type: 'step', name: 'Channel Creation & Transport Setup', summary: 'Create N channels, select transport for each, establish connections.', tags: ['channel'], complexity: 'moderate', filePath: 'src/channel.cc' },
  { id: 'step:communicator-initialization:proxy-launch', type: 'step', name: 'Proxy Thread Launch', summary: 'Spawn proxy thread for ongoing transport polling and data movement.', tags: ['proxy'], complexity: 'simple', filePath: 'src/proxy.cc' },

  // All-Reduce Execution
  { id: 'step:all-reduce-execution:enqueue', type: 'step', name: 'Enqueue Collective Work', summary: 'Validate arguments, allocate work structure, select algorithm (ring/tree).', tags: ['scheduler'], complexity: 'moderate', filePath: 'src/enqueue.cc' },
  { id: 'step:all-reduce-execution:param-calc', type: 'step', name: 'Compute Kernel Parameters', summary: 'Calculate chunk sizes, thread counts, and memory layout for selected algorithm.', tags: ['scheduler'], complexity: 'moderate', filePath: 'src/enqueue.cc' },
  { id: 'step:all-reduce-execution:kernel-launch', type: 'step', name: 'Launch GPU Kernel', summary: 'Compile kernel arguments, launch CUDA kernel on each active channel.', tags: ['cuda'], complexity: 'complex', filePath: 'src/device/all_reduce.h' },
  { id: 'step:all-reduce-execution:proxy-sync', type: 'step', name: 'Proxy Thread Synchronization', summary: 'Proxy thread polls transports for IB/SHM completions, handles async transfer.', tags: ['proxy'], complexity: 'moderate', filePath: 'src/proxy.cc' },
  { id: 'step:all-reduce-execution:completion', type: 'step', name: 'Completion Notification', summary: 'All channels report completion → CUDA stream synchronized → result returned.', tags: ['lifecycle'], complexity: 'simple', filePath: 'src/collectives.cc' },

  // Topology Discovery
  { id: 'step:topology-discovery:nvml-query', type: 'step', name: 'NVML GPU Query', summary: 'Query NVML for GPU PCI bus IDs, NVLink status, and link speeds.', tags: ['nvml'], complexity: 'moderate', filePath: 'src/graph/topo.cc' },
  { id: 'step:topology-discovery:pcie-map', type: 'step', name: 'PCIe Hierarchy Mapping', summary: 'Walk sysfs to build PCIe switch topology and map GPU to CPU socket via NUMA.', tags: ['sysfs'], complexity: 'complex', filePath: 'src/graph/topo.cc' },
  { id: 'step:topology-discovery:graph-build', type: 'step', name: 'Topology Graph Construction', summary: 'Build ncclTopo_t graph from NVML + sysfs data with bandwidth annotations.', tags: ['graph'], complexity: 'complex', filePath: 'src/graph/topo.cc' },

  // Routing Computation
  { id: 'step:routing-computation:search', type: 'step', name: 'Ring Path Search', summary: 'Find optimal ring paths through topology graph using bandwidth-maximized search.', tags: ['graph-algorithm'], complexity: 'complex', filePath: 'src/graph/search.cc' },
  { id: 'step:routing-computation:tree-build', type: 'step', name: 'Tree Construction', summary: 'Build tree structures for broadcast/reduce operations across GPU topology.', tags: ['graph-algorithm'], complexity: 'moderate', filePath: 'src/graph/connect.cc' },
  { id: 'step:routing-computation:algo-select', type: 'step', name: 'Algorithm Selection', summary: 'Choose ring vs tree based on data size, topology bandwidth, and GPU count.', tags: ['tuning'], complexity: 'moderate', filePath: 'src/graph/tuning.cc' },

  // Transport Connection Setup
  { id: 'step:transport-connection-setup:peer-exchange', type: 'step', name: 'Exchange Peer Info', summary: 'Share GPU bus IDs, CUDA devices, and buffer addresses between all ranks.', tags: ['bootstrap'], complexity: 'simple', filePath: 'src/transport.cc' },
  { id: 'step:transport-connection-setup:transport-select', type: 'step', name: 'Transport Selection', summary: 'Select fastest available transport per GPU pair: P2P → SHM → IB → Socket.', tags: ['transport'], complexity: 'moderate', filePath: 'src/transport.cc' },
  { id: 'step:transport-connection-setup:connect', type: 'step', name: 'Establish Connections', summary: 'Create transport-specific connections: QP for IB, FD for socket, CUDA IPC for P2P.', tags: ['transport'], complexity: 'moderate', filePath: 'src/transport/net_ib/connect.cc' },

  // Plugin Loading
  { id: 'step:plugin-loading:dlopen', type: 'step', name: 'Plugin Discovery & dlopen', summary: 'Search LD_PRELOAD, NCCL_NET_PLUGIN env vars, and standard paths for plugin .so files.', tags: ['dlopen'], complexity: 'simple', filePath: 'src/misc/param.cc' },
  { id: 'step:plugin-loading:version-check', type: 'step', name: 'Version Negotiation', summary: 'Verify plugin API version matches NCCL expectations. Reject incompatible plugins.', tags: ['api-version'], complexity: 'simple', filePath: 'src/plugin/net/net_plugin.cc' },
  { id: 'step:plugin-loading:register', type: 'step', name: 'API Registration', summary: 'Register plugin entry points into NCCL internal tables, replacing or augmenting built-in transports.', tags: ['registration'], complexity: 'simple', filePath: 'src/plugin/net/net_plugin.cc' },

  // Kernel Launch Pipeline
  { id: 'step:kernel-launch-pipeline:algo-select', type: 'step', name: 'Algorithm Selection', summary: 'Select ring/tree algorithm and primitive (LL/LL128/Simple) based on data size and dtype.', tags: ['tuning'], complexity: 'moderate', filePath: 'src/enqueue.cc' },
  { id: 'step:kernel-launch-pipeline:param-prep', type: 'step', name: 'Kernel Parameter Preparation', summary: 'Build kernel argument structure with buffer pointers, chunk sizes, and topology info.', tags: ['cuda'], complexity: 'moderate', filePath: 'src/enqueue.cc' },
  { id: 'step:kernel-launch-pipeline:cuda-launch', type: 'step', name: 'CUDA Kernel Launch', summary: 'Launch collective kernel