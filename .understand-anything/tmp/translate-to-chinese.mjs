#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'fs';

const g = JSON.parse(readFileSync('.understand-anything/knowledge-graph.json', 'utf8'));

// ============================================================
// 1. Translate Project Metadata
// ============================================================
g.project.description = '优化的 GPU 间通信原语库。NCCL 实现了 all-reduce、all-gather、reduce-scatter、broadcast、reduce 以及 send/recv 通信模式，针对 PCIe、NVLink、NVswitch、InfiniBand 和 TCP/IP 进行优化。';

// ============================================================
// 2. Translate Tour Steps
// ============================================================
g.tour = [
  { order: 1, title: '项目概览', description: '从 README 开始了解 NCCL 的核心用途：支持 across PCIe、NVLink、NVswitch、InfiniBand 和 TCP/IP 的 all-reduce、all-gather、broadcast、reduce-scatter 等优化通信原语。', nodeIds: g.tour[0]?.nodeIds || [] },
  { order: 2, title: '公开 API — nccl.h', description: '理解核心公开接口。nccl.h 定义了所有 collective 操作（ncclAllReduce、ncclBroadcast 等）、communicator 类型（ncclComm_t）和用户交互的错误码。', nodeIds: g.tour[1]?.nodeIds || [] },
  { order: 3, title: '初始化流程', description: '追踪 init.cc 了解 NCCL 的引导过程：设备检测 → 拓扑发现 → 传输选择 → 通道创建。这是每个 NCCL 应用的入口点。', nodeIds: g.tour[2]?.nodeIds || [] },
  { order: 4, title: '拓扑与路由 — 拓扑发现', description: 'graph 模块分析 GPU 拓扑（NVLink、PCIe 层级）并计算最优 ring/tree 路由路径。这对性能至关重要——它决定了数据如何在 GPU 之间流动。', nodeIds: g.tour[3]?.nodeIds || [] },
  { order: 5, title: '传输层架构', description: 'NCCL 的传输抽象支持多种后端：NVLink (p2p)、Shared Memory (shm)、InfiniBand (net_ib) 和 TCP/IP。每个传输实现相同的接口，但针对其硬件进行了优化。', nodeIds: g.tour[4]?.nodeIds || [] },
  { order: 6, title: '设备内核 — Collective 算法', description: 'all-reduce、all-gather、reduce-scatter 的 GPU 端实现，通过 ring 和 tree 算法完成。这些内核在 GPU 上运行，处理实际的数据移动和归约计算。', nodeIds: g.tour[5]?.nodeIds || [] },
  { order: 7, title: '调度器 — 执行流水线', description: '调度器跨多个通道和传输层编排 collective 操作，管理工作队列并确保高效的 GPU 利用率。', nodeIds: g.tour[6]?.nodeIds || [] },
  { order: 8, title: '插件系统 — 可扩展性', description: 'NCCL 的插件架构支持自定义网络后端（net）、性能调优器（tuner）、调试分析器（profiler）和 GIN 支持。了解如何扩展 NCCL。', nodeIds: [] },
  { order: 9, title: '社区扩展', description: '探索社区贡献的实验性功能：nccl_ep（增强点对点通信）、nccl_m2n（many-to-many 通信）、nccl_ubx 和 checkpoint 支持。', nodeIds: g.tour[8]?.nodeIds || [] },
  { order: 10, title: '构建系统', description: '了解 CMake + Makefile 构建系统，它为不同的 GPU 架构编译 NCCL 并生成分发包（deb、rpm、tarball）。', nodeIds: [] }
];

// ============================================================
// 3. Translate Node Summaries (File-level nodes only)
// ============================================================

// Helper: translate common summary patterns
function translateSummary(summary, filePath, language) {
  if (!summary) return '';
  
  // Pattern: "Source file with X function(s): func1, func2, ..."
  const match = summary.match(/^Source file with (\d+) function\(s\):\s*(.+)$/);
  if (match) {
    const count = match[1];
    const funcs = match[2];
    return `源文件，包含 ${count} 个函数：${funcs}`;
  }
  
  // Pattern: "Source file: filename"
  const fileMatch = summary.match(/^Source file:\s*(.+)$/);
  if (fileMatch) {
    return `源文件：${fileMatch[1]}`;
  }
  
  // Already Chinese or special cases
  if (/^[\u4e00-\u9fff]/.test(summary)) return summary;
  
  return summary;
}

// Translate file-level nodes (most visible in dashboard)
const fileTypes = new Set(['file','config','document','service','pipeline','table','schema','resource','endpoint']);

let translatedCount = 0;
g.nodes.forEach(node => {
  if (fileTypes.has(node.type) && node.summary) {
    const original = node.summary;
    const translated = translateSummary(original, node.filePath, node.language);
    if (translated !== original) {
      node.summary = translated;
      translatedCount++;
    }
  }
  // Also translate function nodes summaries
  if (node.type === 'function' && node.summary) {
    const match = node.summary.match(/^Function in (.+): (\w+)$/);
    if (match) {
      node.summary = `函数：${match[2]}（位于 ${match[1]}）`;
      translatedCount++;
    }
  }
});

console.log(`Translated ${translatedCount} nodes`);

// ============================================================
// 4. Translate Layer Descriptions (already Chinese, but verify)
// ============================================================
g.layers.forEach(layer => {
  if (!/^[\u4e00-\u9fff]/.test(layer.description || '')) {
    // Fallback descriptions for any missing Chinese
    const nameMap = {
      'Core Runtime': '核心运行时',
      'Device Layer (GPU Kernel)': '设备层（GPU 内核）', 
      'Transport Layer': '传输层',
      'Graph & Routing': '拓扑与路由层',
      'Scheduler': '调度层',
      'Plugin System': '插件系统',
      'Headers & API': '头文件与公开 API',
      'Contrib Extensions': '社区扩展',
      'Language Bindings': '语言绑定层',
      'Build & Packaging': '构建与打包',
      'Documentation': '文档层',
      'OS Utilities & RAS': '系统工具与可靠性',
      'Root Configuration': '根目录配置'
    };
    if (nameMap[layer.name]) {
      layer.name = nameMap[layer.name];
    }
  }
});

// ============================================================
// Save
// ============================================================
writeFileSync('.understand-anything/knowledge-graph.json', JSON.stringify(g, null, 2));
console.log('Saved translated knowledge-graph.json');

// Stats
const fileNodes = g.nodes.filter(n => fileTypes.has(n.type)).length;
const funcNodes = g.nodes.filter(n => n.type === 'function').length;
console.log(`Total: ${g.nodes.length} nodes, ${fileNodes} file-level, ${funcNodes} function`);
