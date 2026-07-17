# NCCL 文档参考资料

## 官方文档

- [NCCL User Guide (NVIDIA)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) - 官方权威文档

## 博客与教程

- [NCCL 相关文章](https://www.cnblogs.com/t-bar/p/19652703) - 内容详实，适合参考

---

## 在 5090 测试机上编译运行 NCCL 程序的步骤

### 环境信息
- **机器**: jscs-backuppool-5090test-091606
- **GPU**: 8 × RTX 5090 (32GB)
- **CUDA**: /usr/local/cuda/bin/nvcc (CUDA 12.9)
- **NCCL**: 位于 Python 虚拟环境中 (`/usr/local/lib/python3.9/dist-packages/nvidia/nccl/`)

### 关键路径
```bash
# NCCL 头文件
NCCL_INC="/usr/local/lib/python3.9/dist-packages/nvidia/nccl/include"

# NCCL 库文件
NCCL_LIB="/usr/local/lib/python3.9/dist-packages/nvidia/nccl/lib"

# nvcc 编译器
NVCC="/usr/local/cuda/bin/nvcc"
```

### 编译命令模板
```bash
# 1. 编写 C/CUDA 源文件 (例如: nccl_demo.cu)
# ⚠️ 注意：必须用 .cu 后缀，不要用 .c！
#    因为 nccl.h 包含 CUDA BF16/FP16 的 C++ 模板
#    用 .c 编译会触发 extern "C" 作用域下的名字歧义

cat > nccl_demo.cu << 'EOF'
#include <stdio.h>
#include "nccl.h"

int main() {
    ncclComm_t comms[8];
    ncclResult_t r = ncclCommInitAll(comms, 8, NULL);
    if (r == ncclSuccess) {
        for(int i=0; i<8; i++) ncclCommDestroy(comms[i]);
    }
    return 0;
}
EOF

# 2. 编译
/usr/local/cuda/bin/nvcc -o nccl_demo nccl_demo.cu \
    -I/usr/local/lib/python3.9/dist-packages/nvidia/nccl/include \
    -L/usr/local/lib/python3.9/dist-packages/nvidia/nccl/lib \
    -lnccl \
    -Wno-deprecated-gpu-targets 2>&1

# 3. 运行（带 NCCL Debug 日志）
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
LD_LIBRARY_PATH=/usr/local/lib/python3.9/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
    ./nccl_demo 2>&1 | tee nccl_log.txt
```

### 常见问题排查
| 问题 | 原因 | 解决方案 |
|---|---|---|
| `fatal error: nccl.h: No such file` | 头文件路径不对 | 使用 `-I/usr/local/lib/python3.9/dist-packages/nvidia/nccl/include` |
| `cannot find -lnccl` | 库路径不在 ld 搜索范围 | 使用 `-L/usr/local/lib/python3.9/dist-packages/nvidia/nccl/lib` |
| 大量 C++ 模板报错 (`extern "C" linkage`) | 源文件后缀是 `.c` | **改成 `.cu`**，让 nvcc 走 CUDA 编译流程 |
| 运行时找不到 libnccl.so | LD_LIBRARY_PATH 没设 | 运行前 `export LD_LIBRARY_PATH=.../nvidia/nccl/lib:$LD_LIBRARY_PATH` |

### 抓取 NCCL GRAPH 拓扑日志
```bash
# 开启完整的 INIT + GRAPH 调试输出
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# 运行程序，日志会输出到 stderr
LD_LIBRARY_PATH=/usr/local/lib/python3.9/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
    ./nccl_demo 2>&1 | tee nccl_topo_log.txt

# 关键日志关键字
# "GPU/0-27000 :GPU/0-38000 (2/48.0/PHB)" — GPU 间路径 (跳数/带宽/类型)
# "CPU/0-0 (1/1/4)" — CPU Socket 连接信息
# "+ PCI[48.0] - GPU/0-27000" — PCIe 链路详情
# "Ring 00 : 0 -> 1 -> 2" — NCCL 最终选定的 Ring 通道排列
