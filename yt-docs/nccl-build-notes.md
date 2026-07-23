# NCCL 构建笔记

## 构建方式（官方 Makefile）

NCCL 官方使用 Makefile 构建（**不是 CMake**）。在 5090 上：

```bash
cd /repo/nccl
make -j$(nproc) src.build CUDA_HOME=/usr/local/cuda
```

产物在 `build/` 下：
- `build/lib/libnccl.so.2.30.7`
- `build/lib/libnccl_static.a`
- `build/bin/ncclparam`

## Makefile 默认选项

来自 `makefiles/common.mk`：

```makefile
RDMA_CORE ?= 0    # 不包含 InfiniBand
MLX5DV    ?= 0    # 不包含 MLX5 Direct Verbs
DEBUG     ?= 0    # Release 模式
PROFAPI   ?= 1    # Profiling API
NVTX      ?= 1    # NVTX 注解
MAX_EXT_NET_PLUGINS ?= 0
```

系统 NCCL（pip 包 `nvidia-nccl-cu12`）也是用同样方式构建的。

## 同步方式

本地源码位置：`/Users/lichuan/storage_yuntun/nccl/`
5090 源码位置：`/repo/nccl/`（实际路径 `/mnt/storage00/repo/nccl/`）

同步命令：
```bash
scp <本地文件> jscs-backuppool-5090test-091606:/repo/nccl/<相对路径>
```

### 常用的同步命令
```bash
# xml.cc
scp /Users/lichuan/storage_yuntun/nccl/src/graph/xml.cc \
    jscs-backuppool-5090test-091606:/repo/nccl/src/graph/xml.cc
# topo.cc
scp /Users/lichuan/storage_yuntun/nccl/src/graph/topo.cc \
    jscs-backuppool-5090test-091606:/repo/nccl/src/graph/topo.cc
# debug.h
scp /Users/lichuan/storage_yuntun/nccl/src/include/debug.h \
    jscs-backuppool-5090test-091606:/repo/nccl/src/include/debug.h
```

## 运行测试

### ⚠️ 必须使用动态链接（不要用静态库）

NCCL 2.30.7 在 5090 上**静态链接**（`libnccl_static.a`）会导致 NULL 函数指针崩溃：
- `dmesg` 显示 `segfault at 0 ip 0000000000000000`
- 原因是 CUDA 驱动 PFN（`cuGetProcAddress`）在静态链接时解析失败
- 动态链接 `.so` 完全正常

```bash
# 编译（动态链接）
/usr/local/cuda/bin/nvcc -o /tmp/probe /tmp/probe.cu \
    -I/repo/nccl/build/include \
    -L/repo/nccl/build/lib -lnccl -lnvidia-ml -lcudart -lpthread -ldl

# 运行（指定 LD_LIBRARY_PATH 指向构建的 .so）
LD_LIBRARY_PATH=/repo/nccl/build/lib \
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL /tmp/probe 2>/tmp/probe.log

# 查看 YT-TRACE 输出
grep 'YT-TRACE' /tmp/probe.log
```

> 也可以把 `LD_LIBRARY_PATH` 写入 `~/.bashrc` 或直接替换系统 NCCL 的 .so。
