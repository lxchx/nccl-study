#!/usr/bin/env python3
"""
Export NCCL callgraph trace to an indexed binary format for range-request loading.

File layout:
  [Header 128 bytes]
    magic (4): "NCGT"
    version (4): uint32
    total_nodes (4): uint32
    root_count (4): uint32
    record_size (4): uint32  (= 64)
    nodes_offset (8): uint64  (= 128)
    args_offset (8): uint64
    args_size (8): uint64
    roots (8 * root_count): uint32 each, padded — actually roots go in header
  [Nodes array: total_nodes * 64 bytes, BFS-ordered]
    Each node (64 bytes):
      uint32 call_id          (original call_id)
      uint32 parent_index     (index in nodes array, 0xFFFFFFFF if root)
      uint8  func_id
      uint8  depth
      uint16 child_count
      uint32 first_child_index (0xFFFFFFFF if no visible children)
      uint32 first_collapsed_child_index (0xFFFFFFFF if none; points to a
             "collapsed" sentinel we don't store — instead we store
             collapsed_descendants count and collapsed_child_count inline)
      uint16 collapsed_child_count
      uint16 collapsed_descendants_count_lo  (we store as uint32 below instead)
      int64  duration
      uint32 enter_args_offset (0xFFFFFFFF if none; relative to args section)
      uint8  enter_args_count
      uint32 exit_args_offset  (0xFFFFFFFF if none)
      uint8  exit_args_count
      uint8  padding[3]
  (Note: layout above is > 64 bytes; let me redo to exactly 64.)
"""
import struct, sys, argparse
from collections import defaultdict, deque

HDR_FMT = "<BBBBIIQQ"
HDR_SIZE = 28

FUNC_NAMES = {
    1:"ncclTopoCompute",2:"ncclTopoSearchInit",3:"ncclTopoSearchRec",
    4:"ncclTopoSearchTryGpu",5:"ncclTopoSearchRecGpu",6:"ncclTopoFollowPath",
    7:"ncclTopoCompareGraphs",8:"ncclTopoSearchNextGpuSort",9:"ncclTopoReplayGetGpu",
}
ARG_NAMES = {
    (1,1):["id","pattern","minChannels","maxChannels"],
    (1,0):["nChannels","bwIntra_x1000","bwInter_x1000","nHops","typeIntra","typeInter"],
    (2,1):["ngpus","inter","_"],
    (2,0):["maxBw_x1000","totalBw_x1000"],
    (3,1):["graph_nChannels","saveGraph_nChannels","time","pattern","sameChannels"],
    (3,0):["saveGraph_nChannels_out","time_out"],
    (4,1):["step","forcedOrder","type","index","g","time","graph_nChannels"],
    (4,0):["gpu_nonnull","time_out"],
    (5,1):["gpu_rank","step","backToNet","backToFirstRank","forcedOrder","time","graph_nChannels"],
    (5,0):["time_out","saveGraph_nChannels"],
    (6,1):["type1","index1","type2","index2","mult_x1000","pattern"],
    (6,0):["node_nonnull"],
    (7,1):["graph_nChannels","graph_bwIntra_x1000","ref_nChannels","ref_bwIntra_x1000","graph_nHops","ref_nHops"],
    (7,0):["copy"],
    (8,1):["gpu_rank","graph_nChannels","sortNet"],
    (8,0):["count","next0","next1","next2","next3"],
    (9,1):["graph_nChannels","step"],
    (9,0):["g"],
}
PATTERN_NAMES={1:"BALANCED_TREE",2:"SPLIT_TREE",3:"TREE",4:"RING",5:"NVLS",6:"COLLNET_DIRECT"}
FORCED_ORDER_NAMES={0:"normal",1:"FORCED_ORDER_PCI",2:"FORCED_ORDER_REPLAY"}


def fmt_arg(name, val):
    """Return a string representation for an arg, for the JS side to display."""
    if name == "pattern" and val in PATTERN_NAMES: return f"{PATTERN_NAMES[val]}({val})"
    if name == "forcedOrder" and val in FORCED_ORDER_NAMES: return f"{FORCED_ORDER_NAMES[val]}({val})"
    if name.endswith("_x1000"): return f"{val/1000.0:.1f}"
    if name == "mult_x1000": return f"{val/1000.0:.2f}"
    return str(val)


def parse(path):
    recs=[]
    with open(path,"rb") as f: data=f.read()
    off=0;n=len(data)
    import json as _json
    while off+HDR_SIZE<=n:
        fid,dr,rk,na,cid,pid,tsc,nrec=struct.unpack_from(HDR_FMT,data,off)
        if fid==0 and nrec==0: break
        is_json = (na == 0xFF)
        if is_json:
            jlen = nrec - HDR_SIZE
            if nrec < HDR_SIZE or nrec > HDR_SIZE + 8192: break
            jstr = data[off+HDR_SIZE:off+HDR_SIZE+jlen].decode('utf-8',errors='replace')
            try:
                args = _json.loads(jstr) if jlen>0 else {}
                if not isinstance(args,dict): args={"_value":args}
            except Exception:
                args={"_raw":jstr}
        else:
            if nrec<HDR_SIZE or nrec>HDR_SIZE+64 or na>8: break
            args=[struct.unpack_from("<q",data,off+HDR_SIZE+i*8)[0] for i in range(na)]
        recs.append({"fid":fid,"dir":dr,"cid":cid,"pid":pid,"tsc":tsc,"args":args})
        off+=nrec
    return recs


# === Binary layout constants ===
MAGIC = b"NCGT"
VERSION = 1
HEADER_SIZE = 128
NODE_SIZE = 64
ARGS_MAGIC = b"ARGS"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default="callgraph.bin")
    args=ap.parse_args()

    recs = parse(args.trace)
    enters = {r["cid"]: r for r in recs if r["dir"]==1}
    exits = {r["cid"]: r for r in recs if r["dir"]==0}

    children = defaultdict(list)
    roots = []
    for cid, r in enters.items():
        if r["pid"]==0 or r["pid"] not in enters: roots.append(cid)
        else: children[r["pid"]].append(cid)

    # Compute depth via BFS, and assign BFS index to each node
    depth = {}
    parent_index = {}   # call_id -> BFS index of parent
    bfs_order = []      # list of call_id in BFS order
    index_of = {}       # call_id -> BFS index
    q = deque()
    for r in roots:
        depth[r] = 1
        index_of[r] = len(bfs_order)
        bfs_order.append(r)
        parent_index[r] = 0xFFFFFFFF
        q.append(r)
    while q:
        cid = q.popleft()
        for c in children.get(cid, []):
            depth[c] = depth[cid] + 1
            index_of[c] = len(bfs_order)
            parent_index[c] = index_of[cid]
            bfs_order.append(c)
            q.append(c)

    total_nodes = len(bfs_order)
    print(f"Total nodes: {total_nodes}, roots: {len(roots)}")

    # Compute subtree sizes (descendants) for collapsed display
    # Process in reverse BFS order
    subtree_size = {}
    for cid in reversed(bfs_order):
        s = 1
        for c in children.get(cid, []):
            s += subtree_size.get(c, 1)
        subtree_size[cid] = s

    # Build args section. New JSON trace: args are already a dict from nlohmann/json
    # (preserves float types, nested graph objects, etc.). We store the JSON string
    # directly. For legacy int64 records, args is a list — we wrap as {"_args":[...]}.
    args_blob = bytearray()
    args_offsets = {}  # call_id -> (enter_offset, enter_len, exit_offset, exit_len)
    import json
    for cid in bfs_order:
        e = enters[cid]
        ex = exits.get(cid)
        enter_args = e["args"]
        enter_obj = enter_args if isinstance(enter_args, dict) else {"_args": enter_args}
        exit_obj = None
        if ex:
            exit_args = ex["args"]
            exit_obj = exit_args if isinstance(exit_args, dict) else {"_args": exit_args}
        enter_json = json.dumps(enter_obj, separators=(",",":"), ensure_ascii=False).encode("utf-8")
        exit_json = json.dumps(exit_obj, separators=(",",":"), ensure_ascii=False).encode("utf-8") if exit_obj else b""
        enter_off = len(args_blob)
        args_blob += enter_json
        enter_len = len(enter_json)
        exit_off = len(args_blob)
        args_blob += exit_json
        exit_len = len(exit_json)
        args_offsets[cid] = (enter_off, enter_len, exit_off, exit_len)

    # Build nodes array
    nodes_data = bytearray()
    for i, cid in enumerate(bfs_order):
        e = enters[cid]
        ex = exits.get(cid)
        my_children = children.get(cid, [])
        # children indices in BFS order
        child_indices = [index_of[c] for c in my_children]
        first_child = child_indices[0] if child_indices else 0xFFFFFFFF
        dur = (ex["tsc"] - e["tsc"]) if ex else -1
        enter_off, enter_len, exit_off, exit_len = args_offsets[cid]
        # Pack node (64 bytes):
        # uint32 call_id
        # uint32 parent_index
        # uint8  func_id
        # uint8  depth (cap at 255)
        # uint16 child_count
        # uint32 first_child_index
        # int64  duration
        # uint32 enter_args_offset
        # uint16 enter_args_len
        # uint32 exit_args_offset
        # uint16 exit_args_len
        # uint32 subtree_size  (descendants count, for collapsed display)
        # padding to 64
        d = min(depth[cid], 255)
        node = struct.pack("<IIBBHIqIHIHI",
            cid,                         # I
            parent_index[cid],           # I
            e["fid"],                    # B
            d,                            # B
            len(child_indices),          # H
            first_child,                 # I
            dur,                         # q
            enter_off,                   # I
            enter_len,                   # H
            exit_off,                    # I
            exit_len,                    # H
            subtree_size[cid])           # I
        # Pad to 64 bytes
        node += b"\x00" * (NODE_SIZE - len(node))
        assert len(node) == NODE_SIZE
        nodes_data += node

    args_offset = HEADER_SIZE + len(nodes_data)
    args_size = len(args_blob)

    # Build header (128 bytes)
    # magic(4) version(4) total_nodes(4) root_count(4) record_size(4)
    # nodes_offset(8) args_offset(8) args_size(8)
    # roots: up to (128-48)/4 = 20 roots
    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<I", total_nodes)
    header += struct.pack("<I", len(roots))
    header += struct.pack("<I", NODE_SIZE)
    header += struct.pack("<Q", HEADER_SIZE)   # nodes_offset
    header += struct.pack("<Q", args_offset)
    header += struct.pack("<Q", args_size)
    # roots (each as uint32 BFS index)
    for r in roots:
        header += struct.pack("<I", index_of[r])
    # pad to 128
    header += b"\x00" * (HEADER_SIZE - len(header))

    # Write file
    with open(args.out, "wb") as f:
        f.write(header)
        f.write(nodes_data)
        f.write(args_blob)
    import os
    print(f"Wrote {args.out}: header={HEADER_SIZE} nodes={len(nodes_data)} args={args_size} total={os.path.getsize(args.out)} bytes ({os.path.getsize(args.out)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
