#!/usr/bin/env python3
"""
Export NCCL callgraph trace to a JSON file for the interactive HTML viewer.

Output JSON structure:
{
  "roots": [call_id, ...],
  "nodes": {
    "<call_id>": {
      "func_id": int,
      "func_name": str,
      "depth": int,
      "enter_args": {name: value, ...},   # only if depth <= max_depth
      "exit_args": {name: value, ...},    # only if depth <= max_depth
      "duration": int,                    # tsc ticks, -1 if no exit
      "children": [call_id, ...],         # direct children (call_ids)
      "child_count": int,                 # total descendants (for display)
      "collapsed": bool                   # true if depth > max_depth (no args)
    },
    ...
  },
  "func_names": {...},
  "arg_names": {...},
  "meta": {"total_records": N, "max_depth": N, ...}
}
"""
import struct, sys, json, argparse
from collections import defaultdict, deque

HDR_FMT = "<BBBBIIQQ"
HDR_SIZE = 28

FUNC_NAMES = {
    1: "ncclTopoCompute", 2: "ncclTopoSearchInit", 3: "ncclTopoSearchRec",
    4: "ncclTopoSearchTryGpu", 5: "ncclTopoSearchRecGpu", 6: "ncclTopoFollowPath",
    7: "ncclTopoCompareGraphs", 8: "ncclTopoSearchNextGpuSort", 9: "ncclTopoReplayGetGpu",
}
ARG_NAMES = {
    (1,1): ["id","pattern","minChannels","maxChannels"],
    (1,0): ["nChannels","bwIntra_x1000","bwInter_x1000","nHops","typeIntra","typeInter"],
    (2,1): ["ngpus","inter","_"],
    (2,0): ["maxBw_x1000","totalBw_x1000"],
    (3,1): ["graph_nChannels","saveGraph_nChannels","time","pattern","sameChannels"],
    (3,0): ["saveGraph_nChannels_out","time_out"],
    (4,1): ["step","forcedOrder","type","index","g","time","graph_nChannels"],
    (4,0): ["gpu_nonnull","time_out"],
    (5,1): ["gpu_rank","step","backToNet","backToFirstRank","forcedOrder","time","graph_nChannels"],
    (5,0): ["time_out","saveGraph_nChannels"],
    (6,1): ["type1","index1","type2","index2","mult_x1000","pattern"],
    (6,0): ["node_nonnull"],
    (7,1): ["graph_nChannels","graph_bwIntra_x1000","ref_nChannels","ref_bwIntra_x1000","graph_nHops","ref_nHops"],
    (7,0): ["copy"],
    (8,1): ["gpu_rank","graph_nChannels","sortNet"],
    (8,0): ["count","next0","next1","next2","next3"],
    (9,1): ["graph_nChannels","step"],
    (9,0): ["g"],
}
PATTERN_NAMES = {1:"BALANCED_TREE",2:"SPLIT_TREE",3:"TREE",4:"RING",5:"NVLS",6:"COLLNET_DIRECT"}
FORCED_ORDER_NAMES = {0:"normal",1:"FORCED_ORDER_PCI",2:"FORCED_ORDER_REPLAY"}


def fmt_arg(name, val):
    if name == "pattern" and val in PATTERN_NAMES:
        return f"{PATTERN_NAMES[val]}({val})"
    if name == "forcedOrder" and val in FORCED_ORDER_NAMES:
        return f"{FORCED_ORDER_NAMES[val]}({val})"
    if name.endswith("_x1000"):
        return round(val/1000.0, 1)
    if name == "mult_x1000":
        return round(val/1000.0, 2)
    return val


def parse(path):
    recs = []
    with open(path, "rb") as f:
        data = f.read()
    off = 0; n = len(data)
    while off + HDR_SIZE <= n:
        fid, dr, rk, na, cid, pid, tsc, nrec = struct.unpack_from(HDR_FMT, data, off)
        if fid == 0 and nrec == 0: break
        if nrec < HDR_SIZE or nrec > HDR_SIZE + 64 or na > 8: break
        args = [struct.unpack_from("<q", data, off + HDR_SIZE + i*8)[0] for i in range(na)]
        recs.append({"func_id":fid, "dir":dr, "rank":rk, "call_id":cid, "parent_call_id":pid, "tsc":tsc, "args":args})
        off += nrec
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default="callgraph.json")
    ap.add_argument("--max-depth", type=int, default=12, help="depth <= this: include args; deeper: collapsed")
    args = ap.parse_args()

    recs = parse(args.trace)
    enters = {r["call_id"]: r for r in recs if r["dir"] == 1}
    exits = {r["call_id"]: r for r in recs if r["dir"] == 0}

    # Build children + roots
    children = defaultdict(list)
    roots = []
    for cid, r in enters.items():
        pid = r["parent_call_id"]
        if pid == 0 or pid not in enters:
            roots.append(cid)
        else:
            children[pid].append(cid)

    # Compute depth for each node (BFS from roots)
    depth = {}
    q = deque()
    for r in roots:
        depth[r] = 1
        q.append(r)
    while q:
        n = q.popleft()
        for c in children.get(n, []):
            depth[c] = depth[n] + 1
            q.append(c)

    # Compute descendant count (subtree size) for each node
    subtree_size = {}
    # process in reverse BFS order (deepest first)
    order = sorted(depth.keys(), key=lambda x: -depth[x])
    for cid in order:
        s = 1
        for c in children.get(cid, []):
            s += subtree_size.get(c, 1)
        subtree_size[cid] = s

    # Build node dict — only include nodes with depth <= max_depth
    nodes = {}
    for cid, enter in enters.items():
        d = depth.get(cid, 0)
        if d > args.max_depth:
            continue
        exit_ = exits.get(cid)
        # children: only those also within max_depth
        all_children = children.get(cid, [])
        visible_children = [c for c in all_children if depth.get(c, 999) <= args.max_depth]
        collapsed_children = [c for c in all_children if depth.get(c, 999) > args.max_depth]
        node = {
            "func_id": enter["func_id"],
            "func_name": FUNC_NAMES.get(enter["func_id"], f"fn{enter['func_id']}"),
            "depth": d,
            "duration": (exit_["tsc"] - enter["tsc"]) if exit_ else -1,
            "children": visible_children,
            "collapsed_descendants": sum(subtree_size.get(c, 1) for c in collapsed_children),
            "collapsed_child_count": len(collapsed_children),
            "child_count": subtree_size.get(cid, 1),
        }
        # include args (always, since within max_depth)
        enter_names = ARG_NAMES.get((enter["func_id"], 1), [])
        exit_names = ARG_NAMES.get((enter["func_id"], 0), []) if exit_ else []
        node["enter_args"] = {enter_names[i] if i < len(enter_names) else f"arg{i}": fmt_arg(enter_names[i] if i < len(enter_names) else f"arg{i}", enter["args"][i]) for i in range(len(enter["args"]))}
        if exit_:
            node["exit_args"] = {exit_names[i] if i < len(exit_names) else f"arg{i}": fmt_arg(exit_names[i] if i < len(exit_names) else f"arg{i}", exit_["args"][i]) for i in range(len(exit_["args"]))}
        nodes[str(cid)] = node

    out = {
        "roots": roots,
        "nodes": nodes,
        "func_names": FUNC_NAMES,
        "meta": {
            "total_records": len(recs),
            "total_nodes": len(enters),
            "max_depth_reached": max(depth.values()) if depth else 0,
            "args_included_depth": args.max_depth,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Wrote {args.out}: {len(nodes)} nodes, {len(roots)} roots, max depth {out['meta']['max_depth_reached']}")
    # Size
    import os
    print(f"File size: {os.path.getsize(args.out)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
