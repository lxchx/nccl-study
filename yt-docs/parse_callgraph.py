#!/usr/bin/env python3
"""
Parse NCCL callgraph trace binary files and produce a call graph.

Record layout (variable length, packed):
  uint8  func_id
  uint8  dir       (1=ENTER, 0=EXIT)
  uint8  rank
  uint8  nargs
  uint32 call_id
  uint32 parent_call_id
  uint64 tsc
  uint64 nrec       (total bytes of this record)
  int64  arg[nargs]
Header = 28 bytes. Payload = nargs * 8 bytes.

Usage:
  python3 parse_callgraph.py <trace_file> [--out graph.json] [--md out.md]
"""

import struct
import sys
import os
import json
from collections import defaultdict, Counter

HDR_FMT = "<BBBBIIQQ"  # func_id, dir, rank, nargs, call_id, parent_call_id, tsc, nrec
HDR_SIZE = struct.calcsize(HDR_FMT)
assert HDR_SIZE == 28, f"Header size {HDR_SIZE} != 28"

FUNC_NAMES = {
    1: "ncclTopoCompute",
    2: "ncclTopoSearchInit",
    3: "ncclTopoSearchRec",
    4: "ncclTopoSearchTryGpu",
    5: "ncclTopoSearchRecGpu",
    6: "ncclTopoFollowPath",
    7: "ncclTopoCompareGraphs",
    8: "ncclTopoSearchNextGpuSort",
    9: "ncclTopoReplayGetGpu",
}

# Semantic arg names per (func_id, dir). dir=1 enter, dir=0 exit.
ARG_NAMES = {
    # ncclTopoCompute
    (1, 1): ["id", "pattern", "minChannels", "maxChannels"],
    (1, 0): ["nChannels", "bwIntra_x1000", "bwInter_x1000", "nHops", "typeIntra", "typeInter"],
    # ncclTopoSearchInit
    (2, 1): ["ngpus", "inter", "_"],
    (2, 0): ["maxBw_x1000", "totalBw_x1000"],
    # ncclTopoSearchRec
    (3, 1): ["graph_nChannels", "saveGraph_nChannels", "time", "pattern", "sameChannels"],
    (3, 0): ["saveGraph_nChannels_out", "time_out"],
    # ncclTopoSearchTryGpu
    (4, 1): ["step", "forcedOrder", "type", "index", "g", "time", "graph_nChannels"],
    (4, 0): ["gpu_nonnull", "time_out"],
    # ncclTopoSearchRecGpu
    (5, 1): ["gpu_rank", "step", "backToNet", "backToFirstRank", "forcedOrder", "time", "graph_nChannels"],
    (5, 0): ["time_out", "saveGraph_nChannels"],
    # ncclTopoFollowPath
    (6, 1): ["type1", "index1", "type2", "index2", "mult_x1000", "pattern"],
    (6, 0): ["node_nonnull"],
    # ncclTopoCompareGraphs
    (7, 1): ["graph_nChannels", "graph_bwIntra_x1000", "ref_nChannels", "ref_bwIntra_x1000", "graph_nHops", "ref_nHops"],
    (7, 0): ["copy"],
    # ncclTopoSearchNextGpuSort
    (8, 1): ["gpu_rank", "graph_nChannels", "sortNet"],
    (8, 0): ["count", "next0", "next1", "next2", "next3"],
    # ncclTopoReplayGetGpu
    (9, 1): ["graph_nChannels", "step"],
    (9, 0): ["g"],
}

PATTERN_NAMES = {
    1: "BALANCED_TREE",
    2: "SPLIT_TREE",
    3: "TREE",
    4: "RING",
    5: "NVLS",
    6: "COLLNET_DIRECT",
}

FORCED_ORDER_NAMES = {
    0: "normal",
    1: "FORCED_ORDER_PCI",
    2: "FORCED_ORDER_REPLAY",
}


def parse_file(path):
    """Parse a trace file, return list of records (dicts)."""
    records = []
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    n = len(data)
    count = 0
    while off + HDR_SIZE <= n:
        # Read header
        func_id, dir_, rank, nargs, call_id, parent_call_id, tsc, nrec = struct.unpack_from(HDR_FMT, data, off)
        # Stop at unwritten (zeroed) region
        if func_id == 0 and dir_ == 0 and nargs == 0 and nrec == 0:
            break
        # Sanity check. For JSON records (nargs=0xFF), payload can be long;
        # for int64 records, payload = nargs*8 (max 64 bytes).
        is_json = (nargs == 0xFF)
        max_payload = nrec - HDR_SIZE if is_json else 64
        if nrec < HDR_SIZE or nrec > HDR_SIZE + max_payload + 4096:  # allow up to 4KB JSON
            # Corrupted or unwritten, stop
            break
        if (not is_json) and nargs > 8:
            break
        # Read args
        args = []
        if is_json:
            # JSON variant: payload is a UTF-8 JSON string of length (nrec - HDR_SIZE)
            json_len = nrec - HDR_SIZE
            json_str = data[off + HDR_SIZE : off + HDR_SIZE + json_len].decode('utf-8', errors='replace')
            try:
                args = json.loads(json_str) if json_len > 0 else {}
                # Wrap non-object (e.g. a bare array) so the viewer always sees a dict
                if not isinstance(args, dict):
                    args = {"_value": args}
            except Exception:
                args = {"_raw": json_str}
        else:
            for i in range(nargs):
                (a,) = struct.unpack_from("<q", data, off + HDR_SIZE + i * 8)
                args.append(a)
        rec = {
            "func_id": func_id,
            "func_name": FUNC_NAMES.get(func_id, f"fn{func_id}"),
            "dir": dir_,
            "rank": rank,
            "nargs": nargs,
            "call_id": call_id,
            "parent_call_id": parent_call_id,
            "tsc": tsc,
            "nrec": nrec,
            "args": args,
            "offset": off,
        }
        records.append(rec)
        count += 1
        off += nrec
    return records


def build_call_tree(records):
    """Build a call tree from enter/exit records keyed by call_id."""
    # Match enters and exits by call_id
    enters = {}  # call_id -> enter record
    exits = {}   # call_id -> exit record
    for r in records:
        if r["dir"] == 1:
            enters[r["call_id"]] = r
        else:
            exits[r["call_id"]] = r
    # Build children map by parent_call_id (using enter records)
    children = defaultdict(list)
    roots = []
    for cid, enter in enters.items():
        pid = enter["parent_call_id"]
        if pid == 0 or pid not in enters:
            roots.append(cid)
        else:
            children[pid].append(cid)
    return enters, exits, children, roots


def fmt_args(rec):
    """Format args semantically."""
    fid = rec["func_id"]
    d = rec["dir"]
    names = ARG_NAMES.get((fid, d), [])
    parts = []
    for i, a in enumerate(rec["args"]):
        name = names[i] if i < len(names) else f"arg{i}"
        # Decode special values
        if name in ("pattern",) and a in PATTERN_NAMES:
            parts.append(f"{name}={PATTERN_NAMES[a]}({a})")
        elif name in ("forcedOrder",) and a in FORCED_ORDER_NAMES:
            parts.append(f"{name}={FORCED_ORDER_NAMES[a]}({a})")
        elif name in ("bwIntra_x1000", "bwInter_x1000", "maxBw_x1000", "totalBw_x1000",
                       "graph_bwIntra_x1000", "ref_bwIntra_x1000"):
            parts.append(f"{name}={a/1000.0:.1f}")
        elif name in ("mult_x1000",):
            parts.append(f"mult={a/1000.0:.2f}")
        else:
            parts.append(f"{name}={a}")
    return ", ".join(parts)


def print_tree(node_id, enters, exits, children, depth, max_depth, out, indent=""):
    if depth > max_depth:
        return
    enter = enters.get(node_id)
    exit_ = exits.get(node_id)
    if not enter:
        return
    enter_str = fmt_args(enter)
    exit_str = fmt_args(exit_) if exit_ else "(no exit)"
    # Duration
    dur = (exit_["tsc"] - enter["tsc"]) if exit_ else -1
    out.append(f"{indent}#{node_id} {enter['func_name']}({enter_str}) -> [{exit_str}] dur={dur}")
    for child in children.get(node_id, []):
        print_tree(child, enters, exits, children, depth + 1, max_depth, out, indent + "  ")


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_callgraph.py <trace_file> [--max-depth N] [--stats]")
        sys.exit(1)
    path = sys.argv[1]
    max_depth = 999
    stats_only = False
    for i, a in enumerate(sys.argv[2:], 2):
        if a == "--max-depth" and i + 1 < len(sys.argv):
            max_depth = int(sys.argv[i + 1])
        elif a == "--stats":
            stats_only = True

    records = parse_file(path)
    print(f"Parsed {len(records)} records from {path}")
    if not records:
        return

    # Stats
    fn_counts = Counter()
    dir_counts = Counter()
    for r in records:
        fn_counts[r["func_name"]] += 1
        dir_counts[("ENTER" if r["dir"] == 1 else "EXIT")] += 1
    print("\n=== Function call counts ===")
    for fn, c in fn_counts.most_common():
        print(f"  {fn}: {c}")
    print(f"\nENTER: {dir_counts[('ENTER')]}, EXIT: {dir_counts[('EXIT')]}")

    enters, exits, children, roots = build_call_tree(records)
    print(f"\nRoots: {len(roots)}, Total enters: {len(enters)}")

    if stats_only:
        return

    # Print tree
    print(f"\n=== Call tree (max depth {max_depth}) ===")
    out = []
    for r in roots[:50]:  # limit number of roots printed
        print_tree(r, enters, exits, children, 0, max_depth, out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
