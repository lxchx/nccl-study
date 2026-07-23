#!/usr/bin/env python3
"""Compute max call depth of the call tree."""
import struct, sys
from collections import defaultdict

HDR_FMT = "<BBBBIIQQ"
HDR_SIZE = 28

def parse_parents(path):
    enters = {}
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    n = len(data)
    while off + HDR_SIZE <= n:
        func_id, dir_, rank, nargs, call_id, parent_call_id, tsc, nrec = struct.unpack_from(HDR_FMT, data, off)
        if func_id == 0 and nrec == 0:
            break
        if nrec < HDR_SIZE or nrec > HDR_SIZE + 64 or nargs > 8:
            break
        if dir_ == 1:
            enters[call_id] = (parent_call_id, func_id)
        off += nrec
    return enters

def main():
    enters = parse_parents(sys.argv[1])
    # depth cache
    depth_cache = {}
    def depth(cid):
        if cid in depth_cache:
            return depth_cache[cid]
        if cid not in enters:
            return 0
        p = enters[cid][0]
        d = (depth(p) + 1) if p in enters else 1
        depth_cache[cid] = d
        return d
    max_d = 0
    max_cid = 0
    cnt = defaultdict(int)
    for cid in enters:
        d = depth(cid)
        cnt[d] += 1
        if d > max_d:
            max_d = d
            max_cid = cid
    print(f"Max depth: {max_d} (call_id={max_cid})")
    print("Depth distribution (depth: count):")
    for d in sorted(cnt):
        print(f"  {d}: {cnt[d]}")

if __name__ == "__main__":
    main()
