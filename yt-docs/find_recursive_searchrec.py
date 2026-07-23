#!/usr/bin/env python3
"""Find the first ncclTopoSearchRec that indirectly calls itself,
and print the full call path from root to that recursive call."""
import struct, sys
from collections import defaultdict

HDR_FMT = "<BBBBIIQQ"; HDR_SIZE = 28
FUNC = {1:"Compute",2:"SearchInit",3:"SearchRec",4:"SearchTryGpu",5:"SearchRecGpu",
        6:"FollowPath",7:"CompareGraphs",8:"SearchNextGpuSort",9:"ReplayGetGpu"}

def parse(path):
    recs = []
    with open(path,"rb") as f: data = f.read()
    off=0; n=len(data)
    while off+HDR_SIZE<=n:
        fid,dr,rk,na,cid,pid,tsc,nrec = struct.unpack_from(HDR_FMT,data,off)
        if fid==0 and nrec==0: break
        if nrec<HDR_SIZE or nrec>HDR_SIZE+64 or na>8: break
        args=[struct.unpack_from("<q",data,off+HDR_SIZE+i*8)[0] for i in range(na)]
        recs.append({"fid":fid,"dir":dr,"cid":cid,"pid":pid,"tsc":tsc,"args":args})
        off+=nrec
    return recs

def main():
    recs = parse(sys.argv[1])
    enters = {r["cid"]: r for r in recs if r["dir"]==1}
    children = defaultdict(list)
    roots = []
    for cid, r in enters.items():
        if r["pid"]==0 or r["pid"] not in enters: roots.append(cid)
        else: children[r["pid"]].append(cid)
    # BFS to assign depth, and find the first SearchRec whose ancestor chain contains another SearchRec
    from collections import deque
    parent = {}
    depth = {}
    q = deque()
    for r in roots:
        depth[r] = 1; parent[r] = None; q.append(r)
    first_recursive = None
    while q:
        cid = q.popleft()
        e = enters[cid]
        if e["fid"] == 3:  # SearchRec
            # walk up to check if any ancestor is also SearchRec
            p = parent[cid]
            while p is not None:
                pe = enters[p]
                if pe["fid"] == 3:
                    first_recursive = cid
                    break
                p = parent[p]
            if first_recursive:
                break
        for c in children.get(cid, []):
            parent[c] = cid
            depth[c] = depth[cid] + 1
            q.append(c)
    if not first_recursive:
        print("No recursive SearchRec found")
        return
    # Build the full path from root to first_recursive
    path = []
    cur = first_recursive
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    print(f"First recursive SearchRec: call_id={first_recursive}, depth={depth[first_recursive]}")
    print(f"Path length: {len(path)}")
    print("\n=== Full call path ===")
    for i, cid in enumerate(path):
        e = enters[cid]
        fn = FUNC.get(e["fid"], f"fn{e['fid']}")
        args = e["args"]
        # Semantic decode
        argstr = ",".join(str(a) for a in args)
        print(f"  depth {depth[cid]:2d}: #{cid} {fn}({argstr})")
    # Print which ancestor is the matching SearchRec
    target = enters[first_recursive]
    p = parent[first_recursive]
    while p is not None:
        pe = enters[p]
        if pe["fid"] == 3:
            print(f"\n*** Recursive ancestor: #{p} (depth {depth[p]}) -> #{first_recursive} (depth {depth[first_recursive]})")
            print(f"    Ancestor graph_nChannels={pe['args'][0]}, time={pe['args'][2]}")
            print(f"    Recursive graph_nChannels={target['args'][0]}, time={target['args'][2]}")
            break
        p = parent[p]

if __name__ == "__main__":
    main()
