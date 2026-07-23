#!/usr/bin/env python3
"""Find the deepest path from root, following the recursion chain.
Strategy: at each node, pick the child that leads deepest (preferring
RecGpu over TryGpu, preferring non-FollowPath). This produces a single
chain from root to the deepest leaf, showing the full search recursion."""
import struct, sys, json
from collections import defaultdict, deque

HDR_FMT="<BBBBIIQQ"; HDR_SIZE=28
FUNC={1:"Compute",2:"SearchInit",3:"SearchRec",4:"SearchTryGpu",
      5:"SearchRecGpu",6:"FollowPath",7:"CompareGraphs",
      8:"SearchNextGpuSort",9:"ReplayGetGpu"}

def parse(path):
    recs=[]
    with open(path,"rb") as f: data=f.read()
    off=0;n=len(data)
    while off+HDR_SIZE<=n:
        fid,dr,rk,na,cid,pid,tsc,nrec=struct.unpack_from(HDR_FMT,data,off)
        if fid==0 and nrec==0: break
        if nrec<HDR_SIZE or nrec>HDR_SIZE+64 or na>8: break
        args=[struct.unpack_from("<q",data,off+HDR_SIZE+i*8)[0] for i in range(na)]
        recs.append({"fid":fid,"dir":dr,"cid":cid,"pid":pid,"args":args})
        off+=nrec
    return recs

def main():
    recs=parse(sys.argv[1])
    enters={r["cid"]:r for r in recs if r["dir"]==1}
    exits={r["cid"]:r for r in recs if r["dir"]==0}
    children=defaultdict(list); roots=[]
    for cid,r in enters.items():
        if r["pid"]==0 or r["pid"] not in enters: roots.append(cid)
        else: children[r["pid"]].append(cid)
    # sort children by call order (cid)
    for k in children: children[k].sort()
    roots.sort()
    # BFS to assign index
    index_of={}; bfs_order=[]; parent_idx={}; depth={}
    q=deque()
    for r in roots:
        index_of[r]=len(bfs_order); bfs_order.append(r); parent_idx[r]=None; depth[r]=1; q.append(r)
    while q:
        cid=q.popleft()
        for c in children.get(cid,[]):
            index_of[c]=len(bfs_order); bfs_order.append(c); parent_idx[c]=index_of[cid]; depth[c]=depth[cid]+1; q.append(c)

    # Recursive descent: at each node, pick the child that continues the recursion.
    # Priority: SearchRec(3) > RecGpu(5) > ReplayGetGpu(9) > TryGpu(4) > others
    # Skip FollowPath(6) unless it's the only child.
    PRIORITY={3:0, 5:1, 9:2, 4:3, 8:4, 7:5, 2:6, 6:7, 1:8}
    def best_child(cid):
        cs=children.get(cid,[])
        if not cs: return None
        # pick by priority
        best=None; best_pri=99
        for c in cs:
            f=enters[c]["fid"]
            pri=PRIORITY.get(f, 50)
            # tie-break: prefer child with larger subtree (deeper recursion)
            if pri < best_pri or (pri==best_pri and best is not None):
                best=c; best_pri=pri
        return best

    # find deepest leaf along best_child chain from each root
    def walk(cid, path):
        path=path+[cid]
        nxt=best_child(cid)
        if nxt is None: return path, depth[cid]
        if depth[nxt] <= depth[cid]: return path, depth[cid]  # safety
        return walk(nxt, path)
    best_path=[]; best_depth=0
    for r in roots:
        p,d=walk(r, [])
        if d>best_depth:
            best_depth=d; best_path=p
    # convert to BFS indices + key args
    path_info=[]
    for c in best_path:
        e=enters[c]
        args=e["args"]
        info={"bfs_idx":index_of[c],"call_id":c,"func":FUNC.get(e["fid"],f"fn{e['fid']}"),"depth":depth[c]}
        if e["fid"]==3:
            info["nChannels"]=args[0]; info["saveGraph_nCh"]=args[1]; info["time"]=args[2]; info["pattern"]=args[3]
        elif e["fid"]==4:
            info["step"]=args[0]; info["forcedOrder"]=args[1]; info["g"]=args[4]; info["time"]=args[5]
        elif e["fid"]==5:
            info["gpu_rank"]=args[0]; info["step"]=args[1]; info["backToFirstRank"]=args[3]; info["forcedOrder"]=args[4]
        elif e["fid"]==6:
            info["type1"]=args[0]; info["idx1"]=args[1]; info["type2"]=args[2]; info["idx2"]=args[3]; info["mult"]=args[4]
        elif e["fid"]==9:
            info["nChannels"]=args[0]; info["step"]=args[1]
        path_info.append(info)
    print(json.dumps({"deepest_path":path_info,"path_bfs_indices":[index_of[c] for c in best_path],"max_depth":best_depth}, indent=2))

if __name__=="__main__":
    main()
