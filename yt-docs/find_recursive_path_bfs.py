#!/usr/bin/env python3
"""Find the BFS-index path from root to the first recursive SearchRec."""
import struct, sys, json
from collections import defaultdict, deque

HDR_FMT="<BBBBIIQQ"; HDR_SIZE=28
FUNC={1:"Compute",2:"SearchInit",3:"SearchRec",4:"SearchTryGpu",5:"SearchRecGpu",
      6:"FollowPath",7:"CompareGraphs",8:"SearchNextGpuSort",9:"ReplayGetGpu"}

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
    children=defaultdict(list); roots=[]
    for cid,r in enters.items():
        if r["pid"]==0 or r["pid"] not in enters: roots.append(cid)
        else: children[r["pid"]].append(cid)
    # BFS to assign index
    index_of={}; bfs_order=[]; parent_idx={}; depth={}
    q=deque()
    for r in roots:
        index_of[r]=len(bfs_order); bfs_order.append(r); parent_idx[r]=None; depth[r]=1; q.append(r)
    while q:
        cid=q.popleft()
        for c in children.get(cid,[]):
            index_of[c]=len(bfs_order); bfs_order.append(c); parent_idx[c]=index_of[cid]; depth[c]=depth[cid]+1; q.append(c)
    # BFS again, find first SearchRec whose ancestor is SearchRec
    parent={}
    q=deque()
    for r in roots: parent[r]=None; q.append(r)
    target=None
    while q:
        cid=q.popleft()
        e=enters[cid]
        if e["fid"]==3:
            p=parent[cid]
            while p is not None:
                if enters[p]["fid"]==3:
                    target=cid; break
                p=parent[p]
            if target: break
        for c in children.get(cid,[]):
            parent[c]=cid; q.append(c)
    if not target:
        print(json.dumps({"error":"not found"})); return
    # build path from root to target
    path_call_ids=[]; cur=target
    while cur is not None:
        path_call_ids.append(cur); cur=parent[cur]
    path_call_ids.reverse()
    # convert to BFS indices
    path_indices=[index_of[c] for c in path_call_ids]
    # also output func + key args for each
    path_info=[]
    for c in path_call_ids:
        e=enters[c]
        args=e["args"]
        info={"bfs_idx":index_of[c],"call_id":c,"func":FUNC.get(e["fid"],f"fn{e['fid']}"),"depth":depth[c]}
        # semantic: for SearchRec, show nChannels/time; for TryGpu show step/g/forcedOrder; for RecGpu show step
        if e["fid"]==3: info["nChannels"]=args[0]; info["time"]=args[2]
        elif e["fid"]==4: info["step"]=args[0]; info["forcedOrder"]=args[1]; info["g"]=args[4]
        elif e["fid"]==5: info["gpu_rank"]=args[0]; info["step"]=args[1]
        path_info.append(info)
    print(json.dumps({"target_bfs_idx":index_of[target],"target_call_id":target,"target_depth":depth[target],"path":path_info}, indent=2))

if __name__=="__main__":
    main()
