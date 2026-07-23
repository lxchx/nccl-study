#!/usr/bin/env python3
"""Find the main recursion path: from root to the deepest leaf, always picking
the child that continues the search recursion. Uses lookahead: prefer children
whose subtree contains RecGpu/SearchRec (i.e., the search actually progressed)."""
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
    children=defaultdict(list); roots=[]
    for cid,r in enters.items():
        if r["pid"]==0 or r["pid"] not in enters: roots.append(cid)
        else: children[r["pid"]].append(cid)
    for k in children: children[k].sort()
    roots.sort()
    index_of={}; bfs_order=[]; depth={}
    q=deque()
    for r in roots:
        index_of[r]=len(bfs_order); bfs_order.append(r); depth[r]=1; q.append(r)
    while q:
        cid=q.popleft()
        for c in children.get(cid,[]):
            index_of[c]=len(bfs_order); bfs_order.append(c); depth[c]=depth[cid]+1; q.append(c)

    def subtree_has(child_fid_set, cid, maxdepth=3):
        """BFS up to maxdepth levels to check if any descendant has func in set."""
        q=deque([(cid,0)])
        seen=set()
        while q:
            x,d=q.popleft()
            if x in seen or d>maxdepth: continue
            seen.add(x)
            for c in children.get(x,[]):
                if enters[c]["fid"] in child_fid_set: return True
                q.append((c,d+1))
        return False

    def best_child(cid):
        cs=children.get(cid,[])
        if not cs: return None
        f=enters[cid]["fid"]
        if f==3:  # SearchRec: prefer TryGpu that progresses (has RecGpu child)
            for c in cs:
                cf=enters[c]["fid"]
                if cf==4 and subtree_has({5}, c, 1):  # TryGpu with RecGpu child
                    return c
            # else first TryGpu or SearchRec
            for c in cs:
                if enters[c]["fid"] in (3,4): return c
            return cs[0]
        if f==4:  # TryGpu: prefer RecGpu (the successful branch)
            for c in cs:
                if enters[c]["fid"]==5: return c
            # no RecGpu -> failed TryGpu, take first non-FollowPath? actually just take first
            return cs[0] if cs else None
        if f==5:  # RecGpu: prefer TryGpu that progresses (has RecGpu child)
            for c in cs:
                cf=enters[c]["fid"]
                if cf==4 and subtree_has({5}, c, 1): return c
            # else SearchRec (recursion closure)
            for c in cs:
                if enters[c]["fid"]==3: return c
            for c in cs:
                if enters[c]["fid"]==9: return c  # ReplayGetGpu
            return cs[0] if cs else None
        if f==1:  # Compute: prefer first SearchRec
            for c in cs:
                if enters[c]["fid"]==3: return c
            return cs[0] if cs else None
        if f==9:  # ReplayGetGpu: prefer TryGpu
            for c in cs:
                if enters[c]["fid"]==4: return c
            return cs[0] if cs else None
        return cs[0] if cs else None

    def walk(cid, path, visited):
        path=path+[cid]
        if cid in visited: return path, "cycle"
        visited=visited|{cid}
        nxt=best_child(cid)
        if nxt is None: return path, "leaf"
        if depth[nxt] <= depth[cid]: return path, "no-progress"
        return walk(nxt, path, visited)

    # for each root, find deepest path
    all_paths={}
    for r in roots:
        e=enters[r]
        p,reason=walk(r, [], set())
        all_paths[r]={"func":FUNC.get(e["fid"]),"path":p,"reason":reason,"max_depth":depth[p[-1]] if p else 0}

    # pick the deepest across all roots
    best_r=max(all_paths, key=lambda r: all_paths[r]["max_depth"])
    best=all_paths[best_r]
    path_info=[]
    for c in best["path"]:
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
            info["t1"]=args[0]; info["i1"]=args[1]; info["t2"]=args[2]; info["i2"]=args[3]; info["mult"]=args[4]
        elif e["fid"]==9:
            info["nChannels"]=args[0]; info["step"]=args[1]
        path_info.append(info)
    print(json.dumps({"best_root":best_r,"best_func":best["func"],"max_depth":best["max_depth"],"path_bfs_indices":[index_of[c] for c in best["path"]],"deepest_path":path_info,"all_roots":{str(r):{"func":v["func"],"max_depth":v["max_depth"],"reason":v["reason"],"path_len":len(v["path"])} for r,v in all_paths.items()}}, indent=2))

if __name__=="__main__":
    main()
