#!/usr/bin/env python3
"""B5: N-CMAPSS onset in cycles and cumulative flight hours (rows at 1 Hz) per unit; flight hours per class.
Output $AGTF30_WORK/agtf30_b/b5/ncmapss_onset.json."""
import h5py, numpy as np, glob, os, json
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README
D=NCMAPSS
rows=[]
for fn in sorted(glob.glob(D+"*.h5")):
    b=os.path.basename(fn).replace(".h5","")
    if "DS08d" in b: continue  # truncated on disk
    with h5py.File(fn,"r") as f:
        av=[x.decode() for x in np.array(f["A_var"]).ravel()]
        for part in ("dev","test"):
            A=f["A_"+part][:]
            iu,ic,ih,ifc=av.index("unit"),av.index("cycle"),av.index("hs"),av.index("Fc")
            for u in np.unique(A[:,iu]):
                Au=A[A[:,iu]==u]
                cyc,first,cnt=np.unique(Au[:,ic],return_index=True,return_counts=True)
                hs=Au[first,ih]; hrs=cnt/3600.0
                k=int(np.argmax(hs==0)) if (hs==0).any() else None
                rows.append({"file":b,"part":part,"unit":int(u),"Fc":int(Au[0,ifc]),"eol":int(cyc.max()),
                             "onset":int(cyc[k]) if k is not None else None,
                             "onset_hours":float(hrs[:k].sum()) if k is not None else None,
                             "h_mean":float(hrs.mean()),"h_min":float(hrs.min()),"h_max":float(hrs.max()),"total_hours":float(hrs.sum())})
json.dump(rows,open(f"{WORK}/agtf30_b/b5/ncmapss_onset.json","w"),indent=1)
import collections
by=collections.defaultdict(list)
for r in rows: by[r["Fc"]].append(r)
for c,rs in sorted(by.items()):
    oh=[r["onset_hours"] for r in rs if r["onset_hours"]]; oc=[r["onset"] for r in rs if r["onset"]]
    print("class",c,"n",len(rs),"h/flight mean %.2f min %.2f max %.2f"%(np.mean([r["h_mean"] for r in rs]),min(r["h_min"] for r in rs),max(r["h_max"] for r in rs)),
          "onset cyc med %d [%d,%d]"%(np.median(oc),min(oc),max(oc)),"onset h mean %.1f sd %.1f [%.1f,%.1f]"%(np.mean(oh),np.std(oh),min(oh),max(oh)),
          "eol med %d [%d,%d]"%(np.median([r['eol'] for r in rs]),min(r['eol'] for r in rs),max(r['eol'] for r in rs)))
oh=[r["onset_hours"] for r in rows if r["onset_hours"]]
print("all onset h mean %.1f sd %.1f; cycles"%(np.mean(oh),np.std(oh)), np.percentile([r['onset'] for r in rows if r['onset']],[5,50,95]))
