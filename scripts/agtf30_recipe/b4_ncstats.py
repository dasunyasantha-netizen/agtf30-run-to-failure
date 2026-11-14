#!/usr/bin/env python3
"""B4: N-CMAPSS fan-speed statistics per file (dev; healthy = cycles <= 5) -> the 100 % reference. Output $AGTF30_WORK/agtf30_b/b4/ncmapss_nf_stats.json."""
import h5py, numpy as np, glob, os, json
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README
D=NCMAPSS
out={}
allnf=[];allnc=[]
for fn in sorted(glob.glob(D+"*.h5")):
    b=os.path.basename(fn)
    if "DS08d" in b: continue
    with h5py.File(fn,"r") as f:
        xs=[x.decode() for x in np.array(f["X_s_var"]).ravel()]
        i=xs.index("Nf")
        nf=f["X_s_dev"][:,i]; W=f["W_dev"][:]; A=f["A_dev"][:,:2]
    m=A[:,1]<=5
    T2=W[:,3]; ncor=nf/np.sqrt(T2/518.67)
    out[b]={"rows":int(len(nf)),"nf_max_all":float(nf.max()),"nf_max_c5":float(nf[m].max()),"nf_p999_c5":float(np.percentile(nf[m],99.9)),
            "nf_min_c5":float(nf[m].min()),"ncor_max_c5":float(ncor[m].max()),"tra_max":float(W[:,2].max()),"tra_min":float(W[:,2].min()),
            "alt_max":float(W[:,0].max()),"mach_max":float(W[:,1].max()),"T2min":float(T2.min()),"T2max":float(T2.max())}
    i=np.argmax(np.where(m,nf,0)); out[b]["at_max"]=W[i].tolist()
    print(b,{k:(round(v,1) if isinstance(v,float) else v) for k,v in out[b].items()},flush=True)
json.dump(out,open(f"{WORK}/agtf30_b/b4/ncmapss_nf_stats.json","w"),indent=1)
