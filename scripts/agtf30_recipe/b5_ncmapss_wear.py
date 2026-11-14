#!/usr/bin/env python3
"""B5: empirical N-CMAPSS wear statistics per unit (dev + test): initial wear, normal slope, onset cycle (hs 1->0),
EOL cycle, EOL loss per health parameter. Output $AGTF30_WORK/agtf30_b/b5/ncmapss_wear.json (+ per-unit csv)."""
import h5py, numpy as np, glob, os, json, csv
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README
D=NCMAPSS
O=f"{WORK}/agtf30_b/b5/"
rows=[]
for fn in sorted(glob.glob(D+"*.h5")):
    b=os.path.basename(fn).replace(".h5","")
    if "DS08d" in b: continue  # file truncated on disk (eof mismatch)
    with h5py.File(fn,"r") as f:
        tv=[x.decode() for x in np.array(f["T_var"]).ravel()]
        av=[x.decode() for x in np.array(f["A_var"]).ravel()]
        for part in ("dev","test"):
            T=f["T_"+part][:]; A=f["A_"+part][:]
            iu,ic,ih,ifc=av.index("unit"),av.index("cycle"),av.index("hs"),av.index("Fc")
            for u in np.unique(A[:,iu]):
                m=A[:,iu]==u; Au=A[m]; Tu=T[m]
                cyc,first=np.unique(Au[:,ic],return_index=True)
                Tc=Tu[first]; hs=Au[first,ih]
                on=int(cyc[np.argmax(hs==0)]) if (hs==0).any() else None
                r={"file":b,"part":part,"unit":int(u),"Fc":int(Au[0,ifc]),"eol":int(cyc.max()),"onset":on}
                for j,n in enumerate(tv):
                    x=Tc[:,j]
                    r[n+"_c1"]=float(x[0]); r[n+"_eol"]=float(x[-1])
                    pre=(hs==1)
                    r[n+"_slope_pre"]=float(np.polyfit(cyc[pre],x[pre],1)[0]) if pre.sum()>=3 else None
                rows.append(r)
                print(b,part,int(u),r["eol"],on,flush=True)
json.dump({"T_var":tv,"units":rows},open(O+"ncmapss_wear.json","w"),indent=1)
with open(O+"ncmapss_wear_units.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
