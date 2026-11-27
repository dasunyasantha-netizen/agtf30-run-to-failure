"""Every 50th second of every engine (both releases, dev and test): achieved fan speed Nf (X_s_clean) against the
controller's corrected fan-speed request N1_target (X_a, cntrl.N1cReq) and the fan-inlet temperature T2 (W) to turn it
into a physical target (N1_target * sqrt(T2/518.67)), plus RegNum (active limiter) and PLA_clip. Read-only on the releases.
Paths: released folders $AGTF30_WORK/agtf30_v1 and .../agtf30_v11 (or AGTF30_V10, AGTF30_V11); output
$AGTF30_WORK/agtf30_v11_work/fig_fan/fan_target.npz. Idle CPU priority.

  python extract_fan.py
"""
import ctypes, glob, os, h5py, numpy as np
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception: pass
ROOTW = os.environ.get('AGTF30_WORK', 'work')
V10 = os.environ.get('AGTF30_V10', os.path.join(ROOTW, 'agtf30_v1')); V11 = os.environ.get('AGTF30_V11', os.path.join(ROOTW, 'agtf30_v11'))
OUTD = os.path.join(ROOTW, 'agtf30_v11_work', 'fig_fan')
fs = sorted(glob.glob(os.path.join(V10, 'AGTF30_*.h5')) + glob.glob(os.path.join(V11, 'AGTF30_*.h5')))
out = {k: [] for k in ('nf', 'tgt', 't2', 'reg', 'clip', 'sub')}
B = 16384 * 64
for f in fs:
    name = os.path.basename(f)[7:-3]
    with h5py.File(f, 'r') as h:
        for sp in ('dev', 'test'):
            xa, xs, ww = h[f'X_a_{sp}'], h[f'X_s_clean_{sp}'], h[f'W_{sp}']
            n = xa.shape[0]
            for s in range(0, n, B):
                e = min(n, s + B); off = (-s) % 50
                a = xa[s:e]; x = xs[s:e, 11]; t2 = ww[s:e, 3]
                out['t2'].append(t2[off::50])
                out['nf'].append(x[off::50]); out['tgt'].append(a[off::50, 10])
                out['reg'].append(a[off::50, 9].astype(np.int8)); out['clip'].append(a[off::50, 11].astype(np.int8))
                out['sub'].append(np.full(len(x[off::50]), ['HPT','HPC','LPT','FAN','MIX','HPTLPT','LPCHPC','ALL'].index(name), np.int8))
    print(name, 'done', flush=True)
os.makedirs(OUTD, exist_ok=True)
np.savez_compressed(os.path.join(OUTD, 'fan_target.npz'), **{k: np.concatenate(v) for k, v in out.items()})
print('FAN_DONE')
