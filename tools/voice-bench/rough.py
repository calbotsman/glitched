import numpy as np, sys
from scipy.signal import resample_poly
def aper(x, sr=16000, fmin=70, fmax=450, hop=160, win=480):
    tmin, tmax = int(sr / fmax), int(sr / fmin); out = []
    for s in range(0, len(x) - win - tmax, hop):
        fr = x[s:s + win + tmax].astype(np.float64)
        if np.sqrt(np.mean(fr[:win] ** 2)) < 0.015: out.append(np.nan); continue
        d = np.array([np.sum((fr[:win] - fr[t:t + win]) ** 2) for t in range(tmax + 1)])
        cm = d[1:] * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)
        out.append(cm[tmin - 1:].min())
    return np.array(out)
def f0med(x, sr=16000):
    tmin, tmax = int(sr / 450), int(sr / 60); fs = []
    for s in range(0, len(x) - 480 - tmax, 160):
        fr = x[s:s + 480 + tmax].astype(np.float64)
        if np.sqrt(np.mean(fr[:480] ** 2)) < 0.015: continue
        d = np.array([np.sum((fr[:480] - fr[t:t + 480]) ** 2) for t in range(tmax + 1)])
        cm = d[1:] * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)
        c = np.where(cm[tmin - 1:] < 0.15)[0]
        if len(c): fs.append(sr / (c[0] + tmin))
    return np.median(fs)
if __name__ == '__main__':
    base = {}
    for a in sys.argv[1:]:
        fn, lag, src = a.split(':')
        x = resample_poly(np.fromfile(src, np.float32), 1, 3); y = resample_poly(np.fromfile(fn, np.float32)[int(lag):], 1, 3)[:len(x)]
        ax, ay = aper(x), aper(y); m = ~np.isnan(ax) & ~np.isnan(ay) & (ax < 0.2)   # frames where the INPUT is clearly voiced
        print(f"{fn:24s} roughness in voiced speech: median {np.median(ay[m]):.3f} (input {np.median(ax[m]):.3f}), rough frames(>0.3) {np.mean(ay[m] > 0.3):5.1%}   output median pitch {f0med(y):5.0f} Hz")
