"""When did I fall asleep? Sliding-window wakefulness markers for each run.

Three markers that change at sleep onset (each z-scored across all windows of all runs):
  - global signal amplitude      goes UP   (large slow whole-brain fluctuations)
  - eye-region signal change     goes DOWN (eyes close, fewer eye movements)
  - thalamus-cortex coupling     goes DOWN
Their average is the drowsiness index. A fourth idea, 'does visual cortex still follow the pictures',
needs the same picture order in every run; 06_networks.py shows the runs don't share one.
"""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from nilearn import signal
from mybrain.config import RUNS, TR, DUMMY, FIGURES, RESULTS, deriv
from mybrain import data, space

WINDOW, STEP = 52, 4      # ~60 s windows every ~4.6 s
MARKERS = {'global signal amplitude (%)': +1, 'eye-region activity': -1, 'thalamus-cortex coupling (r)': -1}


def outputs():
    return [RESULTS / 'drowsiness.json', FIGURES / 'drowsiness.png']


def markers(run, thal, cortex):
    gs = data.global_signal(run)
    gs = signal.clean(gs[:, None], detrend=True, high_pass=0.008, t_r=TR, standardize=False)[:, 0] / gs.mean() * 100
    eyes = data.eye_signal(run); eyes = eyes[:, eyes.mean(0) > np.median(eyes.mean(0))]
    ey = signal.clean(eyes, detrend=True, standardize='zscore_sample', high_pass=0.01, t_r=TR)
    eye_change = np.abs(np.diff(ey, axis=0)).mean(1); eye_change = np.r_[eye_change[0], eye_change]
    X = data.clean(run); th = X[:, thal].mean(1); cx = X[:, cortex].mean(1)
    win = [(s, s + WINDOW) for s in range(0, len(gs) - WINDOW + 1, STEP)]
    t = (np.array([a + WINDOW / 2 for a, _ in win]) + DUMMY) * TR / 60
    vals = {'global signal amplitude (%)': [gs[a:b].std() for a, b in win],
            'eye-region activity': [eye_change[a:b].mean() for a, b in win],
            'thalamus-cortex coupling (r)': [np.corrcoef(th[a:b], cx[a:b])[0, 1] for a, b in win]}
    return t, {k: np.array(v) for k, v in vals.items()}


def main():
    _, _, per_voxel = space.schaefer100(); thal = space.thalamus(); cortex = per_voxel > 0
    M = {r: markers(r, thal, cortex) for r in RUNS}
    allv = {k: np.concatenate([M[r][1][k] for r in RUNS]) for k in MARKERS}
    z = {r: {k: sign * (M[r][1][k] - allv[k].mean()) / allv[k].std() for k, sign in MARKERS.items()} for r in RUNS}
    out = {}
    fig, axes = plt.subplots(len(RUNS), 1, figsize=(12, 2.4 * len(RUNS)), sharex=True, sharey=True)
    for ax, r in zip(axes, RUNS):
        t = M[r][0]; idx = np.mean(list(z[r].values()), 0)
        for c, k in zip(['#3b82f6', '#10b981', '#f59e0b'], MARKERS):
            ax.plot(t, z[r][k], lw=.8, c=c, alpha=.7, label=f"{k} {'↑' if MARKERS[k] > 0 else '↓'} = sleepier")
        ax.fill_between(t, 0, idx, where=idx > 0, color='k', alpha=.25, label='drowsiness index')
        ax.plot(t, idx, c='k', lw=1.6); ax.axhline(0, c='k', lw=.5); ax.set_ylabel(f'run {r}'); ax.set_ylim(-3, 3)
        out[f'run{r}'] = dict(mean_index=round(float(idx.mean()), 2), pct_windows_drowsy=round(float((idx > 0.5).mean() * 100), 1),
                              peak_minute=round(float(t[np.argmax(idx)]), 1),
                              **{k: round(float(M[r][1][k].mean()), 3) for k in MARKERS})
    axes[0].legend(fontsize=7, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.02)); axes[-1].set_xlabel('minutes into run')
    plt.tight_layout(); plt.savefig(FIGURES / 'drowsiness.png', dpi=120); plt.close()
    json.dump(out, open(RESULTS / 'drowsiness.json', 'w'), indent=1)
    for k, v in out.items(): print(k, v)


if __name__ == '__main__':
    main()
