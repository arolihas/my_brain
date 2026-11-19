"""When did I fall asleep? Sliding-window wakefulness markers per run."""
import numpy as np, nibabel as nib, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from nilearn import datasets, image, signal, masking
from nilearn.maskers import NiftiMasker
d = 'derivatives/'; F = 'figs/'; TR = 1.155; R = [1, 2, 3, 4, 5]
mni = datasets.load_mni152_template(resolution=2)
gm = image.math_img('img > 0.2', img=image.resample_to_img(datasets.load_mni152_gm_mask(), mni, interpolation='nearest'))
brain_mask = image.math_img('a * b', a=image.math_img('img > 0', img=nib.load(d + 'run2_mni.nii.gz').slicer[..., 0]), b=gm)
bm = NiftiMasker(mask_img=brain_mask).fit()
sch = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
labels = [l.decode() if isinstance(l, bytes) else l for l in sch.labels if (l.decode() if isinstance(l, bytes) else l) != 'Background']
L = masking.apply_mask(image.resample_to_img(sch.maps, mni, interpolation='nearest'), brain_mask).astype(int)
ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
H = masking.apply_mask(image.resample_to_img(ho.maps, mni, interpolation='nearest'), brain_mask).astype(int)
thal = np.isin(H, [i for i, n in enumerate(ho.labels) if 'Thalamus' in n])
vis = np.isin(L, [k + 1 for k, l in enumerate(labels) if '_Vis_' in l])
cortex = L > 0
print('voxels: thalamus', thal.sum(), 'visual', vis.sum())
W, step = 52, 4   # ~60 s windows, ~4.6 s step
def windows(n): return [(s, s + W) for s in range(0, n - W + 1, step)]
def z(x): return (x - x.mean()) / (x.std() + 1e-9)
C = {r: np.load(d + f'clean_run{r}.npy') for r in R}
V = {r: C[r][:, vis].mean(1) for r in R}
n = min(len(V[r]) for r in [2, 3, 4, 5])
# Does visual cortex follow the same timeline in runs 2-5? (same picture sequence)
loo_full = {r: np.corrcoef(V[r][:n], np.mean([V[q][:n] for q in [2, 3, 4, 5] if q != r], 0))[0, 1] for r in [2, 3, 4, 5]}
print('visual cortex vs other runs (whole run r):', {k: round(v, 3) for k, v in loo_full.items()})
out = {'visual_loo_r': {k: round(float(v), 3) for k, v in loo_full.items()}}
fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
summary = {}
for ax, r in zip(axes, R):
    gs = np.load(d + f'gs_run{r}.npy'); gs = signal.clean(gs[:, None], detrend=True, high_pass=0.008, t_r=TR, standardize=False)[:, 0] / gs.mean() * 100
    eyes = np.load(d + f'eyes_run{r}.npy'); eyes = eyes[:, eyes.mean(0) > np.percentile(eyes.mean(0), 50)]
    ey = signal.clean(eyes, detrend=True, standardize='zscore_sample', high_pass=0.01, t_r=TR)
    eyevar = np.abs(np.diff(ey, axis=0)).mean(1); eyevar = np.r_[eyevar[0], eyevar]
    th = C[r][:, thal].mean(1); cx = C[r][:, cortex].mean(1)
    p = np.loadtxt(d + f'run{r}_motion.txt')[5:]; fd = np.r_[0, np.abs(np.diff(p[:, :3], axis=0)).sum(1) + 50 * np.abs(np.diff(p[:, 3:], axis=0)).sum(1)]
    ws = windows(len(gs)); t = np.array([(a + b) / 2 for a, b in ws]) * TR / 60 + 5 * TR / 60
    m = {'global signal amplitude (%)': [gs[a:b].std() for a, b in ws],
         'eye-region activity': [eyevar[a:b].mean() for a, b in ws],
         'thalamus-cortex coupling (r)': [np.corrcoef(th[a:b], cx[a:b])[0, 1] for a, b in ws]}
    if r in [2, 3, 4, 5]:
        tmpl = np.mean([V[q][:n] for q in [2, 3, 4, 5] if q != r], 0)
        m['visual follows the pictures (r)'] = [np.corrcoef(V[r][a:b], tmpl[a:b])[0, 1] if b <= n else np.nan for a, b in ws]
    summary[r] = {k: np.array(v) for k, v in m.items()}; summary[r]['t'] = t
    # combine into one index: z-scored over ALL runs later
    for k, v in m.items():
        ax.plot(t, v, lw=1.2, label=k)
    ax.set_title(f'run {r}'); ax.legend(fontsize=7, ncol=4, loc='upper left')
plt.close()
# drowsiness index: high GS amplitude, low eye activity, low visual following, low thalamocortical coupling (each z-scored across all windows of all runs)
keys = ['global signal amplitude (%)', 'eye-region activity', 'thalamus-cortex coupling (r)']  # visual-follows dropped: runs use different picture orders
sign = {keys[0]: 1, keys[1]: -1, keys[2]: -1}
allv = {k: np.concatenate([summary[r][k] for r in R if k in summary[r]]) for k in keys}
mu = {k: np.nanmean(v) for k, v in allv.items()}; sd = {k: np.nanstd(v) for k, v in allv.items()}
fig, axes = plt.subplots(len(R), 1, figsize=(12, 2.4 * len(R)), sharex=True, sharey=True)
cols = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7']
for ax, r in zip(axes, R):
    t = summary[r]['t']; zs = []
    for c, k in zip(cols, keys):
        if k not in summary[r]: continue
        zk = sign[k] * (summary[r][k] - mu[k]) / sd[k]; zs.append(zk)
        ax.plot(t, zk, lw=.8, c=c, alpha=.7, label=k + (' ↑' if sign[k] > 0 else ' ↓'))
    idx = np.nanmean(zs, 0); summary[r]['index'] = idx
    ax.fill_between(t, 0, idx, where=idx > 0, color='k', alpha=.25, label='drowsiness index')
    ax.plot(t, idx, c='k', lw=1.6); ax.axhline(0, c='k', lw=.5)
    ax.set_ylabel(f'run {r}'); ax.set_ylim(-3, 3)
axes[0].legend(fontsize=7, ncol=5, loc='upper left', bbox_to_anchor=(0, 1.45))
axes[-1].set_xlabel('minutes into run')
plt.tight_layout(); plt.savefig(F + 'drowsiness.png', dpi=120); plt.close()
for r in R:
    idx = summary[r]['index']; t = summary[r]['t']
    out[f'run{r}'] = {'mean_index': round(float(np.nanmean(idx)), 2),
                      'pct_windows_drowsy(index>0.5)': round(float(np.mean(idx > 0.5) * 100), 1),
                      'peak_at_min': round(float(t[np.nanargmax(idx)]), 1),
                      **{k: round(float(np.nanmean(summary[r][k])), 3) for k in keys if k in summary[r]}}
json.dump(out, open('drowsiness.json', 'w'), indent=1); print(json.dumps(out, indent=1))
np.save(d + 'drowsy_summary.npy', summary, allow_pickle=True)
