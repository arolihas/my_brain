"""QC, denoising, connectivity, seed maps, ICA, and cross-run synchrony."""
import numpy as np, nibabel as nib, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from nilearn import datasets, image, masking, signal, plotting
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.decomposition import CanICA
d = 'derivatives/'; F = 'figs/'; TR = 1.155; R = [1, 2, 3, 4, 5]; res = {}
mni = datasets.load_mni152_template(resolution=2)
gm_mask = image.math_img('img > 0.2', img=image.resample_to_img(datasets.load_mni152_gm_mask(), mni, interpolation='nearest'))

# ---------- motion QC ----------
fd = {}
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=False)
for r in R:
    p = np.loadtxt(d + f'run{r}_motion.txt'); p[:, :3] -= p[0, :3]; p[:, 3:] -= p[0, 3:]
    dp = np.abs(np.diff(p, axis=0)); f = np.r_[0, dp[:, :3].sum(1) + 50 * dp[:, 3:].sum(1)]
    fd[r] = f
    res[f'run{r}'] = dict(n_vols=len(f), minutes=round(len(f) * TR / 60, 1), mean_FD=round(float(f.mean()), 3),
                         pct_FD_over_0p5=round(float((f > 0.5).mean() * 100), 1), max_disp_mm=round(float(np.abs(p[:, :3]).max()), 2))
    ax[0].plot(np.arange(len(f)) * TR / 60, f, lw=.7, label=f'run {r}')
ax[0].axhline(.5, c='k', ls=':', lw=.8); ax[0].set_ylabel('FD (mm)'); ax[0].set_xlabel('minutes'); ax[0].legend(ncol=5, fontsize=8)
ax[1].bar([f'run {r}' for r in R], [res[f'run{r}']['mean_FD'] for r in R]); ax[1].set_ylabel('mean FD (mm)')
plt.tight_layout(); plt.savefig(F + 'motion.png', dpi=110); plt.close()

# ---------- tSNR + masking (work in 2D arrays to save memory) ----------
from scipy.ndimage import gaussian_filter
brain_mask = image.math_img('a * b', a=image.math_img('img > 0', img=nib.load(d + 'run2_mni.nii.gz').slicer[..., 0]), b=gm_mask)
bm = NiftiMasker(mask_img=brain_mask).fit()
wm_mask = image.math_img('img > 0.9', img=image.resample_to_img(datasets.load_mni152_wm_mask(), mni, interpolation='nearest'))
M = brain_mask.get_fdata() > 0; W = wm_mask.get_fdata() > 0
sig = 6 / 2.3548 / 2  # 6mm FWHM in 2mm voxels
clean = {}; keep = {}; tsnr_maps = []
import os
cached = all(os.path.exists(d + f'clean_run{r}.npy') for r in R)
if cached:
    for r in R:
        clean[r] = np.load(d + f'clean_run{r}.npy'); keep[r] = fd[r][5:] <= 0.5
        res[f'run{r}']['vols_kept_after_scrub'] = int(keep[r].sum())
    print('using cached cleaned data', flush=True)
for r in ([] if cached else R):
    img = nib.load(d + f'run{r}_mni.nii.gz'); T = img.shape[-1]
    X = np.empty((T - 5, M.sum()), np.float32); 
    vol = img.get_fdata(dtype=np.float32)
    raw = vol[M][:, 5:].T.copy(); Wd = vol[W][:, 5:].T.copy()
    for t in range(5, T): X[t - 5] = gaussian_filter(vol[..., t], sig)[M]
    ijk = np.indices(vol.shape[:3]).reshape(3, -1).T; xyz = ijk @ mni.affine[:3, :3].T + mni.affine[:3, 3]
    eye = ((np.linalg.norm(xyz - [33, 52, -36], axis=1) < 9) | (np.linalg.norm(xyz - [-33, 52, -36], axis=1) < 9)).reshape(vol.shape[:3])
    np.save(d + f'eyes_run{r}.npy', vol[eye][:, 5:].T); np.save(d + f'meanvol_run{r}.npy', vol.mean(-1))
    del vol
    tsnr = raw.mean(0) / (raw.std(0) + 1e-6); gsig = raw.mean(1); del raw
    tsnr_maps.append(tsnr); res[f'run{r}']['median_tSNR_gm'] = round(float(np.median(tsnr)), 1)
    p = np.loadtxt(d + f'run{r}_motion.txt'); dp = np.vstack([np.zeros(6), np.diff(p, axis=0)])
    mot = np.hstack([p, dp, p ** 2, dp ** 2])[5:]
    wz = signal.clean(Wd, detrend=True, standardize='zscore_sample'); u = np.linalg.svd(wz, full_matrices=False)[0][:, :5]
    clean[r] = signal.clean(X, confounds=np.hstack([mot, u]), detrend=True, standardize='zscore_sample', high_pass=0.008, t_r=TR).astype(np.float32)
    keep[r] = fd[r][5:] <= 0.5; res[f'run{r}']['vols_kept_after_scrub'] = int(keep[r].sum())
    np.save(d + f'clean_run{r}.npy', clean[r]); np.save(d + f'gs_run{r}.npy', gsig)
    del X, Wd; print('cleaned run', r, flush=True)
if tsnr_maps: plotting.plot_img(bm.inverse_transform(np.mean(tsnr_maps, 0)), cmap='inferno', colorbar=True, display_mode='z', cut_coords=8,
                  title='temporal SNR (mean over runs)', vmin=0, vmax=80, black_bg=True).savefig(F + 'tsnr.png', dpi=110)
# ---------- atlas connectivity ----------
sch = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
labels = [l.decode() if isinstance(l, bytes) else l for l in sch.labels]
labels = [l for l in labels if l != 'Background']
net = [l.split('_')[2] for l in labels]
lab_img = image.resample_to_img(sch.maps, mni, interpolation='nearest'); L = masking.apply_mask(lab_img, brain_mask).astype(int)
def parcel(X): return np.stack([X[:, L == k + 1].mean(1) for k in range(len(labels))], 1)
masker = NiftiLabelsMasker(lab_img).fit()
full = {r: parcel(clean[r]) for r in R}
ts = {r: full[r][keep[r]] for r in R}
mats = {r: np.corrcoef(ts[r].T) for r in R}
C = np.tanh(np.mean([np.arctanh(np.clip(mats[r], -.999, .999)) for r in R], 0)); np.fill_diagonal(C, 0)
np.save(d + 'connectome.npy', C)
fig, ax = plt.subplots(figsize=(8, 7)); im_ = ax.imshow(C, cmap='RdBu_r', vmin=-.8, vmax=.8)
nets = list(dict.fromkeys(net)); b = [net.index(n) for n in nets] + [len(net)]
for i in b[1:-1]: ax.axhline(i - .5, c='k', lw=.6); ax.axvline(i - .5, c='k', lw=.6)
ax.set_xticks([(b[i] + b[i + 1]) / 2 for i in range(len(nets))]); ax.set_xticklabels(nets, rotation=45)
ax.set_yticks(ax.get_xticks()); ax.set_yticklabels(nets); plt.colorbar(im_, label='correlation'); ax.set_title('Your functional connectome (100 regions, all runs)')
plt.tight_layout(); plt.savefig(F + 'connectome.png', dpi=110); plt.close()
# within vs between network
within = {n: float(np.mean([C[i, j] for i in range(len(net)) for j in range(len(net)) if i != j and net[i] == n and net[j] == n])) for n in nets}
between = float(np.mean([C[i, j] for i in range(len(net)) for j in range(len(net)) if net[i] != net[j]]))
res['network_within_r'] = {k: round(v, 3) for k, v in within.items()}; res['network_between_r'] = round(between, 3)
# run-to-run reliability of connectome (fingerprint)
iu = np.triu_indices(len(net), 1)
rel = np.array([[np.corrcoef(mats[a][iu], mats[b][iu])[0, 1] for b in R] for a in R]); res['connectome_run_similarity'] = np.round(rel, 2).tolist()

# ---------- seed maps ----------
seeds = {'Default mode (posterior cingulate)': (0, -52, 26), 'Motor (left hand area)': (-38, -22, 56),
         'Visual (primary)': (0, -88, 6), 'Salience (anterior insula)': (38, 22, -4),
         'Amygdala (left)': (-22, -4, -18)}
coords = np.array(np.nonzero(M)).T; world = coords @ mni.affine[:3, :3].T + mni.affine[:3, 3]
for name, xyz in seeds.items():
    sel = np.linalg.norm(world - np.array(xyz), axis=1) <= 6
    zs = []
    for r in R:
        v = clean[r][keep[r]]; s_ = v[:, sel].mean(1); s_ = (s_ - s_.mean()) / s_.std()
        rr = (v * s_[:, None]).mean(0) / (v.std(0) + 1e-9); zs.append(np.arctanh(np.clip(rr, -.999, .999)))
    zimg = bm.inverse_transform(np.mean(zs, 0))
    fn = name.split(' ')[0].lower()
    nib.save(zimg, d + f'seed_{fn}.nii.gz')
    plotting.plot_stat_map(zimg, bg_img=d + 't1_in_mni.nii.gz', threshold=0.25, vmax=1, cut_coords=xyz, title=name,
                           black_bg=True).savefig(F + f'seed_{fn}.png', dpi=110)
    v = plotting.view_img(zimg, bg_img=d + 't1_in_mni.nii.gz', threshold=0.25, vmax=1, cut_coords=xyz, title=name, black_bg=True)
    v.save_as_html(F + f'seed_{fn}.html')

# ---------- cross-run synchrony (is something time-locked across runs 2-5?) ----------
n = min(ts[r].shape[0] for r in [2, 3, 4, 5])
n = min(v.shape[0] for v in full.values())
sync = np.zeros(len(labels))
pairs = [(a, b) for a in [2, 3, 4, 5] for b in [2, 3, 4, 5] if a < b]
for k in range(len(labels)):
    sync[k] = np.mean([np.corrcoef(full[a][:n, k], full[b][:n, k])[0, 1] for a, b in pairs])
order = np.argsort(sync)[::-1]
res['cross_run_sync_top'] = [(labels[i], round(float(sync[i]), 3)) for i in order[:10]]
res['cross_run_sync_median'] = round(float(np.median(sync)), 3)
# null: circularly shift
rng = np.random.default_rng(0); null = []
for _ in range(200):
    k = rng.integers(len(labels)); a, b = pairs[rng.integers(len(pairs))]; sh = rng.integers(60, n - 60)
    null.append(np.corrcoef(np.roll(full[a][:n, k], sh), full[b][:n, k])[0, 1])
res['cross_run_sync_null_95pct'] = round(float(np.percentile(null, 95)), 3)
sync_img = masker.inverse_transform(sync[None])
plotting.plot_stat_map(image.index_img(sync_img, 0), bg_img=d + 't1_in_mni.nii.gz', display_mode='z', cut_coords=8,
                       title='Cross-run synchrony (runs 2-5)', cmap='hot', threshold=0.05, black_bg=True).savefig(F + 'sync.png', dpi=110)
# average time course of top region across runs
k = order[0]
fig, ax = plt.subplots(figsize=(11, 3))
for r in [2, 3, 4, 5]: ax.plot(np.arange(n) * TR / 60, full[r][:n, k], lw=.6, alpha=.6, label=f'run {r}')
ax.plot(np.arange(n) * TR / 60, np.mean([full[r][:n, k] for r in [2, 3, 4, 5]], 0), c='k', lw=1.2, label='mean')
ax.set_title(f'{labels[k]}: time course in each run'); ax.set_xlabel('minutes'); ax.legend(ncol=5, fontsize=8)
plt.tight_layout(); plt.savefig(F + 'sync_timecourse.png', dpi=110); plt.close()

# ---------- ICA ----------
from sklearn.decomposition import PCA, FastICA
red = [PCA(60, random_state=0).fit(clean[r][keep[r]]).components_ * 1 for r in R]
G = PCA(20, random_state=0).fit(np.vstack(red)).components_
S = FastICA(20, random_state=0, max_iter=2000, whiten='unit-variance').fit_transform(G.T).T  # spatial ICA
comps = bm.inverse_transform(S); nib.save(comps, d + 'ica20.nii.gz')
yn = ['Visual', 'Somatomotor', 'Dorsal attention', 'Salience/ventral attn', 'Limbic', 'Frontoparietal control', 'Default mode']
codes = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']  # Schaefer parcels carry Yeo-7 labels
parcel_net = np.array([0] + [codes.index(n) + 1 for n in net]); Y = parcel_net[L]; Cm = S
match = []
for i in range(Cm.shape[0]):
    c = Cm[i] * np.sign(Cm[i][np.argmax(np.abs(Cm[i]))]); top = c > np.percentile(c, 95)
    frac = [np.mean(Y[top] == k + 1) for k in range(7)]
    match.append((i, yn[int(np.argmax(frac))], round(float(max(frac)), 2)))
res['ica_matches'] = match
fig = plt.figure(figsize=(14, 9))
best = sorted(match, key=lambda m: -m[2])
seen = []; picks = []
for m in best:
    if m[1] not in seen: seen.append(m[1]); picks.append(m)
for j, (i, nm, fr) in enumerate(picks[:6]):
    ax = fig.add_subplot(3, 2, j + 1)
    c = image.index_img(comps, i); sgn = np.sign(Cm[i][np.argmax(np.abs(Cm[i]))])
    plotting.plot_stat_map(image.math_img(f'{sgn}*img', img=c), bg_img=d + 't1_in_mni.nii.gz', display_mode='ortho', axes=ax,
                           title=f'IC{i}: {nm} ({int(fr*100)}% overlap)', threshold='auto' if False else np.percentile(np.abs(Cm[i]), 95), colorbar=False, black_bg=True)
plt.savefig(F + 'ica_networks.png', dpi=110); plt.close()
json.dump(res, open('results.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
