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

# ---------- tSNR ----------
tsnr_imgs = []
for r in R:
    img = nib.load(d + f'run{r}_mni.nii.gz'); x = img.get_fdata(dtype=np.float32)[..., 5:]
    hp = x - np.mean(x, -1, keepdims=True)
    t = np.where(x.mean(-1) > 0, x.mean(-1) / (x.std(-1) + 1e-6), 0)
    tsnr_imgs.append(nib.Nifti1Image(t.astype(np.float32), img.affine))
    res[f'run{r}']['median_tSNR_gm'] = round(float(np.median(masking.apply_mask(tsnr_imgs[-1], gm_mask))), 1)
brain_mask = image.math_img('img > 0', img=image.mean_img(tsnr_imgs))
brain_mask = image.math_img('a * b', a=brain_mask, b=gm_mask)
plotting.plot_img(image.mean_img(tsnr_imgs), cmap='inferno', colorbar=True, display_mode='z', cut_coords=8,
                  title='temporal SNR (mean over runs)', vmin=0, vmax=80).savefig(F + 'tsnr.png', dpi=110)

# ---------- denoising confounds ----------
csf = image.math_img('img > 0.9', img=image.resample_to_img(datasets.load_mni152_wm_mask(), mni, interpolation='nearest'))
def confounds(r, img):
    p = np.loadtxt(d + f'run{r}_motion.txt'); dp = np.vstack([np.zeros(6), np.diff(p, axis=0)])
    mot = np.hstack([p, dp, p ** 2, dp ** 2])
    wm = masking.apply_mask(img, csf)
    wm = signal.clean(wm, detrend=True, standardize='zscore_sample')
    u, s, vt = np.linalg.svd(wm, full_matrices=False)
    return np.hstack([mot, u[:, :5]])  # 24 motion + 5 aCompCor (white matter)
clean = {}
for r in R:
    img = nib.load(d + f'run{r}_mni.nii.gz')
    img = image.index_img(img, slice(5, None))
    conf = confounds(r, img)
    keep = fd[r][5:] <= 0.5
    clean[r] = image.clean_img(img, confounds=conf, detrend=True, standardize='zscore_sample', high_pass=0.008, t_r=TR, mask_img=brain_mask)
    clean[r] = image.smooth_img(clean[r], 6)
    res[f'run{r}']['vols_kept_after_scrub'] = int(keep.sum())
    clean[r]._keep = keep

# ---------- atlas connectivity ----------
sch = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
labels = [l.decode() if isinstance(l, bytes) else l for l in sch.labels]
labels = [l for l in labels if l != 'Background']
net = [l.split('_')[2] for l in labels]
masker = NiftiLabelsMasker(sch.maps, resampling_target='data')
ts = {r: masker.fit_transform(clean[r])[clean[r]._keep] for r in R}
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
from nilearn.maskers import NiftiSpheresMasker
bm = NiftiMasker(mask_img=brain_mask)
vox = {r: bm.fit_transform(clean[r])[clean[r]._keep] for r in R}
for name, xyz in seeds.items():
    zs = []
    for r in R:
        s = NiftiSpheresMasker([xyz], radius=6).fit_transform(clean[r])[clean[r]._keep]
        s = (s - s.mean()) / s.std(); v = vox[r]
        rr = (v * s).mean(0) / (v.std(0) + 1e-9); zs.append(np.arctanh(np.clip(rr, -.999, .999)))
    zimg = bm.inverse_transform(np.mean(zs, 0))
    fn = name.split(' ')[0].lower()
    nib.save(zimg, d + f'seed_{fn}.nii.gz')
    plotting.plot_stat_map(zimg, bg_img=d + 't1_in_mni.nii.gz', threshold=0.25, vmax=1, cut_coords=xyz, title=name,
                           black_bg=True).savefig(F + f'seed_{fn}.png', dpi=110)
    v = plotting.view_img(zimg, bg_img=d + 't1_in_mni.nii.gz', threshold=0.25, vmax=1, cut_coords=xyz, title=name, black_bg=True)
    v.save_as_html(F + f'seed_{fn}.html')

# ---------- cross-run synchrony (is something time-locked across runs 2-5?) ----------
n = min(ts[r].shape[0] for r in [2, 3, 4, 5])
full = {r: masker.transform(clean[r]) for r in [2, 3, 4, 5]}
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
ica = CanICA(n_components=20, mask=brain_mask, random_state=0, smoothing_fwhm=None, standardize=True, n_init=3)
ica.fit([clean[r] for r in R])
comps = ica.components_img_; nib.save(comps, d + 'ica20.nii.gz')
yeo = datasets.fetch_atlas_yeo_2011()
yeo7 = image.resample_to_img(yeo['thick_7'] if 'thick_7' in yeo else yeo.maps, mni, interpolation='nearest')
yn = ['Visual', 'Somatomotor', 'Dorsal attention', 'Salience/ventral attn', 'Limbic', 'Frontoparietal control', 'Default mode']
Y = masking.apply_mask(yeo7, brain_mask).astype(int); Cm = masking.apply_mask(comps, brain_mask)
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
