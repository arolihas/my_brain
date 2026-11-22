"""Brain networks: connectome, seed-based maps, ICA, and a test for a shared stimulus timeline across runs."""
import json
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from nilearn import plotting, image
from sklearn.decomposition import PCA, FastICA
from mybrain.config import RUNS, TASK_RUNS, TR, FIGURES, RESULTS, deriv
from mybrain import data, space
from mybrain.space import SEEDS



def outputs():
    return [RESULTS / 'networks.json', FIGURES / 'connectome.png', FIGURES / 'ica_networks.png', FIGURES / 'cross_run_sync.png'] + \
           [deriv(f'seed_{k}.nii.gz') for k in SEEDS]


def fisher_mean(rs):
    return np.tanh(np.mean([np.arctanh(np.clip(r, -.999, .999)) for r in rs], 0))


def connectome(parcels, keep, labels, nets, res):
    mats = {r: np.corrcoef(parcels[r][keep[r]].T) for r in RUNS}
    C = fisher_mean(mats.values()); np.fill_diagonal(C, 0); np.save(deriv('connectome.npy'), C)
    order = list(dict.fromkeys(nets)); b = [nets.index(n) for n in order] + [len(nets)]
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(C, cmap='RdBu_r', vmin=-.8, vmax=.8)
    for i in b[1:-1]: ax.axhline(i - .5, c='k', lw=.6); ax.axvline(i - .5, c='k', lw=.6)
    ticks = [(b[i] + b[i + 1]) / 2 for i in range(len(order))]
    ax.set_xticks(ticks, order, rotation=45); ax.set_yticks(ticks, order)
    plt.colorbar(im, label='correlation'); ax.set_title('Functional connectome (100 regions, all runs)')
    plt.tight_layout(); plt.savefig(FIGURES / 'connectome.png', dpi=110); plt.close()
    nets_a = np.array(nets); same = np.equal.outer(nets_a, nets_a); off = ~np.eye(len(nets), dtype=bool)
    res['within_network_r'] = {n: round(float(C[np.outer(nets_a == n, nets_a == n) & off].mean()), 3) for n in order}
    res['between_network_r'] = round(float(C[~same].mean()), 3)
    iu = np.triu_indices(len(nets), 1)
    res['run_to_run_connectome_similarity'] = [[round(float(np.corrcoef(mats[a][iu], mats[b_][iu])[0, 1]), 2) for b_ in RUNS] for a in RUNS]


def seed_maps(clean, keep):
    for key, (title, xyz) in SEEDS.items():
        sel = space.sphere(xyz, 6); zs = []
        for r in RUNS:
            v = clean[r][keep[r]]; s = v[:, sel].mean(1); s = (s - s.mean()) / s.std()
            zs.append(np.arctanh(np.clip((v * s[:, None]).mean(0) / (v.std(0) + 1e-9), -.999, .999)))
        z = space.unmask(np.mean(zs, 0)); nib.save(z, deriv(f'seed_{key}.nii.gz'))
        plotting.plot_stat_map(z, bg_img=space.t1_in_mni(), threshold=0.25, vmax=1, cut_coords=xyz, title=title,
                               black_bg=True).savefig(FIGURES / f'seed_{key}.png', dpi=110)


def cross_run_sync(parcels, labels, res):
    """If runs 2-5 showed the same pictures in the same order, stimulus-driven regions would correlate across runs."""
    n = min(parcels[r].shape[0] for r in TASK_RUNS)
    pairs = [(a, b) for i, a in enumerate(TASK_RUNS) for b in TASK_RUNS[i + 1:]]
    sync = np.array([np.mean([np.corrcoef(parcels[a][:n, k], parcels[b][:n, k])[0, 1] for a, b in pairs]) for k in range(len(labels))])
    rng = np.random.default_rng(0); null = []
    for _ in range(500):   # same statistic with one run circularly shifted: what chance looks like
        k = rng.integers(len(labels)); a, b = pairs[rng.integers(len(pairs))]
        null.append(np.corrcoef(np.roll(parcels[a][:n, k], rng.integers(60, n - 60)), parcels[b][:n, k])[0, 1])
    order = np.argsort(sync)[::-1]
    res['cross_run_sync'] = dict(top_regions=[(labels[i], round(float(sync[i]), 3)) for i in order[:5]],
                                 median=round(float(np.median(sync)), 3), chance_95pct=round(float(np.percentile(null, 95)), 3))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(null, 40, alpha=.5, label='chance (shifted runs)', density=True); ax.hist(sync, 30, alpha=.7, label='100 regions', density=True)
    ax.axvline(np.percentile(null, 95), c='k', ls=':', label='chance 95th pct'); ax.legend(fontsize=8)
    ax.set(xlabel='mean correlation of a region\'s time course between runs 2-5', title='Do runs share a stimulus timeline?')
    plt.tight_layout(); plt.savefig(FIGURES / 'cross_run_sync.png', dpi=110); plt.close()


def ica(clean, keep, nets, per_voxel, res, k=20):
    """Group-style spatial ICA: per-run PCA, concatenate, PCA again, FastICA on the spatial maps."""
    red = np.vstack([PCA(60, random_state=0).fit(clean[r][keep[r]]).components_ for r in RUNS])
    G = PCA(k, random_state=0).fit(red).components_
    S = FastICA(k, random_state=0, max_iter=2000, whiten='unit-variance').fit_transform(G.T).T
    S *= np.sign([s[np.argmax(np.abs(s))] for s in S])[:, None]      # make the dominant lobe positive
    nib.save(image.concat_imgs([space.unmask(s) for s in S]), deriv('ica20.nii.gz'))
    net_of_voxel = np.array([0] + [space.YEO7.index(n) + 1 for n in nets])[per_voxel]
    matches = []
    for i, s in enumerate(S):
        top = s > np.percentile(s, 95); frac = [float(np.mean(net_of_voxel[top] == j + 1)) for j in range(7)]
        matches.append(dict(component=i, closest_network=space.YEO7_NAMES[space.YEO7[int(np.argmax(frac))]], overlap=round(max(frac), 2)))
    res['ica_components'] = matches
    picks, seen = [], set()
    for m in sorted(matches, key=lambda m: -m['overlap']):
        if m['closest_network'] not in seen: seen.add(m['closest_network']); picks.append(m)
    fig = plt.figure(figsize=(14, 9))
    for j, m in enumerate(picks[:6]):
        s = S[m['component']]
        plotting.plot_stat_map(space.unmask(s), bg_img=space.t1_in_mni(), axes=fig.add_subplot(3, 2, j + 1), colorbar=False, black_bg=True,
                               threshold=np.percentile(np.abs(s), 95),
                               title=f"IC{m['component']}: closest to {m['closest_network']} ({int(m['overlap'] * 100)}%)")
    plt.savefig(FIGURES / 'ica_networks.png', dpi=110); plt.close()


def main():
    labels, nets, per_voxel = space.schaefer100()
    clean = {r: data.clean(r) for r in RUNS}; keep = {r: data.keep(r) for r in RUNS}
    parcels = {r: space.parcellate(clean[r], per_voxel, len(labels)) for r in RUNS}
    res = {}
    connectome(parcels, keep, labels, nets, res); print('connectome done', flush=True)
    seed_maps(clean, keep); print('seed maps done', flush=True)
    cross_run_sync(parcels, labels, res); print('cross-run sync done', flush=True)
    ica(clean, keep, nets, per_voxel, res); print('ICA done', flush=True)
    json.dump(res, open(RESULTS / 'networks.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'ica_components'}, indent=1))


if __name__ == '__main__':
    main()
