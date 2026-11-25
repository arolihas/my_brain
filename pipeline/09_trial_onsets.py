"""Recover the task's trial timing from the brain itself (the study's onset files were never shared).

The task (James & Duarte, Neurobiology of Aging 2023): every trial began with a sound that announced the
valence of the next picture (slot machine / whistle / screeching tyres), then a picture to rate on 75% of
trials. Sounds are discrete events, so auditory cortex should carry a train of trial-locked responses.

Method
 1. Mean signal in auditory cortex (Heschl's gyrus + planum temporale), with the global signal removed.
 2. Sparse non-negative deconvolution with the canonical HRF, stopped at ~84 events per run
    ((252 pictures + 84 catch trials) / 4 runs).
 3. Checks that don't use the auditory signal: event-locked averages of visual, motor, default-mode and
    frontal-pole signals against a random-timing null, and lagged cross-correlations between regions.

Outputs: results/trial_onsets/run-{r}_events.tsv (onset in seconds from the first kept volume of the
raw run, i.e. including the dropped dummies), results/trial_onsets.json, figures/trial_onsets.png.
"""
import json
import numpy as np
from scipy.linalg import toeplitz
from sklearn.linear_model import lars_path
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mybrain.config import TR, DUMMY, TASK_RUNS, FIGURES, RESULTS
from mybrain import data, space

TRIALS_PER_RUN = 84
OUTDIR = RESULTS / 'trial_onsets'
WIN = np.arange(-4, 18)                  # event-locked window, in volumes
REGIONS = {
    'auditory': ('Auditory cortex (source of the onsets)', lambda: space.cortical_region("Heschl", "Planum Temporale"), '#2563eb'),
    'visual': ('Visual cortex', lambda: space.cortical_region("Intracalcarine", "Occipital Pole", "Lateral Occipital Cortex, inferior"), '#dc2626'),
    'motor': ('Right motor hand area (left-hand presses)', lambda: space.sphere((38, -22, 56), 8), '#d97706'),
    'default': ('Precuneus (default mode)', lambda: space.cortical_region("Precuneous"), '#7c3aed'),
    'frontal': ('Frontal pole (comparison)', lambda: space.cortical_region("Frontal Pole"), '#6b7280'),
}


def outputs():
    return [RESULTS / 'trial_onsets.json', FIGURES / 'trial_onsets.png'] + [OUTDIR / f'run-{r}_events.tsv' for r in TASK_RUNS]


def region_signals(run, masks):
    X = data.clean(run); g = X.mean(1)
    out = {}
    for k, m in masks.items():
        y = X[:, m].mean(1); y = y - (y @ g) / (g @ g) * g
        out[k] = (y - y.mean()) / y.std()
    return out


def deconvolve(y, k=TRIALS_PER_RUN):
    """Sparse non-negative deconvolution along the LARS-lasso path, stopped at ~k separate events."""
    hrf = spm_hrf(TR, oversampling=1, time_length=30.0); hrf /= hrf.max()
    H = toeplitz(np.r_[hrf, np.zeros(len(y) - len(hrf))], np.zeros(len(y)))
    _, _, path = lars_path(H, y - y.mean(), method='lasso', positive=True, max_iter=4 * k)
    events = []
    for coef in path.T:
        events = merge_adjacent(coef)
        if len(events) >= k: break
    return events


def merge_adjacent(s):
    """Non-zero runs of consecutive volumes become one event at their amplitude-weighted time."""
    ev, i = [], 0
    while i < len(s):
        if s[i] > 0:
            j = i
            while j + 1 < len(s) and s[j + 1] > 0: j += 1
            w = s[i:j + 1]; ev.append(((np.arange(i, j + 1) * w).sum() / w.sum() * TR, w.sum())); i = j + 1
        else:
            i += 1
    return np.array(ev)


def locked(times, y):
    idx = np.round(times / TR).astype(int)
    return np.array([y[i + WIN] for i in idx if i + WIN[0] >= 0 and i + WIN[-1] < len(y)])


def xcorr_peak(a, b, max_lag=8):
    lags = np.arange(-max_lag, max_lag + 1)
    r = [np.corrcoef(a[:len(a) - L], b[L:])[0, 1] if L >= 0 else np.corrcoef(a[-L:], b[:len(b) + L])[0, 1] for L in lags]
    return lags[int(np.argmax(r))] * TR, float(max(r)), float(r[max_lag])


def main():
    masks = {k: v[1]() for k, v in REGIONS.items()}
    sig = {r: region_signals(r, masks) for r in TASK_RUNS}
    onsets = {r: deconvolve(sig[r]['auditory']) for r in TASK_RUNS}
    OUTDIR.mkdir(exist_ok=True)
    for r in TASK_RUNS:
        with open(OUTDIR / f'run-{r}_events.tsv', 'w') as f:
            f.write('onset\tduration\tamplitude\ttrial_type\n')
            for t, a in onsets[r]:
                f.write(f'{t + DUMMY * TR:.2f}\t0\t{a:.3f}\trecovered_cue\n')

    rng = np.random.default_rng(0); res = {'events_per_run': {r: len(onsets[r]) for r in TASK_RUNS}, 'expected_per_run': TRIALS_PER_RUN,
                                           'event_locked': {}, 'cross_correlation': {}}
    fig = plt.figure(figsize=(13, 9)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    # A: a two-minute stretch of one run
    ax = fig.add_subplot(gs[0, :]); r = 4; t = np.arange(len(sig[r]['auditory'])) * TR; sel = (t > 120) & (t < 240)
    ax.plot(t[sel], sig[r]['auditory'][sel] + 2.2, c=REGIONS['auditory'][2], lw=1.4, label='auditory cortex')
    ax.plot(t[sel], sig[r]['visual'][sel] - 2.2, c=REGIONS['visual'][2], lw=1.4, label='visual cortex')
    for ev in onsets[r][(onsets[r][:, 0] > 120) & (onsets[r][:, 0] < 240), 0]: ax.axvline(ev, c='k', lw=.6, alpha=.5)
    ax.set(yticks=[], xlabel='seconds into run 4', title='Recovered trial onsets (black lines) from auditory cortex, run 4')
    ax.legend(loc='upper right', fontsize=8); ax.spines[['left', 'top', 'right']].set_visible(False)
    # B: event-locked averages against a random-timing null
    axs = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for k, (title, _, col) in REGIONS.items():
        rows = np.vstack([locked(onsets[r][:, 0], sig[r][k]) for r in TASK_RUNS]); m = rows.mean(0)
        null = np.array([np.vstack([locked(np.sort(rng.uniform(5 * TR, (len(sig[r][k]) - 20) * TR, len(onsets[r]))), sig[r][k])
                                    for r in TASK_RUNS]).mean(0) for _ in range(500)])
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0); x = WIN * TR
        res['event_locked'][k] = dict(peak_s=round(float(x[np.argmax(m)]), 1), trough_s=round(float(x[np.argmin(m)]), 1),
                                      pct_outside_chance=round(float(np.mean((m > hi) | (m < lo)) * 100)))
        a = axs[0] if k in ('auditory', 'visual', 'motor') else axs[1]
        a.fill_between(x, lo, hi, color=col, alpha=.12, lw=0); a.plot(x, m, c=col, lw=2, label=title)
    for a, tt in zip(axs, ['Task regions', 'Default mode vs a comparison region']):
        a.axvline(0, c='k', lw=.6); a.axhline(0, c='k', lw=.4); a.set(xlabel='seconds after recovered onset', ylabel='signal (z)', title=tt)
        a.legend(fontsize=8, loc='upper left'); a.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .005, 'Shaded bands: 95% range for randomly timed events. The auditory curve is circular (onsets come from it); the others are independent checks.',
             ha='center', fontsize=8, color='#555')
    plt.tight_layout(rect=(0, .02, 1, 1)); plt.savefig(FIGURES / 'trial_onsets.png', dpi=120); plt.close()
    for a, b in [('auditory', 'visual'), ('visual', 'motor'), ('auditory', 'default'), ('auditory', 'frontal')]:
        per = [xcorr_peak(sig[r][a], sig[r][b]) for r in TASK_RUNS]
        res['cross_correlation'][f'{a}->{b}'] = dict(peak_lag_s_per_run=[round(p[0], 1) for p in per],
                                                     peak_r_per_run=[round(p[1], 3) for p in per], r_at_zero_lag_per_run=[round(p[2], 3) for p in per])
    json.dump(res, open(RESULTS / 'trial_onsets.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
