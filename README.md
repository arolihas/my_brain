# my_brain

MRI and fMRI scans of my brain from a research study I took part in (Sept 2019, Siemens Prisma 3T
at the GSU/GT Center for Advanced Brain Imaging, an emotion/depression study where I looked at
pictures). This repo turns the raw scanner files into something I can actually look at, and answers
a few questions along the way: how good is the data, what do my brain networks look like, and when
did I fall asleep in the scanner.

## What came out of it

| | |
|---|---|
| **Data quality** | Very low head motion (mean framewise displacement 0.06–0.19 mm per run). Run 3 was the worst: 6% of volumes were scrubbed. See `figures/qc_motion.png`. |
| **The nap** | Probably **run 3, about 1:58–2:06 pm**. Whole-brain signal swings are 60–80% bigger than in any other run, the eye region goes quiet, and thalamus–cortex coupling drops. Signs of drowsiness start near the end of run 2, and runs 4–5 look awake. See `figures/drowsiness.png`. |
| **Networks** | A textbook default mode network (posterior cingulate + medial prefrontal + angular gyri), clean visual and somatomotor networks, and a connectome that works like a fingerprint (runs 4 and 5 match at r = 0.84). See `figures/seed_default.png`, `figures/connectome.png` and `figures/ica_networks.png`. |
| **The study** | Almost certainly [James & Duarte, *Neurobiology of Aging* 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11212072/). The task: a sound announced each picture's valence (slot machine = positive, whistle = neutral, screeching tyres = negative), then I rated the picture's intensity from 1 to 4. On 25% of trials, no picture came. There were 252 IAPS/NAPS pictures, and a memory test afterwards. No timing files or data are public (I checked Zenodo, OSF and OpenNeuro). |
| **The task timing** | Picture order was randomised per run, so runs share no timeline (best r = 0.06 against a chance level of 0.10). But the sounds leave a trace: auditory cortex has a strong 8–10 s trial rhythm that control regions lack. Deconvolving it recovers 61–70 of the ~84 trials per run. Visual cortex mirrors it, peaking about 6 s after each recovered cue, so I was alternating between hearing a cue and looking at a picture. The recovered onsets are approximate and incomplete: they miss about 20% of trials and can't tell positive from negative or catch trials. See `figures/trial_onsets.png` and `results/trial_onsets/`. |
| **Distortion** | The field maps predict up to about 5 mm of distortion in orbitofrontal cortex and the temporal poles. My own correction didn't measurably improve the fit at this resolution, so the data are left uncorrected (`results/fieldmap_check.json`). Treat orbitofrontal and amygdala results as ±a few mm. |

Numbers are in `results/*.json`. Interactive 3D viewers (open in a browser) are in `figures/interactive/`.

## Layout

```
Brain/                 raw DICOMs from the scanner (headers are pseudonymised: study code, not my name)
nifti/                 the same scans as NIfTI + JSON sidecars (dcm2niix; stored with git LFS)
mybrain/               shared code: paths and scan constants, motion, masks/atlases, data loaders
pipeline/              the analysis, one numbered step per file
run_pipeline.py        runs the steps in order, skipping finished ones
viz_playground.ipynb   notebook for looking around (raw scans, results, make-your-own seed maps)
figures/               figures from the pipeline (committed)
results/               numbers from the pipeline (committed)
derivatives/           large intermediate files, about 12 GB (regenerated, git-ignored)
```

## Running it

```bash
git lfs pull
```

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py
```

A full run takes about 45 minutes on 4 cores, most of it in motion correction. Re-running skips steps
whose outputs exist; `--from N` forces step N onwards, `--only N` forces one step, and `--list` shows status.

| step | what it does |
|---|---|
| `00_convert` | DICOM → NIfTI (only if `nifti/` is missing) |
| `01_motion` | rigid-body motion correction of every volume |
| `02_register` | T1 → MNI template (affine + nonlinear), fMRI → T1 |
| `03_to_mni` | every run into 2 mm MNI space |
| `04_denoise` | smoothing (6 mm) + regressing out 24 motion terms and 5 white-matter components, high-pass filter |
| `05_qc` | motion, temporal SNR, registration check figures |
| `06_networks` | connectome (Schaefer 100), seed maps, ICA, cross-run timeline test |
| `07_drowsiness` | sliding-window sleepiness markers per run |
| `08_interactive` | browser-viewable 3D maps (replaces the old `3d_brain.html` / `4d_brain.html`) |
| `09_trial_onsets` | recovers approximate trial onsets from auditory cortex, and checks them against other regions |
| `x1_fieldmap_check` | optional (`--fieldmap`): field-map distortion correction and the test that rejected it |

Everything runs in plain Python (nilearn, dipy, scikit-learn), with no FSL, AFNI or FreeSurfer.
For publication-grade preprocessing, including validated distortion correction, use fMRIPrep.

## The session

| series | scan | time |
|---|---|---|
| 1 | localizer | 13:25 |
| 2–3 | gradient-echo field map | 13:28 |
| 5 | T1 MEMPRAGE, 1 mm | 13:29 |
| 6 | fMRI_1: 296 volumes, 5.7 min | 13:36 |
| 7–10 | fMRI_2–5: 586 volumes, 11.3 min each | 13:44, 13:57, 14:11, 14:23 |

All fMRI runs: TR 1.155 s, TE 30 ms, multiband 2, 3.4 mm voxels. Runs 2–5 share an identical protocol; run 1 has 36 slices instead of 38.
