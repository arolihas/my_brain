"""Loaders for the denoised per-run arrays produced by pipeline/04_denoise.py."""
import numpy as np
from .config import deriv, DUMMY, FD_THRESH
from .motion import params, framewise_displacement


def clean(run):
    """Denoised, smoothed, z-scored grey-matter data: (volumes after dummies) x (voxels in space.brain_mask())."""
    return np.load(deriv(f'clean_run{run}.npy'))


def keep(run):
    """Boolean per volume (after dummies): True unless scrubbed for motion."""
    return framewise_displacement(params(run))[DUMMY:] <= FD_THRESH


def global_signal(run):
    return np.load(deriv(f'gs_run{run}.npy'))


def eye_signal(run):
    return np.load(deriv(f'eyes_run{run}.npy'))
